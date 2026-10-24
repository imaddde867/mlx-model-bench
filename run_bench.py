#!/usr/bin/env python3
"""Run the pipeline for one model, end to end, in the order that keeps results usable.

Replaces run_p2.sh. One model at a time:

  .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase all
  .venv-mlxlm/bin/python run_bench.py --model spark-x2.5-4b --phase fetch
  .venv-mlxlm/bin/python run_bench.py --model lfm2.5-8b-a1b --phase p2 --trials 5

Phases:
  fetch  download every checkpoint of the model (resumes; verifies nothing)
  p0     load test + one sanity answer + prompt-token count -> results/<id>/p0/<ckpt>.json
  p2     speed/TTFT/memory: in-process, then mlx_lm.server, then the labelled
         prefix-cache run -> results/<id>/<ckpt>-<transport>/
  p3     accuracy (30) + quality (8) -> results/<id>/p3/
  all    fetch, p0, p2, p3

p0 is the gate: if a checkpoint fails to load or produces junk, `all` stops
there rather than spending an hour on nonsense. Run it with the engine's own
interpreter — the subprocesses inherit sys.executable.

Everything model-specific comes from registry.py.
"""

import argparse
import importlib.metadata as md
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_tail(path, n=12):
    try:
        lines = Path(path).read_text().strip().splitlines()
    except OSError:
        return
    for line in lines[-n:]:
        print("    " + line[:160])


def run(cmd, log_path, timeout=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode


def phase_fetch(cfg):
    from huggingface_hub import snapshot_download

    ok = True
    for ckpt in cfg.checkpoints.values():
        if ckpt.available:
            print(f"  {ckpt.tag}: already at {ckpt.local_dir}")
            continue
        print(f"  {ckpt.tag}: downloading {ckpt.hf_repo} -> {ckpt.local_dir}")
        try:
            snapshot_download(
                ckpt.hf_repo,
                local_dir=str(ckpt.local_path),
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
            )
            print(f"    {sum(f.stat().st_size for f in ckpt.local_path.glob('*') if f.is_file()) / 1e9:.2f} GB on disk")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            ok = False
    return ok


def phase_p0(cfg, out):
    p0_dir = out / cfg.id / "p0"
    p0_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for ckpt in cfg.available_checkpoints:
        log = out / cfg.id / "logs" / f"p0-{ckpt.tag}.log"
        rc = run(
            [sys.executable, str(ROOT / "p0_load.py"), str(ckpt.local_path),
             "--template-kwargs", json.dumps(cfg.template_kwargs)],
            log, timeout=900,
        )
        row = None
        for line in reversed(log.read_text().splitlines()):
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if row is None:
            print(f"  {ckpt.tag}: p0 produced no JSON row (see {log})")
            all_ok = False
            continue
        row.update({"model_id": cfg.id, "checkpoint": ckpt.tag, "hf_repo": ckpt.hf_repo,
                    "weights_gb": ckpt.weights_gb, "bits_per_weight": ckpt.bits_per_weight,
                    "template_kwargs": cfg.template_kwargs})
        (p0_dir / f"{ckpt.tag}.json").write_text(json.dumps(row, indent=2))
        status = "ok" if row.get("ok") else "FAIL"
        print(f"  {ckpt.tag}: {status} load {row.get('load_s')}s peak {row.get('load_peak_gb')}GB "
              f"prompt {row.get('prompt_tokens')} tok -> {row.get('sample', row.get('error', ''))[:70]!r}")
        all_ok = all_ok and bool(row.get("ok"))
    return all_ok


def wait_for_server(url, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def phase_p2(cfg, out, trials, prompts, max_tokens, server_version):
    ok = True
    for ckpt in cfg.available_checkpoints:
        # --out must be on every call: without it a smoke run overwrites the real
        # results directory, which is how a 1-trial check once ate a 5-trial set.
        args = [sys.executable, str(ROOT / "bench.py"),
                "--model", cfg.id, "--checkpoint", ckpt.tag,
                "--trials", str(trials), "--max-tokens", str(max_tokens),
                "--out", str(out)]
        print(f"  {ckpt.tag}: in-process")
        log = out / cfg.id / "logs" / f"p2-{ckpt.tag}-inproc.log"
        rc = run([*args, "--transport", "inproc", "--prompts", *prompts], log)
        if rc != 0:
            print(f"    exited {rc}; see {log}")
            log_tail(log)
            ok = False
            continue

        port = free_port()
        print(f"  {ckpt.tag}: mlx_lm.server on :{port}")
        slog = out / cfg.id / "logs" / f"p2-{ckpt.tag}-server.log"
        server = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm.server", "--model", str(ckpt.local_path),
             "--port", str(port)],
            stdout=open(slog, "w"), stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_server(f"http://127.0.0.1:{port}"):
                print(f"    server never became ready; see {slog}")
                ok = False
                continue
            base = f"http://127.0.0.1:{port}"
            http_args = [*args, "--transport", "http", "--base-url", base,
                         "--server-pid", str(server.pid),
                         "--server-version", server_version]
            print(f"  {ckpt.tag}: HTTP, cache-busted")
            run([*http_args, "--prompts", *prompts],
                out / cfg.id / "logs" / f"p2-{ckpt.tag}-http.log")

            print(f"  {ckpt.tag}: HTTP, prefix-cache hit (labelled row)")
            cache_prompts = [p for p in prompts if p in ("short", "medium", "long")]
            run([*http_args, "--no-nonce", "--tag", "cached",
                 "--prompts", *(cache_prompts or prompts)],
                out / cfg.id / "logs" / f"p2-{ckpt.tag}-http-cached.log")
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    return ok


def phase_p3(cfg, out):
    tags = [c.tag for c in cfg.available_checkpoints]
    log = out / cfg.id / "logs" / "p3.log"
    rc = run([sys.executable, str(ROOT / "run_p3.py"), "--model", cfg.id,
              "--checkpoints", *tags, "--out", str(out)], log, timeout=3600)
    if rc != 0:
        print(f"  exited {rc}; see {log}")
        log_tail(log)
    return rc == 0


def server_version_string():
    try:
        return f"mlx-lm {md.version('mlx-lm')}"
    except md.PackageNotFoundError:
        return "mlx-lm unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry id")
    ap.add_argument("--phase", choices=("fetch", "p0", "p2", "p3", "all"), default="all")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", nargs="+", default=["short", "medium", "long", "decode-short", "decode-long"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = registry.get_model(args.model)
    out = Path(args.out)
    print(f"\n{cfg.id} · {cfg.name} · template {json.dumps(cfg.template_kwargs)}")
    print(f"engine {server_version_string()} · python {sys.version.split()[0]}")

    phases = ["fetch", "p0", "p2", "p3"] if args.phase == "all" else [args.phase]
    ok = True
    for ph in phases:
        print(f"\n[{ph}]")
        if ph == "fetch":
            ok = phase_fetch(cfg)
        elif ph == "p0":
            ok = phase_p0(cfg, out)
        elif ph == "p2":
            ok = phase_p2(cfg, out, args.trials, args.prompts, args.max_tokens,
                          server_version_string())
        elif ph == "p3":
            ok = phase_p3(cfg, out)
        if not ok:
            print(f"\n[{ph}] did not complete cleanly — stopping here so the failure is cheap.")
            return 1

    print(f"\n{cfg.id}: {args.phase} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
