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
import hashlib
import importlib.metadata as md
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import gates
import registry

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ["short", "medium", "long", "decode-short", "decode-long"]
DEFAULT_MAX_TOKENS = 256


def optiq_exe():
    for cand in (ROOT / ".venv-optiq/bin/optiq", ROOT / ".venv-optiq/bin/python"):
        if cand.exists():
            return cand
    return None


def optiq_version():
    py = ROOT / ".venv-optiq/bin/python"
    if not py.exists():
        return "mlx-optiq not installed"
    try:
        out = subprocess.run(
            [str(py), "-c", "import importlib.metadata as m; print(m.version('mlx-optiq'))"],
            capture_output=True, text=True, timeout=60,
        )
        return f"mlx-optiq {out.stdout.strip()}" if out.returncode == 0 else "mlx-optiq unknown"
    except Exception:
        return "mlx-optiq unknown"


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
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write(f"\ntimed out after {timeout}s\n")
            return 124
    return proc.returncode


def hub_manifest(hf_repo):
    """(revision, {filename: {size, sha256}}) straight from the Hub, the same
    source P0.md checked the baseline against."""
    url = f"https://huggingface.co/api/models/{hf_repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    return d.get("sha"), {
        s["rfilename"]: {"size": s.get("size"),
                          "sha256": (s.get("lfs") or {}).get("sha256")}
        for s in d.get("siblings", [])
    }


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(ckpt):
    """Byte-for-byte check against the Hub. A quant that does not match its
    published hash is not the file the report claims to have measured."""
    record = {"hf_repo": ckpt.hf_repo, "checkpoint": ckpt.tag, "files": [],
              "revision": None, "verified": None, "error": None}
    try:
        record["revision"], manifest = hub_manifest(ckpt.hf_repo)
    except Exception as e:
        record["error"] = f"manifest fetch failed: {type(e).__name__}: {e}"
        return record

    all_ok = True
    for name, meta in sorted(manifest.items()):
        if not name.endswith(".safetensors"):
            continue
        local = ckpt.local_path / name
        entry = {"file": name, "expected_bytes": meta["size"],
                 "expected_sha256": meta["sha256"], "ok": False}
        if not local.exists():
            entry["error"] = "missing locally"
            all_ok = False
            record["files"].append(entry)
            continue
        entry["local_bytes"] = local.stat().st_size
        if meta["size"] and entry["local_bytes"] != meta["size"]:
            entry["error"] = "size mismatch"
            all_ok = False
        elif meta["sha256"]:
            entry["local_sha256"] = sha256_file(local)
            entry["ok"] = entry["local_sha256"] == meta["sha256"]
            if not entry["ok"]:
                entry["error"] = "sha256 mismatch"
            all_ok = all_ok and entry["ok"]
        else:
            entry["error"] = "Hub publishes no sha256; a size match is not verification"
            all_ok = False
        record["files"].append(entry)
    record["verified"] = bool(all_ok and record["files"] and record["revision"])
    return record


def phase_fetch(cfg, out, selected):
    ok = True
    prov_dir = out / cfg.id / "p0"
    prov_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in selected:
        if ckpt.available:
            print(f"  {ckpt.tag}: already at {ckpt.local_dir}")
        else:
            from huggingface_hub import snapshot_download

            print(f"  {ckpt.tag}: downloading {ckpt.hf_repo} -> {ckpt.local_dir}")
            try:
                snapshot_download(
                    ckpt.hf_repo,
                    local_dir=str(ckpt.local_path),
                    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
                                    "*.py", "*.md", "*LICENSE*", "*.png", ".gitattributes"],
                )
                size = sum(f.stat().st_size for f in ckpt.local_path.glob("*")
                           if f.is_file() and not f.name.startswith("."))
                print(f"    {size / 1e9:.2f} GB on disk")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                ok = False
                continue

        if not ckpt.available:
            print("    not loadable: config.json missing after fetch")
            ok = False
            continue

        print(f"  {ckpt.tag}: verifying against the Hub")
        record = verify_checkpoint(ckpt)
        record.update({"model_id": cfg.id, "local_dir": ckpt.local_dir,
                       "weights_gb": ckpt.weights_gb})
        (prov_dir / f"{ckpt.tag}-provenance.json").write_text(json.dumps(record, indent=2))
        if record["verified"]:
            n = len(record["files"])
            print(f"    {n} weight file(s) match their published sha256")
        else:
            print(f"    VERIFICATION FAILED: {record['error'] or 'see ' + str(prov_dir)} — do not report this checkpoint")
            for f in record["files"]:
                if not f["ok"]:
                    print(f"      {f['file']}: {f.get('error')}")
            ok = False
    return ok


def local(selected):
    """Selected checkpoints that are actually on disk."""
    return [c for c in selected if c.available]


def undownloaded(cfg, selected):
    """Report selected tiers that are not on disk; a missing tier is a failure,
    never a silent skip. True if any (or none selected)."""
    missing = [c for c in selected if not c.available]
    for c in missing:
        print(f"  {cfg.id}/{c.tag}: selected but not downloaded ({c.local_dir}) — run --phase fetch")
    return bool(missing) or not selected


def phase_p0(cfg, out, selected):
    p0_dir = out / cfg.id / "p0"
    p0_dir.mkdir(parents=True, exist_ok=True)
    if undownloaded(cfg, selected) or gates.refuse(cfg, selected, out, need_p0=False):
        return False
    all_ok = True
    for ckpt in selected:
        log = out / cfg.id / "logs" / f"p0-{ckpt.tag}.log"
        rc = run(
            [sys.executable, str(ROOT / "p0_load.py"), str(ckpt.local_path),
             "--template-kwargs", json.dumps(cfg.template_kwargs),
             "--max-tokens", str(cfg.p0_max_tokens)],
            log, timeout=900,
        )
        if rc != 0:
            print(f"  {ckpt.tag}: p0 exited {rc}; see {log}")
            all_ok = False
            continue
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
        prov = json.loads((p0_dir / f"{ckpt.tag}-provenance.json").read_text())
        row.update({"weights_fingerprint": gates.fingerprint(prov), "model_id": cfg.id, "checkpoint": ckpt.tag, "hf_repo": ckpt.hf_repo,
                    "weights_gb": ckpt.weights_gb, "bits_per_weight": ckpt.bits_per_weight,
                    "template_kwargs": cfg.template_kwargs})
        (p0_dir / f"{ckpt.tag}.json").write_text(json.dumps(row, indent=2))
        status = "ok" if row.get("ok") else "FAIL"
        print(f"  {ckpt.tag}: {status} load {row.get('load_s')}s peak {row.get('load_peak_gb')}GB "
              f"prompt {row.get('prompt_tokens')} tok -> {row.get('sample', row.get('error', ''))[:70]!r}")
        all_ok = all_ok and bool(row.get("ok"))
    return all_ok


def wait_for_server(url, timeout=240, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False  # died during startup; waiting out the timeout hides the exit code
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def phase_p2(cfg, out, trials, prompts, max_tokens, server_version, selected):
    ok = True
    if undownloaded(cfg, selected) or gates.refuse(cfg, selected, out):
        return False
    for ckpt in selected:
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
        slog.parent.mkdir(parents=True, exist_ok=True)
        with open(slog, "w") as server_log:
            server = subprocess.Popen(
                [sys.executable, "-m", "mlx_lm.server", "--model", str(ckpt.local_path),
                 "--port", str(port)],
                stdout=server_log, stderr=subprocess.STDOUT,
            )
        try:
            if not wait_for_server(f"http://127.0.0.1:{port}", proc=server):
                print(f"    server never became ready (exit {server.poll()}); see {slog}")
                log_tail(slog)
                ok = False
                continue
            base = f"http://127.0.0.1:{port}"
            http_args = [*args, "--transport", "http", "--base-url", base,
                         "--server-pid", str(server.pid),
                         "--server-version", server_version]
            print(f"  {ckpt.tag}: HTTP, cache-busted")
            http_log = out / cfg.id / "logs" / f"p2-{ckpt.tag}-http.log"
            http_rc = run([*http_args, "--prompts", *prompts], http_log)
            if http_rc != 0:
                print(f"    HTTP benchmark exited {http_rc}")
                log_tail(http_log)
                ok = False

            print(f"  {ckpt.tag}: HTTP, prefix-cache hit (labelled row)")
            cache_prompts = [p for p in prompts if p in ("short", "medium", "long")]
            cache_log = out / cfg.id / "logs" / f"p2-{ckpt.tag}-http-cached.log"
            cache_rc = run([*http_args, "--no-nonce", "--tag", "cached",
                            "--prompts", *(cache_prompts or prompts)], cache_log)
            if cache_rc != 0:
                print(f"    cached HTTP benchmark exited {cache_rc}")
                log_tail(cache_log)
                ok = False
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    return ok


def phase_p2_optiq(cfg, out, trials, prompts, max_tokens, selected):
    """The engine arm: mlx-optiq serving an OptiQ checkpoint.

    For the baseline this arm was unmeasurable (P0.md: an 11.27 GB single Metal
    allocation). Attempting it per model keeps the roster honest: either the arm
    produces a number, or it records why it cannot.

    Returns {checkpoint tag: outcome}. "unmeasurable" (no runtime, server never
    ready, engine produced no measurement at all) is a result, not an error;
    "failed" and "partial" are. Unmeasurable arms leave unmeasurable.json next to
    where the row would have gone.
    """
    targets = [c for c in selected if "mlx-optiq" in c.engines]
    if not targets:
        print("  no OptiQ checkpoint selected — engine arm skipped")
        return {}
    if undownloaded(cfg, targets) or gates.refuse(cfg, targets, out):
        return {c.tag: "failed" for c in targets}
    exe = optiq_exe()
    version = optiq_version() if exe else "mlx-optiq not installed"

    def unmeasurable(ckpt, run_dir, reason, **extra):
        print(f"    {reason}; recorded as unmeasurable")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "unmeasurable.json").write_text(json.dumps({
            "model_id": cfg.id, "checkpoint": ckpt.tag, "engine": version,
            "outcome": "unmeasurable", "reason": reason, **extra,
        }, indent=2))
        return "unmeasurable"

    outcomes = {}
    for ckpt in targets:
        # The same directory bench.py will write into, so a row and the record of
        # why it could not be produced live together.
        run_dir = out / cfg.id / f"{ckpt.tag}-http-optiq-engine"
        if exe is None:
            print(f"  {ckpt.tag}: .venv-optiq not present")
            outcomes[ckpt.tag] = unmeasurable(ckpt, run_dir, "mlx-optiq runtime not installed (.venv-optiq)")
            continue
        port = free_port()
        print(f"  {ckpt.tag}: {exe.name} serve on :{port} ({version})")
        slog = out / cfg.id / "logs" / f"p2-{ckpt.tag}-optiq-engine.log"
        cmd = [str(exe), "serve", "--model", str(ckpt.local_path), "--port", str(port)]
        if exe.name == "python":
            cmd = [str(exe), "-m", "mlx_optiq", "serve", "--model", str(ckpt.local_path),
                   "--port", str(port)]
        slog.parent.mkdir(parents=True, exist_ok=True)
        with open(slog, "w") as server_log:
            try:
                server = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
            except OSError as e:
                outcomes[ckpt.tag] = unmeasurable(ckpt, run_dir, f"cannot launch server: {e}")
                continue
        try:
            if not wait_for_server(f"http://127.0.0.1:{port}", proc=server):
                log_tail(slog)
                outcomes[ckpt.tag] = unmeasurable(
                    ckpt, run_dir, "server did not answer /v1/models",
                    server_exit=server.poll(), log=str(slog))
                continue
            # --tag keeps the engine arm out of the plain mlx_lm HTTP row's
            # directory: two engines on one checkpoint are two rows, and one must
            # never overwrite the other.
            blog = out / cfg.id / "logs" / f"p2-{ckpt.tag}-optiq-engine-bench.log"
            rc = run([sys.executable, str(ROOT / "bench.py"),
                      "--model", cfg.id, "--checkpoint", ckpt.tag, "--transport", "http",
                      "--tag", "optiq-engine", "--engine", "mlx-optiq",
                      "--base-url", f"http://127.0.0.1:{port}", "--server-pid", str(server.pid),
                      "--server-version", version, "--trials", str(trials),
                      "--max-tokens", str(max_tokens), "--out", str(out),
                      "--prompts", *prompts], blog)
            if rc == 0:
                outcomes[ckpt.tag] = "completed"
                continue
            print(f"    engine arm benchmark exited {rc}; see {blog}")
            log_tail(blog)
            try:
                bench_outcome = json.loads((run_dir / "summary.json").read_text()).get("outcome")
            except (OSError, ValueError):
                bench_outcome = None  # bench died before summarising: nothing to trust
            if bench_outcome == "failed":
                # Not one trial succeeded, warm-up included: the engine cannot serve
                # this checkpoint here. The error rows stay in run_dir as evidence.
                outcomes[ckpt.tag] = unmeasurable(
                    ckpt, run_dir, f"bench exited {rc} with no successful trial",
                    tail=blog.read_text().strip().splitlines()[-3:])
            else:
                outcomes[ckpt.tag] = "partial" if bench_outcome == "partial" else "failed"
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    return outcomes


def phase_p3(cfg, out, selected):
    if undownloaded(cfg, selected) or gates.refuse(cfg, selected, out):
        return False
    tags = [c.tag for c in selected]
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
    ap.add_argument("--phase", choices=("fetch", "p0", "p2", "p2-optiq", "p3", "all"),
                    default="all")
    ap.add_argument("--tiers", nargs="+", default=["4bit", "optiq-4bit", "8bit"],
                    choices=("4bit", "optiq-4bit", "8bit"),
                    help="checkpoint tiers to run; 8bit is a reference tier")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = registry.get_model(args.model)
    selected = [c for c in cfg.checkpoints.values() if c.tier in args.tiers]
    out = Path(args.out)
    print(f"\n{cfg.id} · {cfg.name} · template {json.dumps(cfg.template_kwargs)}")
    print(f"engine {server_version_string()} · python {sys.version.split()[0]}")
    if args.phase in ("p2", "all") and args.trials < 5:
        print(f"warning: --trials {args.trials} breaks the median-of-5 reporting rule; "
              f"this row cannot be published as a result (fine for a smoke test)")

    phases = ["fetch", "p0", "p2", "p2-optiq", "p3"] if args.phase == "all" else [args.phase]
    ok = True
    for ph in phases:
        print(f"\n[{ph}]")   
        if ph == "fetch":
            ok = phase_fetch(cfg, out, selected)
        elif ph == "p0":
            ok = phase_p0(cfg, out, selected)
        elif ph == "p2":
            ok = phase_p2(cfg, out, args.trials, args.prompts, args.max_tokens,
                          server_version_string(), selected)
        elif ph == "p2-optiq":
            outcomes = phase_p2_optiq(cfg, out, args.trials, args.prompts, args.max_tokens, selected)
            print(f"  outcomes: {json.dumps(outcomes)}")
            ok = all(o in ("completed", "unmeasurable") for o in outcomes.values())
        elif ph == "p3":
            ok = phase_p3(cfg, out, selected)
        if not ok:
            print(f"\n[{ph}] did not complete cleanly — stopping here so the failure is cheap.")
            return 1

    print(f"\n{cfg.id}: {args.phase} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
