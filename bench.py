#!/usr/bin/env python3
"""P1 harness: one model x one checkpoint x one transport x fixed prompts -> JSON per run.

Everything model-specific comes from registry.py. Nothing here knows which
model it is benchmarking.

In-process, run with the engine's own interpreter:

  .venv-mlxlm/bin/python bench.py --model nemotron-3.5-bolt --checkpoint mlx-4bit
  .venv-mlxlm/bin/python bench.py --model minicpm5-2b --checkpoint optiq-4bit

HTTP, server started separately (mlx_lm.server on the same checkpoint):

  .venv-mlxlm/bin/python bench.py --model minicpm5-2b --transport http \\
      --server-pid <pid> --server-version "mlx-lm 0.32.2"

  # prefix-cache measurement, labelled as such
  ... --transport http --no-nonce --tag cached

Output lands in results/<model-id>/<checkpoint>-<transport>[-<tag>]/.
For HTTP rows `versions` is the client environment; the engine version is
`server_version`, which the operator must supply.

A failed run records an error row, it does not crash the run.
"""

import argparse
import importlib.metadata as md
import json
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"


def versions():
    out = {}
    for pkg in ("mlx", "mlx-lm", "mlx-optiq", "transformers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


class RssSampler:
    """Peak RSS of a server process; mx.get_peak_memory() cannot see another process."""

    def __init__(self, pid, interval=0.1):
        self.pid = str(pid)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ps", "-o", "rss=", "-p", self.pid],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                if out:
                    self.peak = max(self.peak, int(out))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak * 1024 / 1e9 if self.peak else None


def sampling_kwargs(args):
    """Greedy is the default; a card-recommended sampling row must be labelled."""
    kw = {}
    if args.temperature:
        kw["temp"] = args.temperature
    if args.top_p:
        kw["top_p"] = args.top_p
    return kw


def run_mlx(model, tokenizer, text, args, template):
    import mlx.core as mx
    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
        **template,
    )
    mx.reset_peak_memory()  # peak is per run, not cumulative across trials
    first = last = None
    parts = []
    resp = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                                **sampling_kwargs(args)):
        now = time.perf_counter()
        if first is None:
            first = now
        last = now
        parts.append(resp.text)
    wall = time.perf_counter() - t0

    gen = resp.generation_tokens
    return {
        "prompt_tokens": resp.prompt_tokens,
        "generation_tokens": gen,
        "ttft_s": round(first - t0, 4),
        "prefill_tps": round(resp.prompt_tokens / (first - t0), 1),
        "engine_prefill_tps": round(resp.prompt_tps, 1),
        "decode_tps": round((gen - 1) / (last - first), 1) if gen > 1 and last > first else None,
        "wall_s": round(wall, 3),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        "peak_gb_source": "mx_peak",
        "finish_reason": resp.finish_reason,
        "text": "".join(parts).strip(),
    }


def run_http(text, args, template):
    # Omit "model": mlx_lm.server falls back to its loaded checkpoint. Passing an
    # id from /v1/models is unsafe there (it lists every known alias, not the served one).
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template,
    }
    if args.top_p:
        body["top_p"] = args.top_p
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    sampler = RssSampler(args.server_pid) if args.server_pid else None
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    first = last = None
    usage = finish = None
    parts = []
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                finish = choices[0].get("finish_reason") or finish
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    parts.append(delta)
    finally:
        peak = sampler.stop() if sampler else None
    wall = time.perf_counter() - t0

    if first is None:
        raise RuntimeError("no content chunk in stream")

    prompt_tokens = (usage or {}).get("prompt_tokens")
    gen = (usage or {}).get("completion_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen,
        "ttft_s": round(first - t0, 4),
        "prefill_tps": round(prompt_tokens / (first - t0), 1) if prompt_tokens else None,
        "engine_prefill_tps": None,
        "decode_tps": round((gen - 1) / (last - first), 1) if gen and gen > 1 and last > first else None,
        "wall_s": round(wall, 3),
        "peak_gb": round(peak, 3) if peak else None,
        "peak_gb_source": "server_rss" if peak else None,
        "finish_reason": finish,
        "text": "".join(parts).strip(),
    }


def print_list():
    for m in registry.list_models():
        print(f"\n{m.id}  ({m.name}, {m.vendor})")
        for c in m.checkpoints.values():
            state = "downloaded" if c.available else "missing"
            print(f"  {c.tag:11s} {c.weights_gb:5.2f}GB {c.bits_per_weight:4.2f}bpw  {state:10s} {c.hf_repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show the registry and which checkpoints are local")
    ap.add_argument("--model", help="registry id, HF repo or local path")
    ap.add_argument("--checkpoint", help="checkpoint tag within the model (default: the first one)")
    ap.add_argument("--transport", choices=("inproc", "http"), default="inproc")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--prompts", nargs="+", default=["short", "medium", "long", "decode-short", "decode-long"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="default 0.0 = greedy. Any other value makes this a deviation row")
    ap.add_argument("--top-p", type=float, default=None,
                    help="only meaningful with --temperature > 0")
    ap.add_argument("--template-kwargs", default=None,
                    help="override the registry's chat template controls, as JSON")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--server-pid", type=int, help="HTTP: server process to sample RSS from")
    ap.add_argument("--server-version", help="HTTP: engine version string of the server")
    ap.add_argument("--no-nonce", action="store_true",
                    help="send identical prompts (measures prefix-cache hits; label with --tag cached)")
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. cached")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    if args.list:
        print_list()
        return 0
    if not args.model:
        ap.error("--model is required (or use --list)")

    cfg = registry.get_model(args.model)
    ckpt = cfg.checkpoint(args.checkpoint)
    template = json.loads(args.template_kwargs) if args.template_kwargs else cfg.template_kwargs

    # A non-greedy run must never land in the greedy row's directory: it is a
    # labelled deviation, and silently replacing a published summary is the one
    # thing a harness should be incapable of.
    sampling_tag = ""
    if args.temperature:
        sampling_tag = f"-t{args.temperature:g}"
        if args.top_p:
            sampling_tag += f"-p{args.top_p:g}"
    run_id = f"{ckpt.tag}-{args.transport}{sampling_tag}" + (f"-{args.tag}" if args.tag else "")
    out_dir = Path(args.out) / cfg.id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "model_id": cfg.id,
        "model_name": cfg.name,
        "hf_repo": ckpt.hf_repo,
        "checkpoint": ckpt.tag,
        "tier": ckpt.tier,
        "official": ckpt.official,
        "weights_gb": ckpt.weights_gb,
        "bits_per_weight": ckpt.bits_per_weight,
        "publisher": ckpt.publisher,
        "params_total": cfg.params_total,
        "params_active": cfg.params_active,
        "run_id": run_id,
        "transport": args.transport,
        "engine": "mlx_lm",
        "host": f"{platform.machine()} macOS {platform.mac_ver()[0]} python {platform.python_version()}",
        "versions": versions(),
        "server_version": args.server_version,
        "max_tokens": args.max_tokens,
        "template_kwargs": template,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "greedy": args.temperature == 0.0,
        "trials": args.trials,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.transport == "inproc":
        import mlx.core as mx
        from mlx_lm import load

        if not ckpt.available:
            row = {**base, "ok": False, "error": f"checkpoint not downloaded: {ckpt.local_dir}"}
            (out_dir / "load-error.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row))
            return 1

        t0 = time.perf_counter()
        try:
            model, tokenizer = load(str(ckpt.local_path),
                                    tokenizer_config={"trust_remote_code": False})
        except Exception as e:
            row = {**base, "ok": False, "error": f"{type(e).__name__}: {e}",
                   "traceback_tail": traceback.format_exc().strip().splitlines()[-3:]}
            (out_dir / "load-error.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row))
            return 1
        base["model"] = ckpt.local_dir
        base["load_s"] = round(time.perf_counter() - t0, 2)
        base["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
        run = lambda text, a: run_mlx(model, tokenizer, text, a, template)
    else:
        base["model"] = ckpt.hf_repo or ckpt.local_dir
        base["base_url"] = args.base_url
        run = lambda text, a: run_http(text, a, template)

    rows = []
    for name in args.prompts:
        path = Path(name)
        if not path.suffix:
            path = PROMPTS / "speed" / f"{name}.txt"
        text = path.read_text()

        for i in range(args.trials + 1):
            discarded = i == 0
            label = "warmup" if discarded else f"trial{i}"
            row = {
                **base,
                "prompt": path.stem,
                "prompt_file": str(path),
                "trial": i,
                "discarded": discarded,
            }
            # Unique leading nonce defeats server-side prefix caches (mlx_lm.server reuses
            # KV across identical prompts, which turned 8k-token prefill into ~0.09 s).
            run_text = text if args.no_nonce else f"[{random.randrange(10**6):06d}] {text}"
            row["nonce"] = not args.no_nonce
            try:
                row.update(run(run_text, args))
                row["ok"] = True
            except Exception as e:
                row["ok"] = False
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]

            (out_dir / f"{path.stem}-{label}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row))
            rows.append(row)
            if not row["ok"]:
                break  # the remaining trials would hit the same wall

    kept = [r for r in rows if not r["discarded"]]
    summary = {
        **{k: base[k] for k in ("model_id", "model_name", "hf_repo", "checkpoint", "tier",
                                "official", "weights_gb", "bits_per_weight", "publisher",
                                "params_total", "params_active",
                                "run_id", "transport", "engine", "model", "host", "versions",
                                "server_version", "max_tokens", "template_kwargs",
                                "temperature", "top_p", "greedy", "trials")},
        "load_s": base.get("load_s"),
        "load_peak_gb": base.get("load_peak_gb"),
        "base_url": base.get("base_url"),
        "runs_ok": sum(1 for r in kept if r.get("ok")),
        "runs_failed": sum(1 for r in kept if not r.get("ok")),
        "prompts": {},
    }
    for name in args.prompts:
        got = [r for r in kept if r["prompt"] == name and r.get("ok")]
        summary["prompts"][name] = {
            "prompt_tokens": next((r["prompt_tokens"] for r in got), None),
            **{m: stats([r.get(m) for r in got])
               for m in ("ttft_s", "prefill_tps", "decode_tps", "generation_tokens", "peak_gb", "wall_s")},
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(out_dir / "summary.json"), **summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
