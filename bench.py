#!/usr/bin/env python3
"""P1 harness: one arm x fixed prompts x warm-up + N trials -> JSON per run.

In-process (arms 1, 3), run with the engine's own interpreter:
  .venv-mlxlm/bin/python bench.py --arm 1
  .venv-mlxlm/bin/python bench.py --arm 3 --trials 3

HTTP (arms 2, 4, 5), server started separately:
  .venv-mlxlm/bin/python bench.py --arm 2 --base-url http://127.0.0.1:8080 \
      --server-pid <pid> --server-version "mlx-lm 0.32.2" \
      --model-label "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit"

Arm 4/5 have no working transport yet; see P0.md. A failed arm records an
error row, it does not crash the run.

For HTTP rows `versions` is the client environment; the engine version is
`server_version`, which the operator must supply.
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

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
PRIMARY_TEMPLATE = {"enable_thinking": False, "conversational_mode": True}

ARMS = {
    "1": {"engine": "mlx_lm", "model": ROOT / "models/rapid-mlx-4bit"},
    "2": {"engine": "http"},
    "3": {"engine": "mlx_lm", "model": ROOT / "models/optiq-4bit"},
    "4": {"engine": "http"},
    "5": {"engine": "http"},
}


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
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens):
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
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template,
    }
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--prompts", nargs="+", default=["short", "medium", "long", "decode-short", "decode-long"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--template-kwargs", default=json.dumps(PRIMARY_TEMPLATE))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--server-pid", type=int, help="HTTP arms: server process to sample RSS from")
    ap.add_argument("--server-version", help="HTTP arms: engine version string of the server")
    ap.add_argument("--model-label", help="HTTP arms: checkpoint the server was started with")
    ap.add_argument("--no-nonce", action="store_true", help="send identical prompts (measures prefix-cache hits; label with --tag cached)")
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. think")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    template = json.loads(args.template_kwargs)
    arm = ARMS[args.arm]
    out_dir = Path(args.out) / (f"arm{args.arm}" + (f"-{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "arm": args.arm,
        "engine": arm["engine"],
        "transport": "in-process" if arm["engine"] == "mlx_lm" else "http",
        "host": f"{platform.machine()} macOS {platform.mac_ver()[0]} python {platform.python_version()}",
        "versions": versions(),
        "server_version": args.server_version,
        "max_tokens": args.max_tokens,
        "template_kwargs": template,
        "greedy": True,
        "trials": args.trials,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if arm["engine"] == "mlx_lm":
        import mlx.core as mx
        from mlx_lm import load

        t0 = time.perf_counter()
        try:
            model, tokenizer = load(str(arm["model"]))
        except Exception as e:
            row = {**base, "ok": False, "error": f"{type(e).__name__}: {e}",
                   "traceback_tail": traceback.format_exc().strip().splitlines()[-3:]}
            (out_dir / "load-error.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row))
            return 1
        base["model"] = str(arm["model"].relative_to(ROOT))
        base["load_s"] = round(time.perf_counter() - t0, 2)
        base["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
        run = lambda text, a: run_mlx(model, tokenizer, text, a, template)
    else:
        base["model"] = args.model_label
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
        **{k: base[k] for k in ("arm", "engine", "transport", "model", "host", "versions",
                                "server_version", "max_tokens", "template_kwargs", "trials")},
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
