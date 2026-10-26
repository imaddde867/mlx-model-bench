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

import gates
import registry
import sysinfo

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
MIN_PUBLISHABLE_TRIALS = 5  # AGENTS.md: median of 5 and the range
# A trial without these is not a measurement. decode_tps and peak_gb are
# legitimately None for one-token answers / HTTP without --server-pid.
REQUIRED_METRICS = ("prompt_tokens", "generation_tokens", "ttft_s", "prefill_tps", "wall_s")

# Timing boundaries (issue #6). Historical keys keep their original meaning;
# results/ rows already use them. Revised numbers live under new names.
#
# Events. "first token": in-process, the first stream_generate yield (every
# yield is one token); HTTP, the first SSE chunk whose delta.content or
# delta.reasoning is non-empty ("first visible delta" -- a token the server
# strips or has not finished decoding produces no chunk, so this can lag the
# engine's first token). "generation end": in-process, the final yield (the EOS
# or max_tokens token); HTTP, the chunk carrying finish_reason (falls back to the
# last visible delta). Both transports count the EOS token in generation_tokens.
#
# Historical (unchanged, scopes differ by transport -- see ttft_scope):
#   ttft_s       inproc: before chat template -> first token
#                http:   after JSON serialization, request send -> first visible delta
#   prefill_tps  prompt_tokens / ttft_s (so it inherits ttft_s's scope)
#   decode_tps   (gen-1) / (first token -> last token-bearing event); over HTTP the
#                last event is the last visible delta, which on a "stop" finish
#                precedes the EOS token, so it divides gen-1 tokens by gen-2 steps
#   wall_s       inproc: before chat template -> stream exhausted
#                http:   request send -> [DONE] (end-to-end user latency)
# Revised (same boundaries on both transports unless marked):
#   request_ttft_s     messages in hand (before chat template / before JSON
#                      serialization) -> first token. The cross-transport TTFT;
#                      http minus inproc = serialization + transport + server.
#   prep_s             inproc only: chat template rendering. None over HTTP
#                      (the server does it; not observable from the client).
#   engine_ttft_s      inproc only: stream_generate call -> first token
#                      (tokenize + prefill + first sample). None over HTTP.
#   first_content_s    http only: messages in hand -> first non-empty
#                      delta.content; later than request_ttft_s when the stream
#                      is reasoning-first, None when reasoning-only.
#   first_token_kind   http only: "reasoning" or "content", whichever came first.
#   cached_prompt_tokens  prompt tokens served from a prefix cache: 0 in-process
#                      (no prompt_cache is passed), usage.prompt_tokens_details.
#                      cached_tokens over HTTP (None if the server omits it).
#   decode_tokens      generation_tokens - 1 (tokens after the first)
#   decode_s           first token -> generation end
#   decode_window_tps  decode_tokens / decode_s; None whenever decode_flags is
#                      non-empty, so a flagged row cannot be ranked.
#   decode_flags       no_token_event | no_token_count | too_short (decode_tokens <
#                      decode_min_tokens) | no_decode_window | prefix_cache_hit
MIN_DECODE_TOKENS = 32
TTFT_SCOPE = {
    "inproc": "inproc: chat template + tokenize + prefill + first token; no transport",
    "http": "http: request send + transport + server template/tokenize/prefill + first visible "
            "SSE delta; excludes client JSON serialization",
}


def decode_window(t_req, first, end, gen, cached=None, nonce=True):
    """Revised timing fields from one event timeline (see the block above)."""
    decode_tokens = gen - 1 if gen else None
    decode_s = end - first if first is not None and end is not None else None
    flags = []
    if first is None:
        flags.append("no_token_event")
    if not gen:
        flags.append("no_token_count")
    elif decode_tokens < MIN_DECODE_TOKENS:
        flags.append("too_short")
    if decode_s is None or decode_s <= 0:
        flags.append("no_decode_window")
    if nonce and cached:
        flags.append("prefix_cache_hit")
    return {
        "request_ttft_s": round(first - t_req, 4) if first is not None else None,
        "cached_prompt_tokens": cached,
        "decode_tokens": decode_tokens,
        "decode_s": round(decode_s, 4) if decode_s is not None else None,
        "decode_window_tps": None if flags else round(decode_tokens / decode_s, 1),
        "decode_flags": flags,
    }


# peak_gb is only comparable between rows with the same source. Neither is GPU
# VRAM: Apple Silicon has one unified pool. weights_gb is file size, not runtime.
MEMORY_BASES = {
    "mx_peak": {"method": "mx.get_peak_memory()", "scope": "in-process MLX allocator (Metal buffers)",
                "phase": "per-trial: reset before prompt prefill, read after generation"},
    "server_rss": {"method": "ps -o rss= polled every 0.1 s", "scope": "server process resident set (whole process)",
                   "phase": "per-request: sampler runs while the request streams; includes the loaded weights"},
}
MEMORY_UNIT = "GB, decimal (1e9 bytes)"


def memory_record(source, value_gb, reason=None):
    """Provenance for peak_gb. source None = not measured; reason says why."""
    basis = MEMORY_BASES.get(source, {})
    return {"value_gb": value_gb, "source": source, "unit": MEMORY_UNIT,
            "method": basis.get("method"), "scope": basis.get("scope"), "phase": basis.get("phase"),
            "available": value_gb is not None,
            "unavailable_reason": None if value_gb is not None else (reason or "not measured")}


def memory_source(summary):
    """Recorded source of a summary's peak_gb; legacy summaries are inferred from transport."""
    if "memory" in summary:  # new format: None means mixed or unmeasured, never inferred
        return summary["memory"].get("source"), False
    return {"inproc": "mx_peak", "http": "server_rss"}.get(summary.get("transport")), True


def memory_label(summary):
    """Legend suffix from recorded provenance; legacy summaries are marked inferred."""
    src, inferred = memory_source(summary)
    if src is None and len((summary.get("memory") or {}).get("sources_seen") or []) > 1:
        return "mixed sources"
    names = {"mx_peak": "MLX allocator peak", "server_rss": "server RSS", None: "memory n/a"}
    return names.get(src, src) + (" (inferred)" if inferred and src else "")


def memory_delta(a, b):
    """b - a in GB, or None when the two summaries are not on the same memory basis."""
    (sa, _), (sb, _) = memory_source(a), memory_source(b)
    va, vb = a.get("peak_gb"), b.get("peak_gb")
    if sa is None or sa != sb or va is None or vb is None:
        return None
    return round(vb - va, 3)


def host_string():
    return f"{platform.machine()} macOS {platform.mac_ver()[0]} python {platform.python_version()}"


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


def spread_ratio(values):
    """max / min over kept trials; 1.0 is perfectly steady. None when it cannot be computed.

    A 5-trial median hides a run whose trials went 6.9, 6.2, 5.1, 2.3, 1.8 s. The ratio
    makes that visible in the summary instead of only in the per-trial files.
    """
    values = [v for v in values if v]
    if len(values) < 2:
        return None
    return round(max(values) / min(values), 2)


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
    t_engine = time.perf_counter()
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
        "memory": memory_record("mx_peak", round(mx.get_peak_memory() / 1e9, 3)),
        "finish_reason": resp.finish_reason,
        "text": "".join(parts).strip(),
        "prep_s": round(t_engine - t0, 4),
        "engine_ttft_s": round(first - t_engine, 4),
        "first_content_s": None,
        "first_token_kind": None,
        **decode_window(t0, first, last, gen, cached=0),
    }


def run_http(text, args, template):
    t_req = time.perf_counter()
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
    first = last = end = first_content = first_kind = None
    usage = finish = None
    parts = []
    reasoning_parts = []
    done = False
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(f"server reported an error mid-stream: {chunk['error']}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
                    end = time.perf_counter()
                # mlx_lm.server splits a template's thinking block into
                # `delta.reasoning` and leaves `delta.content` null while the model
                # is still thinking. A row that only counted content would both
                # drop those tokens from the text and hard-fail on a
                # reasoning-only answer, so both streams are timed and recorded.
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                thinking = delta.get("reasoning")
                if text or thinking:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        first_kind = "reasoning" if thinking else "content"
                    if text and first_content is None:
                        first_content = now
                    last = now
                    if text:
                        parts.append(text)
                    if thinking:
                        reasoning_parts.append(thinking)
    finally:
        peak = sampler.stop() if sampler else None
    wall = time.perf_counter() - t0

    if not done:
        raise RuntimeError("stream ended without [DONE]; response is truncated")
    # A usage-only stream (no visible delta) is returned, not raised: its token
    # accounting is real, and main() fails the trial on the missing ttft_s.

    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    gen = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen,
        "ttft_s": round(first - t0, 4) if first is not None else None,
        "prefill_tps": round(prompt_tokens / (first - t0), 1) if prompt_tokens and first is not None else None,
        "engine_prefill_tps": None,
        "decode_tps": round((gen - 1) / (last - first), 1) if gen and gen > 1 and first is not None and last > first else None,
        "wall_s": round(wall, 3),
        "peak_gb": round(peak, 3) if peak else None,
        "peak_gb_source": "server_rss" if peak else None,
        "memory": memory_record(
            "server_rss" if peak else None, round(peak, 3) if peak else None,
            None if peak else "no --server-pid" if sampler is None else "sampler read no RSS"),
        "finish_reason": finish,
        "text": "".join(parts).strip(),
        "reasoning_chars": len("".join(reasoning_parts)),
        "reasoning_only": not parts and bool(reasoning_parts),
        "prep_s": None,
        "engine_ttft_s": None,
        "first_content_s": round(first_content - t_req, 4) if first_content is not None else None,
        "first_token_kind": first_kind,
        **decode_window(t_req, first, end if end is not None else last, gen,
                        cached=cached, nonce=not getattr(args, "no_nonce", False)),
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
    ap.add_argument("--engine", default="mlx_lm",
                    help="engine that produced this row; mlx-optiq rows are their own row, never a replacement")
    ap.add_argument("--no-nonce", action="store_true",
                    help="send identical prompts (measures prefix-cache hits; label with --tag cached)")
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. cached")
    ap.add_argument("--condition", choices=("clean", "loaded", "unspecified"), default="unspecified",
                    help="recorded in every row: 'clean' = nothing else running, 'loaded' = a "
                         "realistic desktop (browser, chat apps, CLIs) was open. Does not change the "
                         "output directory; the per-trial system snapshot says what was actually busy")
    ap.add_argument("--max-spread", type=float, default=None,
                    help="if set, a prompt whose kept-trial TTFT or decode max/min exceeds this ratio "
                         "makes the summary non-publishable (e.g. 1.5). Off by default: loaded runs "
                         "are legitimate, they just have to be visible")
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
    if gates.refuse(cfg, [ckpt], args.out):
        print("refusing to measure: provenance/P0 gate failed", file=sys.stderr)
        return 1

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
        "engine": args.engine,
        "host": host_string(),
        "versions": versions(),
        "server_version": args.server_version,
        "max_tokens": args.max_tokens,
        "template_kwargs": template,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "greedy": args.temperature == 0.0,
        "trials": args.trials,
        "ttft_scope": TTFT_SCOPE[args.transport],
        "decode_min_tokens": MIN_DECODE_TOKENS,
        "condition": args.condition,
        # Run start. Per-trial times are `started_at` on each row; before this the
        # same run-start stamp was copied onto every trial, so trials could not be
        # lined up against what else the machine was doing.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.transport == "inproc":
        import mlx.core as mx
        from mlx_lm import load

        if not ckpt.available:
            row = {**base, "ok": False, "outcome": "failed",
                   "error": f"checkpoint not downloaded: {ckpt.local_dir}"}
            (out_dir / "load-error.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row))
            return 1

        t0 = time.perf_counter()
        try:
            model, tokenizer = load(str(ckpt.local_path),
                                    tokenizer_config={"trust_remote_code": False})
        except Exception as e:
            row = {**base, "ok": False, "outcome": "failed", "error": f"{type(e).__name__}: {e}",
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
            # Snapshot BEFORE the timed window opens, so it cannot slow the trial it describes.
            row["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            row["system"] = sysinfo.snapshot()
            try:
                row.update(run(run_text, args))
                missing = [m for m in REQUIRED_METRICS if row.get(m) is None]
                if missing:
                    raise RuntimeError(f"missing required metrics: {', '.join(missing)}")
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
                                "temperature", "top_p", "greedy", "trials",
                                "ttft_scope", "decode_min_tokens")},
        "load_s": base.get("load_s"),
        "load_peak_gb": base.get("load_peak_gb"),
        "load_peak_gb_source": "mx_peak" if "load_peak_gb" in base else None,
        "base_url": base.get("base_url"),
        "runs_ok": sum(1 for r in kept if r.get("ok")),
        "runs_failed": sum(1 for r in rows if not r.get("ok")),
        "reasoning_only_runs": sum(1 for r in kept if r.get("reasoning_only")),
        "prompts": {},
    }
    problems = []
    if args.trials < MIN_PUBLISHABLE_TRIALS:
        problems.append(f"{args.trials} trials < {MIN_PUBLISHABLE_TRIALS} (smoke run)")
    for name in args.prompts:
        got = [r for r in kept if r["prompt"] == name and r.get("ok")]
        warm_ok = any(r["discarded"] and r.get("ok") for r in rows if r["prompt"] == name)
        summary["prompts"][name] = {
            "warmup_ok": warm_ok,
            "kept_ok": len(got),
            "kept_expected": args.trials,
            "prompt_tokens": next((r["prompt_tokens"] for r in got), None),
            **{m: stats([r.get(m) for r in got])
               for m in ("ttft_s", "prefill_tps", "decode_tps", "generation_tokens", "peak_gb", "wall_s",
                         "request_ttft_s", "prep_s", "engine_ttft_s", "first_content_s",
                         "cached_prompt_tokens", "decode_window_tps")},
            # decode_window_tps is None on these rows, so its stats exclude them
            "decode_flagged": sum(1 for r in got if r.get("decode_flags")),
            "spread_ratio": {
                "ttft_s": spread_ratio([r.get("ttft_s") for r in got]),
                "decode_tps": spread_ratio([r.get("decode_tps") for r in got]),
            },
        }
        if args.max_spread:
            wide = {k: v for k, v in summary["prompts"][name]["spread_ratio"].items()
                    if v is not None and v > args.max_spread}
            if wide:
                problems.append(f"{name}: unstable trials, max/min {wide} > {args.max_spread}")
        if not warm_ok:
            problems.append(f"{name}: warm-up did not succeed")
        if len(got) < args.trials:
            problems.append(f"{name}: {len(got)}/{args.trials} kept trials succeeded")
    # completed = everything requested was measured; partial = some measurement
    # exists next to failures; failed = nothing usable. "unmeasurable" (runtime
    # absent, engine cannot serve the checkpoint) is decided by run_bench.py.
    complete = not summary["runs_failed"] and not any(
        not p["warmup_ok"] or p["kept_ok"] < args.trials for p in summary["prompts"].values())
    # One basis per run: a mixed or absent set is recorded, never averaged over.
    mem = [r["memory"] for r in kept if r.get("ok") and r.get("memory")]
    srcs = sorted({m["source"] for m in mem if m["source"]})
    one = MEMORY_BASES.get(srcs[0], {}) if len(srcs) == 1 else {}
    summary["memory"] = {
        "source": srcs[0] if len(srcs) == 1 else None, "sources_seen": srcs, "unit": MEMORY_UNIT,
        "method": one.get("method"), "scope": one.get("scope"), "phase": one.get("phase"),
        "unavailable_rows": sum(1 for m in mem if not m["available"]),
        "weights_gb_is": "checkpoint file size, not runtime memory",
    }
    # Conditions: the run's label plus every distinct caveat any kept trial's snapshot raised.
    summary["condition"] = args.condition
    summary["condition_caveats"] = sorted({w for r in kept if r.get("system")
                                           for w in sysinfo.warnings(r["system"])})
    summary["outcome"] = ("completed" if complete
                          else "partial" if any(r.get("ok") for r in rows) else "failed")
    summary["publishable"] = complete and not problems
    summary["non_publishable_reasons"] = problems
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(out_dir / "summary.json"), **summary}))
    return 0 if summary["outcome"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
