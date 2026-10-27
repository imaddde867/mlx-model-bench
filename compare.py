#!/usr/bin/env python3
"""Cross-model table (REPORT section 5) generated from the committed result rows.

  python compare.py                     # markdown to stdout, in-process arm
  python compare.py --transport http    # the HTTP arm instead
  python compare.py --out results/COMPARE.md

Every number comes from a summary.json or accuracy_summary.json under results/, so the
table cannot drift from the data. Each row carries model, tier, bpw, publisher, prompt
tokens and generation length (AGENTS.md: no number leaves the house without them).

The weight-read proxy is `weights_gb × decode tok/s`, assuming each generated token reads
the loaded checkpoint's file-size worth of weights once. It is not measured memory bandwidth:
it omits cache reuse, tensor metadata, activations and other traffic. For MoE (LFM2.5), only a
subset of weights is active per token, so the value is an upper bound rather than achieved
traffic; those cells are marked `~`.
"""

import argparse
import json
import sys
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent
TIER_ORDER = {"4bit": 0, "optiq-4bit": 1, "8bit": 2}
# params_active / params_total below this means decode reads far fewer bytes than the file holds.
SPARSE_RATIO = 0.8


def _med(prompts, name, metric):
    block = (prompts.get(name) or {}).get(metric)
    return block["median"] if block else None


def _rng(prompts, name, metric):
    block = (prompts.get(name) or {}).get(metric)
    return (block["min"], block["max"]) if block else None


def _accuracy(model_dir):
    """{checkpoint tag: 'passed/total'} from p3/accuracy_summary.json, old or new layout."""
    path = model_dir / "p3" / "accuracy_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for tag, entry in (data.get("checkpoints") or {}).items():
        acc = entry.get("accuracy") or {}
        out[tag] = f"{acc.get('total_passed')}/{acc.get('total_prompts')}"
    # the historical baseline layout: arm1 = 4-bit, arm3 = OptiQ
    for arm, tag in (("arm1", "mlx-4bit"), ("arm3", "optiq-4bit")):
        acc = (data.get(arm) or {}).get("accuracy")
        if acc and tag not in out:
            out[tag] = f"{acc['total_passed']}/{acc['total_prompts']}"
    return out


def _spread(run_dir):
    """Worst per-prompt TTFT max/min over the kept trials, read from the per-trial files.

    Reading the trials (not a summary field) means rows recorded before spread was
    summarised still get one, and a steady run reads 1.0x.
    """
    worst = None
    for prompt in {p.name.rsplit("-trial", 1)[0] for p in run_dir.glob("*-trial*.json")}:
        vals = []
        for f in run_dir.glob(f"{prompt}-trial*.json"):
            row = json.loads(f.read_text())
            if row.get("ok") and not row.get("discarded") and row.get("ttft_s"):
                vals.append(row["ttft_s"])
        if len(vals) >= 2:
            ratio = round(max(vals) / min(vals), 2)
            worst = ratio if worst is None else max(worst, ratio)
    return worst


def _identity(s, model_id, tag):
    """Summary metadata, falling back to registry.py for rows that predate it (the baseline)."""
    keys = ("model_name", "tier", "bits_per_weight", "publisher", "weights_gb", "params_total", "params_active")
    out = {k: s.get(k) for k in keys}
    if any(v is None for v in out.values()):
        try:
            cfg = registry.get_model(model_id)
            ckpt = cfg.checkpoint(tag)
        except Exception:
            return out
        fill = {"model_name": cfg.name, "tier": ckpt.tier, "bits_per_weight": ckpt.bits_per_weight,
                "publisher": ckpt.publisher, "weights_gb": ckpt.weights_gb,
                "params_total": cfg.params_total, "params_active": cfg.params_active}
        out = {k: (v if v is not None else fill[k]) for k, v in out.items()}
    return out


def build_repeat_rows(results_dir):
    """Decode-short confirmation runs, compared to the original in-process row.

    These tagged, single-prompt reruns are kept separate from the 12-row matrix: they
    confirm a narrow measurement and must not silently replace full-run summaries.
    """
    rows = []
    for model_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("*-inproc-decode-check/summary.json")):
            s = json.loads(path.read_text())
            prompt = (s.get("prompts") or {}).get("decode-short") or {}
            if s.get("outcome") != "completed" or s.get("publishable") is not True:
                continue
            tag = s.get("checkpoint") or path.parent.name.removesuffix("-inproc-decode-check")
            ident = _identity(s, model_dir.name, tag)
            original_path = model_dir / f"{tag}-inproc" / "summary.json"
            if not original_path.is_file():
                continue
            original = json.loads(original_path.read_text())
            original_prompt = (original.get("prompts") or {}).get("decode-short") or {}
            before = _med({"decode-short": original_prompt}, "decode-short", "decode_tps")
            after = _med({"decode-short": prompt}, "decode-short", "decode_tps")
            rows.append({
                "model": ident["model_name"] or model_dir.name,
                "checkpoint": tag,
                "bpw": ident["bits_per_weight"],
                "publisher": ident["publisher"],
                "prompt_tokens": prompt.get("prompt_tokens"),
                "generation_tokens": _med({"decode-short": prompt}, "decode-short", "generation_tokens"),
                "original_decode": before,
                "original_range": _rng({"decode-short": original_prompt}, "decode-short", "decode_tps"),
                "repeat_decode": after,
                "repeat_range": _rng({"decode-short": prompt}, "decode-short", "decode_tps"),
                "change_pct": round(100 * (after - before) / before, 1) if before and after else None,
                "original_ttft": _med({"decode-short": original_prompt}, "decode-short", "ttft_s"),
                "repeat_ttft": _med({"decode-short": prompt}, "decode-short", "ttft_s"),
                "summary_path": path.relative_to(Path(results_dir)).as_posix(),
                "condition_caveats": "; ".join(s.get("condition_caveats") or []),
            })
    rows.sort(key=lambda r: (r["model"], r["checkpoint"]))
    return rows


def build_rows(results_dir, transport="inproc"):
    rows = []
    for model_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        accuracy = _accuracy(model_dir)
        for path in sorted(model_dir.glob(f"*-{transport}/summary.json")):
            s = json.loads(path.read_text())
            pr = s.get("prompts") or {}
            tag = s.get("checkpoint") or path.parent.name.rsplit("-", 1)[0]
            ident = _identity(s, model_dir.name, tag)
            weights = ident["weights_gb"]
            short = _med(pr, "decode-short", "decode_tps")
            total, active = ident["params_total"], ident["params_active"]
            sparse = bool(total and active and active / total < SPARSE_RATIO)
            rows.append({
                "model": ident["model_name"] or model_dir.name,
                "tier": ident["tier"] or tag,
                "bpw": ident["bits_per_weight"],
                "publisher": ident["publisher"],
                "weights_gb": weights,
                "params": f"{total}B/{active}B active" if total and active else None,
                "sparse": sparse,
                "prompt_tokens_long": (pr.get("long") or {}).get("prompt_tokens"),
                "gen_tokens_decode_long": _med(pr, "decode-long", "generation_tokens"),
                "decode_short": short,
                "decode_short_range": _rng(pr, "decode-short", "decode_tps"),
                "decode_long": _med(pr, "decode-long", "decode_tps"),
                "ttft_long": _med(pr, "long", "ttft_s"),
                "peak_long_gb": _med(pr, "long", "peak_gb"),
                "gbps": round(weights * short, 1) if weights and short else None,
                "spread_ttft": _spread(path.parent),
                "condition": s.get("condition", "not recorded"),
                "accuracy": accuracy.get(tag),
                "outcome": s.get("outcome", "not recorded"),
                "publishable": s.get("publishable", "not recorded"),
            })
    rows.sort(key=lambda r: (r["model"], TIER_ORDER.get(r["tier"], 9)))
    return rows


def _fmt(v, spec="{}"):
    return "n/a" if v is None else spec.format(v)


def to_markdown(rows, transport="inproc", repeat_rows=()):
    head = ("| Model | Tier | bpw | Publisher | Params | Weights GB | Decode short tok/s [min-max] | "
            "Decode long tok/s | TTFT long s | Prompt tok (long) | Gen tok (decode-long) | "
            "Peak GB (long) | Weight-read proxy GB/s | TTFT spread | Condition | Acc /30 |")
    lines = [f"Transport: {transport}. Medians of 5 kept trials; spread = max/min of kept TTFT, "
             "'not recorded' = the row predates condition recording.",
             "Weight-read proxy = checkpoint weights GB × short-prompt decode tok/s; it is a size-based estimate, not measured memory bandwidth. `~` marks the MoE upper bound because only a subset of experts is active per token.", "", head,
             "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        rng = r["decode_short_range"]
        short = _fmt(r["decode_short"]) + (f" [{rng[0]}-{rng[1]}]" if rng else "")
        gbps = "n/a" if r["gbps"] is None else ("~" if r["sparse"] else "") + str(r["gbps"])
        lines.append("| " + " | ".join([
            r["model"], r["tier"], _fmt(r["bpw"]), _fmt(r["publisher"]), _fmt(r["params"]),
            _fmt(r["weights_gb"]), short, _fmt(r["decode_long"]), _fmt(r["ttft_long"]),
            _fmt(r["prompt_tokens_long"]), _fmt(r["gen_tokens_decode_long"]), _fmt(r["peak_long_gb"]),
            gbps, _fmt(r["spread_ttft"]), str(r["condition"]), _fmt(r["accuracy"]),
        ]) + " |")
    text = "\n".join(lines) + "\n"
    if repeat_rows and transport == "inproc":
        lines = [text.rstrip(), "", "Repeat decode-short checks (five kept trials; separate from full-matrix rows).", "",
                 "| Model | Checkpoint | bpw | Publisher | Prompt tok | Gen tok | Original decode tok/s [min-max] | "
                 "Repeat decode tok/s [min-max] | Change | Original / repeat TTFT s | Rerun summary | Recorded caveats |",
                 "|---|---|---:|---|---:|---:|---|---|---:|---|---|---|"]
        for r in repeat_rows:
            old_rng = r["original_range"]
            new_rng = r["repeat_range"]
            old_text = _fmt(r["original_decode"]) + (f" [{old_rng[0]}-{old_rng[1]}]" if old_rng else "")
            new_text = _fmt(r["repeat_decode"]) + (f" [{new_rng[0]}-{new_rng[1]}]" if new_rng else "")
            caveat = r["condition_caveats"].replace("|", "\\\\|") or "none recorded"
            lines.append("| " + " | ".join([
                r["model"], r["checkpoint"], _fmt(r["bpw"]), _fmt(r["publisher"]),
                _fmt(r["prompt_tokens"]), _fmt(r["generation_tokens"]), old_text, new_text,
                _fmt(r["change_pct"], "{:+.1f}%"),
                f"{_fmt(r['original_ttft'])} / {_fmt(r['repeat_ttft'])}",
                f"[summary.json]({r['summary_path']})", caveat,
            ]) + " |")
        text = "\n".join(lines) + "\n"
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--transport", choices=("inproc", "http"), default="inproc")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    args = ap.parse_args()
    repeat_rows = build_repeat_rows(args.results) if args.transport == "inproc" else []
    text = to_markdown(build_rows(args.results, args.transport), args.transport, repeat_rows)
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
