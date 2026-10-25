#!/usr/bin/env python3
"""Chart every model on the same axes: TTFT, prefill, decode, peak memory.

Reads results/<model>/<checkpoint>-<transport>/summary.json — no model list is
hardcoded, so a model appears on this chart as soon as its P2 run finishes.

  .venv-mlxlm/bin/python visualize_p2.py                    # in-process only
  .venv-mlxlm/bin/python visualize_p2.py --include-http
  .venv-mlxlm/bin/python visualize_p2.py --models minicpm5-2b spark-x2.5-4b

Prompt-token counts differ per tokenizer; each panel says so and the x-axis
shows the pooled median. The per-model count is in each summary.json.
"""

import argparse
import json
import statistics
from pathlib import Path

import bench
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# One hue per model; the mixed-bit checkpoint wears the same hue with a hatch.
COLORS = ["#2e6f9e", "#2e9e5b", "#c9772e", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d"]

PANELS = [  # (metric, title, prompts, log, annotate, skip_cached)
    ("ttft_s", "TTFT, seconds (lower is better)", ["short", "medium", "long"], True, "%.3g", False),
    # Prefill throughput on a cache hit is an artifact (it divides real tokens by
    # cached-prefix time), so cached rows are excluded from this panel on purpose.
    ("prefill_tps", "Prefill tok/s (higher is better)", ["short", "medium", "long"], True, "%.0f", True),
    ("decode_tps", "Decode tok/s, prefill excluded (higher is better)", ["decode-short", "decode-long"], False, "%.0f", False),
    ("peak_gb", "Peak memory, GB decimal (lower is better)", ["short", "long"], False, "%.2f", False),
]


def checkpoint_tags():
    try:
        import registry
        return sorted({c.tag for m in registry.list_models() for c in m.checkpoints.values()},
                      key=len, reverse=True)
    except Exception:
        return ["mlx-4bit", "optiq-4bit"]


def discover(models, include_http, results_dir):
    """Run dirs are results/<model-id>/<checkpoint>-<transport>[-cached].

    Summaries written before the refactor predate model_id/checkpoint, so the
    directory name is the fallback — it is the same convention either way.
    """
    tags = checkpoint_tags()
    out = []
    for path in sorted(Path(results_dir).glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        model_id = s.get("model_id") or path.parent.parent.name
        if models and model_id not in models:
            continue
        name = path.parent.name
        ckpt = s.get("checkpoint")
        if not ckpt:
            ckpt = next((t for t in tags if name.startswith(t + "-")), name.split("-")[0])
        rest = name[len(ckpt) + 1:] if name.startswith(ckpt + "-") else ""
        cached = rest.endswith("cached")
        transport = s.get("transport") or rest.replace("-cached", "") or "inproc"
        if transport == "http" and not include_http:
            continue
        out.append({"path": path, "summary": s, "model_id": model_id,
                    "checkpoint": ckpt, "transport": transport, "cached": cached})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="restrict to these model ids")
    ap.add_argument("--include-http", action="store_true", help="also plot HTTP and cached rows")
    ap.add_argument("--results", default=str(ROOT / "results"), help="results root to read")
    ap.add_argument("--out", default=str(ROOT / "p2_visual.png"))
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    series = discover(args.models, args.include_http, args.results)
    if not series:
        raise SystemExit("no results/*/*/summary.json found — run run_bench.py --phase p2 first")

    # Stable order: registry order, plain checkpoint before mixed-bit.
    order = {"nemotron-3.5-bolt": 0, "minicpm5-2b": 1, "spark-x2.5-4b": 2, "lfm2.5-8b-a1b": 3}
    series.sort(key=lambda s: (order.get(s["model_id"], 99), s["checkpoint"], s["transport"]))

    # Memory gets one panel per recorded source: allocator peaks and server RSS
    # never share an axis, so they cannot be ranked against each other.
    panels = []
    for pn in PANELS:
        if pn[0] != "peak_gb":
            panels.append((*pn, None))
        else:
            panels += [(*pn, lab) for lab in sorted({bench.memory_label(s["summary"]) for s in series})]
    rows_n = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows_n, 2, figsize=(19, 5.25 * rows_n), squeeze=False)
    axes = axes.ravel()
    for extra in axes[len(panels):]:
        extra.set_visible(False)

    for ax, (metric, title, prompts, log, fmt, skip_cached, only) in zip(axes, panels):
        panel = [s for s in series if not (skip_cached and s["cached"])
                 and (only is None or bench.memory_label(s["summary"]) == only)]
        if only:
            title += f" [{only}]"
        prompts = [p for p in prompts if any(s["summary"]["prompts"].get(p, {}).get(metric) for s in panel)]
        if not prompts:
            ax.set_visible(False)
            continue
        w = 0.8 / len(panel)
        for i, s in enumerate(panel):
            xs, med, lo, hi = [], [], [], []
            for j, p in enumerate(prompts):
                m = s["summary"]["prompts"].get(p, {}).get(metric)
                if not m:
                    continue
                xs.append(j + (i - (len(panel) - 1) / 2) * w)
                med.append(m["median"])
                lo.append(m["median"] - m["min"])
                hi.append(m["max"] - m["median"])
            colour = COLORS[order.get(s["model_id"], 3) % len(COLORS)]
            label = f"{s['model_id']} · {s['checkpoint']}"
            engine = s["summary"].get("engine")
            if engine and engine != "mlx_lm":
                label += f" · {engine}"
            if s["cached"]:
                hatch, label = "\\\\", label + " · cache hit"
            elif s["transport"] == "http":
                hatch, label = "..", label + " · http"
            elif s["checkpoint"].startswith("optiq"):
                hatch = "//"
            else:
                hatch = ""
            ax.bar(xs, med, w * 0.92, yerr=[lo, hi], capsize=2.5, color=colour,
                   hatch=hatch, edgecolor="white", label=label)
            for x, v in zip(xs, med):
                ax.text(x, v, fmt % v, ha="center", va="bottom", fontsize=6.5, rotation=90)
        xt, xl = [], []
        for j, p in enumerate(prompts):
            # Cached rows only carry the TTFT prompts, so every lookup here is guarded.
            counts = [s["summary"]["prompts"].get(p, {}).get("prompt_tokens") for s in panel]
            counts = [c for c in counts if c]
            gen = [s["summary"]["prompts"].get(p, {}).get("generation_tokens") for s in panel]
            gen_med = statistics.median([g["median"] for g in gen if g]) if any(gen) else 0
            xt.append(j)
            xl.append(f"{p}\n≈{int(statistics.median(counts)):,} prompt tok\n≈{gen_med:.0f} gen tok")
        ax.set_xticks(xt, xl, fontsize=8)
        ax.set_title(title, fontsize=11)
        if log:
            ax.set_yscale("log")
        if metric == "peak_gb":
            ax.text(0.01, 0.98, "Not comparable with other memory panels (different basis; neither is GPU VRAM).\n"
                    "weights_gb is file size, not runtime memory.",
                    transform=ax.transAxes, va="top", fontsize=7.5)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    fig.legend(handles.values(), handles.keys(), ncol=min(4, len(handles)),
               fontsize=8.5, loc="lower center", frameon=False)
    fig.suptitle(
        "Nemotron 3.5 Bolt E4B and challengers on M4 / 16 GB · greedy, thinking off, "
        "median with min-max whiskers (warm-up discarded)\n"
        "prompt-token counts are each model's own tokenizer — compare within a bar group, "
        "not across the x-axis",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}  ({len(series)} series)")


if __name__ == "__main__":
    main()
