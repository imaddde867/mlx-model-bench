#!/usr/bin/env python3
"""P2 chart: median bars, min-max whiskers, grouped by prompt. Arm 2 'cached' is a separate, hatched bar."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERIES = [  # (results dir, label, colour, hatch)
    ("arm1", "Arm 1  mlx_lm in-process", "#2e9e5b", ""),
    ("arm2", "Arm 2  mlx_lm.server (cache-busted)", "#3b82c4", ""),
    ("arm3", "Arm 3  OptiQ in-process", "#d9534f", ""),
    ("arm2-cached", "Arm 2  server, prefix-cache hit", "#3b82c4", "//"),
]
series = [(d, l, c, h, json.loads((ROOT / "results" / d / "summary.json").read_text()))
          for d, l, c, h in SERIES if (ROOT / "results" / d / "summary.json").exists()]

PANELS = [  # (metric, title, prompts, log)
    ("ttft_s", "TTFT, seconds (lower is better)", ["short", "medium", "long"], True),
    ("prefill_tps", "Prefill tok/s, log scale (higher is better)", ["short", "medium", "long"], True),
    ("decode_tps", "Decode tok/s (higher is better)", ["decode-short", "decode-long"], False),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, (metric, title, prompts, log) in zip(axes, PANELS):
    w = 0.8 / len(series)
    for i, (_, label, colour, hatch, s) in enumerate(series):
        xs, med, lo, hi = [], [], [], []
        for j, p in enumerate(prompts):
            m = s["prompts"].get(p, {}).get(metric)
            if not m:
                continue
            xs.append(j + (i - (len(series) - 1) / 2) * w)
            med.append(m["median"]); lo.append(m["median"] - m["min"]); hi.append(m["max"] - m["median"])
        ax.bar(xs, med, w * 0.92, yerr=[lo, hi], capsize=3, color=colour, hatch=hatch,
               edgecolor="white", label=label)
        for x, v in zip(xs, med):
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)
    labels = []
    for p in prompts:
        pt = series[0][4]["prompts"][p]["prompt_tokens"]
        gt = series[0][4]["prompts"][p]["generation_tokens"]["median"]
        labels.append(f"{p}\n{pt} prompt tok\n~{gt:.0f} gen tok")
    ax.set_xticks(range(len(prompts)), labels, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Nemotron 3.5 Bolt E4B, M4 16 GB, greedy, thinking off · median of 5, whiskers = min-max (warm-up discarded)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "p2_visual.png", dpi=130)
print("wrote p2_visual.png")
