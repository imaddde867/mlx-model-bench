#!/usr/bin/env python3
"""Reproducible, offline decision analysis of committed MLX benchmark evidence.

Unit of analysis: a model/checkpoint configuration. Performance medians describe
one benchmark session; repeated requests in that session are NOT independent
replicate sessions. The accuracy prompts are a fixed, non-random convenience set.
No deployment recommendation or inferential CI is computed from these records.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("arithmetic_ml", "function_call", "instruction_following")
TAGS = ("mlx-4bit", "optiq-4bit", "mlx-8bit")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary, prompt, metric):
    block = summary.get("prompts", {}).get(prompt, {}).get(metric)
    if not isinstance(block, dict) or not isinstance(block.get("median"), (int, float)):
        raise ValueError(f"missing {prompt}/{metric} median")
    return block


def _check_trials(run_dir, prompt, metric, declared):
    """Independently check a recorded median against the existing kept trial files.

    This verifies aggregation, NOT session-to-session reproducibility.
    """
    paths = sorted(run_dir.glob(f"{prompt}-trial*.json"))
    if not paths:
        raise ValueError(f"missing kept trial files: {run_dir}/{prompt}")
    trials = [_read_json(path) for path in paths]
    kept = [r[metric] for r in trials if r.get("ok") is True
            and not r.get("discarded", False) and isinstance(r.get(metric), (int, float))]
    if not kept or len(kept) != len(paths):
        raise ValueError(f"incomplete valid kept {prompt}/{metric} trials: {run_dir}")
    if len(kept) != declared.get("n"):
        raise ValueError(f"{run_dir}: {prompt}/{metric} count mismatch")
    if not math.isclose(statistics.median(kept), declared["median"], rel_tol=0, abs_tol=0.00011):
        raise ValueError(f"{run_dir}: {prompt}/{metric} median disagrees with trials")
    return len(kept)


def _p3_rows(path):
    data = _read_json(path)
    rows = data.get("accuracy_runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"missing P3 accuracy runs in {path}")
    ids = [r.get("id") for r in rows]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError(f"missing or duplicate P3 prompt IDs in {path}")
    if any(r.get("category") not in CATEGORIES or type(r.get("passed")) is not bool for r in rows):
        raise ValueError(f"invalid P3 category or verdict in {path}")
    return {r["id"]: (r["category"], r["passed"]) for r in rows}, data.get("grader_version")


def load_matrix(results, models=None):
    """Require all requested checkpoint arms, no silent dropping of failed rows."""
    models = list(models) if models is not None else registry.list_models()
    matrix = []
    for cfg in models:
        for tag in TAGS:
            ckpt = cfg.checkpoint(tag)
            run_dir = results / cfg.id / f"{tag}-inproc"
            summary_path = run_dir / "summary.json"
            p3_path = results / cfg.id / "p3" / f"{tag}-results.json"
            summary = _read_json(summary_path)
            grades, version = _p3_rows(p3_path)
            if summary.get("model_id", cfg.id) != cfg.id or summary.get("checkpoint", tag) != tag:
                raise ValueError(f"benchmark identity mismatch: {summary_path}")
            if "weights_gb" in summary and not math.isclose(
                    float(summary["weights_gb"]), ckpt.weights_gb, abs_tol=0.015):
                raise ValueError(f"weight metadata mismatch: {summary_path}")
            if "publisher" in summary and summary["publisher"] != ckpt.publisher:
                raise ValueError(f"publisher metadata mismatch: {summary_path}")
            decode = _metric(summary, "decode-short", "decode_tps")
            long_ttft = _metric(summary, "long", "ttft_s")
            _check_trials(run_dir, "decode-short", "decode_tps", decode)
            _check_trials(run_dir, "long", "ttft_s", long_ttft)
            stage = ("historical" if "publishable" not in summary or "outcome" not in summary
                     else "reported complete" if summary["publishable"] is True
                     and summary["outcome"] == "completed" else "nonpublishable")
            by_cat = {cat: (sum(passed for category, passed in grades.values() if category == cat),
                            sum(category == cat for category, _ in grades.values()))
                      for cat in CATEGORIES}
            matrix.append({
                "model_id": cfg.id, "model": cfg.name, "tag": tag,
                "publisher": ckpt.publisher, "bpw": ckpt.bits_per_weight,
                "weights_gb": ckpt.weights_gb, "decode_short": decode["median"],
                "decode_min": decode["min"], "decode_max": decode["max"],
                "ttft_long": long_ttft["median"],
                "prompt_tokens_long": summary["prompts"]["long"].get("prompt_tokens"),
                "grade_version": version or "historical", "grades": grades,
                "categories": by_cat, "condition": summary.get("condition", "not recorded"),
                "stage": stage, "summary_path": summary_path,
                "p3_path": p3_path,
            })
    return matrix


def dominates(a, b, objectives):
    """Objectives are (key, +1 maximize/-1 minimize). Exact measured front."""
    atleast = all(sign * a[key] >= sign * b[key] for key, sign in objectives)
    better = any(sign * a[key] > sign * b[key] for key, sign in objectives)
    return atleast and better


def pareto(rows, objectives):
    """Return nondominated rows; missing metrics are errors, not zeros."""
    for row in rows:
        for key, _ in objectives:
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                raise ValueError(f"missing or invalid Pareto metric {key} for {row.get('model_id')}")
    return [r for r in rows if not any(dominates(other, r, objectives)
                                       for other in rows if other is not r)]


def paired_changes(left, right):
    """Compare the SAME item IDs, not two unpaired aggregate proportions."""
    a, b = left["grades"], right["grades"]
    if a.keys() != b.keys() or any(a[k][0] != b[k][0] for k in a):
        raise ValueError("paired P3 comparisons require identical prompt IDs and categories")
    gained = sorted(k for k in a if not a[k][1] and b[k][1])
    lost = sorted(k for k in a if a[k][1] and not b[k][1])
    return {"gained": gained, "lost": lost, "discordant": len(gained) + len(lost)}


def exact_mcnemar(gained, lost):
    """Exact two-sided conditional binomial calculation, exploratory only.

    The fixed, hand-authored prompt set does NOT justify a claim about the
    general population of tasks, regardless of the resulting p-value.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(gained, lost) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pct_change(new, old):
    return 100 * (new / old - 1) if old else float("nan")


def fmt(v, digits=1):
    return f"{v:.{digits}f}" if v is not None else "n/a"


def _status_p4(results):
    """Surface P4 pilots without treating unvalidated runs as reliability."""
    pilots = []
    for path in sorted(results.glob("p4-*/**/p4/*-results.json")):
        doc = _read_json(path)
        pilots.append((doc.get("model_id", "?"), doc.get("checkpoint", "?"),
                       doc.get("passed"), doc.get("total"), doc.get("status", "missing"), path))
    return pilots


def render(matrix, results):
    lines = [
        "# Local MLX model study — mathematical and statistical readout",
        "",
        "Generated by `python3 study.py --out results/STUDY.md` from the committed result JSON.",
        "**Scope:** Apple M4 / 16 GB, pinned historical MLX harness. This is a descriptive",
        "analysis of the available sessions, not a universal ranking or a replicated experiment.",
        "",
        "## What can be concluded today",
        "",
        "- The 12 model/checkpoint configurations have observable speed, storage and P3 tradeoffs.",
        "- The current P2 matrix has **one session per configuration** and its condition was not recorded.",
        "  Five sequential trials are *within-session repetitions*, not five independent experiments.",
        "- P3 measures a fixed 30-item convenience suite (12 arithmetic, 10 JSON-format tool",
        "  requests, 8 instruction constraints); these are separate outcomes, not one intelligence score.",
        "- Neither P3 JSON correctness nor a first-turn response establishes multi-step agent success.",
        "",
        "## All model/quantization configurations",
        "",
        "| Model | Tier | Publisher | bpw | Weights GB | Decode-short tok/s [min–max] | Long TTFT s | Long prompt tokens | Arithmetic strict | JSON tool | Format | Session condition | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in matrix:
        ar, tool, inst = (f"{r['categories'][c][0]}/{r['categories'][c][1]}" for c in CATEGORIES)
        evidence = "legacy" if r["stage"] == "historical" else r["stage"]
        lines.append(
            f"| {r['model']} | {r['tag']} | {r['publisher']} | {r['bpw']:.2f} | "
            f"{r['weights_gb']:.2f} | {r['decode_short']:.1f} "
            f"[{r['decode_min']:.1f}–{r['decode_max']:.1f}] | "
            f"{r['ttft_long']:.3f} | {r['prompt_tokens_long']} | "
            f"{ar} | {tool} | {inst} | {r['condition']} | {evidence} |"
        )
    lines.extend([
        "",
        "Decode-short is the original session's median throughput; LFM's generated tokens include",
        "reasoning, and Nemotron's short-decode generation lengths differ. Long TTFT is a user-facing",
        "same-text prompt comparison but each tokenizer produces different token counts. Checkpoint",
        "weights are file bytes, **not** peak process/system memory. The five-trial ranges are",
        "observed ranges, **not confidence intervals**. `legacy` records predate current gates.",
        "",
        "## Pareto analysis (two measured dimensions only)",
        "",
        "Objectives: minimize checkpoint weights (GB) and maximize short-decode throughput",
        "(tok/s). A configuration is dominated when another is no worse on both dimensions",
        "and strictly better on at least one. This ignores task quality, runtime memory and",
        "response completion; therefore it is **not an agent suitability frontier**.",
        "",
        "| Nondominated configuration | Weights GB | Decode-short tok/s | Throughput per stored GB |",
        "|---|---:|---:|---:|",
    ])
    valid = [r for r in matrix if r["stage"] != "nonpublishable"]
    for r in pareto(valid, (("weights_gb", -1), ("decode_short", 1))):
        lines.append(f"| {r['model']} / {r['tag']} | {r['weights_gb']:.2f} | "
                     f"{r['decode_short']:.1f} | {r['decode_short']/r['weights_gb']:.1f} |")
    lines.extend([
        "",
        "Throughput per stored GB is an accounting ratio, not measured memory bandwidth or",
        "a universal efficiency rating. The frontier is provisional: timing environments were",
        "not recorded and the smaller per-model prompt-token differences remain relevant.",
        "",
        "## Quantization tradeoffs, paired at the P3 prompt-ID level",
        "",
        "Every row below compares a wider/mixed quantization to that **same model's 4-bit**",
        "checkpoint. `+/-` are the number of P3 tasks that changed from fail to pass /",
        "pass to fail. This is more informative than subtracting 30-item totals alone.",
        "",
        "| Model | Compared with 4-bit | Weight delta | Decode-short delta | P3 pass (4-bit→new) | Gained / lost items | Exact paired p (exploratory) |",
        "|---|---|---:|---:|---|---|---:|",
    ])
    for model in registry.ORDER:
        rows = {r["tag"]: r for r in matrix if r["model_id"] == model}
        if not rows:
            continue
        base = rows["mlx-4bit"]
        for tag in TAGS[1:]:
            other = rows[tag]
            d = paired_changes(base, other)
            old_score = sum(x[0] for x in base["categories"].values())
            new_score = sum(x[0] for x in other["categories"].values())
            # Never claim a common grader if baseline records predate grader-version stamps.
            compatible = base["grade_version"] == other["grade_version"] and base["grade_version"] != "historical"
            p = f"{exact_mcnemar(len(d['gained']), len(d['lost'])):.3f}" if compatible else "n/a (legacy)"
            lines.append(f"| {base['model']} | {tag} | {pct_change(other['weights_gb'], base['weights_gb']):+.1f}% | "
                         f"{pct_change(other['decode_short'],base['decode_short']):+.1f}% | "
                         f"{old_score}/30 → {new_score}/30 | {len(d['gained'])} / {len(d['lost'])} | {p} |")
    lines.extend([
        "",
        "Exact paired p is the two-sided conditional McNemar/binomial test over discordant",
        "items (assuming exchangeable task samples). These hand-authored prompts are **not a random",
        "sample** from everyday work, so p-values are shown for inspection only, without",
        "population-level significance or multiple-testing claims. Historical Nemotron P3",
        "rows lack a stamped grader version; no p-value is reported across that boundary.",
        "Different publishers and bits-per-weight are confounders when comparing quantizers.",
        "",
        "## Agent evidence (P4)",
        "",
    ])
    pilots = _status_p4(results)
    if pilots:
        for model, tag, passed, total, status, path in pilots:
            lines.append(f"- `{model}/{tag}`: {passed}/{total} tasks, status `{status}` "
                         f"(`{path.relative_to(results)}`).")
    else:
        lines.append("- No P4 result files found.")
    lines.extend([
        "",
        "**No P4 reliability value is eligible for optimization yet.** The current pilot",
        "uses a custom JSON-text tool protocol and needs validated native-tool/control",
        "comparisons, held-out tasks and actual tool-to-completion wall time.",
        "",
        "## What to measure next (to support real decisions)",
        "",
        "1. **Session-level replication:** for a shortlist of 4-bit configurations, collect",
        "   at least 5 separately initialized sessions per configuration, counterbalance",
        "   model order, run matched prompt sets, record background load and temperature,",
        "   and retain session IDs. Summarize within session; estimate uncertainty across",
        "   independent sessions (cluster bootstrap or paired session deltas), not by",
        "   resampling five sequential trials as if they were independent machines.",
        "2. **Task distribution and quality:** build separate held-out suites for conversation,",
        "   small coding/file tools and hardware control. Record completion, exact success,",
        "   refusal/hallucination and failures. Maintain task-specific weights, not a single",
        "   aggregate 'intelligence' metric. Gather independent human ratings when needed.",
        "3. **Actual user latency:** report time-to-first-content, time-to-useful-result,",
        "   completed-task p50/p95, tool overhead, tokens spent thinking, and total runtime",
        "   peak/system memory. Label cache regime, prompt lengths and context horizon.",
        "4. **Decision models:** after task-specific success is validated, apply minimum",
        "   reliability/memory/latency constraints, compute Pareto fronts, then evaluate",
        "   explicitly chosen workload weights with sensitivity analysis. Missing quality",
        "   is unknown, never zero or a fabricated score.",
        "",
        "## Reproduce and verify",
        "",
        "```bash",
        "python3 study.py --out results/STUDY.md",
        "python3 study.py --out results/STUDY.md --check",
        "python3 -m unittest discover -s tests -v",
        "```",
        "",
        "The first command recomputes the document from source JSON. The second checks that",
        "the committed document has not become stale. This analysis loads no MLX model and",
        "uses only the Python standard library. No historical result file is changed.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=None, help="write output Markdown, otherwise print")
    ap.add_argument("--check", action="store_true", help="compare output with existing --out file")
    args = ap.parse_args(argv)
    if args.check and args.out is None:
        ap.error("--check requires --out")
    try:
        output = render(load_matrix(args.results), args.results)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"study: INVALID EVIDENCE: {exc}", file=sys.stderr)
        return 2
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != output:
            print(f"study: generated analysis differs from {args.out}", file=sys.stderr)
            return 1
        print(f"study: checked {args.out}")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"study: wrote {args.out}")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
