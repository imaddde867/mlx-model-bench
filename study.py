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
# Locked identity/category manifest for the original fixed, 30-item P3 suite.
# Do not infer the denominator from an incomplete results file.
P3_EXPECTED = {
    **{f"math-{n:02d}": "arithmetic_ml" for n in range(1, 13)},
    **{f"tool-{n:02d}": "function_call" for n in range(1, 11)},
    **{f"instr-{n:02d}": "instruction_following" for n in range(1, 9)},
}
# The two legacy Nemotron P3 records predate checkpoint metadata. Validate
# their recorded arm and model path rather than trusting a renamed filename.
LEGACY_P3 = {
    ("nemotron-3.5-bolt", "mlx-4bit"): "1",
    ("nemotron-3.5-bolt", "optiq-4bit"): "3",
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary, prompt, metric):
    block = summary.get("prompts", {}).get(prompt, {}).get(metric)
    if not isinstance(block, dict):
        raise ValueError(f"missing {prompt}/{metric} summary")
    for label in ("median", "min", "max"):
        value = block.get(label)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"missing or invalid {prompt}/{metric} {label}")
    if not block["min"] <= block["median"] <= block["max"]:
        raise ValueError(f"inconsistent {prompt}/{metric} range")
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
            and not r.get("discarded", False) and type(r.get(metric)) in (int, float)
            and math.isfinite(r[metric])]
    if not kept or len(kept) != len(paths):
        raise ValueError(f"incomplete valid kept {prompt}/{metric} trials: {run_dir}")
    if len(kept) != declared.get("n"):
        raise ValueError(f"{run_dir}: {prompt}/{metric} count mismatch")
    actual = {"median": statistics.median(kept), "min": min(kept), "max": max(kept)}
    for label, value in actual.items():
        if not math.isclose(value, declared[label], rel_tol=0, abs_tol=0.00011):
            raise ValueError(f"{run_dir}: {prompt}/{metric} {label} disagrees with trials")
    return len(kept)


def _p3_rows(path, *, cfg=None, ckpt=None):
    data = _read_json(path)
    rows = data.get("accuracy_runs")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"missing or invalid P3 accuracy runs in {path}")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"missing or duplicate P3 prompt IDs in {path}")
    if any(r.get("category") not in CATEGORIES or type(r.get("passed")) is not bool for r in rows):
        raise ValueError(f"invalid P3 category or verdict in {path}")
    grades = {r["id"]: (r["category"], r["passed"]) for r in rows}
    if (grades.keys() != P3_EXPECTED.keys()
            or any(grades[pid][0] != category for pid, category in P3_EXPECTED.items())):
        missing = sorted(P3_EXPECTED.keys() - grades.keys())
        unexpected = sorted(grades.keys() - P3_EXPECTED.keys())
        wrong = sorted(pid for pid in P3_EXPECTED.keys() & grades.keys()
                       if grades[pid][0] != P3_EXPECTED[pid])
        raise ValueError(f"incomplete or mismatched P3 task manifest {path}: "
                         f"missing={missing}, unexpected={unexpected}, wrong_categories={wrong}")

    if (cfg is None) != (ckpt is None):
        raise ValueError("both cfg and ckpt required to validate P3 identity")
    if cfg is not None:
        key = (cfg.id, ckpt.tag)
        has_id = "model_id" in data
        has_ckpt = "checkpoint" in data
        if not has_id and not has_ckpt:
            # This exception is limited to the two known, explicitly identified
            # baseline files, and does not silently permit incomplete new rows.
            if key not in LEGACY_P3:
                raise ValueError(f"missing modern P3 identity in {path}")
            if str(data.get("arm")) != LEGACY_P3[key]:
                raise ValueError(f"legacy P3 arm identity mismatch in {path}")
        else:
            if data.get("model_id") != cfg.id or data.get("checkpoint") != ckpt.tag:
                raise ValueError(f"P3 checkpoint identity mismatch in {path}")
            expected = {"hf_repo": ckpt.hf_repo, "publisher": ckpt.publisher,
                        "model_name": cfg.name}
            for label, value in expected.items():
                if data.get(label) != value:
                    raise ValueError(f"P3 {label} identity mismatch in {path}")
            for label, value in (("weights_gb", ckpt.weights_gb),
                                 ("bits_per_weight", ckpt.bits_per_weight),
                                 ("params_total", cfg.params_total),
                                 ("params_active", cfg.params_active)):
                actual = data.get(label)
                if value is None:
                    if actual is not None:
                        raise ValueError(f"P3 {label} identity mismatch in {path}")
                elif type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(
                        actual, value, rel_tol=0, abs_tol=0.015):
                    raise ValueError(f"P3 {label} identity mismatch in {path}")
            if not data.get("grader_version"):
                raise ValueError(f"missing modern P3 grader version in {path}")
        recorded_model_path = data.get("model_path")
        if not isinstance(recorded_model_path, str) or Path(recorded_model_path).name != Path(ckpt.local_dir).name:
            raise ValueError(f"P3 model path identity mismatch in {path}")
    return grades, data.get("grader_version")


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
            grades, version = _p3_rows(p3_path, cfg=cfg, ckpt=ckpt)
            if summary.get("model_id", cfg.id) != cfg.id or summary.get("checkpoint", tag) != tag:
                raise ValueError(f"benchmark identity mismatch: {summary_path}")
            if "weights_gb" in summary and not math.isclose(
                    float(summary["weights_gb"]), ckpt.weights_gb, abs_tol=0.015):
                raise ValueError(f"weight metadata mismatch: {summary_path}")
            if "publisher" in summary and summary["publisher"] != ckpt.publisher:
                raise ValueError(f"publisher metadata mismatch: {summary_path}")
            decode = _metric(summary, "decode-short", "decode_tps")
            long_ttft = _metric(summary, "long", "ttft_s")
            decode_gen = _metric(summary, "decode-short", "generation_tokens")
            long_gen = _metric(summary, "long", "generation_tokens")
            for prompt, metric, declared in (
                ("decode-short", "decode_tps", decode),
                ("decode-short", "generation_tokens", decode_gen),
                ("long", "ttft_s", long_ttft),
                ("long", "generation_tokens", long_gen),
            ):
                _check_trials(run_dir, prompt, metric, declared)
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
                "params_total": cfg.params_total, "params_active": cfg.params_active,
                "decode_min": decode["min"], "decode_max": decode["max"],
                "prompt_tokens_decode": summary["prompts"]["decode-short"].get("prompt_tokens"),
                "decode_gen": decode_gen,
                "ttft_long": long_ttft["median"],
                "ttft_min": long_ttft["min"], "ttft_max": long_ttft["max"],
                "prompt_tokens_long": summary["prompts"]["long"].get("prompt_tokens"),
                "long_gen": long_gen,
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
    """Discover default and archived P4 layouts, without scoring their reliability."""
    paths = set(results.glob("*/p4/*-results.json"))
    paths.update(results.glob("p4-*/**/p4/*-results.json"))
    pilots = []
    for path in sorted(paths):
        doc = _read_json(path)
        model = path.parent.parent.name
        checkpoint = path.name.removesuffix("-results.json")
        if doc.get("model_id") != model or doc.get("checkpoint") != checkpoint:
            raise ValueError(f"P4 pilot identity mismatch in {path}")
        passed, total = doc.get("passed"), doc.get("total")
        if (type(passed) is not int or type(total) is not int
                or total <= 0 or not 0 <= passed <= total
                or not isinstance(doc.get("status"), str)):
            raise ValueError(f"invalid P4 pilot counts/status in {path}")
        pilots.append((model, checkpoint, passed, total, doc["status"], path))
    return pilots


def _token_count(prompt, gen):
    """Per-configuration prompt tokens and generated median [observed min-max]."""
    if type(prompt) is not int or prompt < 0:
        raise ValueError("missing or invalid prompt-token count")
    if not isinstance(gen, dict):
        raise ValueError("missing generation-token summary")
    return (f"{prompt} → {gen['median']:g} "
            f"[{gen['min']:g}–{gen['max']:g}]")


def _parameters(row):
    """Total/active parameters in billions; MoE active count is vendor metadata."""
    total = row.get("params_total")
    active = row.get("params_active")
    if type(total) not in (float, int) or not math.isfinite(total) or total <= 0:
        raise ValueError("missing or invalid parameter count")
    if active is None:
        return f"{total:.2f}/unknown"
    if type(active) not in (float, int) or not math.isfinite(active) or active <= 0:
        raise ValueError("invalid active-parameter count")
    return f"{total:.2f}/{active:.2f}"


def render(matrix, results):
    lines = [
        "# Local MLX model study — mathematical and statistical readout",
        "",
        "Generated by `python3 study.py --out results/STUDY.md` from the committed result JSON.",
        "**Scope:** Apple M4 / 16 GB, pinned historical MLX harness. This is a descriptive",
        "analysis of the available sessions, not a universal ranking or a replicated experiment.",
        "",
        "## Replicated follow-up (2026-10-29)",
        "",
        "[Validated 20-session plain-4-bit report](replicated/m4-4bit-20261029/REPLICATED_STUDY.md):",
        "five fresh sessions per model, three fixed duration-retrieval prompts, common 512-token cap.",
        "All 300 kept responses passed the exact duration check. Historical tables below are unchanged.",
        "Every session carries observed background-CPU, swap and unavailable thermal-speed caveats;",
        "this is exploratory replication under recorded conditions, not a verified interference-free run.",
        "LFM includes internal reasoning; the other models produce much shorter completions.",
        "See the linked report for tokenizer-specific lengths, publisher/bpw, ranges and session intervals.",
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
        "| Model / checkpoint | Publisher | bpw | Params B total/active | Weights GB | Short decode tok/s median [min–max] | Short tokens prompt → generated median [min–max] | Long TTFT s median [min–max] | Long tokens prompt → generated median [min–max] | Arithmetic strict | JSON tool | Format | Session condition | Evidence |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---|---|",
    ]
    for r in matrix:
        ar, tool, inst = (f"{r['categories'][c][0]}/{r['categories'][c][1]}" for c in CATEGORIES)
        evidence = "legacy" if r["stage"] == "historical" else r["stage"]
        lines.append(
            f"| {r['model']} / {r['tag']} | {r['publisher']} | {r['bpw']:.2f} | "
            f"{_parameters(r)} | {r['weights_gb']:.2f} | {r['decode_short']:.1f} "
            f"[{r['decode_min']:.1f}–{r['decode_max']:.1f}] | "
            f"{_token_count(r['prompt_tokens_decode'], r['decode_gen'])} | "
            f"{r['ttft_long']:.3f} [{r['ttft_min']:.3f}–{r['ttft_max']:.3f}] | "
            f"{_token_count(r['prompt_tokens_long'], r['long_gen'])} | "
            f"{ar} | {tool} | {inst} | {r['condition']} | {evidence} |"
        )
    lines.extend([
        "",
        "Params B = total/active; the LFM active-parameter figure is vendor metadata, not",
        "a measured per-token FLOP count. The token columns show tokenizer-specific prompt",
        "length and generated median [observed min–max]. LFM generated tokens include reasoning.",
        "Decode-short is the original session's median throughput. Long TTFT is a user-facing",
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
        "| Nondominated model / checkpoint | Publisher | bpw | Params B total/active | Weights GB | Short decode tok/s median [min–max] | Short tokens prompt → generated median [min–max] | Throughput per stored GB |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ])
    valid = [r for r in matrix if r["stage"] != "nonpublishable"]
    for r in pareto(valid, (("weights_gb", -1), ("decode_short", 1))):
        lines.append(f"| {r['model']} / {r['tag']} | {r['publisher']} | {r['bpw']:.2f} | "
                     f"{_parameters(r)} | {r['weights_gb']:.2f} | "
                     f"{r['decode_short']:.1f} [{r['decode_min']:.1f}–{r['decode_max']:.1f}] | "
                     f"{_token_count(r['prompt_tokens_decode'], r['decode_gen'])} | "
                     f"{r['decode_short']/r['weights_gb']:.1f} |")
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
        "| Model (params B total/active) | 4-bit publisher / bpw | New checkpoint publisher / bpw | Weight delta | Short decode tok/s 4-bit → new (median [min–max]) | Short tokens 4-bit → new (prompt → generated median [min–max]) | Decode-short delta | P3 pass (4-bit → new) | Gained / lost | Exact paired p (exploratory) |",
        "|---|---|---|---:|---|---|---:|---|---|---:|",
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
            lines.append(
                f"| {base['model']} ({_parameters(base)}) | "
                f"mlx-4bit: {base['publisher']} / {base['bpw']:.2f} | "
                f"{tag}: {other['publisher']} / {other['bpw']:.2f} | "
                f"{pct_change(other['weights_gb'], base['weights_gb']):+.1f}% | "
                f"{base['decode_short']:.1f} [{base['decode_min']:.1f}–{base['decode_max']:.1f}] → "
                f"{other['decode_short']:.1f} [{other['decode_min']:.1f}–{other['decode_max']:.1f}] | "
                f"{_token_count(base['prompt_tokens_decode'], base['decode_gen'])} → "
                f"{_token_count(other['prompt_tokens_decode'], other['decode_gen'])} | "
                f"{pct_change(other['decode_short'], base['decode_short']):+.1f}% | "
                f"{old_score}/{len(base['grades'])} → {new_score}/{len(other['grades'])} | "
                f"{len(d['gained'])} / {len(d['lost'])} | {p} |"
            )
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
        "The first command recomputes the document from source JSON. The second checks an",
        "existing generated file for drift (commit it only after review). No MLX model is loaded;",
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
