#!/usr/bin/env python3
"""Validate and summarize a blind quality round.

Turns hand-filled scores into the table that goes in the report, and refuses to
do it if the scoring is incomplete — a mean over 6 of 8 prompts is not a result.

  .venv-mlxlm/bin/python score_quality.py --round r1            # check + summarize
  .venv-mlxlm/bin/python score_quality.py --round r1 --unblind  # show which row won

The scores file is results/blind-<round>/scores.json, filled from
scores-template.json: 1-5 on accuracy / coherence / instruction_fit per model
per prompt. Totals are computed here, never typed. Completeness is judged against
results/blind-<round>/key.json (expected prompt IDs, rows, seed); one rater is always
reported as exploratory.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATS = ("accuracy", "coherence", "instruction_fit")
LABELS = "ABCDEFGH"


def validate(data, k):
    """Problems with a filled scores file against the sealed key; [] means complete."""
    problems = []
    rows, expected, key = k["rows"], k["expected_prompt_ids"], k["key"]
    ids = [r.get("id") for r in data.get("scores", [])]
    for qid in sorted({i for i in ids if ids.count(i) > 1}, key=str):
        problems.append(f"{qid}: duplicate prompt id")
    for qid in sorted(set(expected) - set(ids)):
        problems.append(f"{qid}: prompt not scored (incomplete round)")
    for qid in sorted(set(ids) - set(expected), key=str):
        problems.append(f"{qid}: not a prompt of this round")
    for row in data.get("scores", []):
        qid = row.get("id")
        if qid not in expected:
            continue
        for slot in (f"Model {LABELS[i]}" for i in range(len(rows))):
            cell = row.get(slot)
            if not cell:
                problems.append(f"{qid}: {slot} missing")
                continue
            vals = []
            for cat in CATS:
                v = cell.get(cat)
                if v is None:
                    problems.append(f"{qid}: {slot}.{cat} unscored")
                elif isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
                    problems.append(f"{qid}: {slot}.{cat}={v!r} outside 1-5")
                else:
                    vals.append(float(v))
            if len(vals) == len(CATS) and cell.get("total") not in (None, sum(vals)):
                problems.append(f"{qid}: {slot}.total={cell['total']} but categories sum to {sum(vals)}")
    return problems


def summarize(data, k, round_name):
    rows, key = k["rows"], k["key"]
    scored = {r["id"]: r for r in data["scores"]}
    totals = lambda r, slot: sum(float(r[slot][c]) for c in CATS)
    per_row = {r: [] for r in rows}
    for qid in k["expected_prompt_ids"]:
        for i in range(len(rows)):
            slot = f"Model {LABELS[i]}"
            per_row[key[qid][slot]].append(totals(scored[qid], slot))
    n_prompts = len(k["expected_prompt_ids"])
    summary = {label: {
        "prompts": len(t), "total": sum(t),
        "mean_of_15": round(statistics.mean(t), 2),
        "mean_of_5": round(statistics.mean(t) / 3, 2),
        "min": min(t), "max": max(t),
    } for label, t in per_row.items()}

    wins, ties = {label: 0 for label in rows}, 0
    by_slot = {}
    for qid in k["expected_prompt_ids"]:
        t = {key[qid][f"Model {LABELS[i]}"]: totals(scored[qid], f"Model {LABELS[i]}")
             for i in range(len(rows))}
        for i in range(len(rows)):
            by_slot.setdefault(f"Model {LABELS[i]}", []).append(totals(scored[qid], f"Model {LABELS[i]}"))
        leaders = [l for l, v in t.items() if v == max(t.values())]
        ties += len(leaders) > 1
        for l in leaders:
            wins[l] += 1 / len(leaders)
    # Position check: if slot A scores systematically above slot D, the blind is
    # telling you about the rater, not the models.
    slot_means = {s: round(statistics.mean(v), 2) for s, v in by_slot.items()}
    return {"round": round_name, "status": "exploratory", "raters": 1,
            "note": "single rater: a preference signal, not a measurement with an error bar",
            "prompts": n_prompts, "prompt_ids": k["expected_prompt_ids"], "rows": rows,
            "seed": k["seed"], "prompts_sha256": k["prompts_sha256"],
            "summary": summary, "wins": wins, "ties": ties,
            "position_means": slot_means, "slot_balance": k["slot_balance"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="r1")
    ap.add_argument("--unblind", action="store_true", help="map slots back to model/checkpoint labels")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    base = Path(args.out) / f"blind-{args.round}"
    scores_path = base / "scores.json"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} does not exist — copy scores-template.json and fill it in")

    data = json.loads(scores_path.read_text())
    k = json.loads((base / "key.json").read_text())
    problems = validate(data, k)
    prompts_path = base / "prompts.json"
    if prompts_path.exists() and hashlib.sha256(prompts_path.read_bytes()).hexdigest() != k["prompts_sha256"]:
        problems.append("prompts.json does not match the sealed key (changed after blinding)")
    if problems:
        print(f"{len(problems)} problem(s) — not summarizing:")
        for p in problems:
            print("  " + p)
        return 1

    out = summarize(data, k, args.round)
    n_prompts, rows = out["prompts"], out["rows"]
    sm = out["position_means"]
    if len(sm) > 1:
        spread = max(sm.values()) - min(sm.values())
        print("position means: " + ", ".join(f"{s}={m}" for s, m in sorted(sm.items())))
        if spread >= 1.0:
            print(f"  warning: {spread:.2f} points apart across slots — check for position bias before trusting the ranking")

    ordered = sorted(out["summary"].items(), key=lambda kv: (-kv[1]["mean_of_15"], kv[0]))
    print(f"\nBlind quality, round {args.round} (EXPLORATORY, single rater): {n_prompts} prompts x {len(rows)} rows\n")
    print(f"| row | mean /5 | total /{15 * n_prompts} | wins (of {n_prompts}) | range |")
    print("|---|---|---|---|---|")
    for label, s in ordered:
        print(f"| {label} | {s['mean_of_5']} | {s['total']} | {out['wins'][label]:g} | {s['min']}-{s['max']} |")
    if out["ties"]:
        print(f"\n{out['ties']} prompt(s) tied at the top; ties shared half a win each.")
    print(f"\nSingle rater, {n_prompts} prompts: {out['note']}. Say so next to it in the report.")

    if args.unblind:
        out["key"] = k["key"]
    (base / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {base / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
