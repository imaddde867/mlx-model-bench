#!/usr/bin/env python3
"""Validate and summarize a blind quality round.

Turns hand-filled scores into the table that goes in the report, and refuses to
do it if the scoring is incomplete — a mean over 6 of 8 prompts is not a result.

  .venv-mlxlm/bin/python score_quality.py --round r1            # check + summarize
  .venv-mlxlm/bin/python score_quality.py --round r1 --unblind  # show which row won

The scores file is results/blind-<round>/scores.json, filled from
scores-template.json: 1-5 on accuracy / coherence / instruction_fit per model
per prompt. Totals are computed here, never typed.
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATS = ("accuracy", "coherence", "instruction_fit")
LABELS = "ABCDEFGH"


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
    key = json.loads((base / "key.json").read_text())
    rows = data.get("rows") or []

    problems = []
    per_row = {r: [] for r in rows}
    for row in data["scores"]:
        qid = row["id"]
        present = [f"Model {LABELS[i]}" for i in range(len(rows))]
        for slot in present:
            cell = row.get(slot)
            if not cell:
                problems.append(f"{qid}: {slot} missing")
                continue
            vals = []
            for cat in CATS:
                v = cell.get(cat)
                if v is None:
                    problems.append(f"{qid}: {slot}.{cat} unscored")
                    continue
                if not (1 <= float(v) <= 5):
                    problems.append(f"{qid}: {slot}.{cat}={v} outside 1-5")
                vals.append(float(v))
            if len(vals) == len(CATS):
                total = sum(vals)
                if cell.get("total") not in (None, total):
                    problems.append(f"{qid}: {slot}.total={cell['total']} but categories sum to {total}")
                cell["total"] = total
                per_row[key[qid][slot]].append(total)

    if problems:
        print(f"{len(problems)} problem(s) — not summarizing:")
        for p in problems:
            print("  " + p)
        return 1

    n_prompts = len(data["scores"])
    summary = {}
    for label, totals in per_row.items():
        summary[label] = {
            "prompts": len(totals),
            "total": sum(totals),
            "mean_of_15": round(statistics.mean(totals), 2),
            "mean_of_5": round(statistics.mean(totals) / 3, 2),
            "min": min(totals),
            "max": max(totals),
        }

    wins = {label: 0 for label in per_row}
    ties = 0
    for row in data["scores"]:
        totals = {key[row["id"]][f"Model {LABELS[i]}"]: row[f"Model {LABELS[i]}"]["total"]
                  for i in range(len(rows))}
        best = max(totals.values())
        leaders = [l for l, t in totals.items() if t == best]
        if len(leaders) > 1:
            ties += 1
            for l in leaders:
                wins[l] += 1 / len(leaders)
        else:
            wins[leaders[0]] += 1

    # Position check: if slot A scores systematically above slot D, the blind is
    # telling you about the rater, not the models.
    by_slot = {}
    for row in data["scores"]:
        for i in range(len(rows)):
            slot = f"Model {LABELS[i]}"
            if slot in row:
                by_slot.setdefault(slot, []).append(row[slot]["total"])
    slot_means = {s: round(statistics.mean(v), 2) for s, v in by_slot.items() if v}
    if len(slot_means) > 1:
        spread = max(slot_means.values()) - min(slot_means.values())
        print("position means: " + ", ".join(f"{s}={m}" for s, m in sorted(slot_means.items())))
        if spread >= 1.0:
            print(f"  warning: {spread:.2f} points apart across slots — check for position bias before trusting the ranking")

    ordered = sorted(summary.items(), key=lambda kv: -kv[1]["mean_of_15"])
    print(f"\nBlind quality, round {args.round}: {n_prompts} prompts x {len(rows)} rows\n")
    print(f"| row | mean /5 | total /{15 * n_prompts} | wins (of {n_prompts}) | range |")
    print("|---|---|---|---|---|")
    for label, s in ordered:
        print(f"| {label} | {s['mean_of_5']} | {s['total']} | {wins[label]:g} | {s['min']}-{s['max']} |")
    if ties:
        print(f"\n{ties} prompt(s) tied at the top; ties shared half a win each.")
    print(f"\nSingle rater, {n_prompts} prompts: this is a preference signal, not a measurement "
          f"with an error bar. Say so next to it in the report.")

    out = {"round": args.round, "prompts": n_prompts, "rows": rows,
           "summary": summary, "wins": wins, "ties": ties,
           "position_means": slot_means,
           "slot_balance": data.get("slot_balance")}
    if args.unblind:
        out["key"] = key
    (base / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {base / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
