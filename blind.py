#!/usr/bin/env python3
"""Build the blind quality set across every model in the roster.

run_p3.py records raw outputs per model; this assembles the one artifact a human
can score honestly: for each of the 8 quality prompts, every model's answer as
Model A/B/C/D in a random order that is fixed by a seed, plus a key that is not
supposed to be opened until scoring is done.

  .venv-mlxlm/bin/python blind.py --round r1
  .venv-mlxlm/bin/python blind.py --round r1 --all-checkpoints   # plain + OptiQ per model

Default is one row per model (its first checkpoint, the plain 4-bit), because
scoring 8 prompts x 8 checkpoints by hand is where blind evaluation dies.

Writes results/blind-<round>/{prompts.json,key.json,scores-template.json}.
"""

import argparse
import json
import random
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = "ABCDEFGH"

# Pre-refactor result files keyed by arm number, for the Nemotron baseline.
LEGACY_ARM = {"arm1": "mlx-4bit", "arm3": "optiq-4bit"}


def load_runs():
    """Every per-checkpoint P3 result file on disk -> [{label, model_id, checkpoint, runs}]."""
    found = []
    for path in sorted(RESULTS.glob("*/p3/*.json")):
        if path.name in ("accuracy_summary.json", "blind_key.json"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "quality_runs" not in d:
            continue

        if path.name.startswith("arm") and "checkpoint" not in d:
            arm = path.name.split("_")[0]
            ckpt = LEGACY_ARM.get(arm)
            if not ckpt:
                continue
            model_id = path.parent.parent.name
            found.append({
                "label": f"{model_id}/{ckpt}",
                "model_id": model_id,
                "checkpoint": ckpt,
                "legacy": True,
                "runs": d["quality_runs"],
            })
        else:
            model_id = d.get("model_id") or path.parent.parent.name
            ckpt = d.get("checkpoint") or path.stem.removesuffix("-results")
            found.append({
                "label": f"{model_id}/{ckpt}",
                "model_id": model_id,
                "checkpoint": ckpt,
                "legacy": False,
                "runs": d["quality_runs"],
            })
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="r1", help="label for this blind round, e.g. r1")
    ap.add_argument("--models", nargs="+", help="restrict to these model ids")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="include every checkpoint, not just one per model")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    runs = load_runs()
    if args.models:
        runs = [r for r in runs if r["model_id"] in args.models]

    def one_per_model(rows):
        keep, seen = [], set()
        order = {m: i for i, m in enumerate(registry.ORDER)}
        for r in sorted(rows, key=lambda r: (order.get(r["model_id"], 99), r["checkpoint"])):
            if r["model_id"] in seen:
                continue
            seen.add(r["model_id"])
            keep.append(r)
        return keep

    if not args.all_checkpoints:
        runs = one_per_model(runs)
        if len(runs) < 2 and runs:
            # One model with several checkpoints is still a blind comparison.
            print(f"only {len(runs)} model on disk — using every checkpoint of it instead")
            runs = [r for r in load_runs() if not args.models or r["model_id"] in args.models]

    if len(runs) < 2:
        raise SystemExit(f"need at least 2 result sets to blind, found {len(runs)}: "
                         + ", ".join(r["label"] for r in runs))

    by_id = {}
    for r in runs:
        for q in r["runs"]:
            by_id.setdefault(q["id"], {})[r["label"]] = q

    complete = [qid for qid, entries in by_id.items() if len(entries) == len(runs)]
    missing = sorted(set(by_id) - set(complete))
    if missing:
        print(f"warning: {len(missing)} prompt(s) missing a model, excluded: {missing}")
    if not complete:
        raise SystemExit("no quality prompt was answered by every selected row")

    rng = random.Random(args.seed)

    # Slot assignment is balanced, not just shuffled: a rater who anchors on
    # "Model A" would otherwise pay a model that happens to land in slot A five
    # times out of eight. A rotation over a random base order gives each row each
    # slot within one pass through the prompt list.
    base_order = list(runs)
    rng.shuffle(base_order)
    slot_use = {r["label"]: [0] * len(base_order) for r in runs}

    prompts, key, template = [], {}, []
    for i, qid in enumerate(sorted(complete)):
        entries = by_id[qid]
        k = i % len(base_order)
        ordered = base_order[k:] + base_order[:k]
        labels = [r["label"] for r in ordered]
        row = {"id": qid, "task": entries[labels[0]]["task"],
               "prompt": entries[labels[0]]["prompt"], "rubric": entries[labels[0]]["rubric"]}
        key[qid] = {}
        score_row = {"id": qid, "task": row["task"]}
        for slot, label in enumerate(labels):
            slot_name = f"Model {LABELS[slot]}"
            row[slot_name] = entries[label]["response"]
            key[qid][slot_name] = label
            slot_use[label][slot] += 1
            score_row[slot_name] = {
                "accuracy": None, "coherence": None, "instruction_fit": None,
                "total": None, "notes": "",
            }
        prompts.append(row)
        template.append(score_row)

    out_dir = Path(args.out) / f"blind-{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    (out_dir / "key.json").write_text(json.dumps(key, indent=2))
    (out_dir / "scores-template.json").write_text(json.dumps({
        "rubric_categories": ["accuracy", "coherence", "instruction_fit"],
        "scale": "1-5 per category (total 15 per prompt)",
        "round": args.round,
        "rows": [r["label"] for r in runs],
        "slot_balance": slot_use,
        "scores": template,
    }, indent=2))

    print(f"{len(prompts)} prompts x {len(runs)} rows -> {out_dir}/prompts.json")
    for r in runs:
        print(f"  row present: {r['label']}")
    print(f"key:      {out_dir}/key.json          (do not read until scored)")
    print(f"template: {out_dir}/scores-template.json -> copy to scores.json and fill in")
    for label, counts in slot_use.items():
        spread = max(counts) - min(counts)
        print(f"  slots {label:32s} {counts}" + ("  UNBALANCED" if spread > 1 else ""))


if __name__ == "__main__":
    main()
