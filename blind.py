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

Writes results/blind-<round>/{prompts.json,key.json,scores-template.json}. prompts.json and the
template carry no checkpoint IDs; key.json holds the mapping, seed, checkpoint set and
expected prompt IDs. Missing/duplicate/unexpected responses abort the round.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = "ABCDEFGH"
CATS = ("accuracy", "coherence", "instruction_fit")
QUALITY_DIR = ROOT / "prompts" / "quality"
N_PROMPTS = 8

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


def expected_ids():
    ids = sorted(p.stem for p in QUALITY_DIR.glob("*.txt"))
    if len(ids) != N_PROMPTS:
        raise SystemExit(f"expected {N_PROMPTS} quality prompts in {QUALITY_DIR}, found {len(ids)}")
    return ids


def check_runs(runs, expected):
    """Every selected row must answer each expected prompt exactly once."""
    problems, seen = [], set()
    if len(runs) > len(LABELS):
        problems.append(f"{len(runs)} rows but only {len(LABELS)} slot labels")
    for r in runs:
        if r["label"] in seen:
            problems.append(f"{r['label']}: selected twice (duplicate result files)")
        seen.add(r["label"])
        ids = [q["id"] for q in r["runs"]]
        for qid in sorted({i for i in ids if ids.count(i) > 1}):
            problems.append(f"{r['label']}: duplicate response for {qid}")
        for qid in sorted(set(expected) - set(ids)):
            problems.append(f"{r['label']}: missing response for {qid}")
        for qid in sorted(set(ids) - set(expected)):
            problems.append(f"{r['label']}: unexpected prompt {qid}")
    if problems:
        raise SystemExit("incomplete round, not blinding:\n  " + "\n  ".join(problems))


def check_balance(slot_use, n_prompts):
    """Each row must hold each slot floor or ceil of n_prompts/n_rows times."""
    n = len(slot_use)
    lo, hi = n_prompts // n, -(-n_prompts // n)
    for label, counts in slot_use.items():
        if sum(counts) != n_prompts or not all(lo <= c <= hi for c in counts):
            raise SystemExit(f"unbalanced slots for {label}: {counts}, need each in {lo}..{hi}")


def assign(runs, expected, seed):
    """Balanced slot assignment -> (prompts, key, score template, slot_use). Deterministic in seed."""
    by_id = {r["label"]: {q["id"]: q for q in r["runs"]} for r in runs}
    # Slot assignment is balanced, not just shuffled: a rater who anchors on
    # "Model A" would otherwise pay a model that happens to land in slot A five
    # times out of eight. A rotation over a random base order gives each row each
    # slot within one pass through the prompt list.
    base_order = [r["label"] for r in runs]
    random.Random(seed).shuffle(base_order)
    slot_use = {label: [0] * len(base_order) for label in base_order}

    prompts, key, template = [], {}, []
    for i, qid in enumerate(expected):
        k = i % len(base_order)
        labels = base_order[k:] + base_order[:k]
        first = by_id[labels[0]][qid]
        row = {"id": qid, "task": first["task"], "prompt": first["prompt"], "rubric": first["rubric"]}
        key[qid] = {}
        score_row = {"id": qid, "task": first["task"]}
        for slot, label in enumerate(labels):
            slot_name = f"Model {LABELS[slot]}"
            row[slot_name] = by_id[label][qid]["response"]
            key[qid][slot_name] = label
            slot_use[label][slot] += 1
            score_row[slot_name] = {c: None for c in CATS} | {"total": None, "notes": ""}
        prompts.append(row)
        template.append(score_row)
    check_balance(slot_use, len(expected))
    return prompts, key, template, slot_use


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

    runs = sorted(runs, key=lambda r: r["label"])  # seed -> order must not depend on glob order
    expected = expected_ids()
    check_runs(runs, expected)
    prompts, key, template, slot_use = assign(runs, expected, args.seed)

    out_dir = Path(args.out) / f"blind-{args.round}"
    if (out_dir / "scores.json").exists():
        raise SystemExit(f"{out_dir}/scores.json exists — rebuilding would orphan it; pick a new --round")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_text = json.dumps(prompts, indent=2)
    (out_dir / "prompts.json").write_text(prompts_text)
    # Sealed: the only file that maps slots to checkpoints. It also pins the seed,
    # the exact checkpoint set and expected prompt IDs for later audit.
    (out_dir / "key.json").write_text(json.dumps({
        "round": args.round,
        "seed": args.seed,
        "rows": [r["label"] for r in runs],
        "expected_prompt_ids": expected,
        "prompts_sha256": hashlib.sha256(prompts_text.encode()).hexdigest(),
        "slot_balance": slot_use,
        "key": key,
    }, indent=2))
    (out_dir / "scores-template.json").write_text(json.dumps({
        "rubric_categories": list(CATS),
        "scale": "1-5 per category (total 15 per prompt)",
        "round": args.round,
        "scores": template,
    }, indent=2))

    print(f"{len(prompts)} prompts x {len(runs)} rows -> {out_dir}/prompts.json")
    for r in runs:
        print(f"  row present: {r['label']}")
    print(f"key:      {out_dir}/key.json          (do not read until scored; holds seed {args.seed})")
    print(f"template: {out_dir}/scores-template.json -> copy to scores.json and fill in")
    for label, counts in slot_use.items():
        print(f"  slots {label:32s} {counts}")


if __name__ == "__main__":
    main()
