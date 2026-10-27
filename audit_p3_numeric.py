#!/usr/bin/env python3
"""Recompute the REPORT.md numerical audit from the committed P3 rows.

The strict grader requires the whole response to be one number. This audit
instead reads the last number in the extracted `response` (the final numeric
claim), so a correct value inside `\\boxed{...}` or prose counts. It never
changes the committed `passed` verdicts.

  python audit_p3_numeric.py            # per-prompt values and verdicts
  python audit_p3_numeric.py --check    # assert the totals printed in REPORT.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

import registry

ROOT = Path(__file__).resolve().parent

# Same numeric literal shape as p3_suite, plus exponent notation.
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

EXPECTED = {
    ("nemotron-3.5-bolt", "mlx-4bit"): 6,
    ("nemotron-3.5-bolt", "optiq-4bit"): 7,
    ("nemotron-3.5-bolt", "mlx-8bit"): 6,
    ("minicpm5-2b", "mlx-4bit"): 5,
    ("minicpm5-2b", "optiq-4bit"): 3,
    ("minicpm5-2b", "mlx-8bit"): 5,
    ("spark-x2.5-4b", "mlx-4bit"): 3,
    ("spark-x2.5-4b", "optiq-4bit"): 3,
    ("spark-x2.5-4b", "mlx-8bit"): 4,
    ("lfm2.5-8b-a1b", "mlx-4bit"): 12,
    ("lfm2.5-8b-a1b", "optiq-4bit"): 11,
    ("lfm2.5-8b-a1b", "mlx-8bit"): 11,
}


def last_number(text):
    found = NUMBER.findall(text or "")
    return float(found[-1].replace(",", "")) if found else None


def decisions(path):
    data = json.loads(path.read_text())
    out = []
    for run in data.get("accuracy_runs", []):
        if run.get("category") != "arithmetic_ml":
            continue
        value = last_number(run.get("response"))
        target = run["target"]
        tolerance = 1e-4 if isinstance(target, float) else 0
        out.append((run["id"], value, value is not None and abs(value - target) <= tolerance))
    return out


def fmt(value):
    if value is None:
        return "none"
    return str(int(value)) if value.is_integer() else f"{value:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert the totals printed in REPORT.md")
    args = ap.parse_args()

    order = {model.id: i for i, model in enumerate(registry.list_models())}
    paths = sorted(
        (ROOT / "results").glob("*/p3/*-results.json"),
        key=lambda p: (order.get(p.parent.parent.name, 99), p.name),
    )
    totals = {}
    for path in paths:
        model = path.parent.parent.name
        checkpoint = path.name.removesuffix("-results.json")
        rows = decisions(path)
        totals[(model, checkpoint)] = sum(ok for _, _, ok in rows)
        print(f"{model} / {checkpoint}: {totals[(model, checkpoint)]}/{len(rows)}")
        print("  " + " ".join(f"{pid}={fmt(value)}{'✓' if ok else '✗'}" for pid, value, ok in rows))

    if not args.check:
        return 0
    bad = {key: (totals.get(key), want) for key, want in EXPECTED.items() if totals.get(key) != want}
    if bad:
        print(f"audit totals do not match REPORT.md: {bad}", file=sys.stderr)
        return 1
    print("all audit totals match REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
