"""Offline tests: blind round completeness, balance, sealed key, and scoring validation."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import blind
import score_quality

IDS = [p.stem for p in sorted((ROOT / "prompts" / "quality").glob("*.txt"))]
CKPTS = ["m1/4bit", "m2/4bit", "m3/4bit"]


def runs_for(labels, ids=IDS):
    return [{"label": l, "model_id": l.split("/")[0], "checkpoint": l.split("/")[1], "legacy": False,
             "runs": [{"id": q, "task": "t", "prompt": f"p-{q}", "rubric": "r",
                       "response": f"resp {q}"} for q in ids]}
            for l in labels]


def build_round(tmp, labels=CKPTS, seed=7, ids=IDS):
    """Run blind.main against synthetic runs; returns the round dir."""
    argv = ["blind.py", "--round", "rt", "--seed", str(seed), "--out", str(tmp), "--all-checkpoints"]
    with patch.object(blind, "load_runs", return_value=runs_for(labels, ids)), \
            patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        blind.main()
    return Path(tmp) / "blind-rt"


def fill(d, value=lambda q, slot: 4):
    data = json.loads((d / "scores-template.json").read_text())
    for r in data["scores"]:
        for slot, cell in r.items():
            if slot.startswith("Model "):
                for c in score_quality.CATS:
                    cell[c] = value(r["id"], slot)
    (d / "scores.json").write_text(json.dumps(data))
    return data


def run_scoring(tmp):
    buf = io.StringIO()
    with patch.object(sys, "argv", ["score_quality.py", "--round", "rt", "--out", str(tmp)]), \
            contextlib.redirect_stdout(buf):
        rc = score_quality.main()
    return rc, buf.getvalue()


class BlindBuild(unittest.TestCase):
    def test_prompts_hide_checkpoints_and_key_is_sealed_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build_round(tmp)
            for f in ("prompts.json", "scores-template.json"):
                text = (d / f).read_text()
                for label in CKPTS:
                    self.assertNotIn(label, text)
                    self.assertNotIn(label.split("/")[0], text)
            key = json.loads((d / "key.json").read_text())
            self.assertEqual(key["seed"], 7)
            self.assertEqual(sorted(key["rows"]), CKPTS)
            self.assertEqual(key["expected_prompt_ids"], IDS)
            self.assertEqual(len(IDS), 8)

    def test_balanced_and_repeat_is_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            da, db = build_round(a), build_round(b)
            for f in ("prompts.json", "key.json", "scores-template.json"):
                self.assertEqual((da / f).read_text(), (db / f).read_text())
            for counts in json.loads((da / "key.json").read_text())["slot_balance"].values():
                self.assertTrue(all(2 <= c <= 3 for c in counts), counts)

    def test_missing_response_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                build_round(tmp, ids=IDS[:-1])
            self.assertIn(f"missing response for {IDS[-1]}", str(cm.exception))
            self.assertFalse((Path(tmp) / "blind-rt").exists())

    def test_duplicate_response_fails(self):
        runs = runs_for(CKPTS)
        runs[1]["runs"].append(copy.deepcopy(runs[1]["runs"][0]))
        with self.assertRaises(SystemExit) as cm:
            blind.check_runs(runs, IDS)
        self.assertIn(f"duplicate response for {IDS[0]}", str(cm.exception))

    def test_duplicate_checkpoint_fails(self):
        with self.assertRaises(SystemExit):
            blind.check_runs(runs_for([CKPTS[0], CKPTS[0]]), IDS)

    def test_imbalance_fails(self):
        with self.assertRaises(SystemExit):
            blind.check_balance({"a": [8, 0], "b": [0, 8]}, 8)
        blind.check_balance({"a": [4, 4], "b": [4, 4]}, 8)

    def test_does_not_orphan_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            fill(build_round(tmp))
            with self.assertRaises(SystemExit):
                build_round(tmp)


class Scoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = build_round(self.tmp.name)

    def assert_rejected(self, edit, text):
        data = fill(self.d)
        edit(data)
        (self.d / "scores.json").write_text(json.dumps(data))
        rc, out = run_scoring(self.tmp.name)
        self.assertEqual(rc, 1)
        self.assertIn(text, out)
        self.assertFalse((self.d / "summary.json").exists())

    def test_complete_round_is_deterministic_and_exploratory(self):
        fill(self.d, lambda q, slot: 3 + (len(q) + ord(slot[-1])) % 3)
        rc1, out1 = run_scoring(self.tmp.name)
        s1 = (self.d / "summary.json").read_text()
        rc2, out2 = run_scoring(self.tmp.name)
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertEqual(s1, (self.d / "summary.json").read_text())
        self.assertEqual(out1, out2)
        summary = json.loads(s1)
        self.assertEqual((summary["status"], summary["raters"], summary["seed"]), ("exploratory", 1, 7))
        self.assertEqual(summary["prompt_ids"], IDS)
        self.assertIn("EXPLORATORY", out1)

    def test_incomplete_round(self):
        self.assert_rejected(lambda d: d["scores"].pop(), "prompt not scored")

    def test_duplicate_prompt_id(self):
        self.assert_rejected(lambda d: d["scores"].append(copy.deepcopy(d["scores"][0])),
                             "duplicate prompt id")

    def test_missing_cell(self):
        self.assert_rejected(lambda d: d["scores"][0]["Model B"].update(coherence=None),
                             "Model B.coherence unscored")

    def test_missing_slot(self):
        self.assert_rejected(lambda d: d["scores"][0].pop("Model C"), "Model C missing")

    def test_out_of_scale(self):
        self.assert_rejected(lambda d: d["scores"][0]["Model A"].update(accuracy=6), "outside 1-5")
        self.assert_rejected(lambda d: d["scores"][0]["Model A"].update(accuracy="x"), "outside 1-5")

    def test_edited_prompts_detected(self):
        fill(self.d)
        (self.d / "prompts.json").write_text("[]")
        rc, out = run_scoring(self.tmp.name)
        self.assertEqual(rc, 1)
        self.assertIn("prompts.json does not match", out)


if __name__ == "__main__":
    unittest.main()
