"""Offline checks for the experimental LFM3 parity fixture (no MLX, torch or network).

The fixture and run records are produced on a Mac by experimental/lfm3/; these
tests check that what is committed is internally consistent, matches its pins,
and that the comparison and gate arithmetic does what the README says.
"""

import copy
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental" / "lfm3"))

import lfm3_parity as lp

FX = lp.load()
RUNS = sorted(lp.RUNS.glob("*.json"))


def redigest(fx):
    fx["fixture_sha256"] = lp.digest(fx)
    return fx


def perfect_mlx(fx):
    """What an exact port would return: the reference itself."""
    ref = fx["reference"]
    return ref["topk_logits"], [row[0] for row in ref["topk_ids"]], list(ref["greedy_ids"])


def run_record(fx, **rows):
    base = lp.compare(fx["reference"], *perfect_mlx(fx))
    record = {
        "fixture_sha256": fx["fixture_sha256"],
        "versions": {"mlx": lp.PINS["mlx"], "mlx_lm_commit": lp.PINS["mlx_lm_commit"]},
        "rows": {"fp32": base, "cache_fp32": {"max_abs_diff": 0.0, "top1": 1.0},
                 "4bit": dict(base, top1=0.5, greedy_identical=3)},
    }
    record["rows"].update(rows)
    return record


class FixtureTests(unittest.TestCase):
    def test_committed_fixture_is_valid(self):
        self.assertEqual(lp.fixture_problems(FX), [])

    def test_fixture_runs_past_twice_the_window(self):
        self.assertEqual(len(FX["input"]["ids"]), lp.N_TOKENS)
        self.assertGreater(lp.N_TOKENS, 2 * FX["reference"]["window"])

    def test_requirements_match_pins(self):
        req = (lp.HERE / "requirements.txt").read_text()
        self.assertIn(f"mlx=={lp.PINS['mlx']}\n", req)
        self.assertIn(f"transformers=={lp.PINS['transformers']}\n", req)
        self.assertIn(f"torch=={lp.PINS['torch']}\n", req)
        self.assertRegex(req, rf"mlx-lm @ git\+https://github\.com/ml-explore/mlx-lm@{lp.PINS['mlx_lm_commit']}\n")

    def test_readme_names_every_pin(self):
        readme = (lp.HERE / "README.md").read_text()
        for key in ("mlx_lm_commit", "mlx", "transformers", "torch", "reference_revision", "quant_revision"):
            self.assertIn(lp.PINS[key], readme, key)
        self.assertIn(FX["fixture_sha256"], readme)

    def test_edited_logit_breaks_the_digest(self):
        fx = copy.deepcopy(FX)
        fx["reference"]["topk_logits"][300][0] += 0.5
        self.assertIn("fixture_sha256 does not match its content", lp.fixture_problems(fx))

    def test_changed_input_text_is_caught(self):
        fx = redigest(dict(copy.deepcopy(FX), input=dict(FX["input"], text_sha256="0" * 64)))
        self.assertTrue(any("changed since the fixture" in p for p in lp.fixture_problems(fx)))

    def test_moved_pin_is_caught(self):
        fx = redigest(dict(copy.deepcopy(FX), pins=dict(FX["pins"], quant_revision="main")))
        self.assertIn("pins differ from lfm3_parity.PINS", lp.fixture_problems(fx))

    def test_unsorted_topk_is_caught(self):
        fx = copy.deepcopy(FX)
        fx["reference"]["topk_logits"][5].reverse()
        self.assertTrue(any("not sorted" in p for p in lp.fixture_problems(redigest(fx))))

    def test_file_outside_the_fetch_allow_list_is_caught(self):
        fx = copy.deepcopy(FX)
        fx["weights"][lp.PINS["reference_repo"]]["README.md"] = "0" * 64
        self.assertTrue(any("outside CHECKPOINT_FILES" in p for p in lp.fixture_problems(redigest(fx))))

    def test_dump_round_trips_and_keeps_rows_on_one_line(self):
        text = lp.dump(FX)
        self.assertEqual(json.loads(text), FX)
        self.assertLess(text.count("\n"), 2 * lp.N_TOKENS + 200)


class CompareTests(unittest.TestCase):
    def test_exact_port_scores_perfectly(self):
        row = lp.compare(FX["reference"], *perfect_mlx(FX))
        self.assertEqual(row["max_abs_diff_topk"], 0)
        self.assertEqual((row["top1"], row["top1_after_window"]), (1.0, 1.0))
        self.assertEqual(row["greedy_identical"], row["greedy_len"])

    def test_flip_past_the_window_shows_in_the_tail(self):
        gathered, argmax, gen = perfect_mlx(FX)
        argmax[400] = argmax[400] + 1
        row = lp.compare(FX["reference"], gathered, argmax, gen)
        self.assertLess(row["top1_after_window"], 1.0)
        self.assertLess(row["top1"], 1.0)

    def test_first_divergence(self):
        self.assertEqual(lp.first_divergence([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(lp.first_divergence([1, 2], [1, 2, 3]), 2)
        self.assertEqual(lp.first_divergence([], [1]), 0)


class GateTests(unittest.TestCase):
    def test_committed_runs_pass(self):
        self.assertTrue(RUNS, "no run record under experimental/lfm3/runs/")
        for path in RUNS:
            run = json.loads(path.read_text())
            with self.subTest(run=path.name):
                self.assertEqual(lp.run_problems(run, FX), [])
                self.assertTrue(run["ok"])

    def test_exact_port_passes_and_4bit_is_not_gated(self):
        self.assertEqual(lp.run_problems(run_record(FX), FX), [])

    def test_fp32_logit_drift_fails(self):
        row = dict(lp.compare(FX["reference"], *perfect_mlx(FX)), max_abs_diff_topk=0.01)
        self.assertTrue(any("fp32: max abs diff" in p for p in lp.run_problems(run_record(FX, fp32=row), FX)))

    def test_greedy_divergence_fails(self):
        row = dict(lp.compare(FX["reference"], *perfect_mlx(FX)), greedy_identical=44)
        self.assertIn("fp32: greedy decode diverges at token 44", lp.run_problems(run_record(FX, fp32=row), FX))

    def test_cache_mismatch_fails(self):
        problems = lp.run_problems(run_record(FX, cache_fp32={"max_abs_diff": 0.2, "top1": 0.99}), FX)
        self.assertEqual(len([p for p in problems if p.startswith("cache_fp32")]), 2)

    def test_unpinned_mlx_lm_or_other_fixture_fails(self):
        run = run_record(FX)
        run["versions"]["mlx_lm_commit"] = None
        run["fixture_sha256"] = "f" * 64
        problems = lp.run_problems(run, FX)
        self.assertIn("run was made against a different fixture", problems)
        self.assertTrue(any(re.match(r"mlx_lm_commit None", p) for p in problems))


if __name__ == "__main__":
    unittest.main()
