"""Offline tests for compare.py: cross-model table built from result rows."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compare


def block(med, lo=None, hi=None):
    return {"median": med, "min": lo if lo is not None else med, "max": hi if hi is not None else med, "n": 5}


def summary(**over):
    s = {
        "model_name": "MiniCPM5-2B", "checkpoint": "mlx-4bit", "tier": "4bit", "bits_per_weight": 4.55,
        "publisher": "vendor", "weights_gb": 1.43, "params_total": 2.517, "params_active": 2.517,
        "condition": "loaded", "outcome": "completed", "publishable": True,
        "prompts": {
            "decode-short": {"decode_tps": block(69.0, 54.7, 71.7)},
            "decode-long": {"decode_tps": block(54.6), "generation_tokens": block(256)},
            "long": {"prompt_tokens": 8354, "ttft_s": block(14.6), "peak_gb": block(2.465)},
        },
    }
    s.update(over)
    return s


def write_trials(run_dir, ttfts):
    for i, t in enumerate(ttfts, 1):
        (run_dir / f"long-trial{i}.json").write_text(json.dumps({"ok": True, "discarded": False, "ttft_s": t}))
    (run_dir / "long-warmup.json").write_text(json.dumps({"ok": True, "discarded": True, "ttft_s": 99.0}))


class CompareTests(unittest.TestCase):
    def make(self, tmp, model="minicpm5-2b", run="mlx-4bit-inproc", s=None, ttfts=(1.0, 1.0, 1.1, 1.0, 1.0)):
        run_dir = Path(tmp) / model / run
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps(s or summary()))
        write_trials(run_dir, ttfts)
        return run_dir

    def test_row_carries_identity_tokens_and_effective_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp)
            (row,) = compare.build_rows(tmp)
        self.assertEqual(row["model"], "MiniCPM5-2B")
        self.assertEqual(row["prompt_tokens_long"], 8354)
        self.assertEqual(row["gen_tokens_decode_long"], 256)
        self.assertEqual(row["gbps"], 98.7)  # 1.43 GB x 69.0 tok/s
        self.assertFalse(row["sparse"])
        self.assertEqual(row["condition"], "loaded")
        self.assertIn("not measured memory bandwidth", compare.to_markdown([row]))
        self.assertIn("Weight-read proxy GB/s", compare.to_markdown([row]))

    def test_spread_ignores_the_discarded_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp, ttfts=(6.857, 6.2372, 5.0529, 2.29, 1.8398))
            (row,) = compare.build_rows(tmp)
        self.assertEqual(row["spread_ttft"], 3.73)

    def test_moe_bandwidth_is_marked_as_a_ceiling_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp, s=summary(params_total=8.468, params_active=1.3))
            rows = compare.build_rows(tmp)
        self.assertTrue(rows[0]["sparse"])
        self.assertIn("| ~98.7 |", compare.to_markdown(rows))

    def test_legacy_summary_metadata_comes_from_the_registry(self):
        legacy = {"prompts": summary()["prompts"], "transport": "inproc"}  # no identity fields at all
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp, model="nemotron-3.5-bolt", s=legacy)
            (row,) = compare.build_rows(tmp)
        self.assertEqual(row["model"], "NVIDIA Nemotron 3.5 Bolt E4B")
        self.assertEqual(row["tier"], "4bit")
        self.assertEqual(row["condition"], "not recorded")

    def test_accuracy_reads_new_and_historical_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp)
            p3 = Path(tmp) / "minicpm5-2b" / "p3"
            p3.mkdir()
            (p3 / "accuracy_summary.json").write_text(json.dumps({"checkpoints": {
                "mlx-4bit": {"accuracy": {"total_passed": 19, "total_prompts": 30}}}}))
            self.assertEqual(compare.build_rows(tmp)[0]["accuracy"], "19/30")
            old = Path(tmp) / "old"
            old.mkdir()
            (old / "p3").mkdir()
            (old / "p3" / "accuracy_summary.json").write_text(json.dumps({
                "arm1": {"accuracy": {"total_passed": 24, "total_prompts": 30}}}))
            self.assertEqual(compare._accuracy(old), {"mlx-4bit": "24/30"})

    def test_repeat_run_is_listed_separately_without_replacing_matrix_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp, ttfts=(1.0, 1.0, 1.1, 1.0, 1.0))
            repeat_dir = Path(tmp) / "minicpm5-2b" / "mlx-4bit-inproc-decode-check"
            repeat_dir.mkdir()
            repeat = summary(condition="loaded", prompts={
                "decode-short": {"prompt_tokens": 55, "decode_tps": block(71.6, 71.4, 72.1),
                                 "generation_tokens": block(256), "ttft_s": block(0.1133, 0.1107, 0.118)},
            })
            (repeat_dir / "summary.json").write_text(json.dumps(repeat))
            (repeat_dir / "decode-short-trial1.json").write_text(json.dumps({"ok": True, "discarded": False}))
            rows = compare.build_rows(tmp)
            repeats = compare.build_repeat_rows(tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decode_short"], 69.0)
            self.assertEqual(len(repeats), 1)
            self.assertEqual(repeats[0]["repeat_decode"], 71.6)
            text = compare.to_markdown(rows, repeat_rows=repeats)
            self.assertIn("Repeat decode-short checks", text)
            self.assertIn("not measured memory bandwidth", text)
            self.assertIn("71.6 [71.4-72.1]", text)
            self.assertIn("69.0 [54.7-71.7]", text)

    def test_http_rows_are_a_separate_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make(tmp, run="mlx-4bit-http")
            self.assertEqual(compare.build_rows(tmp, "inproc"), [])
            self.assertEqual(len(compare.build_rows(tmp, "http")), 1)


if __name__ == "__main__":
    unittest.main()
