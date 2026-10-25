"""Offline: P3 summary is merged per checkpoint and never overwrites a published tag. No model, no inference."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_p3


def entry(n):
    return {"accuracy": {"total_passed": n}}


class TestMerge(unittest.TestCase):
    def test_legacy_summary_without_checkpoints_keeps_its_keys(self):
        self.path.write_text(json.dumps({"timestamp": "t0", "arm1": {"name": "a"}}))
        run_p3.merge_summary(self.d, {"model_id": "m", "checkpoints": {"mlx-8bit": entry(25)}})
        got = json.loads(self.path.read_text())
        self.assertEqual(got["arm1"], {"name": "a"})
        self.assertEqual(got["checkpoints"], {"mlx-8bit": entry(25)})

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        self.path = self.d / "accuracy_summary.json"

    def test_new_tier_keeps_old_entries_byte_identical(self):
        old = {"model_id": "m", "timestamp": "t0", "checkpoints": {"mlx-4bit": entry(20)}}
        self.path.write_text(json.dumps(old, indent=2))
        run_p3.merge_summary(self.d, {"model_id": "m", "timestamp": "t1", "checkpoints": {"mlx-8bit": entry(25)}})
        got = json.loads(self.path.read_text())
        self.assertEqual(json.dumps(got["checkpoints"]["mlx-4bit"]), json.dumps(old["checkpoints"]["mlx-4bit"]))
        self.assertEqual(got["checkpoints"]["mlx-8bit"], entry(25))
        self.assertEqual(got["timestamp"], "t0")

    def test_creates_when_absent(self):
        run_p3.merge_summary(self.d, {"model_id": "m", "checkpoints": {"mlx-8bit": entry(1)}})
        self.assertEqual(list(json.loads(self.path.read_text())["checkpoints"]), ["mlx-8bit"])

    def test_existing_tag_refused(self):
        self.path.write_text(json.dumps({"checkpoints": {"mlx-4bit": entry(20)}}))
        self.assertEqual(run_p3.collisions(self.d, ["mlx-8bit"]), [])
        self.assertIn("mlx-4bit", run_p3.collisions(self.d, ["mlx-8bit", "mlx-4bit"])[0])

    def test_existing_results_file_refused(self):
        (self.d / "mlx-8bit-results.json").write_text("{}")
        self.assertIn("mlx-8bit-results.json", run_p3.collisions(self.d, ["mlx-8bit"])[0])


class FinalAnswerTests(unittest.TestCase):
    def test_strips_finished_thinking(self):
        self.assertEqual(run_p3.final_answer("<think>\nnotes 17*23\n</think>\n391"), "391")

    def test_unfinished_thinking_is_empty_not_graded(self):
        self.assertEqual(run_p3.final_answer("<think>\nstill reasoning 391"), "")

    def test_plain_answers_unchanged(self):
        self.assertEqual(run_p3.final_answer("391"), "391")

    def test_budget_scales_only_for_thinking_model(self):
        import registry
        self.assertEqual(registry.get_model("lfm2.5-8b-a1b").p3_budget_x, 4)
        self.assertEqual(registry.get_model("nemotron-3.5-bolt").p3_budget_x, 1)


if __name__ == "__main__":
    unittest.main()
