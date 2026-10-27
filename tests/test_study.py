"""Offline invariants for the descriptive MLX study; no model or network needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study


class StudyMathTests(unittest.TestCase):
    def test_pareto_keeps_tradeoffs_and_excludes_dominated(self):
        rows = [
            {"name": "mini", "weights_gb": 1.0, "decode_short": 65.0},
            {"name": "slow", "weights_gb": 2.0, "decode_short": 50.0},
            {"name": "fast", "weights_gb": 5.0, "decode_short": 75.0},
        ]
        front = study.pareto(rows, (("weights_gb", -1), ("decode_short", 1)))
        self.assertEqual([r["name"] for r in front], ["mini", "fast"])

    def test_pareto_does_not_make_missing_data_zero(self):
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            study.pareto([{"weights_gb": 1.0}], (("weights_gb", -1), ("decode_short", 1)))
        with self.assertRaises(ValueError):
            study.pareto([{"weights_gb": 1.0, "decode_short": float('nan')}],
                         (("weights_gb", -1), ("decode_short", 1)))

    def test_exact_paired_test(self):
        self.assertEqual(study.exact_mcnemar(0, 0), 1.0)
        self.assertEqual(study.exact_mcnemar(3, 0), 0.25)
        self.assertEqual(study.exact_mcnemar(1, 2), 1.0)
        self.assertEqual(study.exact_mcnemar(5, 0), 0.0625)

    def test_pairing_is_by_id_not_order_and_detects_drift(self):
        a = {"grades": {"a": ("arithmetic_ml", True), "b": ("function_call", False)}}
        b = {"grades": {"b": ("function_call", True), "a": ("arithmetic_ml", False)}}
        self.assertEqual(study.paired_changes(a, b),
                         {"gained": ["b"], "lost": ["a"], "discordant": 2})
        with self.assertRaises(ValueError):
            study.paired_changes(a, {"grades": {"a": ("arithmetic_ml", True)}})
        with self.assertRaises(ValueError):
            study.paired_changes(a, {"grades": {"a": ("function_call", False),
                                                 "b": ("function_call", True)}})

    def test_p3_rejects_duplicates_or_non_boolean_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3.json"
            path.write_text(json.dumps({"accuracy_runs": [
                {"id": "same", "category": "function_call", "passed": True},
                {"id": "same", "category": "function_call", "passed": False}]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                study._p3_rows(path)
            path.write_text(json.dumps({"accuracy_runs": [
                {"id": "ok", "category": "function_call", "passed": 1}]}))
            with self.assertRaisesRegex(ValueError, "invalid"):
                study._p3_rows(path)

    def test_trials_validate_median_and_require_complete_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, value in enumerate((4.0, 2.0, 3.0), 1):
                (root / f"decode-short-trial{i}.json").write_text(json.dumps({
                    "ok": True, "discarded": False, "decode_tps": value}))
            declared = {"median": 3.0, "n": 3}
            self.assertEqual(study._check_trials(root, "decode-short", "decode_tps", declared), 3)
            with self.assertRaisesRegex(ValueError, "median disagrees"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 2.0, "n": 3})
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 3.0, "n": 5})
            (root / "decode-short-trial2.json").write_text(json.dumps({
                "ok": False, "discarded": False, "decode_tps": 2.0}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study._check_trials(root, "decode-short", "decode_tps", declared)

    def test_fixed_pilot_is_not_included_in_pareto(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p4-round1" / "minicpm5-2b" / "p4" / "mlx-4bit-results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"model_id": "minicpm5-2b", "checkpoint": "mlx-4bit",
                                        "passed": 0, "total": 20, "status": "pilot_unvalidated"}))
            self.assertEqual(study._status_p4(Path(tmp))[0][4], "pilot_unvalidated")
            # Pareto only takes explicitly provided performance rows; pilots are separate.
            self.assertNotIn("passed", study.pareto(
                [{"weights_gb": 1.0, "decode_short": 50.0}],
                (("weights_gb", -1), ("decode_short", 1)))[0])

    def test_render_quantization_uses_passes_not_denominators(self):
        # Patch the registry order for a small synthetic fixture.
        rows = []
        for tag, weight, speed, verdicts in (
            ("mlx-4bit", 1.0, 50.0, [True, False]),
            ("optiq-4bit", 1.5, 40.0, [False, False]),
            ("mlx-8bit", 2.0, 30.0, [True, True]),
        ):
            grades = {"math-01": ("arithmetic_ml", verdicts[0]),
                      "tool-01": ("function_call", verdicts[1])}
            rows.append({"model_id": "mock", "model": "Mock model", "tag": tag,
                         "weights_gb": weight, "decode_short": speed,
                         "decode_min": speed, "decode_max": speed,
                         "ttft_long": 1.0, "prompt_tokens_long": 100, "bpw": 4.5,
                         "publisher": "vendor", "categories": {
                            "arithmetic_ml": (int(verdicts[0]), 1),
                            "function_call": (int(verdicts[1]), 1),
                            "instruction_following": (0, 0)},
                         "grades": grades, "stage": "reported complete", "condition": "clean",
                         "grade_version": "p3-grader-v2"})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(study.registry, "ORDER", ["mock"]):
            report = study.render(rows, Path(tmp))
        self.assertIn("1/30 → 0/30", report)
        self.assertIn("1/30 → 2/30", report)
        self.assertIn("not confidence intervals", report)
        self.assertIn("No P4 reliability value is eligible", report)


if __name__ == "__main__":
    unittest.main()
