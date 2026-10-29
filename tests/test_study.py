"""Offline invariants for the descriptive MLX study; no model or network needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study


class StudyMathTests(unittest.TestCase):
    def test_historical_report_does_not_certify_absent_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = study.render([], Path(tmp))
        self.assertIn("Archived follow-up", report)
        self.assertNotIn("All 300 kept responses passed", report)
        self.assertNotIn("Validated 20-session", report)

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
            declared = {"median": 3.0, "min": 2.0, "max": 4.0, "n": 3}
            self.assertEqual(study._check_trials(root, "decode-short", "decode_tps", declared), 3)
            with self.assertRaisesRegex(ValueError, "median disagrees"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 2.0, "min": 2.0, "max": 4.0, "n": 3})
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 3.0, "min": 2.0, "max": 4.0, "n": 5})
            with self.assertRaisesRegex(ValueError, "min disagrees"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 3.0, "min": 1.0, "max": 4.0, "n": 3})
            with self.assertRaisesRegex(ValueError, "max disagrees"):
                study._check_trials(root, "decode-short", "decode_tps", {"median": 3.0, "min": 2.0, "max": 5.0, "n": 3})
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


    def test_p3_manifest_rejects_same_missing_task_across_every_quant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3.json"
            all_rows = [{"id": pid, "category": category, "passed": True}
                        for pid, category in study.P3_EXPECTED.items()]
            self.assertEqual(len(all_rows), 30)
            for tag in study.TAGS:
                path.write_text(json.dumps({"accuracy_runs": all_rows[:-1]}))
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "missing=.*instr-08"):
                    study._p3_rows(path)
            all_rows[-1]["category"] = "function_call"
            path.write_text(json.dumps({"accuracy_runs": all_rows}))
            with self.assertRaisesRegex(ValueError, "wrong_categories"):
                study._p3_rows(path)

    def test_p3_requires_identity_and_exact_checkpoint_metadata(self):
        cfg = SimpleNamespace(id="mock", name="Mock model", params_total=2.5, params_active=2.5)
        ckpt = SimpleNamespace(tag="mlx-4bit", hf_repo="vendor/checkpoint", publisher="vendor",
                               weights_gb=1.4, bits_per_weight=4.5, local_dir="models/mock-4bit")
        data = {"model_id": cfg.id, "model_name": cfg.name, "checkpoint": ckpt.tag,
                "hf_repo": ckpt.hf_repo, "publisher": ckpt.publisher, "weights_gb": ckpt.weights_gb,
                "bits_per_weight": ckpt.bits_per_weight, "params_total": cfg.params_total,
                "params_active": cfg.params_active, "model_path": ckpt.local_dir,
                "grader_version": "p3-grader-v2", "accuracy_runs": [
                    {"id": pid, "category": category, "passed": True}
                    for pid, category in study.P3_EXPECTED.items()]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3.json"
            path.write_text(json.dumps(data))
            grades, version = study._p3_rows(path, cfg=cfg, ckpt=ckpt)
            self.assertEqual(len(grades), 30)
            self.assertEqual(version, "p3-grader-v2")
            for key, wrong in (("model_id", "other"), ("checkpoint", "mlx-8bit"),
                               ("hf_repo", "other/repo"), ("publisher", "third-party"),
                               ("weights_gb", 7.1), ("bits_per_weight", 8.5),
                               ("model_path", "models/other-4bit")):
                modified = dict(data, **{key: wrong})
                path.write_text(json.dumps(modified))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                    study._p3_rows(path, cfg=cfg, ckpt=ckpt)
            missing = dict(data)
            missing.pop("model_id")
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "checkpoint identity mismatch"):
                study._p3_rows(path, cfg=cfg, ckpt=ckpt)

    def test_legacy_p3_allowlist_validates_arm_and_model_path(self):
        cfg = SimpleNamespace(id="nemotron-3.5-bolt", name="Nemotron")
        ckpt = SimpleNamespace(tag="optiq-4bit", local_dir="models/optiq-4bit")
        doc = {"arm": "3", "model_path": "/old/user/models/optiq-4bit",
               "accuracy_runs": [{"id": pid, "category": category, "passed": False}
                                 for pid, category in study.P3_EXPECTED.items()]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_p3.json"
            path.write_text(json.dumps(doc))
            _, version = study._p3_rows(path, cfg=cfg, ckpt=ckpt)
            self.assertIsNone(version)
            path.write_text(json.dumps(dict(doc, arm="1")))
            with self.assertRaisesRegex(ValueError, "legacy P3 arm identity mismatch"):
                study._p3_rows(path, cfg=cfg, ckpt=ckpt)
            ckpt = SimpleNamespace(tag="mlx-8bit", local_dir="models/optiq-4bit")
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "missing modern P3 identity"):
                study._p3_rows(path, cfg=cfg, ckpt=ckpt)

    def test_p4_discovers_archived_and_default_layout_and_refuses_misattribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"model_id": "minicpm5-2b", "checkpoint": "mlx-4bit",
                       "passed": 0, "total": 20, "status": "pilot_unvalidated"}
            paths = [root / "p4-round1/minicpm5-2b/p4/mlx-4bit-results.json",
                     root / "minicpm5-2b/p4/mlx-4bit-results.json"]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
            self.assertEqual(len(study._status_p4(root)), 2)
            paths[-1].write_text(json.dumps(dict(payload, model_id="other")))
            with self.assertRaisesRegex(ValueError, "P4 pilot identity mismatch"):
                study._status_p4(root)

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
                         "ttft_long": 1.0, "ttft_min": 0.9, "ttft_max": 1.1,
                         "prompt_tokens_long": 100, "prompt_tokens_decode": 22,
                         "decode_gen": {"median": 120, "min": 117, "max": 123},
                         "long_gen": {"median": 35, "min": 30, "max": 40},
                         "params_total": 8.468, "params_active": 1.3,
                         "bpw": 4.5,
                         "publisher": "vendor", "categories": {
                            "arithmetic_ml": (int(verdicts[0]), 1),
                            "function_call": (int(verdicts[1]), 1),
                            "instruction_following": (0, 0)},
                         "grades": grades, "stage": "reported complete", "condition": "clean",
                         "grade_version": "p3-grader-v2"})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(study.registry, "ORDER", ["mock"]):
            report = study.render(rows, Path(tmp))
        self.assertIn("1/2 → 0/2", report)
        self.assertIn("1/2 → 2/2", report)
        self.assertIn("1.000 [0.900–1.100]", report)
        self.assertIn("22 → 120 [117–123]", report)
        self.assertIn("100 → 35 [30–40]", report)
        self.assertIn("8.47/1.30", report)
        self.assertIn("mlx-4bit: vendor / 4.50", report)
        self.assertIn("not confidence intervals", report)
        self.assertIn("No P4 reliability value is eligible", report)


if __name__ == "__main__":
    unittest.main()
