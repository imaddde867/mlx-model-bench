"""Offline completion-pilot regressions; never load model weights."""
import contextlib
import io
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PilotTests(unittest.TestCase):
    def test_pilot_uses_study_cap(self):
        import pilot
        import replicated
        self.assertEqual(replicated.MAX_TOKENS, 512)
        self.assertEqual(pilot.MAX_TOKENS, replicated.MAX_TOKENS)

    def test_help_runs_without_loading_models(self):
        result = subprocess.run([sys.executable, str(ROOT / "pilot.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--evidence", result.stdout)

    def test_completion_gate_rejects_wrong_or_unfinished_answers(self):
        import pilot
        for text, finish, flag in (("Forty", "stop", "wrong_answer"),
                                   ("<think>work", "stop", "reasoning_only"),
                                   ("Forty years", "length", "truncated"),
                                   ("Forty years", None, "unexpected_finish_reason:None"),
                                   ("", "stop", "empty_output")):
            with self.subTest(text=text, finish=finish):
                self.assertIn(flag, pilot.classify(text, 3, 256, finish))
        self.assertEqual(pilot.classify("<think>work</think>Forty years.", 8, 256, "stop"), [])

    def test_main_records_evidence_and_returns_failure_for_skipped_model(self):
        import pilot
        with tempfile.TemporaryDirectory() as tmp:
            args = ["pilot.py", "--out", tmp, "--condition", "clean", "--evidence", tmp]
            with (mock.patch.object(sys, "argv", args),
                  mock.patch.object(pilot, "run_id", return_value="test-run"),
                  mock.patch.object(pilot.sysinfo, "snapshot", return_value={"power": "ac"}),
                  mock.patch.object(pilot.sysinfo, "warnings", return_value=[]),
                  mock.patch.object(pilot.metadata, "version", return_value="test"),
                  mock.patch("registry.Checkpoint.available", new_callable=mock.PropertyMock, return_value=True),
                  mock.patch.object(pilot, "run_model", side_effect=[{"status": "ok"}] * 3 +
                                    [{"status": "skipped_unverified"}]) as run,
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(pilot.main(), 1)
                self.assertEqual(run.call_args.args[-1], Path(tmp))
                manifest = json.loads((Path(tmp) / "test-run" / "manifest.json").read_text())
                self.assertEqual(manifest["condition_warnings_observed"], [])
                self.assertEqual(manifest["evidence_dir"], str(Path(tmp).resolve()))
                with self.assertRaises(SystemExit):
                    pilot.main()


if __name__ == "__main__":
    unittest.main()
