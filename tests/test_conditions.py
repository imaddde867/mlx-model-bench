"""Offline tests for run-condition recording: sysinfo parsers, per-trial snapshots, spread gate."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bench
import gates
import registry
import sysinfo
from test_runner import install_fake_mlx

QUIET = {"swap_used_gb": 0.0, "mem_free_pct": 60, "power": "ac", "cpu_speed_limit_pct": 100,
         "load_1m": 0.5, "top_cpu": []}


class ParserTests(unittest.TestCase):
    def test_swap_megabytes_and_gigabytes(self):
        self.assertEqual(sysinfo.parse_swap("total = 2048.00M  used = 1024.00M  free = 1024.00M  (encrypted)"), 1.0)
        self.assertEqual(sysinfo.parse_swap("total = 4.00G  used = 2.50G  free = 1.50G"), 2.5)
        self.assertIsNone(sysinfo.parse_swap("garbage"))
        self.assertIsNone(sysinfo.parse_swap(None))

    def test_memory_free_percentage(self):
        text = "The system has 17179869184 (1048576 pages)\nSystem-wide memory free percentage: 41%\n"
        self.assertEqual(sysinfo.parse_free_pct(text), 41)
        self.assertIsNone(sysinfo.parse_free_pct("nothing here"))

    def test_power_source(self):
        self.assertEqual(sysinfo.parse_power("Now drawing from 'AC Power'\n -InternalBattery-0"), "ac")
        self.assertEqual(sysinfo.parse_power("Now drawing from 'Battery Power'"), "battery")
        self.assertIsNone(sysinfo.parse_power(""))

    def test_thermal_limit_absent_is_unknown_not_100(self):
        self.assertEqual(sysinfo.parse_cpu_speed_limit("CPU_Speed_Limit = 80"), 80)
        self.assertIsNone(sysinfo.parse_cpu_speed_limit("Note: No thermal warning level has been recorded"))

    def test_top_cpu_excludes_self_sorts_and_uses_basename(self):
        ps = "\n".join([
            "  10  45.0 /Applications/Safari.app/Contents/MacOS/Safari",
            "  11   2.0 /usr/sbin/WindowServer",
            "  12  90.0 /usr/bin/python3",
            "  13   0.2 /usr/bin/idle",
            "bad line",
        ])
        got = sysinfo.parse_top_cpu(ps, own_pid=12)
        self.assertEqual([p["proc"] for p in got], ["Safari", "WindowServer"])
        self.assertEqual(got[0]["cpu_pct"], 45.0)

    def test_warnings_name_each_cause(self):
        snap = {"power": "battery", "cpu_speed_limit_pct": 70, "swap_used_gb": 2.0, "mem_free_pct": 10,
                "top_cpu": [{"proc": "Safari", "cpu_pct": 60.0}, {"proc": "tiny", "cpu_pct": 3.0}]}
        text = " | ".join(sysinfo.warnings(snap))
        for needle in ("battery", "throttle", "swap", "memory free 10%", "Safari 60%"):
            self.assertIn(needle, text)
        self.assertNotIn("tiny", text)
        self.assertEqual(sysinfo.warnings(QUIET), [])

    def test_snapshot_never_raises_when_commands_are_missing(self):
        with patch.object(sysinfo.subprocess, "run", side_effect=FileNotFoundError):
            snap = sysinfo.snapshot()
        self.assertIsNone(snap["swap_used_gb"])
        self.assertEqual(snap["top_cpu"], [])


class SpreadTests(unittest.TestCase):
    def test_spread_ratio(self):
        self.assertEqual(bench.spread_ratio([6.857, 6.2372, 5.0529, 2.29, 1.8398]), 3.73)
        self.assertEqual(bench.spread_ratio([1.0, 1.0]), 1.0)
        self.assertIsNone(bench.spread_ratio([1.0]))
        self.assertIsNone(bench.spread_ratio([None, None]))


class RecordingTests(unittest.TestCase):
    GOOD = {"prompt_tokens": 7, "generation_tokens": 3, "ttft_s": 0.1, "prefill_tps": 70.0,
            "wall_s": 0.5, "decode_tps": 30.0}

    def setUp(self):
        # Gate behaviour is covered in test_gates.py; these tests exercise run recording.
        patcher = patch.object(gates, "refuse", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, ttfts, extra=(), snap=QUIET):
        cfg = registry.get_model("nemotron-3.5-bolt")
        rows = iter({**self.GOOD, "ttft_s": t} for t in [ttfts[0]] + list(ttfts))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "prompts" / "speed").mkdir(parents=True)
            (temp / "prompts" / "speed" / "fixture.txt").write_text("synthetic prompt")
            argv = ["bench.py", "--model", cfg.id, "--checkpoint", "mlx-4bit", "--prompts", "fixture",
                    "--trials", str(len(ttfts)), "--max-tokens", "8", "--out", str(temp), *extra]
            with (
                patch.dict(sys.modules, install_fake_mlx()),
                patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
                patch.object(bench, "versions", return_value={"mlx": "fixture"}),
                patch.object(bench, "PROMPTS", temp / "prompts"),
                patch.object(bench, "run_mlx", side_effect=lambda *a, **k: next(rows)),
                patch.object(bench.sysinfo, "snapshot", return_value=dict(snap)),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = bench.main()
            out = temp / cfg.id / "mlx-4bit-inproc"
            summary = json.loads((out / "summary.json").read_text())
            trial = json.loads((out / "fixture-trial1.json").read_text())
            return status, summary, trial

    def test_every_trial_records_its_own_start_and_the_system_snapshot(self):
        _, _, trial = self.run_cli([0.1] * 5)
        self.assertIn("started_at", trial)
        self.assertEqual(trial["system"]["power"], "ac")

    def test_condition_label_and_caveats_reach_the_summary(self):
        busy = {**QUIET, "top_cpu": [{"proc": "Safari", "cpu_pct": 80.0}]}
        _, summary, _ = self.run_cli([0.1] * 5, ["--condition", "loaded"], snap=busy)
        self.assertEqual(summary["condition"], "loaded")
        self.assertEqual(summary["condition_caveats"], ["busy: Safari 80%"])

    def test_default_condition_is_unspecified_and_quiet_has_no_caveats(self):
        _, summary, _ = self.run_cli([0.1] * 5)
        self.assertEqual(summary["condition"], "unspecified")
        self.assertEqual(summary["condition_caveats"], [])

    def test_spread_is_recorded_and_does_not_block_by_default(self):
        status, summary, _ = self.run_cli([6.857, 6.2372, 5.0529, 2.29, 1.8398])
        self.assertEqual(status, 0)
        self.assertEqual(summary["prompts"]["fixture"]["spread_ratio"]["ttft_s"], 3.73)
        self.assertTrue(summary["publishable"])

    def test_max_spread_makes_a_wobbly_run_non_publishable(self):
        _, summary, _ = self.run_cli([6.857, 6.2372, 5.0529, 2.29, 1.8398], ["--max-spread", "1.5"])
        self.assertFalse(summary["publishable"])
        self.assertTrue(any("unstable trials" in r for r in summary["non_publishable_reasons"]))

    def test_max_spread_passes_a_steady_run(self):
        _, summary, _ = self.run_cli([0.100, 0.101, 0.102, 0.100, 0.101], ["--max-spread", "1.5"])
        self.assertTrue(summary["publishable"])


if __name__ == "__main__":
    unittest.main()
