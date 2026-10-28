"""Offline regression coverage for the session-level experiment (no MLX, network or Mac)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import replicated as r


class ReplicatedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = ["model-a", "model-b", "model-c", "model-d"]
        self.plan = {"schema": r.VERSION, "seed": 23,
                     "models": [{"model_id": m, "checkpoint": "mlx-4bit", "tier": "4bit",
                                 "hf_repo": f"vendor/{m}", "publisher": "vendor", "bpw": 4.5,
                                 "weights_gb": 2.0, "template_kwargs": {"enable_thinking": False}}
                                for m in self.ids],
                     "prompts": [{"regime": p, "sha256": "abc"} for p in r.REGIMES],
                     "settings": {"transport": "inproc", "engine": "mlx_lm",
                                  "temperature": 0.0, "nonce": False,
                                  "prompt_cache": "none (fresh process, no prompt_cache)",
                                  "max_tokens": 256, "warmups_per_regime": 1,
                                  "kept_trials_per_regime": 5},
                     "schedule": r.schedule(self.ids, 23), "source_commit": "f" * 40,
                     "machine": {"arch": "arm64", "macos": "27.0.1",
                                 "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"}}}
        self.plan["plan_id"] = r.digest(self.plan)
        r.write_once(self.root / "plan.json", self.plan)
        self.snap = {"power": "ac", "cpu_speed_limit_pct": 100, "swap_used_gb": 0,
                     "mem_free_pct": 70, "top_cpu": [], "load_1m": 1.0}

    def fixture(self, block=1, pos=1, corrupt=None):
        model_id = self.plan["schedule"][block - 1]["order"][pos - 1]
        path = r.slot_path(self.root, block, pos, model_id)
        path.mkdir(parents=True)
        started = {"schema": 1, "session_id": "uuid123", "plan_id": self.plan["plan_id"],
                   "block": block, "position": pos, "model_id": model_id,
                   "operator_condition": "clean", "pre_system": self.snap,
                   "started_utc": "2026-10-28T08:00:00+00:00"}
        r.write_once(path / "started.json", started)
        raw = path / "raw" / model_id / "mlx-4bit-inproc"
        raw.mkdir(parents=True)
        summary = {"model_id": model_id, "checkpoint": "mlx-4bit", "tier": "4bit",
                   "hf_repo": f"vendor/{model_id}", "publisher": "vendor", "transport": "inproc",
                   "engine": "mlx_lm", "trials": 5, "max_tokens": 256,
                   "versions": self.plan["machine"]["packages"],
                   "greedy": True, "template_kwargs": {"enable_thinking": False},
                   "condition": "clean", "outcome": "completed", "publishable": True,
                   "session_id": "uuid123", "prompts": {}}
        for ri, regime in enumerate(r.REGIMES):
            values = []
            for trial in range(6):
                row = {"model_id": model_id, "checkpoint": "mlx-4bit", "prompt": regime,
                       "trial": trial, "discarded": trial == 0, "condition": "clean",
                       "ok": True, "nonce": False, "max_tokens": 256,
                       "session_id": "uuid123",
                       "template_kwargs": {"enable_thinking": False},
                       "hf_repo": f"vendor/{model_id}", "publisher": "vendor",
                       "versions": self.plan["machine"]["packages"],
                       "prompt_file": f"/source/prompts/speed/{regime}.txt",
                       "system": self.snap, "started_at": "2026-10-28T08:00:00+00:00",
                       "finish_reason": "stop", "wall_s": 1.0 + ri + trial / 10,
                       "request_ttft_s": 0.1 + trial / 100, "peak_gb": 2 + trial / 10,
                       "prompt_tokens": 80 + ri * 100, "generation_tokens": 50}
                r.write_once(raw / f"{regime}-{'warmup' if trial == 0 else f'trial{trial}'}.json", row)
                if trial:
                    values.append(row)
            summary["prompts"][regime] = {"kept_ok": 5, "warmup_ok": True,
                                          "prompt_tokens": values[0]["prompt_tokens"],
                                          **{k: {"median": r.statistics.median(v[k] for v in values)}
                                             for k in ("wall_s", "request_ttft_s", "peak_gb")}}
        r.write_once(raw / "summary.json", summary)
        (path / "bench.log").write_text("simulated - not real inference\n")
        s = {"schema": 1, "session_id": "uuid123", "plan_id": self.plan["plan_id"],
             "block": block, "position": pos, "model_id": model_id,
             "operator_condition": "clean", "return_code": 0, "status": "completed",
             "observed": {"before": self.snap, "after": self.snap, "caveats": []},
             "host_memory": {"source": "vm_stat: total - free - inactive - speculative; host-wide estimate",
                             "samples": 30, "peak_gb_est": 8.5},
             "artifacts_sha256": r.evidence_hashes(path)}
        r.write_once(path / "session.json", s)
        return path, model_id

    def test_schedule_balances_positions_and_is_seeded(self):
        seq = r.schedule(self.ids, 23)
        self.assertEqual(seq, r.schedule(self.ids, 23))
        self.assertNotEqual(seq, r.schedule(self.ids, 24))
        self.assertEqual(len(seq), 5)
        for model_id in self.ids:
            places = [b["order"].index(model_id) for b in seq[:4]]
            self.assertEqual(sorted(places), list(range(4)))
        for block in seq:
            self.assertEqual(set(block["order"]), set(self.ids))

    def test_plan_is_content_addressed_and_tamper_detected(self):
        r.check_plan(self.plan)
        broken = dict(self.plan, seed=99)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            r.check_plan(broken)
        broken["plan_id"] = r.digest({k: v for k, v in broken.items() if k != "plan_id"})
        with self.assertRaisesRegex(ValueError, "run order"):
            r.check_plan(broken)

    def test_mac_gate_excludes_other_16gb_macs_and_linux(self):
        m = {"os": "Darwin", "arch": "arm64", "chip": "Apple M4",
             "cpu_count": 10, "ram_bytes": 16 * 2**30,
             "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"}}
        r.require_mac(m)
        for changed in ({"chip": "Apple M2"}, {"cpu_count": 8},
                        {"os": "Linux"}, {"ram_bytes": 32 * 2**30},
                        {"packages": {"mlx": "0.32.5", "mlx-lm": "0.32.2"}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                r.require_mac(dict(m, **changed))

    def test_vm_stat_estimate_is_explicit_and_fails_closed(self):
        s = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\nPages speculative: 50.\n"
        self.assertAlmostEqual(r.parse_vm_stat(s, 16 * 2**30), (16 * 2**30 - 350 * 16384) / 1e9, places=4)
        self.assertIsNone(r.parse_vm_stat("", 16 * 2**30))
        self.assertIsNone(r.parse_vm_stat(s, None))

    def test_exclusive_write_and_session_slot(self):
        p = self.root / "once.json"
        r.write_once(p, {"x": 1})
        with self.assertRaises(FileExistsError):
            r.write_once(p, {"x": 2})
        self.assertEqual(r.load_json(p), {"x": 1})

    def test_valid_fixture_and_evidence_tamper(self):
        p, mid = self.fixture()
        self.assertEqual(r.validate_session(p, self.plan, 1, 1, mid), [])
        trial = p / "raw" / mid / "mlx-4bit-inproc" / "short-trial1.json"
        data = r.load_json(trial)
        data["wall_s"] = 400
        trial.write_text(json.dumps(data))
        self.assertIn("SHA256", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_condition_label_not_rewritten_by_observed_cpu(self):
        p, mid = self.fixture()
        s = r.load_json(p / "session.json")
        s["observed"]["caveats"] = ["busy: mediaanalysisd 225%"]
        s["observed"]["before"]["top_cpu"] = [{"proc": "mediaanalysisd", "cpu_pct": 225}]
        self.assertEqual(s["operator_condition"], "clean")
        self.assertIn("busy", s["observed"]["caveats"][0])
        # Changing the operator label does not change the original per-trial label.
        s["operator_condition"] = "loaded"
        (p / "session.json").write_text(json.dumps(s))
        self.assertIn("condition provenance", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_invalid_median_missing_trial_and_truncation(self):
        p, mid = self.fixture()
        raw = p / "raw" / mid / "mlx-4bit-inproc"
        summ = r.load_json(raw / "summary.json")
        summ["prompts"]["short"]["wall_s"]["median"] = 999
        (raw / "summary.json").write_text(json.dumps(summ))
        self.assertIn("median mismatch", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        # Preserve, flag and disqualify truncated outputs instead of treating as successful responses.
        t = raw / "short-trial2.json"
        row = r.load_json(t)
        row["finish_reason"] = "length"
        t.write_text(json.dumps(row))
        self.assertIn("truncated", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        (raw / "long-trial5.json").unlink()
        self.assertIn("absent evidence", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_incomplete_design_cannot_be_reported(self):
        p, mid = self.fixture()
        _, passed, problems = r.validate(self.root)
        self.assertEqual(len(passed), 1)
        self.assertEqual(len(problems), 19)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            r.report(self.root)

    def test_complete_synthetic_design_generates_deterministic_report(self):
        for block in self.plan["schedule"]:
            for position in range(1, 5):
                self.fixture(block["block"], position)
        _, good, errors = r.validate(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(len(good), 20)
        first = r.report(self.root)
        self.assertEqual(first, r.report(self.root))
        self.assertIn("5 counterbalanced blocks", first)
        self.assertIn("95% block bootstrap", first)
        self.assertIn("Operator-selected condition", first)
        self.assertIn("vm_stat", first)
        self.assertIn("time to a correct/usable answer", first)
        self.assertEqual(first.count("| short |"), 7)  # four model rows plus three contrasts

    def test_block_bootstrap_not_trial_bootstrap(self):
        result = r.bootstrap_median([1, 2, 3, 4, 5], 7)
        self.assertEqual(result[0], 3)
        self.assertLessEqual(result[1], result[0])
        self.assertGreaterEqual(result[2], result[0])
        self.assertEqual(result, r.bootstrap_median([1, 2, 3, 4, 5], 7))
        with self.assertRaisesRegex(ValueError, "fewer than five"):
            r.bootstrap_median([1, 2, 3, 4], 7)

    def test_no_overwrite_even_when_resume_requested_for_bad_slot(self):
        p, _ = self.fixture()
        s = r.load_json(p / "session.json")
        s["status"] = "failed"
        (p / "session.json").write_text(json.dumps(s))
        with mock.patch.object(r, "verify_host"):
            with self.assertRaises(FileExistsError):
                r.run_block(self.root, 1, "clean", resume=True)
        self.assertEqual(r.load_json(p / "session.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()