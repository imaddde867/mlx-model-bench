"""Offline regression coverage for the session-level experiment (no MLX, network or Mac)."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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
                     "prompts": [{"regime": p, "path": f"prompts/replicated-duration/{p}.txt", "sha256": "abc"}
                                 for p in r.REGIMES],
                     "settings": {"transport": "inproc", "engine": "mlx_lm",
                                  "temperature": 0.0, "nonce": False,
                                  "prompt_cache": "none (fresh process, no prompt_cache)",
                                  "max_tokens": 256, "warmups_per_regime": 1,
                                  "kept_trials_per_regime": 5},
                     "schedule": r.schedule(self.ids, 23), "source_commit": "f" * 40,
                     "evidence_dir": str(self.root),
                     "machine": {"arch": "arm64", "macos": "27.0.1", "ram_bytes": 16 * 2**30,
                                 "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"}}}
        self.plan["plan_id"] = r.digest(self.plan)
        r.write_once(self.root / "plan.json", self.plan)
        self.snap = {"power": "ac", "low_power_mode": 0, "cpu_speed_limit_pct": 100, "swap_used_gb": 0,
                     "mem_free_pct": 70, "top_cpu": [], "load_1m": 1.0}

    def fixture(self, block=1, pos=1, corrupt=None, overlap=False, duplicate_id=False, generation_tokens=50):
        model_id = self.plan["schedule"][block - 1]["order"][pos - 1]
        path = r.slot_path(self.root, block, pos, model_id)
        path.mkdir(parents=True)
        number = (block - 1) * 4 + pos - 1
        start = datetime(2026, 10, 28, 8, tzinfo=timezone.utc) + timedelta(minutes=number)
        if overlap:
            start -= timedelta(minutes=1)
        start_at = start.isoformat()
        end_at = (start + timedelta(seconds=30)).isoformat()
        sid = "session-1-1" if duplicate_id else f"session-{block}-{pos}"
        started = {"schema": r.VERSION, "session_id": sid, "plan_id": self.plan["plan_id"],
                   "block": block, "position": pos, "model_id": model_id,
                   "operator_condition": "clean", "pre_system": self.snap,
                   "started_utc": start_at}
        r.write_once(path / "started.json", started)
        raw = path / "raw" / model_id / "mlx-4bit-inproc"
        raw.mkdir(parents=True)
        summary = {"model_id": model_id, "checkpoint": "mlx-4bit", "tier": "4bit",
                   "hf_repo": f"vendor/{model_id}", "publisher": "vendor", "transport": "inproc",
                   "engine": "mlx_lm", "trials": 5, "max_tokens": self.plan["settings"]["max_tokens"],
                   "versions": self.plan["machine"]["packages"],
                   "greedy": True, "template_kwargs": {"enable_thinking": False},
                   "condition": "clean", "outcome": "completed", "publishable": True,
                   "session_id": sid, "prompts": {}, "condition_caveats": []}
        for ri, regime in enumerate(r.REGIMES):
            values = []
            for trial in range(6):
                row = {"model_id": model_id, "checkpoint": "mlx-4bit", "prompt": regime,
                       "trial": trial, "discarded": trial == 0, "condition": "clean",
                       "ok": True, "nonce": False, "max_tokens": self.plan["settings"]["max_tokens"],
                       "session_id": sid,
                       "template_kwargs": {"enable_thinking": False},
                       "hf_repo": f"vendor/{model_id}", "publisher": "vendor",
                       "versions": self.plan["machine"]["packages"],
                       "prompt_file": str(ROOT / "prompts" / "replicated-duration" / f"{regime}.txt"),
                       "system": self.snap, "started_at": start_at,
                       "finish_reason": "stop", "text": "Synthetic answer", "wall_s": 1.0 + ri + trial / 10,
                       "request_ttft_s": 0.1 + trial / 100, "peak_gb": 2 + trial / 10,
                       "prompt_tokens": 80 + ri * 100, "generation_tokens": generation_tokens}
                r.write_once(raw / f"{regime}-{'warmup' if trial == 0 else f'trial{trial}'}.json", row)
                if trial:
                    values.append(row)
            summary["prompts"][regime] = {"kept_ok": 5, "warmup_ok": True,
                                          "prompt_tokens": values[0]["prompt_tokens"],
                                          **{k: {"median": r.statistics.median(v[k] for v in values),
                                                 "min": min(v[k] for v in values),
                                                 "max": max(v[k] for v in values), "n": 5}
                                             for k in ("wall_s", "request_ttft_s", "peak_gb", "generation_tokens")}}
        r.write_once(raw / "summary.json", summary)
        (path / "bench.log").write_text("simulated - not real inference\n")
        s = {"schema": r.VERSION, "session_id": sid, "plan_id": self.plan["plan_id"],
             "block": block, "position": pos, "model_id": model_id,
             "operator_condition": "clean", "return_code": 0, "status": "completed",
             "error": None, "started_utc": start_at, "ended_utc": end_at,
             "observed": {"before": self.snap, "after": self.snap, "caveats": []},
             "host_memory": {"source": "vm_stat: total - free - inactive - speculative; host-wide estimate",
                             "samples": 3, "peak_gb_est": 8.5, "unit": "GB decimal", "interval_s": 0.5,
                             "sampled_gb": [7.7, 8.5, 8.1]}}
        r.write_once(path / "session_payload.json", s)
        s["artifacts_sha256"] = r.evidence_hashes(path)
        r.write_once(path / "session.json", s)
        return path, model_id

    def test_new_plan_freezes_512_cap(self):
        with mock.patch.object(r, "require_mac"), \
             mock.patch.object(r, "verified_identity", side_effect=lambda cfg, evidence: {"model_id": cfg.id}):
            plan = r.make_plan(self.root, 23, m={}, source_commit="test-commit")
        self.assertEqual(plan["settings"]["max_tokens"], 512)
        r.check_plan(plan)

    def test_512_cap_validates_longer_completed_generation(self):
        self.plan["settings"]["max_tokens"] = 512
        self.plan["plan_id"] = r.digest({k: v for k, v in self.plan.items() if k != "plan_id"})
        r.check_plan(self.plan)
        path, mid = self.fixture(generation_tokens=300)
        self.assertEqual(r.validate_session(path, self.plan, 1, 1, mid), [])
        trial = path / "raw" / mid / "mlx-4bit-inproc" / "short-warmup.json"
        row = r.load_json(trial)
        row["max_tokens"] = 256
        trial.write_text(json.dumps(row))
        self.assertIn("max_tokens mismatch", " ".join(
            r.validate_session(path, self.plan, 1, 1, mid)))
        row["max_tokens"] = 512
        row["generation_tokens"] = 513
        trial.write_text(json.dumps(row))
        self.assertIn("generation exceeds token cap", " ".join(
            r.validate_session(path, self.plan, 1, 1, mid)))

    def test_plan_rejects_unapproved_or_noninteger_caps(self):
        for cap in (255, 257, 1024, True, 256.0, 512.0, "512", None):
            with self.subTest(cap=cap):
                self.plan["settings"]["max_tokens"] = cap
                self.plan["plan_id"] = r.digest({k: v for k, v in self.plan.items() if k != "plan_id"})
                with self.assertRaisesRegex(ValueError, "token cap"):
                    r.check_plan(self.plan)

    def test_run_uses_512_cap_frozen_in_plan(self):
        self.plan["settings"]["max_tokens"] = 512
        self.plan["plan_id"] = r.digest({k: v for k, v in self.plan.items() if k != "plan_id"})
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        sampler = mock.MagicMock()
        sampler.stop.return_value = {"sampled_gb": []}
        with mock.patch.object(r, "verify_host"), \
             mock.patch.object(r, "MemorySampler", return_value=sampler), \
             mock.patch.object(r.sysinfo, "snapshot", return_value=self.snap), \
             mock.patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, "invalid and preserved"):
                r.run_block(self.root, 1, "clean")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--max-tokens") + 1], "512")

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

    def test_replicated_prompts_match_task_and_preserve_context(self):
        question = "Question: How long has the lighthouse not been lit? Give the duration as a number followed by its unit (for example, \"ten days\"). Reply with only the duration."
        speed = ROOT / "prompts" / "speed"
        for regime in r.REGIMES:
            text = (r.PROMPT_DIR / f"{regime}.txt").read_text()
            original = (speed / f"{regime if regime != 'short' else 'medium'}.txt").read_text().strip()
            if regime == "short":
                original = original.split("\n\n")[0]
            self.assertEqual(text, original + "\n\n" + question + "\n")

    def test_original_prompt_plan_keeps_its_frozen_directory(self):
        for prompt in self.plan["prompts"]:
            prompt["path"] = f"prompts/replicated/{prompt['regime']}.txt"
        self.plan["plan_id"] = r.digest({k: v for k, v in self.plan.items() if k != "plan_id"})
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        r.check_plan(self.plan)
        sampler = mock.MagicMock()
        sampler.stop.return_value = {"sampled_gb": []}
        with mock.patch.object(r, "verify_host"), \
             mock.patch.object(r, "MemorySampler", return_value=sampler), \
             mock.patch.object(r.sysinfo, "snapshot", return_value=self.snap), \
             mock.patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, "invalid and preserved"):
                r.run_block(self.root, 1, "clean")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--prompt-dir") + 1], str(ROOT / "prompts" / "replicated"))

    def test_mixed_prompt_revisions_are_rejected(self):
        self.plan["prompts"][0]["path"] = "prompts/replicated/short.txt"
        self.plan["plan_id"] = r.digest({k: v for k, v in self.plan.items() if k != "plan_id"})
        with self.assertRaisesRegex(ValueError, "prompt path"):
            r.check_plan(self.plan)

    def test_plan_is_content_addressed_and_tamper_detected(self):
        r.check_plan(self.plan)
        broken = dict(self.plan, seed=99)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            r.check_plan(broken)
        broken["plan_id"] = r.digest({k: v for k, v in broken.items() if k != "plan_id"})
        with self.assertRaisesRegex(ValueError, "run order"):
            r.check_plan(broken)
        broken = {**self.plan, "prompts": [dict(p) for p in self.plan["prompts"]]}
        broken["prompts"][0]["path"] = "prompts/speed/short.txt"
        broken["plan_id"] = r.digest({k: v for k, v in broken.items() if k != "plan_id"})
        with self.assertRaisesRegex(ValueError, "wrong replicated prompt path"):
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

    def test_block_host_preflight_requires_ac_and_low_power_mode_off(self):
        model = {"model_id": "model-a"}
        plan = {"machine": {}, "source_commit": "commit", "prompts": [],
                "models": [model], "evidence_dir": str(self.root)}
        with mock.patch.object(r, "machine", return_value={}), \
             mock.patch.object(r, "require_mac"), mock.patch.object(r, "git_head", return_value="commit"), \
             mock.patch.dict(r.registry.MODELS, {"model-a": object()}), \
             mock.patch.object(r, "verified_identity", return_value=model), \
             mock.patch.object(r.sysinfo, "snapshot") as snap:
            snap.return_value = {"power": "ac", "low_power_mode": 0}
            r.verify_host(plan)
            for condition in ({"power": "battery", "low_power_mode": 0},
                              {"power": None, "low_power_mode": 0},
                              {"power": "ac", "low_power_mode": 1},
                              {"power": "ac", "low_power_mode": None}):
                with self.subTest(condition=condition), self.assertRaises(ValueError):
                    snap.return_value = condition
                    r.verify_host(plan)

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

    def test_missing_condition_measurements_are_not_silent(self):
        snap = dict(self.snap, power=None, low_power_mode=None, cpu_speed_limit_pct=None, top_cpu=None)
        warnings = r.observed_caveats(snap, self.snap, [])
        for label in ("power source", "low power mode", "CPU thermal speed limit", "CPU interference"):
            self.assertIn(label + " unavailable", " ".join(warnings))

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
        self.assertIn("median/range mismatch", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        # Preserve, flag and disqualify truncated outputs instead of treating as successful responses.
        t = raw / "short-trial2.json"
        row = r.load_json(t)
        row["finish_reason"] = "length"
        t.write_text(json.dumps(row))
        self.assertIn("truncated", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        (raw / "long-trial5.json").unlink()
        self.assertIn("absent evidence", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_recorded_prompt_path_must_name_frozen_prompt_not_historical_prompt(self):
        p, mid = self.fixture()
        trial = p / "raw" / mid / "mlx-4bit-inproc" / "short-trial1.json"
        row = r.load_json(trial)
        row["prompt_file"] = "/different-checkout/prompts/replicated-duration/short.txt"
        trial.write_text(json.dumps(row))
        self.assertNotIn("prompt file identity", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        row["prompt_file"] = "/different-checkout/prompts/speed/short.txt"
        trial.write_text(json.dumps(row))
        self.assertIn("prompt file identity", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_stop_reason_and_inprocess_answer_are_validated(self):
        p, mid = self.fixture()
        trial = p / "raw" / mid / "mlx-4bit-inproc" / "short-trial1.json"
        original = r.load_json(trial)
        for changes, expected in [
            ({"finish_reason": "cancelled"}, "unknown stop reason"),
            ({"text": ""}, "empty or reasoning-only"),
            ({"text": "<think>unfinished"}, "empty or reasoning-only"),
            ({"text": "<think>reasoning</think>  "}, "empty or reasoning-only"),
        ]:
            with self.subTest(changes=changes):
                trial.write_text(json.dumps({**original, **changes}))
                self.assertIn(expected, " ".join(r.validate_session(p, self.plan, 1, 1, mid)))
        trial.write_text(json.dumps({**original, "text": "<think>reasoning</think> answer"}))
        self.assertNotIn("empty or reasoning-only", " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

    def test_invalid_metric_domains_rejected(self):
        p, mid = self.fixture()
        trial = p / "raw" / mid / "mlx-4bit-inproc" / "short-trial1.json"
        original = r.load_json(trial)
        for changes, expected in [
            ({"prompt_tokens": float("inf")}, "missing/invalid prompt_tokens"),
            ({"generation_tokens": 2.5}, "generation_tokens must be an integer"),
            ({"generation_tokens": 257}, "generation exceeds token cap"),
            ({"request_ttft_s": 5.0}, "TTFT exceeds response-end"),
        ]:
            with self.subTest(changes=changes):
                trial.write_text(json.dumps({**original, **changes}))
                self.assertIn(expected, " ".join(r.validate_session(p, self.plan, 1, 1, mid)))

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
        self.assertIn("Prompt tokens (model tokenizer)", first)
        self.assertIn("Completion tokens median [min–max]", first)
        self.assertIn("five kept trials and their ranges", first)
        self.assertIn("50 [50–50]", first)
        self.assertEqual(first.count("| short |"), 27)  # four models, three contrasts, 20 sessions

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
            with self.assertRaisesRegex(ValueError, "existing session evidence is invalid"):
                r.run_block(self.root, 1, "clean", resume=True)
        self.assertEqual(r.load_json(p / "session.json")["status"], "failed")

    def test_later_block_refuses_missing_previous_sessions_before_inference(self):
        with mock.patch.object(r, "verify_host") as host, mock.patch.object(r.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "prior session"):
                r.run_block(self.root, 2, "clean")
        host.assert_not_called()
        run.assert_not_called()
        self.assertFalse((self.root / "sessions").exists())

    def test_completed_process_with_invalid_evidence_stops_the_block(self):
        sampler = mock.MagicMock()
        sampler.stop.return_value = {"sampled_gb": []}
        with mock.patch.object(r, "verify_host"), \
             mock.patch.object(r, "MemorySampler", return_value=sampler), \
             mock.patch.object(r.sysinfo, "snapshot", return_value=self.snap), \
             mock.patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, "invalid and preserved"):
                r.run_block(self.root, 1, "clean")
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--prompt-dir") + 1], str(r.PROMPT_DIR))
        self.assertEqual(len(list((self.root / "sessions").iterdir())), 1)

    def test_experiment_wide_lock_blocks_other_blocks_and_releases_after_error(self):
        with r.experiment_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "another benchmark block"):
                r.run_block(self.root, 2, "clean")
        # Also release after an exception; the file itself is intentionally persistent.
        with self.assertRaisesRegex(ValueError, "test failure"):
            with r.experiment_lock(self.root):
                raise ValueError("test failure")
        with r.experiment_lock(self.root):
            self.assertTrue((self.root / ".experiment.lock").is_file())

    def test_overlapping_synthetic_sessions_are_rejected_before_reporting(self):
        for block in self.plan["schedule"]:
            for pos in range(1, 5):
                self.fixture(block["block"], pos, overlap=(block["block"] == 1 and pos == 2))
        _, good, errors = r.validate(self.root)
        self.assertEqual(len(good), 20)  # each individual session passes independently
        self.assertIn("overlapping independent sessions", " ".join(errors))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            r.report(self.root)

    def test_reused_session_uuid_across_blocks_is_not_independent(self):
        self.fixture(1, 1)
        self.fixture(1, 2, duplicate_id=True)
        _, good, errors = r.validate(self.root)
        self.assertEqual(len(good), 2)
        self.assertIn("reused session UUID", " ".join(errors))

    def test_manifest_observations_cannot_be_edited_independently(self):
        path, mid = self.fixture()
        manifest = r.load_json(path / "session.json")
        manifest["host_memory"]["peak_gb_est"] = 5.1
        manifest["observed"]["caveats"].clear()
        (path / "session.json").write_text(json.dumps(manifest))
        self.assertIn("hashed session payload", " ".join(
            r.validate_session(path, self.plan, 1, 1, mid)))

    def test_sample_peak_is_verified_even_with_all_manifest_hashes_updated(self):
        path, mid = self.fixture()
        payload = r.load_json(path / "session_payload.json")
        payload["host_memory"]["peak_gb_est"] = 7.2  # original max(sampled_gb) is 8.5
        (path / "session_payload.json").write_text(json.dumps(payload))
        manifest = dict(payload, artifacts_sha256=r.evidence_hashes(path))
        (path / "session.json").write_text(json.dumps(manifest))
        self.assertIn("host-wide memory estimate", " ".join(
            r.validate_session(path, self.plan, 1, 1, mid)))

    def test_busy_trial_derived_warning_cannot_be_suppressed_by_clean_label(self):
        path, mid = self.fixture()
        with mock.patch.object(r.sysinfo, "warnings", side_effect=lambda snap: (
            ["busy: mediaanalysisd 225%"] if snap.get("top_cpu") else [])):
            trial = path / "raw" / mid / "mlx-4bit-inproc" / "medium-trial1.json"
            row = r.load_json(trial)
            row["system"] = {**self.snap, "top_cpu": [{"proc": "mediaanalysisd", "cpu_pct": 225}]}
            trial.write_text(json.dumps(row))
            manifest = r.load_json(path / "session.json")
            manifest["artifacts_sha256"] = r.evidence_hashes(path)
            (path / "session.json").write_text(json.dumps(manifest))
            errors = " ".join(r.validate_session(path, self.plan, 1, 1, mid))
            self.assertIn("observed warning derivation mismatch", errors)
            self.assertIn("summary warning derivation mismatch", errors)
            self.assertEqual(manifest["operator_condition"], "clean")

    def test_checkpoint_freeze_includes_config_tokenizer_and_template(self):
        ckdir = self.root / "model"
        ckdir.mkdir()
        (ckdir / "config.json").write_text('{"model_type":"test"}')
        (ckdir / "tokenizer.json").write_text('{"vocab":{}}')
        (ckdir / "chat_template.jinja").write_text('hello')
        (ckdir / "model.safetensors").write_bytes(b"mock-weights")
        weights = {"model.safetensors": r.sha_file(ckdir / "model.safetensors")}
        provdir = self.root / "evidence" / "model-x" / "p0"
        provdir.mkdir(parents=True)
        (provdir / "mlx-4bit-provenance.json").write_text(json.dumps({
            "revision": "abc123", "files": [{"file": "model.safetensors",
                                                "expected_sha256": weights["model.safetensors"]}]}))
        ckpt = SimpleNamespace(local_path=ckdir, tag="mlx-4bit", hf_repo="vendor/model-x",
                               tier="4bit", publisher="vendor", bits_per_weight=4.5,
                               weights_gb=2.0, official=True)
        cfg = SimpleNamespace(id="model-x", checkpoints={"mlx-4bit": ckpt},
                              template_kwargs={"enable_thinking": False})
        with mock.patch.object(r.gates, "problems", return_value=[]), \
                mock.patch.object(r.gates, "fingerprint", return_value="fp"):
            frozen = r.verified_identity(cfg, self.root / "evidence")
            self.assertEqual(set(frozen["checkpoint_file_sha256"]), {
                "config.json", "tokenizer.json", "chat_template.jinja"})
            for changed in ("config.json", "tokenizer.json", "chat_template.jinja"):
                target = ckdir / changed
                original = target.read_text()
                target.write_text(original + "changed")
                self.assertNotEqual(r.verified_identity(cfg, self.root / "evidence"), frozen)
                target.write_text(original)
            (ckdir / "extra.safetensors").write_bytes(b"surprise")
            with self.assertRaisesRegex(ValueError, "weight file set drift"):
                r.verified_identity(cfg, self.root / "evidence")


if __name__ == "__main__":
    unittest.main()
