"""Offline fixture smoke for readiness.py: success and failure paths, no model, no network."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench
import blind
import gates
import readiness
import registry
import run_bench

WEIGHTS = b"fixture weights"
SHA = hashlib.sha256(WEIGHTS).hexdigest()
ENV = {"host": "arm64 macOS 27.0.1 python 3.12.10",
       "versions": {"mlx": "0.32.4", "mlx-lm": "0.32.2", "mlx-optiq": None, "transformers": "5"},
       "optiq": "mlx-optiq not installed"}
SUITE_OK = lambda: (True, "OK")
HISTORICAL = "nemotron-3.5-bolt"  # real id: the baseline whose results must never be overwritten


def model(tmp, mid, tiers=("4bit", "optiq-4bit")):
    cks = {}
    for tier in tiers:
        tag = "mlx-" + tier if tier != "optiq-4bit" else tier
        d = Path(tmp) / "w" / mid / tag
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "m.safetensors").write_bytes(WEIGHTS)
        cks[tag] = registry.Checkpoint(
            tag=tag, hf_repo=f"fx/{mid}-{tag}", local_dir=str(d), weights_gb=1.0, params=2.0,
            quant="q", publisher="vendor", tier=tier,
            engines=("mlx_lm", "mlx-optiq") if tier == "optiq-4bit" else ("mlx_lm",))
    return registry.Model(id=mid, name=mid, vendor="v", arch="a", params_total=2.0, params_active=None,
                          context_train=1, template_kwargs={"enable_thinking": False},
                          template_notes="", checkpoints=cks)


def seed_gate(out, cfg, p0=True, **p0_over):
    """Verified provenance (and a current P0 row) for every checkpoint of cfg."""
    p0_dir = Path(out) / cfg.id / "p0"
    p0_dir.mkdir(parents=True, exist_ok=True)
    for ck in cfg.checkpoints.values():
        prov = {"hf_repo": ck.hf_repo, "checkpoint": ck.tag, "revision": "rev-" + ck.tag,
                "verified": True, "files": [{"file": "m.safetensors", "expected_sha256": SHA,
                                             "local_sha256": SHA, "local_bytes": len(WEIGHTS), "ok": True}]}
        (p0_dir / f"{ck.tag}-provenance.json").write_text(json.dumps(prov))
        if p0:
            row = {"hf_repo": ck.hf_repo, "ok": True, "sanity": True, "deterministic": True,
                   "weights_fingerprint": gates.fingerprint(prov), "template_kwargs": cfg.template_kwargs}
            (p0_dir / f"{ck.tag}.json").write_text(json.dumps(row | p0_over))


def summary(out, mid, run_id, **over):
    d = Path(out) / mid / run_id
    d.mkdir(parents=True, exist_ok=True)
    s = {"outcome": "completed", "publishable": True, "trials": 5, "host": ENV["host"],
         "versions": {"mlx": "0.32.4", "mlx-lm": "0.32.2"}} | over
    (d / "summary.json").write_text(json.dumps(s))


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.out = Path(tmp.name) / "results"
        self.a, self.b = model(self.tmp, "fx-a"), model(self.tmp, HISTORICAL)
        patcher = patch.dict(registry.MODELS, {"fx-a": self.a, HISTORICAL: self.b})
        patcher.start()
        self.addCleanup(patcher.stop)
        for cfg in (self.a, self.b):
            seed_gate(self.out, cfg)

    def run_ev(self, models=("fx-a", HISTORICAL), tiers=("4bit", "optiq-4bit"), trials=5, env=ENV,
               suite=SUITE_OK, round_name="r1"):
        return readiness.evaluate(list(models), self.out, list(tiers), trials, 256,
                                  ["short", "medium"], round_name, env, suite)

    def assertBlocked(self, area, text="", **kw):
        bad, _ = self.run_ev(**kw)
        self.assertTrue(any(b.startswith(f"[{area}]") and text in b for b in bad), bad)

    def test_targeted_phase_checks_external_evidence_and_single_model(self):
        new = Path(self.tmp) / "fresh"
        p3 = new / "fx-a" / "p3"
        p3.mkdir(parents=True)
        (p3 / "mlx-4bit-results.json").write_text("leave untouched")
        bad, lines = readiness.evaluate(["fx-a"], new, ["4bit"], 5, 256, ["short"],
            "r1", ENV, SUITE_OK, evidence=self.out, phase="p2")
        self.assertEqual(bad, [])
        self.assertIn(str(self.out), "\n".join(lines))
        summary(new, "fx-a", "mlx-4bit-inproc")
        bad, _ = readiness.evaluate(["fx-a"], new, ["4bit"], 1, 256, ["short"],
            "r1", ENV, SUITE_OK, evidence=self.out, phase="p3")
        self.assertTrue(any(b.startswith("[collision]") for b in bad))
        new2 = Path(self.tmp) / "p3fresh"
        bad, _ = readiness.evaluate(["fx-a"], new2, ["4bit"], 1, 256, ["short"],
            "r1", ENV, SUITE_OK, evidence=self.out, phase="p3")
        self.assertEqual(bad, [])
        bad, _ = readiness.evaluate(["fx-a"], new2, ["4bit"], 5, 256, ["short"],
            "r1", ENV, SUITE_OK, evidence=new2, phase="p2")
        self.assertTrue(any(b.startswith("[provenance]") for b in bad))

    def test_optiq_readiness_ignores_unused_tiers(self):
        other = self.a.checkpoints["mlx-4bit"]
        (Path(other.local_dir) / "config.json").unlink()
        (self.out / self.a.id / "p0" / "mlx-4bit-provenance.json").unlink()
        bad, lines = readiness.evaluate(
            ["fx-a"], Path(self.tmp) / "fresh-optiq", ["4bit", "optiq-4bit"],
            5, 256, ["short"], "r1", ENV, SUITE_OK,
            evidence=self.out, phase="p2-optiq")
        self.assertEqual(bad, [])
        self.assertIn("1 checkpoint(s) checked", "\n".join(lines))
        self.assertFalse(any("mlx-4bit" in line for line in lines))

    def test_p3_new_tier_reuses_summary_with_own_log(self):
        dest = Path(self.tmp) / "p3-rerun"
        out = dest / "fx-a"
        p3 = out / "p3"
        p3.mkdir(parents=True)
        (p3 / "accuracy_summary.json").write_text(
            json.dumps({"checkpoints": {"mlx-4bit": {"accuracy": 1}}}))
        first_log = run_bench.p3_log_path(self.a, [self.a.checkpoint("mlx-4bit")], dest)
        first_log.parent.mkdir(parents=True)
        first_log.write_text("historical log")
        bad, _ = readiness.evaluate(["fx-a"], dest, ["optiq-4bit"], 5,
            256, ["short"], "r1", ENV, SUITE_OK, evidence=self.out, phase="p3")
        self.assertEqual(bad, [])
        bad, _ = readiness.evaluate(["fx-a"], dest, ["4bit"], 5,
            256, ["short"], "r1", ENV, SUITE_OK, evidence=self.out, phase="p3")
        self.assertTrue(any(b.startswith("[collision]") for b in bad))

    def test_ready_and_checklist(self):
        bad, lines = self.run_ev()
        self.assertEqual(bad, [])
        text = "\n".join(lines)
        for needle in (ENV["host"], "mlx 0.32.4", "mlx-lm 0.32.2", "fx/fx-a-optiq-4bit @ rev-optiq-4bit",
                       "prompts/speed/short.txt  sha256", "max_tokens 256", "enable_thinking",
                       "5 kept per prompt", "fx-a/mlx-4bit-inproc"):
            self.assertIn(needle, text)

    def test_existing_rows_of_other_tiers_do_not_block(self):
        summary(self.out, HISTORICAL, "mlx-4bit-inproc")  # published history for a tier not selected
        bad, _ = self.run_ev(tiers=("optiq-4bit",))
        self.assertEqual(bad, [])

    def test_missing_model_checkpoint_and_download(self):
        self.assertBlocked("models", "not in the registry", models=("fx-a", "nope"))
        self.assertBlocked("models", "no 8bit checkpoint", tiers=("8bit",))
        (Path(self.a.checkpoints["optiq-4bit"].local_dir) / "config.json").unlink()
        self.assertBlocked("models", "not downloaded")

    def test_provenance_and_load_gate(self):
        (self.out / "fx-a/p0/mlx-4bit-provenance.json").unlink()
        self.assertBlocked("provenance", "missing mlx-4bit-provenance.json")
        (self.out / "fx-a/p0/mlx-4bit.json").unlink()
        seed_gate(self.out, self.a, p0=False)
        self.assertBlocked("load gate", "missing mlx-4bit.json")
        seed_gate(self.out, self.a, ok=False, error="oom")
        self.assertBlocked("load gate", "P0 failed: oom")

    def test_stale_gate_evidence(self):
        seed_gate(self.out, self.a, weights_fingerprint="old")
        self.assertBlocked("load gate", "predates the current provenance")

    def test_too_few_trials(self):
        self.assertBlocked("trials", "--trials 3", trials=3)
        summary(self.out, "fx-a", "mlx-4bit-inproc", trials=3, publishable=False)
        self.assertBlocked("trials", "recorded 3 trials")

    def test_inconsistent_versions_and_host(self):
        summary(self.out, "fx-a", "mlx-8bit-inproc", versions={"mlx": "0.99", "mlx-lm": "0.32.2"})
        self.assertBlocked("versions", "mlx differs")
        summary(self.out, "fx-a", "mlx-8bit-inproc", host="arm64 macOS 26 python 3.12.10")
        self.assertBlocked("versions", "host differs")
        env = {**ENV, "versions": {**ENV["versions"], "mlx-lm": None}}
        self.assertBlocked("versions", "mlx-lm is not installed", env=env)

    def test_output_collisions_never_overwrite_baseline(self):
        base = self.out / HISTORICAL / "mlx-4bit-inproc"
        summary(self.out, HISTORICAL, "mlx-4bit-inproc")
        before = base.joinpath("summary.json").read_text()
        self.assertBlocked("collision", f"{HISTORICAL}/mlx-4bit-inproc")
        for extra in (self.out / "blind-r1",):
            extra.mkdir(parents=True, exist_ok=True)
            self.assertBlocked("collision", extra.name)
        self.assertEqual(base.joinpath("summary.json").read_text(), before)  # evaluate wrote nothing

    def test_p3_summary_collides_only_for_published_tags(self):
        p3 = self.out / "fx-a/p3"
        p3.mkdir(parents=True)
        (p3 / "accuracy_summary.json").write_text(json.dumps({"checkpoints": {"mlx-8bit": {}}}))
        bad, _ = self.run_ev(models=("fx-a",))  # planned tiers: mlx-4bit, optiq-4bit
        self.assertFalse([b for b in bad if b.startswith("[collision]")], bad)
        (p3 / "accuracy_summary.json").write_text(json.dumps({"checkpoints": {"optiq-4bit": {}}}))
        self.assertBlocked("collision", "entry for optiq-4bit", models=("fx-a",))

    def test_speed_results_audit(self):
        summary(self.out, "fx-a", "mlx-4bit-http", outcome="partial", publishable=False)
        self.assertBlocked("speed results", "not a completed, publishable result")
        (self.out / "fx-a/mlx-4bit-http/summary.json").write_text("{broken")
        self.assertBlocked("speed results", "mlx-4bit-http")

    def test_optional_package_versions(self):
        summary(self.out, "fx-a", "mlx-8bit-inproc",
                versions={"mlx": "0.32.4", "mlx-lm": "0.32.2", "mlx-optiq": None})  # lacking it is fine
        bad, _ = self.run_ev()
        self.assertEqual(bad, [])
        summary(self.out, "fx-a", "mlx-8bit-inproc", versions={"mlx": "0.32.4", "mlx-lm": "0.32.2",
                                                                "transformers": "4"})
        self.assertBlocked("versions", "transformers differs")
        summary(self.out, "fx-a", "mlx-8bit-inproc", versions={"mlx-optiq": "0.7.3"})
        env = {**ENV, "versions": {**ENV["versions"], "mlx-optiq": "0.7.2"}}
        self.assertBlocked("versions", "mlx-optiq differs", env=env)

    def test_unreadable_prompt_is_not_a_traceback(self):
        with patch.object(readiness, "sha256", side_effect=PermissionError("denied")):
            self.assertBlocked("report inputs", "unreadable")

    def test_scoring_round_completeness(self):
        rd = self.out / "blind-r1"
        rd.mkdir()
        key = {"rows": ["fx-a/mlx-4bit", f"{HISTORICAL}/mlx-4bit"], "expected_prompt_ids": ["q1"],
               "key": {"q1": {"Model A": "x", "Model B": "y"}}}
        (rd / "key.json").write_text(json.dumps(key))
        (rd / "scores.json").write_text(json.dumps({"scores": [{"id": "q1", "Model A": {"accuracy": 5}}]}))
        self.assertBlocked("scoring", "unscored")
        self.assertBlocked("scoring", "at least 2 models", models=("fx-a",))

    def test_report_inputs(self):
        with patch.object(bench, "PROMPTS", self.out):  # no prompt files at all
            self.assertBlocked("report inputs", "speed prompt file missing")
        with patch.object(blind, "expected_ids", side_effect=SystemExit("expected 8 quality prompts")):
            self.assertBlocked("scoring", "expected 8")
        self.a.checkpoints["mlx-4bit"].publisher = ""
        self.assertBlocked("report inputs", "registry lacks publisher")

    def test_unit_suite_failure_and_no_recursion(self):
        self.assertBlocked("tests", "boom", suite=lambda: (False, "boom"))
        with patch.dict("os.environ", {readiness.NESTED: "1"}), \
                patch.object(readiness.subprocess, "run", side_effect=AssertionError("recursed")):
            self.assertFalse(readiness.unit_suite()[0])
            self.assertIn("not a pass", readiness.unit_suite()[1])

    def test_cli_verdict_and_exit_code(self):
        argv = ["--models", "fx-a", HISTORICAL, "--tiers", "4bit", "optiq-4bit", "--out", str(self.out)]
        env = patch.object(readiness, "environment", return_value=ENV)
        suite = patch.object(readiness, "unit_suite", SUITE_OK)

        def run(extra=()):
            buf = io.StringIO()
            with env, suite, contextlib.redirect_stdout(buf):
                rc = readiness.main(argv + list(extra))
            return rc, buf.getvalue().strip().splitlines()

        rc, out = run()
        self.assertEqual(rc, 0)
        self.assertTrue(any(line.startswith("READY") for line in out))
        self.assertTrue(any("independently reviewed" in line for line in out))
        rc, out = run(["--trials", "2"])
        self.assertEqual(rc, 1)
        self.assertTrue(any(line.startswith("NOT READY") for line in out))
        self.assertFalse(any(line.startswith("READY") for line in out))


if __name__ == "__main__":
    unittest.main()
