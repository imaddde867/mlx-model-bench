"""Offline tests: provenance verification, the P0 verdict, and gate enforcement."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench
import gates
import registry
import run_bench

CFG = registry.get_model("nemotron-3.5-bolt")
WEIGHTS = b"fixture weights"
SHA = hashlib.sha256(WEIGHTS).hexdigest()


class Env:
    """A temp results dir plus a checkpoint whose weights live in the same temp dir."""

    def __init__(self, tmp, tag="mlx-4bit"):
        self.out = Path(tmp) / "results"
        self.p0 = self.out / CFG.id / "p0"
        self.p0.mkdir(parents=True)
        real = CFG.checkpoint(tag)
        self.ckpt = SimpleNamespace(tag=tag, hf_repo=real.hf_repo, local_dir="w",
                                    local_path=Path(tmp) / "w")
        self.ckpt.local_path.mkdir()
        (self.ckpt.local_path / "m.safetensors").write_bytes(WEIGHTS)

    def provenance(self, **over):
        rec = {"hf_repo": self.ckpt.hf_repo, "checkpoint": self.ckpt.tag, "revision": "abc123",
               "verified": True, "files": [{
                   "file": "m.safetensors", "expected_sha256": SHA, "local_sha256": SHA,
                   "local_bytes": len(WEIGHTS), "ok": True}]}
        rec.update(over)
        (self.p0 / f"{self.ckpt.tag}-provenance.json").write_text(json.dumps(rec))
        return rec

    def p0_row(self, prov, **over):
        row = {"hf_repo": self.ckpt.hf_repo, "ok": True, "sanity": True, "deterministic": True,
               "weights_fingerprint": gates.fingerprint(prov),
               "template_kwargs": CFG.template_kwargs}
        row.update(over)
        (self.p0 / f"{self.ckpt.tag}.json").write_text(json.dumps(row))

    def problems(self, **kw):
        return gates.problems(CFG, self.ckpt, self.out, **kw)


class GateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.env = Env(tmp.name)

    def good(self):
        prov = self.env.provenance()
        self.env.p0_row(prov)
        return prov

    def test_good_evidence_passes(self):
        self.good()
        self.assertEqual(self.env.problems(), [])

    def test_missing_provenance_blocks(self):
        self.assertIn("missing", self.env.problems()[0])

    def test_missing_p0_blocks(self):
        self.env.provenance()
        self.assertIn("run --phase p0", self.env.problems()[0])

    def test_no_revision_blocks(self):
        prov = self.env.provenance(revision=None)
        self.env.p0_row(prov)
        self.assertTrue(any("revision" in p for p in self.env.problems()))

    def test_unverified_or_size_only_file_blocks(self):
        prov = self.env.provenance(verified=False)
        prov["files"][0]["expected_sha256"] = None
        (self.env.p0 / "mlx-4bit-provenance.json").write_text(json.dumps(prov))
        self.env.p0_row(prov)
        self.assertTrue(any("sha256" in p for p in self.env.problems()))

    def test_file_changed_after_verification_blocks(self):
        self.good()
        (self.env.ckpt.local_path / "m.safetensors").write_bytes(WEIGHTS + b"x")
        self.assertTrue(any("differs" in p for p in self.env.problems()))

    def test_wrong_answer_blocks(self):
        prov = self.env.provenance()
        self.env.p0_row(prov, ok=False, sanity=False, error="sanity answer wrong or empty")
        self.assertTrue(any("sanity" in p for p in self.env.problems()))

    def test_nondeterminism_blocks(self):
        prov = self.env.provenance()
        self.env.p0_row(prov, ok=False, deterministic=False)
        self.assertTrue(any("reproducible" in p for p in self.env.problems()))

    def test_p0_older_than_provenance_is_stale(self):
        prov = self.good()
        prov["files"][0]["expected_sha256"] = "0" * 64
        prov["files"][0]["local_sha256"] = "0" * 64
        (self.env.p0 / "mlx-4bit-provenance.json").write_text(json.dumps(prov))
        self.assertTrue(any("predates" in p for p in self.env.problems()))

    def test_different_template_controls_block(self):
        prov = self.env.provenance()
        self.env.p0_row(prov, template_kwargs={"enable_thinking": True, "x": 1})
        self.assertTrue(any("template" in p for p in self.env.problems()))

    def test_provenance_only_gate_for_p0_phase(self):
        self.env.provenance()
        self.assertEqual(self.env.problems(need_p0=False), [])


class VerifyCheckpointTests(unittest.TestCase):
    def verify(self, manifest, content=WEIGHTS):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "m.safetensors").write_bytes(content)
            ckpt = SimpleNamespace(hf_repo="r/x", tag="t", local_path=path)
            with patch.object(run_bench, "hub_manifest", return_value=("rev1", manifest)):
                return run_bench.verify_checkpoint(ckpt)

    def test_matching_hash_verifies_and_records_revision(self):
        rec = self.verify({"m.safetensors": {"size": len(WEIGHTS), "sha256": SHA}})
        self.assertTrue(rec["verified"])
        self.assertEqual(rec["revision"], "rev1")

    def test_bad_hash_fails(self):
        rec = self.verify({"m.safetensors": {"size": len(WEIGHTS), "sha256": "0" * 64}})
        self.assertFalse(rec["verified"])
        self.assertEqual(rec["files"][0]["error"], "sha256 mismatch")

    def test_size_only_match_is_not_verification(self):
        rec = self.verify({"m.safetensors": {"size": len(WEIGHTS), "sha256": None}})
        self.assertFalse(rec["verified"])

    def test_missing_revision_is_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "m.safetensors").write_bytes(WEIGHTS)
            ckpt = SimpleNamespace(hf_repo="r/x", tag="t", local_path=Path(tmp))
            with patch.object(run_bench, "hub_manifest", return_value=(
                    None, {"m.safetensors": {"size": len(WEIGHTS), "sha256": SHA}})):
                self.assertFalse(run_bench.verify_checkpoint(ckpt)["verified"])


def load_p0(generate):
    """Import p0_load against a fake mlx stack whose generate() is `generate`."""
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.get_peak_memory = lambda: 1
    mlx = types.ModuleType("mlx")
    mlx.__path__, mlx.core = [], mlx_core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = generate
    tok = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "p", encode=lambda t: t.split())
    mlx_lm.load = lambda *_a, **_k: (object(), tok)
    return {"mlx": mlx, "mlx.core": mlx_core, "mlx_lm": mlx_lm}


def run_p0(generate):
    sys.modules.pop("p0_load", None)
    buf = io.StringIO()
    with patch.dict(sys.modules, load_p0(generate)), \
            patch.object(sys, "argv", ["p0_load.py", "m"]), contextlib.redirect_stdout(buf):
        import p0_load
        p0_load.main()
    sys.modules.pop("p0_load", None)
    return json.loads(buf.getvalue())


class P0VerdictTests(unittest.TestCase):
    def test_correct_and_reproducible_passes(self):
        row = run_p0(lambda *_a, **_k: "17 x 23 = 391.")
        self.assertTrue(row["ok"] and row["sanity"] and row["deterministic"])

    def test_wrong_answer_fails(self):
        row = run_p0(lambda *_a, **_k: "It is 400.")
        self.assertFalse(row["ok"])
        self.assertIn("sanity", row["error"])

    def test_empty_answer_fails(self):
        self.assertFalse(run_p0(lambda *_a, **_k: "")["ok"])

    def test_nondeterminism_fails(self):
        outs = iter(["391 first", "391 second"])
        row = run_p0(lambda *_a, **_k: next(outs))
        self.assertFalse(row["ok"])
        self.assertIn("differs", row["error"])

    def test_truncated_thinking_answer_fails(self):
        row = run_p0(lambda *_a, **_k: "<think>17 times 23, first 17 x 20")
        self.assertFalse(row["ok"])


class P0BudgetTests(unittest.TestCase):
    def cmd_for(self, cfg):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sel = [cfg.checkpoint()]
        with patch.object(run_bench, "undownloaded", return_value=[]), \
                patch.object(run_bench.gates, "refuse", return_value=False), \
                patch.object(run_bench, "run", return_value=1) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run_bench.phase_p0(cfg, Path(tmp.name), sel)
        return run.call_args[0][0]

    def test_per_model_budget_reaches_subprocess(self):
        cmd = self.cmd_for(registry.MODELS["lfm2.5-8b-a1b"])
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "512")

    def test_default_budget_is_32(self):
        cmd = self.cmd_for(CFG)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "32")


class EnforcementTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        available = patch.object(registry.Checkpoint, "available",
                                 new_callable=PropertyMock, return_value=True)
        available.start()
        self.addCleanup(available.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_phases_refuse_without_evidence(self):
        sel = [CFG.checkpoint("mlx-4bit")]
        with patch.object(run_bench, "run", side_effect=AssertionError("must not launch")):
            self.assertFalse(run_bench.phase_p2(CFG, self.tmp, 5, ["short"], 8, "v", sel))
            self.assertFalse(run_bench.phase_p3(CFG, self.tmp, sel))
            self.assertFalse(run_bench.phase_p0(CFG, self.tmp, sel))

    def test_direct_bench_refuses_without_evidence(self):
        argv = ["bench.py", "--model", CFG.id, "--checkpoint", "mlx-4bit", "--out", str(self.tmp)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.main(), 1)
        self.assertFalse((self.tmp / CFG.id / "mlx-4bit-inproc").exists())

    def test_unavailable_selected_tier_is_reported_not_skipped(self):
        with patch.object(registry.Checkpoint, "available",
                          new_callable=PropertyMock, return_value=False):
            sel = [CFG.checkpoint("mlx-8bit")]
            self.assertTrue(run_bench.undownloaded(CFG, sel))
            self.assertFalse(run_bench.phase_p2(CFG, self.tmp, 5, ["short"], 8, "v", sel))

    def test_fetch_honors_tier_selection(self):
        sel = [CFG.checkpoint("mlx-4bit")]
        rec = {"verified": True, "files": [], "error": None}
        with patch.object(run_bench, "verify_checkpoint", return_value=rec):
            self.assertTrue(run_bench.phase_fetch(CFG, self.tmp, sel))
        written = sorted(p.name for p in (self.tmp / CFG.id / "p0").iterdir())
        self.assertEqual(written, ["mlx-4bit-provenance.json"])


if __name__ == "__main__":
    unittest.main()
