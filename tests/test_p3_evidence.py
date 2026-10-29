"""Capability evidence fields; synthetic streams, no model inference."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import registry
import run_p3
import run_p4
import test_runner


class CapabilityEvidenceTests(unittest.TestCase):
    def test_run_single_keeps_stop_reason_and_observes_system_before_timing(self):
        for reason in ("stop", "length", None):
            with self.subTest(reason=reason):
                clock = [0.0]
                def snapshot():
                    clock[0] = 5.0
                    return {"power": "ac"}
                def stream(*_args, **_kwargs):
                    clock[0] = 5.2
                    if reason is not None:
                        yield SimpleNamespace(text="answer", generation_tokens=1,
                                              prompt_tokens=7, finish_reason=reason)
                modules = test_runner.install_fake_mlx()
                modules["mlx_lm"].stream_generate = stream
                tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "prompt")
                with mock.patch.dict(sys.modules, modules), \
                     mock.patch.object(run_p3.sysinfo, "snapshot", side_effect=snapshot), \
                     mock.patch.object(run_p3.time, "perf_counter", side_effect=lambda: clock[0]):
                    row = run_p3.run_single(object(), tokenizer, "question", {}, 8)
                self.assertEqual(row["finish_reason"], reason)
                self.assertEqual(row["system"], {"power": "ac"})
                self.assertEqual(row["wall_s"], 0.2)
                self.assertEqual(row["raw_text"], "answer" if reason else "")

    def test_p3_persists_stop_and_conditions_for_accuracy_and_quality(self):
        modules = test_runner.install_fake_mlx()
        modules["mlx.core"].clear_cache = lambda: None
        cfg = registry.MODELS["minicpm5-2b"]
        gen = {"text": "answer", "raw_text": "<think>notes</think>answer",
               "prompt_tokens": 7, "generation_tokens": 8, "wall_s": 0.2,
               "ttft_s": 0.1, "decode_tps": 70.0, "peak_gb": 0.123,
               "finish_reason": "length", "system": {"power": "ac"}}
        accuracy = {"id": "fixture", "category": "format", "prompt": "question", "checker": "fixture"}
        quality = {"id": "quality", "task": "rewrite", "prompt": "question", "rubric": "clear"}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(sys.modules, modules), \
             mock.patch.object(run_p3, "run_single", return_value=gen), \
             mock.patch.object(run_p3, "MATH_PROMPTS", [accuracy]), \
             mock.patch.object(run_p3, "TOOL_PROMPTS", []), \
             mock.patch.object(run_p3, "INSTR_PROMPTS", []), \
             mock.patch.object(run_p3, "QUALITY_PROMPTS", [quality]), \
             mock.patch.dict(run_p3.CHECKERS, {"fixture": lambda _text: True}), \
             mock.patch.object(run_p3.time, "sleep"):
            result = run_p3.evaluate_checkpoint(cfg, cfg.checkpoint("mlx-4bit"), Path(tmp))
        self.assertIn("mlx-lm", result["versions"])
        self.assertIn("transformers", result["versions"])
        for row in result["accuracy_runs"] + result["quality_runs"]:
            self.assertEqual(row["finish_reason"], "length")
            self.assertEqual(row["system"], {"power": "ac"})
            self.assertEqual(row["raw_response"], gen["raw_text"])

    def test_p4_preserves_step_evidence_and_measures_tool_overhead_separately(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "read-workers")
        responses = iter(['{"tool":"read_file","arguments":{"path":"/config/worker.env"}}',
                          '{"final":"6"}'])
        clock = [10.0]
        def generate(_messages):
            clock[0] += 0.15
            return {"text": next(responses), "wall_s": 0.1,
                    "finish_reason": "stop", "system": {"power": "ac"}}
        real_tool = run_p4.tool_call
        def tool(action, files):
            clock[0] += 0.3
            return real_tool(action, files)
        with mock.patch.object(run_p4.time, "perf_counter", side_effect=lambda: clock[0]), \
             mock.patch.object(run_p4, "tool_call", side_effect=tool), \
             mock.patch.object(run_p4.sysinfo, "snapshot", return_value={}):
            result = run_p4.run_task(case, generate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["model_wall_s"], 0.2)
        self.assertEqual(result["task_wall_s"], 0.6)
        for step in result["steps"]:
            self.assertEqual(step["finish_reason"], "stop")
            self.assertEqual(step["system"], {"power": "ac"})


if __name__ == "__main__":
    unittest.main()
