"""P4 pilot protocol regression tests; no model, network, or disk writes by tools."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_p4


def scripted(*answers):
    responses = iter(answers)
    def generate(_messages):
        return {"text": next(responses), "wall_s": 0.1, "ttft_s": 0.02,
                "generation_tokens": 5, "peak_gb": 1.5}
    return generate


class AgentPilotTests(unittest.TestCase):
    def test_fixture_has_unique_cases_and_categories(self):
        cases = run_p4.tasks()
        self.assertEqual(len(cases), 20)
        self.assertEqual({t["category"] for t in cases},
                         {"operations", "incident", "data", "coding"})

    def test_read_requires_actual_tool_use(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "read-workers")
        self.assertFalse(run_p4.run_task(case, scripted('{"final":"6"}'))["passed"])
        result = run_p4.run_task(case, scripted(
            '{"tool":"read_file","arguments":{"path":"/config/worker.env"}}',
            '{"final":"6"}',
        ))
        self.assertTrue(result["passed"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["model_wall_s"], 0.2)

    def test_edit_requires_correct_file_and_read(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "fix-timeout")
        before = case["files"]["/config/service.env"]
        after = case["expected_files"]["/config/service.env"]
        bad = run_p4.run_task(case, scripted(
            '{"tool":"write_file","arguments":{"path":"/config/service.env","content":'
            + __import__("json").dumps(after) + '}}', '{"final":"DONE"}'))
        self.assertFalse(bad["passed"])
        good = run_p4.run_task(case, scripted(
            '{"tool":"read_file","arguments":{"path":"/config/service.env"}}',
            '{"tool":"write_file","arguments":{"path":"/config/service.env","content":'
            + __import__("json").dumps(after) + '}}', '{"final":"DONE"}'))
        self.assertTrue(good["passed"])
        self.assertEqual(case["files"]["/config/service.env"], before)

    def test_comparison_requires_both_reads(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "compare-cache")
        responses = ['{"tool":"read_file","arguments":{"path":"' + p + '"}}'
                     for p in case["reads"]]
        self.assertTrue(run_p4.run_task(case, scripted(
            *responses, '{"final":"CACHE_TTL"}'))["passed"])

    def test_search_does_not_read_disk_or_guess(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "search-alert")
        self.assertFalse(run_p4.run_task(case, scripted(
            '{"final":"/logs/db.log"}'))["passed"])
        self.assertTrue(run_p4.run_task(case, scripted(
            '{"tool":"search_files","arguments":{"query":"ALERT"}}',
            '{"final":"/logs/db.log"}'))["passed"])

    def test_sandbox_rejects_non_fixture_paths_and_unknown_tools(self):
        files = {"/safe": "value"}
        for act in (
            {"tool": "read_file", "arguments": {"path": "/etc/passwd"}},
            {"tool": "write_file", "arguments": {"path": "/tmp/x", "content": "x"}},
            {"tool": "run_shell", "arguments": {"command": "rm -rf /"}},
            {"tool": "read_file", "arguments": {"path": "/safe", "extra": True}},
        ):
            with self.assertRaises(ValueError):
                run_p4.tool_call(act, files)
        self.assertEqual(files, {"/safe": "value"})

    def test_invalid_output_and_step_limit_fail_closed(self):
        case = next(t for t in run_p4.tasks() if t["id"] == "read-workers")
        result = run_p4.run_task(case, scripted("not json", "still invalid"), max_steps=2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["tool_errors"], 2)
        self.assertIsNone(result["final"])


if __name__ == "__main__":
    unittest.main()
