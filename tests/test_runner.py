"""Offline regression tests for benchmark execution and result records."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench
import registry
import run_bench


class FakeProcess:
    next_pid = 100

    def __init__(self, cmd, stdout=None, stderr=None):
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None
        self.terminated = False
        self.waited = False
        if stdout:
            stdout.write("synthetic server started\n")

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        self.returncode = self.returncode or 0
        return self.returncode

    def kill(self):
        self.returncode = -9


class SyntheticHTTPHandler(BaseHTTPRequestHandler):
    request_body = None
    fail = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        type(self).request_body = json.loads(self.rfile.read(size))
        if type(self).fail:
            self.send_error(500, "synthetic transport failure")
            return

        events = [
            {"choices": [{"delta": {"reasoning": "checking "}}]},
            {"choices": [{"delta": {"content": "fixture answer"}}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        ]
        payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        payload += "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(payload.encode())


def install_fake_mlx(stream_error=False):
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.reset_peak_memory = lambda: None
    mlx_core.get_peak_memory = lambda: 123_000_000

    mlx_package = types.ModuleType("mlx")
    mlx_package.__path__ = []
    mlx_package.core = mlx_core

    mlx_lm = types.ModuleType("mlx_lm")

    def stream_generate(_model, _tokenizer, **_kwargs):
        if stream_error:
            raise RuntimeError("synthetic generation failure")
        yield SimpleNamespace(
            text="fixture answer",
            generation_tokens=3,
            prompt_tokens=7,
            prompt_tps=70.0,
            finish_reason="stop",
        )

    mlx_lm.stream_generate = stream_generate
    mlx_lm.load = lambda *_args, **_kwargs: (
        object(),
        SimpleNamespace(apply_chat_template=lambda *_a, **_k: "formatted prompt"),
    )
    return {"mlx": mlx_package, "mlx.core": mlx_core, "mlx_lm": mlx_lm}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = registry.get_model("nemotron-3.5-bolt")

    def test_stats_reports_median_range_and_successful_sample_count(self):
        self.assertEqual(bench.stats([1.0, None, 3.0]), {
            "median": 2.0, "min": 1.0, "max": 3.0, "n": 2,
        })
        self.assertIsNone(bench.stats([None]))

    def test_inproc_transport_uses_synthetic_stream_and_records_metrics(self):
        modules = install_fake_mlx()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "formatted prompt")
        args = SimpleNamespace(max_tokens=8, temperature=0.0, top_p=None)
        with patch.dict(sys.modules, modules):
            row = bench.run_mlx(object(), tokenizer, "prompt", args, {"enable_thinking": False})
        self.assertEqual(row["text"], "fixture answer")
        self.assertEqual(row["prompt_tokens"], 7)
        self.assertEqual(row["generation_tokens"], 3)
        self.assertEqual(row["peak_gb_source"], "mx_peak")

    def test_http_transport_parses_reasoning_first_sse_fixture(self):
        SyntheticHTTPHandler.fail = False
        SyntheticHTTPHandler.request_body = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = SimpleNamespace(
                base_url=f"http://127.0.0.1:{server.server_port}",
                server_pid=None, max_tokens=8, temperature=0.0, top_p=None,
            )
            row = bench.run_http("prompt", args, {"enable_thinking": False})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(row["text"], "fixture answer")
        self.assertEqual(row["generation_tokens"], 3)
        self.assertEqual(row["reasoning_chars"], len("checking "))
        self.assertFalse(row["reasoning_only"])
        self.assertEqual(
            SyntheticHTTPHandler.request_body["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_inproc_cli_writes_a_complete_success_result_from_fixture(self):
        modules = install_fake_mlx()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            prompts_dir = temp / "prompts"
            (prompts_dir / "speed").mkdir(parents=True)
            prompt = prompts_dir / "speed" / "fixture.txt"
            prompt.write_text("synthetic prompt")
            argv = [
                "bench.py", "--model", self.cfg.id, "--checkpoint", "mlx-4bit",
                "--prompts", prompt.stem, "--trials", "1", "--max-tokens", "8",
                "--out", str(temp),
            ]
            with (
                patch.dict(sys.modules, modules),
                patch.object(
                registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True
            ),
                patch.object(bench, "versions", return_value={"mlx": "fixture"}),
                patch.object(bench, "PROMPTS", prompts_dir),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(bench.main(), 0)

            out_dir = temp / self.cfg.id / "mlx-4bit-inproc"
            trial = json.loads((out_dir / f"{prompt.stem}-warmup.json").read_text())
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertTrue(trial["ok"])
            self.assertEqual(trial["text"], "fixture answer")
            self.assertEqual(trial["max_tokens"], 8)
            self.assertEqual(summary["runs_ok"], 1)
            self.assertEqual(summary["prompts"][prompt.stem]["generation_tokens"]["n"], 1)

    def test_failed_inproc_cli_persists_an_actionable_error_row(self):
        modules = install_fake_mlx(stream_error=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            prompts_dir = temp / "prompts"
            (prompts_dir / "speed").mkdir(parents=True)
            prompt = prompts_dir / "speed" / "failure.txt"
            prompt.write_text("synthetic prompt")
            argv = [
                "bench.py", "--model", self.cfg.id, "--checkpoint", "mlx-4bit",
                "--prompts", prompt.stem, "--trials", "1", "--max-tokens", "8",
                "--out", str(temp),
            ]
            with (
                patch.dict(sys.modules, modules),
                patch.object(
                registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True
            ),
                patch.object(bench, "versions", return_value={"mlx": "fixture"}),
                patch.object(bench, "PROMPTS", prompts_dir),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(bench.main(), 1)

            out_dir = temp / self.cfg.id / "mlx-4bit-inproc"
            trial = json.loads((out_dir / f"{prompt.stem}-warmup.json").read_text())
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertFalse(trial["ok"])
            self.assertIn("synthetic generation failure", trial["error"])
            self.assertEqual(summary["runs_failed"], 1)
            self.assertEqual(summary["runs_ok"], 0)
            self.assertIsNone(summary["prompts"]["failure"]["decode_tps"])

    def run_http_cli_fixture(self, fail):
        SyntheticHTTPHandler.fail = fail
        SyntheticHTTPHandler.request_body = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                prompts_dir = temp / "prompts"
                (prompts_dir / "speed").mkdir(parents=True)
                prompt = prompts_dir / "speed" / "fixture.txt"
                prompt.write_text("synthetic prompt")
                argv = [
                    "bench.py", "--model", self.cfg.id, "--checkpoint", "mlx-4bit",
                    "--transport", "http", "--base-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--prompts", prompt.stem, "--trials", "1", "--max-tokens", "8",
                    "--out", str(temp),
                ]
                with (
                    patch.object(bench, "PROMPTS", prompts_dir),
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    status = bench.main()

                out_dir = temp / self.cfg.id / "mlx-4bit-http"
                trial = json.loads((out_dir / f"{prompt.stem}-warmup.json").read_text())
                summary = json.loads((out_dir / "summary.json").read_text())
                return status, trial, summary
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_cli_persists_successful_result_and_summary(self):
        status, trial, summary = self.run_http_cli_fixture(fail=False)

        self.assertEqual(status, 0)
        self.assertTrue(trial["ok"])
        self.assertEqual(trial["text"], "fixture answer")
        self.assertEqual(trial["prompt_tokens"], 7)
        self.assertEqual(summary["runs_ok"], 1)
        self.assertEqual(summary["runs_failed"], 0)

    def test_failed_http_cli_persists_actionable_error_and_summary(self):
        status, trial, summary = self.run_http_cli_fixture(fail=True)

        self.assertEqual(status, 1)
        self.assertFalse(trial["ok"])
        self.assertIn("synthetic transport failure", trial["error"])
        self.assertEqual(summary["runs_ok"], 0)
        self.assertEqual(summary["runs_failed"], 1)

    def test_p2_orchestration_propagates_http_subprocess_failure(self):
        launched = []
        commands = []

        def fake_run(cmd, log_path, timeout=None):
            commands.append(list(cmd))
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("synthetic transport result\n")
            cache_busted_http = (
                "--no-nonce" not in cmd
                and "--transport" in cmd
                and cmd[cmd.index("--transport") + 1] == "http"
            )
            return 4 if cache_busted_http else 0

        def fake_popen(*args, **kwargs):
            process = FakeProcess(*args, **kwargs)
            launched.append(process)
            return process

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(run_bench, "run", side_effect=fake_run),
            patch.object(run_bench, "free_port", return_value=8765),
            patch.object(run_bench, "wait_for_server", return_value=True),
            patch.object(run_bench, "log_tail"),
            patch.object(run_bench.subprocess, "Popen", side_effect=fake_popen),
        ):
            ok = run_bench.phase_p2(
                self.cfg, Path(temp_dir), trials=5, prompts=["short"], max_tokens=8,
                server_version="mlx-lm fixture", selected=[self.cfg.checkpoint("mlx-4bit")],
            )

        self.assertFalse(ok)
        self.assertEqual(len(commands), 3)
        self.assertTrue(launched[0].terminated)

    def test_p2_orchestration_stops_when_inproc_subprocess_fails(self):
        calls = []

        def fake_run(cmd, log_path, timeout=None):
            calls.append(cmd)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("synthetic in-process failure\n")
            return 1

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(run_bench, "run", side_effect=fake_run),
            patch.object(
                run_bench.subprocess,
                "Popen",
                side_effect=AssertionError("server must not start"),
            ),
        ):
            ok = run_bench.phase_p2(
                self.cfg, Path(temp_dir), trials=1, prompts=["short"], max_tokens=8,
                server_version="mlx-lm fixture", selected=[self.cfg.checkpoint("mlx-4bit")],
            )

        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertIn("--out", calls[0])

    def test_p2_orchestration_runs_inproc_http_and_labelled_cached_http(self):
        launched = []
        commands = []

        def fake_run(cmd, log_path, timeout=None):
            commands.append(list(cmd))
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("synthetic run completed\n")
            return 0

        def fake_popen(*args, **kwargs):
            process = FakeProcess(*args, **kwargs)
            launched.append(process)
            return process

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(run_bench, "run", side_effect=fake_run),
            patch.object(run_bench, "free_port", return_value=8765),
            patch.object(run_bench, "wait_for_server", return_value=True),
            patch.object(run_bench.subprocess, "Popen", side_effect=fake_popen),
        ):
            ok = run_bench.phase_p2(
                self.cfg, Path(temp_dir), trials=5, prompts=["short", "decode-short"],
                max_tokens=16, server_version="mlx-lm fixture",
                selected=[self.cfg.checkpoint("mlx-4bit")],
            )

        self.assertTrue(ok)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0][commands[0].index("--transport") + 1], "inproc")
        self.assertEqual(commands[1][commands[1].index("--transport") + 1], "http")
        self.assertNotIn("--no-nonce", commands[1])
        self.assertIn("--no-nonce", commands[2])
        self.assertEqual(commands[2][commands[2].index("--tag") + 1], "cached")
        self.assertEqual(commands[2][commands[2].index("--prompts") + 1:], ["short"])
        self.assertTrue(launched[0].terminated)
        self.assertTrue(launched[0].waited)


if __name__ == "__main__":
    unittest.main()
