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
import gates
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

    def poll(self):
        return self.returncode

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
    truncate = False

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
        if not type(self).truncate:
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
        # Gate behaviour is covered in test_gates.py; these tests exercise the runner.
        patcher = patch.object(gates, "refuse", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

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
            patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
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
            patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
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
            patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
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


    def run_inproc_cli(self, trials, results):
        """bench.main() in-process with run_mlx replaced by `results` (a row dict or an exception per call)."""
        results = iter(results)

        def fake_run_mlx(*_a, **_k):
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return dict(r)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            prompts_dir = temp / "prompts"
            (prompts_dir / "speed").mkdir(parents=True)
            (prompts_dir / "speed" / "fixture.txt").write_text("synthetic prompt")
            argv = [
                "bench.py", "--model", self.cfg.id, "--checkpoint", "mlx-4bit",
                "--prompts", "fixture", "--trials", str(trials), "--max-tokens", "8",
                "--out", str(temp),
            ]
            with (
                patch.dict(sys.modules, install_fake_mlx()),
                patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
                patch.object(bench, "versions", return_value={"mlx": "fixture"}),
                patch.object(bench, "PROMPTS", prompts_dir),
                patch.object(bench, "run_mlx", side_effect=fake_run_mlx),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = bench.main()
            out_dir = temp / self.cfg.id / "mlx-4bit-inproc"
            return status, json.loads((out_dir / "summary.json").read_text())

    GOOD = {"prompt_tokens": 7, "generation_tokens": 3, "ttft_s": 0.1, "prefill_tps": 70.0,
            "wall_s": 0.5, "decode_tps": 30.0}

    def test_complete_five_trial_run_is_publishable(self):
        status, summary = self.run_inproc_cli(5, [self.GOOD] * 6)

        self.assertEqual(status, 0)
        self.assertEqual(summary["outcome"], "completed")
        self.assertTrue(summary["publishable"])
        self.assertEqual(summary["non_publishable_reasons"], [])
        self.assertEqual(summary["prompts"]["fixture"]["kept_ok"], 5)

    def test_smoke_run_succeeds_but_is_not_publishable(self):
        status, summary = self.run_inproc_cli(1, [self.GOOD] * 2)

        self.assertEqual(status, 0)
        self.assertEqual(summary["outcome"], "completed")
        self.assertFalse(summary["publishable"])
        self.assertIn("1 trials < 5", summary["non_publishable_reasons"][0])

    def test_failed_kept_trial_makes_the_arm_partial_and_exit_nonzero(self):
        status, summary = self.run_inproc_cli(
            3, [self.GOOD, self.GOOD, RuntimeError("synthetic trial failure")])

        self.assertEqual(status, 1)
        self.assertEqual(summary["outcome"], "partial")
        self.assertFalse(summary["publishable"])
        self.assertEqual(summary["prompts"]["fixture"]["kept_ok"], 1)
        self.assertTrue(any("1/3 kept trials" in r for r in summary["non_publishable_reasons"]))

    def test_missing_required_metric_fails_the_trial(self):
        incomplete = {k: v for k, v in self.GOOD.items() if k != "ttft_s"}
        status, summary = self.run_inproc_cli(1, [self.GOOD, incomplete])

        self.assertEqual(status, 1)
        self.assertEqual(summary["outcome"], "partial")
        self.assertEqual(summary["runs_failed"], 1)

    def test_failed_warmup_leaves_nothing_measured(self):
        status, summary = self.run_inproc_cli(2, [RuntimeError("synthetic warm-up failure")])

        self.assertEqual(status, 1)
        self.assertEqual(summary["outcome"], "failed")
        self.assertFalse(summary["prompts"]["fixture"]["warmup_ok"])

    def test_http_stream_without_done_marker_is_an_error(self):
        SyntheticHTTPHandler.fail = False
        SyntheticHTTPHandler.truncate = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = SimpleNamespace(
                base_url=f"http://127.0.0.1:{server.server_port}",
                server_pid=None, max_tokens=8, temperature=0.0, top_p=None,
            )
            with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
                bench.run_http("prompt", args, {})
        finally:
            SyntheticHTTPHandler.truncate = False
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def phase_optiq(self, out, exe=None, run_rc=0, summary=None, ready=True):
        ckpt = self.cfg.checkpoint("optiq-4bit")
        run_dir = Path(out) / self.cfg.id / "optiq-4bit-http-optiq-engine"

        def fake_run(cmd, log_path, timeout=None):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("synthetic bench log\n")
            if summary:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(json.dumps(summary))
            return run_rc

        with (
            patch.object(registry.Checkpoint, "available", new_callable=PropertyMock, return_value=True),
            patch.object(run_bench, "optiq_exe", return_value=exe),
            patch.object(run_bench, "optiq_version", return_value="mlx-optiq fixture"),
            patch.object(run_bench, "run", side_effect=fake_run),
            patch.object(run_bench, "free_port", return_value=8765),
            patch.object(run_bench, "wait_for_server", return_value=ready),
            patch.object(run_bench, "log_tail"),
            patch.object(run_bench.subprocess, "Popen", side_effect=FakeProcess),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            outcomes = run_bench.phase_p2_optiq(self.cfg, Path(out), 5, ["short"], 8, [ckpt])
        return outcomes, run_dir

    def test_missing_optiq_runtime_is_recorded_as_unmeasurable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outcomes, run_dir = self.phase_optiq(temp_dir, exe=None)
            record = json.loads((run_dir / "unmeasurable.json").read_text())

        self.assertEqual(outcomes, {"optiq-4bit": "unmeasurable"})
        self.assertEqual(record["outcome"], "unmeasurable")
        self.assertIn("not installed", record["reason"])

    def test_optiq_server_that_never_starts_is_unmeasurable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outcomes, _ = self.phase_optiq(temp_dir, exe=Path("optiq"), ready=False)

        self.assertEqual(outcomes, {"optiq-4bit": "unmeasurable"})

    def test_optiq_bench_outcomes_are_distinguished(self):
        cases = [(0, None, "completed"), (1, {"outcome": "partial"}, "partial"),
                 (1, {"outcome": "failed"}, "unmeasurable"), (1, None, "failed")]
        for rc, summary, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp_dir:
                outcomes, _ = self.phase_optiq(temp_dir, exe=Path("optiq"), run_rc=rc, summary=summary)
                self.assertEqual(outcomes, {"optiq-4bit": expected})

    def test_wait_for_server_gives_up_when_the_server_process_exited(self):
        proc = SimpleNamespace(poll=lambda: 3)
        self.assertFalse(run_bench.wait_for_server("http://127.0.0.1:9", timeout=60, proc=proc))

    def test_run_reports_a_timeout_as_a_failing_exit_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rc = run_bench.run([sys.executable, "-c", "import time; time.sleep(30)"],
                               Path(temp_dir) / "log.txt", timeout=0.2)
        self.assertEqual(rc, 124)


if __name__ == "__main__":
    unittest.main()
