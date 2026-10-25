"""Synthetic event-timeline tests for TTFT / prefill / decode boundaries (issue #6)."""

import contextlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bench
import gates
import registry
import test_runner


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def tok(text, gen, finish=None):
    return SimpleNamespace(text=text, generation_tokens=gen, prompt_tokens=7,
                           prompt_tps=70.0, finish_reason=finish)


def run_inproc(timeline, prep=0.05):
    """timeline: [(t, response)]; the chat template takes `prep` seconds."""
    clock = Clock()
    seen = {}

    def template(messages, **kwargs):
        seen["messages"], seen["kwargs"] = messages, kwargs
        clock.t += prep
        return "formatted prompt"

    def stream_generate(*_a, **_k):
        for t, resp in timeline:
            clock.t = t
            yield resp

    modules = test_runner.install_fake_mlx()
    modules["mlx_lm"].stream_generate = stream_generate
    tokenizer = SimpleNamespace(apply_chat_template=template)
    args = SimpleNamespace(max_tokens=64, temperature=0.0, top_p=None)
    with patch.dict(sys.modules, modules), patch.object(bench, "time", SimpleNamespace(perf_counter=clock)):
        row = bench.run_mlx(object(), tokenizer, "prompt", args, {"enable_thinking": False})
    return row, seen


def run_http(timeline, no_nonce=False):
    """timeline: [(t, chunk dict or "[DONE]")]; the request is sent at t=0."""
    clock = Clock()
    seen = {}

    def lines():
        for t, ev in timeline:
            clock.t = t
            yield f"data: {ev if isinstance(ev, str) else json.dumps(ev)}\n".encode()

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return contextlib.nullcontext(lines())

    args = SimpleNamespace(base_url="http://fixture", server_pid=None, max_tokens=64,
                           temperature=0.0, top_p=None, no_nonce=no_nonce)
    with (
        patch.object(bench, "time", SimpleNamespace(perf_counter=clock)),
        patch.object(bench.urllib.request, "urlopen", urlopen),
    ):
        row = bench.run_http("prompt", args, {"enable_thinking": False})
    return row, seen


def delta(**d):
    return {"choices": [{"delta": d}]}


def finish(reason, **d):
    return {"choices": [{"delta": d, "finish_reason": reason}]}


def usage(prompt, completion, cached=None):
    u = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [], "usage": u}


def visible(n, start=0.2, step=0.02, key="content"):
    """n visible single-token deltas, one every `step` seconds from `start`."""
    return [(start + step * k, delta(**{key: f"t{k} "})) for k in range(n)]


class TimingTests(unittest.TestCase):
    def test_inproc_splits_template_prep_from_engine_ttft(self):
        # first token at 0.30, then 40 more tokens 20 ms apart; the last is EOS
        timeline = [(0.30 + 0.02 * k, tok(f"t{k} ", k + 1)) for k in range(40)]
        timeline.append((0.30 + 0.02 * 40, tok("", 41, "stop")))
        row, _ = run_inproc(timeline)

        self.assertEqual(row["ttft_s"], 0.30)          # historical: before template -> first token
        self.assertEqual(row["request_ttft_s"], 0.30)  # same start boundary in-process
        self.assertEqual(row["prep_s"], 0.05)
        self.assertEqual(row["engine_ttft_s"], 0.25)
        self.assertEqual(row["cached_prompt_tokens"], 0)
        self.assertEqual((row["decode_tokens"], row["decode_s"]), (40, 0.8))
        self.assertEqual(row["decode_window_tps"], 50.0)
        self.assertEqual(row["decode_tps"], 50.0)
        self.assertEqual(row["decode_flags"], [])

    def test_http_stop_stream_keeps_historical_decode_tps_and_adds_window(self):
        # 40 visible tokens, then the EOS token arrives as an empty finish chunk
        timeline = visible(40) + [(1.0, finish("stop", content="")), (1.0, usage(7, 41)), (1.0, "[DONE]")]
        row, _ = run_http(timeline)

        self.assertEqual(row["ttft_s"], 0.2)
        self.assertEqual(row["request_ttft_s"], 0.2)
        self.assertEqual(row["first_token_kind"], "content")
        self.assertEqual(row["first_content_s"], 0.2)
        self.assertEqual(row["generation_tokens"], 41)
        self.assertEqual(row["decode_tps"], 51.3)         # 40 tokens / 0.78 s: historical, unchanged
        self.assertEqual(row["decode_window_tps"], 50.0)  # 40 tokens / 0.80 s: ends at the EOS chunk
        self.assertIsNone(row["prep_s"])
        self.assertIsNone(row["engine_ttft_s"])

    def test_http_reasoning_first_stream_times_first_token_and_first_content(self):
        timeline = (visible(10, key="reasoning") + visible(31, start=0.5)
                    + [(1.2, finish("stop")), (1.2, usage(7, 42)), (1.2, "[DONE]")])
        row, _ = run_http(timeline)

        self.assertEqual(row["first_token_kind"], "reasoning")
        self.assertEqual(row["request_ttft_s"], 0.2)
        self.assertEqual(row["first_content_s"], 0.5)
        self.assertEqual(row["generation_tokens"], 42)
        self.assertEqual(row["decode_s"], 1.0)
        self.assertEqual(row["decode_window_tps"], 41.0)
        self.assertGreater(row["reasoning_chars"], 0)
        self.assertTrue(row["text"].startswith("t0 "))

    def test_http_role_only_and_empty_deltas_are_not_first_tokens(self):
        timeline = ([(0.05, delta(role="assistant", content="")), (0.1, delta(reasoning=""))]
                    + visible(40) + [(1.0, finish("stop")), (1.0, usage(7, 41)), (1.0, "[DONE]")])
        row, _ = run_http(timeline)
        self.assertEqual(row["request_ttft_s"], 0.2)

    def test_http_usage_only_stream_keeps_token_count_but_fails_the_trial(self):
        row, _ = run_http([(0.4, finish("stop")), (0.4, usage(7, 1)), (0.4, "[DONE]")])

        self.assertIsNone(row["ttft_s"])
        self.assertIsNone(row["request_ttft_s"])
        self.assertEqual(row["generation_tokens"], 1)
        self.assertIn("no_token_event", row["decode_flags"])
        self.assertIsNone(row["decode_window_tps"])
        missing = [m for m in bench.REQUIRED_METRICS if row.get(m) is None]
        self.assertIn("ttft_s", missing)  # main() turns this into a failed trial

    def test_http_stream_cut_before_done_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            run_http(visible(40) + [(1.0, usage(7, 41))])

    def test_http_max_tokens_finish_is_a_valid_unflagged_row(self):
        timeline = (visible(63) + [(0.2 + 0.02 * 63, finish("length", content="last")),
                                   (1.5, usage(7, 64)), (1.5, "[DONE]")])
        row, _ = run_http(timeline)

        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["generation_tokens"], 64)
        self.assertEqual(row["decode_flags"], [])
        self.assertEqual(row["decode_window_tps"], 50.0)

    def test_too_short_completion_is_flagged_not_ranked(self):
        timeline = visible(4) + [(0.28, finish("stop")), (0.28, usage(7, 5)), (0.28, "[DONE]")]
        row, _ = run_http(timeline)

        self.assertEqual(row["decode_flags"], ["too_short"])
        self.assertIsNone(row["decode_window_tps"])
        self.assertIsNotNone(row["decode_tps"])  # historical key still reports it

    def test_prefix_cache_hit_flags_a_nonce_row_only(self):
        timeline = visible(40) + [(1.0, finish("stop")), (1.0, usage(7, 41, cached=4)), (1.0, "[DONE]")]
        busted, _ = run_http(timeline)
        cached, _ = run_http(timeline, no_nonce=True)

        self.assertEqual(busted["cached_prompt_tokens"], 4)
        self.assertEqual(busted["decode_flags"], ["prefix_cache_hit"])
        self.assertEqual(cached["decode_flags"], [])

    def test_both_transports_send_identical_messages_and_template_controls(self):
        _, inproc = run_inproc([(0.3, tok("x", 1, "stop"))])
        _, http = run_http(visible(1) + [(0.3, finish("stop")), (0.3, usage(7, 1)), (0.3, "[DONE]")])

        self.assertEqual(inproc["messages"], http["body"]["messages"])
        self.assertEqual(inproc["kwargs"]["enable_thinking"],
                         http["body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertNotEqual(bench.TTFT_SCOPE["inproc"], bench.TTFT_SCOPE["http"])


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = registry.get_model("nemotron-3.5-bolt")
        patcher = patch.object(gates, "refuse", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_summary_labels_scope_and_excludes_flagged_decode_rows(self):
        good = {**test_runner.RunnerTests.GOOD, "decode_window_tps": 40.0, "decode_flags": []}
        short = {**test_runner.RunnerTests.GOOD, "decode_window_tps": None, "decode_flags": ["too_short"]}
        _, summary = test_runner.RunnerTests.run_inproc_cli(self, 3, [good, good, good, short])

        prompt = summary["prompts"]["fixture"]
        self.assertEqual(summary["ttft_scope"], bench.TTFT_SCOPE["inproc"])
        self.assertEqual(summary["decode_min_tokens"], bench.MIN_DECODE_TOKENS)
        self.assertEqual(prompt["decode_flagged"], 1)
        self.assertEqual(prompt["decode_window_tps"]["n"], 2)
        self.assertEqual(prompt["decode_tps"]["n"], 3)  # historical stats unchanged


if __name__ == "__main__":
    unittest.main()
