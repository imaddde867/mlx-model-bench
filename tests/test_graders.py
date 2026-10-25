"""Offline, deterministic tests for the P3 objective graders (issue #4).

Every checker has positive, negative and adversarial cases. Responses are raw
strings; nothing here loads a model or touches the network.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import p3_suite as S


class Table(unittest.TestCase):
    def check(self, fn, passing, failing, *args):
        for t in passing:
            with self.subTest(expect="pass", text=t):
                self.assertTrue(fn(t, *args))
        for t in failing:
            with self.subTest(expect="fail", text=t):
                self.assertFalse(fn(t, *args))


class TestInt(Table):
    def test_int_match(self):
        self.check(S.check_int_match,
                   ["112", " 112\n"],
                   ["", "113", "112.0", "112.", "-112"],
                   112)
        self.check(S.check_int_match, ["32,768,000", "32768000"], ["32,76,8000", "32 768 000"], 32768000)
        self.check(S.check_int_match, ["-4"], ["4"], -4)

    def test_correct_number_quoted_but_other_final_answer(self):
        self.check(S.check_int_match, [], [
            "The answer is 113, not 112.",
            "112 is wrong, the answer is 113",
            "112\n113",
            "112 113",
            "113 (not 112)",
            "Step 1: 224 + 6 = 230 ... Answer: 112",
            "112 pixels",
            "**112**",
            "1112",
            "112,5",
            "32000 * 256 * 4",
        ], 112)

    def test_float_text_is_not_an_int(self):
        self.check(S.check_int_match, [], ["3.5", "4.0"], 4)


class TestFloat(Table):
    def test_float_match(self):
        self.check(S.check_float_match,
                   ["0.015", " 0.0150 ", ".015", "0.01501", "0.01499"],
                   ["", "0.0125", "0.017", "0.015.", "0.016"],
                   0.015)

    def test_tolerance_boundary(self):
        self.assertTrue(S.check_float_match("0.8001", 0.8))
        self.assertFalse(S.check_float_match("0.8002", 0.8))
        self.assertTrue(S.check_float_match("0.8002", 0.8, tolerance=1e-3))

    def test_adversarial(self):
        self.check(S.check_float_match, [], [
            "0.75 or 0.8",
            "0.8 0.6",
            "Precision is 0.6667, not 0.8",
            "80%",
            "4/5",
            "8e-1",
            "nan",
            "inf",
            "0.8 (80%)",
            "3.5 GB",
        ], 0.8)
        self.check(S.check_float_match, ["3.5", "3,5".replace(",", ".")], ["3,5", "35"], 3.5)

    def test_integer_text_accepted_for_integral_target(self):
        self.assertTrue(S.check_float_match("2", 2.0))


def call(**over):
    base = {"name": "f", "arguments": {"path": "/a", "n": 20, "lr": 0.001, "ks": ["x"]}}
    base.update(over)
    return json.dumps(base)


TARGET = {"name": "f", "arguments": {"path": "/a", "n": 20, "lr": 0.001, "ks": ["x"]}}


class TestToolCall(Table):
    def test_positive(self):
        self.check(S.check_json_tool_match, [
            call(),
            " " + call() + "\n",
            '{"arguments": {"ks": ["x"], "lr": 1e-3, "n": 20, "path": "/a"}, "name": "f"}',
            json.dumps({"name": "f", "arguments": {"path": "/a", "n": 20, "lr": 0.00100004, "ks": ["x"]}}),
        ], [], TARGET)

    def test_structure(self):
        a = TARGET["arguments"]
        self.check(S.check_json_tool_match, [], [
            "",
            "not json",
            "[]",
            "null",
            json.dumps([TARGET]),
            json.dumps({"name": "f"}),
            json.dumps({"arguments": a}),
            json.dumps({"name": "g", "arguments": a}),
            json.dumps({"name": "f", "arguments": json.dumps(a)}),
            json.dumps({"name": "f", "arguments": [a]}),
            json.dumps({"name": "f", "arguments": a, "extra": 1}),
            json.dumps({"function": {"name": "f", "arguments": a}}),
        ], TARGET)

    def test_required_and_extra_arguments(self):
        a = TARGET["arguments"]
        for k in a:
            with self.subTest(missing=k):
                self.assertFalse(S.check_json_tool_match(
                    json.dumps({"name": "f", "arguments": {x: v for x, v in a.items() if x != k}}), TARGET))
        self.assertFalse(S.check_json_tool_match(
            json.dumps({"name": "f", "arguments": {**a, "invented": True}}), TARGET))
        self.assertFalse(S.check_json_tool_match(
            json.dumps({"name": "f", "arguments": {**a, "n": 20, "N": 20}}), TARGET))

    def test_wrong_values(self):
        a = TARGET["arguments"]
        for k, bad in [("path", "/b"), ("n", 21), ("lr", 0.01), ("ks", ["y"]), ("ks", []), ("ks", ["x", "x"])]:
            with self.subTest(k=k, bad=bad):
                self.assertFalse(S.check_json_tool_match(
                    json.dumps({"name": "f", "arguments": {**a, k: bad}}), TARGET))

    def test_no_type_coercion(self):
        a = TARGET["arguments"]
        for k, bad in [
            ("n", "20"),        # str for int
            ("n", 20.0),        # float for int
            ("n", True),        # bool is not an int
            ("lr", "0.001"),    # str for float
            ("lr", True),
            ("lr", [0.001]),
            ("path", 1),        # number for str
            ("path", " /a"),    # no whitespace stripping on strings
            ("path", ["/a"]),
            ("ks", "x"),        # str for list
            ("ks", [1]),
        ]:
            with self.subTest(k=k, bad=bad):
                self.assertFalse(S.check_json_tool_match(
                    json.dumps({"name": "f", "arguments": {**a, k: bad}}), TARGET))
        flag = {"name": "f", "arguments": {"on": True, "k": 1}}
        self.assertTrue(S.check_json_tool_match(json.dumps(flag), flag))
        self.assertFalse(S.check_json_tool_match('{"name":"f","arguments":{"on":1,"k":1}}', flag))
        self.assertFalse(S.check_json_tool_match('{"name":"f","arguments":{"on":true,"k":true}}', flag))

    def test_adversarial_wrapping_and_json_tricks(self):
        self.check(S.check_json_tool_match, [], [
            "```json\n" + call() + "\n```",          # prompt forbids code fences
            "Sure! " + call(),
            call() + "\nThis calls f.",
            call() + call(),
            '{"name": "f", "name": "g", "arguments": %s}' % json.dumps(TARGET["arguments"]),
            '{"name": "g", "name": "f", "arguments": %s}' % json.dumps(TARGET["arguments"]),
            '{"name": "f", "arguments": {"path": "/a", "n": NaN, "lr": 0.001, "ks": ["x"]}}',
            '{"name": "f", "arguments": {"path": "/a", "n": 20, "lr": 0.001, "ks": ["x"],}}',
            "{'name': 'f'}",
        ], TARGET)


class TestTwoSentences(Table):
    def test_two_sentences_no_lists(self):
        self.check(S.check_two_sentences_no_lists, [
            "It caps gradient norm. This stops exploding updates.",
            "Does it help? Yes, it stabilises training!",
            "It limits updates, e.g. by norm. That keeps training stable.",
            "First sentence.\n\nSecond sentence.",
            'It says "stop." Then it stops.',
            "Version 1.5 clips values. It helps.",
        ], [
            "",
            "One sentence only.",
            "One. Two. Three.",
            "- It caps norm.\n- It stops blowups.",
            "1. It caps norm.\n2) It stops blowups.",
            "It caps norm. This stops blowups.\n* extra",
            "• It caps norm. It helps.",
            "First sentence.\n- Second sentence.",
            "No terminator here\n\nand another paragraph\n\nand a third",
        ])


class TestNoE(Table):
    def test_no_letter_e(self):
        self.check(S.check_no_letter_e, [
            "A columnar data pack.",
            "Parquit is a columnar block, packing data.",
            "Ok",
        ], [
            "",
            "   ",
            "Parquet is a columnar format.",
            "A columnar data pack. E",
            "COLUMNAR STORE",
            "café",
            "E",
            "ｅ",
            "e",
            "A columnar data pack.\nEnd",
        ])


class TestQKV(Table):
    def test_qkv_single_line(self):
        self.check(S.check_qkv_single_line, [
            "Query,Key,Value",
            "Query, Key, Value",
            "value, KEY, query",
            "  query , key , value \n",
        ], [
            "",
            "Query, Key",
            "Query, Key, Value, Output",
            "Query, Key, Value.",
            "Query\nKey\nValue",
            "Query, Key,\nValue",
            "Query vector, Key vector, Value vector",
            "Query, Key, and Value",
            "Query, Key, Query",
            "Query, Key, Value,",
            "Q, K, V",
            "The names are: Query, Key, Value",
            "(Query, Key, Value)",
        ])


def words(n, joiner=" "):
    return joiner.join(f"w{i}" for i in range(n))


class TestWordCount(Table):
    def test_word_count_25_35(self):
        self.check(S.check_word_count_25_35, [
            words(25), words(30), words(35),
            words(24) + " well-known",       # hyphenated compound is one word
            words(34) + " ... done.",         # bare punctuation tokens are not words
            words(25, "\n"),
        ], [
            "", words(1), words(24), words(36),
            words(24) + " - - -",
            words(35) + " extra",
        ])

    def test_contractions_and_hyphens_are_single_words(self):
        self.assertTrue(S.check_word_count_25_35(words(23) + " don't well-known"))
        self.assertFalse(S.check_word_count_25_35(words(22) + " don't well-known"))
        self.assertTrue(S.check_word_count_25_35(words(24) + " don't"))
        self.assertFalse(S.check_word_count_25_35(words(36).replace(" w1 ", " w1-w2 ", 1)))


class TestFiveHyphenLines(Table):
    GOOD = "- MSE\n- Cross-Entropy\n- Hinge\n- Huber\n- KL Divergence"

    def test_five_hyphen_lines(self):
        self.check(S.check_five_hyphen_lines, [
            self.GOOD,
            self.GOOD + "\n",
            "\n" + self.GOOD,
            self.GOOD.replace("\n", "\n\n"),
        ], [
            "",
            "- a\n- b\n- c\n- d",
            self.GOOD + "\n- Extra",
            "Here are five:\n" + self.GOOD,
            self.GOOD + "\nThese are common.",
            "* MSE\n* Hinge\n* Huber\n* KL\n* CE",
            "-MSE\n-Hinge\n-Huber\n-KL\n-CE",
            "  - MSE\n- Hinge\n- Huber\n- KL\n- CE",
            "- MSE\n- Hinge\n- Huber\n- KL\n-",
            "- MSE\n- Hinge\n- Huber\n- KL\n-  ",
            "- MSE\n- MSE\n- mse\n- Huber\n- KL",
            "- MSE, Hinge, Huber, KL, CE",
            "1. MSE\n2. Hinge\n3. Huber\n4. KL\n5. CE",
        ])


class TestNoRealtimeLatency(Table):
    def test_no_realtime_latency(self):
        self.check(S.check_no_realtime_latency, [
            "Batch processes data in chunks. Streaming handles events continuously.",
            "Streaming is continuous, batch is periodic.",
        ], [
            "",
            "   ",
            "Streaming gives real-time results.",
            "Streaming gives Real-Time results.",
            "Streaming gives REAL TIME results.",
            "Streaming is realtime.",
            "Streaming is real‑time.",   # non-breaking hyphen
            "Streaming is real–time.",   # en dash
            "Streaming is real—time.",   # em dash
            "Streaming is real‐time.",
            "Streaming has low latency.",
            "Streaming has low-LATENCY output.",
            "Streaming has Latency.",
            "Batch has high latency. Streaming is fast.",
            "ｒｅａｌ-time",    # full-width letters fold under NFKC
        ])


class TestAllCaps(Table):
    def test_all_caps(self):
        self.check(S.check_all_caps, [
            "DROPOUT RANDOMLY ZEROES ACTIVATIONS, PREVENTING CO-ADAPTATION.",
            "DROPOUT\nZEROES 50% OF UNITS",
            "CO_ADAPTATION",
        ], [
            "",
            "123 !!!",
            "Dropout randomly zeroes activations.",
            "dropout",
            "DROPOUT ZEROES ACTIVATIONS. Thus",
            "DROPOUT ZEROES x ACTIVATIONS",
            "STRAßE",
            "DROPOUT\nzeroes",
        ])


class TestOneSentence(Table):
    def test_one_sentence_no_gradient(self):
        self.check(S.check_one_sentence_no_gradient, [
            "Backpropagation adjusts weights by passing the error backward through the network.",
            "Backpropagation adjusts weights, e.g. by passing error backward, to reduce loss",
            "Version 2.0 adjusts weights by passing error backward.",
        ], [
            "",
            "Backpropagation adjusts weights. It reduces loss.",
            "Backpropagation uses the gradient to adjust weights.",
            "Backpropagation uses Gradients to adjust weights.",
            "Backpropagation uses derivatives to adjust weights.",
            "Backpropagation uses a DERIVATIVE to adjust weights.",
            "Backpropagation adjusts weights\n\nIt reduces loss",
            "Backpropagation adjusts weights!? It reduces loss.",
            "...",
        ])


class TestSentenceCounter(unittest.TestCase):
    def test_counts(self):
        for text, n in [
            ("", 0), ("...", 0), ("A.", 1), ("A. B.", 2), ("A! B? C.", 3),
            ("Use e.g. Adam. Done.", 2), ("Pi is 3.14 today.", 1), ('He said "go." Then left.', 2),
        ]:
            with self.subTest(text=text):
                self.assertEqual(S.count_sentences(text), n)


class TestRegistry(unittest.TestCase):
    def test_every_prompt_checker_exists_and_target_passes_itself(self):
        for p in S.MATH_PROMPTS + S.TOOL_PROMPTS + S.INSTR_PROMPTS:
            with self.subTest(id=p["id"]):
                self.assertIn(p["checker"], S.CHECKERS)
        for p in S.MATH_PROMPTS:
            self.assertTrue(S.CHECKERS[p["checker"]](str(p["target"]), p["target"]), p["id"])
        for p in S.TOOL_PROMPTS:
            self.assertTrue(S.check_json_tool_match(json.dumps(p["target"]), p["target"]), p["id"])

    def test_numeric_prompts_say_output_only(self):
        for p in S.MATH_PROMPTS:
            self.assertRegex(p["prompt"], r"Output only", p["id"])

    def test_recorded_legacy_results_do_not_drift(self):
        # Original P3 artifacts (no grader_version) are never rewritten; the
        # hardened graders must reproduce their recorded verdicts from raw responses.
        files = [f for f in (ROOT / "results").glob("*/p3/*-results.json")]
        for f in files:
            data = json.loads(f.read_text())
            if "grader_version" in data:
                continue
            for r in data["accuracy_runs"]:
                chk = S.CHECKERS[r["checker"]]
                new = chk(r["response"], r["target"]) if r["target"] is not None else chk(r["response"])
                with self.subTest(file=f.name, id=r["id"]):
                    self.assertEqual(new, r["passed"])


if __name__ == "__main__":
    unittest.main()
