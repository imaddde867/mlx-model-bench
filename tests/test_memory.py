"""Memory provenance: mixed sources, absent sampling, same-basis comparison (issue #7)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bench
import test_runner


class MemoryTests(unittest.TestCase):
    def test_inproc_row_records_allocator_provenance_beside_peak_gb(self):
        modules = test_runner.install_fake_mlx()
        tok = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "p")
        args = SimpleNamespace(max_tokens=8, temperature=0.0, top_p=None)
        with patch.dict(sys.modules, modules):
            row = bench.run_mlx(object(), tok, "x", args, {})
        m = row["memory"]
        self.assertEqual(row["peak_gb_source"], "mx_peak")
        self.assertEqual((m["source"], m["value_gb"], m["available"]), ("mx_peak", row["peak_gb"], True))
        self.assertIn("decimal", m["unit"])
        self.assertTrue(m["method"] and m["scope"] and m["phase"])

    def test_absent_sampling_is_an_explicit_unavailable_state(self):
        m = bench.memory_record(None, None, "no --server-pid")
        self.assertFalse(m["available"])
        self.assertIsNone(m["source"])
        self.assertEqual(m["unavailable_reason"], "no --server-pid")

    def test_mixed_sources_are_never_subtracted(self):
        a = {"transport": "inproc", "memory": {"source": "mx_peak"}, "peak_gb": 4.0}
        b = {"transport": "http", "memory": {"source": "server_rss"}, "peak_gb": 5.0}
        self.assertIsNone(bench.memory_delta(a, b))

    def test_same_basis_comparison_is_valid(self):
        a = {"memory": {"source": "server_rss"}, "peak_gb": 4.0}
        b = {"memory": {"source": "server_rss"}, "peak_gb": 5.5}
        self.assertEqual(bench.memory_delta(a, b), 1.5)
        self.assertIsNone(bench.memory_delta(a, {"memory": {"source": "server_rss"}, "peak_gb": None}))

    def test_labels_come_from_recorded_provenance_and_legacy_is_marked_inferred(self):
        self.assertEqual(bench.memory_label({"memory": {"source": "server_rss"}, "transport": "inproc"}),
                         "server RSS")
        self.assertEqual(bench.memory_label({"transport": "inproc"}), "MLX allocator peak (inferred)")
        self.assertEqual(bench.memory_label({"memory": {"source": None}, "transport": "x"}), "memory n/a")

    def test_new_format_summary_without_source_is_never_inferred(self):
        mixed = {"transport": "http", "peak_gb": 4.0, "memory": {"source": None, "sources_seen": ["mx_peak", "server_rss"]}}
        none = {"transport": "http", "peak_gb": 4.0, "memory": {"source": None, "sources_seen": []}}
        rss = {"memory": {"source": "server_rss"}, "peak_gb": 5.0}
        self.assertEqual(bench.memory_label(mixed), "mixed sources")
        self.assertEqual(bench.memory_label(none), "memory n/a")
        self.assertIsNone(bench.memory_delta(mixed, rss))
        self.assertIsNone(bench.memory_delta(rss, none))

    def test_summary_memory_aggregates_recorded_sources(self):
        t = test_runner.RunnerTests("test_complete_five_trial_run_is_publishable")
        t.setUp()
        mx = bench.memory_record("mx_peak", 4.0)
        rss = bench.memory_record("server_rss", 5.0)
        gone = bench.memory_record(None, None, "no --server-pid")
        _, one = t.run_inproc_cli(2, [{**t.GOOD, "memory": mx}] * 3)
        self.assertEqual((one["memory"]["source"], one["memory"]["unavailable_rows"]), ("mx_peak", 0))
        self.assertEqual(one["memory"]["method"], "mx.get_peak_memory()")
        _, mixed = t.run_inproc_cli(2, [{**t.GOOD, "memory": mx}, {**t.GOOD, "memory": mx}, {**t.GOOD, "memory": rss}])
        self.assertIsNone(mixed["memory"]["source"])
        self.assertEqual(mixed["memory"]["sources_seen"], ["mx_peak", "server_rss"])
        _, absent = t.run_inproc_cli(2, [{**t.GOOD, "memory": gone}] * 3)
        self.assertEqual((absent["memory"]["source"], absent["memory"]["unavailable_rows"]), (None, 2))
        self.assertEqual(bench.memory_label(mixed), "mixed sources")

if __name__ == "__main__":
    unittest.main()
