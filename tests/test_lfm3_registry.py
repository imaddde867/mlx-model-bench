"""Offline registration checks for the separately evaluated LFM3 challenger."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import registry


class Lfm3RegistryTests(unittest.TestCase):
    def test_lfm3_is_explicitly_selected_without_expanding_frozen_roster(self):
        cfg = registry.get_model("lfm3-1.7b")
        self.assertEqual(registry.ORDER, ["nemotron-3.5-bolt", "minicpm5-2b",
                                          "spark-x2.5-4b", "lfm2.5-8b-a1b"])
        self.assertNotIn(cfg, registry.list_models())
        self.assertEqual(set(cfg.checkpoints), {"mlx-4bit"})
        ckpt = cfg.checkpoint("mlx-4bit")
        for alias in (cfg.id, ckpt.hf_repo, ckpt.local_dir, str(ckpt.local_path)):
            with self.subTest(alias=alias):
                self.assertIs(registry.get_model(alias), cfg)

    def test_lfm3_records_verified_vendor_metadata_and_explicit_template_controls(self):
        cfg = registry.get_model("lfm3-1.7b")
        ckpt = cfg.checkpoint("mlx-4bit")
        self.assertEqual(cfg.template_kwargs, {"enable_thinking": False,
                                               "preserve_thinking": False})
        self.assertEqual(cfg.params_total, 1.728801536)
        self.assertEqual(cfg.params_active, cfg.params_total)
        self.assertEqual(cfg.context_train, 262144)
        self.assertEqual(ckpt.weights_gb, 973083386 / 1e9)
        self.assertEqual(ckpt.params, cfg.params_total)
        self.assertEqual(ckpt.bits_per_weight, 4.5)
        self.assertEqual(ckpt.publisher, "vendor")
        self.assertTrue(ckpt.official)
        self.assertEqual(ckpt.engines, ("mlx_lm",))
        self.assertEqual(ckpt.tier, "4bit")
        self.assertGreaterEqual(cfg.p0_max_tokens, 128)
        self.assertEqual(cfg.p3_budget_x, 1)
        self.assertIn("9027ac5b062175fba890c612c3905d5d8d5b5294", cfg.notes)
        self.assertIn("d0ca73dcfa70ce18733754a492cdc1fb0123262d", ckpt.notes)


if __name__ == "__main__":
    unittest.main()
