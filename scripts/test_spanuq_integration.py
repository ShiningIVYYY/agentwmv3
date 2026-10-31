#!/usr/bin/env python3
"""CPU contract tests; mocks/negative fixtures are never experimental scores."""
import json
import os
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.spanuq_adapter import (
    CompatibilityError, SpanUQAdapter, compare_state,
    official_classes, translate_released_config)
from internal_signals.hidden_states import response_alignment, extract_hidden_states

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("SPANUQ_SOURCE", str(ROOT / "external/SpanUQ"))
BACKBONE = os.environ.get("QWEN3_BACKBONE", str(ROOT / "checkpoints/qwen3-4b"))


class StrictContracts(unittest.TestCase):
    def test_unexpected_architecture_keys_are_reported(self):
        report = compare_state({"a": torch.zeros(1)}, {"a": torch.zeros(1),
                               "u_conditioner.cond_mlp.0.weight": torch.zeros(2, 3)})
        self.assertEqual(report["unexpected_keys"], ["u_conditioner.cond_mlp.0.weight"])

    def test_missing_head_is_reported(self):
        self.assertEqual(compare_state({"head": torch.zeros(2)}, {})["missing_keys"], ["head"])

    def test_pe_shape_mismatch_is_reported(self):
        report = compare_state({"pe": torch.zeros(1, 4, 2)}, {"pe": torch.zeros(1, 16, 2)})
        self.assertEqual(len(report["shape_mismatches"]), 1)

    def test_explicit_legacy_fields(self):
        cls, _ = official_classes(SOURCE)
        raw = json.loads((ROOT / "docs/audit/DamonDemon__SpanUQ__Qwen3-4B__model_config.json").read_text())
        config, cap, changes = translate_released_config(raw, cls)
        self.assertEqual(config.mob_n_components, 3)
        self.assertEqual(cap, 2048)
        self.assertEqual(config.seq_aggregation, "importance")
        self.assertIn("mob_k", changes)

    def test_unknown_config_rejected(self):
        cls, _ = official_classes(SOURCE)
        raw = json.loads((ROOT / "docs/audit/DamonDemon__SpanUQ__Qwen3-4B__model_config.json").read_text())
        with self.assertRaisesRegex(CompatibilityError, "Unknown"):
            translate_released_config({**raw, "invented_field": True}, cls)

    def test_real_checkpoint_refuses_scoring(self):
        with self.assertRaises(CompatibilityError) as caught:
            SpanUQAdapter(SOURCE, ROOT / "checkpoints/spanuq/Qwen3-4B")
        self.assertFalse(caught.exception.report["strict_load_passed"])
        self.assertEqual(len(caught.exception.report["unexpected_keys"]), 4)
        self.assertIn("u_conditioner.cond_mlp.0.weight", caught.exception.report["unexpected_keys"])


@unittest.skipUnless(Path(BACKBONE, "tokenizer.json").exists(), "Downloaded real tokenizer required")
class RealTokenizerAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)

    def prompt(self):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain melting."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text, self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def test_full_response_offsets(self):
        prompt, ids = self.prompt()
        response = "Ice melts at zero degrees Celsius."
        full = self.tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
        self.assertEqual(full[:len(ids)], ids)
        text, offsets = response_alignment(self.tokenizer, prompt, ids, full[len(ids):])
        self.assertEqual(text, response)
        self.assertEqual(len(offsets), len(full) - len(ids))
        self.assertEqual(offsets[0][0], 0)
        self.assertEqual(offsets[-1][1], len(response))

    def test_empty_response_rejected(self):
        prompt, ids = self.prompt()
        with self.assertRaisesRegex(ValueError, "No response"):
            response_alignment(self.tokenizer, prompt, ids, [])

    def test_prompt_overflow_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceeds extraction limit"):
            extract_hidden_states(None, self.tokenizer, [{"role": "user", "content": "test"}], [42], max_total_tokens=1)

    def test_special_response_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "Special response"):
            extract_hidden_states(None, self.tokenizer, [{"role": "user", "content": "test"}], [self.tokenizer.eos_token_id])

    def test_changed_prompt_rejected(self):
        prompt, ids = self.prompt()
        response_ids = self.tokenizer("Ice melts.", add_special_tokens=False)["input_ids"]
        with self.assertRaises(ValueError):
            response_alignment(self.tokenizer, prompt + "modified", ids, response_ids)


if __name__ == "__main__":
    unittest.main()
