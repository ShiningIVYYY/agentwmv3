#!/usr/bin/env python3
"""CPU unit fixtures only; no benchmark/research measurements."""
import math
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.token_nll import selected_token_nll, summarize_fields, teacher_forced_nll

TEXT = "Premise: Door is closed.\nExpected Consequence: Door opens.\nAction: open door"


class TokenNLLTests(unittest.TestCase):
    def test_known_uniform_distribution(self):
        actual = selected_token_nll(torch.zeros(2, 4), [1, 3])
        self.assertTrue(torch.allclose(actual, torch.full((2,), math.log(4))))

    def test_teacher_forced_next_token_shift(self):
        class FakeLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
            def forward(self, input_ids, **kwargs):
                from types import SimpleNamespace
                logits = torch.zeros(1, input_ids.shape[1], 4)
                logits[0, 1, 2] = 5.
                logits[0, 2, 3] = 5.
                return SimpleNamespace(logits=logits)
        actual = teacher_forced_nll(FakeLM(), [0, 1], [2, 3])
        expected = math.log(math.exp(5) + 3) - 5
        self.assertTrue(torch.allclose(actual, torch.full((2,), expected), atol=1e-6))

    def test_confident_wrong_token_has_large_nll(self):
        self.assertGreater(float(selected_token_nll(torch.tensor([[10., -10.]]), [1])[0]), 19)

    def test_invalid_alignment_rejected(self):
        with self.assertRaises(ValueError):
            selected_token_nll(torch.zeros(3, 4), [1])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            selected_token_nll(torch.tensor([[float("nan"), 0.]]), [0])

    def test_local_max_global_all_response(self):
        offsets = [(i, i + 1) for i in range(len(TEXT))]
        values = [1. if i < TEXT.index("Expected Consequence:") else 3. for i in range(len(TEXT))]
        report = summarize_fields(TEXT, offsets, values)
        self.assertEqual(report["premise_uncertainty"], 1.)
        self.assertEqual(report["consequence_uncertainty"], 3.)
        self.assertEqual(report["local_uncertainty"], 3.)
        self.assertAlmostEqual(report["global_uncertainty"], sum(values) / len(values))
        self.assertFalse(report["missing_span"])

    def test_invalid_or_duplicate_labels_not_zero_confidence(self):
        for text in [TEXT + "\nPremise: extra", TEXT.replace("Premise:", "Reason:")]:
            report = summarize_fields(text, [(i, i + 1) for i in range(len(text))], [1.] * len(text))
            self.assertTrue(report["missing_span"])
            self.assertIsNone(report["local_uncertainty"])
            self.assertEqual(report["global_uncertainty"], 1.)

    def test_boundary_token_is_counted_and_reported(self):
        offsets = [(0, 11)] + [(i, i + 1) for i in range(11, len(TEXT))]
        report = summarize_fields(TEXT, offsets, [1.] * len(offsets))
        self.assertIn(0, report["field_spans"][0]["boundary_overhang_tokens"])

    def test_empty_field_missing(self):
        text = TEXT.replace("Door is closed.", " ")
        report = summarize_fields(text, [(i, i + 1) for i in range(len(text))], [1.] * len(text))
        self.assertIsNone(report["local_uncertainty"])

    def test_score_scale_not_clipped_to_probability(self):
        report = summarize_fields(TEXT, [(i, i + 1) for i in range(len(TEXT))], [5.] * len(TEXT))
        self.assertEqual(report["local_uncertainty"], 5.)
        self.assertFalse(report["calibrated_error_probability"])


if __name__ == "__main__":
    unittest.main()
