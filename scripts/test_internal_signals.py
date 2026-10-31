#!/usr/bin/env python3
"""Unit tests. Fixtures do not constitute Agent or uncertainty evaluation data."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.provider import BACKEND, TokenNLLProvider
from internal_signals.decision import messages, action_from_text, bounded

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Premise: The door is closed.\nExpected Consequence: The door opens.\nAction: open door"


class ProviderContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "checkpoints/qwen3-4b", local_files_only=True)
        cls.config = json.loads((ROOT / "configs/internal_signals/scienceworld_token_nll_v1.json").read_text())

    def record(self, text=TEXT):
        msg = [{"role": "user", "content": "Unit input"}]
        prompt = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(prompt + text, add_special_tokens=False)["input_ids"]
        self.assertEqual(full[:len(ids)], ids)
        response = full[len(ids):]
        return {"prompt_text": prompt, "prompt_ids": ids, "response_ids": response,
                "response": text, "token_nll": [1.] * len(response),
                "generation_tokens": len(response) + 1, "truncated": False,
                "reduction_seconds": 0., "backend": BACKEND, "dtype": "bfloat16",
                "model_revision": self.config["model_revision"]}

    def test_real_tokenizer_field_scoring(self):
        result = TokenNLLProvider(self.tokenizer, self.config).extract(self.record())
        self.assertEqual(result.local_uncertainty, 1.)
        self.assertEqual(result.global_uncertainty, 1.)
        self.assertFalse(result.missing_span)

    def test_parse_failure_uses_explicit_global_fallback(self):
        result = TokenNLLProvider(self.tokenizer, self.config).extract(self.record("Action: look around"))
        self.assertIsNone(result.local_uncertainty)
        self.assertTrue(result.missing_span)
        self.assertEqual(result.details["fallback_uncertainty"], 1.)

    def test_parse_failure_can_use_null_fallback(self):
        cfg = {**self.config, "missing_policy": "null"}
        result = TokenNLLProvider(self.tokenizer, cfg).extract(self.record("Action: look around"))
        self.assertIsNone(result.details["effective_uncertainty"])

    def test_alignment_failure_never_supplies_global_fallback(self):
        r = self.record()
        r["response"] += "changed"
        result = TokenNLLProvider(self.tokenizer, self.config).extract(r)
        self.assertIsNone(result.global_uncertainty)
        self.assertIsNone(result.details["effective_uncertainty"])
        self.assertTrue(result.errors[0].startswith("alignment_or_numeric_error"))

    def test_truncation_cannot_look_valid(self):
        r = self.record()
        r["truncated"] = True
        result = TokenNLLProvider(self.tokenizer, self.config).extract(r)
        self.assertTrue(result.missing_span)
        self.assertIsNone(result.global_uncertainty)

    def test_provider_precision_and_revision_mixing_rejected(self):
        for key, value in [("dtype", "float32"), ("backend", "hf_teacher_forced"),
                           ("model_revision", "changed")]:
            r = self.record()
            r[key] = value
            with self.assertRaises(ValueError):
                TokenNLLProvider(self.tokenizer, self.config).extract(r)

    def test_special_response_token_never_scores(self):
        record = self.record()
        record["response_ids"] = [self.tokenizer.eos_token_id]
        record["response"] = self.tokenizer.decode(record["response_ids"], skip_special_tokens=False)
        record["token_nll"] = [1.]
        result = TokenNLLProvider(self.tokenizer, self.config).extract(record)
        self.assertTrue(result.missing_span)
        self.assertIsNone(result.global_uncertainty)

    def test_legacy_actor_default_does_not_load_internal_backend(self):
        import tempfile
        from scripts.run_audited_paper_eval import Agent
        with tempfile.TemporaryDirectory() as folder:
            actor = Agent(SimpleNamespace(disable_thinking=False), Path(folder))
        self.assertIsNone(actor.internal_actor)

    def test_unknown_fallback_rejected(self):
        with self.assertRaises(ValueError):
            TokenNLLProvider(self.tokenizer, {**self.config, "missing_policy": "zero"})

    def test_empty_and_nonfinite_tokens_missing_not_zero(self):
        for values in [[], [float("nan")]]:
            r = self.record()
            r["token_nll"] = values
            result = TokenNLLProvider(self.tokenizer, self.config).extract(r)
            self.assertTrue(result.missing_span)
            self.assertIsNone(result.local_uncertainty)
            self.assertIsNone(result.global_uncertainty)


class DecisionBoundary(unittest.TestCase):
    def test_prompt_only_observable_past_not_future_metadata(self):
        steps = [{"observation": "old", "action": "wait", "next_observation": "FUTURE_SENTINEL",
                  "score": 100, "private_state": "SECRET"}]
        text = messages("goal", "current", steps, 2)[0]["content"]
        self.assertNotIn("FUTURE_SENTINEL", text)
        self.assertNotIn("SECRET", text)
        self.assertIn("old", text)
        self.assertIn("current", text)

    def test_recent_history_count(self):
        steps = [{"observation": "obs_" + str(i), "action": "wait"} for i in range(3)]
        text = messages("goal", "current", steps, 2)[0]["content"]
        self.assertNotIn("obs_0", text)
        self.assertIn("obs_1", text)
        self.assertIn("obs_2", text)

    def test_native_action_tail_and_number_preserved(self):
        for action in ["open door", "0", "connect battery to wire", "look around"]:
            text = TEXT.rsplit("Action:", 1)[0] + "Action: " + action
            self.assertEqual(action_from_text(text), action)
            self.assertEqual(action_from_text(text), text.rsplit("Action:", 1)[-1].strip())

    def test_required_input_overflow_raises(self):
        actor = SimpleNamespace(config={"history_length": 2, "max_prompt_tokens": 1})
        agent = SimpleNamespace(internal_actor=actor, count=lambda msg: 999)
        with self.assertRaises(ValueError):
            bounded(agent, "goal", "current", [])


if __name__ == "__main__":
    unittest.main()
