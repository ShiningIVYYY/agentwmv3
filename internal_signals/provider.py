"""Official Phase 2 TokenNLLProvider. No implicit provider or dtype fallback."""
import math
import time

from .base import InternalSignals
from .hidden_states import response_alignment
from .token_nll import summarize_fields

BACKEND = "hf_generation_raw_bf16_v1"
QWEN35_BACKEND = "hf_qwen35_generation_raw_bf16_v1"


class TokenNLLProvider:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config
        self.backend = config["backend"]
        if self.backend not in (BACKEND, QWEN35_BACKEND) or config["dtype"] != "bfloat16":
            raise ValueError("Current provider accepts only its pinned raw-generation BF16 backend")
        if config["missing_policy"] not in ("global_only", "null"):
            raise ValueError("Unknown missing-field fallback")

    def extract(self, record):
        started = time.perf_counter()
        if record["backend"] != self.backend or record["dtype"] != self.config["dtype"]:
            raise ValueError("Generation provider/dtype mismatch; scores cannot be mixed")
        if record["model_revision"] != self.config["model_revision"]:
            raise ValueError("Generation model revision mismatch")
        values = list(record["token_nll"])
        global_value = None
        errors = []
        details = {}
        try:
            if len(values) != len(record["response_ids"]) or not values:
                raise ValueError("Token NLL length/empty response")
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Non-finite or negative token NLL")
            text, offsets = response_alignment(
                self.tokenizer, record["prompt_text"], record["prompt_ids"], record["response_ids"],
                left_truncated_tokens=record.get("left_truncated_prompt_tokens", 0))
            if text != record["response"]:
                raise ValueError("Decoded response differs from captured output")
            if any(t in self.tokenizer.all_special_ids for t in record["response_ids"]):
                raise ValueError("Special token in scoring range")
            details = summarize_fields(text, offsets, values)
            details["offsets"] = offsets
            global_value = details["global_uncertainty"]
        except ValueError as error:
            errors.append("alignment_or_numeric_error: " + str(error))
        if record.get("truncated"):
            errors.append("generation_truncated")
        if errors:
            local = premise = consequence = None
            # Fail closed even if an apparently complete field precedes truncation.
            details.update(missing_span=True, missing_reason="; ".join(errors), format_valid=False)
        else:
            local = details["local_uncertainty"]
            premise = details["premise_uncertainty"]
            consequence = details["consequence_uncertainty"]
            if details["missing_span"]:
                errors.append(details["missing_reason"])
        missing = bool(details.get("missing_span", True))
        fallback = global_value if missing and self.config["missing_policy"] == "global_only" else None
        if record.get("truncated") or any(e.startswith("alignment_or_numeric_error") for e in errors):
            fallback = None
            global_value = None
        details.update({
            "fallback_policy": self.config["missing_policy"],
            "fallback_uncertainty": fallback, "fallback_applied": fallback is not None,
            "effective_uncertainty": local if not missing else fallback,
            "method": "fixed_field_mean_token_nll_v1", "score_unit": "nats_per_token",
            "calibrated_error_probability": False, "prompt_tokens": len(record["prompt_ids"]),
            "generation_tokens_including_eos": record["generation_tokens"],
            "scoring_path": "raw_generation_logits_before_temperature",
            "additional_backbone_forward_passes": 0,
            "raw_logit_reduction_seconds": record["reduction_seconds"]})
        return InternalSignals(
            premise, consequence, local, global_value, missing, len(values),
            record["reduction_seconds"] + time.perf_counter() - started,
            self.backend, self.config["dtype"], self.config["model_revision"],
            "partial" if missing else "ok", errors, details)
