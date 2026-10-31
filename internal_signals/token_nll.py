"""Phase 1B standalone feasibility prototype; no Agent/Gate integration."""
import math
import re

import torch
import torch.nn.functional as F

METHOD = "fixed_field_mean_token_nll_v1"
FORMAT = re.compile(
    r"\APremise: (?P<premise>[^\r\n]+)\n"
    r"Expected Consequence: (?P<consequence>[^\r\n]+)\n"
    r"Action: (?P<action>[^\r\n]+)\n?\Z"
)


def selected_token_nll(logits, token_ids):
    """Raw frozen-model surprisal in nats, not temperature-warped sampling NLL."""
    if logits.ndim != 2 or logits.shape[0] != len(token_ids) or not token_ids:
        raise ValueError("Need aligned nonempty [response_tokens, vocabulary] logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite raw logits")
    targets = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.float(), targets, reduction="none")


def summarize_fields(response, offsets, nll):
    """Mean token NLL for fixed field values; include intersecting boundary tokens."""
    values = [float(x) for x in nll]
    if not values or len(values) != len(offsets):
        raise ValueError("Offsets and NLL must be aligned and nonempty")
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("NLL must be finite and nonnegative")
    if any(not 0 <= a < b <= len(response) for a, b in offsets):
        raise ValueError("Invalid response character offsets")
    match = FORMAT.fullmatch(response)
    spans, missing = [], []
    if match:
        for field in ("premise", "consequence"):
            raw = match[field]
            left, right = match.span(field)
            left += len(raw) - len(raw.lstrip())
            right -= len(raw) - len(raw.rstrip())
            tokens = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
            if left >= right or not tokens:
                missing.append(field)
                continue
            spans.append({
                "field": field, "char_start": left, "char_end": right,
                "text": response[left:right], "token_indices": tokens,
                "token_count": len(tokens),
                "boundary_overhang_tokens": [
                    i for i in tokens if offsets[i][0] < left or offsets[i][1] > right],
                "uncertainty": math.fsum(values[i] for i in tokens) / len(tokens)})
        if len(spans) == 2 and set(spans[0]["token_indices"]) & set(spans[1]["token_indices"]):
            raise ValueError("A token intersects both separated field values")
    else:
        missing = ["premise", "consequence"]
    by_name = {span["field"]: span["uncertainty"] for span in spans}
    return {
        "schema_version": 1, "method": METHOD, "unit": "nats_per_token",
        "score_range": "[0,infinity)", "calibrated_error_probability": False,
        "local_uncertainty": max(by_name.values()) if not missing else None,
        "global_uncertainty": math.fsum(values) / len(values),
        "premise_uncertainty": by_name.get("premise"),
        "consequence_uncertainty": by_name.get("consequence"),
        "field_spans": spans, "missing_span": bool(missing),
        "missing_fields": missing, "format_valid": match is not None,
        "missing_reason": "invalid_fixed_format_or_empty_unscorable_field" if missing else None,
        "boundary_policy": "include_whole_token_if_character_interval_intersects_field_value",
        "global_token_policy": "all_response_tokens_including_labels_and_action_excluding_eos_prompt_padding",
        "response_tokens": len(values)}


@torch.inference_mode()
def teacher_forced_nll(model, prompt_ids, response_ids):
    """Exactly one extra causal LM forward with the next-token shift."""
    if not prompt_ids or not response_ids:
        raise ValueError("Prompt and response cannot be empty")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, return_dict=True)
    start = len(prompt_ids) - 1
    logits = output.logits[0, start:start + len(response_ids)]
    return selected_token_nll(logits, response_ids).detach().cpu()
