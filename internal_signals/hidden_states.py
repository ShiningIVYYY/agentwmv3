"""Exact chat-token extraction using the official batch producer's block convention."""
from dataclasses import dataclass
import hashlib
import json
import time

import torch


@dataclass
class Extraction:
    response: str
    prompt_ids: list
    response_ids: list
    offsets: list
    hidden_states: list
    metadata: dict


def response_alignment(tokenizer, prompt_text, prompt_ids, response_ids):
    """Fail on retokenization/merge drift; never trim arrays to the minimum length."""
    if not response_ids:
        raise ValueError("No response tokens")
    response = tokenizer.decode(response_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    encoded = tokenizer(prompt_text + response, add_special_tokens=False,
                        return_offsets_mapping=True)
    full_ids = prompt_ids + response_ids
    if encoded["input_ids"] != full_ids:
        raise ValueError("Full prompt/response retokenization differs from original IDs")
    boundary = len(prompt_text)
    absolute = encoded["offset_mapping"][len(prompt_ids):]
    if len(absolute) != len(response_ids):
        raise ValueError("Response token/offset lengths differ")
    offsets = []
    for start, end in absolute:
        if not boundary <= start < end <= len(prompt_text + response):
            raise ValueError("Token crosses prompt boundary or has an invalid offset")
        offsets.append((start - boundary, end - boundary))
    if tokenizer(prompt_text, add_special_tokens=False)["input_ids"] != prompt_ids:
        raise ValueError("Prompt token IDs changed")
    return response, offsets


@torch.inference_mode()
def extract_hidden_states(model, tokenizer, messages, response_ids, *,
                          blocks=(21, 22, 23), verify_layer_mapping=False,
                          max_total_tokens=4096, max_response_tokens=2048):
    total_started = time.perf_counter()
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(response_ids) > max_response_tokens:
        raise ValueError("Response exceeds checkpoint token cap")
    if len(prompt_ids) + len(response_ids) > max_total_tokens:
        raise ValueError("Prompt plus response exceeds extraction limit; no truncation")
    if any(token in set(tokenizer.all_special_ids) for token in response_ids):
        raise ValueError("Special response tokens must be explicitly excluded")
    response, offsets = response_alignment(tokenizer, prompt_text, prompt_ids, response_ids)
    if len(set(blocks)) != len(blocks) or any(i < 0 or i >= len(model.model.layers) for i in blocks):
        raise ValueError("Invalid decoder block selection")
    if model.config.model_type != "qwen3" or model.config.hidden_size != 2560:
        raise ValueError("Backbone must match the published Qwen3-4B hidden dimension")
    captured, handles = {}, []
    boundary = len(prompt_ids)

    def make_hook(index):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index] = tensor[:, boundary:, :].detach().clone()
        return hook

    for index in blocks:
        handles.append(model.model.layers[index].register_forward_hook(make_hook(index)))
    device = next(model.parameters()).device
    full = torch.tensor([prompt_ids + response_ids], device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        # The frozen decoder supplies the same states without allocating vocabulary logits.
        output = model.model(input_ids=full, attention_mask=torch.ones_like(full),
                             use_cache=False, output_hidden_states=verify_layer_mapping,
                             return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checks = {}
    if verify_layer_mapping:
        for index in blocks:
            reference = output.hidden_states[index + 1][:, boundary:, :]
            # Final block tuple entries may include final norm; these selected
            # intermediate blocks are strictly before the final decoder layer.
            if index == len(model.model.layers) - 1:
                raise ValueError("Final-layer normalization requires a separate convention")
            error = float((reference.float() - captured[index].float()).abs().max())
            if error != 0:
                raise ValueError("Hook differs from HF hidden_states[block+1]")
            checks[str(index)] = {"hf_index": index + 1, "max_absolute_error": error}
    hidden = [captured[index] for index in blocks]
    for tensor in hidden:
        if tuple(tensor.shape) != (1, len(response_ids), 2560) or not torch.isfinite(tensor).all():
            raise ValueError("Invalid response hidden states")
    key = hashlib.sha256(json.dumps(
        {"ids": prompt_ids + response_ids, "blocks": blocks,
         "backbone": getattr(model.config, "_commit_hash", None),
         "dtype": str(hidden[0].dtype)}, sort_keys=True).encode()).hexdigest()
    return Extraction(response, prompt_ids, list(response_ids), offsets, hidden,
                      {"decoder_blocks": list(blocks), "hf_hidden_state_indices": [i + 1 for i in blocks],
                       "layer_convention": "zero_based_decoder_block_output",
                       "shapes": [list(x.shape) for x in hidden], "finite": True,
                       "prompt_tokens": boundary, "response_tokens": len(response_ids),
                       "decoder_forward_seconds": elapsed,
                       "extraction_seconds": time.perf_counter() - total_started,
                       "backbone_forward_passes": 1,
                       "layer_mapping_checks": checks, "input_fingerprint": key,
                       "cache_hit": False, "cache_enabled": False})
