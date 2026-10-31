#!/usr/bin/env python3
"""Real backbone/extraction smoke. Checkpoint failure makes overall exit code 2."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.hidden_states import extract_hidden_states
from internal_signals.spanuq_adapter import SpanUQAdapter, CompatibilityError, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-dir", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This smoke measures a real CUDA backbone")
    report = {"acceptance_passed": False,
              "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu": torch.cuda.get_device_name(device),
              "versions": {name: importlib.metadata.version(name)
                           for name in ["torch", "transformers", "accelerate", "tokenizers", "huggingface-hub"]},
              "cuda_runtime": torch.version.cuda,
              "queries": [],
              "phase2_started": False,
              "code_sha256": {str(path): sha256(path) for path in [
                  Path(__file__), Path("internal_signals/hidden_states.py"),
                  Path("internal_signals/spanuq_adapter.py")]},
              "revisions": json.loads(Path("configs/internal_signals/qwen3_4b.json").read_text())}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    try:
        probe = SpanUQAdapter(args.source, args.checkpoint_dir, device=args.device)
        report["probe_strict_loading"] = probe.metadata
    except CompatibilityError as error:
        probe = None
        report["probe_strict_loading"] = {"error": str(error), **error.report}
        save()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone_dir, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    torch.cuda.synchronize(device)
    report["backbone_loading_seconds"] = time.perf_counter() - started
    report["backbone_load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["backbone_load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report["backbone_config"] = {"model_type": model.config.model_type,
                                "layers": model.config.num_hidden_layers,
                                "hidden_size": model.config.hidden_size}
    save()
    for query in [
        "In two short sentences, explain why ice melts when heated.",
        "In two short sentences, explain the roles of roots and leaves in a plant.",
        "Who wrote Pride and Prejudice, and in which year was it first published? Answer briefly.",
    ]:
        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=model.generation_config.eos_token_id)
        torch.cuda.synchronize(device)
        gen_seconds = time.perf_counter() - started
        generated_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
        terminators = model.generation_config.eos_token_id
        if isinstance(terminators, int):
            terminators = [terminators]
        response_ids = list(generated_ids)
        while response_ids and response_ids[-1] in terminators:
            response_ids.pop()
        extraction = extract_hidden_states(model, tokenizer, messages, response_ids,
                                            verify_layer_mapping=True)
        # Verify repeat extraction without keeping all hidden states between samples.
        repeated = extract_hidden_states(model, tokenizer, messages, response_ids)
        max_delta = max(float((a.float() - b.float()).abs().max())
                        for a, b in zip(extraction.hidden_states, repeated.hidden_states))
        if max_delta != 0:
            raise RuntimeError("Identical real forward passes were not reproducible")
        record = {"query": query, "messages": messages, "response": extraction.response,
                  "generated_token_ids_including_eos": generated_ids,
                  "prompt_token_ids": extraction.prompt_ids,
                  "response_token_ids": response_ids,
                  "response_offsets": extraction.offsets,
                  "enable_thinking": False, "generation": {"do_sample": False,
                  "max_new_tokens": args.max_new_tokens, "seed": 0,
                  "purpose": "official-style minimal inference; not benchmark evaluation"},
                  "generation_seconds": gen_seconds, **extraction.metadata,
                  "repeat_hidden_state_max_absolute_error": max_delta,
                  "validation_backbone_forward_passes": 2,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        if probe is None:
            record["spanuq_status"] = "not_run_checkpoint_architecture_incompatible"
            record["detected_spans"] = None
            record["uncertainty_score"] = None
        else:
            record["spanuq"] = probe.score(extraction)
            record["spanuq_status"] = "ran"
        report["queries"].append(record)
        save()
        print(json.dumps({k: v for k, v in record.items()
                         if k not in ["prompt_token_ids", "response_token_ids",
                                      "generated_token_ids_including_eos", "response_offsets"]}, indent=2))
        del extraction, repeated, inputs, generated
    report["backbone_hidden_state_checks_passed"] = True
    report["acceptance_passed"] = probe is not None
    save()
    return 0 if report["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
