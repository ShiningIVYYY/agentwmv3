#!/usr/bin/env python3
"""Three synthetic prompts, real frozen Qwen inference. No environment/Agent loop."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.hidden_states import response_alignment
from internal_signals.token_nll import selected_token_nll, summarize_fields, teacher_forced_nll
from internal_signals.spanuq_adapter import sha256

CASES = [
    "Observation: A wooden door in front of you is closed and has an unlocked handle. Goal: Open the door.",
    "Observation: A dry plant sits beside a watering can full of water. Goal: Water the plant.",
    "Observation: A lamp is off and its power switch is within reach. Goal: Turn on the lamp.",
]
INSTRUCTION = """Choose one next action based only on the given observation and goal.
Output exactly three lines, with one short sentence per field and no extra text:
Premise: <current observable condition the action depends on>
Expected Consequence: <predicted immediate effect, not observed feedback>
Action: <one short concrete action>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    device = torch.device(args.device)

    def sync():
        torch.cuda.synchronize(device)

    config = json.loads(Path("configs/internal_signals/qwen3_4b.json").read_text())
    asset_manifest = json.loads(Path("docs/phase1/assets_manifest.json").read_text())
    for name, row in asset_manifest[config["backbone_id"]]["files"].items():
        if sha256(Path(args.backbone_dir) / name) != row["sha256"]:
            raise ValueError("Backbone/tokenizer differs from Phase 1 asset manifest: " + name)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone_dir, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    report = {
        "purpose": "standalone real-model feasibility on synthetic text inputs; no benchmark outcomes",
        "backbone": config["backbone_id"], "backbone_revision": config["backbone_revision"],
        "dtype": args.dtype, "enable_thinking": False,
        "seed": 0, "temperature": .4, "top_p": 1., "top_k": 0,
        "max_new_tokens": 128, "nll_source": "raw_pre_temperature_logits",
        "equivalence_tolerance_nats": .1,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "code_sha256": {name: sha256(name) for name in
                        ["internal_signals/token_nll.py", "scripts/token_nll_feasibility.py"]},
        "cases": []}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for index, observation in enumerate(CASES):
        online_started = time.perf_counter()
        messages = [{"role": "user", "content": INSTRUCTION + "\n" + observation}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0].tolist()
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=True, temperature=.4, top_p=1., top_k=0,
                repetition_penalty=1., max_new_tokens=128,
                return_dict_in_generate=True, output_logits=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id)
        sync()
        generation_seconds = time.perf_counter() - started
        original_ids = generated.sequences[0, len(prompt_ids):].tolist()
        response_ids = list(original_ids)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        while response_ids and response_ids[-1] in eos:
            response_ids.pop()
        if any(x in tokenizer.all_special_ids for x in response_ids):
            raise ValueError("Unexpected special token in response")
        response, offsets = response_alignment(tokenizer, prompt, prompt_ids, response_ids)
        sync()
        started = time.perf_counter()
        raw_logits = torch.stack([step[0] for step in generated.logits[:len(response_ids)]])
        generation_nll = selected_token_nll(raw_logits, response_ids).cpu()
        processed = torch.stack([step[0] for step in generated.scores[:len(response_ids)]])
        processed_nll = selected_token_nll(processed, response_ids).cpu()
        signal = summarize_fields(response, offsets, generation_nll)
        sync()
        reduction_seconds = time.perf_counter() - started
        online_seconds = time.perf_counter() - online_started
        online_peak = torch.cuda.max_memory_allocated(device)
        saved_logits_bytes = sum(x.numel() * x.element_size() for x in generated.logits)
        saved_scores_bytes = sum(x.numel() * x.element_size() for x in generated.scores)
        del generated, raw_logits, processed
        gc.collect()
        sync()
        started = time.perf_counter()
        teacher_nll = teacher_forced_nll(model, prompt_ids, response_ids)
        sync()
        extra_forward_seconds = time.perf_counter() - started
        delta = (generation_nll - teacher_nll).abs()
        teacher_signal = summarize_fields(response, offsets, teacher_nll)
        record = {
            "synthetic_prompt": observation, "messages": messages,
            "response": response, "prompt_token_ids": prompt_ids,
            "response_token_ids": response_ids, "offsets": offsets,
            "raw_generation_token_nll": generation_nll.tolist(),
            "teacher_forced_token_nll": teacher_nll.tolist(),
            "processed_sampling_token_nll": processed_nll.tolist(),
            "raw_vs_teacher_max_abs_nats": float(delta.max()),
            "raw_vs_teacher_mean_abs_nats": float(delta.mean()),
            "raw_vs_teacher_within_predeclared_tolerance": bool(delta.max() <= .1),
            "raw_vs_processed_mean_abs_nats": float((generation_nll - processed_nll).abs().mean()),
            "signal": signal, "teacher_forced_signal": teacher_signal,
            "cost": {"generation_with_logits_and_scores_seconds": generation_seconds,
                     "online_reduction_and_summary_seconds": reduction_seconds,
                     "online_total_seconds": online_seconds,
                     "extra_teacher_forward_seconds": extra_forward_seconds,
                     "online_additional_backbone_forward_passes": 0,
                     "validation_extra_backbone_forward_passes": 1,
                     "forward_tokens": len(prompt_ids) + len(response_ids),
                     "saved_raw_logits_bytes": saved_logits_bytes,
                     "saved_processed_scores_bytes_validation_only": saved_scores_bytes,
                     "online_peak_allocated_bytes": online_peak,
                     "validation_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)},
            "generated_tokens_including_eos": len(original_ids),
            "generation_hit_token_limit": len(original_ids) == 128 and original_ids[-1] not in eos}
        report["cases"].append(record)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in
                          ["response", "signal", "raw_vs_teacher_max_abs_nats", "cost"]}, indent=2))
    report["all_formats_valid"] = all(x["signal"]["format_valid"] and not x["signal"]["missing_span"] for x in report["cases"])
    report["all_numeric_checks_passed"] = all(x["raw_vs_teacher_within_predeclared_tolerance"] for x in report["cases"])
    report["phase2_started"] = False
    out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_formats_valid"] and report["all_numeric_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
