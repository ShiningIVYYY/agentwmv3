"""Unscored frozen generation for baselines/reviews; scored drafts reuse Phase2 verbatim."""
import time
import torch
from internal_signals.token_nll import FORMAT


@torch.inference_mode()
def unscored_generate(actor, messages, role):
    started = time.perf_counter()
    prompt = actor.render(messages)
    inputs = actor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(actor.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    if len(prompt_ids) > actor.config["max_prompt_tokens"]: raise ValueError("Actor prompt over unchanged limit")
    if actor.device.type == "cuda": torch.cuda.reset_peak_memory_stats(actor.device)
    actor.sync(); generation_start = time.perf_counter()
    with torch.random.fork_rng(devices=[actor.device.index] if actor.device.type == "cuda" else []):
        torch.manual_seed(actor.config["seed"])
        output = actor.model.generate(**inputs, do_sample=True, temperature=actor.config["temperature"],
            top_p=actor.config["top_p"], top_k=actor.config["top_k"], repetition_penalty=actor.config["repetition_penalty"],
            max_new_tokens=actor.config["max_new_tokens"], return_dict_in_generate=True,
            output_logits=False, pad_token_id=actor.tokenizer.eos_token_id)
    actor.sync(); generation_seconds = time.perf_counter() - generation_start
    emitted = output.sequences[0, len(prompt_ids):].tolist()
    eos = actor.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    response_ids = list(emitted)
    while response_ids and response_ids[-1] in eos: response_ids.pop()
    text = actor.tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    truncated = len(emitted) >= actor.config["max_new_tokens"] and emitted[-1] not in eos
    metadata = {"role": role, "prompt_tokens": len(prompt_ids), "finish_reason": "length" if truncated else "stop",
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(emitted), "total_tokens": len(prompt_ids) + len(emitted)},
        "decision_format_valid": bool(FORMAT.fullmatch(text)) and not truncated,
        "generation_record": {"messages": messages, "prompt_text": prompt, "prompt_ids": prompt_ids,
            "response": text, "response_ids": response_ids, "generation_tokens": len(emitted), "truncated": truncated,
            "backend": "hf_unscored_generation_bf16_v1", "dtype": actor.config["dtype"], "model_revision": actor.config["model_revision"]},
        "signal_computation": {"internal": False, "raw_logits_retained": False, "additional_forward_passes": 0},
        "internal_cost": {"generation_seconds": generation_seconds, "extraction_seconds": 0,
            "additional_backbone_forward_passes": 0, "retained_raw_logits_bytes": 0,
            "output_tokens_including_eos": len(emitted),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(actor.device) if actor.device.type == "cuda" else None},
        "unscored_total_seconds": time.perf_counter() - started}
    return text, metadata


class GateActorAdapter:
    def __init__(self, agent, needs_internal):
        self.agent, self.original, self.needs_internal = agent, agent.generate, needs_internal
        self.reviewing = False
        self.calls = []

    def generate(self, messages, task_id, step, response_schema=None, response_regex=None):
        if response_schema is not None or response_regex is not None:
            raise ValueError("Phase4 preserves unconstrained existing JSON review protocol")
        role = "review" if self.reviewing else "draft"
        before = time.time_ns()
        if self.needs_internal and role == "draft":
            text, meta = self.original(messages, task_id, step)
            meta["signal_computation"] = {"internal": True, "raw_logits_retained": True, "additional_forward_passes": 0}
        else:
            text, meta = unscored_generate(self.agent.internal_actor, messages, role)
            self.agent.total_calls += 1
        meta["role"] = role
        self.calls.append({"role": role, "task_id": task_id, "step": step, "started_at_ns": before,
                           "finished_at_ns": time.time_ns(), "usage": meta["usage"], "cost": meta["internal_cost"]})
        return text, meta
