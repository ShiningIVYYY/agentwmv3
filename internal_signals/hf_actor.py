"""Local frozen Actor behind the existing Agent.generate/count interface."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decision import SCHEMA, action_token_indices
from .provider import BACKEND, TokenNLLProvider
from .spanuq_adapter import sha256
from .token_nll import selected_token_nll


class HFInternalActor:
    def __init__(self, config_file, model_dir, device="cuda:0"):
        self.config = json.loads(Path(config_file).read_text())
        c = self.config
        if c["prompt_schema"] != SCHEMA or c["backend"] != BACKEND or c["dtype"] != "bfloat16":
            raise ValueError("Unvalidated backend/dtype/prompt schema")
        if importlib.metadata.version("transformers") != c["transformers_version"]:
            raise ValueError("Transformers differs from the pinned scoring provider")
        if importlib.metadata.version("torch").split("+")[0] != c["torch_version"]:
            raise ValueError("Torch differs from the pinned scoring provider")
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "docs/phase1/assets_manifest.json").read_text())[c["model_id"]]
        if manifest["revision"] != c["model_revision"]:
            raise ValueError("Asset revision differs from configured Actor")
        self.asset_hashes = {}
        for name, row in manifest["files"].items():
            digest = sha256(Path(model_dir) / name)
            if digest != row["sha256"]:
                raise ValueError("Backbone/tokenizer asset differs: " + name)
            self.asset_hashes[name] = digest
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(self.device).eval().requires_grad_(False)
        if self.model.config.model_type != "qwen3" or self.model.config.hidden_size != 2560:
            raise ValueError("Backbone configuration differs from Qwen3-4B")
        if any(p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise ValueError("Mixed/unexpected model precision")
        self.provider = TokenNLLProvider(self.tokenizer, c)
        self.fingerprint = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()

    def count(self, messages):
        return len(self.tokenizer(self.render(messages), add_special_tokens=False)["input_ids"])

    def render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def generate(self, messages, task_id, step, response_schema=None, response_regex=None):
        if response_schema is not None or response_regex is not None:
            raise ValueError("Phase 2 has no constrained WM review generation")
        prompt = self.render(messages)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        removed_prompt_tokens = 0
        if len(prompt_ids) > self.config["max_prompt_tokens"]:
            if not getattr(self, "evaluation_left_truncation", False):
                raise ValueError("Prompt over token budget")
            removed_prompt_tokens = len(prompt_ids) - self.config["max_prompt_tokens"]
            inputs = {key: value[:, removed_prompt_tokens:] for key, value in inputs.items()}
            prompt_ids = inputs["input_ids"][0].tolist()
        requested_seed = getattr(self, "request_seed", None)
        effective_seed = self.config["seed"] if requested_seed is None else requested_seed
        # Local RNG isolation avoids changing other callers; no global training RNG.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
            torch.manual_seed(effective_seed)
            generated = self.model.generate(
                **inputs, do_sample=True, temperature=self.config["temperature"],
                top_p=self.config["top_p"], top_k=self.config["top_k"],
                repetition_penalty=self.config["repetition_penalty"],
                max_new_tokens=self.config["max_new_tokens"],
                return_dict_in_generate=True, output_logits=True,
                pad_token_id=self.tokenizer.eos_token_id)
        self.sync()
        generation_seconds = time.perf_counter() - started
        emitted = generated.sequences[0, len(prompt_ids):].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        response_ids = list(emitted)
        while response_ids and response_ids[-1] in eos:
            response_ids.pop()
        truncated = len(emitted) >= self.config["max_new_tokens"] and emitted[-1] not in eos
        text = self.tokenizer.decode(response_ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        self.sync()
        started = time.perf_counter()
        if response_ids:
            logits = torch.stack([x[0] for x in generated.logits[:len(response_ids)]])
            values = selected_token_nll(logits, response_ids).cpu().tolist()
        else:
            values = []
        self.sync()
        reduction_seconds = time.perf_counter() - started
        record = {
            "backend": BACKEND, "dtype": self.config["dtype"], "model_revision": self.config["model_revision"],
            "messages": messages, "prompt_text": prompt, "prompt_ids": prompt_ids,
            "response": text, "response_ids": response_ids, "token_nll": values,
            "generation_tokens": len(emitted), "truncated": truncated,
            "left_truncated_prompt_tokens": removed_prompt_tokens,
            "reduction_seconds": reduction_seconds}
        signal = self.provider.extract(record).to_dict()
        offsets = signal["details"].get("offsets", [])
        action_tokens = action_token_indices(text, offsets)
        cost = {
            "generation_seconds": generation_seconds, "extraction_seconds": signal["extraction_latency"],
            "additional_backbone_forward_passes": 0,
            "retained_raw_logits_bytes": sum(x.numel() * x.element_size() for x in generated.logits),
            "output_tokens_including_eos": len(emitted),
            "action_value_tokens": len(action_tokens),
            "non_action_response_tokens_including_labels": len(response_ids) - len(action_tokens),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None}
        metadata = {
            "prompt_tokens": len(prompt_ids), "finish_reason": "length" if truncated else "stop",
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(emitted),
                      "total_tokens": len(prompt_ids) + len(emitted)},
            "internal_signals": signal, "decision_format_valid": signal["details"].get("format_valid", False),
            "internal_cost": cost,
            "generation_record": record,
            "actor_metadata": {"config_sha256": self.fingerprint, "model_revision": self.config["model_revision"],
                               "tokenizer_assets_sha256": self.asset_hashes,
                               "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                               "attention": "sdpa", "generation_seed": effective_seed,
                               "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"}}
        return text, metadata
