"""Pinned post-trained9B, one shared frozen model for scored drafts and review."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration
from internal_signals.hf_actor import HFInternalActor
from internal_signals.provider import TokenNLLProvider, QWEN35_BACKEND
from internal_signals.spanuq_adapter import sha256
from gates.generation import unscored_generate


class Qwen35Actor(HFInternalActor):
    def __init__(self, config_file, model_dir, device='cuda:0'):
        self.config = json.loads(Path(config_file).read_text())
        c = self.config
        if c['model_id'] != 'Qwen/Qwen3.5-9B' or c['backend'] != QWEN35_BACKEND or c['dtype'] != 'bfloat16':
            raise ValueError('Independent pinned Qwen3.5-9B BF16 profile required')
        if importlib.metadata.version('transformers') != c['transformers_version']:
            raise ValueError('Unpinned Transformers')
        if torch.__version__.split('+')[0] != c['torch_version']:
            raise ValueError('Unpinned PyTorch')
        model_dir = Path(model_dir)
        manifest = json.loads((model_dir/'assets_manifest.json').read_text())
        if manifest['model_id'] != c['model_id'] or manifest['revision'] != c['model_revision']:
            raise ValueError('Model revision mismatch')
        self.asset_hashes = {}
        for name, expected in manifest['files'].items():
            actual = sha256(model_dir/name)
            if actual != expected['sha256']:
                raise ValueError('Changed model/tokenizer asset: '+name)
            self.asset_hashes[name] = actual
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs', 'conversion_errors')):
            raise ValueError('Checkpoint loading is not exact: '+str(info))
        # Transformers5 returns sets in loading_info; normalize metadata only.
        self.loading_info = {k: sorted(v) if isinstance(v, set) else v for k,v in info.items()}
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        if self.model.config.model_type != 'qwen3_5' or any(p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise ValueError('Unexpected model architecture or mixed parameter dtype')
        original_eos = self.model.generation_config.eos_token_id
        original_eos = original_eos if isinstance(original_eos,list) else [original_eos]
        official_eos = sorted(set(original_eos+[self.tokenizer.eos_token_id]))
        if official_eos != sorted(c['eos_token_ids']):
            raise ValueError('Official model/chat tokenizer EOS differs from pinned configuration')
        # Explicit configuration: do not inherit model-card sampling defaults.
        self.model.generation_config = GenerationConfig(
            do_sample=True, temperature=c['temperature'], top_p=c['top_p'], top_k=c['top_k'],
            repetition_penalty=c['repetition_penalty'], eos_token_id=official_eos,
            pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        self.provider = TokenNLLProvider(self.tokenizer, c)
        self.fingerprint = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        self.evaluation_left_truncation = True

    def generate_role(self, messages, role, seed, scored=False):
        self.request_seed = seed
        if scored:
            text, meta = super().generate(messages, role, 0)
        else:
            text, meta = unscored_generate(self, messages, role)
        meta.update(role=role, shared_model_id=self.config['model_id'],
                    shared_model_revision=self.config['model_revision'], config_sha256=self.fingerprint)
        return text, meta
