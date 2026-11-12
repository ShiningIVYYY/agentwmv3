"""Frozen, pinned ScienceWorld SR policy. Separate identity; no old Gate thresholds."""
import hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from internal_signals.hf_actor import HFInternalActor
from internal_signals.provider import TokenNLLProvider
from gates.generation import unscored_generate

BACKEND='hf_ee_scienceworld_sr_raw_bf16_v1'
REVISION='4b8f0835a3f613e40541619462ae906db8dba95c'


class SRTokenNLLProvider(TokenNLLProvider):
    def __init__(self,tokenizer,config):
        if config['backend']!=BACKEND or config['dtype']!='bfloat16' or config['model_revision']!=REVISION:
            raise ValueError('Unvalidated SR signal identity')
        if config['missing_policy'] not in ('global_only','null'):raise ValueError('Unknown missing fallback')
        # Reuse the verified reduction/alignment method, with a NEW exact identity.
        self.tokenizer=tokenizer;self.config=config;self.backend=BACKEND


class EESRActor(HFInternalActor):
    def __init__(self,config_file,model_dir,device='cuda:0'):
        self.config=json.loads(Path(config_file).read_text());c=self.config
        if c['model_id']!='osunlp/early-experience-scienceworld' or c['subfolder']!='sr' or c['model_revision']!=REVISION:raise ValueError('Wrong checkpoint')
        import importlib.metadata
        if importlib.metadata.version('transformers')!=c['transformers_version'] or torch.__version__.split('+')[0]!=c['torch_version']:raise ValueError('Unpinned software')
        model_dir=Path(model_dir);manifest=json.loads((model_dir.parent/'assets_manifest.json').read_text())
        if manifest['revision']!=REVISION or manifest['subfolder']!='sr':raise ValueError('Model asset identity mismatch')
        self.asset_hashes={}
        for name,row in manifest['files'].items():
            path=model_dir.parent/name;h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Asset changed '+name)
            self.asset_hashes[name]=h.hexdigest()
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        self.model,info=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors')):raise ValueError('Checkpoint did not load exactly: '+str(info))
        self.loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()}
        if self.model.config.model_type!='qwen2' or self.model.config.hidden_size!=3584 or self.model.config.num_hidden_layers!=28:raise ValueError('Wrong architecture')
        self.device=torch.device(device);self.model=self.model.to(self.device).eval().requires_grad_(False)
        if any(p.dtype!=torch.bfloat16 for p in self.model.parameters()):raise ValueError('Mixed parameter dtype')
        if self.tokenizer.eos_token_id!=151645:raise ValueError('Tokenizer EOS mismatch')
        official=self.model.generation_config.eos_token_id
        if sorted(official)!=sorted(c['eos_token_ids']):raise ValueError('Official generation EOS mismatch')
        self.original_generation_config=self.model.generation_config.to_dict()
        self.model.generation_config=GenerationConfig(do_sample=True,temperature=c['temperature'],top_p=c['top_p'],top_k=c['top_k'],
            repetition_penalty=c['repetition_penalty'],eos_token_id=c['eos_token_ids'],pad_token_id=c['pad_token_id'],use_cache=True)
        self.provider=SRTokenNLLProvider(self.tokenizer,c);self.fingerprint=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        self.evaluation_left_truncation=False
    def render(self,messages):
        # Qwen2.5 checkpoint supplies its own non-thinking ChatML template.
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    def generate_role(self,messages,role,seed,scored=False):
        if self.count(messages)>self.config['max_prompt_tokens']:raise ValueError('Prompt exceeds trial limit; no silent truncation')
        self.request_seed=seed
        text,meta=super().generate(messages,role,0) if scored else unscored_generate(self,messages,role)
        meta.update(role=role,shared_model_id=self.config['model_id'],shared_model_revision=REVISION,model_subfolder='sr',config_sha256=self.fingerprint,
            new_signal_backend=BACKEND,task_specific_training_by_checkpoint_authors=True,additional_training_in_this_project=False)
        return text,meta
