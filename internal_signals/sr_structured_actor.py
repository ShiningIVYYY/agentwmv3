"""Opt-in SR format adapter; original SR/Qwen3.5 profiles remain intact."""
import hashlib,json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from internal_signals.ee_sr_actor import EESRActor,REVISION
from internal_signals.hf_actor import HFInternalActor
from internal_signals.provider import TokenNLLProvider
from internal_signals.structured_decode import FormatMachine
from internal_signals.sr_alignment import extract_signal
from gates.generation import unscored_generate
from reflection.replay_review_v2 import ReviewerProtocol

BACKEND='hf_ee_sr_structured_raw_bf16_v1'


class StructuredSRProvider(TokenNLLProvider):
    def __init__(self,tokenizer,config):
        if config['backend']!=BACKEND or config['model_revision']!=REVISION or config['dtype']!='bfloat16':
            raise ValueError('Unvalidated structured SR identity')
        if config['format_constraint']!='three-line-and-compact-json-fsm-v1':raise ValueError('Wrong grammar')
        if config['missing_policy'] not in ('global_only','null'):raise ValueError('Unknown missing fallback')
        self.tokenizer=tokenizer;self.config=config;self.backend=BACKEND
    def extract(self,record):return extract_signal(self,record)


class StructuredSRActor(EESRActor):
    def __init__(self,config_file,model_dir,device='cuda:0'):
        c=json.loads(Path(config_file).read_text())
        root=Path(__file__).resolve().parents[1]
        old=root/'configs/internal_signals/ee_scienceworld_sr_v1.json'
        original=json.loads(old.read_text())
        for k,v in original.items():
            if k not in ('backend','purpose') and c.get(k)!=v:raise ValueError('Inherited SR setting changed: '+k)
        if c.get('format_constraint')!='three-line-and-compact-json-fsm-v1':raise ValueError('Wrong output adaptation')
        # Reuse the exact audited strict loading and asset checks; no state-dict edits.
        super().__init__(old,model_dir,device)
        self.config=c;self.provider=StructuredSRProvider(self.tokenizer,c)
        self.fingerprint=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        self.evaluation_left_truncation=True
        self.last_constraint=None

    @contextmanager
    def constrained(self,kind,cap,draft=None):
        original=self.model.generate
        instance_override='generate' in self.model.__dict__
        previous_override=self.model.__dict__.get('generate')
        machine=[]
        def invoke(*args,**kwargs):
            if args or kwargs.get('logits_processor'):raise ValueError('Unexpected generation path/processor')
            if kwargs['max_new_tokens']!=cap:raise ValueError('Generation cap changed')
            m=FormatMachine(self.tokenizer,kwargs['input_ids'].shape[1],cap,kind,draft)
            kwargs['logits_processor']=[m]
            out=original(**kwargs)
            emitted=out.sequences[0,kwargs['input_ids'].shape[1]:].tolist()
            machine.append(m.verify_finished(emitted))
            return out
        self.model.generate=invoke
        try:yield machine
        finally:
            if instance_override:self.model.generate=previous_override
            else:del self.model.generate

    def generate_role(self,messages,role,seed,scored=False):
        self.request_seed=seed
        with self.constrained('draft',self.config['max_new_tokens']) as receipt:
            text,meta=HFInternalActor.generate(self,messages,role,0) if scored else unscored_generate(self,messages,role)
        meta.update(role=role,shared_model_id=self.config['model_id'],shared_model_revision=REVISION,
            config_sha256=self.fingerprint,model_subfolder='sr',format_constraint=receipt[0],
            task_specific_training_by_checkpoint_authors=True,additional_training_in_this_project=False)
        return text,meta

    def generate_review(self,messages,seed,cap):
        payload=json.loads(messages[-1]['content'].rsplit('\n',1)[-1]);draft=payload['draft_action']
        proxy=SimpleNamespace(config=dict(self.config,max_new_tokens=cap),model=self.model,tokenizer=self.tokenizer,
            device=self.device,render=self.render,sync=self.sync,request_seed=seed,evaluation_left_truncation=True)
        with self.constrained('review',cap,draft) as receipt:
            text,meta=unscored_generate(proxy,messages,'review')
        meta.update(role='review',shared_model_id=self.config['model_id'],shared_model_revision=REVISION,
            config_sha256=self.fingerprint,reviewer_version='compact-json-conservative-v2',
            reviewer_max_new_tokens=cap,grammar_constrained_decoding=True,format_constraint=receipt[0])
        return text,meta


class SRReviewerProtocol(ReviewerProtocol):
    def __init__(self):super().__init__('compact-json-conservative-v2')
    def generate(self,actor,messages,seed):return actor.generate_review(messages,seed,self.max_new_tokens)
