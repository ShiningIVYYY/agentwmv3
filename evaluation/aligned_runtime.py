"""Frozen runtime with only provider alignment changed; vectors/contracts unchanged."""
from evaluation.activation_runtime import Runtime
from internal_signals.actual_token_nll import ActualTokenNLLProvider
from internal_signals.actual_token_alignment import VERSION
from evaluation.independent_dev import atomic
import json


class AlignedRuntime(Runtime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.actor.provider=ActualTokenNLLProvider(self.actor.tokenizer,self.actor.config)
        p=self.out/'model.json';d=json.loads(p.read_text());d['alignment_version']=VERSION
        d['capture_contract_unchanged']=True;atomic(p,d)
