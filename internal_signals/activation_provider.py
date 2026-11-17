"""Versioned supervised provider; scores are not Token NLL or calibrated probabilities."""
import hashlib,time
from pathlib import Path
from internal_signals.activation_training import Ensemble


class ActivationProvider:
    def __init__(self,root,selection):
        if selection['status']!='trained':raise ValueError('Unavailable head must remain Unknown')
        self.selection=selection;self.ensemble=Ensemble(root,selection['checkpoints'])
        self.hashes={p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in selection['checkpoints']}

    def extract(self,values,contract):
        start=time.perf_counter();score=self.ensemble.score(values,contract)
        return dict(status='ok',provider='activation_probe_v1',score_type='supervised_uncalibrated',
            score=score,target=self.ensemble.target,threshold=self.selection['threshold'],
            missing_span=False,calibrated_probability=False,layer=self.ensemble.layer,
            extraction_latency=time.perf_counter()-start,checkpoint_sha256=self.hashes,
            actor_revision=contract['actor_revision'],dtype=contract['capture']['dtype'],
            contract=contract)
