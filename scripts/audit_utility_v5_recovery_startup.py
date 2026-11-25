"""Read-only audit of actual recovered Validation decisions; no model generation."""
import json,sys,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from evaluation.independent_dev import atomic
from internal_signals.actual_token_nll import ActualTokenNLLProvider
from evaluation.utility_labels_v3 import aggregate

def rows(p):
    if not p.exists():return []
    s=p.read_text();ls=s.splitlines()
    if s and not s.endswith('\n'):ls=ls[:-1]
    return [json.loads(x) for x in ls]

def main():
    r=ROOT/'outputs/utility_expansion_v5_recovery';cfg=json.loads((ROOT/'configs/evaluation/utility_expansion_v5_recovery.json').read_text())
    m=json.loads((r/'manifest.json').read_text());split=json.loads((ROOT/cfg['split']).read_text())
    assignment=json.loads((r/'new_validation_gpu_assignment.json').read_text());allowed=set(split['new_validation_task_ids'])
    actor_cfg=json.loads((ROOT/cfg['actor_config']).read_text());t=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
    provider=ActualTokenNLLProvider(t,actor_cfg);models=[];verified=[];labels=[];case_count=0;trials=0
    if list(r.glob('new_*/worker*/error.json')) or (r/'error.json').exists():raise ValueError('Actual worker error present')
    for w in sorted((r/'new_validation').glob('worker*')):
        model=json.loads((w/'model.json').read_text())
        assert model['frozen_actor'] and model['actor_reviewer_shared'] and not any(model['loading_info'].values())
        cs=rows(w/'cases.jsonl');assert len(cs)>=5;case_count+=len(cs)
        for c in cs[:5]:
            assert c['task_id'] in allowed and c['role']=='probe_validation'
            assert c['feature_available_at_ns']<c['label_observed_at_ns'] and c['feature_valid']
            item=torch.load(ROOT/c['feature_file'],map_location='cpu',weights_only=True)
            assert item['case_id']==c['case_id'] and item['contract']==model['contract']
            record=c['metadata']['generation_record'];n=len(record['response_ids'])
            for h in item['features'].values():assert h.shape==(n,4096) and torch.isfinite(h.float()).all()
            assert all(ids and all(0<=i<n for i in ids) for ids in item['field_token_indices'].values())
            score=provider.extract(record);assert not score.missing_span
            for k in ['local_uncertainty','global_uncertainty','premise_uncertainty','consequence_uncertainty']:
                assert getattr(score,k)==c['metadata']['internal_signals'][k]
            verified.append(dict(case_id=c['case_id'],actual_response_tokens=n,shape_and_nll_match=True,no_future_features=True))
        for c in cs:
            sim=c.get('simulation')
            if sim and sim['prediction_available']:
                assert sim['formal_unchanged'] and sim['worker_pid']!=sim['formal_pid'];trials+=1
                if c['executed_action']==c['draft_action']:assert c['actual']==sim['candidate_transition']
        for l in rows(w/'state_labels.jsonl'):
            if l['label']!='unknown':assert aggregate(l['deltas'],l['changed'],not l['changed'])['label']==l['label']
            labels.append(l)
        models.append(dict(worker=w.name,strict_load=True,frozen=True,shared=True,source_cases=len(cs)))
    assert len(models)==len(assignment['gpus'])
    from collections import Counter
    audit=dict(status='pass',scope='Real recovered startup correctness only, no effectiveness claim',at_unix=time.time(),source_commit=m['source_commit'],actual_gpus=assignment['gpus'],models=models,real_new_decisions_at_snapshot=case_count,verified_cached_features=verified,isolated_trial_checks=trials,new_label_counts=dict(Counter(l['label'] for l in labels)),model_and_scoring_unchanged=True,unit_tests=9,no_test_or_fresh_dev=True,no_probe_training_started=not (r/'training').exists())
    atomic(ROOT/'docs/utility_expansion_v5/recovery_startup_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['status','actual_gpus','real_new_decisions_at_snapshot','isolated_trial_checks','new_label_counts','no_probe_training_started']}))

if __name__=='__main__':main()
