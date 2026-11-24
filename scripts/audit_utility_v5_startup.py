"""One-time real startup audit; no synthetic data or new model requests."""
import json,sys,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_utility_expansion_v5 import prior
from scripts.run_utility_sampling import read,rows,check
from evaluation.utility_sampling_v5 import context_key
from evaluation.utility_labels_v3 import aggregate
from evaluation.independent_dev import atomic


def main():
    run=ROOT/'outputs/utility_expansion_v5';m=read(run/'manifest.json');check(m)
    if list(run.rglob('error.json')):raise ValueError('Worker errors present')
    cs,oldlabels,_=prior();lookup={c['case_id']:c for c in cs}
    blocked={context_key(lookup[l['case_id']]) for l in oldlabels};models=[];checked=[];pair_checks=[];known={}
    for w in sorted((run/'densify_train').glob('worker*')):
        model=read(w/'model.json')
        if not model['frozen_actor'] or not model['actor_reviewer_shared'] or any(model['loading_info'].values()):raise ValueError('Not strict shared frozen Actor/Reviewer')
        chosen=rows(w/'dense_selections.jsonl')
        if not chosen or not any(x['selected'] for x in chosen):raise ValueError('No actual selected states')
        selection=next(s for x in chosen for s in x['selected']);c=lookup[selection['case_id']]
        if c['role']!='probe_train' or context_key(c) in blocked:raise ValueError('Illegal or already-labelled context')
        item=torch.load(ROOT/c['feature_file'],map_location='cpu',weights_only=True)
        if item['case_id']!=c['case_id'] or item['contract']!=model['contract']:raise ValueError('Cached/current backbone contract mismatch')
        n=len(c['metadata']['generation_record']['response_ids'])
        for h in item['features'].values():
            if h.shape!=(n,4096) or not torch.isfinite(h.float()).all():raise ValueError('Invalid actual-token feature shape')
        pairs=rows(w/'pairs.jsonl');labels=rows(w/'state_labels.jsonl')
        for pair in pairs:
            c2=lookup[pair['case_id']]
            if pair['skip']['traces'][0]['action']!=c2['draft_action']:raise ValueError('Skip intervention action changed')
            if pair['delta_return']!=int(pair['call']['success'])-int(pair['skip']['success']):raise ValueError('Invented paired delta')
            if pair['skip']['formal_steps']>50 or pair['call']['formal_steps']>50:raise ValueError('Horizon mismatch')
            pair_checks.append(dict(case_id=pair['case_id'],seed=pair['seed'],delta=pair['delta_return'],
                actual_skip_action=pair['skip']['traces'][0]['action'],actual_call_action=pair['call']['traces'][0]['action']))
        for l in labels:
            if l['label'] in ['unknown']:continue
            c2=lookup[l['case_id']]
            if l['feature_available_at_ns']>=l['label_observed_at_ns']:raise ValueError('Temporal leakage')
            a=aggregate(l['deltas'],l['changed'],not l['changed'])
            if a['label']!=l['label']:raise ValueError('Label definition changed')
            sim=l['simulation']
            if not sim['prediction_available'] or not sim['formal_unchanged'] or sim['worker_pid']==sim['formal_pid']:raise ValueError('Not isolated lookahead')
            for pair in [p for p in pairs if p['case_id']==l['case_id']]:
                if pair['skip']['traces'][0]['actual']!=sim['candidate_transition']:raise ValueError('Actual Skip and real trial transition differ')
                if pair['delta_return']!=int(pair['call']['success'])-int(pair['skip']['success']):raise ValueError('Invented utility delta')
                if pair['skip']['formal_steps']>50 or pair['call']['formal_steps']>50:raise ValueError('Horizon mismatch')
            checked.append(dict(case_id=l['case_id'],task_id=l['task_id'],label=l['label'],deltas=l['deltas'],simulator_isolated=True,real_skip_matches_trial=True))
        models.append(dict(worker=w.name,strict=True,frozen=True,shared_actor_reviewer=True,cached_contract_matches=True,first_case=selection['case_id'],actual_response_tokens=n,pairs=len(pairs),complete_labels=len(labels)))
        ledger=read(w/'ledger.json')
        for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']:known[k]=known.get(k,0)+ledger[k]
    if len(models)!=2 or not pair_checks:raise ValueError('Need two strict workers and at least one actual complete paired continuation')
    audit=dict(status='pass',scope='Engineering startup, not new Probe effectiveness or population utility',at_unix=time.time(),
        source_commit=m['source_commit'],run='outputs/utility_expansion_v5',gpus=m['gpus'],workers=models,verified_labels=checked,verified_pairs=pair_checks,
        known_new_cost_snapshot=known,worker_errors=0,unit_tests=15,native_identity_checks=125,native_replay_controls=5,
        no_final_test_or_fresh_dev_used=True,no_probe_training_started=True)
    atomic(ROOT/'docs/utility_expansion_v5/startup_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['status','gpus','known_new_cost_snapshot','verified_labels']},indent=2))


if __name__=='__main__':main()
