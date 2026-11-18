"""Same V2 counterfactual execution with the corrected derived-label aggregator."""
import time
from evaluation.independent_dev import append,ResourceLimit
from evaluation.utility_labels_v3 import aggregate
from scripts.run_utility_sampling import rollout


def label_state(rt,row,case,selected,source_cases,source_result):
    remaining=50-case['step'];maximum=dict(model_calls=10*(remaining-1)+1,tokens=(10*(remaining-1)+1)*2560,
        transition_queries=12*(case['step']+remaining+2),audit_capture_bundles=12*(case['step']+remaining+4))
    base={k:case[k] for k in ['case_id','task_id','goal_group','role','step','feature_file','feature_available_at_ns','state_sha256','source_policy']}
    try:
        for k,v in maximum.items():rt.ledger.check(k,v)
    except ResourceLimit as e:
        if rt.ledger.stop_file.exists() or time.time()>rt.ledger.deadline:raise
        result=dict(**base,selection=selected,label='unknown',utility_help=None,utility_harm=None,error='whole_state_budget_reservation_failed',
            label_observed_at_ns=time.time_ns());append(rt.out/'state_labels.jsonl',result);return result
    env=rt.rebuild(row,case)
    try:
        sim=rt.simulate(env,case['draft_action']);review,meta=rt.review(env,case['past'],row,case['draft_action'],sim,seed=0)
    finally:env.close()
    if not sim['prediction_available']:
        result=dict(**base,selection=selected,label='unknown',utility_help=None,utility_harm=None,error='preview_reconstruction_failed',label_observed_at_ns=time.time_ns())
        append(rt.out/'state_labels.jsonl',result);return result
    final=review['final_action'];changed=final!=case['draft_action'];pairs=[]
    seeds=[1,2,3] if changed else [1]
    index=0
    while index<len(seeds):
        seed=seeds[index]
        if seed==1 and case['source_policy']=='never':
            # This is the actual previously executed Skip trajectory, not a generated outcome.
            suffix=[c for c in source_cases if c['step']>=case['step']]
            if not suffix or suffix[0]['case_id']!=case['case_id']:raise ValueError('Cached Skip identity mismatch')
            skip=dict(success=source_result['success'],score=source_result['score'],formal_steps=source_result['formal_steps'],
                traces=[dict(action=c['executed_action'],actual=c['actual']) for c in suffix],seed=1,format_abort=source_result['format_abort'],
                reused_observed_source=True,source_episode_id=source_result['episode_id'])
        else:skip=rollout(rt,row,case,case['draft_action'],seed,'skip')
        call=rollout(rt,row,case,final,seed,'fixed_review_action')
        if not changed and call['traces']!=skip['traces']:
            raise ValueError('Unchanged intervention produced unequal fixed-seed continuation')
        d=int(call['success'])-int(skip['success']);pair=dict(case_id=case['case_id'],seed=seed,delta_return=d,skip=skip,call=call,
            intervention='same verified one-step trial and fixed Reviewer seed0; only downstream Actor seed changes')
        append(rt.out/'pairs.jsonl',pair);pairs.append(pair);index+=1
        if changed and index==3 and any(p['delta_return'] for p in pairs):seeds.extend([4,5])
    summary=aggregate([p['delta_return'] for p in pairs],changed,not changed)
    result=dict(**base,**summary,selection=selected,changed=changed,review=review,simulation=sim,review_metadata=meta,
        continuation_seeds=seeds,deltas=[p['delta_return'] for p in pairs],pair_count=len(pairs),label_observed_at_ns=time.time_ns(),
        input_features_exclude_outcomes=True,action_correctness='unknown',fixed_reviewer_seed=0)
    assert result['feature_available_at_ns']<result['label_observed_at_ns']
    append(rt.out/'state_labels.jsonl',result);return result
