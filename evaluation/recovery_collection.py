"""Keep complete real trajectories, quarantine only unalignable feature samples."""
import copy,time
from pathlib import Path
from evaluation.independent_dev import append
from internal_signals.actual_token_alignment import field_masks,VERSION
from evaluation.activation_runtime import native_label
from evaluation.utility_sampling import select_cases,public_flags
from scripts.run_utility_sampling import delta,mine_training_candidates
from worldmodel.alfworld_environment import fingerprint
ROOT=Path(__file__).resolve().parents[1]


def source_episode(rt,row,role,policy,memory,thresholds):
    import torch
    start=time.perf_counter();before=dict(rt.ledger.state);env=rt.new_env(row);past=[];steps=[];abort=False
    episode_id='utility-v2/'+row['task_id']+'/'+policy+'/seed1'
    try:
        for step in range(50):
            state=copy.deepcopy(env.current);prefix=copy.deepcopy(env.history)
            parsed,text,meta=rt.draft(env,past,row,seed=1,scored=True)
            if not parsed['valid']:
                abort=True;append(rt.out/'format_errors.jsonl',dict(episode_id=episode_id,step=step,raw=text,metadata=meta));break
            decision_at=time.time_ns();draft=parsed['action'];signals=meta['internal_signals']
            external,_=memory.evaluate(dict(task_id=row['task_id'],episode_id=episode_id,goal=env.goal,observation=state['observation'],
                draft_action=draft,commands=state['admissible_commands'],available_at_ns=decision_at))
            sampling=public_flags(draft,state['admissible_commands'],state['observation'],past,external,signals['local_uncertainty'],signals['global_uncertainty'],thresholds)
            eligible=True;error=None;masks=None;feature_at=None;cost=None;f=None
            try:
                if signals['status']!='ok':raise ValueError(';'.join(signals['errors']) or 'Missing internal spans')
                masks=field_masks(rt.actor,meta)
                if any(not x for x in masks.values()):raise ValueError('Missing actual-ID field mask')
            except ValueError as e:
                eligible=False;error=str(e)
                append(rt.out/'alignment_quarantine.jsonl',dict(case_id=episode_id+'/'+str(step),task_id=row['task_id'],role=role,
                    error=error,metadata=meta,decision_at_ns=decision_at,no_uncertainty_zero_substitution=True))
            if eligible:
                # Shape/nonfinite/backbone failures remain fatal; never mask them as alignment.
                values,cost=rt.features(meta);feature_at=time.time_ns()
                f=rt.out/'features'/row['task_id']/policy/(str(step)+'.pt');f.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(features=values,contract=rt.contract,case_id=episode_id+'/'+str(step),field_token_indices=masks),f)
            final=draft;sim=review=None
            if policy=='always':
                sim=rt.simulate(env,draft);review,_=rt.review(env,past,row,draft,sim,seed=0);final=review['final_action']
            actual=rt.step(env,final);label_at=time.time_ns()
            if sim and sim['prediction_available'] and final==draft and actual!=sim['candidate_transition']:raise ValueError('Source KEEP mismatch')
            draft_actual=sim['candidate_transition'] if sim and sim['prediction_available'] else actual if final==draft else None
            c=dict(case_id=episode_id+'/'+str(step),episode_id=episode_id,task_id=row['task_id'],role=role,source_policy=policy,source_seed=1,
                goal_group=row['goal_group'],step=step,goal=env.goal,observation=state['observation'],commands=state['admissible_commands'],
                draft=text,draft_action=draft,executed_action=final,metadata=meta,external=external,sampling=sampling,
                feature_valid=eligible,alignment_version=VERSION,alignment_error=error,feature_file=str(f.relative_to(ROOT)) if f else None,
                capture_cost=cost,decision_at_ns=decision_at,feature_available_at_ns=feature_at,label_observed_at_ns=label_at,
                actual=actual,native_rejection=native_label(draft,state['admissible_commands'],draft_actual) if draft_actual else None,
                prefix=prefix,past=copy.deepcopy(past),state_sha256=fingerprint(state),simulation=sim,review=review)
            assert decision_at<label_at
            append(rt.out/'source_steps.jsonl',c)
            if eligible:
                assert feature_at<label_at;append(rt.out/'cases.jsonl',c)
            steps.append(c);past.append(dict(observation=state['observation'],action=final))
            if actual['done']:break
        result=dict(episode_id=episode_id,task_id=row['task_id'],role=role,policy=policy,goal_group=row['goal_group'],seed=1,
            success=env.current['success'],score=env.current['score'],formal_steps=env.steps,format_abort=abort,
            case_ids=[c['case_id'] for c in steps],quarantined_feature_steps=sum(not c['feature_valid'] for c in steps),
            seconds=time.perf_counter()-start,cost=delta(rt.ledger,before))
        append(rt.out/'source_episodes.jsonl',result);return steps,result
    finally:env.close()


def select_pair(never,always,role,task):
    valid=[c for c in never+always if c.get('feature_valid',True)];selected=select_cases(valid,task);ids={s['case_id'] for s in selected};first=None
    for a,b in zip(never,always):
        if a['executed_action']!=b['executed_action']:
            first=dict(step=a['step'],never_case_id=a['case_id'],always_case_id=b['case_id'],exact_common_state=a['state_sha256']==b['state_sha256'],
                same_draft=a['draft_action']==b['draft_action'],feature_eligible=a.get('feature_valid',True))
            break
        if a['state_sha256']!=b['state_sha256'] or a['actual']!=b['actual']:raise ValueError('Common prefix state mismatch')
    if role=='probe_train' and first and first['exact_common_state'] and first['same_draft'] and first['feature_eligible'] and first['never_case_id'] not in ids:
        selected.append(dict(case_id=first['never_case_id'],step=first['step'],kind='first_shared_state_action_divergence',inclusion_probability=None,selection_uses_future_outcomes=True))
    return selected,first


def mine_candidates(selected,never,always,*args):
    # Label rollouts still receive ALL real source steps, including quarantined ones.
    return mine_training_candidates(selected,[c for c in never if c.get('feature_valid',True)],
        [c for c in always if c.get('feature_valid',True)],*args)
