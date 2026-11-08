#!/usr/bin/env python3
"""Prespecified descriptive, instance-weighted independent dev analysis.

Reads completed REAL trajectories only. No routing/threshold/model modification.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash
from gates.core import RuleGate,GateInput,ExternalInput
from scripts.audit_phase57_replay import feedback_status
from scripts.export_phase57_evidence import compact


def mean(values):return sum(values)/len(values)


def interval(rows,key,plan):
    groups=defaultdict(list)
    for r in rows:groups[r['goal_group']].append(float(r[key]))
    names=sorted(groups);rng=np.random.default_rng(plan['bootstrap_seed']);draw=[]
    for _ in range(plan['bootstrap_resamples']):
        chosen=rng.integers(0,len(names),len(names));total=0.;count=0
        for i in chosen:
            values=groups[names[int(i)]];total+=sum(values);count+=len(values)
        draw.append(total/count)
    tail=(1-plan['confidence'])/2
    return dict(method='descriptive_goal_cluster_percentile_bootstrap',clusters=len(names),resamples=len(draw),
        seed=plan['bootstrap_seed'],low=float(np.quantile(draw,tail)),high=float(np.quantile(draw,1-tail)),
        no_significance_claim=True)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def rejection_reason(text,score):
    t=text.casefold()
    status=feedback_status(text)
    if status!='explicit_rejection':
        if status=='accepted_feedback' and score<0:return 'accepted_execution_but_task_failure'
        if 'which' in t and ('mean' in t or 'object' in t):return 'ambiguity_prompt_unknown'
        return status
    if 'no known action' in t:return 'syntax_or_reference_resolution'
    if any(s in t for s in ('not open','not closed','must first','already open','already closed')):return 'state_precondition'
    if any(s in t for s in ('not visible','not accessible','could not find','cannot find','can\'t find')):return 'reference_or_visibility'
    if 'which' in t and ('mean' in t or 'object' in t):return 'ambiguity_prompt_unknown'
    if feedback_status(text)=='explicit_rejection':return 'other_explicit_rejection'
    if score<0:return 'task_failure_execution_may_be_accepted'
    return 'no_explicit_rejection_or_unknown'


def analyze(run,out):
    run=Path(run);out=Path(out)
    if out.exists():raise ValueError('New analysis output required')
    if not (run/'complete.json').exists():raise ValueError('All96 completed Episodes required before final analysis')
    audit=json.loads((run/'final_audit.json').read_text())
    if audit['status']!='engineering_checks_pass':raise ValueError('Engineering audit failed')
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile'];plan=profile['analysis_plan']
    out.mkdir(parents=True);(out/'decisions').mkdir()
    episodes=[];traces=[];index=[];lookaheads=[]
    for g in manifest['schedule']:
        for policy in ('never','always','internal','full'):
            p=run/'episodes'/g['task_id']/('seed'+str(g['seed']))/policy
            e=json.loads((p/'complete.json').read_text());rows=read_rows(p/'steps.jsonl')
            ss=[t['simulation'] for t in rows if t['simulation']]
            e['extra_transition_queries']=sum(sum(z['simulation_cost'][k] for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls')) for z in ss)
            e['total_transition_queries']=1+len(rows)+e['extra_transition_queries']
            e['simulation_seconds']=sum(z['simulation_latency'] for z in ss)
            e['actor_calls']=len(rows)
            episodes.append(e);traces.extend(rows)
            for name in ('steps.jsonl','generation_calls.jsonl','initial.json','complete.json'):
                index.append(dict(path=str((p/name).relative_to(run)),sha256=file_hash(p/name)))
            decisions=[]
            for t in rows:
                s=t['metadata'].get('internal_signals');sim=t['simulation'];review=t['review'];actual=t['actual'];ext=t['resolved_external']
                decisions.append(dict(policy=policy,task_id=t['task_id'],rollout_seed=t['rollout_seed'],step=t['step']+1,
                    local_nll=s.get('local_uncertainty') if s else None,global_nll=s.get('global_uncertainty') if s else None,
                    missing_span=s.get('missing_span') if s else None,call_wm=t['gate']['call_wm'],reason=t['gate']['reason'],
                    triggered_signals=json.dumps(t['gate']['triggered_signals']),budget_after=t['budget_after'],
                    coverage=ext['coverage'],conflict=ext['conflict'],sufficiency=ext['sufficiency'],reliability=ext['reliability'],
                    evidence_ids=json.dumps(ext['evidence_ids']),draft_action=t['draft_action'],final_action=t['final_action'],
                    prediction_available=sim['prediction_available'] if sim else None,
                    review_decision=review['parsed']['decision'] if review else None,
                    score=actual['score'],reward=actual['reward'],done=actual['done'],feedback_status=feedback_status(actual['observation']),
                    actor_tokens=t['metadata']['usage']['total_tokens'],reviewer_tokens=t['review_metadata']['usage']['total_tokens'] if t['review_metadata'] else 0,
                    kept_state_match=t['kept_draft_transition_match'],actual_feedback_sha256=__import__('hashlib').sha256(actual['observation'].encode()).hexdigest()))
                if sim:lookaheads.append(t)
            write_csv(out/'decisions'/(g['task_id']+'-seed'+str(g['seed'])+'-'+policy+'.csv'),decisions)
    if len(episodes)!=96:raise ValueError('Missing episodes')
    write_csv(out/'episodes.csv',episodes);atomic(out/'episodes.json',episodes)
    policy_summaries=[]
    for policy in ('never','always','internal','full'):
        es=[e for e in episodes if e['policy']==policy];ts=[t for t in traces if t['policy']==policy]
        sr=[]
        for seed in profile['rollout_seeds']:
            subset=[e for e in es if e['rollout_seed']==seed]
            sr.append(dict(seed=seed,instances=len(subset),mean_task_score=mean([e['task_score'] for e in subset]),
                success_rate=mean([int(e['success']) for e in subset])))
        sims=[t['simulation'] for t in ts if t['simulation']];simcost=Counter()
        for sim in sims:simcost.update(sim['simulation_cost'])
        changed=[t for t in ts if t['draft_action']!=t['final_action']]
        external=[t['resolved_external'] for t in ts if t['external']]
        row=dict(policy=policy,instances=len(es),tasks=len({e['task_id'] for e in es}),seeds=sr,
            mean_task_score=mean([e['task_score'] for e in es]),success_rate=mean([int(e['success']) for e in es]),
            score_seed_sample_sd=statistics.stdev([x['mean_task_score'] for x in sr]),
            success_seed_sample_sd=statistics.stdev([x['success_rate'] for x in sr]),
            score_resampling_interval=interval(es,'task_score',plan),success_resampling_interval=interval(es,'success',plan),
            formal_steps=sum(e['formal_environment_actions'] for e in es),
            lookahead_attempts=len(sims),successful_lookaheads=sum(s['prediction_available'] for s in sims),
            rejected_lookaheads=sum(not s['prediction_available'] for s in sims),reviewer_calls=sum(e['reviewer_calls'] for e in es),
            revised_actions=len(changed),revision_execution_feedback=dict(Counter(feedback_status(t['actual']['observation']) for t in changed)),
            revision_rejection_reasons=dict(Counter(rejection_reason(t['actual']['observation'],t['actual']['score']) for t in changed)),
            review_decisions=dict(Counter(t['review']['parsed']['decision'] for t in ts if t['review_metadata'])),
            environment_initializations=len(es)+int(simcost['jvm_initializations']),
            input_tokens=sum(e['input_tokens'] for e in es),output_tokens=sum(e['output_tokens'] for e in es),
            total_tokens=sum(e['total_tokens'] for e in es),
            transition_queries=len(es)+sum(e['formal_environment_actions'] for e in es)+sum(simcost[k] for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls')),
            extra_transition_queries=sum(simcost[k] for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls')),
            simulation_cost=dict(simcost),simulation_seconds=sum(s['simulation_latency'] for s in sims),
            total_episode_seconds=sum(e['total_seconds_including_cleanup'] for e in es),
            actor_generation_seconds=sum(t['metadata']['internal_cost']['generation_seconds'] for t in ts),
            reviewer_generation_seconds=sum(t['review_metadata']['internal_cost']['generation_seconds'] for t in ts if t['review_metadata']),
            internal_extractions=sum('internal_signals' in t['metadata'] for t in ts),
            internal_extraction_seconds=sum(t['metadata']['internal_cost']['extraction_seconds'] for t in ts),
            extra_nll_forward_passes=sum(t['metadata']['internal_cost']['additional_backbone_forward_passes'] for t in ts),
            missing_internal_steps=sum(t['metadata'].get('internal_signals',{}).get('missing_span',False) for t in ts),
            invalid_draft_formats=sum(not t['metadata']['decision_format_valid'] for t in ts),
            retrieval_calls=len(external),retrieval_seconds=sum(t['external']['retrieval_latency'] for t in ts if t['external']),
            rule_judge_seconds=sum(t['external']['judge_latency'] for t in ts if t['external']),llm_judge_calls=0,
            gate_seconds=sum(t['gate']['gate_latency'] for t in ts),
            gate_reasons=dict(Counter(t['gate']['reason'] for t in ts)),triggered_signals=dict(Counter(x for t in ts for x in t['gate']['triggered_signals'])),
            coverage=dict(Counter(x['coverage'] for x in external)),conflict=dict(Counter(x['conflict'] for x in external)),
            sufficiency=dict(Counter(x['sufficiency'] for x in external)),
            peak_torch_allocated_bytes=max(m['internal_cost']['peak_allocated_bytes'] for t in ts for m in [t['metadata'],t['review_metadata']] if m))
        policy_summaries.append(row)
    atomic(out/'policy_summary.json',policy_summaries)
    lookup={(e['task_id'],e['rollout_seed'],e['policy']):e for e in episodes};pairs=[];comparisons=[]
    for baseline in ('never','always','internal'):
        rr=[]
        for g in manifest['schedule']:
            full=lookup[g['task_id'],g['seed'],'full'];other=lookup[g['task_id'],g['seed'],baseline]
            rr.append(dict(task_id=g['task_id'],rollout_seed=g['seed'],goal_group=full['goal_group'],baseline=baseline,
                score_delta=full['task_score']-other['task_score'],success_delta=int(full['success'])-int(other['success']),
                token_delta=full['total_tokens']-other['total_tokens'],
                transition_query_delta=full['total_transition_queries']-other['total_transition_queries'],
                episode_latency_delta=full['total_seconds_including_cleanup']-other['total_seconds_including_cleanup'],
                full_score=full['task_score'],baseline_score=other['task_score'],
                full_tokens=full['total_tokens'],baseline_tokens=other['total_tokens'],
                full_transition_queries=full['total_transition_queries'],baseline_transition_queries=other['total_transition_queries']))
        pairs.extend(rr)
        task_effects={task:mean([x['score_delta'] for x in rr if x['task_id']==task]) for task in {x['task_id'] for x in rr}}
        positive=sorted([v for v in task_effects.values() if v>0],reverse=True)
        leave_task=[mean([x['score_delta'] for x in rr if x['task_id']!=task]) for task in task_effects]
        goals={x['goal_group'] for x in rr};leave_goal=[mean([x['score_delta'] for x in rr if x['goal_group']!=goal]) for goal in goals]
        comparisons.append(dict(comparison='full-'+baseline,paired_instances=len(rr),mean_score_delta=mean([x['score_delta'] for x in rr]),
            mean_success_delta=mean([x['success_delta'] for x in rr]),mean_token_delta=mean([x['token_delta'] for x in rr]),
            mean_transition_query_delta=mean([x['transition_query_delta'] for x in rr]),
            mean_episode_latency_delta=mean([x['episode_latency_delta'] for x in rr]),
            score_interval=interval(rr,'score_delta',plan),success_interval=interval(rr,'success_delta',plan),
            per_seed_score_delta={str(seed):mean([x['score_delta'] for x in rr if x['rollout_seed']==seed]) for seed in (0,1)},
            per_task_mean_score_delta=task_effects,positive_tasks=sum(v>0 for v in task_effects.values()),negative_tasks=sum(v<0 for v in task_effects.values()),zero_tasks=sum(v==0 for v in task_effects.values()),
            top1_fraction_of_positive_task_gains=positive[0]/sum(positive) if positive else None,
            top3_fraction_of_positive_task_gains=sum(positive[:3])/sum(positive) if positive else None,
            leave_one_task_out_score_delta_range=[min(leave_task),max(leave_task)],leave_one_goal_out_score_delta_range=[min(leave_goal),max(leave_goal)]))
    write_csv(out/'paired_differences.csv',pairs);atomic(out/'paired_summary.json',comparisons)
    gate=json.loads((ROOT/profile['gate_config']).read_text());extra=[]
    for t in [x for x in traces if x['policy']=='full']:
        context=GateInput(t['gate']['decision_id'],t['gate']['budget_remaining'],t['metadata'].get('internal_signals'),
            ExternalInput(**t['resolved_external']),t['gate']['budget_remaining'],t['metadata']['decision_format_valid'])
        counter=RuleGate(dict(gate,policy='internal')).decide(context)
        if t['gate']['call_wm'] and not counter['call_wm']:
            value=compact(t);value['rollout_seed']=t['rollout_seed'];value['internal_counter_same_state']=counter
            value['execution_feedback_status']=feedback_status(t['actual']['observation'])
            value['changed_action']=t['draft_action']!=t['final_action'];extra.append(value)
    atomic(out/'external_added_calls.json',dict(scope='same_Full_recorded_state_and_budget; not same_trajectory_as_Internal',calls=extra,
        count=len(extra),changed=sum(x['changed_action'] for x in extra),
        changed_feedback=dict(Counter(x['execution_feedback_status'] for x in extra if x['changed_action'])),
        immediate_score_deltas=dict(Counter(str(x['immediate_score_delta']) for x in extra if x['immediate_score_delta'] is not None)),
        downstream_causal_utility_established=False))
    examples=[]
    for label,predicate in [
        ('successful_terminal',lambda t:t['actual']['done'] and t['actual']['score']==100),
        ('positive_immediate_correction',lambda t:t.get('one_step_score_delta_actual_minus_trial_draft',0)>0),
        ('negative_immediate_correction',lambda t:t.get('one_step_score_delta_actual_minus_trial_draft',0)<0),
        ('explicitly_rejected_revision',lambda t:t['draft_action']!=t['final_action'] and feedback_status(t['actual']['observation'])=='explicit_rejection'),
        ('natural_sufficiency',lambda t:t['resolved_external']['sufficiency']=='true'),
        ('replay_rejection',lambda t:t['simulation'] and not t['simulation']['prediction_available'])]:
        found=next((t for t in traces if predicate(t)),None)
        if found:examples.append(dict(label=label,rollout_seed=found['rollout_seed'],**compact(found)))
    atomic(out/'examples.json',examples)
    atomic(out/'raw_file_index.json',dict(run=str(run),files=index))
    atomic(out/'engineering_audit.json',audit)
    atomic(out/'analysis_provenance.json',dict(script_sha256=file_hash(Path(__file__)),run_manifest_sha256=file_hash(run/'manifest.json'),
        analysis_plan=plan,policy_effectiveness_is_descriptive=True,exact_downstream_WM_counterfactual_utility=False))
    print(json.dumps(dict(episodes=len(episodes),decisions=len(traces),external_added_calls=len(extra),output=str(out)),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.run,a.output)


if __name__=='__main__':main()
