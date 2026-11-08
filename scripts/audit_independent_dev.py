#!/usr/bin/env python3
"""Outcome-independent acceptance checks; descriptive analysis is separate."""
from collections import Counter
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows,verify_frozen,request_seed
from experience.splits import file_hash
from gates.core import RuleGate,GateInput,ExternalInput
from internal_signals.decision import messages
from reflection.replay_review import review_messages


def require(value,why):
    if not value:raise ValueError(why)


def recorded_seed(metadata):
    # The accepted scored and unscored providers have distinct REAL schemas.
    # Require actual recorded values; never substitute the expected seed.
    values=[]
    record=metadata.get('generation_record',{})
    actor=metadata.get('actor_metadata',{})
    if 'request_seed' in record:values.append(record['request_seed'])
    if 'generation_seed' in actor:values.append(actor['generation_seed'])
    if not values or any(not isinstance(v,int) or isinstance(v,bool) for v in values):
        raise ValueError('Missing/invalid recorded generation seed')
    if len(set(values))!=1:raise ValueError('Inconsistent recorded generation seeds')
    return values[0]


def audit(run,stage):
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    verify_frozen(ROOT,profile)
    for name,expected in manifest['code_hashes'].items():require(file_hash(ROOT/name)==expected,'Changed evaluation code: '+name)
    groups=[g for g in manifest['schedule'] if stage=='all' or g['round']==1]
    config=json.loads((ROOT/profile['gate_config']).read_text());episodes=[];traces=[];initials={};call_count=0;tokens=0;queries=0;sim_count=0;keep=0
    for group in groups:
        task=group['task_id'];seed=group['seed']
        for policy in ('never','always','internal','full'):
            path=run/'episodes'/task/('seed'+str(seed))/policy
            result=json.loads((path/'complete.json').read_text());rows=read_rows(path/'steps.jsonl');calls=read_rows(path/'generation_calls.jsonl')
            initial=json.loads((path/'initial.json').read_text());initials.setdefault(task,set()).add(initial['state_sha256'])
            require(initial['simulator']==profile['engine'],'Initial engine mismatch')
            require(result['steps_sha256']==file_hash(path/'steps.jsonl'),'Changed completed trace')
            require(result['generation_calls_sha256']==file_hash(path/'generation_calls.jsonl'),'Changed generation ledger')
            require(result['task_id']==task and result['rollout_seed']==seed and result['policy']==policy,'Episode identity drift')
            gate=RuleGate(dict(config,policy=policy));past=[];budget=profile['wm_invocation_budgets'][policy];ep_queries=1;ep_tokens=0;ep_calls=0;ep_sims=0;ep_reviews=0
            for step,t in enumerate(rows):
                meta=t['metadata'];record=meta['generation_record'];signals=meta.get('internal_signals')
                require(t['step']==step and t['task_id']==task and t['rollout_seed']==seed,'Step identity/count')
                require(record['messages']==messages(t['goal'],t['observation'],past,keep=2),'Actor future/different prompt')
                require(meta['shared_model_revision']==config['internal_signature']['model_revision'],'Actor model revision drift')
                require(recorded_seed(meta)==request_seed(task,step,'draft',seed),'Draft random seed drift')
                if past:require(t['observation']==rows[step-1]['actual']['observation'],'Observation not last actual feedback')
                if not gate.needs_internal:require(signals is None and meta['internal_cost']['extraction_seconds']==0,'Unused Internal cost')
                if not gate.needs_external:require(t['external'] is None,'Unused External cost')
                context=GateInput(t['gate']['decision_id'],budget,signals,ExternalInput(**t['resolved_external']),budget,meta['decision_format_valid'])
                expected=gate.decide(context)
                for key in ('call_wm','reason','triggered_signals'):require(t['gate'][key]==expected[key],'Gate not reproducible')
                sim=t['simulation'];review=t['review_metadata']
                require(bool(sim)==expected['call_wm'],'Call/Skip actual dispatch')
                ep_tokens+=meta['usage']['total_tokens'];ep_calls+=1
                if sim:
                    ep_sims+=1;budget-=1
                    require(sim['source']=='ground_truth_environment' and sim['simulator']==profile['engine'],'Wrong WM')
                    require(sim['formal_unchanged'],'Formal environment polluted')
                    require(t['decided_at_ns']<sim['available_at_ns']<t['executed_at_ns'],'Future WM feedback')
                    ep_queries+=sum(sim['simulation_cost'][k] for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls'))
                    if sim['prediction_available']:
                        require(sim['worker_pid']!=sim['formal_pid'],'Shared JVM')
                        require(sim['verified_prefix_steps']==step and sim['simulation_cost']['replay_step_calls']==step,'Incomplete replay/cost')
                        require(sim['simulation_cost']['candidate_step_calls']==1,'Not one-step WM')
                        if t['final_action']==t['draft_action']:
                            require(t['kept_draft_transition_match'] is True,'KEEP state mismatch');keep+=1
                            require(t['formal_after_state_sha256']==sim['candidate_state_sha256'],'KEEP projection mismatch')
                        if review:
                            require(review['generation_record']['messages']==review_messages(t['goal'],t['observation'],past,t['draft_action'],sim),'Private/different Reviewer prompt')
                            require(review['shared_model_revision']==meta['shared_model_revision'] and review['config_sha256']==meta['config_sha256'],'Reviewer weights differ')
                            require(recorded_seed(review)==request_seed(task,step,'review',seed),'Reviewer seed drift')
                            ep_calls+=1;ep_reviews+=1;ep_tokens+=review['usage']['total_tokens']
                    else:
                        require(review is None and t['final_action']==t['draft_action'],'Failed replay used')
                        require(not any(k.startswith('predicted_') for k in sim),'Unverified feedback leaked')
                else:require(review is None and t['final_action']==t['draft_action'],'Skip invoked Reviewer/action correction')
                require(t['budget_after']==budget and budget>=0,'Budget accounting')
                require(t['actual']['action']==t['final_action'],'Formal executed wrong action')
                require(t['decided_at_ns']<t['executed_at_ns'],'Action before Gate')
                require(t['metadata']['internal_cost']['additional_backbone_forward_passes']==0,'Unexpected extra NLL forward')
                ep_queries+=1;past.append(dict(observation=t['observation'],action=t['final_action']))
            require(len(rows)==result['formal_environment_actions'] and len(rows)<=20,'Episode step limit')
            require(result['done'] or len(rows)==20,'Unexpected early episode stop')
            require(result['task_score']==rows[-1]['actual']['score'],'Score extraction')
            require(result['success']==(result['done'] and result['task_score']==100),'Success extraction')
            require(ep_tokens==result['total_tokens']==sum(c['metadata']['usage']['total_tokens'] for c in calls),'Token accounting')
            require(ep_calls==len(calls) and ep_reviews==result['reviewer_calls'],'Actual model request accounting')
            require(ep_sims==result['lookahead_attempts'],'Lookahead accounting')
            if policy=='never':require(ep_sims==0,'Never called WM')
            if policy=='always':require(ep_sims==len(rows),'Always skipped with budget')
            episodes.append(result);traces.extend(rows);tokens+=ep_tokens;call_count+=ep_calls;queries+=ep_queries;sim_count+=ep_sims
    require(all(len(s)==1 for s in initials.values()),'Different initial state across strategies/seeds')
    report=dict(status='engineering_checks_pass',stage=stage,episodes=len(episodes),formal_actions=len(traces),
        model_calls=call_count,llm_tokens=tokens,transition_queries=queries,lookahead_attempts=sim_count,
        verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in traces),
        rejected_lookaheads=sum(bool(t['simulation'] and not t['simulation']['prediction_available']) for t in traces),
        verified_keep_projection_matches=keep,method_hashes_verified=True,evaluation_code_hashes_verified=True,
        identical_initial_states=True,policy_outcomes_used_for_acceptance=False,scientific_effectiveness_established=False)
    if stage=='all':
        ledger={k:0 for k in ('tokens','model_calls','transition_queries','audit_capture_bundles')}
        for i in range(2):
            state=json.loads((run/('worker'+str(i))/'ledger.json').read_text())
            require(state['pending_model'] is None and state['pending_transition'] is None,'Unresolved resource receipt')
            for key in ledger:ledger[key]+=state[key]
        require(ledger['tokens']==tokens and ledger['model_calls']==call_count and ledger['transition_queries']==queries,'Global resource ledger differs')
        require(tokens<=profile['limits']['max_llm_tokens'] and call_count<=profile['limits']['max_model_calls'] and queries<=profile['limits']['max_transition_queries'],'Run budget exceeded')
        report['ledger']=ledger
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',choices=['round1','all'],required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.run,a.stage);atomic(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
