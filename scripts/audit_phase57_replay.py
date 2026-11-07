#!/usr/bin/env python3
"""Audit completed true replay traces/costs; does not select a policy or threshold."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experience.splits import file_hash,validate_manifest
from gates.core import RuleGate,GateInput,ExternalInput
from internal_signals.decision import messages
from reflection.replay_review import review_messages


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def require(value,reason):
    if not value:raise ValueError(reason)


def feedback_status(text):
    # Conservative public-feedback diagnosis, not an oracle action-validity flag.
    if re.search(r'(?i)(no known action|unknown action|not a valid action|you can.t|you cannot|not open|not visible|not accessible|could not find)',text):
        return 'explicit_rejection'
    if text.startswith(('You open ','You close ','You move ','You go ','You focus ','You pick ','You activate ',
                        'You deactivate ','You connect ','You disconnect ','You decide to wait ',
                        'This room is called ','In your inventory, you see:')):
        return 'accepted_feedback'
    if re.fullmatch(r'The .+ is now (?:open|closed)\.',text.strip()):
        return 'accepted_feedback'
    return 'unknown'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output)
    if out.exists():raise ValueError('New audit output required')
    require(json.loads((run/'progress.json').read_text())['stage']=='complete','Run incomplete; no final metrics')
    manifest=json.loads((run/'manifest.json').read_text())
    config=json.loads((run/'gate_config.json').read_text())
    root_summary=json.loads((run/'summary.json').read_text())
    episodes=load_rows(run/'episodes.jsonl');traces=load_rows(run/'evaluation_steps.jsonl')
    calibration=load_rows(run/'calibration_steps.jsonl');calls=load_rows(run/'generation_calls.jsonl')
    splits=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());validate_manifest(splits,require_native=True)
    for key,name in [('split_sha256','docs/phase3/split_manifest.json'),('memory_sha256','docs/phase5/memory_snapshot_v2.json'),
                     ('paper_protocol_sha256','docs/PAPER_ALIGNED_EVAL.md')]:
        require(manifest[key]==file_hash(ROOT/name),'Changed protected input: '+name)
    for name,expected in manifest['source_hashes'].items():
        require(expected==file_hash(ROOT/name),'Changed executed source: '+name)
    evaluation={r['task_id'] for r in manifest['evaluation_tasks']}
    calids={r['task_id'] for r in manifest['calibration_tasks']}
    require(not evaluation&calids,'Calibration ID leakage')
    require(len(episodes)==4*len(evaluation),'Missing complete episodes')
    require({t['task_id'] for t in traces}==evaluation,'Unexpected evaluation ID')
    require({t['task_id'] for t in calibration}==calids,'Unexpected calibration ID')
    model_revision=manifest['model_config']['model_revision']
    policy_audits=[];corrections=[];representative=[]
    for policy in ('never','always','internal','full'):
        policy_rows=[t for t in traces if t['policy']==policy]
        bytask={task:[t for t in policy_rows if t['task_id']==task] for task in evaluation}
        gate=RuleGate(dict(config,policy=policy))
        reasons=Counter();mismatches=Counter();feedback=Counter();review_decisions=Counter()
        simcost=Counter();generations=[];retrieval_seconds=0.;gate_seconds=0.;extraction_seconds=0.;simulation_seconds=0.;rule_judge_seconds=0.
        external_only_calls=0;coverage=Counter();conflict=Counter();sufficiency=Counter()
        kept=0
        for task,rows in bytask.items():
            past=[];budget=manifest['pilot_profile']['wm_invocation_budgets'][policy]
            for step,t in enumerate(rows):
                require(t['step']==step,'Missing/duplicate formal step')
                require(t['metadata']['generation_record']['messages']==messages(t['goal'],t['observation'],past,keep=2),
                        'Actor prompt differs from observable committed prefix')
                if past:require(t['observation']==rows[step-1]['actual']['observation'],'Current observation is not last real feedback')
                meta=t['metadata'];generations.append(meta)
                require(meta['shared_model_revision']==model_revision,'Actor revision drift')
                if not gate.needs_internal:
                    require('internal_signals' not in meta and meta['internal_cost']['extraction_seconds']==0,'Unused NLL charged')
                context=GateInput(t['gate']['decision_id'],budget,meta.get('internal_signals'),ExternalInput(**t['resolved_external']),budget,meta['decision_format_valid'])
                expected=gate.decide(context)
                if policy=='full':
                    internal_same_state=RuleGate(dict(config,policy='internal')).decide(context)
                    external_only_calls+=bool(expected['call_wm'] and not internal_same_state['call_wm'])
                for key in ('call_wm','reason','triggered_signals'):
                    require(expected[key]==t['gate'][key],'Gate not reproducible from pre-action signals')
                sim=t['simulation'];review=t['review_metadata']
                require(bool(sim)==t['gate']['call_wm'],'Call/Skip dispatch mismatch')
                if sim:
                    budget-=1;require(sim['formal_unchanged'],'Formal environment polluted')
                    require(sim['source']=='ground_truth_environment','Not a real simulator transition')
                    require(sim['available_at_ns']<t['executed_at_ns'] and t['decided_at_ns']<sim['available_at_ns'],'Future feedback leakage')
                    for k,v in sim['simulation_cost'].items():simcost[k]+=v
                    simulation_seconds+=sim['simulation_latency']
                    if sim['prediction_available']:
                        require(sim['worker_pid']!=sim['formal_pid'],'Shared JVM')
                        require(sim['verified_prefix_steps']==step,'Incomplete verified replay')
                        require(sim['simulation_cost']['candidate_step_calls']==1,'Not one-step lookahead')
                        if review:
                            require(review['generation_record']['messages']==review_messages(t['goal'],t['observation'],past,t['draft_action'],sim),
                                    'Reviewer received private or different evidence')
                            require(review['shared_model_revision']==meta['shared_model_revision'] and review['config_sha256']==meta['config_sha256'],'Reviewer weights/config differ')
                            generations.append(review);review_decisions[t['review']['parsed']['decision']]+=1
                        if t['final_action']==t['draft_action']:
                            require(t['kept_draft_transition_match'] is True,'Accepted KEEP transition mismatch');kept+=1
                        else:
                            corrections.append(dict(policy=policy,task_id=task,step=step,draft=t['draft_action'],final=t['final_action'],
                                trial_feedback=sim['predicted_observation'],actual_feedback=t['actual']['observation'],
                                immediate_score_delta=t['actual']['score']-sim['predicted_score'],reason=t['review']['parsed'].get('reason')))
                    else:
                        require(review is None and t['final_action']==t['draft_action'],'Failed reconstruction used for review/action change')
                        require(not any(k.startswith('predicted_') for k in sim),'Failed simulation exposes unverified prediction')
                        require(sim['simulation_cost']['candidate_step_calls']==0,'Unexpected candidate after failed reconstruction')
                        mismatches.update(sim.get('mismatch',{}).get('fields',{}).keys())
                else:
                    require(review is None and t['final_action']==t['draft_action'],'Skip invoked Reviewer or changed draft')
                require(t['budget_after']==budget,'Budget count differs from actual attempts')
                require(t['actual']['action']==t['final_action'],'Executed action differs')
                require(t['decided_at_ns']<t['executed_at_ns'],'Action before Gate')
                reasons[t['gate']['reason']]+=1;feedback[feedback_status(t['actual']['observation'])]+=1
                extraction_seconds+=meta['internal_cost']['extraction_seconds'];gate_seconds+=t['gate']['gate_latency']
                if t['external']:
                    require(policy=='full','Unexpected external computation')
                    retrieval_seconds+=t['external'].get('retrieval_latency',0.)
                    rule_judge_seconds+=t['external'].get('judge_latency',0.)
                    coverage[t['external']['coverage']]+=1;conflict[t['external']['conflict']]+=1;sufficiency[t['external']['sufficiency']]+=1
                past.append(dict(observation=t['observation'],action=t['final_action']))
            episode=next(e for e in episodes if e['policy']==policy and e['task_id']==task)
            require(episode['task_score']==rows[-1]['actual']['score'],'Score extraction differs')
            require(episode['formal_environment_actions']==len(rows) and len(rows)<=20,'Formal action count differs')
            require(episode['done'] or len(rows)==20,'Episode ended early without cause')
            require(episode['total_tokens']==sum(t['metadata']['usage']['total_tokens']+(t['review_metadata']['usage']['total_tokens'] if t['review_metadata'] else 0) for t in rows),'Double/omitted token accounting')
        if policy=='never':require(not any(t['simulation'] for t in policy_rows),'Never actually called WM')
        if policy=='always':require(all(t['gate']['call_wm'] for t in policy_rows),'Always skipped with available budget')
        if policy_rows:
            representative.append(policy_rows[0])
        policy_audits.append(dict(policy=policy,actor_calls=len(policy_rows),reviewer_calls=sum(t['review_metadata'] is not None for t in policy_rows),
            internal_extractions=sum('internal_signals' in t['metadata'] for t in policy_rows),retrieval_calls=sum(t['external'] is not None for t in policy_rows),
            judge_calls=0,learned_wm_calls=0,gate_calls=len(policy_rows),gate_seconds=gate_seconds,
            actor_generation_seconds=sum(t['metadata']['internal_cost']['generation_seconds'] for t in policy_rows),
            reviewer_generation_seconds=sum(t['review_metadata']['internal_cost']['generation_seconds'] for t in policy_rows if t['review_metadata']),
            internal_extraction_seconds=extraction_seconds,retrieval_seconds=retrieval_seconds,simulation_seconds=simulation_seconds,
            external_rule_judge_seconds=rule_judge_seconds,external_only_calls_at_same_recorded_state=external_only_calls,
            external_coverage=dict(coverage),external_conflict=dict(conflict),external_sufficiency=dict(sufficiency),
            missing_internal_steps=sum(t['metadata'].get('internal_signals',{}).get('missing_span',False) for t in policy_rows),
            invalid_draft_formats=sum(not t['metadata']['decision_format_valid'] for t in policy_rows),
            left_truncated_draft_prompts=sum(t['metadata']['generation_record']['left_truncated_prompt_tokens']>0 for t in policy_rows),
            simulation_cost=dict(simcost),reconstruction_mismatch_fields=dict(mismatches),gate_reasons=dict(reasons),
            public_feedback_execution_status=dict(feedback),review_decisions=dict(review_decisions),verified_keep_matches=kept,
            peak_torch_allocated_bytes=max(m['internal_cost'].get('peak_allocated_bytes') or 0 for m in generations)))
    eval_calls=[c for c in calls if c['metadata']['role'] in ('draft','review')]
    require(len(eval_calls)==sum(p['actor_calls']+p['reviewer_calls'] for p in policy_audits),'Model request count differs from eager log')
    require(sum(c['metadata']['usage']['total_tokens'] for c in eval_calls)==sum(e['total_tokens'] for e in episodes),'Usage accounting differs from eager log')
    deltas=[c['immediate_score_delta'] for c in corrections]
    replace_feedback=Counter(feedback_status(c['actual_feedback']) for c in corrections)
    all_reviews=[t for t in traces if t['review_metadata']]
    report=dict(status='engineering_trace_audit_pass',policies=policy_audits,episodes=len(episodes),formal_actions=len(traces),
        exact_full_hidden_state_equivalence=False,independent_raw_latency_scope='episode seconds exclude formal JVM construction/disposal; whole-run wall includes them',
        replacement_one_step_score_delta=dict(positive=sum(v>0 for v in deltas),zero=sum(v==0 for v in deltas),negative=sum(v<0 for v in deltas)),
        downstream_counterfactual_utility_established=False,scientific_effectiveness_established=False,
        actual_replacements=len(corrections),replacement_feedback_status=dict(replace_feedback),
        replace_same_action_proposals=sum(t['review']['parsed']['decision']=='REPLACE' and t['final_action']==t['draft_action'] for t in all_reviews),
        review_format_errors=sum(t['review']['parsed']['decision']=='INVALID' for t in all_reviews),
        feedback_validity_scope='Conservative public-feedback execution acceptance; Unknown retained; not task correctness or exhaustive native validity',
        calibration_reference_steps=len(calibration),calibration_reused_without_new_generation=bool(root_summary['calibration_source']),
        source_hashes_verified=True,protected_inputs_verified=True,early_model_call_logs=len(calls),
        model_call_tokens=sum(c['metadata']['usage']['total_tokens'] for c in calls),
        final_run_summary=root_summary)
    out.mkdir(parents=True)
    for name,value in [('audit.json',report),('corrections.json',corrections),('representative_traces.json',representative)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('policies','final_run_summary')},indent=2))


if __name__=='__main__':main()
