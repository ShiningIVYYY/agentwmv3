#!/usr/bin/env python3
"""Engineering checks and descriptive reviewer audit. No tuning or model calls."""
from collections import Counter,defaultdict
import argparse,csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows,request_seed
from experience.splits import file_hash
from gates.core import RuleGate,GateInput,ExternalInput
from internal_signals.decision import messages
from reflection.replay_review_v2 import ReviewerProtocol,VERSION
from scripts.audit_independent_dev import recorded_seed
from scripts.audit_phase57_replay import feedback_status
from scripts.prepare_reviewer_repair import source_case,events


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def mean(xs):return sum(xs)/len(xs) if xs else None


def utility(values):return dict(positive=sum(x>0 for x in values),zero=sum(x==0 for x in values),negative=sum(x<0 for x in values),mean=mean(values))


def validity(text):
    status=feedback_status(text);low=text.lower()
    if status=='explicit_rejection':
        if 'no known action' in low:return status,'syntax_or_reference_unresolved'
        if any(x in low for x in ('not open',"isn\'t open",'not visible','not accessible')):return status,'state_precondition_or_access'
        return status,'other_explicit_rejection'
    if status=='unknown' and 'which' in low and 'mean' in low:return status,'ambiguity_choice_required'
    if status=='unknown' and 'already open' in low:return status,'state_already_satisfied'
    return status,'accepted_execution_not_semantic_correctness' if status=='accepted_feedback' else 'not_classifiable_from_public_text'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output)
    if out.exists():raise ValueError('Fresh analysis output required')
    complete=json.loads((run/'complete.json').read_text());assert complete['episodes']==15
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    for name,sha in manifest['file_hashes'].items():assert file_hash(ROOT/name)==sha,name
    out.mkdir(parents=True)
    gate=json.loads((ROOT/profile['gate_config']).read_text());allrows=[];summaries=[];calls=read_rows(run/'generation_calls.jsonl');qs=0;captures=0;kept=0;lookaheads=0;source_index=[]
    initial={};first_drafts={}
    for path in sorted((run/'episodes').rglob('complete.json')):
        e=json.loads(path.read_text());rows=read_rows(path.parent/'steps.jsonl');protocol=ReviewerProtocol(e['reviewer_version']);past=[];budget=profile['wm_invocation_budgets'][e['policy']]
        callrows=read_rows(path.parent/'generation_calls.jsonl');calls.extend(callrows);reviewcalls=0
        init=json.loads((path.parent/'initial.json').read_text());initial.setdefault(e['task_id'],set()).add(init['state_sha256'])
        first=rows[0]['metadata']['generation_record'];first_drafts.setdefault(e['task_id'],set()).add(json.dumps([first['prompt_ids'],first['response_ids']]))
        simq=0
        for step,t in enumerate(rows):
            assert t['step']==step and t['reviewer_version']==e['reviewer_version'];meta=t['metadata']
            assert meta['generation_record']['messages']==messages(t['goal'],t['observation'],past,keep=2)
            assert recorded_seed(meta)==request_seed(t['task_id'],step,'draft',0)
            ctx=GateInput(t['gate']['decision_id'],budget,meta.get('internal_signals'),ExternalInput(**t['resolved_external']),budget,meta['decision_format_valid'])
            expected=RuleGate(dict(gate,policy=e['policy'])).decide(ctx)
            for key in ('call_wm','reason','triggered_signals'):assert expected[key]==t['gate'][key]
            sim=t['simulation'];rm=t['review_metadata'];assert bool(sim)==expected['call_wm']
            assert t['decided_at_ns']<t['executed_at_ns']
            if sim:
                budget-=1;lookaheads+=1;assert sim['formal_unchanged'] and sim['source']=='ground_truth_environment'
                simq+=sum(sim['simulation_cost'][key] for key in ('initialization_step_calls','replay_step_calls','candidate_step_calls'))
                assert sim['available_at_ns']<t['executed_at_ns'] and t['decided_at_ns']<sim['available_at_ns']
                if sim['prediction_available']:
                    assert sim['formal_pid']!=sim['worker_pid'];assert sim['verified_prefix_steps']==step
                    assert sim['simulation_cost']['candidate_step_calls']==1
                    if t['final_action']==t['draft_action']:assert t['kept_draft_transition_match'] is True;kept+=1
                else:assert rm is None and t['final_action']==t['draft_action']
            else:assert rm is None and t['final_action']==t['draft_action']
            if rm:
                reviewcalls+=1;assert sim['prediction_available']
                assert rm['generation_record']['messages']==protocol.messages(t['goal'],t['observation'],past,t['draft_action'],sim)
                assert t['review']['parsed']==protocol.parse(t['review']['raw'],t['draft_action'])
                assert rm['reviewer_version']==e['reviewer_version'] and rm['reviewer_max_new_tokens']==protocol.max_new_tokens
                assert recorded_seed(rm)==request_seed(t['task_id'],step,'review',0)
                assert rm['usage']['completion_tokens']<=protocol.max_new_tokens
            assert t['budget_after']==budget;assert t['actual']['action']==t['final_action']
            if past:assert t['observation']==rows[step-1]['actual']['observation']
            assert 'internal_signals' in meta if e['policy']=='full' else 'internal_signals' not in meta
            assert bool(t['external'])==(e['policy']=='full')
            allrows.append(t);past.append(dict(observation=t['observation'],action=t['final_action']))
        assert e['task_score']==rows[-1]['actual']['score'] and e['success']==(rows[-1]['actual']['done'] and rows[-1]['actual']['score']==100)
        assert rows[-1]['actual']['done'] or len(rows)==20
        assert len(callrows)==len(rows)+reviewcalls
        assert e['total_tokens']==sum(x['metadata']['usage']['total_tokens'] for x in callrows)
        e['total_transition_queries']=1+len(rows)+simq;e['extra_lookahead_queries']=simq;qs+=e['total_transition_queries'];summaries.append(e)
        for name in ('initial.json','steps.jsonl','generation_calls.jsonl','complete.json'):source_index.append(dict(path=str((path.parent/name).relative_to(run)),sha256=file_hash(path.parent/name)))
    assert len(summaries)==15 and all(len(s)==1 for s in initial.values()) and all(len(s)==1 for s in first_drafts.values())
    offline=read_rows(run/'offline_reviews.jsonl');offline_groups=[]
    for panel in ('old_debug','phase59_format_failure','phase59_full_quality'):
        rs=[r for r in offline if r['ref']['panel']==panel];old=[source_case(r['ref'])[0]['review_metadata'] for r in rs]
        original_parses=[source_case(r['ref'])[0]['review']['parsed'] for r in rs]
        weighted=sum(len(r['ref'].get('source_events',[r['ref']])) for r in rs)
        offline_groups.append(dict(panel=panel,unique_requests=len(rs),source_events=weighted,
            original_errors=sum(q['decision']=='INVALID' for q in original_parses),fixed_errors=sum(r['parsed']['decision']=='INVALID' for r in rs),
            original_event_errors=sum(len(r['ref'].get('source_events',[r['ref']])) for r,q in zip(rs,original_parses) if q['decision']=='INVALID'),
            fixed_event_errors=sum(len(r['ref'].get('source_events',[r['ref']])) for r in rs if r['parsed']['decision']=='INVALID'),
            original_mean_output_tokens=mean([m['usage']['completion_tokens'] for m in old]),fixed_mean_output_tokens=mean([r['metadata']['usage']['completion_tokens'] for r in rs]),
            original_mean_total_tokens=mean([m['usage']['total_tokens'] for m in old]),fixed_mean_total_tokens=mean([r['metadata']['usage']['total_tokens'] for r in rs]),
            original_mean_generation_seconds=mean([m['internal_cost']['generation_seconds'] for m in old]),fixed_mean_generation_seconds=mean([r['metadata']['internal_cost']['generation_seconds'] for r in rs]),
            latency_comparison_uses_historical_original_hardware_load=True,original_keep_changed=sum(q['decision']=='KEEP' and r['parsed']['final_action']!=source_case(r['ref'])[0]['draft_action'] for r,q in zip(rs,original_parses))))
    comparisons=read_rows(run/'one_step_comparisons.jsonl');assert len(comparisons)==32
    controls=[]
    for c in comparisons:
        bs=c['branches'];assert len({b['before_sha256'] for b in bs})==1
        assert all(b['before_sha256']==source_case(c['ref'])[0]['simulation']['checked_state_sha256'] for b in bs)
        qs+=sum(b['transition_queries'] for b in bs)
        controls.append(dict(case=c['ref']['prompt_id'],panel=c['ref']['panel'],task_id=c['ref']['task_id'],seed=c['ref']['seed'],step=c['ref']['step'],
            draft_action=bs[0]['action'],original_action=bs[1]['action'],fixed_action=bs[2]['action'],
            draft_score=bs[0]['actual']['score'],original_score=bs[1]['actual']['score'],fixed_score=bs[2]['actual']['score'],
            draft_feedback=bs[0]['actual']['observation'],original_feedback=bs[1]['actual']['observation'],fixed_feedback=bs[2]['actual']['observation'],
            fixed_minus_original=c['fixed_minus_original'],original_minus_skip=c['original_minus_skip'],fixed_minus_skip=c['fixed_minus_skip']))
    control_summaries=[]
    for panel in ('old_debug','phase59_full_quality'):
        cs=[c for c in comparisons if c['ref']['panel']==panel];base=dict(panel=panel,cases=len(cs),same_state_projection_verified=True)
        for variant,i in [('original',1),('fixed',2)]:
            changed=[c for c in cs if c['branches'][i]['action']!=c['branches'][0]['action']]
            base[variant]=dict(actual_revisions=len(changed),feedback=dict(Counter(feedback_status(c['branches'][i]['actual']['observation']) for c in changed)),
                one_step_vs_skip=utility([c['branches'][i]['actual']['score']-c['branches'][0]['actual']['score'] for c in cs]),
                successful_draft_replaced=sum(c['branches'][0]['actual']['done'] and c['branches'][0]['actual']['score']==100 and c['branches'][i]['action']!=c['branches'][0]['action'] for c in cs),
                successful_draft_cases=sum(c['branches'][0]['actual']['done'] and c['branches'][0]['actual']['score']==100 for c in cs))
        base['fixed_minus_original']=utility([c['fixed_minus_original'] for c in cs]);control_summaries.append(base)
    groups=[]
    for policy,reviewer in [('never','original'),('always','original'),('always',VERSION),('full','original'),('full',VERSION)]:
        es=[e for e in summaries if e['policy']==policy and e['reviewer_version']==reviewer];ts=[t for t in allrows if t['policy']==policy and t['reviewer_version']==reviewer];reviews=[t for t in ts if t['review_metadata']];changed=[t for t in reviews if t['draft_action']!=t['final_action']]
        groups.append(dict(policy=policy,reviewer=reviewer,episodes=len(es),mean_score=mean([e['task_score'] for e in es]),successes=sum(e['success'] for e in es),
            success_rate=mean([e['success'] for e in es]),formal_steps=len(ts),wm_calls=sum(t['simulation'] is not None for t in ts),verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in ts),
            reviewer_calls=len(reviews),format_errors=sum(t['review']['parsed']['decision']=='INVALID' for t in reviews),
            reviewer_decisions=dict(Counter(t['review']['parsed']['decision'] for t in reviews)),
            actual_revisions=len(changed),revision_feedback=dict(Counter(feedback_status(t['actual']['observation']) for t in changed)),
            revision_semantic_correctness='unknown_unless_supported_by_separate_controlled_evidence',
            total_tokens=sum(e['total_tokens'] for e in es),input_tokens=sum(e['input_tokens'] for e in es),output_tokens=sum(e['output_tokens'] for e in es),
            reviewer_tokens=sum(t['review_metadata']['usage']['total_tokens'] for t in reviews),actor_tokens=sum(t['metadata']['usage']['total_tokens'] for t in ts),
            episode_seconds=sum(e['total_seconds_including_cleanup'] for e in es),reviewer_seconds=sum(t['review_metadata']['internal_cost']['generation_seconds'] for t in reviews),
            lookahead_seconds=sum(t['simulation']['simulation_latency'] for t in ts if t['simulation']),
            total_transition_queries=sum(e['total_transition_queries'] for e in es),extra_lookahead_queries=sum(e['extra_lookahead_queries'] for e in es),
            peak_torch_allocated_bytes=max(m['internal_cost']['peak_allocated_bytes'] for t in ts for m in [t['metadata'],t['review_metadata']] if m)))
    for c in calls:
        m=c['metadata'];assert m['shared_model_revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        assert m['config_sha256']=='cad3791bc197e7f95baa16837d072ab26b68b6f76cb12cf164e0ccfcaf71a7a6'
    recovery=manifest.get('recovery',{})
    qs+=recovery.get('inherited_failed_diagnostic_transition_queries',0)+recovery.get('additional_probe_transition_queries',0)
    ledger=json.loads((run/'ledger.json').read_text());assert ledger['tokens']==sum(c['metadata']['usage']['total_tokens'] for c in calls)
    assert ledger['model_calls']==len(calls) and ledger['transition_queries']==qs
    assert ledger['pending_model'] is None and ledger['pending_transition'] is None
    # Full58 classification keeps execution, semantic utility and format on separate axes.
    original,_=events();full58=[]
    for e in original:
        t=e['t']
        if e['legacy'] or t['policy']!='full' or t['final_action']==t['draft_action']:continue
        status,subtype=validity(t['actual']['observation']);delta=t['one_step_score_delta_actual_minus_trial_draft']
        full58.append(dict(ref=e['ref'],goal=t['goal'],observation=t['observation'],draft=t['draft_action'],final=t['final_action'],
            feedback=t['actual']['observation'],execution_status=status,execution_subtype=subtype,trial_score=t['simulation']['predicted_score'],actual_score=t['actual']['score'],
            one_step_score_difference=delta,goal_relevance='not_automatically_inferred',correct_revision='not_established_by_format_or_acceptance',
            actual_harm_observed=delta<0,original_reason=t['review']['parsed'].get('reason')))
    assert len(full58)==58
    atomic(out/'original_full58_classifications.json',full58)
    atomic(out/'summary.json',dict(episode_groups=groups,offline_groups=offline_groups,controlled_one_step=control_summaries,
        ledger=ledger,engineering_audit=dict(status='pass',episodes=15,formal_steps=len(allrows),lookahead_attempts=lookaheads,exact_keep_projections=kept,
            identical_initial_projections=True,identical_initial_actor_tokens=True,causality_prompts_gate_budgets_and_costs=True),
        isolated_goal_generalization=False,independent_new_goal_groups=0,old_results_preserved=True,no_gate_threshold_memory_change=True,
        model_load=json.loads((run/'model.json').read_text()),complete=complete,monetary_cost=None,flops=None,
        query_scope='new online+offline controlled native transitions; original historic queries excluded; state captures separate',
        downstream_causal_utility_established=False))
    csv_write(out/'episodes.csv',summaries);csv_write(out/'controlled_one_step.csv',controls)
    atomic(out/'paired_episode_differences.json',[dict(task_id=row['task_id'],policy=policy,
        fixed_minus_original_score=next(e['task_score'] for e in summaries if e['task_id']==row['task_id'] and e['policy']==policy and e['reviewer_version']==VERSION)-next(e['task_score'] for e in summaries if e['task_id']==row['task_id'] and e['policy']==policy and e['reviewer_version']=='original')) for row in profile['tasks'] for policy in ('always','full')])
    atomic(out/'provenance.json',dict(script_sha256=file_hash(Path(__file__)),run_manifest_sha256=file_hash(run/'manifest.json'),raw_file_index=source_index,
        all_frozen_source_hashes_pass=True,all_model_and_transition_receipts_match_ledger=True))
    print(json.dumps(dict(groups=groups,offline=offline_groups,controlled=control_summaries,ledger=ledger),indent=2))


if __name__=='__main__':main()
