"""Audit and aggregate the real Experience-Only supplement; no tuning or inference."""
import argparse,csv,json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows,request_seed
from scripts.run_external_only import frozen
from scripts.audit_independent_dev import recorded_seed
from scripts.analyze_independent_dev import interval,rejection_reason
from scripts.audit_phase57_replay import feedback_status
from internal_signals.decision import messages
from reflection.replay_review import review_messages,parse_review
from gates.core import RuleGate,GateInput,ExternalInput
from experience.splits import file_hash


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def analyze(run,out):
    if out.exists():raise ValueError('New analysis output required')
    assert (run/'complete.json').exists()
    m=json.loads((run/'manifest.json').read_text());p=m['profile'];old,_=frozen(p)
    for n,h in m['new_source_hashes'].items():assert file_hash(ROOT/n)==h,n
    gate_config=json.loads((ROOT/p['gate_config']).read_text());memory=json.loads((ROOT/p['memory_snapshot']).read_text())
    evidence_ids={c['case_id'] for c in memory['cases']};out.mkdir(parents=True);(out/'decisions').mkdir()
    episodes=[];traces=[];index=[];prefix_checks=[];total_queries=0;keep_matches=0;model_calls=0;tokens=0
    for g in m['schedule']:
        task=g['task_id'];seed=g['seed'];ep=run/'episodes'/task/f'seed{seed}'/'experience'
        base=ROOT/'outputs/phase59_independent_v2/episodes'/task/f'seed{seed}'/'never'
        e=json.loads((ep/'complete.json').read_text());initial=json.loads((ep/'initial.json').read_text())
        baseline_initial=json.loads((base/'initial.json').read_text());baseline=read_rows(base/'steps.jsonl')
        assert initial['state_sha256']==baseline_initial['state_sha256'] and initial['simulator']==p['engine']
        assert e['steps_sha256']==file_hash(ep/'steps.jsonl') and e['generation_calls_sha256']==file_hash(ep/'generation_calls.jsonl')
        ts=read_rows(ep/'steps.jsonl');calls=read_rows(ep/'generation_calls.jsonl');past=[];budget=5;epqueries=1;prefix=True;matched=0
        for step,t in enumerate(ts):
            meta=t['metadata'];sim=t['simulation'];rev=t['review_metadata'];ext=t['resolved_external']
            assert t['task_id']==task and t['rollout_seed']==seed and t['step']==step
            assert meta.get('internal_signals') is None and meta['internal_cost']['extraction_seconds']==0
            assert not meta['signal_computation']['raw_logits_retained'] and not meta['signal_computation']['internal']
            assert meta['shared_model_revision']==gate_config['internal_signature']['model_revision']
            assert meta['config_sha256']==gate_config['internal_signature']['config_sha256']
            assert recorded_seed(meta)==request_seed(task,step,'draft',seed)
            assert meta['generation_record']['messages']==messages(t['goal'],t['observation'],past,keep=2)
            assert t['external'] is not None and t['external']['additional_tokens']==0 and t['external']['judge_model_calls']==0
            assert set(ext['evidence_ids'])<=evidence_ids
            assert t['external']['query_at_ns']<=t['decided_at_ns']<t['executed_at_ns']
            ctx=GateInput(t['gate']['decision_id'],budget,None,ExternalInput(**ext),budget,meta['decision_format_valid'])
            expected=RuleGate(dict(gate_config,policy='experience')).decide(ctx)
            for k in ['call_wm','reason','triggered_signals']:assert t['gate'][k]==expected[k],k
            assert bool(sim)==expected['call_wm']
            if prefix:
                assert step<len(baseline)
                b=baseline[step]
                assert t['observation']==b['observation']
                assert meta['generation_record']['response_ids']==b['metadata']['generation_record']['response_ids'],'Shared Never prefix differs'
                matched+=1
            if sim:
                budget-=1
                assert sim['formal_unchanged'] and sim['source']=='ground_truth_environment' and sim['simulator']==p['engine']
                assert t['decided_at_ns']<sim['available_at_ns']<t['executed_at_ns']
                epqueries+=sum(sim['simulation_cost'][k] for k in ['initialization_step_calls','replay_step_calls','candidate_step_calls'])
                if sim['prediction_available']:
                    assert sim['formal_pid']!=sim['worker_pid'] and sim['verified_prefix_steps']==step
                    if t['final_action']==t['draft_action']:
                        assert t['kept_draft_transition_match'] is True;keep_matches+=1
                    if rev:
                        assert rev['generation_record']['messages']==review_messages(t['goal'],t['observation'],past,t['draft_action'],sim)
                        assert recorded_seed(rev)==request_seed(task,step,'review',seed)
                        assert rev['shared_model_revision']==meta['shared_model_revision'] and rev['config_sha256']==meta['config_sha256']
                        assert t['review']['parsed']==parse_review(t['review']['raw'],t['draft_action'])
                        assert rev['usage']['completion_tokens']<=512
                else:assert rev is None and t['final_action']==t['draft_action']
            else:assert rev is None and t['final_action']==t['draft_action']
            assert t['actual']['action']==t['final_action'] and t['budget_after']==budget>=0
            if prefix and t['final_action']==t['draft_action']:
                assert all(t['actual'][k]==baseline[step]['actual'][k] for k in ['observation','reward','score','done','moves'])
            elif prefix:prefix=False
            if sim and t['final_action']!=t['draft_action']:prefix=False
            past.append(dict(observation=t['observation'],action=t['final_action']));epqueries+=1
        assert e['done'] or len(ts)==20
        assert len(ts)==e['formal_environment_actions']<=20 and e['lookahead_attempts']<=5
        assert e['task_score']==ts[-1]['actual']['score'] and e['success']==(e['done'] and e['task_score']==100)
        usages=[t['metadata']['usage'] for t in ts]+[t['review_metadata']['usage'] for t in ts if t['review_metadata']]
        assert e['total_tokens']==sum(u['total_tokens'] for u in usages)==sum(c['metadata']['usage']['total_tokens'] for c in calls)
        assert len(calls)==len(usages)
        e.update(total_transition_queries=epqueries,actor_calls=len(ts));episodes.append(e);traces.extend(ts);total_queries+=epqueries;model_calls+=len(calls);tokens+=e['total_tokens']
        prefix_checks.append(dict(task_id=task,seed=seed,matched_actor_decisions_before_first_changed_action=matched,initial_state_equal=True,never_prefix_exact=True))
        compact=[]
        for t in ts:
            compact.append({k:t[k] for k in ['policy','task_id','rollout_seed','step','goal','observation','draft_output','draft_action','external','resolved_external','gate','simulation','review','final_action','actual','kept_draft_transition_match']})
        atomic(out/'decisions'/f'{task}-seed{seed}.json',compact)
        for name in ['steps.jsonl','generation_calls.jsonl','initial.json','complete.json']:
            index.append(dict(path=str((ep/name).relative_to(run)),sha256=file_hash(ep/name)))
    ledger={k:sum(json.loads((run/f'worker{i}'/'ledger.json').read_text())[k] for i in range(2)) for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']}
    assert ledger['tokens']==tokens and ledger['model_calls']==model_calls and ledger['transition_queries']==total_queries
    for i in range(2):
        l=json.loads((run/f'worker{i}'/'ledger.json').read_text());assert l['pending_model'] is None and l['pending_transition'] is None
    assert len(episodes)==24 and len({(e['task_id'],e['rollout_seed']) for e in episodes})==24
    seeds=[dict(seed=s,instances=12,mean_task_score=statistics.mean(e['task_score'] for e in episodes if e['rollout_seed']==s),success_rate=statistics.mean(int(e['success']) for e in episodes if e['rollout_seed']==s)) for s in [0,1]]
    sims=[t['simulation'] for t in traces if t['simulation']];changed=[t for t in traces if t['draft_action']!=t['final_action']]
    row=dict(policy='experience',instances=24,mean_task_score=statistics.mean(e['task_score'] for e in episodes),success_rate=statistics.mean(int(e['success']) for e in episodes),seeds=seeds,
        score_seed_sample_sd=statistics.stdev(s['mean_task_score'] for s in seeds),score_resampling_interval=interval(episodes,'task_score',p['analysis_plan']),
        formal_steps=len(traces),lookahead_attempts=len(sims),successful_lookaheads=sum(s['prediction_available'] for s in sims),rejected_lookaheads=sum(not s['prediction_available'] for s in sims),
        reviewer_calls=sum(e['reviewer_calls'] for e in episodes),revised_actions=len(changed),revision_execution_feedback=dict(Counter(feedback_status(t['actual']['observation']) for t in changed)),
        revision_rejection_reasons=dict(Counter(rejection_reason(t['actual']['observation'],t['actual']['score']) for t in changed)),
        review_decisions=dict(Counter(t['review']['parsed']['decision'] for t in traces if t['review_metadata'])),total_tokens=tokens,input_tokens=sum(e['input_tokens'] for e in episodes),output_tokens=sum(e['output_tokens'] for e in episodes),
        transition_queries=total_queries,internal_extractions=0,retrieval_calls=len(traces),retrieval_seconds=sum(t['external']['retrieval_latency'] for t in traces),
        rule_judge_seconds=sum(t['external']['judge_latency'] for t in traces),llm_judge_calls=0,
        coverage=dict(Counter(t['resolved_external']['coverage'] for t in traces)),conflict=dict(Counter(t['resolved_external']['conflict'] for t in traces)),sufficiency=dict(Counter(t['resolved_external']['sufficiency'] for t in traces)),
        gate_reasons=dict(Counter(t['gate']['reason'] for t in traces)),simulation_seconds=sum(s['simulation_latency'] for s in sims),
        total_episode_seconds=sum(e['total_seconds_including_cleanup'] for e in episodes),
        actor_generation_seconds=sum(t['metadata']['internal_cost']['generation_seconds'] for t in traces),reviewer_generation_seconds=sum(t['review_metadata']['internal_cost']['generation_seconds'] for t in traces if t['review_metadata']))
    oldrows=json.loads((ROOT/'docs/phase59/results/policy_summary.json').read_text());summary=oldrows+[row]
    baseline_es=json.loads((ROOT/'docs/phase59/results/episodes.json').read_text());pairs=[]
    for e in episodes:
        for olde in baseline_es:
            if olde['task_id']==e['task_id'] and olde['rollout_seed']==e['rollout_seed']:
                pairs.append(dict(task_id=e['task_id'],rollout_seed=e['rollout_seed'],baseline=olde['policy'],external_score=e['task_score'],baseline_score=olde['task_score'],score_delta=e['task_score']-olde['task_score'],token_delta=e['total_tokens']-olde['total_tokens']))
    csv_write(out/'episodes.csv',episodes);csv_write(out/'paired_differences.csv',pairs);atomic(out/'episodes.json',episodes);atomic(out/'policy_summary.json',summary)
    atomic(out/'audit.json',dict(status='pass',episodes=24,ledger=ledger,initial_and_shared_never_prefix_checks=prefix_checks,verified_KEEP_matches=keep_matches,no_internal_computation=True,original_reviewer_verified=True,old_method_hashes_verified=True,no_test_rollouts=True,no_downstream_causal_utility_claim=True))
    atomic(out/'raw_index.json',index);atomic(out/'analysis_provenance.json',dict(script_sha256=file_hash(Path(__file__)),run_manifest_sha256=file_hash(run/'manifest.json'),old_summary_sha256=file_hash(ROOT/'docs/phase59/results/policy_summary.json')))
    print(json.dumps(row,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(Path(a.run),Path(a.output))
