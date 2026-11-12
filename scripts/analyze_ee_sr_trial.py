#!/usr/bin/env python3
"""Descriptive retrospective Actor comparison; incomplete formats never counted as scored episodes."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import read_rows,atomic,request_seed
from experience.splits import file_hash
from internal_signals.ee_sr_prompt import prompt,parse_action


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sr-run',default='outputs/ee_sr_actor_trial_v1');ap.add_argument('--control-run',default='outputs/ee_qwen35_native_control_v1');ap.add_argument('--output',required=True);a=ap.parse_args()
    sr=Path(a.sr_run);control=Path(a.control_run);out=Path(a.output)
    if out.exists():raise ValueError('Fresh analysis output required')
    assert json.loads((sr/'complete.json').read_text())['episodes']==6
    assert json.loads((control/'complete.json').read_text())['status']=='complete'
    m=json.loads((sr/'manifest.json').read_text());profile=m['profile'];source=json.loads((ROOT/profile['source_prompt']).read_text())
    for n,h in m['code_hashes'].items():assert file_hash(ROOT/n)==h,n
    for n,h in profile['protected_hashes'].items():assert file_hash(ROOT/n)==h,n
    out.mkdir(parents=True);rows=[];examples=[];calls=read_rows(sr/'generation_calls.jsonl');primary=[c for c in calls if c['identity']['role']=='actor'];sigs=[]
    for task in profile['tasks']:
        tid=task['task_id'];expected=json.loads((ROOT/'outputs/phase510_reviewer_repair_v4/episodes'/tid/'never-original/initial.json').read_text())['state_sha256']
        for mode in ['structured','native_h2']:
            p=sr/'episodes'/tid/mode;e=json.loads((p/'complete.json').read_text());assert json.loads((p/'initial.json').read_text())['state_sha256']==expected
            ts=read_rows(p/'steps.jsonl') if (p/'steps.jsonl').exists() else [];history=[]
            for t in ts:
                ps=prompt(mode,t['goal'],t['observation'],history,source);meta=t['metadata'];assert ps==meta['generation_record']['messages']
                assert parse_action(t['raw'])['action']==t['action']==t['actual']['action']
                assert t['actor_seed']==request_seed(tid,t['step'],'draft',0)
                assert meta['shared_model_revision']=='4b8f0835a3f613e40541619462ae906db8dba95c'
                history.append(dict(observation=t['observation'],action=t['action']))
                if mode=='structured':sigs.append(meta['internal_signals'])
            cs=[c for c in primary if c['identity']['task']==tid and c['identity']['mode']==mode]
            assert e['total_tokens']==sum(c['metadata']['usage']['total_tokens'] for c in cs)
            assert e['formal_steps']==len(ts)
            if e['episode_complete']:assert e['done'] or len(ts)==20
            else:assert e['status']=='parse_failure' and (p/'parse_failures.jsonl').exists()
            rows.append(dict(model='SR7B',mode=mode,task_id=tid,score=e['task_score'] if e['episode_complete'] else None,partial_native_score=e['task_score'],
                status=e['status'],complete=e['episode_complete'],success=e['success'] if e['episode_complete'] else None,steps=e['formal_steps'],calls=e['actor_calls'],tokens=e['total_tokens'],queries=e['transition_queries'],seconds=e['seconds_including_init_cleanup']))
            if ts:examples.append(dict(model='SR7B',mode=mode,task=tid,first_raw=ts[0]['raw'],first_feedback=ts[0]['actual'],last_raw=ts[-1]['raw'],last_feedback=ts[-1]['actual']))
            if (p/'parse_failures.jsonl').exists():
                for f in read_rows(p/'parse_failures.jsonl'):examples.append(dict(model='SR7B',mode=mode,task=tid,format_failure=f['parsed'],raw=f['raw'],signals=f['metadata'].get('internal_signals')))
        p=control/'episodes'/tid;e=json.loads((p/'complete.json').read_text());assert json.loads((p/'initial.json').read_text())['state_sha256']==expected
        rows.append(dict(model='Qwen35_9B',mode='native_h2',task_id=tid,score=e['task_score'] if e['episode_complete'] else None,partial_native_score=e['task_score'],status=e['status'],complete=e['episode_complete'],success=e['success'] if e['episode_complete'] else None,steps=e['formal_steps'],calls=e['actor_calls'],tokens=e['total_tokens'],queries=e['transition_queries'],seconds=e['seconds_including_init_cleanup']))
        old=json.loads((ROOT/'outputs/phase510_reviewer_repair_v4/episodes'/tid/'never-original/complete.json').read_text())
        rows.append(dict(model='Qwen35_9B',mode='structured',task_id=tid,score=old['task_score'],partial_native_score=old['task_score'],status='historical_phase510_never',complete=True,success=old['success'],steps=old['formal_environment_actions'],calls=old['formal_environment_actions'],tokens=old['total_tokens'],queries=1+old['formal_environment_actions'],seconds=old['total_seconds_including_cleanup']))
    ledger=json.loads((sr/'ledger.json').read_text());assert ledger['tokens']==sum(c['metadata']['usage']['total_tokens'] for c in calls) and ledger['model_calls']==len(calls)
    assert ledger['transition_queries']==sum(e['queries'] for e in rows if e['model']=='SR7B')
    ccalls=read_rows(control/'generation_calls.jsonl');cl=json.loads((control/'ledger.json').read_text());assert cl['tokens']==sum(c['metadata']['usage']['total_tokens'] for c in ccalls) and cl['model_calls']==len(ccalls)
    assert cl['transition_queries']==sum(e['queries'] for e in rows if e['model']=='Qwen35_9B' and e['mode']=='native_h2')
    for l in [ledger,cl]:assert l['pending_model'] is None and l['pending_transition'] is None
    primary_scored=[c['metadata']['internal_signals'] for c in primary if c['identity']['mode']=='structured']
    groups=[]
    for model,mode in [('Qwen35_9B','structured'),('Qwen35_9B','native_h2'),('SR7B','structured'),('SR7B','native_h2')]:
        es=[e for e in rows if e['model']==model and e['mode']==mode];complete=[e for e in es if e['complete']]
        groups.append(dict(model=model,mode=mode,attempts=len(es),complete_episodes=len(complete),format_failures=len(es)-len(complete),
            mean_score=sum(e['score'] for e in complete)/len(complete) if complete else None,success_rate=sum(e['success'] for e in complete)/len(complete) if complete else None,
            all_attempts_comparable_as_complete=len(complete)==len(es),total_tokens=sum(e['tokens'] for e in es),steps=sum(e['steps'] for e in es),queries=sum(e['queries'] for e in es),seconds=sum(e['seconds'] for e in es)))
    atomic(out/'summary.json',dict(groups=groups,sr_structured_signal_attempts=len(primary_scored),sr_structured_missing_span=sum(x['missing_span'] for x in primary_scored),
        sr_executed_structured_signal_steps=len(sigs),initial_states_match_across_all_conditions=True,old_baseline_not_regenerated=True,
        numerical_validation=json.loads((sr/'numerical_validation.json').read_text()),new_measurement_cost=dict(model_calls=ledger['model_calls']+cl['model_calls'],tokens=ledger['tokens']+cl['tokens'],transition_queries=ledger['transition_queries']+cl['transition_queries'],audit_capture_bundles=ledger['audit_capture_bundles']+cl['audit_capture_bundles'],
            sr_ledger=ledger,prompt_control_ledger=cl,downloaded_checkpoint_bytes=15247179567,additional_experience_training_gate_wm_reviewer_calls=0),
        not_independent_generalization=True,no_claim_of_exact_checkpoint_training_membership=True,scientific_gate_effectiveness_not_evaluated=True))
    atomic(out/'examples.json',examples)
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    atomic(out/'provenance.json',dict(analysis_script_sha256=file_hash(Path(__file__)),sr_manifest_sha256=file_hash(sr/'manifest.json'),control_manifest_sha256=file_hash(control/'manifest.json'),
        primary_scored_model_metadata_and_missing_signals_retained=True))
    print(json.dumps(dict(groups=groups,new_calls=ledger['model_calls']+cl['model_calls'],new_tokens=ledger['tokens']+cl['tokens']),indent=2))


if __name__=='__main__':main()
