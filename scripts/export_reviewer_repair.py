#!/usr/bin/env python3
"""Compact real reviewer inputs/outcomes and accounting; no model or environment calls."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash
from reflection.replay_review_v2 import ReviewerProtocol,VERSION
from scripts.prepare_reviewer_repair import source_case
from scripts.audit_independent_dev import recorded_seed


def compact_meta(m):
    return dict(usage=m['usage'],finish_reason=m['finish_reason'],request_seed=recorded_seed(m),
        generation_seconds=m['internal_cost']['generation_seconds'],truncated=m['generation_record']['truncated'],
        prompt_tokens=m['prompt_tokens'],response_ids=m['generation_record']['response_ids'],
        model_revision=m['shared_model_revision'],actor_config_sha256=m['config_sha256'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--analysis',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    run=Path(a.run);analysis=Path(a.analysis);out=Path(a.output)
    summary=json.loads((analysis/'summary.json').read_text());assert summary['engineering_audit']['status']=='pass'
    if out.exists():raise ValueError('New output required')
    shutil.copytree(analysis,out);(out/'cases').mkdir()
    for name in ['manifest.json','complete.json','model.json','assets_verified.json','ledger.json','fixed_reviewer_lock.json','reused_data_provenance.json']:
        if (run/name).exists():shutil.copyfile(run/name,out/('run_'+name))
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());dev={r['task_id']:r for r in split['partitions']['development']}
    index=[]
    for x in read_rows(run/'offline_reviews.jsonl'):
        ref=x['ref'];t,past,prefix=source_case(ref);row=dev[t['task_id']]
        public={k:t['simulation'][k] for k in ('predicted_observation','predicted_reward','predicted_done','predicted_score')}
        inputs=dict(goal=t['goal'],current_observation=t['observation'],past_committed_history=past[-2:],draft_action=t['draft_action'],isolated_trial_feedback=public)
        assert x['metadata']['generation_record']['messages']==ReviewerProtocol(VERSION).messages(t['goal'],t['observation'],past,t['draft_action'],t['simulation'])
        value=dict(ref=ref,reviewer_input_public_only=inputs,source_original_raw=t['review']['raw'],source_original_parse=t['review']['parsed'],
            source_original_metadata=compact_meta(t['review_metadata']),fixed_raw=x['raw'],fixed_parse=x['parsed'],fixed_metadata=compact_meta(x['metadata']),
            audit_only_not_reviewer_input=dict(task=row,simulation_spec=dict(taskName=row['taskName'],variationIdx=row['variationIdx'],seed=profile['simulator_seed'],simplifications=profile['simplifications'],environment_step_limit=profile['environment_step_limit'],simulator_profile=profile['simulator_profile']),
                full_committed_action_prefix=prefix,expected_pre_full_digest=t['simulation']['checked_state_sha256'],expected_draft_post_fingerprint=t['simulation']['candidate_state_sha256'],
                original_final_action=t['final_action'],original_final_action_feedback=t['actual'],original_formal_post_fingerprint=t.get('formal_after_state_sha256'),
                historical_original_post_fingerprint_missing='formal_after_state_sha256' not in t))
        path=out/'cases'/(ref['prompt_id']+'.json');atomic(path,value);index.append(dict(path=str(path.relative_to(out)),sha256=file_hash(path),source_ref=ref))
    atomic(out/'case_index.json',index)
    lengths=[]
    for x in read_rows(run/'length_diagnostics.jsonl'):
        lengths.append(dict(ref=x['ref'],original_metadata=compact_meta(x['original_metadata']),extended_metadata=compact_meta(x['extended_metadata']),
            original_parse=x['original_parse'],extended_parse=x['extended_parse'],extended_raw=x['extended_raw'],same_first512_token_ids=x['extended_metadata']['generation_record']['response_ids'][:512]==x['original_metadata']['generation_record']['response_ids']))
    atomic(out/'bounded_length_diagnostics.json',lengths)
    loads=[json.loads((run/'model.json').read_text())];recovery=manifest.get('recovery')
    while recovery:
        loads.append(recovery['original_model_load']);recovery=recovery.get('source_chain')
    atomic(out/'all_model_loads.json',loads)
    atomic(out/'cost_scope.json',dict(total_ledger=summary['ledger'],model_loads=len(loads),model_load_seconds=sum(x['load_seconds'] for x in loads),
        episode_online_tokens=sum(g['total_tokens'] for g in summary['episode_groups']),
        episode_online_queries=sum(g['total_transition_queries'] for g in summary['episode_groups']),
        offline_model_tokens=sum(x['metadata']['usage']['total_tokens'] for x in read_rows(run/'generation_calls.jsonl')),
        offline_model_requests=len(read_rows(run/'generation_calls.jsonl')),
        controlled_comparison_queries=sum(b['transition_queries'] for c in read_rows(run/'one_step_comparisons.jsonl') for b in c['branches']),
        failed_and_probe_engineering_queries=manifest['recovery']['inherited_failed_diagnostic_transition_queries']+manifest['recovery']['additional_probe_transition_queries'],
        inherited_ledgers_already_include_all_original_offline_model_calls=True,never_add_reused_model_cost_twice=True,
        historical_original_source_inference_cost_not_charged_as_new_measurement=True,
        original_phase59_experience_construction_unchanged=True,new_offline_memory_synthesis=0,
        low_level_py4j_rpc_count=None,gpu_kernel_hours=None,monetary_cost=None,flops=None,
        final_process_wall_seconds=summary['complete']['seconds'],
        calendar_since_first_real_model_run_seconds=(summary['complete']['finished_at_ns']-json.loads((ROOT/'outputs/phase510_reviewer_repair_v2/manifest.json').read_text())['started_at_ns'])/1e9,
        note='Online episode, offline regression/controlled queries and failed engineering overhead are separate; no all-cost-zero claim. Timing scopes overlap; do not sum wall and components.'))
    atomic(out/'export_provenance.json',dict(script_sha256=file_hash(Path(__file__)),analysis_script_sha256=file_hash(ROOT/'scripts/analyze_reviewer_repair.py'),
        source_manifest_sha256=file_hash(run/'manifest.json'),exported_at_ns=time.time_ns(),case_pack_is_genuine=True,
        raw_complete_prompts_and_trace_logs_retained_on_lab=True,original_experiments_not_overwritten=True,
        full_driver_reproduction_requires_historical_source_files=True,
        case_input_and_audit_only_outcome_fields_separated=True))
    files=[f for f in out.rglob('*') if f.is_file()];assert not [f for f in files if f.stat().st_size>1048576]
    print(json.dumps(dict(files=len(files),bytes=sum(f.stat().st_size for f in files),out=str(out)),indent=2))


if __name__=='__main__':main()
