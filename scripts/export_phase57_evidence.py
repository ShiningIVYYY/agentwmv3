#!/usr/bin/env python3
"""Copy compact TRUE diagnostic evidence, not weights/caches/full trajectories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def compact(t):
    sim=t['simulation'];internal=t['metadata'].get('internal_signals')
    signal={k:internal[k] for k in ('premise_uncertainty','consequence_uncertainty','local_uncertainty','global_uncertainty','missing_span','backend','dtype','model_revision')} if internal else None
    public={k:sim.get(k) for k in ('status','prediction_available','source','verified_prefix_steps','predicted_observation','predicted_reward','predicted_done','predicted_score','error','simulation_cost','simulation_latency')} if sim else None
    if sim and sim.get('mismatch'):public['rejected_state_fields']=list(sim['mismatch']['fields'])
    return dict(source_record_sha256=digest(t),policy=t['policy'],task_id=t['task_id'],step_index=t['step']+1,
                goal=t['goal'],observation=t['observation'],draft_decision=t['draft_output'],internal_signals=signal,
                external=t['resolved_external'],gate=t['gate'],simulation=public,review=t['review'],final_action=t['final_action'],
                actual=t['actual'],gate_decided_at_ns=t['decided_at_ns'],executed_at_ns=t['executed_at_ns'],
                actor_usage=t['metadata']['usage'],review_usage=t['review_metadata']['usage'] if t['review_metadata'] else None,
                immediate_score_delta=t.get('one_step_score_delta_actual_minus_trial_draft'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gate-config-output',help='Optional explicit target; never overwrite different frozen thresholds')
    a=p.parse_args()
    run=Path(a.run);audit=Path(a.audit);cal=Path(a.calibration);out=Path(a.output)
    if out.exists():raise ValueError('New evidence directory required')
    if json.loads((run/'progress.json').read_text())['stage']!='complete':raise ValueError('Incomplete run')
    out.mkdir(parents=True)
    for src,name in [(run/'summary.json','summary.json'),(run/'manifest.json','run_manifest.json'),
                     (run/'gate_config.json','gate_config.json'),(run/'calibration.json','calibration.json'),
                     (run/'model_smoke.json','model_smoke.json'),(cal/'model_smoke.json','calibration_model_smoke.json'),
                     (run/'calibration_provenance.json','calibration_provenance.json'),(audit/'audit.json','audit.json'),
                     (audit/'corrections.json','corrections.json'),
                     (ROOT/'checkpoints/qwen35_9b/assets_manifest.json','assets_manifest.json'),
                     (ROOT/'outputs/phase57_setup/official_weight_checks.json','official_weight_checks.json'),
                     (ROOT/'outputs/phase57_setup/replay_minimal_failure.json','minimal_replay_failure.json')]:
        shutil.copyfile(src,out/name)
    traces=rows(run/'evaluation_steps.jsonl');calibration=rows(cal/'calibration_steps.jsonl');episodes=rows(run/'episodes.jsonl')
    write(out/'episodes.json',episodes)
    values=[]
    for t in calibration:
        s=t['metadata']['internal_signals']
        values.append(dict(task_id=t['task_id'],step_index=t['step']+1,local=s['local_uncertainty'],global_nll=s['global_uncertainty'],
                           premise=s['premise_uncertainty'],consequence=s['consequence_uncertainty'],missing=s['missing_span'],
                           token_count=s['token_count'],status=s['status'],generation_record_sha256=digest(t['metadata']['generation_record'])))
    write(out/'calibration_values.json',dict(source_file_sha256=sha(cal/'calibration_steps.jsonl'),
          model_config=json.loads((run/'manifest.json').read_text())['model_config'],records=values))
    predicates=[('positive_immediate_score',lambda t:t.get('one_step_score_delta_actual_minus_trial_draft',0)>0),
                ('closed_route_repair',lambda t:t['review_metadata'] and t['draft_action'].startswith('go ') and t['final_action'].startswith('open ')),
                ('rejected_reconstruction',lambda t:t['simulation'] and not t['simulation']['prediction_available']),
                ('low_signal_failure',lambda t:t['policy'] in ('internal','full') and t['gate']['reason']=='no_active_trigger' and t['gate']['budget_remaining']>0 and t['actual']['score']<0),
                ('external_conflict',lambda t:t['policy']=='full' and t['gate']['reason']=='reliable_action_conflict_requires_verification_correction_off'),
                ('invalid_replacement',lambda t:t['review_metadata'] and t['final_action']!=t['draft_action'] and t['actual']['observation'].startswith('No known action')),
                ('unsupported_semantic_reason',lambda t:t['review_metadata'] and 'water' in t['review']['raw'].lower() and 'living thing' in t['review']['raw'].lower())]
    examples=[]
    for label,predicate in predicates:
        matching=[t for t in traces if predicate(t)]
        if matching:examples.append(dict(label=label,**compact(matching[0])))
    write(out/'trace_examples.json',dict(source_run=str(run),source_trace_sha256=sha(run/'evaluation_steps.jsonl'),examples=examples))
    full=[t for t in traces if t['policy']=='full'];unique_evidence=sorted({x for t in full for x in t['resolved_external']['evidence_ids']})
    write(out/'external_evidence.json',dict(coverage=dict(Counter(t['resolved_external']['coverage'] for t in full)),
          conflict=dict(Counter(t['resolved_external']['conflict'] for t in full)),sufficiency=dict(Counter(t['resolved_external']['sufficiency'] for t in full)),
          source_evidence_ids=unique_evidence,snapshot_sha256=sha(ROOT/'docs/phase5/memory_snapshot_v2.json'),
          no_new_experience=True,memory_correction=False,confirmed_skip_enabled=False))
    probes=[]
    for version in (1,2,3):
        d=ROOT/('outputs/phase57_replay_validation_v'+str(version))
        item=dict(version=version,summary=json.loads((d/'summary.json').read_text()),trials_sha256=sha(d/'trials.jsonl'))
        if (d/'provenance.json').exists():item['provenance']=json.loads((d/'provenance.json').read_text())
        probes.append(item)
    write(out/'replay_validation.json',probes)
    jobs=[]
    for version in (1,2,3,4,5):
        name='v'+str(version);job=ROOT/('outputs/phase57_setup/qwen35_job_'+name+'.json');log=ROOT/('outputs/phase57_setup/qwen35_runtime_'+name+'.txt')
        jobs.append(dict(version=version,job=json.loads(job.read_text()),log_sha256=sha(log),
                         outcome='complete' if version==5 else log.read_text().splitlines()[-1]))
    write(out/'run_attempts.json',jobs)
    tokens_cal=sum(t['metadata']['usage']['total_tokens'] for t in calibration)
    extra_cal=2*calibration[0]['metadata']['usage']['total_tokens'] # Proven equal prompt/response IDs in v4 smoke.
    job5=jobs[-1]['job'];audit_data=json.loads((audit/'audit.json').read_text())
    pids={job5['pid']}
    for t in traces:
        if t['simulation']:
            pids.update(t['simulation'][k] for k in ('worker_pid','formal_pid') if k in t['simulation'])
    live=[pid for pid in sorted(pids) if Path('/proc/'+str(pid)).exists()]
    if live:raise ValueError('Owned logged processes still present: '+str(live))
    counters={}
    for item in audit_data['policies']:
        c=item['simulation_cost'];counters[item['policy']]=sum(c.get(k,0) for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls'))
    forward_tokens=0
    for version in (3,4,5):
        previous=ROOT/('outputs/phase57_replay_9b_v'+str(version))
        record=json.loads((previous/'ordinary_validation_generation.json').read_text())['generation_record']
        checked=json.loads((previous/'model_smoke.json').read_text())['checked_tokens']
        forward_tokens+=len(record['prompt_ids'])+checked
    write(out/'cost_scope.json',dict(evaluation_model_calls=sum(p['actor_calls']+p['reviewer_calls'] for p in audit_data['policies']),
          evaluation_tokens=sum(p['total_tokens'] for p in json.loads((run/'summary.json').read_text())['policies']),
          run_model_calls=len(rows(run/'generation_calls.jsonl')),run_tokens=audit_data['model_call_tokens'],
          calibration_model_calls=len(calibration),calibration_tokens=tokens_cal,
          v3_ordinary_validation_tokens=json.loads((ROOT/'outputs/phase57_replay_9b_v3/ordinary_validation_generation.json').read_text())['usage']['total_tokens'],
          v4_ordinary_validation_tokens=json.loads((cal/'ordinary_validation_generation.json').read_text())['usage']['total_tokens'],
          v4_extra_validation_tokens_derived_from_proven_parity=extra_cal,
          v5_wall_to_summary_seconds=((run/'summary.json').stat().st_mtime_ns-job5['started_at_ns'])/1e9,
          formal_episode_initialization_queries=len(episodes),formal_actions=len(traces),lookahead_transition_queries=counters,
          calibration_initialization_queries=len({t['task_id'] for t in calibration}),
          extra_diagnostic_forward_passes_v3_v4_v5=3,
          diagnostic_forward_input_tokens=forward_tokens,
          unknown_historical_generation_token_receipts=['v1 first ordinary generation','v4 first evaluation draft before Query schema error'],
          historical_stage_total_tokens_exact=False,offline_new_experience_cost=0,
          owned_logged_processes_checked=len(pids),owned_logged_processes_remaining=live))
    if a.gate_config_output:
        target=Path(a.gate_config_output)
        if target.exists():
            if json.loads(target.read_text())!=json.loads((run/'gate_config.json').read_text()):
                raise ValueError('Do not overwrite different frozen pilot thresholds')
        else:shutil.copyfile(run/'gate_config.json',target)
    print(json.dumps(dict(evidence_directory=str(out),files=len(list(out.iterdir())),examples=len(examples),calibration_tokens=tokens_cal),indent=2))


if __name__=='__main__':main()
