#!/usr/bin/env python3
"""Export compact genuine repair results; never manufacture unavailable outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash
from scripts.export_phase57_evidence import compact


def rows(p):return [json.loads(line) for line in p.read_text().splitlines()]
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--controlled',required=True);p.add_argument('--histories',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);audit=Path(a.audit);out=Path(a.output)
    assert json.loads((run/'progress.json').read_text())['stage']=='complete'
    out.mkdir(parents=True,exist_ok=False)
    for src,name in [(run/'manifest.json','run_manifest.json'),(run/'summary.json','summary.json'),
            (run/'model_smoke.json','model_smoke.json'),(run/'calibration.json','calibration.json'),
            (run/'gate_config.json','gate_config.json'),(audit/'audit.json','audit.json'),
            (audit/'corrections.json','corrections.json')]:shutil.copyfile(src,out/name)
    controlled=Path(a.controlled);histories=Path(a.histories)
    write(out/'replay_validation.json',dict(
        controlled=dict(summary=json.loads((controlled/'summary.json').read_text()),
            provenance=json.loads((controlled/'provenance.json').read_text()),
            trials_sha256=file_hash(controlled/'trials.jsonl')),
        histories=dict(summary=json.loads((histories/'summary.json').read_text()),
            provenance=json.loads((histories/'manifest.json').read_text()),
            trials_sha256=file_hash(histories/'trials.jsonl'),diagnostics_sha256=file_hash(histories/'diagnostics.json'))))
    trace=rows(run/'evaluation_steps.jsonl');calls=rows(run/'generation_calls.jsonl');cal=rows(run/'calibration_steps.jsonl')
    write(out/'episodes.json',rows(run/'episodes.jsonl'))
    write(out/'calibration_values.json',dict(source_sha256=file_hash(run/'calibration_steps.jsonl'),
        collection='fresh_130_not_reused',records=[dict(task_id=r['task_id'],step_index=r['step']+1,
            local=r['metadata']['internal_signals']['local_uncertainty'],
            global_nll=r['metadata']['internal_signals']['global_uncertainty'],
            missing=r['metadata']['internal_signals']['missing_span']) for r in cal]))
    simulations=[t['simulation'] for t in trace if t['simulation']]
    evalmeta=[t['metadata'] for t in trace]+[t['review_metadata'] for t in trace if t['review_metadata']]
    cost=Counter()
    for sim in simulations:cost.update(sim['simulation_cost'])
    job=json.loads((ROOT/'outputs/replay_fix_130_process.json').read_text())
    write(out/'cost_scope.json',dict(
        process_timing=job,
        whole_run_to_summary_seconds_estimate=((run/'summary.json').stat().st_mtime_ns-job['process_start_unix_ns_estimate'])/1e9,
        evaluation_calls=len(evalmeta),evaluation_tokens=sum(m['usage']['total_tokens'] for m in evalmeta),
        all_run_calls=len(calls),all_run_tokens=sum(c['metadata']['usage']['total_tokens'] for c in calls),
        calls_by_role=dict(Counter(c['metadata']['role'] for c in calls)),
        tokens_by_role={role:sum(c['metadata']['usage']['total_tokens'] for c in calls if c['metadata']['role']==role)
            for role in sorted({c['metadata']['role'] for c in calls})},
        formal_actions=len(trace),formal_initialization_queries=12,online_simulation_cost=dict(cost),
        online_simulation_latency=sum(s['simulation_latency'] for s in simulations),
        formal_environment_version='1.3.0',prediction_available=sum(s['prediction_available'] for s in simulations),
        reconstruction_failures=sum(not s['prediction_available'] for s in simulations),
        checked_keep_transitions=sum(t['kept_draft_transition_match'] is True for t in trace),
        negative_trial_predictions=sum(s.get('predicted_score',0)<0 for s in simulations),
        terminal_trial_predictions=sum(s.get('predicted_done',False) for s in simulations),
        diagnostic_forward_passes=json.loads((run/'model_smoke.json').read_text())['extra_validation_forward_passes'],
        offline_new_experience_cost=0,no_money_conversion=True))
    examples=[]
    predicates=[('ordinary_verified_keep',lambda t:t['kept_draft_transition_match'] is True),
        ('negative_terminal_trial',lambda t:t['simulation'] and t['simulation'].get('predicted_score',0)<0),
        ('terminal_transition',lambda t:t['simulation'] and t['simulation'].get('predicted_done',False)),
        ('positive_immediate_correction',lambda t:t.get('one_step_score_delta_actual_minus_trial_draft',0)>0),
        ('rejected_replacement',lambda t:t['draft_action']!=t['final_action'] and t['actual']['observation'].startswith('No known action')),
        ('external_conflict',lambda t:t['policy']=='full' and t['resolved_external']['conflict']=='true')]
    for label,predicate in predicates:
        found=next((t for t in trace if predicate(t)),None)
        if found:
            value=compact(found)
            value['checked_candidate_state_sha256']=found['simulation'].get('candidate_state_sha256') if found['simulation'] else None
            value['kept_draft_transition_match']=found['kept_draft_transition_match']
            examples.append(dict(label=label,**value))
    write(out/'trace_examples.json',dict(trace_sha256=file_hash(run/'evaluation_steps.jsonl'),examples=examples))
    write(out/'provenance.json',dict(
        scripts={str(p.relative_to(ROOT)):file_hash(p) for p in
            [ROOT/'scripts/export_replay_fix_evidence.py',ROOT/'scripts/validate_replay_histories.py',
             ROOT/'scripts/prepare_scienceworld_replay.py']},
        original_report_unchanged_evidence='docs/phase57_replay',
        real_simulator_access='oracle_like_not_learned_wm',phase6_started=False))
    print(json.dumps(dict(files=len(list(out.iterdir())),examples=len(examples),verified=len(simulations)),indent=2))


if __name__=='__main__':main()
