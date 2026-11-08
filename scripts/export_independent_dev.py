#!/usr/bin/env python3
"""Compact genuine results/config/cost provenance; no model inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);analysis=Path(a.analysis);out=Path(a.output)
    if not (run/'complete.json').exists():raise ValueError('Run incomplete')
    if json.loads((analysis/'engineering_audit.json').read_text())['status']!='engineering_checks_pass':raise ValueError('Audit incomplete')
    if out.exists():raise ValueError('New evidence directory required')
    shutil.copytree(analysis,out)
    for name,target in [('manifest.json','run_manifest.json'),('complete.json','run_complete.json'),
        ('assets_verified.json','model_assets_verified.json'),('environment_snapshot.json','environment.json'),
        ('round1_acceptance.json','round1_acceptance.json')]:
        shutil.copyfile(run/name,out/target)
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    for key,target in [('actor_config','actor_config.json'),('gate_config','gate_config.json')]:
        shutil.copyfile(ROOT/profile[key],out/target)
    recovered=bool(manifest.get('recovery'))
    if recovered:
        shutil.copyfile(run/'reused_data_provenance.json',out/'reused_data_provenance.json')
    original=Path(manifest['recovery']['source_run']) if recovered else run
    loads=[];setup=[];workers=read_rows(run/'owned_workers.jsonl')
    for i in range(2):
        w=run/('worker'+str(i));loads.extend(read_rows(w/'model_loads.jsonl'));setup.extend(read_rows(w/'setup_events.jsonl'))
    atomic(out/'model_load_events.json',loads);atomic(out/'shared_setup_events.json',setup)
    parent=json.loads((original/'manifest.json').read_text())
    error=json.loads((original/'error.json').read_text()) if recovered else None
    complete=json.loads((run/'complete.json').read_text());audit=json.loads((run/'final_audit.json').read_text())
    phase1_end=max(error['available_at_ns'],(original.parent/(original.name+'.log')).stat().st_mtime_ns) if recovered else 0
    round1=json.loads((run/'round1_acceptance.json').read_text())
    oldpids={r['pid'] for r in read_rows(original/'owned_workers.jsonl')} if recovered else set()
    worker_residency=[]
    for w in workers:
        end=phase1_end if w['pid'] in oldpids else complete['finished_at_ns']
        worker_residency.append(dict(**w,process_residency_upper_seconds=max(0,(end-w['started_at_ns'])/1e9),
            scope='worker spawn to supervising-run completion; includes CPU setup/idle, not measured GPU kernel time'))
    atomic(out/'cost_scope.json',dict(model_calls=audit['model_calls'],llm_tokens=audit['llm_tokens'],
        transition_queries=audit['transition_queries'],audit_capture_bundles=audit['ledger']['audit_capture_bundles'],
        original_round1_calls=round1['model_calls'],original_round1_tokens=round1['llm_tokens'],round1_measurements_included_without_regeneration=True,recovery_import=recovered,
        unique_measurement_episodes=len(json.loads((analysis/'episodes.json').read_text())),
        total_model_loads_including_original_and_recovery=len(loads),additional_model_loads_due_to_recovery=len(loads)-2,
        shared_model_load_seconds=sum(x['load_seconds'] for x in loads),
        shared_memory_load_seconds=sum(x['memory_load_seconds'] for x in setup),
        worker_residency_upper_intervals=worker_residency,
        assigned_worker_process_hours_upper=sum(x['process_residency_upper_seconds'] for x in worker_residency)/3600,
        calendar_study_wall_seconds=(complete['finished_at_ns']-parent['started_at_ns'])/1e9,
        original_supervisor_wall_to_error_seconds=(phase1_end-parent['started_at_ns'])/1e9 if recovered else None,
        recovered_supervisor_seconds=complete['seconds'],
        counter_import_note='Recovery ledgers already include original counters; never add them twice.' if recovered else 'No imported counters.',
        query_unit='native initialization/action/replay/candidate transitions; inspection bundles separate; low-level RPC count unmeasured',
        offline_new_experience_construction=0,historical_memory_construction='unchanged; see Phase3/5 provenance',
        monetary_cost=None,flops=None,resource_caps=profile['limits']))
    atomic(out/'export_provenance.json',dict(export_script_sha256=file_hash(Path(__file__)),
        source_run_manifest_sha256=file_hash(run/'manifest.json'),source_analysis=str(analysis),
        dependency_hashes={name:file_hash(ROOT/name) for name in ['scripts/analyze_independent_dev.py',
            'scripts/audit_independent_dev.py','scripts/audit_phase57_replay.py','scripts/export_phase57_evidence.py',
            'scripts/plot_independent_dev.py','evaluation/independent_dev.py']},
        native_environment_queries_are_additional_real_access=True,no_downstream_counterfactual_utility_claim=True))
    files=list(out.rglob('*'));large=[str(x) for x in files if x.is_file() and x.stat().st_size>1048576]
    if large:raise ValueError('Compact evidence exceeded file limit; split before commit: '+str(large))
    print(json.dumps(dict(files=sum(x.is_file() for x in files),bytes=sum(x.stat().st_size for x in files if x.is_file()),output=str(out)),indent=2))


if __name__=='__main__':main()
