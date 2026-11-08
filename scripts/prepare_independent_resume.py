#!/usr/bin/env python3
"""Preserve failed audit version; reuse ALL20 complete unchanged measurements.

Only the post-run auditor seed-schema reader may differ. Generation, method,
cohort and budgets must match. Physical resource receipts remain cumulative.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,verify_frozen
from experience.splits import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    old=Path(a.source).resolve();new=Path(a.output).resolve()
    if new.exists():raise ValueError('New explicit recovery Run ID required')
    m=json.loads((old/'manifest.json').read_text());verify_frozen(ROOT,m['profile'])
    actual={name:file_hash(ROOT/name) for name in m['code_hashes']}
    changed=[name for name,h in actual.items() if h!=m['code_hashes'][name]]
    if changed!=['scripts/audit_independent_dev.py']:raise ValueError('Only known auditor repair allowed: '+str(changed))
    expected={(g['task_id'],g['seed'],policy) for g in m['schedule'] if g['round']==1 for policy in ('never','always','internal','full')}
    results=[];hashes=[]
    for path in (old/'episodes').rglob('complete.json'):
        e=json.loads(path.read_text());results.append(e)
        for name,field in [('steps.jsonl','steps_sha256'),('generation_calls.jsonl','generation_calls_sha256')]:
            if file_hash(path.parent/name)!=e[field]:raise ValueError('Completed data modified')
    if {(r['task_id'],r['rollout_seed'],r['policy']) for r in results}!=expected or len(results)!=20:
        raise ValueError('Recovery only reuses the complete fixed first round, never selected outcomes')
    for i in range(2):
        ledger=json.loads((old/('worker'+str(i))/'ledger.json').read_text())
        if ledger['pending_model'] or ledger['pending_transition']:raise ValueError('Unknown receipt requires separate accounting')
    new.mkdir(parents=True);shutil.copytree(old/'episodes',new/'episodes')
    for path in (old/'episodes').rglob('*'):
        if path.is_file():
            target=new/path.relative_to(old)
            if file_hash(path)!=file_hash(target):raise ValueError('Copy changed raw data')
            hashes.append(dict(path=str(path.relative_to(old)),sha256=file_hash(path)))
    for i in range(2):
        src=old/('worker'+str(i));dst=new/('worker'+str(i));dst.mkdir()
        for name in ('ledger.json','episodes.jsonl','model_loads.jsonl','setup_events.jsonl'):
            if (src/name).exists():shutil.copyfile(src/name,dst/name)
    for name in ('owned_workers.jsonl','environment_snapshot.json'):
        if (old/name).exists():shutil.copyfile(old/name,new/name)
    replacement=copy.deepcopy(m);replacement.update(code_hashes=actual,
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        resume_created_at_ns=time.time_ns(),
        recovery=dict(source_run=str(old),source_manifest_sha256=file_hash(old/'manifest.json'),
            data_generation_commit=m['source_git_commit'],original_code_hashes=m['code_hashes'],
            changed_files=changed,reason='strict auditor reads existing scored actor_metadata.generation_seed',
            reused_complete_episodes=20,regenerated_reused_episodes=0,cumulative_ledgers_imported=True,
            deadline_and_all_method_settings_unchanged=True))
    atomic(new/'manifest.json',replacement)
    atomic(new/'reused_data_provenance.json',dict(source_run=str(old),files=hashes,all_bytes_identical=True,
        fixed_round1_reused_in_full=True,selection_by_outcomes=False))
    print(json.dumps(dict(reused_episodes=20,copied_raw_files=len(hashes),only_changed_evaluation_file=changed,
        output=str(new),old_run_preserved=True),indent=2))


if __name__=='__main__':main()
