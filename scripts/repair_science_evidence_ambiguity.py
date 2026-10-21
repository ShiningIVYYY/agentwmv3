#!/usr/bin/env python3
"""Audited continuation after the native duplicate-label adapter failure.

Retain all completed task bytes and old configs in explicit cohorts. Only the
previously crashed Oracle task is retried; no sampling/prompt/policy changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.matrix import read,save,sha,verify
from experiments.scheduler import active_runners,matches_process
from worldmodel.runtime import audit_settings

ALLOWED=('worldmodel/contexts.py','worldmodel/runtime.py',
    'scripts/run_science_evidence_experiments.py','scripts/test_science_evidence_experiments.py')


def repair(out):
    target=out/'experiment.json';matrix=read(target);state=read(out/'runstate.json')
    archive=out/'maintenance/duplicate_ambiguity_v1'
    if (archive/'transition.json').exists():
        raise RuntimeError('Repair already recorded; resume the existing manager instead')
    if matches_process(read(out/'manager.json')):
        raise RuntimeError('Manager is still running')
    for runner in active_runners():
        if any(str(out) in str(x) for x in runner['command']):
            raise RuntimeError('Experiment evaluator still running')
    if state.get('phase')!='error' or 'Runner exhausted retries: full/stage1_guard/oracle' not in state.get('error',''):
        raise RuntimeError('Unexpected failure; this repair is narrowly scoped')
    changed={p:(old,sha(p)) for p,old in matrix['hashes'].items() if sha(p)!=old}
    if {str(Path(p).relative_to(ROOT)) for p in changed}!=set(ALLOWED):
        raise RuntimeError('Unexpected frozen source changes: '+str(list(changed)))
    old_matrix=copy.deepcopy(matrix);old_state=copy.deepcopy(state)
    completed={};cohorts={};errors=[]
    for key,job in matrix['jobs'].items():
        rows={}
        for p in (Path(job['output'])/'tasks').glob('*.json'):
            row=read(p)
            if row['status']=='completed':
                rows[row['task_id']]=sha(p);completed[str(p)]=sha(p)
            else:errors.append((key,row))
        if rows:cohorts[key]=rows
    if len(completed)!=399 or len(errors)!=1 or errors[0][0]!='stage1_guard/oracle' or errors[0][1]['task_id']!='sciworld_581' or 'ambiguity choices lack unique labels' not in errors[0][1]['error']:
        raise RuntimeError('Expected 399 completed tasks and the one known adapter failure')
    archive.mkdir(parents=True)
    save(archive/'experiment.before.json',old_matrix);save(archive/'runstate.before.json',old_state)
    save(archive/'failed_task.before.json',errors[0][1])
    for key in cohorts:
        save(archive/'configs'/key/'config.before.json',read(out/key/'config.json'))
    for p,(old,new) in changed.items():
        rel=str(Path(p).relative_to(ROOT))
        before=subprocess.check_output(['git','show',matrix['commit']+':'+rel],cwd=ROOT)
        if hashlib.sha256(before).hexdigest()!=old:raise RuntimeError('Original source does not match archive')
        dest=archive/'sources.before'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(before)
    config_path=out/'stage1_guard/oracle/config.json'
    config=read(config_path);feature=audit_settings(SimpleNamespace(**config),ROOT)['wm_reflection']
    after=dict(config,wm_reflection=feature)
    # All generated args, model, task IDs and policy names stay identical.
    if {k:v for k,v in config.items() if k!='wm_reflection'}!={k:v for k,v in after.items() if k!='wm_reflection'}:
        raise RuntimeError('Continuation changed evaluation arguments')
    transition=dict(protocol='science-evidence-infrastructure-continuation-v1',updated=time.time(),
        cause='native ambiguity options can have identical labels with distinct numbers',
        behavior='unique-label menus unchanged; duplicate-label menus preserve native bytes and all native numbers',
        previous_matrix_sha256=sha(target),previous_commit=matrix['commit'],
        repair_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        changed_sources={str(Path(p).relative_to(ROOT)):dict(before=a,after=b) for p,(a,b) in changed.items()},
        retained_completed_task_hashes=completed,retained_task_cohorts=cohorts,
        old_config_paths={key:str(archive/'configs'/key/'config.before.json') for key in cohorts},
        retried_task_ids=['sciworld_581'],extra_launch_budget=1,
        unchanged=['SR/WM checkpoints','actor and reflection prompts','selection rules','sampling','manifest','GPU allocation'])
    save(archive/'config.oracle.after.json',after)
    save(archive/'transition.json',transition)
    for p,(_,new) in changed.items():matrix['hashes'][p]=new
    matrix['hashes'].update(completed)
    for p in [archive/'transition.json',Path(__file__),ROOT/'scripts/test_wm_reflection.py']:
        matrix['hashes'][str(p)]=sha(p)
    matrix.setdefault('maintenance_transitions',[]).append(str(archive/'transition.json'))
    save(config_path,after)
    save(target,matrix)
    record=state['jobs']['full/stage1_guard/oracle']
    record['maintenance_restarts']=record.get('maintenance_restarts',0)+1
    record['maintenance_transition']=str(archive/'transition.json')
    record['status']='pending'
    state.update(matrix_sha256=sha(target),phase='maintenance_repaired')
    state.pop('error',None);save(out/'runstate.json',state)
    verify(matrix)
    for p,digest in completed.items():
        if sha(p)!=digest:raise RuntimeError('Completed task changed during repair')
    print(json.dumps(dict(passed=True,retained_completed_tasks=len(completed),
        retry_task='sciworld_581',recorded_extra_launches=1,transition=str(archive/'transition.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    repair(p.parse_args().output.resolve())
