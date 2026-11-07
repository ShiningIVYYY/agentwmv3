#!/usr/bin/env python3
"""Real SDK preflight. Independent reset/replay is NOT claimed to be cloning.

Run with the existing SDK Python and PYTHONNOUSERSITE=1. No Actor, model
service, formal-environment rollback, speculative callback or fake feedback.
Exit42 means the required full isolated-clone capability remains unverified.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH
from worldmodel.science_snapshot import load_bridge
from evaluation.profile import canonicalize_observation
from experience.splits import validate_manifest


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def projection(e):
    tree=e.getObjectTree()
    return dict(observation=canonicalize_observation(e.look()),inventory=e.inventory(),
        moves=e.get_num_moves(),score=int(round(100*e.server.getScore())),completed=bool(e.server.getCompleted()),
        sdk_last_score=e.lastStepScore,executed_action_history=str(e.server.getActionHistory()),object_tree_projection_sha256=digest(tree))


def transition(e,action):
    started=time.perf_counter();obs,reward,done,info=e.step(action)
    return dict(observation=canonicalize_observation(obs),raw_observation=obs,reward=reward,done=done,score=info['score'],
        simulator_moves=info['moves'],latency_seconds=time.perf_counter()-started)


def serial_probe(e,obj):
    try:
        stream=e._gateway.jvm.java.io.ByteArrayOutputStream();out=e._gateway.jvm.java.io.ObjectOutputStream(stream)
        out.writeObject(obj);out.flush();return dict(success=True,bytes=stream.size())
    except Exception as error:return dict(success=False,error=str(error)[:1800])


def seed(e,value):
    cls=e._gateway.jvm.java.lang.Class.forName('scala.util.Random$',True,e.server.getClass().getClassLoader())
    cls.getField('MODULE$').get(None).setSeed(value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();dest=Path(a.output)
    if dest.exists():raise ValueError('Use a new preflight Run ID')
    dest.mkdir(parents=True);started=time.perf_counter()
    plan=json.loads((ROOT/'docs/phase5/development_plan.json').read_text())
    manifest_path=ROOT/'docs/phase3/split_manifest.json'
    manifest=json.loads(manifest_path.read_text());validate_manifest(manifest,require_native=True)
    dev={r['task_id']:r for r in manifest['partitions']['development']}
    selected=[r for r in plan['validation_tasks'] if r['task_id'] in ('sciworld_128','sciworld_2417','sciworld_3070')]
    if len(selected)!=3:raise ValueError('Approved dev tasks missing')
    for row in selected:
        allowed=dev.get(row['task_id'])
        if row.get('split')!='development' or allowed is None or any(row[k]!=allowed[k] for k in ('taskName','variationIdx','goal_group')):
            raise ValueError('Not an approved isolated development instance')
    inventory=None;trials=[];all_pids=[]
    for row in selected:
        for value in (0,1):
            ref=ScienceWorldEnv();worker=ScienceWorldEnv();loader=None
            try:
                for e in (ref,worker):
                    seed(e,value);e.envStepLimit=21;e.load(row['taskName'],row['variationIdx']);e.step('look around')
                ids=[ref._gateway.java_process.pid,worker._gateway.java_process.pid];all_pids+=ids
                if ids[0]==ids[1]:raise RuntimeError('Independent instances share JVM process')
                if inventory is None:
                    cls=ref.server.getClass();fields={}
                    for f in cls.getDeclaredFields():
                        if f.getName() in ('agentInterface','agent','actionHandler','taskMaker','currentHistory'):
                            f.setAccessible(True);fields[f.getName()]=serial_probe(ref,f.get(ref.server))
                    inventory=dict(sdk_version=importlib.metadata.version('scienceworld'),sdk_file=inspect.getfile(ScienceWorldEnv),
                        sdk_source_sha256=hashlib.sha256(Path(inspect.getfile(ScienceWorldEnv)).read_bytes()).hexdigest(),
                        jar_sha256=hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),java_root_class=cls.getName(),
                        sdk_public_methods=sorted(n for n in dir(ScienceWorldEnv) if not n.startswith('_')),
                        java_public_methods=sorted({m.getName() for m in cls.getMethods()}),
                        root_java_serialization=serial_probe(ref,ref.server),critical_fields_java_serialization=fields,
                        criu_path=shutil.which('criu'),capability_lines=[s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith(('CapEff:','CapPrm:'))],
                        snapshot_implementation='same-JVM live object references; no cross-JVM export/import')
                bridge,loader=load_bridge(ref,ROOT)
                initial_ref=projection(ref);initial_worker=projection(worker);steps=[];ref_count=worker_count=0
                for action in ('look around','wait','focus on air'):
                    before=projection(ref);witness=bridge.take(ref.server)
                    preview=transition(worker,action);worker_count+=1
                    # Read-only witness; never restore/execute speculatively on ref.
                    heap_unchanged=bool(witness.matches());after=projection(ref)
                    if not heap_unchanged or before!=after:raise RuntimeError('Independent worker mutated reference')
                    actual=transition(ref,action);ref_count+=1
                    keys=('observation','reward','done','score','simulator_moves')
                    steps.append(dict(action=action,worker=preview,reference_normal_execution=actual,
                        observable_transition_match=all(preview[k]==actual[k] for k in keys),
                        reference_unchanged_during_worker_step=True,read_only_heap_witness_matches=True,
                        reference_steps=ref_count,worker_steps=worker_count,
                        full_snapshot_import_performed=False))
                    if preview['done'] or actual['done']:break
                trials.append(dict(task_id=row['task_id'],taskName=row['taskName'],variationIdx=row['variationIdx'],probe_seed=value,
                    distinct_jvm_pids=ids,instance_isolation_observed=True,
                    initial_observable_projection_match=initial_ref==initial_worker,initial_ref=initial_ref,initial_worker=initial_worker,
                    steps=steps,scope='independent seeded reset/lockstep diagnostic, NOT a copied complete state'))
                (dest/'progress.json').write_text(json.dumps(dict(completed_trials=len(trials),total=6))+'\n')
            finally:
                if loader is not None:loader.close()
                worker.close();ref.close()
    report=dict(schema='phase57-isolated-clone-preflight-v1',status='blocked_required_clone_not_validated',
        inventory=inventory,trials=trials,independent_instances_verified=True,
        full_state_transfer_implemented=False,isolated_clone_verified=False,formal_environment_speculative_steps=0,
        actor_calls=0,reviewer_calls=0,learned_wm_calls=0,new_model_weights_downloaded=False,
        required_next_step='validated full-state export/import into another JVM, or separately validated process checkpoint infrastructure',
        no_formal_rollback_or_legacy_fallback=True,seconds=time.perf_counter()-started,owned_probe_java_pids=all_pids)
    report['split_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report['development_plan_sha256']=hashlib.sha256((ROOT/'docs/phase5/development_plan.json').read_bytes()).hexdigest()
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],trials=len(trials),real_transitions=sum(2*len(t['steps']) for t in trials),
        full_state_clone_verified=False,model_calls=0),indent=2))
    raise SystemExit(42)


if __name__=='__main__':main()
