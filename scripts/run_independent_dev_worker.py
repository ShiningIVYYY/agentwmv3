#!/usr/bin/env python3
"""One persistent shared Actor/Reviewer per owned GPU; bounded dev queue only."""
import argparse
from itertools import islice
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,verify_frozen,Ledger,run_episode,ResourceLimit
from experience.memory import ExperienceMemory
from experience.splits import file_hash
from internal_signals.qwen35_actor import Qwen35Actor


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--model-dir',required=True);a=p.parse_args();run=Path(a.run);worker=run/('worker'+str(a.worker))
    worker.mkdir(exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile'];split=verify_frozen(ROOT,profile)
    for name,expected in manifest['code_hashes'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Evaluation code changed: '+name)
    def stop(signum,frame):raise ResourceLimit('Owned worker stop signal')
    signal.signal(signal.SIGTERM,stop)
    ledger=Ledger(worker/'ledger.json',run/'STOP',manifest['deadline_unix'],profile['worker_limits'],profile['batch_limits'])
    started=time.perf_counter();actor=Qwen35Actor(ROOT/profile['actor_config'],a.model_dir,'cuda:0')
    gate=json.loads((ROOT/profile['gate_config']).read_text())
    if actor.fingerprint!=gate['internal_signature']['config_sha256']:raise ValueError('Actor scoring signature changed')
    model_info=dict(pid=os.getpid(),loading_info=actor.loading_info,load_seconds=time.perf_counter()-started,
        model_revision=actor.config['model_revision'],config_sha256=actor.fingerprint,
        shared_actor_reviewer_object=True,backend=actor.config['backend'],dtype=actor.config['dtype'],
        base_model_generation_config=actor.model.generation_config.to_dict())
    append(worker/'model_loads.jsonl',model_info)
    atomic(worker/'model.json',model_info)
    begin=time.perf_counter()
    memory=ExperienceMemory(json.loads((ROOT/profile['memory_snapshot']).read_text()),split)
    append(worker/'setup_events.jsonl',dict(pid=os.getpid(),memory_load_seconds=time.perf_counter()-begin,available_at_ns=time.time_ns()))
    schedule=manifest['schedule'];owned=[g for g in schedule if g['worker']==a.worker]
    try:
        for stage in (1,2):
            if stage==2:
                atomic(worker/'progress.json',dict(stage='await_round1_audit',pid=os.getpid()))
                start_wait=time.monotonic()
                while not (run/'round1_acceptance.json').exists():
                    if (run/'STOP').exists() or time.monotonic()-start_wait>600:raise ResourceLimit('No timely first-round engineering acceptance')
                    time.sleep(2)
                accepted=json.loads((run/'round1_acceptance.json').read_text())
                if accepted['status']!='engineering_checks_pass':raise ValueError('Round1 not accepted')
                verify_frozen(ROOT,profile)
            groups=[g for g in owned if g['round']==stage]
            for offset in range(0,len(groups),3):
                batch=groups[offset:offset+3];ledger.begin_batch();begin=time.perf_counter();results=[]
                batch_id='round'+str(stage)+'-batch'+str(offset//3)
                for group in batch:
                    row=next(r for r in profile['tasks'] if r['task_id']==group['task_id'])
                    for policy in ('never','always','internal','full'):
                        out=run/'episodes'/row['task_id']/('seed'+str(group['seed']))/policy
                        if (out/'complete.json').exists():
                            complete=json.loads((out/'complete.json').read_text())
                            if complete['steps_sha256']!=file_hash(out/'steps.jsonl'):raise ValueError('Completed trace changed')
                            results.append(complete);continue
                        if out.exists():raise ResourceLimit('Partial episode preserved; explicit engineering audit required before retry')
                        atomic(worker/'progress.json',dict(stage='running',round=stage,batch=batch_id,task_id=row['task_id'],seed=group['seed'],policy=policy,pid=os.getpid()))
                        result=run_episode(actor,memory,row,policy,group['seed'],profile,gate,out,ledger)
                        append(worker/'episodes.jsonl',result);results.append(result)
                atomic(worker/(batch_id+'.json'),dict(stage='complete',groups=batch,episodes=len(results),
                    seconds=time.perf_counter()-begin,ledger=ledger.state,batch_start=ledger.batch_start))
            atomic(worker/('round'+str(stage)+'_complete.json'),dict(stage='complete',groups=len(groups),pid=os.getpid()))
        atomic(worker/'progress.json',dict(stage='complete',ledger=ledger.state,pid=os.getpid()))
    except BaseException as error:
        atomic(worker/'error.json',dict(error_type=type(error).__name__,error=str(error),ledger=ledger.state,available_at_ns=time.time_ns()))
        raise


if __name__=='__main__':main()
