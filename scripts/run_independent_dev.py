#!/usr/bin/env python3
"""Two-worker bounded Phase5.9 supervisor; Round2 waits for engineering audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,verify_frozen,verify_model_assets
from experience.splits import file_hash


def schedule(profile):
    result=[]
    for seed in profile['rollout_seeds']:
        for row in profile['tasks']:
            i=len(result)
            result.append(dict(task_id=row['task_id'],seed=seed,worker=i%2,
                round=1 if seed==0 and row['task_id'] in profile['round1_task_ids'] else 2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',default='configs/evaluation/phase59_independent_dev_v1.json')
    p.add_argument('--model-dir',required=True);p.add_argument('--run',required=True);p.add_argument('--gpus',default='6,7')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    run=Path(a.run).resolve();profile=json.loads((ROOT/a.profile).read_text());verify_frozen(ROOT,profile)
    gpus=[int(s) for s in a.gpus.split(',')]
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Exactly two distinct owned GPUs required')
    for gpu in gpus:
        free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<22000:raise ValueError('GPU '+str(gpu)+' is not free; do not displace other jobs')
    code=[ROOT/p for p in ['evaluation/independent_dev.py','scripts/run_independent_dev.py',
        'scripts/run_independent_dev_worker.py','scripts/audit_independent_dev.py','scripts/prepare_independent_dev_eval.py']]
    hashes={str(p.relative_to(ROOT)):file_hash(p) for p in code}
    if run.exists():
        if not a.resume:raise ValueError('Existing run requires explicit --resume')
        old=json.loads((run/'manifest.json').read_text())
        if old['profile']!=profile or old['code_hashes']!=hashes:raise ValueError('Resume source/config mismatch')
        if (run/'complete.json').exists():raise ValueError('Run already complete; do not regenerate episodes')
        (run/'STOP').unlink(missing_ok=True)
        append(run/'resume_events.jsonl',dict(available_at_ns=time.time_ns(),pid=os.getpid()))
    else:
        run.mkdir(parents=True)
        atomic(run/'manifest.json',dict(schema='phase59-independent-run-v1',profile=profile,
            profile_sha256=file_hash(ROOT/a.profile),code_hashes=hashes,schedule=schedule(profile),
            source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            started_at_ns=time.time_ns(),deadline_unix=time.time()+profile['limits']['max_wall_seconds'],gpus=gpus,
            no_training=True,no_test_evaluation=True))
    started=time.perf_counter()
    verified=verify_model_assets(ROOT,Path(a.model_dir))
    atomic(run/'assets_verified.json',verified)
    children=[];logs=[];error=None
    try:
        for index,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            f=(run/('worker'+str(index)+'.log')).open('ab');logs.append(f)
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_independent_dev_worker.py'),
                '--run',str(run),'--worker',str(index),'--model-dir',a.model_dir],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);append(run/'owned_workers.jsonl',dict(worker=index,pid=child.pid,gpu=gpu,started_at_ns=time.time_ns()))
        while not all((run/('worker'+str(i))/'round1_complete.json').exists() for i in range(2)):
            if any(c.poll() is not None for c in children):raise RuntimeError('Worker exited before first-round audit')
            if time.perf_counter()-started>profile['limits']['max_wall_seconds']:raise RuntimeError('Whole run wall budget')
            atomic(run/'progress.json',dict(stage='round1',completed_episodes=len(list((run/'episodes').rglob('complete.json')))))
            time.sleep(5)
        if not (run/'round1_acceptance.json').exists():
            subprocess.run([sys.executable,'scripts/audit_independent_dev.py','--run',str(run),
                '--stage','round1','--output',str(run/'round1_acceptance.json')],cwd=ROOT,check=True)
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Worker failed; stop the bounded run')
            if time.perf_counter()-started>profile['limits']['max_wall_seconds']:raise RuntimeError('Whole run wall budget')
            atomic(run/'progress.json',dict(stage='round2',completed_episodes=len(list((run/'episodes').rglob('complete.json')))))
            time.sleep(5)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Worker exit error')
        subprocess.run([sys.executable,'scripts/audit_independent_dev.py','--run',str(run),
            '--stage','all','--output',str(run/'final_audit.json')],cwd=ROOT,check=True)
        atomic(run/'complete.json',dict(stage='complete',episodes=96,seconds=time.perf_counter()-started,finished_at_ns=time.time_ns()))
        atomic(run/'progress.json',dict(stage='complete',episodes=96))
    except BaseException as caught:
        error=caught;atomic(run/'STOP',dict(error_type=type(caught).__name__,error=str(caught)))
        atomic(run/'error.json',dict(error_type=type(caught).__name__,error=str(caught),available_at_ns=time.time_ns()))
        raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        for f in logs:f.close()


if __name__=='__main__':main()
