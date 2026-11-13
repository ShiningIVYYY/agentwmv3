"""Bounded two-worker supplement; actual Phase5.9 runtime is reused byte-for-byte."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash
from evaluation.independent_dev import atomic,append


def frozen(profile):
    base=json.loads((ROOT/profile['baseline_profile']).read_text())
    assert file_hash(ROOT/profile['baseline_profile'])==profile['baseline_profile_sha256']
    for k,v in base.items():
        if k in ('purpose','limits','worker_limits','wm_invocation_budgets'):continue
        assert profile[k]==v,k
    assert profile['wm_invocation_budgets']==dict(base['wm_invocation_budgets'],experience=5)
    path=ROOT/profile['frozen_runtime'];assert file_hash(path)==profile['frozen_runtime_sha256']
    spec=importlib.util.spec_from_file_location('phase59_exact_external_runtime',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    split=mod.verify_frozen(ROOT,base)
    return mod,split


def worker(run,index,model_dir):
    from internal_signals.qwen35_actor import Qwen35Actor
    from experience.memory import ExperienceMemory
    m=json.loads((run/'manifest.json').read_text());p=m['profile'];old,split=frozen(p)
    for n,h in m['new_source_hashes'].items():assert file_hash(ROOT/n)==h,n
    d=run/('worker'+str(index));d.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(old.ResourceLimit('Worker terminated')))
    ledger=old.Ledger(d/'ledger.json',run/'STOP',m['deadline_unix'],p['worker_limits'],p['worker_limits'])
    before=time.perf_counter();actor=Qwen35Actor(ROOT/p['actor_config'],model_dir)
    gate=json.loads((ROOT/p['gate_config']).read_text());assert actor.fingerprint==gate['internal_signature']['config_sha256']
    atomic(d/'model.json',dict(model_id=actor.config['model_id'],revision=actor.config['model_revision'],loading_info=actor.loading_info,
        config_sha256=actor.fingerprint,backend=actor.config['backend'],dtype=actor.config['dtype'],load_seconds=time.perf_counter()-before,
        shared_actor_reviewer_weights=True,original_reviewer=True,internal_extraction=False,pid=os.getpid()))
    memory=ExperienceMemory(json.loads((ROOT/p['memory_snapshot']).read_text()),split)
    for group in m['schedule']:
        if group['worker']!=index:continue
        row=next(r for r in p['tasks'] if r['task_id']==group['task_id'])
        ep=run/'episodes'/row['task_id']/('seed'+str(group['seed']))/'experience'
        if (ep/'complete.json').exists():continue
        if ep.exists():raise ValueError('Partial attempt retained; manual receipt audit required')
        old.run_episode(actor,memory,row,'experience',group['seed'],p,gate,ep,ledger)
        atomic(d/'progress.json',dict(stage='running',task=row['task_id'],seed=group['seed'],completed=len(list((run/'episodes').rglob('complete.json')))))
    frozen(p)
    atomic(d/'complete.json',dict(status='complete',ledger=ledger.state))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model-dir',default='checkpoints/qwen35_9b');ap.add_argument('--gpus',default='2,6');ap.add_argument('--worker',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    run=Path(a.run).absolute()
    if a.worker is not None:worker(run,a.worker,a.model_dir);return
    p=json.loads((ROOT/'configs/evaluation/phase59_external_only_v1.json').read_text());old,_=frozen(p)
    gpus=a.gpus.split(',');assert len(gpus)==2 and len(set(gpus))==2
    for g in gpus:
        free=int(subprocess.check_output(['nvidia-smi','--id='+g,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<22000:raise ValueError('GPU not free; no jobs displaced')
    hashes={n:file_hash(ROOT/n) for n in ['scripts/run_external_only.py','scripts/prepare_external_only.py','configs/evaluation/phase59_external_only_v1.json']}
    if run.exists():
        assert a.resume
        m=json.loads((run/'manifest.json').read_text());assert m['profile']==p and m['new_source_hashes']==hashes
        assert not (run/'complete.json').exists();(run/'STOP').unlink(missing_ok=True)
    else:
        run.mkdir(parents=True);schedule=[]
        for seed in p['rollout_seeds']:
            for row in p['tasks']:schedule.append(dict(task_id=row['task_id'],seed=seed,worker=len(schedule)%2))
        m=dict(profile=p,new_source_hashes=hashes,source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            schedule=schedule,deadline_unix=time.time()+p['limits']['max_wall_seconds'],gpu_ids=gpus,started_ns=time.time_ns(),
            baseline_runtime_reused_exactly=True,only_added_policy='experience',old_four_policies_not_regenerated=True)
        atomic(run/'manifest.json',m)
    atomic(run/'assets_verified.json',old.verify_model_assets(ROOT,a.model_dir))
    children=[];logs=[];started=time.perf_counter()
    try:
        for index,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(run/f'worker{index}.log').open('ab');logs.append(log)
            child=subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--model-dir',a.model_dir,'--worker',str(index)],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);append(run/'owned_workers.jsonl',dict(pid=child.pid,gpu=gpu,index=index))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Worker failure; preserve partial attempts')
            if time.time()>m['deadline_unix']:raise RuntimeError('Whole-run budget exhausted')
            atomic(run/'progress.json',dict(stage='running',completed=len(list((run/'episodes').rglob('complete.json')))));time.sleep(5)
        assert all(c.returncode==0 for c in children)
        assert len(list((run/'episodes').rglob('complete.json')))==24
        frozen(p);atomic(run/'complete.json',dict(status='complete',episodes=24,seconds=time.perf_counter()-started,finished_ns=time.time_ns()))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'STOP',dict(reason=str(e)));raise
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait(timeout=10)
        for log in logs:log.close()


if __name__=='__main__':main()
