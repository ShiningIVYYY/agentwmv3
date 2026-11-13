"""Scheduling-only continuation, <=4 GPUs, frozen calibration and exact episode AST."""
import argparse,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,Ledger
from experience.splits import file_hash
from scripts.build_alfworld_episode import validate


def verify(source):
    m=json.loads((source/'manifest.json').read_text())
    for n,h in m['source_hashes'].items():
        if file_hash(ROOT/n)!=h:raise ValueError('Original method changed: '+n)
    freeze=json.loads((source/'threshold_freeze.json').read_text())
    if freeze['gate_sha256']!=file_hash(source/'gate_config.json'):raise ValueError('Threshold changed')
    if freeze['calibration_sha256']!=file_hash(source/'calibration_steps.jsonl'):raise ValueError('Calibration changed')
    if file_hash(ROOT/'outputs/alfworld_memory_v2/snapshot.json')!=m['memory_sha256']:raise ValueError('Experience changed')
    identity=json.loads((ROOT/'docs/alfworld_pilot/parallel_episode_identity.json').read_text())
    digest=validate((ROOT/'scripts/run_alfworld_pilot.py').read_text(),(ROOT/'evaluation/alfworld_episode.py').read_text())
    if identity['episode_ast_sha256']!=digest:raise ValueError('Episode AST changed')
    return m


def worker(run,index):
    from internal_signals.qwen35_actor import Qwen35Actor
    from experience.alfworld_memory import AlfExperienceMemory
    from evaluation.alfworld_episode import run_episode
    m=json.loads((run/'manifest.json').read_text());source=Path(m['source_run']);old=verify(source);p=old['profile'];split=old['split']
    for n,h in m['new_source_hashes'].items():
        if file_hash(ROOT/n)!=h:raise ValueError('Scheduling source changed: '+n)
    out=run/f'worker{index}';out.mkdir(exist_ok=True)
    def stop(*_):raise InterruptedError('Owned parallel supervisor stop')
    signal.signal(signal.SIGTERM,stop)
    caps=m['worker_caps'][index];ledger=Ledger(out/'ledger.json',run/'STOP',m['deadline_unix'],caps,caps)
    started=time.perf_counter();actor=Qwen35Actor(ROOT/p['actor_config'],ROOT/'checkpoints/qwen35_9b');actor.evaluation_left_truncation=False
    old_model=json.loads((source/'model.json').read_text())
    if actor.fingerprint!=old_model['config_sha256']:raise ValueError('Model configuration differs')
    atomic(out/'model.json',dict(model_id=actor.config['model_id'],revision=actor.config['model_revision'],config_sha256=actor.fingerprint,loading_info=actor.loading_info,
        load_seconds=time.perf_counter()-started,actor_reviewer_shared=True,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    memory=AlfExperienceMemory(json.loads((ROOT/'outputs/alfworld_memory_v2/snapshot.json').read_text()),split)
    gate=json.loads((source/'gate_config.json').read_text());native=json.loads((ROOT/'docs/alfworld_pilot/native_goal_identity.json').read_text());native_goals={r['task_id']:r for r in native}
    if file_hash(ROOT/'docs/alfworld_pilot/native_goal_identity.json')!=old['native_goals_sha256']:raise ValueError('Native task identity changed')
    episodes=[];a=SimpleNamespace(env_python=old['env_python'],data_root=old['data_root'])
    try:
        for entry in m['schedule']:
            if entry.get('worker')!=index:continue
            row=next(r for r in split['evaluation_tasks'] if r['task_id']==entry['task_id'])
            run_episode(row=row,policy=entry['policy'],seed=entry['seed'],actor=actor,memory=memory,gate_config=gate,p=p,out=out,a=a,
                ledger=ledger,native_goals=native_goals,episodes=episodes)
        verify(source)
        atomic(out/'complete.json',dict(status='complete',episodes=len(episodes),seconds=time.perf_counter()-started,ledger=ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=ledger.state));raise


def weights(entry,p):
    h=p['maximum_interaction_turns'];budget=p['wm_invocation_budgets'][entry['policy']]
    return dict(model_calls=h+budget,transition_queries=1+h+sum(i+2 for i in range(h-budget,h)),audit_capture_bundles=1+h+sum(i+4 for i in range(h-budget,h)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source',default='outputs/alfworld_five_policy_v1');ap.add_argument('--gpus',default='4,5,6,7');ap.add_argument('--worker',type=int);a=ap.parse_args()
    run=Path(a.run).absolute()
    if a.worker is not None:worker(run,a.worker);return
    source=Path(a.source).absolute();old=verify(source);p=old['profile'];prior=json.loads((source/'ledger.json').read_text())
    if prior['pending_model'] or prior['pending_transition']:raise ValueError('Unknown original receipt; do not resume')
    old_super=json.loads(Path(str(source)+'-supervisor.json').read_text())
    if not old_super['model_exited']:raise ValueError('Old model must exit before parallel expansion')
    gpus=a.gpus.split(',')
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus):raise ValueError('At most4 distinct GPUs')
    for gpu in gpus:
        free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<22000:raise ValueError('GPU not free; never displace another job')
    if run.exists():raise ValueError('Fresh parallel Run ID required')
    run.mkdir(parents=True);schedule=[];next_worker=0
    for row in old['split']['evaluation_tasks']:
        for seed in p['rollout_seeds']:
            for policy in p['policies']:
                ep=source/'episodes'/row['task_id']/policy
                entry=dict(task_id=row['task_id'],policy=policy,seed=seed)
                if (ep/'complete.json').exists():
                    result=json.loads((ep/'complete.json').read_text())
                    if result['steps_sha256']!=file_hash(ep/'steps.jsonl'):raise ValueError('Completed old Episode changed')
                    entry.update(reused_source=str(ep),steps_sha256=result['steps_sha256'])
                else:entry['worker']=next_worker%len(gpus);next_worker+=1
                schedule.append(entry)
    cap_keys=['tokens','model_calls','transition_queries','audit_capture_bundles'];remaining={k:p['limits'][k]-prior[k] for k in cap_keys}
    if any(v<=0 for v in remaining.values()):raise ValueError('Original whole-run resource budget exhausted')
    raw=[]
    for i in range(len(gpus)):
        entries=[e for e in schedule if e.get('worker')==i];parts=[weights(e,p) for e in entries]
        raw.append({k:sum(x[k] for x in parts) for k in ['model_calls','transition_queries','audit_capture_bundles']})
    caps=[{} for _ in gpus]
    for key in cap_keys:
        wk='model_calls' if key=='tokens' else key;total=sum(x[wk] for x in raw);used=0
        for i in range(len(gpus)):
            amount=remaining[key]-used if i==len(gpus)-1 else int(remaining[key]*raw[i][wk]/total)
            caps[i][key]=amount;used+=amount
    base_start=json.loads((ROOT/'outputs/alfworld_five_policy_job_v1.json').read_text())['started_ns']/1e9
    names=['scripts/run_alfworld_parallel.py','scripts/build_alfworld_episode.py','evaluation/alfworld_episode.py']
    manifest=dict(source_run=str(source),source_manifest_sha256=file_hash(source/'manifest.json'),original_spent=prior,profile=p,schedule=schedule,
        worker_caps=caps,gpus=gpus,deadline_unix=base_start+p['limits']['wall_seconds'],
        new_source_hashes={n:file_hash(ROOT/n) for n in names},source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        same_frozen_gate_memory_prompts_seeds=True,recalibration=False,user_authorized_max_gpus=4,resource_change_only=True,started_ns=time.time_ns())
    atomic(run/'manifest.json',manifest);children=[];logs=[];error=None;started=time.perf_counter()
    try:
        for i,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(run/f'worker{i}.log').open('ab');logs.append(log)
            child=subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--worker',str(i)],env=env,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        atomic(run/'owned_workers.json',dict(workers=[dict(pid=c.pid,gpu=gpus[i]) for i,c in enumerate(children)]))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Parallel worker failed; preserve receipts')
            if time.time()>manifest['deadline_unix']:raise TimeoutError('Original2h whole-run wall bound')
            count=len([e for e in schedule if 'reused_source' in e])+len(list(run.glob('worker*/episodes/*/*/complete.json')))
            atomic(run/'progress.json',dict(stage='running',completed=count,total=15,max_gpus=len(gpus)));time.sleep(5)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Worker exit failure')
        count=len([e for e in schedule if 'reused_source' in e])+len(list(run.glob('worker*/episodes/*/*/complete.json')))
        if count!=15:raise ValueError('Missing/duplicate final Episode')
        spent={k:prior[k]+sum(json.loads((run/f'worker{i}'/'ledger.json').read_text())[k] for i in range(len(gpus))) for k in cap_keys}
        if any(spent[k]>p['limits'][k] for k in cap_keys):raise ValueError('Whole-run cap exceeded')
        atomic(run/'complete.json',dict(status='complete',episodes=15,seconds=time.perf_counter()-started,total_resource_spent_including_initial_partial=spent))
        atomic(run/'progress.json',dict(stage='complete',completed=15,total=15,max_gpus=len(gpus)))
    except BaseException as e:error=str(e);atomic(run/'error.json',dict(type=type(e).__name__,message=error));atomic(run/'STOP',dict(reason=error));raise
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait(timeout=10)
        from scripts.launch_alfworld_pilot import native_children
        cleanup=[]
        for i in range(len(children)):
            for pid in native_children(run/f'worker{i}'):
                try:os.killpg(pid,signal.SIGTERM);cleanup.append(pid)
                except ProcessLookupError:pass
        for log in logs:log.close()
        atomic(run/'supervisor_complete.json',dict(error=error,workers_exited=[c.poll() is not None for c in children],owned_native_cleanup_pids=cleanup))


if __name__=='__main__':main()
