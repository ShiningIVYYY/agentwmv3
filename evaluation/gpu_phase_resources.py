"""Per-phase free-GPU checks and advisory leases; never stop external jobs."""
import fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from evaluation.independent_dev import atomic
from scripts.run_utility_sampling import phase_cost,rows


def candidates(text):
    out=[]
    for line in text.strip().splitlines():
        i,free,used=[int(x.strip()) for x in line.split(',')]
        if free>=22000 and used<=128:out.append(i)
    return out


def acquire(count,deadline,run,phase,min_count=None):
    if min_count is None:min_count=count
    if not 1<=min_count<=count<=4:raise ValueError('Invalid GPU concurrency bounds')
    while time.time()<deadline:
        data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
        chosen=[];locks=[]
        for gpu in candidates(data):
            try:
                handle=open('/tmp/agentwmv3-gpu-'+str(gpu)+'.lock','a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except OSError:handle.close();continue
                chosen.append(gpu);locks.append(handle)
            except PermissionError:continue
            if len(chosen)==count:break
        if len(chosen)>=min_count:return chosen,locks
        for handle in locks:handle.close()
        atomic(run/'progress.json',dict(stage='waiting_for_free_gpus',phase=phase,required=count,available=len(chosen),deadline=deadline))
        time.sleep(15)
    raise TimeoutError('Bounded per-phase GPU wait expired; no external jobs displaced')


def execute(root,run,phase,cfg,m,entry,disk_check):
    spent=phase_cost(run);stage=cfg['phase_limits'][phase]
    limits={k:min(stage[k],cfg['global_limits'][k]-spent.get(k,0)) for k in stage}
    if any(v<=0 for v in limits.values()):raise RuntimeError('Original remaining budget exhausted')
    maximum=len(cfg['gpus']);gpus,locks=acquire(maximum,min(m['deadline_unix'],time.time()+cfg['gpu_wait_seconds']),run,phase,cfg.get('minimum_gpus',maximum))
    n=len(gpus)
    from experience.splits import file_hash
    phase_config=run/(phase+'_runtime_config.json');atomic(phase_config,dict(cfg,gpus=gpus))
    atomic(run/(phase+'_gpu_assignment.json'),dict(gpus=gpus,config_sha256=file_hash(phase_config),advisory_leases=True,external_jobs_cannot_be_locked=True))
    atomic(run/(phase+'_budgets.json'),[{k:v//n for k,v in limits.items()} for _ in range(n)])
    children=[];logs=[];last_disk=0
    try:
        for i,gpu in enumerate(gpus):
            free,used=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')]
            if free<22000 or used>128:raise RuntimeError('GPU became busy before spawn; preserve run, never evict')
            log=(run/(phase+'_worker'+str(i)+'.log')).open('ab');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            children.append(subprocess.Popen([sys.executable,'-u',str(entry),'--run',str(run),'--phase',phase,'--worker',str(i)],cwd=root,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        atomic(run/(phase+'_owned_workers.json'),[dict(pid=p.pid,gpu=gpus[i]) for i,p in enumerate(children)])
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Recovery worker failed; no automatic rollout retry')
            if time.time()>m['deadline_unix']:raise TimeoutError('Original72h deadline expired')
            if time.time()-last_disk>120:disk_check(run,cfg);last_disk=time.time()
            atomic(run/'progress.json',dict(stage=phase,new_recovery_cost=phase_cost(run),
                completed_new_tasks=sum(len(rows(p)) for p in run.glob('new_*/worker*/completed_tasks.jsonl')),
                new_labels=sum(len(rows(p)) for p in run.glob('new_*/worker*/state_labels.jsonl')),retained_old_task_operations=100))
            time.sleep(15)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Worker exit failure')
    finally:
        if any(p.poll() is None for p in children):atomic(run/'STOP',dict(reason='owned_controller_stop'))
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=40)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        from scripts.launch_alfworld_pilot import native_children
        for i in range(len(children)):
            for pid in native_children(run/phase/('worker'+str(i))):
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for log in logs:log.close()
        for lock in locks:lock.close()
        atomic(run/(phase+'_supervisor.json'),dict(workers_exited=[p.poll() is not None for p in children]))
