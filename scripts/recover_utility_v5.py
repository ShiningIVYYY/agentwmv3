"""Resume only fixed80Train/40Val; preserve old source and original deadline/costs."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_utility_expansion_v5 as base
from scripts.run_utility_sampling import read,rows,phase_cost,check
from evaluation.independent_dev import atomic,append
from evaluation.gpu_phase_resources import execute
from experience.splits import file_hash

CONFIG='configs/evaluation/utility_expansion_v5_recovery.json'
original_prior=base.prior;original_finish=base.finish


def prior(verify=False):
    cs,ls,ps=original_prior(verify)
    receipt=read(ROOT/'docs/utility_expansion_v5/recovery_receipt.json')
    for p,h in receipt['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Stopped V5 raw artifact changed: '+p)
    old=ROOT/receipt['prior_run']
    ls += [l for p in (old/'densify_train').glob('worker*/state_labels.jsonl') for l in rows(p)]
    return cs,ls,ps


base.CONFIG=CONFIG;base.prior=prior


def disk_check(run,cfg):
    base.disk_check(run,cfg)
    paths=[ROOT/cfg['original_run'],run]
    total=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)
    if total>cfg['maximum_run_bytes']:raise RuntimeError('Original V5+recovery disk cap reached')


def finish(run,cfg):
    original_finish(run,cfg)
    s=read(run/'summary.json');receipt=read(ROOT/cfg['recovery_receipt'])
    s['recovery_only_cost']=phase_cost(run)
    for key in ['new_known_cost','known_cumulative_cost','conservative_cumulative_cost']:
        for metric,value in receipt['prior_actual_cost'].items():s[key][metric]+=value
    s.update(resource_recovery=True,retained_densification_labels=294,retained_densification_task_operations=100,
        retained_densification_cost=receipt['prior_actual_cost'],original_deadline_unix=cfg['original_deadline_unix'],
        failed_load_compute_note=receipt['new_train_failed_load_compute'])
    atomic(run/'summary.json',s);atomic(ROOT/'docs/utility_expansion_v5/results/summary.json',s)
    (ROOT/'docs/UTILITY_EXPANSION_V5_REPORT.md').write_text('\n'.join(['# Utility expansion v5 with resource recovery','',
        'Retain completed100 old-task densification operations and294 labels, including140 budget-Unknowns. Run only fixed80newTrain/40newValidation; no old-task rerun.',
        '', 'Root cause: external GPU use during the phase-switch reload. New per-phase free-card checks/advisory leases reduce this race; other users do not honor our leases. No external jobs stopped.',
        '', 'Combined labels: '+json.dumps(s['by_role']),'', 'Training: '+s['training']['status'],
        '', 'Whole V5 confirmed cost (old phase + resumed phases + one native preflight): '+json.dumps(s['new_known_cost']),
        '', 'Cumulative recorded utility-sampling cost: '+json.dumps(s['known_cumulative_cost']),
        '', 'Conservative cumulative cost: '+json.dumps(s['conservative_cumulative_cost']),
        '', 'Original64M-token/72h budget preserved; interrupted model loading used compute but issued no LLM/native requests. Precise failed-load GPU time is unavailable.',
        '', 'No final Test/fresh Dev or automatic Gate deployment. [Full results](utility_expansion_v5/results/summary.json).'])+'\n')


def launch(run):
    cfg=read(ROOT/CONFIG);receipt=read(ROOT/cfg['recovery_receipt'])
    if run.exists():raise ValueError('Fresh audited recovery only')
    if len(cfg['phase_order'])!=2 or set(cfg['phase_order'])!={'new_train','new_validation'}:raise ValueError('Only remaining fixed phases allowed')
    if time.time()>=cfg['original_deadline_unix']:raise TimeoutError('Original deadline expired; no reset')
    prior(verify=True)
    if read(ROOT/'docs/utility_expansion_v5/recovery_tests.json')['status']!='pass':raise ValueError('Recovery tests missing')
    paths=list(read(ROOT/'outputs/utility_expansion_v5/manifest.json')['source_hashes'])
    paths += [CONFIG,cfg['recovery_receipt'],'scripts/recover_utility_v5.py','evaluation/gpu_phase_resources.py','docs/utility_expansion_v5/recovery_tests.json']
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in set(paths)},
        started_unix=time.time(),deadline_unix=cfg['original_deadline_unix'],maximum_concurrency=len(cfg['gpus']),actual_gpus_by_phase='*_gpu_assignment.json',profile=cfg,retained_prior=receipt)
    run.mkdir(parents=True);atomic(run/'manifest.json',m)
    imported=run/'densify_train/worker_imported';imported.mkdir(parents=True)
    for r in receipt['completed_densification']:append(imported/'completed_tasks.jsonl',r)
    def stop(*_):raise InterruptedError('Owned recovery stopped/original deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,m['deadline_unix']-time.time())
    try:
        for phase in cfg['phase_order']:execute(ROOT,run,phase,cfg,m,Path(__file__),disk_check)
        check(m);finish(run,cfg);atomic(run/'progress.json',dict(stage='complete',summary='summary.json'))
        try:
            allowed=['docs/UTILITY_EXPANSION_V5_REPORT.md','docs/utility_expansion_v5/results']
            dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
            if any(not any(x[3:]==p or x[3:].startswith(p+'/') for p in allowed) for x in dirty):raise RuntimeError('Unrelated worktree edits block publication')
            subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
            subprocess.run(['git','commit','-m','Report retained utility labels and resumed fixed Train Validation cohort'],check=True)
            subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20',GIT_TERMINAL_PROMPT='0'))
            atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        except Exception as e:atomic(run/'publication.json',dict(status='needs_review',message=str(e)))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase');p.add_argument('--worker',type=int);a=p.parse_args()
    if a.worker is None:launch(Path(a.run).absolute())
    else:
        run=Path(a.run).absolute();path=run/(a.phase+'_runtime_config.json')
        assignment=read(run/(a.phase+'_gpu_assignment.json'));phase_cfg=read(path);root_cfg=read(ROOT/CONFIG)
        if file_hash(path)!=assignment['config_sha256']:raise ValueError('Runtime allocation changed')
        if set(phase_cfg)!=set(root_cfg) or any(phase_cfg[k]!=v for k,v in root_cfg.items() if k!='gpus'):raise ValueError('Runtime allocation changed a method parameter')
        if phase_cfg['gpus']!=assignment['gpus'] or not 1<=len(phase_cfg['gpus'])<=4:raise ValueError('Invalid actual GPU count')
        base.CONFIG=str(path.relative_to(ROOT));base.worker(run,a.phase,a.worker)
