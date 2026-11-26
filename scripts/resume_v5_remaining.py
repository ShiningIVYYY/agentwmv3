"""Retain38 completed Val tasks; finish2 Val and80 Train inside original caps."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import recover_utility_v5 as previous
from evaluation.gpu_phase_resources import execute as resource_execute
from scripts.run_utility_sampling import read,rows,phase_cost,check
from evaluation.independent_dev import atomic,append
from experience.splits import file_hash

CONFIG='configs/evaluation/utility_expansion_v5_budget_resume.json'
saved_prior=previous.prior;saved_finish=previous.finish;saved_disk=previous.disk_check
base=previous.base;source_episode=base.source_episode;label_state=base.label_state;tasks_for=base.tasks_for


def prior(verify=False):
    cs,ls,ps=saved_prior(verify);cfg=read(ROOT/CONFIG);receipt=read(ROOT/cfg['recovery_receipt']);old=ROOT/receipt['prior_recovery_run']
    for p,h in receipt['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Sealed recovery data changed: '+p)
    complete={x['task_id'] for x in receipt['completed_validation']};remaining=set(receipt['remaining_validation_ids'])
    cs += [c for p in old.glob('new_validation/worker*/cases.jsonl') for c in rows(p) if c['task_id'] in complete or (c['task_id'] in remaining and c['source_policy']=='never')]
    ls += [l for p in old.glob('new_validation/worker*/state_labels.jsonl') for l in rows(p) if l['task_id'] in complete]
    ps += [p for path in old.glob('new_validation/worker*/task_pairs.jsonl') for p in rows(path) if p['task_id'] in complete]
    return cs,ls,ps


def remaining_tasks(phase,split):
    rs=tasks_for(phase,split)
    if phase=='new_validation':
        receipt=read(ROOT/read(ROOT/CONFIG)['recovery_receipt']);ids=set(receipt['remaining_validation_ids']);rs=[r for r in rs if r['task_id'] in ids]
    return rs


def reuse_or_collect(rt,row,role,policy,memory,thresholds):
    cfg=read(ROOT/CONFIG);receipt=read(ROOT/cfg['recovery_receipt'])
    if role=='probe_validation' and policy=='never' and row['task_id'] in receipt['remaining_validation_ids']:
        old=ROOT/receipt['prior_recovery_run'];e=next(x for x in receipt['reuse_never_episodes'] if x['task_id']==row['task_id'])
        steps=[c for p in old.glob('new_validation/worker*/source_steps.jsonl') for c in rows(p) if c['episode_id']==e['episode_id']]
        if [c['case_id'] for c in steps]!=e['case_ids']:raise ValueError('Retained complete Never source mismatch')
        result=dict(e,reused_observed_source_episode=True,observed_original_cost=e['cost'],cost={k:0 for k in e['cost']})
        append(rt.out/'reused_sources.jsonl',dict(task_id=row['task_id'],episode_id=e['episode_id'],source=str(old),no_new_model_or_native_requests=True))
        return steps,result
    return source_episode(rt,row,role,policy,memory,thresholds)


base.prior=prior;base.tasks_for=remaining_tasks;base.source_episode=reuse_or_collect
previous.prior=prior;previous.CONFIG=CONFIG


def disk_check(run,cfg):
    saved_disk(run,cfg)
    import subprocess
    paths=[ROOT/cfg['original_run'],ROOT/read(ROOT/cfg['recovery_receipt'])['prior_recovery_run'],run]
    if sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)>cfg['maximum_run_bytes']:raise RuntimeError('Combined original disk cap')


def execute(root,run,phase,cfg,m,entry,unused_disk):
    if 'scripts/resume_v5_remaining.py' not in m['source_hashes']:
        for p in ['scripts/resume_v5_remaining.py','docs/utility_expansion_v5/budget_resume_tests.json']:
            m['source_hashes'][p]=file_hash(ROOT/p)
        atomic(run/'manifest.json',m)
        imported=run/'new_validation/worker_retained';imported.mkdir(parents=True)
        for row in read(ROOT/cfg['recovery_receipt'])['completed_validation']:append(imported/'completed_tasks.jsonl',row)
    shadow=dict(cfg,gpus=cfg['gpus'][:cfg['phase_maximum_gpus'][phase]])
    return resource_execute(root,run,phase,shadow,m,Path(__file__),disk_check)


def finish(run,cfg):
    saved_finish(run,cfg);s=read(run/'summary.json');receipt=read(ROOT/cfg['recovery_receipt'])
    s.update(retained_densification_cost=receipt['densification_cost'],retained_validation_cost=receipt['recovery1_cost'],
        retained_complete_validation_tasks=38,completed_never_episodes_reused=2,source_first_label_second=True)
    atomic(run/'summary.json',s);atomic(ROOT/'docs/utility_expansion_v5/results/summary.json',s)
    with (ROOT/'docs/UTILITY_EXPANSION_V5_REPORT.md').open('a') as f:
        f.write('\nBudget continuation retained38 completed Val tasks and two complete Never episodes. Only unfinished Always sources restarted at fixed seeds; both partial Always datasets excluded. Source trajectories were collected before label rollouts to protect fixed-task coverage. Whole V5 costs above retain all prior unsuccessful work; no extra budget/time.\n')


previous.execute=execute;previous.finish=finish;previous.disk_check=disk_check


def worker(run,phase,index):
    root_cfg=read(ROOT/CONFIG);path=run/(phase+'_runtime_config.json');cfg=read(path);assignment=read(run/(phase+'_gpu_assignment.json'))
    if file_hash(path)!=assignment['config_sha256'] or set(cfg)!=set(root_cfg) or any(cfg[k]!=v for k,v in root_cfg.items() if k!='gpus'):raise ValueError('Only actual GPU allocation may differ')
    if cfg['gpus']!=assignment['gpus']:raise ValueError('GPU mapping mismatch')
    base.CONFIG=str(path.relative_to(ROOT));queued=[];done=[];original_append=base.append;original_atomic=base.atomic;original_label=base.label_state
    def queue(*args):queued.append(args)
    def append_source(p,value):
        if Path(p).name=='completed_tasks.jsonl':done.append(value);return original_append(Path(p).with_name('source_tasks_complete.jsonl'),value)
        return original_append(p,value)
    def atomic_source(p,value):
        if Path(p).name=='complete.json':return original_atomic(Path(p).with_name('source_pass_complete.json'),value)
        return original_atomic(p,value)
    base.label_state=queue;base.append=append_source;base.atomic=atomic_source
    out=run/phase/('worker'+str(index));runtime=None
    try:
        base.worker(run,phase,index)
        for row in done:
            args_for_task=[args for args in queued if args[1]['task_id']==row['task_id']]
            for args in args_for_task:
                runtime=args[0];check(read(run/'manifest.json'))
                atomic(out/'progress.json',dict(stage='counterfactual_after_source_pass',task=row['task_id'],case=args[2]['case_id']))
                label_state(*args)
            append(out/'completed_tasks.jsonl',row)
        ledger=runtime.ledger.state if runtime else read(out/'source_pass_complete.json')['ledger']
        atomic(out/'complete.json',dict(status='complete',source_first_label_second=True,ledger=ledger))
    except BaseException as e:
        existing=read(out/'error.json') if (out/'error.json').exists() else {}
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=runtime.ledger.state if runtime else existing.get('ledger')));raise
    finally:
        base.label_state=original_label;base.append=original_append;base.atomic=original_atomic


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase');p.add_argument('--worker',type=int);a=p.parse_args();run=Path(a.run).absolute()
    if a.worker is not None:worker(run,a.phase,a.worker)
    else:
        if read(ROOT/'docs/utility_expansion_v5/budget_resume_tests.json')['status']!='pass':raise ValueError('Tests not passed')
        previous.launch(run)
