"""Bounded TRAIN-state densification and fresh outcome-blind Validation expansion."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from evaluation.aligned_runtime import AlignedRuntime
from evaluation.recovery_collection import source_episode
from evaluation.utility_labeling_v3 import label_state
from evaluation.utility_sampling_v5 import context_key,changed_candidates,train_selection,validation_selection
from experience.alfworld_memory import AlfExperienceMemory
from experience.splits import file_hash
from scripts.run_recovery_v4 import all_records as recovery_records
from scripts.run_utility_expansion import audit_records
from scripts.run_utility_sampling import read,rows,phase_cost,check

CONFIG='configs/evaluation/utility_expansion_v5.json'


def prior(verify=False):
    receipt=read(ROOT/'docs/utility_expansion_v5/prior_corpus.json')
    for p,h in receipt['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Immutable prior source changed: '+p)
    if verify:
        for p,h in receipt['feature_hashes'].items():
            if file_hash(ROOT/p)!=h:raise ValueError('Prior feature changed: '+p)
    cfg=read(ROOT/'configs/evaluation/recovery_v4.json')
    cs,ls,ps=recovery_records(ROOT/'outputs/recovery_v4',cfg,verify=verify)
    if len(cs)!=receipt['cases'] or len(ls)!=receipt['labels']:raise ValueError('Prior record count changed')
    return cs,ls,ps


def all_records(run,verify=False):
    cs,ls,ps=prior(verify)
    cs += [c for p in run.glob('*/worker*/cases.jsonl') for c in rows(p)]
    ls += [c for p in run.glob('*/worker*/state_labels.jsonl') for c in rows(p)]
    ps += [c for p in run.glob('*/worker*/task_pairs.jsonl') for c in rows(p)]
    return cs,ls,ps


def tasks_for(phase,split):
    if phase=='densify_train':
        new=set(split['new_train_task_ids']);return [r for r in split['partitions']['probe_train'] if r['task_id'] not in new]
    role='probe_train' if phase=='new_train' else 'probe_validation'
    ids=set(split['new_train_task_ids' if role=='probe_train' else 'new_validation_task_ids'])
    return [r for r in split['partitions'][role] if r['task_id'] in ids]


def worker(run,phase,index):
    cfg=read(ROOT/CONFIG);split=read(ROOT/cfg['split']);m=read(run/'manifest.json');check(m)
    out=run/phase/('worker'+str(index));out.mkdir(parents=True);rt=None
    def stop(*_):raise InterruptedError('Owned utility-v5 worker stopped')
    signal.signal(signal.SIGTERM,stop)
    try:
        c=copy.deepcopy(cfg);c['stage_limits']=read(run/(phase+'_budgets.json'))[index]
        rt=AlignedRuntime(ROOT,out,c,m['deadline_unix']);memory=AlfExperienceMemory(read(ROOT/cfg['memory_snapshot']),split)
        oldcases,oldlabels,oldpairs=prior();lookup={c['case_id']:c for c in oldcases}
        blocked={context_key(lookup[l['case_id']]) for l in oldlabels}
        for ordinal,row in enumerate(tasks_for(phase,split)):
            if ordinal%len(cfg['gpus'])!=index:continue
            check(m)
            if shutil.disk_usage(ROOT).free<cfg['minimum_free_disk_bytes']:raise RuntimeError('Free disk safety floor reached')
            role='probe_validation' if phase=='new_validation' else 'probe_train'
            if phase=='densify_train':
                collections={p:sorted([x for x in oldcases if x['task_id']==row['task_id'] and x['source_policy']==p],key=lambda x:x['step']) for p in ['never','always']}
                results=next(p for p in oldpairs if p['task_id']==row['task_id'])
                selected=changed_candidates(collections['always'],role,row['task_id'],blocked,cfg['extra_train_states_per_task'])
                append(out/'dense_selections.jsonl',dict(task_id=row['task_id'],role=role,selected=selected,source_episodes_reused=True))
            else:
                collections={};results={}
                for policy in ['never','always']:
                    atomic(out/'progress.json',dict(stage='source_episode',task=row['task_id'],role=role,policy=policy))
                    collections[policy],results[policy]=source_episode(rt,row,role,policy,memory,cfg['sampling_nll_thresholds'])
                if role=='probe_train':selected,first=train_selection(collections['never'],collections['always'],row['task_id'],cfg['extra_train_states_per_task'])
                else:selected=validation_selection(collections['never']+collections['always'],row['task_id'],cfg['validation_states_per_window']);first=None
                append(out/'task_pairs.jsonl',dict(task_id=row['task_id'],role=role,never=results['never'],always=results['always'],first_divergence=first,selected=selected))
            local={x['case_id']:x for cs in collections.values() for x in cs}
            for s in selected:
                case=local[s['case_id']]
                if not case.get('feature_valid',True):raise ValueError('Unusable state selected')
                atomic(out/'progress.json',dict(stage='state_counterfactual',task=row['task_id'],role=role,case=case['case_id']))
                label_state(rt,row,case,s,collections[case['source_policy']],results[case['source_policy']])
            append(out/'completed_tasks.jsonl',dict(task_id=row['task_id'],role=role,selected=len(selected)))
        atomic(out/'complete.json',dict(status='complete',ledger=rt.ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=rt.ledger.state if rt else None));raise


def disk_check(run,cfg):
    if shutil.disk_usage(ROOT).free<cfg['minimum_free_disk_bytes']:raise RuntimeError('Free disk safety floor reached')
    used=int(subprocess.check_output(['du','-sb',str(run)],text=True).split()[0])
    if used>cfg['maximum_run_bytes']:raise RuntimeError('New-run disk budget reached')


def execute(run,phase,cfg,m):
    spent=phase_cost(run);pre=read(ROOT/'outputs/utility_expansion_v5_preflight/complete.json')
    stage=cfg['phase_limits'][phase]
    limits={k:min(stage[k],cfg['global_limits'][k]-spent.get(k,0)-pre.get(k,0)) for k in stage}
    if any(v<=0 for v in limits.values()):raise RuntimeError('New batch remaining budget exhausted')
    n=len(cfg['gpus']);atomic(run/(phase+'_budgets.json'),[{k:v//n for k,v in limits.items()} for _ in range(n)])
    children=[];logs=[];last_disk=0
    try:
        for i,gpu in enumerate(cfg['gpus']):
            log=(run/(phase+'_worker'+str(i)+'.log')).open('ab');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            children.append(subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--phase',phase,'--worker',str(i)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        atomic(run/(phase+'_owned_workers.json'),[dict(pid=p.pid,gpu=cfg['gpus'][i]) for i,p in enumerate(children)])
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Owned worker failed; preserve all receipts')
            if time.time()>m['deadline_unix']:raise TimeoutError('New72h batch window expired')
            if time.time()-last_disk>120:disk_check(run,cfg);last_disk=time.time()
            atomic(run/'progress.json',dict(stage=phase,new_known_cost=phase_cost(run),
                completed_task_operations=sum(len(rows(p)) for p in run.glob('*/worker*/completed_tasks.jsonl')),
                new_labels=sum(len(rows(p)) for p in run.glob('*/worker*/state_labels.jsonl')),
                reused_old_source_tasks=100,new_training_tasks=80,new_validation_tasks=40))
            time.sleep(15)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Worker exit failure')
    finally:
        if any(p.poll() is None for p in children):atomic(run/'STOP',dict(reason='controller_stop'))
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
        atomic(run/(phase+'_supervisor.json'),dict(workers_exited=[p.poll() is not None for p in children]))


def finish(run,cfg):
    cs,ls,ps=all_records(run,verify=True);split=read(ROOT/cfg['split']);audit=audit_records(cs,ls,split)
    for phase,expected in [('densify_train',100),('new_train',80),('new_validation',40)]:
        done=[r for p in (run/phase).glob('worker*/completed_tasks.jsonl') for r in rows(p)]
        if len(done)!=expected or len({r['task_id'] for r in done})!=expected:raise ValueError('Incomplete fixed task denominator')
    for p in run.glob('*/worker*/ledger.json'):
        l=read(p);calls=rows(p.parent/'generation_calls.jsonl')
        if l['pending_model'] or l['pending_transition']:raise ValueError('Unknown new receipt; never fabricate cost')
        if len(calls)!=l['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=l['tokens']:raise ValueError('New journal mismatch')
    if len({p['task_id'] for p in ps if p['role']=='probe_train'})!=180:raise ValueError('Expected180 training task pairs')
    if len({p['task_id'] for p in ps if p['role']=='probe_validation'})!=52:raise ValueError('Expected52 validation task pairs')
    from internal_signals.linear_utility_training import train_head
    d=run/'training';d.mkdir();atomic(d/'data_audit.json',audit)
    profile=read(ROOT/cfg['probe_training_profile']);trained=train_head(ROOT,d,ls,'utility_help',profile['layers'],profile)
    atomic(d/'selection.json',dict(heads={'utility_help':trained},deployment_allowed=False,no_actor_updates=True,scientific_effectiveness_validated=False))
    pre=read(ROOT/'outputs/utility_expansion_v5_preflight/complete.json');new=phase_cost(run)
    new={k:v+pre.get(k,0) for k,v in new.items()};receipt=read(ROOT/'docs/utility_expansion_v5/prior_corpus.json')
    cumulative={k:v+receipt['known_cumulative_cost'][k] for k,v in new.items()}
    upper={k:v+receipt['conservative_cumulative_cost'][k] for k,v in new.items()}
    byrole={role:dict(Counter(l['label'] for l in ls if l['role']==role)) for role in ['probe_train','probe_validation','gate_calibration']}
    newlabels=[l for p in run.glob('*/worker*/state_labels.jsonl') for l in rows(p)]
    result=dict(status='complete',data_audit=audit,by_role=byrole,new_by_role={r:dict(Counter(l['label'] for l in newlabels if l['role']==r)) for r in byrole},
        training=trained,new_known_cost=new,known_cumulative_cost=cumulative,conservative_cumulative_cost=upper,
        unknown_prior_receipts=receipt['unknown_prior_receipts'],new_task_family_counts=split['new_family_counts'],
        new_state_index=[{k:l.get(k) for k in ['case_id','task_id','role','step','label','deltas','source_policy','feature_file','selection']} for l in newlabels],
        old_train_source_tasks_reused=100,new_train_tasks=80,new_validation_tasks=40,fresh_independent_dev_reserved=12,
        fixed_cohorts_completed=True,no_gate_deployment=True,no_test_or_fresh_dev_rollouts=True,
        caveats=['TRAIN accepted-revision enrichment is not natural prevalence','New untouched goals span only3 families',
            'Finite3/5 continuation seeds are not statistical significance','No TaskScore/onlineProbe utility claim before independent evaluation'])
    atomic(run/'summary.json',result);dest=ROOT/'docs/utility_expansion_v5/results';dest.mkdir(parents=True,exist_ok=True);atomic(dest/'summary.json',result)
    (ROOT/'docs/UTILITY_EXPANSION_V5_REPORT.md').write_text('\n'.join(['# Training utility and independent Validation extension v5','',
        'Old100 TRAIN source tasks reused;80 new TRAIN and40 new Validation tasks fixed before outputs.12 new Dev goals reserved, never reset.',
        '', 'Combined label counts: '+json.dumps(byrole),'', 'New labels: '+json.dumps(result['new_by_role']),
        '', 'Linear Help training: '+trained['status'],'', 'New cost incl native preflight: '+json.dumps(new),
        '', 'Cumulative confirmed cost: '+json.dumps(cumulative),'', 'Conservative upper accounting retains2 historical unknown outputs: '+json.dumps(upper),
        '', 'TRAIN sampling is enriched; Validation uses outcome-blind uniform windows. Old and new sampling strata remain identifiable.',
        '', 'No automatic Gate deployment, final Test or fresh-Dev rollout. Full metrics: [JSON](utility_expansion_v5/results/summary.json).'])+'\n')


def launch(run,autopublish):
    cfg=read(ROOT/CONFIG)
    if run.exists():raise ValueError('Fresh Run ID only; no duplicate restart')
    if read(ROOT/'docs/utility_expansion_v5/tests.json')['status']!='pass':raise ValueError('CPU data/selection tests not passed')
    pre=read(ROOT/'outputs/utility_expansion_v5_preflight/complete.json')
    if pre['status']!='pass' or pre['split_sha256']!=cfg['split_sha256']:raise ValueError('Native preflight mismatch')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256'] or file_hash(ROOT/cfg['memory_snapshot'])!=cfg['memory_sha256']:raise ValueError('Split or Memory changed')
    prior(verify=True)
    if len(cfg['gpus'])>4:raise ValueError('At most4 GPUs authorized')
    for gpu in cfg['gpus']:
        if int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<22000:raise ValueError('GPU unavailable; never displace another job')
    if shutil.disk_usage(ROOT).free<cfg['minimum_free_disk_bytes']:raise RuntimeError('Insufficient free disk')
    paths=list(read(ROOT/'outputs/recovery_v4/manifest.json')['source_hashes'])
    paths += [CONFIG,cfg['split'],'docs/utility_expansion_v5/prior_corpus.json','docs/utility_expansion_v5/tests.json','scripts/run_utility_expansion_v5.py','evaluation/utility_sampling_v5.py',cfg['probe_training_profile']]
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in set(paths)},
        started_unix=time.time(),deadline_unix=time.time()+cfg['wall_seconds'],gpus=cfg['gpus'],profile=cfg,native_preflight=pre)
    run.mkdir(parents=True);atomic(run/'manifest.json',m)
    def stop(*_):raise InterruptedError('Owned utility-v5 controller stopped/deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,cfg['wall_seconds'])
    try:
        for phase in ['densify_train','new_train','new_validation']:execute(run,phase,cfg,m)
        check(m);atomic(run/'progress.json',dict(stage='audit_and_conditional_linear_training'));finish(run,cfg)
        atomic(run/'progress.json',dict(stage='complete',summary='summary.json'))
        if autopublish:
            try:
                allowed=['docs/UTILITY_EXPANSION_V5_REPORT.md','docs/utility_expansion_v5/results']
                dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
                if any(not any(l[3:]==p or l[3:].startswith(p+'/') for p in allowed) for l in dirty):raise RuntimeError('Unrelated edits block publication')
                subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
                subprocess.run(['git','commit','-m','Report utility-label and independent Validation expansion v5'],check=True)
                subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20',GIT_TERMINAL_PROMPT='0'))
                atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
            except Exception as e:atomic(run/'publication.json',dict(status='needs_review',message=str(e),results_preserved=True))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase');p.add_argument('--worker',type=int);p.add_argument('--auto-publish',action='store_true');a=p.parse_args()
    if a.worker is not None:worker(Path(a.run),a.phase,a.worker)
    else:launch(Path(a.run).absolute(),a.auto_publish)
