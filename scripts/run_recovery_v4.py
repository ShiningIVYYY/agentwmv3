"""Audited recovery, fixed100-TRAIN-goal collection, optional supervised probes."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from evaluation.aligned_runtime import AlignedRuntime
from evaluation.recovery_collection import source_episode,select_pair,mine_candidates
from evaluation.utility_labeling_v3 import label_state
from evaluation.utility_labels_v3 import SCHEMA
from scripts.run_utility_expansion import prior_records,feasibility,audit_records
from scripts.run_utility_sampling import read,rows,check,phase_cost
from experience.alfworld_memory import AlfExperienceMemory
from experience.splits import file_hash


def imported(cfg,verify=False):
    cs,ls,ps=prior_records(cfg,verify_features=verify);receipt=read(ROOT/'docs/recovery_v4/recovery_receipt.json')
    for p,h in receipt['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Stopped raw artifacts changed: '+p)
    if verify:
        for p,h in receipt['feature_hashes'].items():
            if file_hash(ROOT/p)!=h:raise ValueError('Retained stopped feature changed')
    good=set(receipt['completed_task_ids']);r=ROOT/cfg['failed_run']
    cs += [x for p in r.glob('train_pool/worker*/cases.jsonl') for x in rows(p) if x['task_id'] in good]
    ls += [x for p in r.glob('train_pool/worker*/state_labels.jsonl') for x in rows(p) if x['task_id'] in good]
    ps += [x for p in r.glob('train_pool/worker*/task_pairs.jsonl') for x in rows(p) if x['task_id'] in good]
    return cs,ls,ps


def all_records(run,cfg,verify=False):
    cs,ls,ps=imported(cfg,verify)
    cs += [x for p in run.glob('*/worker*/cases.jsonl') for x in rows(p)]
    ls += [x for p in run.glob('*/worker*/state_labels.jsonl') for x in rows(p)]
    ps += [x for p in run.glob('*/worker*/task_pairs.jsonl') for x in rows(p)]
    return cs,ls,ps


def tasks_for(phase,split):
    if phase=='recover_original':ids=set(split['remaining_original_task_ids']);return [('probe_train',r) for r in split['partitions']['probe_train'] if r['task_id'] in ids]
    if phase=='expand_train':ids=set(split['new_train_task_ids']);return [('probe_train',r) for r in split['partitions']['probe_train'] if r['task_id'] in ids]
    return [(role,r) for role in ['probe_validation','gate_calibration'] for r in split['partitions'][role]]


def worker(run,phase,index):
    cfg=read(ROOT/'configs/evaluation/recovery_v4.json');split=read(ROOT/cfg['split']);m=read(run/'manifest.json');check(m)
    out=run/phase/('worker'+str(index));out.mkdir(parents=True);rt=None
    def stop(*_):raise InterruptedError('Owned recovery worker stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        c=copy.deepcopy(cfg);c['stage_limits']=read(run/(phase+'_budgets.json'))[index]
        deadline=min(m['deadline_unix'],cfg['original_recovery_deadline']) if phase=='recover_original' else m['deadline_unix']
        rt=AlignedRuntime(ROOT,out,c,deadline);memory=AlfExperienceMemory(read(ROOT/cfg['memory_snapshot']),split)
        for ordinal,(role,row) in enumerate(tasks_for(phase,split)):
            if ordinal%len(cfg['gpus'])!=index:continue
            check(m);collections={};results={}
            for policy in ['never','always']:
                atomic(out/'progress.json',dict(stage='source_episode',phase=phase,task=row['task_id'],role=role,policy=policy))
                collections[policy],results[policy]=source_episode(rt,row,role,policy,memory,cfg['sampling_nll_thresholds'])
            selected,first=select_pair(collections['never'],collections['always'],role,row['task_id'])
            selected=mine_candidates(selected,collections['never'],collections['always'],role,results['never']['success'],results['always']['success'])
            append(out/'task_pairs.jsonl',dict(task_id=row['task_id'],role=role,never=results['never'],always=results['always'],first_divergence=first,selected=selected))
            lookup={c['case_id']:c for cs in collections.values() for c in cs}
            for s in selected:
                case=lookup[s['case_id']]
                if not case['feature_valid']:raise ValueError('Quarantined state selected for training')
                atomic(out/'progress.json',dict(stage='state_counterfactual',task=row['task_id'],case=case['case_id'],role=role))
                label_state(rt,row,case,s,collections[case['source_policy']],results[case['source_policy']])
            append(out/'completed_tasks.jsonl',dict(task_id=row['task_id'],role=role))
        atomic(out/'complete.json',dict(status='complete',ledger=rt.ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=rt.ledger.state if rt else None));raise


def execute(run,phase,cfg,m):
    spent=phase_cost(run);remaining={k:cfg['global_limits'][k]-spent.get(k,0) for k in cfg['stage_limits']}
    stage=cfg['recovery_limits'] if phase=='recover_original' else cfg['expansion_limits'] if phase=='expand_train' else remaining
    limits={k:min(stage[k],remaining[k]) for k in remaining};n=len(cfg['gpus'])
    if any(v<=0 for v in limits.values()):raise RuntimeError('Original+extension budget exhausted')
    atomic(run/(phase+'_budgets.json'),[{k:v//n for k,v in limits.items()} for _ in range(n)])
    deadline=min(m['deadline_unix'],cfg['original_recovery_deadline']) if phase=='recover_original' else m['deadline_unix']
    children=[];logs=[]
    try:
        for i,gpu in enumerate(cfg['gpus']):
            log=(run/(phase+'_worker'+str(i)+'.log')).open('ab');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            children.append(subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--phase',phase,'--worker',str(i)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        atomic(run/(phase+'_owned_workers.json'),[dict(pid=c.pid,gpu=cfg['gpus'][i]) for i,c in enumerate(children)])
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Recovery worker failed; raw receipts preserved')
            if time.time()>deadline:raise TimeoutError('Original recovery or new extension deadline')
            atomic(run/'progress.json',dict(stage=phase,new_known_cost=phase_cost(run),completed_tasks=sum(p.read_text().count('\n') for p in run.glob('*/worker*/completed_tasks.jsonl')),
                new_state_labels=sum(p.read_text().count('\n') for p in run.glob('*/worker*/state_labels.jsonl')),
                quarantined_steps=sum(p.read_text().count('\n') for p in run.glob('*/worker*/alignment_quarantine.jsonl'))))
            time.sleep(15)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Worker exit failure')
    finally:
        if any(c.poll() is None for c in children):atomic(run/'STOP',dict(reason='controller_stop'))
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=40)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait(timeout=10)
        from scripts.launch_alfworld_pilot import native_children
        for i in range(len(children)):
            for pid in native_children(run/phase/('worker'+str(i))):
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for log in logs:log.close()
        atomic(run/(phase+'_supervisor.json'),dict(workers_exited=[c.poll() is not None for c in children]))


def finish(run,cfg,expanded):
    cases,labels,pairs=all_records(run,cfg,verify=True);split=read(ROOT/cfg['split']);audit=audit_records(cases,labels,split)
    for p in run.glob('*/worker*/ledger.json'):
        l=read(p);calls=rows(p.parent/'generation_calls.jsonl')
        if l['pending_model'] or l['pending_transition']:raise ValueError('Unknown new receipt; do not fabricate final cost')
        if len(calls)!=l['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=l['tokens']:raise ValueError('New journal mismatch')
    trainpairs=[p for p in pairs if p['role']=='probe_train']
    if len(trainpairs)!=100 or len({p['task_id'] for p in trainpairs})!=100:raise ValueError('Fixed100 task denominator incomplete')
    training={}
    if expanded:
        from internal_signals.linear_utility_training import train_head
        d=run/'training';d.mkdir();atomic(d/'data_audit.json',audit)
        profile=read(ROOT/cfg['probe_training_profile'])
        training['utility_help']=train_head(ROOT,d,labels,'utility_help',profile['layers'],profile)
        atomic(d/'selection.json',dict(heads=training,deployment_allowed=False,study_stage='engineering_exploratory',scientific_effectiveness_validated=False,recommended_future_help_positive_target=50,label_schema=SCHEMA,alignment_version='actual-generated-id-alignment-v1',
            note='No independent benchmark or natural-traffic calibration claim; missing heads unavailable, no automatic Gate deployment'))
    old=read(ROOT/'docs/utility_expansion_v3/prior_corpus.json');stopped=read(ROOT/'docs/recovery_v4/recovery_receipt.json');new=phase_cost(run)
    known={k:new[k]+old['prior_cost'][k]+stopped['known_completed_cost'][k]+cfg.get('auxiliary_cost',{}).get(k,0) for k in new}
    upper=dict(known);upper['tokens']+=stopped['unknown_token_upper_bound'];upper['model_calls']+=len(stopped['unreceipted_requests'])
    byrole={role:dict(Counter(l['label'] for l in labels if l['role']==role)) for role in ['probe_train','probe_validation','gate_calibration']}
    compute=[x for p in run.glob('*/worker*/compute.jsonl') for x in rows(p)];quarantine=[x for p in run.glob('*/worker*/alignment_quarantine.jsonl') for x in rows(p)]
    result=dict(status='complete',data_audit=audit,by_role=byrole,feasibility=feasibility(labels),training=training,
        old_completed_training_goals_reused=47,recovered_original_goals=13,additional_new_goals=40,total_training_goals=100,heldout_collected=expanded,
        new_known_cost=new,known_cost_all_three_sampling_runs=known,cost_upper_bound_for_budget=upper,
        unknown_prior_receipts=stopped['unreceipted_requests'],actual_prior_unknown_tokens=None,quarantined_steps=len(quarantine),
        activation_compute=dict(forwards=len(compute),seconds=sum(x['cost']['seconds'] for x in compute)),no_test_or_independent_dev_rollouts=True,
        model_actor_reviewer_gate_memory_frozen=True,alignment_version='actual-generated-id-alignment-v1',no_gate_deployment=True,
        states=[{k:l.get(k) for k in ['case_id','task_id','role','label','deltas','source_policy','step','changed','reliability']} for l in labels],
        task_pairs=[dict(task_id=p['task_id'],role=p['role'],never_success=p['never']['success'],always_success=p['always']['success']) for p in pairs])
    atomic(run/'summary.json',result);dest=ROOT/'docs/recovery_v4/results';dest.mkdir(parents=True,exist_ok=True);atomic(dest/'summary.json',result)
    lines=['# Actual-ID alignment recovery and100-goal training results','', 'Fixed100 training tasks completed; old completed data reused, partial tasks rerun at unchanged seeds.',
        '', 'Labels by role: '+json.dumps(byrole), '', 'Head statuses: '+json.dumps({k:v['status'] for k,v in training.items()}),
        '', 'Two historical model outputs remain unknown. Their reserved upper bounds are budget charges, not invented token receipts.',
        '', 'Known combined cost: '+json.dumps(known), '', 'Budget upper bound: '+json.dumps(upper),
        '', 'Quarantined feature steps: '+str(len(quarantine)), '', 'No automatic Gate deployment or Test/independent-Dev evaluation.',
        '', 'Full sources/metrics/limitations: [JSON](recovery_v4/results/summary.json).']
    (ROOT/'docs/RECOVERY_V4_REPORT.md').write_text('\n'.join(lines)+'\n')


def launch(run,autopublish):
    cfg=read(ROOT/'configs/evaluation/recovery_v4.json');receipt=read(ROOT/'docs/recovery_v4/recovery_receipt.json')
    if run.exists():raise ValueError('Fresh recovery Run ID required')
    if time.time()>cfg['original_recovery_deadline']:raise TimeoutError('Original recovery window expired; do not reset silently')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256'] or file_hash(ROOT/cfg['memory_snapshot'])!=cfg['memory_sha256']:raise ValueError('Split/Memory changed')
    if read(ROOT/'docs/recovery_v4/alignment_cpu_audit.json')['status']!='pass':raise ValueError('CPU alignment audit missing')
    if read(ROOT/'outputs/recovery_v4_gpu_smoke/complete.json')['status']!='pass':raise ValueError('Real GPU regression missing')
    if read(ROOT/'outputs/recovery_v4_preflight/complete.json')['status']!='pass':raise ValueError('New identity preflight missing')
    imported(cfg,verify=True)
    if len(cfg['gpus'])>4:raise ValueError('Max4 GPUs')
    for gpu in cfg['gpus']:
        if int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<22000:raise ValueError('GPU unavailable')
    run.mkdir(parents=True)
    paths=['configs/evaluation/recovery_v4.json',cfg['split'],cfg['actor_config'],cfg['memory_snapshot'],'docs/recovery_v4/recovery_receipt.json',
        'docs/utility_expansion_v3/prior_corpus.json','scripts/run_recovery_v4.py','evaluation/recovery_collection.py','evaluation/aligned_runtime.py',
        'internal_signals/actual_token_alignment.py','internal_signals/actual_token_nll.py','internal_signals/hidden_states.py','internal_signals/provider.py',
        'evaluation/activation_runtime.py','scripts/run_utility_expansion.py','scripts/run_utility_sampling.py','evaluation/utility_sampling.py',
        'evaluation/utility_labeling_v3.py','evaluation/utility_labels_v3.py','internal_signals/activation_protocol.py','internal_signals/activation_capture.py',
        'internal_signals/activation_training.py','internal_signals/linear_utility_training.py',cfg['probe_training_profile'],'internal_signals/activation_probe.py','internal_signals/qwen35_actor.py','internal_signals/hf_actor.py',
        'worldmodel/alfworld_environment.py','scripts/alfworld_worker.py','experience/alfworld_memory.py','reflection/replay_review.py']
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in paths},
        started_unix=time.time(),deadline_unix=time.time()+cfg['wall_seconds'],gpus=cfg['gpus'],profile=cfg,original_recovery_deadline=cfg['original_recovery_deadline'])
    atomic(run/'manifest.json',m)
    def stop(*_):raise InterruptedError('Owned recovery controller stopped/deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,cfg['wall_seconds'])
    try:
        execute(run,'recover_original',cfg,m);execute(run,'expand_train',cfg,m)
        _,labels,_=all_records(run,cfg);f=feasibility(labels);atomic(run/'feasibility.json',f)
        if f['expand']:execute(run,'heldout_pools',cfg,m)
        check(m);atomic(run/'progress.json',dict(stage='audit_optional_probe_training'));finish(run,cfg,f['expand'])
        atomic(run/'progress.json',dict(stage='complete',summary='summary.json'))
        if autopublish:
            try:
                allowed=['docs/RECOVERY_V4_REPORT.md','docs/recovery_v4/results']
                dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
                if any(not any(l[3:]==p or l[3:].startswith(p+'/') for p in allowed) for l in dirty):raise RuntimeError('Unrelated edits block publication')
                subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
                subprocess.run(['git','commit','-m','Report repaired alignment recovery and100-goal training data'],check=True)
                subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20',GIT_TERMINAL_PROMPT='0'))
                atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
            except Exception as e:atomic(run/'publication.json',dict(status='needs_review',message=str(e),results_preserved=True))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase');p.add_argument('--worker',type=int);p.add_argument('--auto-publish',action='store_true');a=p.parse_args()
    if a.worker is not None:worker(Path(a.run),a.phase,a.worker)
    else:launch(Path(a.run).absolute(),a.auto_publish)
