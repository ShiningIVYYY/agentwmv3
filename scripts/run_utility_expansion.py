"""Fixed24-goal lawful TRAIN expansion; immutable V2 corpus is reused, not rerun."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from evaluation.activation_runtime import Runtime
from evaluation.utility_labeling_v3 import label_state
from evaluation.utility_labels_v3 import relabel,SCHEMA
from scripts.run_utility_sampling import source_episode,select_pair,mine_training_candidates,phase_cost,read,rows,check
from experience.alfworld_memory import AlfExperienceMemory
from experience.splits import file_hash


def prior_records(cfg,verify_features=False):
    receipt=read(ROOT/'docs/utility_expansion_v3/prior_corpus.json')
    for p,h in receipt['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Prior raw corpus changed: '+p)
    if verify_features:
        for p,h in receipt['feature_hashes'].items():
            if file_hash(ROOT/p)!=h:raise ValueError('Prior activation changed: '+p)
    r=ROOT/cfg['prior_run']
    cases=[c for p in sorted(r.glob('train_pool/worker*/cases.jsonl')) for c in rows(p)]
    labels=[relabel(c) for p in sorted(r.glob('train_pool/worker*/state_labels.jsonl')) for c in rows(p)]
    pairs=[c for p in sorted(r.glob('train_pool/worker*/task_pairs.jsonl')) for c in rows(p)]
    if len(cases)!=receipt['cases'] or len(labels)!=receipt['state_labels']:raise ValueError('Prior denominator changed')
    return cases,labels,pairs


def feasibility(labels,minimum=5,min_tasks=3):
    helpful=[l for l in labels if l['role']=='probe_train' and l['label']=='help']
    tasks={l['task_id'] for l in helpful}
    return dict(helpful_states=len(helpful),helpful_tasks=len(tasks),expand=len(helpful)>=minimum and len(tasks)>=min_tasks,
        rule='At least5 repeated-help candidates across3 TRAIN tasks; complete all24 new goals before checking; inherited evidence retained')


def worker(run,phase,index):
    m=read(run/'manifest.json');cfg=read(ROOT/'configs/evaluation/utility_expansion_v3.json');split=read(ROOT/cfg['split']);check(m)
    out=run/phase/('worker'+str(index));out.mkdir(parents=True);rt=None
    def stop(*_):raise InterruptedError('Owned expansion worker stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        runtime_cfg=copy.deepcopy(cfg);runtime_cfg['stage_limits']=read(run/(phase+'_budgets.json'))[index]
        rt=Runtime(ROOT,out,runtime_cfg,m['deadline_unix']);memory=AlfExperienceMemory(read(ROOT/cfg['memory_snapshot']),split)
        if phase=='train_pool':
            ids=set(split['expansion_train_task_ids']);tasks=[('probe_train',r) for r in split['partitions']['probe_train'] if r['task_id'] in ids]
        else:tasks=[(role,r) for role in ['probe_validation','gate_calibration'] for r in split['partitions'][role]]
        for ordinal,(role,row) in enumerate(tasks):
            if ordinal%len(cfg['gpus'])!=index:continue
            check(m);collections={};results={}
            for policy in ['never','always']:
                atomic(out/'progress.json',dict(stage='source_episode',task=row['task_id'],role=role,policy=policy))
                collections[policy],results[policy]=source_episode(rt,row,role,policy,memory,cfg['sampling_nll_thresholds'])
            selected,first=select_pair(collections['never'],collections['always'],role,row['task_id'])
            selected=mine_training_candidates(selected,collections['never'],collections['always'],role,results['never']['success'],results['always']['success'])
            append(out/'task_pairs.jsonl',dict(task_id=row['task_id'],role=role,never=results['never'],always=results['always'],first_divergence=first,selected=selected))
            lookup={c['case_id']:c for cs in collections.values() for c in cs}
            for selection in selected:
                case=lookup[selection['case_id']];atomic(out/'progress.json',dict(stage='state_counterfactual',task=row['task_id'],case=case['case_id'],role=role))
                label_state(rt,row,case,selection,collections[case['source_policy']],results[case['source_policy']])
            append(out/'completed_tasks.jsonl',dict(task_id=row['task_id'],role=role))
        atomic(out/'complete.json',dict(status='complete',ledger=rt.ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=rt.ledger.state if rt else None));raise


def all_records(run,cfg,verify=False):
    cases,labels,pairs=prior_records(cfg,verify_features=verify)
    cases += [c for p in sorted(run.glob('*/worker*/cases.jsonl')) for c in rows(p)]
    labels += [c for p in sorted(run.glob('*/worker*/state_labels.jsonl')) for c in rows(p)]
    pairs += [c for p in sorted(run.glob('*/worker*/task_pairs.jsonl')) for c in rows(p)]
    return cases,labels,pairs


def audit_records(cases,labels,split):
    legal={r['task_id']:(role,r['goal_group']) for role in ['probe_train','probe_validation','gate_calibration'] for r in split['partitions'][role]}
    ids={};groups={}
    for c in cases:
        if c['case_id'] in ids:raise ValueError('Duplicate combined case ID')
        if legal.get(c['task_id'])!=(c['role'],c['goal_group']):raise ValueError('Role/goal leakage')
        if not c['feature_available_at_ns']<c['label_observed_at_ns']:raise ValueError('Feature future leakage')
        ids[c['case_id']]=c;groups.setdefault(c['role'],set()).add(c['goal_group'])
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:raise ValueError('Role goal overlap')
    seen=set()
    for l in labels:
        if l['case_id'] in seen:raise ValueError('Duplicate combined state label')
        seen.add(l['case_id']);c=ids[l['case_id']]
        if any(l[k]!=c[k] for k in ['role','goal_group','feature_file','state_sha256']):raise ValueError('Label source mismatch')
        if not l['feature_available_at_ns']<l['label_observed_at_ns']:raise ValueError('Label timing leakage')
        if l['label']=='neutral' and any(l.get('deltas',[])):raise ValueError('Mixed outcomes cannot be neutral')
    return dict(cases=len(cases),labels=len(labels),groups={k:len(v) for k,v in groups.items()},no_role_overlap=True,no_future_features=True,label_schema=SCHEMA)


def finish(run,cfg,expanded):
    cases,labels,pairs=all_records(run,cfg,verify=True);split=read(ROOT/cfg['split']);audit=audit_records(cases,labels,split)
    for p in run.glob('*/worker*/ledger.json'):
        l=read(p);calls=rows(p.parent/'generation_calls.jsonl')
        if l['pending_model'] or l['pending_transition']:raise ValueError('Unknown pending new receipt')
        if len(calls)!=l['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=l['tokens']:raise ValueError('New token journal mismatch')
    training={}
    if expanded:
        from internal_signals.activation_training import train_head
        dest=run/'training';dest.mkdir();atomic(dest/'data_audit.json',audit)
        for target in ['native_rejection','utility_help','utility_harm']:
            training[target]=train_head(ROOT,dest,cases if target=='native_rejection' else labels,target,[10,15,21])
        atomic(dest/'selection.json',dict(heads=training,label_schema=SCHEMA,deployment_allowed=False,
            reason='Training/Validation/Calibration quotas and stratification unchanged; probability/sampling audit required before online Gate deployment'))
    prior=read(ROOT/'docs/utility_expansion_v3/prior_corpus.json');preflight=read(ROOT/'outputs/utility_expansion_v3_preflight/complete.json')
    newcost=phase_cost(run);cumulative={k:newcost[k]+prior['prior_cost'][k]+preflight.get(k,0) for k in newcost}
    byrole={role:dict(Counter(l['label'] for l in labels if l['role']==role)) for role in ['probe_train','probe_validation','gate_calibration']}
    outcomes={role:dict(Counter(('always_win' if p['always']['success'] else 'always_loss')+'_'+('never_win' if p['never']['success'] else 'never_loss') for p in pairs if p['role']==role)) for role in byrole}
    compute=[r for p in run.glob('*/worker*/compute.jsonl') for r in rows(p)]
    models=[read(p) for p in run.glob('*/worker*/model.json')]
    result=dict(status='complete',data_audit=audit,expanded_validation_calibration=expanded,by_role=byrole,training=training,
        feasibility=feasibility(labels),paired_task_outcomes_by_role=outcomes,label_schema=SCHEMA,prior_label_corrections=prior['corrections'],
        new_main_cost=newcost,previous_sunk_cost=prior['prior_cost'],new_preflight_cost=preflight,cumulative_cost_for_v2_v3_sampling_runs=cumulative,
        new_activation_compute=dict(forwards=len(compute),seconds=sum(r['cost']['seconds'] for r in compute),
            gpu_peak_allocated_bytes=max([r['cost'].get('gpu_peak_allocated_bytes') or 0 for r in compute],default=0)),
        model_loading_seconds=sum(m['load_seconds'] for m in models),no_gate_deployment=True,no_test_or_independent_dev_rollouts=True,
        states=[{k:l.get(k) for k in ['case_id','task_id','role','goal_group','step','source_policy','label','utility_help','utility_harm','legacy_label',
            'deltas','reliability','mean_delta','selection','changed','pair_count','feature_file']} for l in labels],
        task_pairs=[dict(task_id=p['task_id'],role=p['role'],first_divergence=p['first_divergence'],never_success=p['never']['success'],
            always_success=p['always']['success'],never_steps=p['never']['formal_steps'],always_steps=p['always']['formal_steps']) for p in pairs],
        caveats=['TRAIN enrichment/stratified heldout labels are not natural prevalence or independent Dev performance',
            'Five-seed condition is not a significance test','Unchanged action does not mean correct action','No Actor/Reviewer/Memory/Gate changes'])
    atomic(run/'summary.json',result);dest=ROOT/'docs/utility_expansion_v3/results';dest.mkdir(parents=True,exist_ok=True);atomic(dest/'summary.json',result)
    lines=['# Legal utility training expansion v3 results','',
        'Original36 Train tasks reused;24 new TRAIN goals were frozen before outcomes. Val/Cal roles and final Test remain protected.',
        '', 'Combined labels by role: '+json.dumps(byrole), '', 'Combined feasibility: '+json.dumps(result['feasibility']),
        '', 'Validation/Calibration collected: '+str(expanded), '', 'Legacy mixed-neutral correction: '+str(len(prior['corrections']))+' records; original raw files unchanged.',
        '', 'Head statuses: '+json.dumps({k:v['status'] for k,v in training.items()}),
        '', 'New main cost: '+json.dumps(newcost), '', 'No automatic learned Gate deployment or final/independent evaluation.',
        '', 'All trial metrics, finite-seed labels, cost provenance and limitations: [JSON](utility_expansion_v3/results/summary.json).']
    (ROOT/'docs/UTILITY_EXPANSION_V3_REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(run):
    try:
        allowed=['docs/UTILITY_EXPANSION_V3_REPORT.md','docs/utility_expansion_v3/results']
        dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
        if any(not any(l[3:]==p or l[3:].startswith(p+'/') for p in allowed) for l in dirty):raise RuntimeError('Unrelated worktree changes; raw results saved')
        subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
        subprocess.run(['git','commit','-m','Report legal training expansion and combined repeated WM utility labels'],check=True)
        subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20'))
        atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    except Exception as e:atomic(run/'publication.json',dict(status='needs_review',message=str(e),results_preserved=True))


def launch(run,autopublish):
    cfg=read(ROOT/'configs/evaluation/utility_expansion_v3.json')
    if run.exists():raise ValueError('Fresh run required; do not duplicate or rerun old corpus')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256'] or file_hash(ROOT/cfg['memory_snapshot'])!=cfg['memory_sha256']:raise ValueError('Split/Memory changed')
    pre=read(ROOT/'outputs/utility_expansion_v3_preflight/complete.json')
    if pre['status']!='pass' or pre['split_sha256']!=cfg['split_sha256']:raise ValueError('Native preflight not passed for this split')
    prior_records(cfg,verify_features=True)
    if len(cfg['gpus'])>4:raise ValueError('Max4 authorized GPUs')
    for gpu in cfg['gpus']:
        if int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<22000:raise ValueError('GPU not free; do not displace other jobs')
    run.mkdir(parents=True)
    paths=['configs/evaluation/utility_expansion_v3.json',cfg['actor_config'],cfg['split'],cfg['memory_snapshot'],
        'docs/utility_expansion_v3/prior_corpus.json','scripts/run_utility_expansion.py','scripts/run_utility_sampling.py',
        'evaluation/utility_labels_v3.py','evaluation/utility_labeling_v3.py','evaluation/utility_sampling.py','evaluation/activation_runtime.py',
        'internal_signals/activation_protocol.py','internal_signals/activation_capture.py','internal_signals/activation_training.py','internal_signals/activation_probe.py',
        'internal_signals/qwen35_actor.py','internal_signals/hf_actor.py','internal_signals/provider.py','internal_signals/hidden_states.py','internal_signals/token_nll.py',
        'worldmodel/alfworld_environment.py','scripts/alfworld_worker.py','experience/alfworld_memory.py','reflection/replay_review.py']
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in paths},
        started_unix=time.time(),deadline_unix=time.time()+cfg['global_limits']['wall_seconds'],gpus=cfg['gpus'],profile=cfg,
        preflight=pre,prior_corpus_sha256=file_hash(ROOT/'docs/utility_expansion_v3/prior_corpus.json'))
    atomic(run/'manifest.json',m)
    def stop(*_):raise InterruptedError('Owned expansion controller stop/deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,cfg['global_limits']['wall_seconds'])
    try:
        execute_phase(run,'train_pool',cfg,m)
        _,labels,_=all_records(run,cfg);f=feasibility(labels);atomic(run/'feasibility.json',f)
        if f['expand']:execute_phase(run,'heldout_pools',cfg,m)
        check(m);atomic(run/'progress.json',dict(stage='audit_and_optional_probe_training'));finish(run,cfg,f['expand'])
        atomic(run/'progress.json',dict(stage='complete',summary='summary.json',expanded=f['expand']))
        if autopublish:publish(run)
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


def execute_phase(run,phase,cfg,m):
    spent=phase_cost(run);fraction=cfg['train_pool_budget_fraction'] if phase=='train_pool' else 1.
    limits={k:int((cfg['global_limits'][k]*fraction if phase=='train_pool' else cfg['global_limits'][k]-spent.get(k,0))) for k in cfg['stage_limits']}
    budgets=[{k:v//len(cfg['gpus']) for k,v in limits.items()} for _ in cfg['gpus']];atomic(run/(phase+'_budgets.json'),budgets)
    children=[];logs=[]
    try:
        for i,gpu in enumerate(cfg['gpus']):
            log=(run/(phase+'_worker'+str(i)+'.log')).open('ab');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            children.append(subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--phase',phase,'--worker',str(i)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        atomic(run/(phase+'_owned_workers.json'),[dict(pid=c.pid,gpu=cfg['gpus'][i]) for i,c in enumerate(children)])
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Utility worker failed; no synthetic recovery')
            if time.time()>m['deadline_unix']:raise TimeoutError('Whole-run deadline')
            atomic(run/'progress.json',dict(stage=phase,cost=phase_cost(run),completed_tasks=sum(p.read_text().count('\n') for p in run.glob('*/worker*/completed_tasks.jsonl')),
                completed_state_labels=sum(p.read_text().count('\n') for p in run.glob('*/worker*/state_labels.jsonl')),
                completed_seed_pairs=sum(p.read_text().count('\n') for p in run.glob('*/worker*/pairs.jsonl'))))
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



if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--phase');ap.add_argument('--worker',type=int);ap.add_argument('--auto-publish',action='store_true');a=ap.parse_args()
    if a.worker is not None:worker(Path(a.run),a.phase,a.worker)
    else:launch(Path(a.run).absolute(),a.auto_publish)
