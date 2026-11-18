"""Bounded TRAIN-only feasibility, paired Never/Always and repeated state utility.

No Actor/Reviewer/Memory/Gate changes. No final Test or independent-dev rollouts.
"""
import argparse,copy,json,os,signal,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,ResourceLimit
from evaluation.activation_runtime import Runtime,field_masks,native_label
from evaluation.utility_sampling import select_cases,public_flags,aggregate
from experience.alfworld_memory import AlfExperienceMemory
from experience.splits import file_hash
from worldmodel.alfworld_environment import fingerprint


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
def check(m):
    for p,h in m['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Frozen experiment changed: '+p)


def delta(ledger,before):return {k:ledger.state[k]-before[k] for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']}


def source_episode(rt,row,role,policy,memory,thresholds):
    import torch
    start=time.perf_counter();before=dict(rt.ledger.state)
    env=rt.new_env(row);past=[];cases=[];abort=False
    episode_id='utility-v2/'+row['task_id']+'/'+policy+'/seed1'
    try:
        for step in range(50):
            state=copy.deepcopy(env.current);prefix=copy.deepcopy(env.history)
            parsed,text,meta=rt.draft(env,past,row,seed=1,scored=True)
            if not parsed['valid']:
                abort=True;append(rt.out/'format_errors.jsonl',dict(episode_id=episode_id,step=step,raw=text));break
            draft=parsed['action'];signals=meta['internal_signals']
            external,_=memory.evaluate(dict(task_id=row['task_id'],episode_id=episode_id,goal=env.goal,observation=state['observation'],
                draft_action=draft,commands=state['admissible_commands'],available_at_ns=time.time_ns()))
            sampling=public_flags(draft,state['admissible_commands'],state['observation'],past,external,
                signals['local_uncertainty'],signals['global_uncertainty'],thresholds)
            values,cost=rt.features(meta);feature_at=time.time_ns();case_id=episode_id+'/'+str(step)
            f=rt.out/'features'/row['task_id']/policy/(str(step)+'.pt');f.parent.mkdir(parents=True,exist_ok=True)
            masks=field_masks(rt.actor,meta)
            if any(not x for x in masks.values()):raise ValueError('Missing field alignment')
            torch.save(dict(features=values,contract=rt.contract,case_id=case_id,field_token_indices=masks),f)
            final=draft;sim=review=rm=None
            if policy=='always':
                sim=rt.simulate(env,draft);review,rm=rt.review(env,past,row,draft,sim,seed=0);final=review['final_action']
            actual=rt.step(env,final);label_at=time.time_ns()
            if sim and sim['prediction_available'] and final==draft and actual!=sim['candidate_transition']:raise ValueError('Source KEEP replay mismatch')
            # Risk label for an unexecuted draft is NOT inferred from a revised action.
            draft_actual=sim['candidate_transition'] if sim and sim['prediction_available'] else actual if final==draft else None
            c=dict(case_id=case_id,episode_id=episode_id,source_policy=policy,source_seed=1,task_id=row['task_id'],role=role,
                goal_group=row['goal_group'],step=step,goal=env.goal,observation=state['observation'],commands=state['admissible_commands'],
                draft=text,draft_action=draft,executed_action=final,metadata=meta,external=external,sampling=sampling,
                feature_file=str(f.relative_to(ROOT)),capture_cost=cost,feature_available_at_ns=feature_at,label_observed_at_ns=label_at,
                actual=actual,native_rejection=native_label(draft,state['admissible_commands'],draft_actual) if draft_actual else None,
                prefix=prefix,past=copy.deepcopy(past),state_sha256=fingerprint(state),simulation=sim,review=review)
            assert feature_at<label_at
            append(rt.out/'cases.jsonl',c);cases.append(c);past.append(dict(observation=state['observation'],action=final))
            if actual['done']:break
        result=dict(episode_id=episode_id,task_id=row['task_id'],role=role,policy=policy,goal_group=row['goal_group'],seed=1,
            success=env.current['success'],score=env.current['score'],formal_steps=env.steps,format_abort=abort,
            case_ids=[c['case_id'] for c in cases],seconds=time.perf_counter()-start,cost=delta(rt.ledger,before))
        append(rt.out/'source_episodes.jsonl',result);return cases,result
    finally:env.close()


def select_pair(never,always,role,task):
    selected=select_cases(never+always,task);existing={s['case_id'] for s in selected};first=None
    for a,b in zip(never,always):
        if a['executed_action']!=b['executed_action']:
            first=dict(step=a['step'],never_case_id=a['case_id'],always_case_id=b['case_id'],
                exact_common_state=a['state_sha256']==b['state_sha256'],same_draft=a['draft_action']==b['draft_action'])
            break
        if a['state_sha256']!=b['state_sha256'] or a['actual']!=b['actual']:
            raise ValueError('Same committed prefix produced unequal states/transitions')
    # Post-hoc paired-outcome mining is restricted to Probe Train.
    if role=='probe_train' and first and first['exact_common_state'] and first['same_draft'] and first['never_case_id'] not in existing:
        selected.append(dict(case_id=first['never_case_id'],step=first['step'],kind='first_shared_state_action_divergence',
            inclusion_probability=None,selection_uses_future_outcomes=True))
    return selected,first


def mine_training_candidates(selected,never,always,role,never_success,always_success):
    if role!='probe_train' or never_success or not always_success:return selected
    lookup={c['case_id']:c for c in never+always}
    identities={(lookup[s['case_id']]['state_sha256'],lookup[s['case_id']]['draft_action']) for s in selected}
    added=0
    for c in always:
        identity=(c['state_sha256'],c['draft_action'])
        if c['executed_action']==c['draft_action'] or identity in identities:continue
        selected.append(dict(case_id=c['case_id'],step=c['step'],kind='train_always_success_never_failure_revision_candidate',
            inclusion_probability=None,selection_uses_future_outcomes=True))
        identities.add(identity);added+=1
        if added==2:break
    # Outcome mining affects only offline Train scheduling, never online features.
    return sorted(selected,key=lambda s:not s['selection_uses_future_outcomes'])


def rollout(rt,row,case,action,seed,branch):
    start=time.perf_counter();before=dict(rt.ledger.state)
    env=rt.rebuild(row,case);past=copy.deepcopy(case['past']);trace=[];abort=False
    try:
        obs=env.current['observation'];actual=rt.step(env,action)
        trace.append(dict(action=action,actual=actual));past.append(dict(observation=obs,action=action))
        append(rt.out/'branch_receipts.jsonl',dict(case_id=case['case_id'],seed=seed,branch=branch,step=env.steps,action=action,actual=actual))
        while not env.current['done']:
            obs=env.current['observation'];parsed,text,_=rt.draft(env,past,row,seed=seed,scored=False)
            if not parsed['valid']:abort=True;break
            action=parsed['action'];actual=rt.step(env,action);past.append(dict(observation=obs,action=action));trace.append(dict(action=action,actual=actual))
            append(rt.out/'branch_receipts.jsonl',dict(case_id=case['case_id'],seed=seed,branch=branch,step=env.steps,action=action,actual=actual))
        return dict(success=env.current['success'],score=env.current['score'],native_done=env.current['done'],formal_steps=env.steps,
            traces=trace,format_abort=abort,continuation='frozen_never_wm',seed=seed,reused_observed_source=False,seconds=time.perf_counter()-start,cost=delta(rt.ledger,before))
    finally:env.close()


def label_state(rt,row,case,selected,source_cases,source_result):
    remaining=50-case['step'];maximum=dict(model_calls=10*(remaining-1)+1,tokens=(10*(remaining-1)+1)*2560,
        transition_queries=12*(case['step']+remaining+2),audit_capture_bundles=12*(case['step']+remaining+4))
    base={k:case[k] for k in ['case_id','task_id','goal_group','role','step','feature_file','feature_available_at_ns','state_sha256','source_policy']}
    try:
        for k,v in maximum.items():rt.ledger.check(k,v)
    except ResourceLimit as e:
        if rt.ledger.stop_file.exists() or time.time()>rt.ledger.deadline:raise
        result=dict(**base,selection=selected,label='unknown',utility_help=None,utility_harm=None,error='whole_state_budget_reservation_failed',
            label_observed_at_ns=time.time_ns());append(rt.out/'state_labels.jsonl',result);return result
    env=rt.rebuild(row,case)
    try:
        sim=rt.simulate(env,case['draft_action']);review,meta=rt.review(env,case['past'],row,case['draft_action'],sim,seed=0)
    finally:env.close()
    if not sim['prediction_available']:
        result=dict(**base,selection=selected,label='unknown',utility_help=None,utility_harm=None,error='preview_reconstruction_failed',label_observed_at_ns=time.time_ns())
        append(rt.out/'state_labels.jsonl',result);return result
    final=review['final_action'];changed=final!=case['draft_action'];pairs=[]
    seeds=[1,2,3] if changed else [1]
    index=0
    while index<len(seeds):
        seed=seeds[index]
        if seed==1 and case['source_policy']=='never':
            # This is the actual previously executed Skip trajectory, not a generated outcome.
            suffix=[c for c in source_cases if c['step']>=case['step']]
            if not suffix or suffix[0]['case_id']!=case['case_id']:raise ValueError('Cached Skip identity mismatch')
            skip=dict(success=source_result['success'],score=source_result['score'],formal_steps=source_result['formal_steps'],
                traces=[dict(action=c['executed_action'],actual=c['actual']) for c in suffix],seed=1,format_abort=source_result['format_abort'],
                reused_observed_source=True,source_episode_id=source_result['episode_id'])
        else:skip=rollout(rt,row,case,case['draft_action'],seed,'skip')
        call=rollout(rt,row,case,final,seed,'fixed_review_action')
        if not changed and call['traces']!=skip['traces']:
            raise ValueError('Unchanged intervention produced unequal fixed-seed continuation')
        d=int(call['success'])-int(skip['success']);pair=dict(case_id=case['case_id'],seed=seed,delta_return=d,skip=skip,call=call,
            intervention='same verified one-step trial and fixed Reviewer seed0; only downstream Actor seed changes')
        append(rt.out/'pairs.jsonl',pair);pairs.append(pair);index+=1
        if changed and index==3 and any(p['delta_return'] for p in pairs):seeds.extend([4,5])
    summary=aggregate([p['delta_return'] for p in pairs],changed,not changed)
    result=dict(**base,**summary,selection=selected,changed=changed,review=review,simulation=sim,review_metadata=meta,
        continuation_seeds=seeds,deltas=[p['delta_return'] for p in pairs],pair_count=len(pairs),label_observed_at_ns=time.time_ns(),
        input_features_exclude_outcomes=True,action_correctness='unknown',fixed_reviewer_seed=0)
    assert result['feature_available_at_ns']<result['label_observed_at_ns']
    append(rt.out/'state_labels.jsonl',result);return result


def worker(run,phase,index):
    m=read(run/'manifest.json');cfg=read(ROOT/'configs/evaluation/utility_sampling_v2.json');split=read(ROOT/cfg['split']);check(m)
    rt=None;out=run/phase/('worker'+str(index));out.mkdir(parents=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Owned utility worker stop')))
    try:
        runtime_cfg=copy.deepcopy(cfg);runtime_cfg['stage_limits']=read(run/(phase+'_budgets.json'))[index]
        rt=Runtime(ROOT,out,runtime_cfg,m['deadline_unix']);memory=AlfExperienceMemory(read(ROOT/cfg['memory_snapshot']),split)
        tasks=[('probe_train',r) for r in split['partitions']['probe_train']] if phase=='train_pool' else [(role,r) for role in ['probe_validation','gate_calibration'] for r in split['partitions'][role]]
        for ordinal,(role,row) in enumerate(tasks):
            if ordinal%len(cfg['gpus'])!=index:continue
            check(m);collections={};results={}
            for policy in ['never','always']:
                atomic(out/'progress.json',dict(stage='source_episode',task=row['task_id'],role=role,policy=policy))
                collections[policy],results[policy]=source_episode(rt,row,role,policy,memory,cfg['sampling_nll_thresholds'])
            selected,first=select_pair(collections['never'],collections['always'],role,row['task_id'])
            selected=mine_training_candidates(selected,collections['never'],collections['always'],role,
                results['never']['success'],results['always']['success'])
            pair=dict(task_id=row['task_id'],role=role,never=results['never'],always=results['always'],first_divergence=first,selected=selected)
            append(out/'task_pairs.jsonl',pair)
            lookup={c['case_id']:c for cs in collections.values() for c in cs}
            for selection in selected:
                case=lookup[selection['case_id']];atomic(out/'progress.json',dict(stage='state_counterfactual',task=row['task_id'],case=case['case_id'],role=role))
                label_state(rt,row,case,selection,collections[case['source_policy']],results[case['source_policy']])
            append(out/'completed_tasks.jsonl',dict(task_id=row['task_id'],role=role))
        atomic(out/'complete.json',dict(status='complete',ledger=rt.ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=rt.ledger.state if rt else None));raise


def phase_cost(run):
    c=Counter()
    for p in run.glob('*/worker*/ledger.json'):
        for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']:c[k]+=read(p)[k]
    return dict(c)


def pool_labels(run):return [x for p in run.glob('*/worker*/state_labels.jsonl') for x in rows(p)]


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


def finish(run,expanded):
    labels=pool_labels(run);cases=[x for p in run.glob('*/worker*/cases.jsonl') for x in rows(p)]
    ids={c['case_id']:c for c in cases};roles={}
    for c in cases:
        if not c['feature_available_at_ns']<c['label_observed_at_ns']:raise ValueError('Future feature leakage')
        roles.setdefault(c['role'],set()).add(c['goal_group'])
    for a in roles:
        for b in roles:
            if a!=b and roles[a]&roles[b]:raise ValueError('Goal split leakage')
    for l in labels:
        source=ids[l['case_id']]
        if any(l[k]!=source[k] for k in ['role','goal_group','feature_file','state_sha256']):raise ValueError('Label source mismatch')
        if not l['feature_available_at_ns']<l['label_observed_at_ns']:raise ValueError('Label time leakage')
    for p in run.glob('*/worker*/ledger.json'):
        ledger=read(p);calls=rows(p.parent/'generation_calls.jsonl')
        if ledger['pending_model'] or ledger['pending_transition']:raise ValueError('Unknown pending receipt')
        if len(calls)!=ledger['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=ledger['tokens']:raise ValueError('Token journal mismatch')
    taskpairs=[x for p in run.glob('*/worker*/task_pairs.jsonl') for x in rows(p)]
    paired=Counter(('always_success' if p['always']['success'] else 'always_failure')+'_'+('never_success' if p['never']['success'] else 'never_failure') for p in taskpairs)
    training={}
    if expanded:
        from internal_signals.activation_training import train_head
        out=run/'training';out.mkdir();layers=[10,15,21]
        for target in ['native_rejection','utility_help','utility_harm']:
            training[target]=train_head(ROOT,out,cases if target=='native_rejection' else labels,target,layers)
        atomic(out/'selection.json',dict(heads=training,deployment_allowed=False,
            reason='Outcome-enriched Train and stratified state Validation; calibration/deployment requires separate natural-traffic weighting audit. No new Gate is automatically evaluated.'))
    result=dict(status='complete',cases=len(cases),state_labels=len(labels),labels=dict(Counter(l['label'] for l in labels)),
        by_role={role:dict(Counter(l['label'] for l in labels if l['role']==role)) for role in roles},
        expanded_validation_calibration=expanded,paired_task_outcomes=dict(paired),task_pairs=taskpairs,states=labels,
        training=training,cost=phase_cost(run),no_test_or_independent_dev_rollouts=True,no_actor_reviewer_or_memory_updates=True,
        caveats=['Outcome-enriched training cases do not estimate natural positive prevalence','Five-seed majority is a diagnostic reliability rule, not statistical proof',
            'Fixed Reviewer seed0, Never-WM continuation; does not capture arbitrary repeated-intervention synergy',
            'Unchanged action/equal state is not an action-correctness label','No trained Gate deployment or final benchmark in this stage'])
    atomic(run/'summary.json',result)
    dest=ROOT/'docs/utility_sampling_v2/results';dest.mkdir(parents=True,exist_ok=True)
    compact=copy.deepcopy(result)
    compact['states']=[{k:l.get(k) for k in ['case_id','task_id','role','goal_group','step','source_policy','label','utility_help','utility_harm',
        'changed','deltas','reliability','mean_delta','selection','pair_count','action_correctness','feature_available_at_ns','label_observed_at_ns']} for l in labels]
    compact['task_pairs']=[dict(task_id=p['task_id'],role=p['role'],first_divergence=p['first_divergence'],
        never={k:p['never'][k] for k in ['success','score','formal_steps','format_abort','seconds','cost']},
        always={k:p['always'][k] for k in ['success','score','formal_steps','format_abort','seconds','cost']}) for p in taskpairs]
    atomic(dest/'summary.json',compact)
    lines=['# Utility sampling v2 results','','Paired TRAIN-pool Never/Always source trajectories, temporal/public-risk sampling and repeated same-state interventions.',
        '',f"Real decision cases: {len(cases)}; selected state labels: {len(labels)}.",
        '', 'State labels: '+json.dumps(result['labels']), '', 'Paired task outcomes: '+json.dumps(dict(paired)),
        '', 'Validation/Calibration expanded: '+str(expanded)+'. No final Test or independent Dev task was rolled out.',
        '', 'No Actor/Reviewer/Memory/Gate changes. No automatic learned Gate deployment. Full actual examples, seed differences, cost and heads: [results](utility_sampling_v2/results/summary.json).',
        '', 'Insufficient positive labels remain a feasibility limitation; no success or future feedback was invented.']
    (ROOT/'docs/UTILITY_SAMPLING_V2_REPORT.md').write_text('\n'.join(lines)+'\n')


def launch(run,publish):
    cfg=read(ROOT/'configs/evaluation/utility_sampling_v2.json');split=read(ROOT/cfg['split'])
    if run.exists():raise ValueError('Fresh run required, do not duplicate')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256'] or file_hash(ROOT/cfg['memory_snapshot'])!=cfg['memory_sha256']:raise ValueError('Split/Memory changed')
    for gpu in cfg['gpus']:
        if int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<22000:raise ValueError('GPU not free')
    if len(cfg['gpus'])>4:raise ValueError('Maximum four authorized GPUs')
    run.mkdir(parents=True)
    paths=['configs/evaluation/utility_sampling_v2.json',cfg['actor_config'],cfg['split'],cfg['memory_snapshot'],
        'scripts/run_utility_sampling.py','evaluation/utility_sampling.py','evaluation/activation_runtime.py','internal_signals/activation_protocol.py',
        'internal_signals/activation_capture.py','internal_signals/activation_training.py','internal_signals/activation_probe.py',
        'internal_signals/qwen35_actor.py','internal_signals/hf_actor.py','internal_signals/provider.py','internal_signals/hidden_states.py','internal_signals/token_nll.py',
        'worldmodel/alfworld_environment.py','scripts/alfworld_worker.py','experience/alfworld_memory.py','reflection/replay_review.py']
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in paths},
        started_unix=time.time(),deadline_unix=time.time()+cfg['global_limits']['wall_seconds'],gpus=cfg['gpus'],profile=cfg)
    atomic(run/'manifest.json',m)
    def stop(*_):raise InterruptedError('Owned controller stopped/deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,cfg['global_limits']['wall_seconds'])
    try:
        execute_phase(run,'train_pool',cfg,m)
        trainlabels=pool_labels(run);helpful=[l for l in trainlabels if l['label']=='help']
        expanded=len(helpful)>=5 and len({l['task_id'] for l in helpful})>=3
        atomic(run/'feasibility.json',dict(helpful_states=len(helpful),helpful_tasks=len({l['task_id'] for l in helpful}),expand=expanded,
            rule='At least5 five-seed repeated-help candidates across3 TRAIN tasks; fixed before collecting outcomes'))
        if expanded:execute_phase(run,'heldout_pools',cfg,m)
        check(m);atomic(run/'progress.json',dict(stage='audit_and_optional_probe_training'));finish(run,expanded)
        atomic(run/'progress.json',dict(stage='complete',summary='summary.json',expanded=expanded))
        if publish:
            allowed=['docs/UTILITY_SAMPLING_V2_REPORT.md','docs/utility_sampling_v2/results']
            dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
            if any(not any(l[3:]==p or l[3:].startswith(p+'/') for p in allowed) for l in dirty):raise RuntimeError('Results saved, unrelated changes block publication')
            subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
            subprocess.run(['git','commit','-m','Report paired task and repeated-state WM utility sampling'],check=True)
            subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20',GIT_TERMINAL_PROMPT='0'))
            atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--phase');ap.add_argument('--worker',type=int);ap.add_argument('--auto-publish',action='store_true');a=ap.parse_args()
    if a.worker is not None:worker(Path(a.run),a.phase,a.worker)
    else:launch(Path(a.run).absolute(),a.auto_publish)
