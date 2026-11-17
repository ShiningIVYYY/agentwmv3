"""Bounded AP0-AP5 research pilot, real labels only; no backbone parameter updates."""
import argparse,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from experience.splits import file_hash
from evaluation.activation_runtime import Runtime,collect_row,utility_pair


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def check(root,manifest):
    for path,expected in manifest['source_hashes'].items():
        if file_hash(root/path)!=expected:raise ValueError('Experiment source changed: '+path)


def worker(run,phase,index):
    import torch,numpy as np
    cfg=read(ROOT/'configs/evaluation/activation_v1.json');split=read(ROOT/cfg['split']);manifest=read(run/'manifest.json');check(ROOT,manifest)
    out=run/phase/('worker'+str(index));rt=None
    out.mkdir(parents=True,exist_ok=False)
    def stop(*_):raise InterruptedError('Owned activation worker stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        rt=Runtime(ROOT,out,cfg,manifest['deadline_unix'])
        if phase=='collect':
            selected=[(role,row) for role in ['probe_train','probe_validation','gate_calibration'] for row in split['partitions'][role]]
            for ordinal,(role,row) in enumerate(selected):
                if ordinal%len(cfg['gpus'])!=index:continue
                check(ROOT,manifest);atomic(out/'progress.json',dict(stage='collect',task=row['task_id'],role=role))
                cases=collect_row(rt,row,role,cfg['collection_steps'])
                chosen=next((c for c in cases if c['step']==cfg['utility_step']),None)
                if chosen:
                    atomic(out/'progress.json',dict(stage='utility_pair',task=row['task_id'],role=role))
                    utility_pair(rt,row,chosen)
                else:append(out/'missing_utility_labels.jsonl',dict(task_id=row['task_id'],role=role,reason='no_valid_predeclared_step2; not a zero utility'))
        elif phase=='evaluation':
            from experience.alfworld_memory import AlfExperienceMemory
            from internal_signals.activation_provider import ActivationProvider
            from evaluation.activation_evaluation import run_eval
            selection=read(run/'training/selection.json');heads={}
            for target,result in selection['heads'].items():
                if result['status']=='trained':heads[target]=dict(result=result,provider=ActivationProvider(ROOT,result))
            memory=AlfExperienceMemory(read(ROOT/cfg['memory_snapshot']),split)
            policies=['never','always','nll','nll_external','admissible','random','periodic','review_only']
            if 'native_rejection' in heads:policies+=['risk','risk_external']
            if all(t in heads for t in ['utility_help','utility_harm']):policies+=['utility','utility_external']
            atomic(out/'policies.json',policies)
            for ordinal,row in enumerate(split['partitions']['independent_dev']):
                if ordinal%len(cfg['gpus'])!=index:continue
                for policy in policies:
                    check(ROOT,manifest);atomic(out/'progress.json',dict(stage='evaluation',task=row['task_id'],policy=policy))
                    # A model-format abort returns an explicit result; other tasks continue.
                    run_eval(rt,row,policy,heads,selection['nll_calibration'],memory)
        else:raise ValueError('Unknown worker stage')
        atomic(out/'complete.json',dict(status='complete',phase=phase,ledger=rt.ledger.state))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=rt.ledger.state if rt else None));raise


def train(run):
    import numpy as np
    from internal_signals.activation_training import train_head,binary_metrics
    cases=[c for p in run.glob('collect/worker*/cases.jsonl') for c in rows(p)]
    labels=[c for p in run.glob('collect/worker*/utility_labels.jsonl') for c in rows(p)]
    allids=set();groups={}
    for c in cases:
        if c['case_id'] in allids:raise ValueError('Duplicate case')
        allids.add(c['case_id']);groups.setdefault(c['role'],set()).add(c['goal_group'])
        if not c['feature_available_at_ns']<c['label_observed_at_ns']:raise ValueError('Future label leakage')
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:raise ValueError('Group split leakage')
    for label in labels:
        source=next((c for c in cases if c['case_id']==label['case_id']),None)
        if source is None or any(label[k]!=source[k] for k in ['feature_file','role','goal_group','state_sha256']):raise ValueError('Utility/feature source mismatch')
        if not label['feature_available_at_ns']<label['label_observed_at_ns']:raise ValueError('Utility label timing leakage')
        label['utility_help']=int(label['delta_return']==1) if label['valid'] else None
        label['utility_harm']=int(label['delta_return']==-1) if label['valid'] else None
    out=run/'training';out.mkdir();atomic(out/'data_audit.json',dict(cases=len(cases),utility_labels=len(labels),label_counts=dict(Counter(l['label'] for l in labels)),
        roles={r:len(v) for r,v in groups.items()},no_task_overlap=True,no_future_features=True))
    # Direct public-rule and old-signal controls, evaluated only on Probe Validation.
    validation=[c for c in cases if c['role']=='probe_validation' and c['native_rejection'] is not None]
    baselines={}
    if validation:
        y=[c['native_rejection'] for c in validation]
        baselines['admissible_rule']=binary_metrics(y,[float(c['draft_action'] not in c['commands']) for c in validation])
        for field in ['local_uncertainty','global_uncertainty']:
            valid=[c for c in validation if c['metadata']['internal_signals']['status']=='ok']
            if valid:baselines[field]=binary_metrics([c['native_rejection'] for c in valid],[c['metadata']['internal_signals'][field] for c in valid])
    atomic(out/'offline_baselines.json',dict(scope='Probe Validation only; no independent-dev tuning',metrics=baselines))
    layers=next(iter(read(p)['contract']['capture']['layers'] for p in run.glob('collect/worker*/model.json')))
    heads={}
    heads['native_rejection']=train_head(ROOT,out,cases,'native_rejection',layers)
    for target in ['utility_help','utility_harm']:heads[target]=train_head(ROOT,out,labels,target,layers)
    cal=[c for c in cases if c['role']=='gate_calibration' and c['metadata']['internal_signals']['status']=='ok']
    if len(cal)<12:raise ValueError('Insufficient new-protocol NLL calibration; do not reuse old thresholds')
    nll=dict(tau_local=float(np.quantile([c['metadata']['internal_signals']['local_uncertainty'] for c in cal],.75)),
        tau_global=float(np.quantile([c['metadata']['internal_signals']['global_uncertainty'] for c in cal],.75)),n=len(cal),used_task_outcomes=False)
    u=np.array([[c['metadata']['internal_signals'][k] for k in ['local_uncertainty','global_uncertainty']] for c in cal])
    def ranks(v):return np.array([sum(v<x)+(sum(v==x)-1)/2 for x in v],dtype=float)
    nll['pearson']=float(np.corrcoef(u.T)[0,1]) if np.std(u[:,0]) and np.std(u[:,1]) else None
    nll['spearman']=float(np.corrcoef(ranks(u[:,0]),ranks(u[:,1]))[0,1]) if np.std(u[:,0]) and np.std(u[:,1]) else None
    nll['quantile_call_rates']={str(q):float(np.mean((u[:,0]>np.quantile(u[:,0],q))|(u[:,1]>np.quantile(u[:,1],q)))) for q in [.5,.75,.9]}
    atomic(out/'selection.json',dict(heads=heads,nll_calibration=nll,utility_deployed=all(heads[t]['status']=='trained' for t in ['utility_help','utility_harm']),
        note='Sparse/unknown labels do not create a fake utility head; missing heads are not scored as zero or deployed'))


def summarize(run):
    episodes=[r for p in run.glob('evaluation/worker*/episodes.jsonl') for r in rows(p)]
    by={}
    for policy in sorted({e['policy'] for e in episodes}):
        es=[e for e in episodes if e['policy']==policy]
        if len(es)!=12 or len({e['task_id'] for e in es})!=12:raise ValueError('Incomplete evaluation denominator')
        by[policy]=dict(n=12,successes=sum(e['success'] for e in es),success_rate=sum(e['success'] for e in es)/12,
            agent_format_aborts=sum(e['termination']=='agent_format_abort' for e in es),tokens=sum(e['cost']['tokens'] for e in es),
            wm_calls=sum(e['wm_calls'] for e in es),reviewer_calls=sum(e['reviewer_calls'] for e in es),
            formal_steps=sum(e['formal_steps'] for e in es),activation_forwards=sum(e['activation_forwards'] for e in es),
            activation_forward_seconds=sum(e['activation_forward_seconds'] for e in es),probe_seconds=sum(e['probe_seconds'] for e in es),
            environment_queries=sum(e['cost']['transition_queries'] for e in es),seconds=sum(e['seconds'] for e in es))
    cost=Counter()
    for p in run.glob('*/worker*/ledger.json'):
        data=read(p)
        if data['pending_model'] or data['pending_transition']:raise ValueError('Unknown pending receipt')
        calls=rows(p.parent/'generation_calls.jsonl')
        if len(calls)!=data['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=data['tokens']:raise ValueError('Model cost journal mismatch')
        for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']:cost[k]+=data[k]
    selection=read(run/'training/selection.json')
    compute=[r for p in run.glob('*/worker*/compute.jsonl') for r in rows(p)]
    models=[read(p) for p in run.glob('*/worker*/model.json')]
    report=dict(status='complete',policies=by,episodes=episodes,training=selection,data_audit=read(run/'training/data_audit.json'),
        offline_baselines=read(run/'training/offline_baselines.json'),all_recorded_cost=dict(cost),
        activation_compute=dict(forwards=len(compute),seconds=sum(r['cost']['seconds'] for r in compute)),
        model_loading_seconds=sum(r['load_seconds'] for r in models),
        limitations=['Small new-protocol independent12-task single-seed pilot, not full benchmark','Native rejection probe may only approximate public admissible rule',
            'Utility labels single continuation seed; sparse heads explicitly unavailable','Teacher-forced features add a full forward, included separately',
            'Prompt/reviewer/error protocolv2 differs from historical47; do not combine outcomes','Oracle-like native environment access, not learned WM'])
    atomic(run/'summary.json',report)
    dest=ROOT/'docs/activation_v1/results';dest.mkdir(parents=True,exist_ok=True)
    atomic(dest/'summary.json',report)
    lines=['# Activation Probe v1 pilot results','','This new protocol trains only small probes; Actor/Reviewer remain frozen. Historical47-task outcomes are not merged.','',
        '| Policy | Success | SR | Tokens | WM calls | Format aborts |','|---|---:|---:|---:|---:|---:|']
    for name,e in by.items():lines.append(f"| {name} | {e['successes']}/12 | {100*e['success_rate']:.1f}% | {e['tokens']} | {e['wm_calls']} | {e['agent_format_aborts']} |")
    lines+=['','Utility label counts: '+json.dumps(report['data_audit']['label_counts']),
        '','Utility deployment: '+str(selection['utility_deployed'])+'. Insufficient heads are omitted, not fabricated.',
        '', 'Full train/validation/calibration metrics, all3 seeds, selected layer/architecture, costs and limitations: [JSON](activation_v1/results/summary.json).',
        '', 'No significance or causal generalization claim. Native rejection direct-rule baseline is essential; Risk classification alone is not WM Utility. No original benchmark results were overwritten.']
    (ROOT/'docs/ACTIVATION_V1_REPORT.md').write_text('\n'.join(lines)+'\n')


def launch(run,auto_publish=False):
    cfg=read(ROOT/'configs/evaluation/activation_v1.json')
    if run.exists():raise ValueError('Fresh run required; no accidental duplicate or outcome retry')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256'] or file_hash(ROOT/cfg['memory_snapshot'])!=cfg['memory_sha256']:raise ValueError('Preregistered split or frozen Memory changed')
    preflight=read(ROOT/'outputs/activation_preflight_v1/complete.json')
    if preflight['status']!='pass' or preflight['split_sha256']!=cfg['split_sha256']:raise ValueError('Native preflight missing or for another split')
    if read(ROOT/'outputs/activation_smoke_v1/complete.json')['status']!='pass':raise ValueError('Real activation smoke failed')
    for gpu in cfg['gpus']:
        free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free<22000:raise ValueError('GPU unavailable; do not displace other jobs')
    run.mkdir(parents=True)
    paths=['configs/evaluation/activation_v1.json',cfg['actor_config'],cfg['split'],cfg['memory_snapshot'],
        'internal_signals/activation_protocol.py','internal_signals/activation_capture.py','internal_signals/activation_probe.py',
        'internal_signals/activation_training.py','internal_signals/activation_provider.py','evaluation/activation_runtime.py','evaluation/activation_evaluation.py',
        'internal_signals/qwen35_actor.py','internal_signals/hf_actor.py','internal_signals/provider.py','worldmodel/alfworld_environment.py',
        'internal_signals/hidden_states.py','internal_signals/token_nll.py','internal_signals/alfworld_prompts.py','experience/alfworld_memory.py',
        'experience/memory.py','reflection/replay_review.py','scripts/run_alfworld_parallel.py','evaluation/independent_dev.py',
        'scripts/alfworld_worker.py','scripts/run_activation_experiment.py']
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:file_hash(ROOT/p) for p in paths},
        started_unix=time.time(),deadline_unix=time.time()+cfg['global_limits']['wall_seconds'],gpus=cfg['gpus'],no_foundation_training=True,
        native_preflight=preflight,preflight_sha256=file_hash(ROOT/'outputs/activation_preflight_v1/complete.json'),
        engineering_smoke_sha256=file_hash(ROOT/'outputs/activation_smoke_v1/complete.json'))
    atomic(run/'manifest.json',manifest)
    def stop(*_):raise InterruptedError('Owned activation supervisor stop')
    signal.signal(signal.SIGTERM,stop)
    def deadline(*_):raise TimeoutError('36h whole-run cap, including CPU probe training')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(1,manifest['deadline_unix']-time.time()))
    try:
        for phase in ['collect','evaluation']:
            if phase=='evaluation':
                atomic(run/'progress.json',dict(stage='training_probes_on_cpu',no_actor_training=True));train(run)
            children=[];logs=[]
            try:
                for i,gpu in enumerate(cfg['gpus']):
                    log=(run/(phase+'_worker'+str(i)+'.log')).open('ab');logs.append(log)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                    children.append(subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--phase',phase,'--worker',str(i)],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                atomic(run/(phase+'_owned_workers.json'),[dict(pid=c.pid,gpu=cfg['gpus'][i]) for i,c in enumerate(children)])
                while any(c.poll() is None for c in children):
                    if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Activation worker failed; preserve evidence, no fake results')
                    if time.time()>manifest['deadline_unix']:raise TimeoutError('36h activation run cap')
                    totals=Counter()
                    for p in run.glob('*/worker*/ledger.json'):
                        for k,v in read(p).items():
                            if k in cfg['global_limits']:totals[k]+=v
                    if any(totals[k]>cfg['global_limits'][k] for k in totals):raise RuntimeError('Global resource cap')
                    atomic(run/'progress.json',dict(stage=phase,cost=dict(totals),cases=sum(p.read_text().count('\n') for p in run.glob('collect/worker*/cases.jsonl')),
                        utility_labels=sum(p.read_text().count('\n') for p in run.glob('collect/worker*/utility_labels.jsonl')),
                        episodes=sum(p.read_text().count('\n') for p in run.glob('evaluation/worker*/episodes.jsonl'))))
                    time.sleep(15)
                if any(c.returncode!=0 for c in children):raise RuntimeError('Worker exit failure')
            finally:
                if any(c.poll() is None for c in children):atomic(run/'STOP',dict(reason='supervisor_stop'))
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
        check(ROOT,manifest);summarize(run)
        atomic(run/'progress.json',dict(stage='complete',summary='summary.json'))
        if auto_publish:
            allowed=['docs/ACTIVATION_V1_REPORT.md','docs/activation_v1/results']
            dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
            if any(not any(line[3:]==p or line[3:].startswith(p+'/') for p in allowed) for line in dirty):raise RuntimeError('Results saved; unrelated worktree changes block auto publication')
            subprocess.run(['git','add','--',*allowed],check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
            subprocess.run(['git','commit','-m','Report supervised activation-probe pilot and real utility-label results'],check=True)
            subprocess.run(['git','push','origin','HEAD:main'],check=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20'))
            atomic(run/'publication.json',dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    except BaseException as e:
        atomic(run/'error.json',dict(type=type(e).__name__,message=str(e)));atomic(run/'progress.json',dict(stage='error',message=str(e)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--phase');ap.add_argument('--worker',type=int);ap.add_argument('--auto-publish',action='store_true');a=ap.parse_args()
    if a.worker is not None:worker(Path(a.run),a.phase,a.worker)
    else:launch(Path(a.run).absolute(),a.auto_publish)
