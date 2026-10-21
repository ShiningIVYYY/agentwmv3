#!/usr/bin/env python3
"""Frozen, staged ScienceWorld intervention ablations on retained GPUs 2--6."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.matrix import ARGUMENTS,read,save,sha,source_files,statistics,verify
from experiments.scheduler import process_identity,matches_process
from reflection.science_evidence import facts,options
import run_sr_experiments as sr
from run_scienceworld_sr_wm import audit_baseline

PROFILE='scienceworld_ee_sr'
BASE=ROOT/'outputs/wm_reflection/ee_sr_science_context_v3'
STAGES=(('stage1_guard','science-evidence-v1',('oracle','word2world')),
        ('stage2_compare','science-evidence-v2',('oracle','word2world')),
        ('stage3_memory','science-evidence-v3',('oracle','word2world')),
        ('stage4_observation_only','science-evidence-v3-obs',('oracle',)))


def prepare(out,workers):
    path=out/'experiment.json'
    if path.exists():
        matrix=read(path)
        if matrix['output']!=str(out) or matrix['workers']!=workers:
            raise RuntimeError('Scheduling differs from frozen experiment')
        verify(matrix);return matrix
    out.mkdir(parents=True,exist_ok=True)
    audit=audit_baseline(BASE)
    frozen=read(BASE/'wm_baseline_audit.json')
    if audit['baseline_result_hashes']!=frozen['baseline_result_hashes']:
        raise RuntimeError('Completed baseline files changed')
    save(out/'baseline_audit.json',audit)
    reference=read(BASE/'wm_experiment.json')
    profile=dict(reference['profiles'][PROFILE])
    config=dict(profile['config']);config.pop('execution',None)
    profile['baseline']=str(BASE/'none'/PROFILE)
    dev=read(out/'dev_manifest.json')
    dev_ids=[r['task_id'] for r in dev['tasks']]
    if len(dev_ids)!=6 or set(dev_ids)&set(profile['task_ids']):
        raise RuntimeError('Development tasks must be six distinct non-test training tasks')
    dev_audit=read(out/'dev_manifest_audit.json')
    if not dev_audit['passed'] or dev_audit['development_manifest_sha256']!=sha(out/'dev_manifest.json'):
        raise RuntimeError('Development manifest was not verified')
    jobs={}
    for stage,protocol,backends in STAGES:
        for backend in backends:
            key=stage+'/'+backend
            c=dict(config,output=str(out/key),wm_backend=backend,
                   reflection_review_protocol=protocol,reflection_max_rounds=3)
            pre=dict(c,manifest=str(out/'dev_manifest.json'))
            jobs[key]=dict(profile=PROFILE,backend=backend,protocol=protocol,stage=stage,
                config=c,preflight_config=pre,expected=200,
                workers=workers if len(backends)==2 else min(32,2*workers),
                output=str(out/key),preflight=str(out/'preflight'/key))
    files=source_files(ROOT)+[Path(__file__),ROOT/'scripts/prepare_science_evidence_dev.py',
        ROOT/'scripts/test_science_evidence.py',ROOT/'scripts/test_science_evidence_experiments.py',
        Path(config['manifest']),Path(config['science_demo']),
        out/'dev_manifest.json',out/'dev_manifest_audit.json',out/'baseline_audit.json',
        BASE/'none'/PROFILE/'config.json',
        ROOT/'checkpoints/early-experience-scienceworld/sr/sr_provenance.json']
    hashes={str(p):sha(p) for p in files};hashes.update(audit['baseline_result_hashes'])
    services=reference['services']
    if {s['gpu'] for s in services}!={3,4,5,6} or profile['wm']['gpu']!=2:
        raise RuntimeError('Only the retained GPU allocation is authorized')
    matrix=dict(root=str(ROOT),output=str(out),protocol='science-evidence-ablations-v1',
        created=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        workers=workers,profiles={PROFILE:profile},jobs=jobs,hashes=hashes,
        stages=[x[0] for x in STAGES],development_task_ids=dev_ids,
        services=services,inference_router=reference['inference_router'],sampling=reference['sampling'],
        baseline_success_rate=audit['success_rate'],baseline_reference=str(BASE),
        old_WM_reference=str(BASE),policy=dict(workers=6,smoke_limit=6,smoke_steps=6,
            schedule='fixed-stages/both-train-only-preflights-before-paired-full-arms/no-selection-by-test-results',
            max_runner_launches=3,structured_outputs=sr.DECODE,released_gpus=[0,1,7],
            development_usage='interface-and-invariant-validation-only/no-success-rate-tuning',
            baseline='reuse-frozen-200-corrected-pure-SR-tasks',
            full_tasks_per_arm=200,total_full_task_instances=1400))
    save(path,matrix)
    (out/'services').mkdir(exist_ok=True)
    router=read(BASE/'services/science_router.json')
    if not matches_process(router):raise RuntimeError('Retained router is not running')
    save(out/'services/science_router.json',router)
    old=read(BASE/'wm_runstate.json')
    save(out/'runstate.json',dict(matrix_sha256=sha(path),phase='prepared',jobs={},
         services={str(s['port']):old['services'][str(s['port'])] for s in services}))
    return matrix


def command(matrix,key,smoke=False):
    argv=sr.command(matrix,key,smoke)
    argv[argv.index('--reflection-review-protocol')+1]=matrix['jobs'][key]['protocol']
    c=matrix['jobs'][key]['preflight_config' if smoke else 'config']
    for field in ('science_context','science_demo'):
        argv+=['--'+field.replace('_','-'),str(c[field])]
    return argv


def launch_limit(matrix,record,smoke):
    # Extra launches require a recorded infrastructure repair; never reset the
    # original failure count or silently expand retry budgets.
    return 1 if smoke else matrix['policy'].get('max_runner_launches',3)+record.get('maintenance_restarts',0)


class Scheduler(sr.Scheduler):
    def __init__(self,matrix):
        self.matrix=matrix;self.root=Path(matrix['root']);self.out=Path(matrix['output'])
        self.state=read(self.out/'runstate.json');self.children={}
        if self.state['matrix_sha256']!=sha(self.out/'experiment.json'):
            raise RuntimeError('Frozen experiment changed')

    def persist(self):
        self.state['updated']=time.time();save(self.out/'runstate.json',self.state)

    def event(self,event,**fields):
        row=dict(event=event,updated=time.time(),**fields)
        print(json.dumps(row),flush=True)
        with (self.out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    def validate(self,key,smoke):
        job=self.matrix['jobs'][key];path=Path(job['preflight'] if smoke else job['output'])
        actual=read(path/'config.json');expected=job['preflight_config' if smoke else 'config']
        for name in ARGUMENTS+('science_context','science_demo'):
            value=self.matrix['policy']['smoke_steps'] if smoke and name=='max_steps' else expected.get(name)
            if actual.get(name)!=value:raise RuntimeError('Configuration differs: '+key+'/'+name)
        if actual.get('manifest_sha256')!=sha(expected['manifest']):raise RuntimeError('Manifest changed')
        if actual.get('wm_backend')!=job['backend'] or actual.get('reflection_review_protocol')!=job['protocol'] or actual.get('reflection_max_rounds')!=3:
            raise RuntimeError('Wrong WM or reflection settings')
        if actual.get('execution',{}).get('workers')!=(6 if smoke else job['workers']):raise RuntimeError('Wrong workers')
        if actual.get('limit')!=(6 if smoke else None):raise RuntimeError('Wrong limit')
        ids=self.matrix['development_task_ids'] if smoke else self.matrix['profiles'][PROFILE]['task_ids']
        opt=options(job['protocol']);steps_count=0;reviews=[];protected=0;changes=0;comparisons=0
        for tid in ids:
            row=read(path/'tasks'/(tid+'.json'))
            if row['status']!='completed' or not row['steps']:raise RuntimeError('Incomplete task '+tid)
            if row['success']!=bool(row['done'] and row['score']==100):raise RuntimeError('Wrong success accounting')
            for i,s in enumerate(row['steps']):
                r=s['reflection'];steps_count+=1;reviews+=r['reviews']
                if r['review_protocol']!=job['protocol'] or r['rounds']>3:
                    raise RuntimeError('Wrong review policy/budget')
                if r['final_action']!=s['action'] or r['predictions'][-1]['action']!=s['action']:
                    raise RuntimeError('Final forecast/commit mismatch')
                if row['task_description'] not in s['messages'][-1]['content'] or s['prompt_tokens']>4096:
                    raise RuntimeError('Actor context changed')
                if job['backend']=='oracle' and not r['prediction_exact_match']:
                    raise RuntimeError('Oracle committed state differs from forecast')
                if r.get('transition_metadata_visible')!=opt['metadata']:
                    raise RuntimeError('Metadata ablation mismatch')
                if opt['memory'] and r['memory_evidence']!=facts(row['steps'][:i]):
                    raise RuntimeError('Memory contains uncommitted facts')
                p=next(p for p in r['predictions'] if p['action']==r['initial_action'])
                won=(job['backend']=='oracle' and opt['metadata'] and p.get('done') is True and p.get('success') is True and p.get('score')==100)
                if won:
                    protected+=1
                    if s['action']!=r['initial_action']:raise RuntimeError('Successful action vetoed')
                changed=s['action']!=r['initial_action'];changes+=changed
                compared=any('comparison' in rv for rv in r['reviews']);comparisons+=compared
                if opt['compare'] and changed and not compared:
                    final=r['predictions'][-1]
                    direct_success=(job['backend']=='oracle' and opt['metadata'] and final.get('success') is True) or (job['backend']=='word2world' and final.get('predicted_success') is True)
                    if not direct_success and 'safe_fallback_from_verified_terminal_failure' not in r['policy_events']:
                        raise RuntimeError('Alternative accepted without comparison')
                if not opt['metadata']:
                    for rv in r['reviews']:
                        packets=[rv.get('messages',[])]+[rv[x]['messages'] for x in ('replan','comparison') if x in rv]
                        for ms in packets:
                            for m in ms:
                                data=json.loads(m['content'].split('\nINPUT:\n',1)[1])
                                for name in ('original_forecast','alternative_forecast'):
                                    if any(k in data.get(name,{}) for k in ('reward','done','success','score','predicted_success')):
                                        raise RuntimeError('Privileged metadata leaked to observation-only model')
        model_reviews=[r for r in reviews if r.get('generation')]
        errors=sum(r.get('error_kind')=='invalid_format' for r in reviews)
        truncated=sum(r.get('generation',{}).get('finish_reason')=='length' for r in model_reviews)
        if smoke and model_reviews and (errors/max(1,len(reviews))>.1 or truncated/len(model_reviews)>.1):
            raise RuntimeError('Systemic review format/truncation failure')
        save(path/'validation.json',dict(passed=True,steps=steps_count,model_reviews=len(model_reviews),
             format_errors=errors,truncated_reviews=truncated,protected_successes=protected,
             changed_steps=changes,comparison_steps=comparisons,metadata_visible=opt['metadata'],
             train_only=smoke))

    def drive(self,key,smoke):
        phase=('preflight/' if smoke else 'full/')+key
        r=self.state['jobs'].setdefault(phase,dict(status='pending',launches=0))
        if r['status']=='completed':return True
        job=self.matrix['jobs'][key];path=Path(job['preflight'] if smoke else job['output'])
        ids=self.matrix['development_task_ids'] if smoke else self.matrix['profiles'][PROFILE]['task_ids']
        if r.get('pid'):
            child=self.children.get(phase);code=child.poll() if child else None
            if code is None and matches_process(r):return False
            native=read(path/'parallel_state.json') if (path/'parallel_state.json').exists() else {}
            code=code if code is not None else (0 if native.get('status')=='completed' else -1)
            r.update(pid=None,exit_code=code)
            if code==0 and statistics(path,ids)['complete']:
                self.validate(key,smoke);r['status']='completed';self.persist()
                self.event('job_completed',job=key,smoke=smoke);return True
            r['status']='failed';self.event('job_failed',job=key,smoke=smoke,exit_code=code)
        if r['launches'] >= launch_limit(self.matrix,r,smoke):raise RuntimeError('Runner exhausted retries: '+phase)
        verify(self.matrix);path.mkdir(parents=True,exist_ok=True);argv=command(self.matrix,key,smoke)
        with (path/'runner.log').open('a') as stream:
            child=subprocess.Popen(argv,cwd=self.root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1'))
        self.children[phase]=child
        r.update(pid=child.pid,**process_identity(child.pid),status='running',launches=r['launches']+1)
        self.persist();self.event('job_started',job=key,smoke=smoke,pid=child.pid);return False

    def refresh(self):
        profile=self.matrix['profiles'][PROFILE];ids=profile['task_ids'];results={};comparisons={}
        for key,job in self.matrix['jobs'].items():
            value=statistics(job['output'],ids)
            value.update(status=self.state['jobs'].get('full/'+key,{}).get('status','pending'),
                preflight=self.state['jobs'].get('preflight/'+key,{}).get('status','pending'))
            results[key]=value;comparisons[key]=sr.paired(self.matrix,key)
        baseline=statistics(profile['baseline'],ids)
        complete=all(v['complete'] and v['status']=='completed' for v in results.values())
        save(self.out/'summary.json',dict(updated=time.time(),complete=complete,phase=self.state.get('phase'),
            error=self.state.get('error'),manager_pid=os.getpid(),baseline=baseline,results=results))
        save(self.out/'comparison.json',comparisons)
        lines=['# ScienceWorld evidence-gated WM ablations','',
            'Fixed pure SR baseline: 123/200 (61.5%). Stages are fixed in advance; no stage selection by test success.',
            '', '| Group | Completed | Success rate | Mean score | Errors | Status |',
            '|---|---:|---:|---:|---:|---|']
        if self.matrix.get('maintenance_transitions'):
            lines[2]+=' Infrastructure repair cohorts are recorded in maintenance_transitions; completed task files are retained byte-for-byte.'
        for key,v in results.items():
            rate='-' if v['success_rate'] is None else '%.2f%%'%v['success_rate']
            score='-' if v['average_score'] is None else '%.2f'%v['average_score']
            lines.append('| %s | %d/200 | %s | %s | %d | %s |'%(key,v['completed'],rate,score,v['errors'],v['status']))
        lines+=['','| Group vs pure SR | Matched | Failure to success | Success to failure | Net |','|---|---:|---:|---:|---:|']
        for key,v in comparisons.items():
            lines.append('| %s | %d | %d | %d | %+d |'%(key,v['matched_tasks'],v['improved'],v['degraded'],v['net_successes']))
        (self.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        return complete

    def run(self):
        with (self.out/'.manager.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            save(self.out/'manager.json',dict(pid=os.getpid(),**process_identity(os.getpid())))
            try:
                self.state.pop('error',None);self.state['phase']='services';self.persist();self.refresh()
                self.services();self.ensure_router()
                for stage in self.matrix['stages']:
                    keys=[k for k,j in self.matrix['jobs'].items() if j['stage']==stage]
                    for smoke in (True,False):
                        self.state['phase']=('preflight/' if smoke else 'full/')+stage;self.persist()
                        while True:
                            self.ensure_router()
                            done=[self.drive(k,smoke) for k in keys]
                            self.refresh()
                            if all(done):break
                            time.sleep(5 if smoke else 10)
                    self.event('fixed_stage_completed',stage=stage)
                self.state['phase']='completed';self.persist();self.refresh();self.event('all_ablations_completed')
            except Exception as error:
                self.state.update(phase='error',error=repr(error));self.persist();self.refresh()
                self.event('manager_error',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,17),default=16)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--status',action='store_true')
    args=p.parse_args();out=args.output.resolve()
    if args.status:print((out/'RESULTS.md').read_text())
    else:
        matrix=prepare(out,args.workers)
        if not args.prepare_only:Scheduler(matrix).run()
