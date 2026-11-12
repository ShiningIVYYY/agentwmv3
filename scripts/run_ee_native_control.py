#!/usr/bin/env python3
"""Same native-H2 prompt control on the preserved9B Actor; no Gate, training or WM."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,Ledger,request_seed
from experience.splits import file_hash,goal_key
from internal_signals.qwen35_actor import Qwen35Actor
from internal_signals.ee_sr_prompt import prompt,parse_action
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,state_fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sr-run',default='outputs/ee_sr_actor_trial_v1');a=ap.parse_args();out=Path(a.run)
    if out.exists():raise ValueError('New control directory required')
    out.mkdir(parents=True);profile=json.loads((ROOT/'configs/evaluation/ee_sr_actor_trial_v1.json').read_text())
    ledger=Ledger(out/'ledger.json',out/'STOP',time.time()+3600,dict(tokens=150000,model_calls=60,transition_queries=63,audit_capture_bundles=200))
    names=['scripts/run_ee_native_control.py','internal_signals/ee_sr_prompt.py','configs/internal_signals/qwen35_9b_replay_v1.json','docs/ee_sr_trial/source_prompt.json']
    manifest=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_at_ns=time.time_ns(),source_sr_manifest_sha256=file_hash(Path(a.sr_run)/'manifest.json'),hashes={n:file_hash(ROOT/n) for n in names},matched_tasks=profile['tasks'],purpose='post-SR prompt-confounding control; no policy tuning')
    atomic(out/'manifest.json',manifest);started=time.perf_counter()
    try:
        before=time.perf_counter();actor=Qwen35Actor(ROOT/'configs/internal_signals/qwen35_9b_replay_v1.json',ROOT/'checkpoints/qwen35_9b')
        atomic(out/'model.json',dict(load_seconds=time.perf_counter()-before,loading_info=actor.loading_info,config_sha256=actor.fingerprint,model_revision=actor.config['model_revision']))
        source=json.loads((ROOT/'docs/ee_sr_trial/source_prompt.json').read_text())
        for row in profile['tasks']:
            ep=out/'episodes'/row['task_id'];ep.mkdir(parents=True);env=None;history=[];records=[];metas=[];begin=time.perf_counter();status='horizon20'
            atomic(out/'progress.json',dict(task=row['task_id']))
            try:
                spec=dict(taskName=row['taskName'],variationIdx=row['variationIdx'],simplifications='',seed=0,environment_step_limit=21,simulator_profile='scienceworld130')
                ledger.begin_transition(1,dict(kind='init',task=row['task_id']));env=ReplayScienceEnvironment(spec);ledger.finish_transition(1);ledger.captures(env.capture_count)
                sha=state_fingerprint(env.initial);expected=json.loads((Path(a.sr_run)/'episodes'/row['task_id']/'native_h2/initial.json').read_text())['state_sha256'];assert sha==expected
                atomic(ep/'initial.json',dict(state_sha256=sha,spec=spec,pid=env.pid,goal=env.goal))
                assert goal_key(env.goal)==row['goal_group']
                for step in range(20):
                    ps=prompt('native_h2',env.goal,env.current['observation'],history,source);seed=request_seed(row['task_id'],step,'draft',0)
                    ledger.begin_model(actor.count(ps)+512,dict(task=row['task_id'],step=step));text,m=actor.generate_role(ps,'draft',seed,False);ledger.finish_model(m);metas.append(m)
                    append(out/'generation_calls.jsonl',dict(task_id=row['task_id'],step=step,metadata=m))
                    parsed=parse_action(text)
                    if m['finish_reason']=='length':parsed=dict(action=None,error='truncated')
                    if parsed['action'] is None:status='parse_failure';append(ep/'parse_failures.jsonl',dict(raw=text,parsed=parsed,metadata=m));break
                    count=env.capture_count;obs=env.current['observation'];ledger.begin_transition(1,dict(task=row['task_id'],step=step));actual=env.step(parsed['action']);ledger.finish_transition(1);ledger.captures(env.capture_count-count)
                    t=dict(task_id=row['task_id'],step=step,observation=obs,raw=text,action=parsed['action'],actual=actual,metadata=m);records.append(t);append(ep/'steps.jsonl',t);history.append(dict(observation=obs,action=parsed['action']))
                    if actual['done']:status='native_done';break
                result=dict(task_id=row['task_id'],mode='native_h2',model='qwen35_9b',status=status,task_score=env.current['score'],success=env.current['done'] and env.current['score']==100,
                    episode_complete=status!='parse_failure',formal_steps=len(records),actor_calls=len(metas),total_tokens=sum(m['usage']['total_tokens'] for m in metas),transition_queries=1+len(records))
            finally:
                if env is not None:env.close()
            result['seconds_including_init_cleanup']=time.perf_counter()-begin;atomic(ep/'complete.json',result);append(out/'episodes.jsonl',result)
        for n,h in manifest['hashes'].items():assert file_hash(ROOT/n)==h
        atomic(out/'complete.json',dict(status='complete',seconds=time.perf_counter()-started,ledger=ledger.state));atomic(out/'progress.json',dict(stage='complete'))
    except BaseException as e:atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=ledger.state));raise


if __name__=='__main__':main()
