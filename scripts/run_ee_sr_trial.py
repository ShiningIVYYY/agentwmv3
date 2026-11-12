#!/usr/bin/env python3
"""Bounded frozen SR Actor-only trial; known-goal retrospective, no Gate/Reviewer/WM."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,read_rows,Ledger,request_seed
from experience.splits import file_hash,goal_key
from internal_signals.ee_sr_actor import EESRActor
from internal_signals.ee_sr_prompt import prompt,parse_action
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,state_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model-dir',default='checkpoints/ee_scienceworld_sr/sr');a=p.parse_args();out=Path(a.run)
    if out.exists():raise ValueError('New output required; no outcome-based retry')
    out.mkdir(parents=True)
    profile=json.loads((ROOT/'configs/evaluation/ee_sr_actor_trial_v1.json').read_text())
    for n,h in profile['protected_hashes'].items():assert file_hash(ROOT/n)==h,n
    code=['internal_signals/ee_sr_actor.py','internal_signals/ee_sr_prompt.py','internal_signals/hf_actor.py','internal_signals/provider.py','internal_signals/token_nll.py','internal_signals/hidden_states.py','scripts/run_ee_sr_trial.py','configs/internal_signals/ee_scienceworld_sr_v1.json','configs/evaluation/ee_sr_actor_trial_v1.json','docs/ee_sr_trial/source_prompt.json']
    manifest=dict(profile=profile,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes={n:file_hash(ROOT/n) for n in code},started_at_ns=time.time_ns(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),task_training_corpus_overlap_disclosed=True)
    atomic(out/'manifest.json',manifest);ledger=Ledger(out/'ledger.json',out/'STOP',time.time()+7200,profile['limits'],profile['limits'])
    started=time.perf_counter();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    try:
        before=time.perf_counter();actor=EESRActor(ROOT/profile['actor_config'],a.model_dir)
        atomic(out/'model.json',dict(load_seconds=time.perf_counter()-before,loading_info=actor.loading_info,config_sha256=actor.fingerprint,config=actor.config,
            model_id=actor.config['model_id'],revision=actor.config['model_revision'],subfolder='sr',gpu=__import__('torch').cuda.get_device_name(0),dtype='bfloat16',
            official_generation_defaults=actor.original_generation_config,actual_generation_config=actor.model.generation_config.to_dict(),inference_use_cache_override=True,additional_parameter_updates=False))
        source_prompt=json.loads((ROOT/'docs/ee_sr_trial/source_prompt.json').read_text());parity_done=False
        def generate(ps,seed,scored,identity):
            ledger.begin_model(actor.count(ps)+actor.config['max_new_tokens'],identity)
            text,meta=actor.generate_role(ps,'draft',seed,scored)
            append(out/'generation_calls.jsonl',dict(identity=identity,metadata=meta));ledger.finish_model(meta);return text,meta
        for row in profile['tasks']:
            for mode in ('structured','native_h2'):
                ledger.begin_batch();ep=out/'episodes'/row['task_id']/mode;ep.mkdir(parents=True);env=None;history=[];traces=[];decision_metas=[];begin=time.perf_counter();status='running'
                spec=dict(taskName=row['taskName'],variationIdx=row['variationIdx'],simplifications='',seed=0,environment_step_limit=21,simulator_profile='scienceworld130')
                atomic(out/'progress.json',dict(task=row['task_id'],mode=mode,stage='episode'))
                try:
                    ledger.begin_transition(1,dict(task=row['task_id'],mode=mode,stage='init'));env=ReplayScienceEnvironment(spec);ledger.finish_transition(1);ledger.captures(env.capture_count)
                    assert goal_key(env.goal)==row['goal_group'];atomic(ep/'initial.json',dict(spec=spec,goal=env.goal,state_sha256=state_fingerprint(env.initial),pid=env.pid))
                    for step in range(20):
                        obs=env.current['observation'];ps=prompt(mode,env.goal,obs,history,source_prompt);seed=request_seed(row['task_id'],step,'draft',0)
                        raw,meta=generate(ps,seed,mode=='structured',dict(task=row['task_id'],mode=mode,step=step,role='actor'));decision_metas.append(meta)
                        if not parity_done:
                            before_sha=state_fingerprint(env.capture());other,unscored=generate(ps,seed,False,dict(task=row['task_id'],role='scored_unscored_parity'))
                            repeat,again=generate(ps,seed,True,dict(task=row['task_id'],role='repeated_scored_nll'))
                            assert meta['generation_record']['response_ids']==unscored['generation_record']['response_ids']==again['generation_record']['response_ids']
                            values=meta['generation_record']['token_nll'];more=again['generation_record']['token_nll'];assert len(values)==len(more)>0
                            delta=max(abs(x-y) for x,y in zip(values,more));assert delta==0
                            assert before_sha==state_fingerprint(env.capture())
                            atomic(out/'numerical_validation.json',dict(response_token_parity=True,repeated_raw_nll_max_delta=delta,response_tokens=len(values),signals=meta['internal_signals'],
                                actor_prompt_ids_sha256=hashlib.sha256(json.dumps(meta['generation_record']['prompt_ids']).encode()).hexdigest(),new_actor_thresholds_calibrated=False,no_extra_forward_pass=True,additional_generation_calls=2))
                            ledger.captures(2);parity_done=True
                        parsed=parse_action(raw)
                        if meta['finish_reason']=='length':parsed=dict(action=None,error='generation_truncated_no_committed_action')
                        if parsed['action'] is None:
                            append(ep/'parse_failures.jsonl',dict(step=step,raw=raw,parsed=parsed,metadata=meta));status='parse_failure';break
                        count=env.capture_count;ledger.begin_transition(1,dict(task=row['task_id'],mode=mode,step=step));actual=env.step(parsed['action']);ledger.finish_transition(1);ledger.captures(env.capture_count-count)
                        trace=dict(task_id=row['task_id'],mode=mode,step=step,goal=env.goal,observation=obs,raw=raw,action=parsed['action'],metadata=meta,actual=actual,actor_seed=seed)
                        append(ep/'steps.jsonl',trace);traces.append(trace);history.append(dict(observation=obs,action=parsed['action']))
                        if actual['done']:status='native_done';break
                    if status=='running':status='horizon20'
                    result=dict(task_id=row['task_id'],mode=mode,status=status,task_score=env.current['score'],success=env.current['done'] and env.current['score']==100,
                        done=env.current['done'],formal_steps=len(traces),native_moves=env.current['moves'],actor_calls=len(decision_metas),
                        missing_internal_steps=sum(t['metadata'].get('internal_signals',{}).get('missing_span',False) for t in traces) if mode=='structured' else None,
                        structured_format_valid_steps=sum(t['metadata']['decision_format_valid'] for t in traces) if mode=='structured' else None,
                        input_tokens=sum(m['usage']['prompt_tokens'] for m in decision_metas),output_tokens=sum(m['usage']['completion_tokens'] for m in decision_metas),total_tokens=sum(m['usage']['total_tokens'] for m in decision_metas),
                        wm_calls=0,reviewer_calls=0,memory_calls=0,gate_calls=0,transition_queries=1+len(traces),episode_complete=status!='parse_failure')
                finally:
                    if env is not None:env.close()
                result['seconds_including_init_cleanup']=time.perf_counter()-begin;atomic(ep/'complete.json',result);append(out/'episodes.jsonl',result)
        for n,h in manifest['code_hashes'].items():assert file_hash(ROOT/n)==h,n
        for n,h in profile['protected_hashes'].items():assert file_hash(ROOT/n)==h,n
        atomic(out/'complete.json',dict(status='complete',episodes=len(read_rows(out/'episodes.jsonl')),seconds=time.perf_counter()-started,ledger=ledger.state,finished_at_ns=time.time_ns()))
        atomic(out/'progress.json',dict(stage='complete'))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=ledger.state));raise


if __name__=='__main__':main()
