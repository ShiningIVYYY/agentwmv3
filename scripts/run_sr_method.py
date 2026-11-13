"""Shared runtime, paired isolated cohort, new model-specific calibration only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,Ledger,run_episode,request_seed
from experience.splits import file_hash,goal_key
from experience.memory import ExperienceMemory
from experience.collection import fields
from internal_signals.decision import messages
from internal_signals.qwen35_actor import Qwen35Actor
from internal_signals.sr_structured_actor import StructuredSRActor,SRReviewerProtocol
from internal_signals.token_nll import selected_token_nll
from reflection.replay_review_v2 import ReviewerProtocol
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,state_fingerprint
from scripts.run_phase57_replay import calibrate
import torch


def verify(profile):
    for n,h in profile['protected_hashes'].items():assert file_hash(ROOT/n)==h,n
    assert file_hash(ROOT/profile['isolation_manifest'])==profile['isolation_sha256']
    isolate=json.loads((ROOT/profile['isolation_manifest']).read_text())
    assert profile['tasks']==isolate['evaluation_tasks'] and profile['calibration_tasks']==isolate['calibration_tasks']
    groups={r['goal_group'] for r in profile['tasks']+profile['calibration_tasks']}
    assert len(groups)==9 and not groups&set(isolate['published_source_goal_keys'])
    return isolate


def spec(row,p):return dict(taskName=row['taskName'],variationIdx=row['variationIdx'],simplifications=p['simplifications'],seed=p['simulator_seed'],
    environment_step_limit=p['environment_step_limit'],simulator_profile=p['simulator_profile'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model',choices=['sr','qwen35'],required=True);a=ap.parse_args()
    out=Path(a.run)
    if out.exists():raise ValueError('Fresh Run ID required; preserve partial failures')
    out.mkdir(parents=True);p=json.loads((ROOT/'configs/evaluation/sr_isolated_integration_v1.json').read_text());verify(p)
    code=['internal_signals/structured_decode.py','internal_signals/sr_structured_actor.py','internal_signals/sr_alignment.py','scripts/run_sr_method.py',
          'evaluation/independent_dev.py','scripts/run_phase57_replay.py','reflection/replay_review_v2.py','configs/evaluation/sr_isolated_integration_v1.json',p['sr_actor_config'],p['qwen35_actor_config']]
    manifest=dict(profile=p,model=a.model,source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_hashes={n:file_hash(ROOT/n) for n in code},started_ns=time.time_ns(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        public_training_sources_filtered=True,checkpoint_optimizer_manifest_unavailable=True,no_full_training_isolation_claim=True)
    atomic(out/'manifest.json',manifest)
    limits={k:v for k,v in p['limits'].items() if k!='wall_seconds'};ledger=Ledger(out/'ledger.json',out/'STOP',time.time()+p['limits']['wall_seconds'],limits,limits)
    started=time.perf_counter();actor=None
    try:
        before=time.perf_counter()
        if a.model=='sr':actor=StructuredSRActor(ROOT/p['sr_actor_config'],ROOT/'checkpoints/ee_scienceworld_sr/sr');reviewer=SRReviewerProtocol()
        else:actor=Qwen35Actor(ROOT/p['qwen35_actor_config'],ROOT/'checkpoints/qwen35_9b');reviewer=ReviewerProtocol(p['reviewer_version'])
        atomic(out/'model.json',dict(model_id=actor.config['model_id'],model_revision=actor.config['model_revision'],backend=actor.config['backend'],
            config_sha256=actor.fingerprint,config=actor.config,load_seconds=time.perf_counter()-before,loading_info=actor.loading_info,
            actor_reviewer_same_model_object=True,reviewer_cap=reviewer.max_new_tokens,reviewer_prompt_identical_to_9b=True,grammar_sr_only=a.model=='sr'))
        calibration=[];first=True
        for row in p['calibration_tasks']:
            env=None;history=[]
            try:
                ledger.begin_transition(1,dict(kind='calibration_initial',task=row['task_id']));env=ReplayScienceEnvironment(spec(row,p));ledger.finish_transition(1);ledger.captures(env.capture_count)
                if goal_key(env.goal)!=row['goal_group']:raise ValueError('Native calibration goal mismatch')
                for step in range(p['calibration_max_actions']):
                    prompt=messages(env.goal,env.current['observation'],history,keep=2);seed=request_seed(row['task_id'],step,'draft',0)
                    captured=[]
                    def hook(module,args,output):
                        if len(captured)<8:captured.append(output.logits[0,-1].detach().float().clone())
                    handle=actor.model.register_forward_hook(hook) if first else None
                    ledger.begin_model(min(actor.count(prompt),4096)+512,dict(kind='calibration_draft',task=row['task_id'],step=step))
                    try:text,meta=actor.generate_role(prompt,'calibration_draft',seed,True)
                    finally:
                        if handle is not None:handle.remove()
                    append(out/'calibration_generations.jsonl',dict(metadata=meta));ledger.finish_model(meta)
                    if first:
                        ids=meta['generation_record']['response_ids'][:len(captured)]
                        raw=selected_token_nll(torch.stack(captured[:len(ids)]),ids).tolist()
                        delta=max(abs(x-y) for x,y in zip(raw,meta['generation_record']['token_nll']))
                        receipts=[]
                        for role,scored in [('parity',False),('repeat',True)]:
                            ledger.begin_model(min(actor.count(prompt),4096)+512,dict(kind='numerical_validation',role=role))
                            tt,mm=actor.generate_role(prompt,role,seed,scored);append(out/'validation_generations.jsonl',dict(metadata=mm));ledger.finish_model(mm)
                            receipts.append(mm)
                        validation=dict(raw_forward_hook_nll_delta=delta,scored_unscored_token_parity=receipts[0]['generation_record']['response_ids']==meta['generation_record']['response_ids'],
                            repeated_token_and_nll_exact=receipts[1]['generation_record']['response_ids']==meta['generation_record']['response_ids'] and receipts[1]['generation_record']['token_nll']==meta['generation_record']['token_nll'],
                            extra_forward_passes=0,additional_generation_calls=2,actual_draft=text)
                        atomic(out/'numerical_validation.json',validation)
                        if delta>1e-5 or not validation['scored_unscored_token_parity'] or not validation['repeated_token_and_nll_exact']:raise ValueError('Actual constrained scoring parity failed')
                        first=False
                    if not meta['decision_format_valid'] or meta['internal_signals']['status']!='ok':raise ValueError('Calibration cannot feed missing/invalid signals to new Gate')
                    observation=env.current['observation'];_,_,action=fields(text);count=env.capture_count
                    ledger.begin_transition(1,dict(kind='calibration_formal',task=row['task_id'],step=step));actual=env.step(action);ledger.finish_transition(1);ledger.captures(env.capture_count-count)
                    t=dict(task_id=row['task_id'],step=step,observation=observation,output=text,action=action,metadata=meta,actual=actual)
                    calibration.append(t);append(out/'calibration_steps.jsonl',t);history.append(dict(observation=observation,action=action))
                    atomic(out/'progress.json',dict(stage='calibration',task=row['task_id'],valid_steps=len(calibration)))
                    if actual['done']:break
            finally:
                if env is not None:env.close()
        gate,report=calibrate(calibration,actor);atomic(out/'gate_config.json',gate);atomic(out/'calibration.json',report)
        # Both thresholds frozen before this model's FIRST evaluation episode.
        atomic(out/'threshold_freeze.json',dict(frozen_ns=time.time_ns(),calibration_steps_sha256=file_hash(out/'calibration_steps.jsonl'),gate_sha256=file_hash(out/'gate_config.json')))
        split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());memory=ExperienceMemory(json.loads((ROOT/p['memory_snapshot']).read_text()),split)
        for row in p['tasks']:
            for seed in p['rollout_seeds']:
                for policy in p['policies']:
                    ep=out/'episodes'/row['task_id']/f'seed{seed}'/policy
                    result=run_episode(actor,memory,row,policy,seed,p,gate,ep,ledger,reviewer=reviewer)
                    append(out/'episodes.jsonl',result)
                    atomic(out/'progress.json',dict(stage='evaluation',task=row['task_id'],seed=seed,policy=policy,completed=len(list((out/'episodes').rglob('complete.json')))))
        for n,h in manifest['code_hashes'].items():assert file_hash(ROOT/n)==h,n
        verify(p);atomic(out/'complete.json',dict(status='complete',episodes=12,seconds=time.perf_counter()-started,ledger=ledger.state))
        atomic(out/'progress.json',dict(stage='complete',episodes=12))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=ledger.state));raise


if __name__=='__main__':main()
