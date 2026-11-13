"""Frozen9B five-policy ALFWorld pilot; no training or final140-game evaluation."""
import argparse,json,os,subprocess,sys,time,signal
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,Ledger,request_seed
from experience.splits import file_hash
from experience.collection import fields
from experience.alfworld_memory import AlfExperienceMemory
from gates.core import RuleGate,GateInput,ExternalInput
from internal_signals.qwen35_actor import Qwen35Actor
from internal_signals.alfworld_prompts import draft_messages,review_messages,parse_review
from worldmodel.alfworld_environment import AlfEnvironment,fingerprint,AlfGroundTruthWM
from scripts.run_phase57_replay import calibrate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-python',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--memory',required=True);ap.add_argument('--run',required=True);ap.add_argument('--native-validation',required=True);a=ap.parse_args()
    out=Path(a.run)
    if out.exists():raise ValueError('Fresh Run ID required; keep all partial attempts')
    out.mkdir(parents=True);p=json.loads((ROOT/'configs/evaluation/alfworld_pilot_v1.json').read_text())
    def stop(*_):raise InterruptedError('Owned supervisor stop; retain partial receipts')
    signal.signal(signal.SIGTERM,stop)
    if file_hash(ROOT/p['split_manifest'])!=p['split_sha256']:raise ValueError('Data split identity changed')
    split=json.loads((ROOT/p['split_manifest']).read_text());snapshot=json.loads(Path(a.memory).read_text());memory=AlfExperienceMemory(snapshot,split)
    validation=Path(a.native_validation)
    if json.loads((validation/'complete.json').read_text())['status']!='pass':raise ValueError('Real replay validation not passed')
    native_goals={r['task_id']:r for r in json.loads((validation/'goal_identity.json').read_text())}
    names=['scripts/run_alfworld_pilot.py','scripts/launch_alfworld_pilot.py','scripts/alfworld_worker.py','worldmodel/alfworld_environment.py','experience/alfworld_memory.py',
        'internal_signals/alfworld_prompts.py','internal_signals/qwen35_actor.py','internal_signals/hf_actor.py','internal_signals/provider.py',
        'gates/core.py','scripts/run_phase57_replay.py','configs/evaluation/alfworld_pilot_v1.json',p['actor_config']]
    manifest=dict(profile=p,split=split,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:file_hash(ROOT/n) for n in names},memory_sha256=file_hash(a.memory),memory_cards=len(snapshot['cards']),
        env_python=a.env_python,data_root=a.data_root,model_revision=p['model_revision'],pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        native_validation_sha256=file_hash(validation/'complete.json'),native_goals_sha256=file_hash(validation/'goal_identity.json'),
        no_model_training=True,no_test_rollouts=True,information_access='oracle-like real-environment one-step queries',started_ns=time.time_ns())
    atomic(out/'manifest.json',manifest)
    limits={k:v for k,v in p['limits'].items() if k!='wall_seconds'};ledger=Ledger(out/'ledger.json',out/'STOP',time.time()+p['limits']['wall_seconds'],limits,limits)
    started=time.perf_counter();episodes=[];numerical_done=False
    try:
        before=time.perf_counter();actor=Qwen35Actor(ROOT/p['actor_config'],ROOT/'checkpoints/qwen35_9b')
        actor.evaluation_left_truncation=False # ALF prompt overflow is an error, never silent goal deletion.
        atomic(out/'model.json',dict(model_id=p['model_id'],revision=actor.config['model_revision'],loading_info=actor.loading_info,
            config_sha256=actor.fingerprint,load_seconds=time.perf_counter()-before,config=actor.config,
            shared_actor_reviewer_model_object=True,thinking=False,original_three_line_reviewer=True))
        def generate(prompt,role,seed,scored,where):
            count=actor.count(prompt)
            if count>2048:raise ValueError('ALFWorld prompt exceeds fixed2048; no hidden truncation/history change')
            ledger.begin_model(count+512,dict(role=role,where=where))
            text,meta=actor.generate_role(prompt,role,seed,scored)
            append(out/'generation_calls.jsonl',dict(where=where,role=role,available_at_ns=time.time_ns(),metadata=meta));ledger.finish_model(meta)
            return text,meta
        def new_env(row):
            ledger.begin_transition(1,dict(kind='initialization',task=row['task_id']));env=AlfEnvironment(row,a.env_python,a.data_root,out/'native')
            ledger.finish_transition(1);ledger.captures(env.capture_count)
            expected=native_goals[row['task_id']]
            if env.goal!=expected['goal'] or fingerprint(env.initial)!=expected['initial_sha256']:
                env.close();raise ValueError('Native task goal/state identity changed')
            return env
        def formal_step(env,action,identity):
            before=env.capture_count;ledger.begin_transition(1,identity);actual=env.step(action);ledger.finish_transition(1);ledger.captures(env.capture_count-before);return actual
        records=[]
        for row in split['calibration_tasks']:
            env=new_env(row);past=[]
            try:
                for step in range(8):
                    prompt=draft_messages(env.goal,env.current['observation'],past,env.current['admissible_commands']);seed=request_seed(row['task_id'],step,'draft',0)
                    text,meta=generate(prompt,'draft',seed,True,'calibration/'+row['task_id'])
                    if not numerical_done:
                        tt,other=generate(prompt,'draft',seed,False,'scored_unscored_parity')
                        rr,repeated=generate(prompt,'draft',seed,True,'repeat_nll_parity')
                        same=meta['generation_record']['response_ids']==other['generation_record']['response_ids']==repeated['generation_record']['response_ids']
                        exact=meta['generation_record']['token_nll']==repeated['generation_record']['token_nll']
                        atomic(out/'numerical_validation.json',dict(token_parity=same,raw_nll_repeat_exact=exact,extra_generation_calls=2,extra_forward_passes=0,
                            first_real_draft=text,internal_signals=meta['internal_signals']))
                        if not same or not exact:raise ValueError('ALF scored/unscored/repeat path differs')
                        numerical_done=True
                    _,_,action=fields(text)
                    if not action or '\n' in action:raise ValueError('No single draft command; retain format failure')
                    observation=env.current['observation'];actual=formal_step(env,action,dict(kind='calibration',task=row['task_id'],step=step))
                    record=dict(task_id=row['task_id'],step=step,observation=observation,output=text,action=action,metadata=meta,actual=actual)
                    records.append(record);append(out/'calibration_steps.jsonl',record);past.append(dict(observation=observation,action=action))
                    atomic(out/'progress.json',dict(stage='calibration',task=row['task_id'],steps=len(records)))
                    if actual['done']:break
            finally:env.close()
        if any(r['metadata']['internal_signals']['status']!='ok' for r in records):raise ValueError('ALF calibration has missing/invalid spans; stop before Gate evaluation')
        gate_config,calibration=calibrate(records,actor);calibration['budgets_selected_independently']=p['wm_invocation_budgets']
        atomic(out/'gate_config.json',gate_config);atomic(out/'calibration.json',calibration)
        atomic(out/'threshold_freeze.json',dict(frozen_ns=time.time_ns(),gate_sha256=file_hash(out/'gate_config.json'),calibration_sha256=file_hash(out/'calibration_steps.jsonl')))
        for row in split['evaluation_tasks']:
            for seed in p['rollout_seeds']:
                for policy in p['policies']:
                    env=new_env(row);ep=out/'episodes'/row['task_id']/policy;ep.mkdir(parents=True);past=[];traces=[];budget=p['wm_invocation_budgets'][policy]
                    gate=RuleGate(dict(gate_config,policy=policy));begin=time.perf_counter();passive=0
                    atomic(ep/'initial.json',dict(task=row,initial=env.initial,goal=env.goal,formal_pid=env.pid))
                    try:
                        for step in range(50):
                            observation=env.current['observation'];commands=env.current['admissible_commands']
                            prompt=draft_messages(env.goal,observation,past,commands)
                            text,meta=generate(prompt,'draft',request_seed(row['task_id'],step,'draft',seed),gate.needs_internal,policy+'/'+row['task_id'])
                            premise,consequence,draft=fields(text)
                            if not draft or '\n' in draft:raise ValueError('Invalid single-action draft')
                            query=dict(task_id=row['task_id'],episode_id=str(ep),goal=env.goal,observation=observation,draft_action=draft,commands=commands,available_at_ns=time.time_ns())
                            external,resolved=memory.evaluate(query) if gate.needs_external else (None,ExternalInput())
                            context=GateInput(str(ep)+'/'+str(step),budget,meta.get('internal_signals'),resolved,budget,meta['decision_format_valid'])
                            decision=gate.decide(context);decided=time.time_ns();final=draft;simulation=review=review_meta=None
                            if decision['call_wm']:
                                budget-=1;count=env.capture_count;ledger.begin_transition(len(env.history)+2,dict(kind='lookahead',task=row['task_id'],step=step))
                                simulation=AlfGroundTruthWM().simulate(env,draft);cost=simulation['simulation_cost']
                                ledger.finish_transition(cost['initializations']+cost['replay_steps']+cost['candidate_steps']);ledger.captures(cost['state_capture_bundles']+env.capture_count-count)
                                passive+=cost['formal_inspections']
                                if simulation['prediction_available']:
                                    rp=review_messages(env.goal,observation,past,commands,draft,simulation)
                                    raw,review_meta=generate(rp,'review',request_seed(row['task_id'],step,'review',seed),False,policy+'/'+row['task_id'])
                                    review=dict(raw=raw,parsed=parse_review(raw,draft));final=review['parsed']['final_action']
                                else:review=dict(parsed=dict(decision='NOT_CALLED',final_action=draft,error='reconstruction_failed_keep_draft'))
                            execution=time.time_ns();actual=formal_step(env,final,dict(kind='formal',task=row['task_id'],policy=policy,step=step))
                            kept_match=None
                            if simulation and simulation['prediction_available'] and final==draft:
                                kept_match=actual==simulation['candidate_transition']
                                if not kept_match:raise RuntimeError('ALF verified KEEP transition differs')
                            trace=dict(benchmark='alfworld',task_id=row['task_id'],policy=policy,rollout_seed=seed,step=step,goal=env.goal,
                                observation=observation,admissible_commands=commands,draft_output=text,draft_action=draft,metadata=meta,
                                external=external,resolved_external=asdict(resolved),gate=decision,decided_at_ns=decided,simulation=simulation,
                                review=review,review_metadata=review_meta,final_action=final,executed_at_ns=execution,actual=actual,
                                kept_draft_transition_match=kept_match,budget_after=budget,draft_admissible=draft in commands,final_admissible=final in commands)
                            append(ep/'steps.jsonl',trace);traces.append(trace);past.append(dict(observation=observation,action=final))
                            if not decided<execution:raise RuntimeError('Gate decided after formal action')
                            if actual['done']:break
                        usage=[t['metadata']['usage'] for t in traces]+[t['review_metadata']['usage'] for t in traces if t['review_metadata']]
                        result=dict(task_id=row['task_id'],policy=policy,rollout_seed=seed,family=row['family'],goal_group=row['goal_group'],success=env.current['success'],
                            score_100_won=env.current['score'],done=env.current['done'],horizon_truncated=env.current['truncated'],formal_steps=len(traces),
                            wm_calls=sum(t['simulation'] is not None for t in traces),verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in traces),
                            reviewer_calls=sum(t['review_metadata'] is not None for t in traces),internal_extractions=sum('internal_signals' in t['metadata'] for t in traces),
                            retrieval_calls=sum(t['external'] is not None for t in traces),tokens=sum(u['total_tokens'] for u in usage),input_tokens=sum(u['prompt_tokens'] for u in usage),
                            output_tokens=sum(u['completion_tokens'] for u in usage),passive_state_queries=passive,total_seconds=time.perf_counter()-begin)
                    finally:env.close()
                    result['steps_sha256']=file_hash(ep/'steps.jsonl');atomic(ep/'complete.json',result);append(out/'episodes.jsonl',result);episodes.append(result)
                    atomic(out/'progress.json',dict(stage='evaluation',task=row['task_id'],policy=policy,completed=len(episodes)))
        for n,h in manifest['source_hashes'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Run source changed: '+n)
        atomic(out/'complete.json',dict(status='complete',episodes=len(episodes),seconds=time.perf_counter()-started,ledger=ledger.state));atomic(out/'progress.json',dict(stage='complete',episodes=len(episodes)))
    except BaseException as e:
        atomic(out/'error.json',dict(type=type(e).__name__,message=str(e),ledger=ledger.state));raise


if __name__=='__main__':main()
