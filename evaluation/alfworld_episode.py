"""Generated verbatim episode logic; scheduling only. Do not edit by hand."""
import time
from dataclasses import asdict
from scripts.run_alfworld_pilot import (atomic,append,AlfEnvironment,AlfGroundTruthWM,RuleGate,GateInput,ExternalInput,
    draft_messages,review_messages,parse_review,request_seed,fields,file_hash,fingerprint)


def run_episode(*,row,policy,seed,actor,memory,gate_config,p,out,a,ledger,native_goals,episodes):
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

    return result
