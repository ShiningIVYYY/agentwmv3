"""New-protocol real episodes; future feedback never enters pre-action gate."""
import copy,json,random,time
from pathlib import Path
import numpy as np
from evaluation.independent_dev import atomic,append,request_seed
from experience.alfworld_memory import AlfExperienceMemory
from internal_signals.activation_protocol import GUIDANCE,INSTRUCTIONS,parse_review_guarded
from internal_signals.activation_training import Ensemble


def run_eval(rt,row,policy,heads,calibration,memory):
    baseline=dict(rt.ledger.state)
    env=rt.new_env(row);past=[];budget=rt.config['horizon'] if policy=='always' else 0 if policy=='never' else rt.config['budget']
    start=time.perf_counter();traces=[];termination=None
    try:
        for step in range(rt.config['horizon']):
            obs=env.current['observation'];commands=env.current['admissible_commands']
            scored=policy not in ['never','always','admissible','random','periodic']
            parsed,text,meta=rt.draft(env,past,row,scored=scored)
            if not parsed['valid']:
                termination='agent_format_abort'
                append(rt.out/'evaluation_format_errors.jsonl',dict(task_id=row['task_id'],policy=policy,step=step,raw=text,usage=meta['usage']))
                break
            draft=parsed['action'];external=None;conflict=False;probe={};capture_cost=None
            if policy.endswith('_external'):
                external,resolved=memory.evaluate(dict(task_id=row['task_id'],episode_id=policy+'/'+row['task_id'],goal=env.goal,
                    observation=obs,draft_action=draft,commands=commands,available_at_ns=time.time_ns()))
                conflict=resolved.evidence_verified and resolved.conflict=='true'
            if policy.startswith(('risk','utility')):
                values,capture_cost=rt.features(meta)
                needed=['native_rejection'] if policy.startswith('risk') else ['utility_help','utility_harm']
                for target in needed:
                    probe[target]=heads[target]['provider'].extract(values,rt.contract)
            signal=meta.get('internal_signals');high_nll=False
            if signal:
                high_nll=(signal['status']!='ok' or signal['local_uncertainty']>calibration['tau_local'] or signal['global_uncertainty']>calibration['tau_global'])
            if policy=='never':requested=False;reason='never'
            elif policy=='always':requested=True;reason='always'
            elif policy=='admissible':requested=draft not in commands;reason='public_action_membership'
            elif policy=='random':requested=random.Random(request_seed(row['task_id'],step,'random',0)).random()<.1;reason='predeclared_random_p0.1'
            elif policy=='periodic':requested=step%10==0;reason='periodic_every10'
            elif policy in ['nll','nll_external','review_only']:requested=high_nll or conflict;reason='new_protocol_nll_or_verified_conflict'
            elif policy.startswith('risk'):
                requested=probe['native_rejection']['score']>probe['native_rejection']['threshold'] or conflict;reason='activation_risk_or_verified_conflict'
            elif policy.startswith('utility'):
                threshold=heads['utility_help']['result']['threshold_conflict' if conflict else 'threshold']
                requested=probe['utility_help']['score']>threshold and probe['utility_harm']['score']<heads['utility_harm']['result']['threshold'];reason='utility_help_with_harm_veto'
            else:raise ValueError('Unknown policy')
            call=bool(requested and budget>0);decided=time.time_ns();final=draft;sim=review=review_meta=None
            if call:
                budget-=1
                if policy=='review_only':
                    msg=[dict(role='user',content=GUIDANCE+INSTRUCTIONS+'\nReason: one short clause, <=20 words. No simulation was performed; there is NO predicted feedback. '
                        +'Review only current public evidence.\nGoal: '+env.goal+'\nCurrent observation: '+obs+'\nExecuted history: '+json.dumps(past[-2:])+
                        '\nCURRENT admissible commands: '+json.dumps(commands)+'\nDraft Action: '+draft)]
                    raw,review_meta=rt.generate(msg,'review',row['task_id'],step,0,False)
                    parsed_review=parse_review_guarded(raw,draft,commands) if review_meta['finish_reason']!='length' else dict(decision='INVALID',final_action=draft,error='review_truncated',fallback='keep_draft')
                    review=dict(raw=raw,**parsed_review);final=review['final_action']
                else:
                    sim=rt.simulate(env,draft);review,review_meta=rt.review(env,past,row,draft,sim);final=review['final_action']
            execution=time.time_ns();actual=rt.step(env,final)
            assert decided<execution
            if sim and sim['prediction_available'] and final==draft and actual!=sim['candidate_transition']:raise ValueError('Evaluation KEEP transition mismatch')
            t=dict(task_id=row['task_id'],goal_group=row['goal_group'],policy=policy,step=step,observation=obs,commands=commands,draft=text,draft_action=draft,
                metadata=meta,probe=probe,capture_cost=capture_cost,external=external,gate=dict(call=call,requested=requested,reason=reason,budget_after=budget),
                decided_at_ns=decided,executed_at_ns=execution,simulation=sim,review=review,review_metadata=review_meta,final_action=final,
                final_admissible=final in commands,actual=actual)
            tracefile=rt.out/'traces'/row['task_id']/(policy+'.jsonl');tracefile.parent.mkdir(parents=True,exist_ok=True)
            append(tracefile,t);traces.append(t);past.append(dict(observation=obs,action=final))
            if actual['done']:break
        result=dict(task_id=row['task_id'],goal_group=row['goal_group'],family=row['family'],policy=policy,seed=0,success=env.current['success'],
            score_100_won=env.current['score'],native_done=env.current['done'],termination=termination or 'native_or_horizon',formal_steps=env.steps,
            gate_calls=sum(t['gate']['call'] for t in traces),wm_calls=sum(t['simulation'] is not None for t in traces),
            verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in traces),
            reviewer_calls=sum(t['review_metadata'] is not None for t in traces),revisions=sum(t['final_action']!=t['draft_action'] for t in traces),
            nonadmissible_final=sum(not t['final_admissible'] for t in traces),seconds=time.perf_counter()-start,
            cost={k:rt.ledger.state[k]-baseline[k] for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']},
            activation_forwards=sum(t['capture_cost'] is not None for t in traces),
            activation_forward_seconds=sum(t['capture_cost']['seconds'] for t in traces if t['capture_cost']),
            probe_seconds=sum(p['extraction_latency'] for t in traces for p in t['probe'].values()))
        append(rt.out/'episodes.jsonl',result)
        atomic(rt.out/'episodes'/row['task_id']/(policy+'.json'),result)
        return result
    finally:env.close()
