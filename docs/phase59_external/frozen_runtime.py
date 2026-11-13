"""Frozen-method independent dev runtime; reuses the accepted decision sequence.

Resource caps stop execution; they never alter Gate thresholds, review depth or
which proposals are accepted. Completed episodes are resumable. Partial attempts
are preserved and charged, never silently counted as completed measurements.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time
from experience.collection import fields
from experience.memory import ExperienceMemory
from experience.schema import ExternalQuery
from experience.splits import file_hash,goal_key,validate_manifest
from gates.core import RuleGate,GateInput,ExternalInput
from gates.evidence import resolve_external
from internal_signals.decision import messages
from reflection.replay_review import review_messages,parse_review
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,GroundTruthEnvironmentWM,state_fingerprint


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def request_seed(task,step,role,rollout_seed):
    text='phase57-rollout'+str(rollout_seed)+'|'+task+'|'+str(step)+'|'+role
    return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%(2**31)


def verify_frozen(root,profile):
    for name,expected in profile['method_hashes'].items():
        if file_hash(root/name)!=expected:raise ValueError('Frozen method changed: '+name)
    if file_hash(root/profile['exposure_audit'])!=profile['exposure_audit_sha256']:
        raise ValueError('Exposure audit changed')
    split=json.loads((root/'docs/phase3/split_manifest.json').read_text());validate_manifest(split,require_native=True)
    dev={r['task_id']:r for r in split['partitions']['development']}
    if len(profile['tasks'])!=12 or len({r['task_id'] for r in profile['tasks']})!=12:
        raise ValueError('Frozen12-task cohort required')
    exposure=json.loads((root/profile['exposure_audit']).read_text())
    for row in profile['tasks']:
        if row!=dev[row['task_id']]:raise ValueError('Native dev identity changed')
        if row['task_id'] in exposure['observed_ids'] or row['task_id'] in exposure['declared_ids']:
            raise ValueError('Previously exposed task')
        if row['goal_group'] in exposure['exposed_dev_goal_groups']:raise ValueError('Previously exposed goal')
    if profile['rollout_seeds']!=[0,1] or profile['wm_invocation_budgets']!={'never':0,'always':20,'internal':5,'full':5}:
        raise ValueError('Frozen seed/budget setting changed')
    return split


def verify_model_assets(root,model_dir):
    expected=json.loads((root/'docs/phase57_replay/assets_manifest.json').read_text())
    results={}
    for name,entry in expected['files'].items():
        path=Path(model_dir)/name;h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        actual=h.hexdigest()
        if actual!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise ValueError('Pinned model asset changed: '+name)
        results[name]=dict(sha256=actual,bytes=path.stat().st_size)
    return dict(status='all_pinned_assets_match',model_id=expected['model_id'],revision=expected['revision'],files=results,
                expected_manifest_sha256=file_hash(root/'docs/phase57_replay/assets_manifest.json'),network_used=False)


class ResourceLimit(RuntimeError):pass


class Ledger:
    def __init__(self,path,stop_file,deadline,limits,batch_limits=None):
        self.path=Path(path);self.stop_file=Path(stop_file);self.deadline=deadline;self.limits=limits
        self.batch_limits=batch_limits or dict(tokens=500000,model_calls=330,transition_queries=1512,audit_capture_bundles=12000)
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(
            tokens=0,model_calls=0,transition_queries=0,audit_capture_bundles=0,pending_model=None,pending_transition=None)
        if self.state['pending_model'] or self.state['pending_transition']:
            raise ResourceLimit('Interrupted request has unknown receipt; audit before retry')
        self.batch_start={k:self.state[k] for k in ('tokens','model_calls','transition_queries','audit_capture_bundles')}

    def save(self):atomic(self.path,self.state)

    def check(self,key,extra):
        if self.stop_file.exists():raise ResourceLimit('Supervisor stop requested')
        if time.time()>self.deadline:raise ResourceLimit('Whole-run deadline')
        if self.state[key]+extra>self.limits[key]:raise ResourceLimit('Worker '+key+' cap')
        batch_caps=self.batch_limits
        if self.state[key]-self.batch_start[key]+extra>batch_caps[key]:raise ResourceLimit('Batch '+key+' cap')

    def begin_model(self,max_tokens,identity):
        self.check('model_calls',1);self.check('tokens',max_tokens)
        if self.state['pending_model']:raise ResourceLimit('Overlapping model request')
        self.state['pending_model']=dict(max_tokens=max_tokens,identity=identity,started_ns=time.time_ns());self.save()

    def finish_model(self,metadata):
        if not self.state['pending_model']:raise ResourceLimit('Missing model reservation')
        usage=metadata['usage']['total_tokens']
        if usage>self.state['pending_model']['max_tokens']:raise ResourceLimit('Model receipt exceeds reservation')
        self.state['tokens']+=usage;self.state['model_calls']+=1;self.state['pending_model']=None;self.save()

    def begin_transition(self,maximum,identity):
        self.check('transition_queries',maximum)
        if self.state['pending_transition']:raise ResourceLimit('Overlapping transition request')
        self.state['pending_transition']=dict(maximum=maximum,identity=identity,started_ns=time.time_ns());self.save()

    def finish_transition(self,actual):
        if actual>self.state['pending_transition']['maximum']:raise ResourceLimit('Environment receipt exceeds reservation')
        self.state['transition_queries']+=actual;self.state['pending_transition']=None;self.save()

    def captures(self,actual):
        self.check('audit_capture_bundles',actual);self.state['audit_capture_bundles']+=actual;self.save()

    def begin_batch(self):
        self.batch_start={k:self.state[k] for k in self.batch_start}


def run_episode(actor,memory,row,policy,rollout_seed,profile,gate_config,episode_dir,ledger):
    out=Path(episode_dir);out.mkdir(parents=True,exist_ok=False)
    spec=dict(taskName=row['taskName'],variationIdx=row['variationIdx'],simplifications=profile['simplifications'],
              seed=profile['simulator_seed'],environment_step_limit=profile['environment_step_limit'],simulator_profile=profile['simulator_profile'])
    gate=RuleGate(dict(gate_config,policy=policy));wm=GroundTruthEnvironmentWM();env=None
    started=time.perf_counter();past=[];traces=[];budget=profile['wm_invocation_budgets'][policy]
    atomic(out/'identity.json',dict(task=row,policy=policy,rollout_seed=rollout_seed,spec=spec,started_at_ns=time.time_ns()))
    raw_generate=actor.generate_role
    def logged_generate(prompt,role,seed,scored):
        reservation=min(actor.count(prompt),actor.config['max_prompt_tokens'])+actor.config['max_new_tokens']
        ledger.begin_model(reservation,dict(episode=str(out),step=len(traces),role=role))
        text,meta=raw_generate(prompt,role,seed,scored)
        append(out/'generation_calls.jsonl',dict(available_at_ns=time.time_ns(),metadata=meta))
        ledger.finish_model(meta)
        return text,meta
    try:
        ledger.begin_transition(1,dict(episode=str(out),kind='formal_initialization'))
        begin=time.perf_counter();env=ReplayScienceEnvironment(spec);init_seconds=time.perf_counter()-begin
        ledger.finish_transition(1);ledger.captures(env.capture_count)
        if goal_key(env.goal)!=row['goal_group']:raise ValueError('Native development goal differs; do not replace the task')
        atomic(out/'initial.json',dict(state_sha256=state_fingerprint(env.initial),goal=env.goal,public=env.current,
            simulator=env.simulator,formal_pid=env.pid,initialization_seconds=init_seconds))
        for step in range(profile['maximum_interaction_turns']):
            capture_start=env.capture_count
            ledger.check('audit_capture_bundles',6+3*step+10)
            observation=env.current['observation'];prompt=messages(env.goal,observation,past,keep=2)
            text,meta=logged_generate(prompt,'draft',request_seed(row['task_id'],step,'draft',rollout_seed),gate.needs_internal)
            premise,consequence,draft=fields(text)
            query=ExternalQuery(benchmark='scienceworld',task_id=row['task_id'],episode_id=str(out),step_index=step+1,
                available_at_ns=time.time_ns(),goal=env.goal,observation=observation,draft_action=draft,
                premise=premise,expected_consequence=consequence)
            external=memory.evaluate(query) if gate.needs_external else None
            resolved=resolve_external(external,query,memory) if gate.needs_external else ExternalInput()
            context=GateInput(decision_id=str(out)+'/'+str(step),remaining_wm_budget=budget,remaining_forecast_budget=budget,
                internal=meta.get('internal_signals'),external=resolved,draft_action_valid=meta['decision_format_valid'])
            decision=gate.decide(context);decided_ns=time.time_ns();final=draft
            simulation=review=review_meta=None
            if decision['call_wm']:
                budget-=1
                ledger.begin_transition(step+2,dict(episode=str(out),step=step,kind='lookahead'))
                simulation=wm.simulate_one_step(env,draft)
                cost=simulation['simulation_cost']
                ledger.finish_transition(sum(cost[k] for k in ('initialization_step_calls','replay_step_calls','candidate_step_calls')))
                ledger.captures(cost['audit_capture_bundles'])
                if simulation['prediction_available']:
                    rp=review_messages(env.goal,observation,past,draft,simulation)
                    if actor.count(rp)<=actor.config['max_prompt_tokens']:
                        raw,review_meta=logged_generate(rp,'review',request_seed(row['task_id'],step,'review',rollout_seed),False)
                        review=dict(raw=raw,parsed=parse_review(raw,draft));final=review['parsed']['final_action']
                    else:review=dict(parsed=dict(decision='NOT_CALLED',final_action=draft,error='review_context_overflow'))
                else:review=dict(parsed=dict(decision='NOT_CALLED',final_action=draft,error='reconstruction_failed_keep_draft'))
            ledger.begin_transition(1,dict(episode=str(out),step=step,kind='formal_action'))
            execution_ns=time.time_ns();actual=env.step(final);ledger.finish_transition(1)
            after_sha=state_fingerprint(env.capture());match=None
            if simulation and simulation['prediction_available'] and final==draft:
                match=(all(simulation['candidate_transition'][k]==actual[k] for k in ('observation','reward','done','score','moves'))
                    and simulation['candidate_state_sha256']==after_sha)
            trace=dict(policy=policy,task_id=row['task_id'],rollout_seed=rollout_seed,step=step,goal=env.goal,
                observation=observation,draft_output=text,draft_action=draft,metadata=meta,external=external,
                resolved_external=asdict(resolved),gate=decision,decided_at_ns=decided_ns,simulation=simulation,
                review=review,review_metadata=review_meta,final_action=final,executed_at_ns=execution_ns,actual=actual,
                kept_draft_transition_match=match,formal_after_state_sha256=after_sha,budget_after=budget)
            if simulation and simulation['prediction_available']:
                trace['one_step_score_delta_actual_minus_trial_draft']=actual['score']-simulation['predicted_score']
            append(out/'steps.jsonl',trace);traces.append(trace)
            ledger.captures(env.capture_count-capture_start)
            atomic(out/'progress.json',dict(stage='episode',step=step+1,done=actual['done']))
            if execution_ns<=decided_ns or match is False:raise RuntimeError('Causal/KEEP state consistency violation')
            past.append(dict(observation=observation,action=final))
            if actual['done']:break
        usage=[t['metadata']['usage'] for t in traces]+[t['review_metadata']['usage'] for t in traces if t['review_metadata']]
        result=dict(policy=policy,task_id=row['task_id'],rollout_seed=rollout_seed,variation=row['variationIdx'],
            task_name=row['taskName'],goal_group=row['goal_group'],category=row['category'],task_score=env.current['score'],
            success=env.current['done'] and env.current['score']==100,formal_environment_actions=len(traces),
            simulator_moves=env.current['moves'],done=env.current['done'],action_turn_cap_reached=len(traces)==20,
            lookahead_attempts=sum(t['simulation'] is not None for t in traces),
            verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in traces),
            reconstruction_failures=sum(bool(t['simulation'] and not t['simulation']['prediction_available']) for t in traces),
            reviewer_calls=sum(t['review_metadata'] is not None for t in traces),
            action_corrections=sum(t['draft_action']!=t['final_action'] for t in traces),
            input_tokens=sum(u['prompt_tokens'] for u in usage),output_tokens=sum(u['completion_tokens'] for u in usage),
            total_tokens=sum(u['total_tokens'] for u in usage),seconds=time.perf_counter()-started,
            latency_scope='includes formal initialization and online work; cleanup recorded separately')
    finally:
        if env is not None:
            begin=time.perf_counter();env.close();cleanup=time.perf_counter()-begin
        else:cleanup=0.
    result['formal_cleanup_seconds']=cleanup
    result['total_seconds_including_cleanup']=time.perf_counter()-started
    result['steps_sha256']=file_hash(out/'steps.jsonl');result['generation_calls_sha256']=file_hash(out/'generation_calls.jsonl')
    atomic(out/'complete.json',result);atomic(out/'progress.json',dict(stage='complete',steps=len(traces)))
    return result
