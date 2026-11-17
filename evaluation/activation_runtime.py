"""Shared real-model/real-environment runtime for activation experiments v1."""
import copy,json,time,os
from pathlib import Path
from types import SimpleNamespace
from evaluation.independent_dev import atomic,append,request_seed,ResourceLimit
from scripts.run_alfworld_parallel import ReceiptLedger
from internal_signals.activation_protocol import draft_messages,parse_draft,review_messages,parse_review_guarded,protocol_hash
from internal_signals.activation_capture import ActivationCapture,feature_record
from internal_signals.qwen35_actor import Qwen35Actor
from worldmodel.alfworld_environment import AlfEnvironment,AlfGroundTruthWM,fingerprint
from internal_signals.token_nll import FORMAT
from internal_signals.hidden_states import response_alignment


def field_masks(actor,meta):
    r=meta['generation_record'];text,offsets=response_alignment(actor.tokenizer,r['prompt_text'],r['prompt_ids'],r['response_ids'])
    m=FORMAT.fullmatch(text)
    if m is None:raise ValueError('Cannot mask an invalid decision')
    return {name:[i for i,(a,b) in enumerate(offsets) if a<m.end(name) and b>m.start(name)] for name in ['premise','consequence','action']}


class Runtime:
    def __init__(self,root,out,config,deadline):
        self.root=Path(root);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.config=dict(config)
        for key in ['env_python','data_root']:
            self.config[key]=os.path.expandvars(self.config[key])
            if '$' in self.config[key]:raise ValueError('Missing runtime environment variable: '+key)
        self.ledger=ReceiptLedger(self.out/'ledger.json',self.out.parent.parent/'STOP',deadline,config['stage_limits'],config['stage_limits'])
        start=time.perf_counter();self.actor=Qwen35Actor(self.root/config['actor_config'],self.root/'checkpoints/qwen35_9b')
        self.actor.evaluation_left_truncation=False;self.capture=ActivationCapture(self.actor)
        import hashlib
        assets={k:v for k,v in self.actor.asset_hashes.items() if any(s in k for s in ['tokenizer','vocab','merges','chat_template'])}
        if not assets:raise ValueError('Missing verified tokenizer assets')
        self.contract=dict(actor_config_sha256=self.actor.fingerprint,actor_revision=self.actor.config['model_revision'],protocol_sha256=protocol_hash(),
            capture=self.capture.description,tokenizer_sha256=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest())
        atomic(self.out/'model.json',dict(contract=self.contract,loading_info=self.actor.loading_info,load_seconds=time.perf_counter()-start,
            frozen_actor=not any(p.requires_grad for p in self.actor.model.parameters()),actor_reviewer_shared=True))

    def generate(self,messages,role,task,step,seed=0,scored=False):
        n=self.actor.count(messages)
        if n>2048:raise ValueError('mandatory_prompt_overflow')
        self.ledger.begin_model(n+512,dict(role=role,task=task,step=step,seed=seed))
        text,meta=self.actor.generate_role(messages,role,request_seed(task,step,role,seed),scored)
        append(self.out/'generation_calls.jsonl',dict(role=role,task_id=task,step=step,rollout_seed=seed,metadata=meta,at_ns=time.time_ns()))
        self.ledger.finish_model(meta)
        return text,meta

    def draft(self,env,past,row,seed=0,scored=True):
        text,meta=self.generate(draft_messages(env.goal,env.current['observation'],past,env.current['admissible_commands']),
            'draft',row['task_id'],env.steps,seed,scored)
        return parse_draft(text,meta['finish_reason']=='length'),text,meta

    def features(self,meta):
        self.ledger.check('model_calls',0)
        values,cost=self.capture.extract(feature_record(meta))
        append(self.out/'compute.jsonl',dict(kind='activation_forward',cost=cost,at_ns=time.time_ns()))
        return values,cost

    def new_env(self,row):
        self.ledger.begin_transition(1,dict(kind='reset',task=row['task_id']))
        env=AlfEnvironment(row,self.config['env_python'],self.config['data_root'],self.out/'native')
        self.ledger.finish_transition(1);self.ledger.captures(env.capture_count)
        return env

    def step(self,env,action):
        before=env.capture_count;self.ledger.begin_transition(1,dict(kind='step',task=env.row['task_id'],step=env.steps))
        actual=env.step(action);self.ledger.finish_transition(1);self.ledger.captures(env.capture_count-before)
        return actual

    def simulate(self,env,action):
        before=env.capture_count;self.ledger.begin_transition(len(env.history)+2,dict(kind='lookahead',task=env.row['task_id'],step=env.steps))
        sim=AlfGroundTruthWM().simulate(env,action);sc=sim['simulation_cost']
        self.ledger.finish_transition(sc['initializations']+sc['replay_steps']+sc['candidate_steps'])
        self.ledger.captures(sc['state_capture_bundles']+env.capture_count-before)
        return sim

    def review(self,env,past,row,draft,sim,seed=0):
        if not sim['prediction_available']:return dict(decision='UNAVAILABLE',final_action=draft,error=sim['status']),None
        raw,meta=self.generate(review_messages(env.goal,env.current['observation'],past,env.current['admissible_commands'],draft,sim),
            'review',row['task_id'],env.steps,seed,False)
        parsed=parse_review_guarded(raw,draft,env.current['admissible_commands']) if meta['finish_reason']!='length' else dict(decision='INVALID',final_action=draft,error='review_truncated',fallback='keep_draft')
        return dict(raw=raw,**parsed),meta

    def rebuild(self,row,case):
        env=self.new_env(row)
        try:
            for action,expected in case['prefix']:
                if self.step(env,action)!=expected:raise ValueError('Label prefix transition mismatch')
            if fingerprint(env.current)!=case['state_sha256'] or env.goal!=case['goal']:raise ValueError('Label source state mismatch')
            return env
        except BaseException:
            env.close();raise


def native_label(action,commands,actual):
    if actual['observation'].strip()=='Nothing happens.':return 1 if action not in commands else None
    return 0


def collect_row(rt,row,role,max_steps=12):
    import torch
    env=rt.new_env(row);past=[];cases=[];start=time.perf_counter()
    try:
        for step in range(max_steps):
            state=copy.deepcopy(env.current);prefix=copy.deepcopy(env.history)
            parsed,text,meta=rt.draft(env,past,row,scored=True)
            if not parsed['valid']:
                append(rt.out/'format_errors.jsonl',dict(task_id=row['task_id'],role=role,step=step,raw=text,error=parsed['error'],last_public_state=state))
                break
            values,cost=rt.features(meta);feature_at=time.time_ns()
            case_id=row['task_id']+'/'+str(step);file=rt.out/'features'/row['task_id']/(str(step)+'.pt');file.parent.mkdir(parents=True,exist_ok=True)
            masks=field_masks(rt.actor,meta)
            if any(not indices for indices in masks.values()):raise ValueError('Missing aligned field mask')
            torch.save(dict(features=values,contract=rt.contract,case_id=case_id,field_token_indices=masks),file)
            # Only after the features are fixed may the real current outcome be observed.
            actual=rt.step(env,parsed['action']);label_at=time.time_ns()
            r=dict(case_id=case_id,task_id=row['task_id'],goal_group=row['goal_group'],role=role,step=step,goal=env.goal,
                observation=state['observation'],commands=state['admissible_commands'],draft=text,draft_action=parsed['action'],
                metadata=meta,feature_file=str(file.relative_to(rt.root)),capture_cost=cost,feature_available_at_ns=feature_at,
                label_observed_at_ns=label_at,native_rejection=native_label(parsed['action'],state['admissible_commands'],actual),
                actual=actual,state_sha256=fingerprint(state),prefix=prefix,past=copy.deepcopy(past),wm_called=False)
            assert feature_at<label_at
            append(rt.out/'cases.jsonl',r);cases.append(r);past.append(dict(observation=state['observation'],action=parsed['action']))
            if actual['done']:break
    finally:env.close()
    atomic(rt.out/('collection_'+row['task_id']+'.json'),dict(task_id=row['task_id'],role=role,cases=len(cases),scope='prefix_collection_not_benchmark_episode',seconds=time.perf_counter()-start))
    return cases


def utility_pair(rt,row,case,seed=0):
    # Reserve the worst-case whole pair before starting either branch.
    remaining=50-case['step'];calls=2*(remaining-1)+1
    maximum=dict(model_calls=calls,tokens=calls*(2048+512),transition_queries=2+2*case['step']+2*remaining+case['step']+2,
        audit_capture_bundles=2+2*case['step']+2*remaining+case['step']+4)
    try:
        for key,amount in maximum.items():rt.ledger.check(key,amount)
    except ResourceLimit as e:
        if rt.ledger.stop_file.exists() or time.time()>rt.ledger.deadline:raise
        result=dict(case_id=case['case_id'],task_id=row['task_id'],goal_group=row['goal_group'],role=case['role'],step=case['step'],
            feature_file=case['feature_file'],state_sha256=case['state_sha256'],seed=seed,valid=False,delta_return=None,label='unknown',
            branches={},error='whole_pair_budget_reservation_failed: '+str(e),maximum_cost_reservation=maximum,
            feature_available_at_ns=case['feature_available_at_ns'],label_observed_at_ns=time.time_ns())
        append(rt.out/'utility_labels.jsonl',result);return result
    outputs={}
    for branch in ['skip','call']:
        env=rt.rebuild(row,case);past=copy.deepcopy(case['past']);traces=[];sim=review=None
        try:
            action=case['draft_action'];before=env.current['observation']
            if branch=='call':
                sim=rt.simulate(env,action)
                if not sim['prediction_available']:
                    outputs[branch]=dict(valid=False,error='reconstruction_failed');continue
                review,_=rt.review(env,past,row,action,sim,seed);action=review['final_action']
            actual=rt.step(env,action)
            append(rt.out/'utility_traces.jsonl',dict(case_id=case['case_id'],branch=branch,formal_step=env.steps,action=action,actual=actual))
            if sim and action==case['draft_action'] and actual!=sim['candidate_transition']:raise ValueError('Utility KEEP mismatch')
            past.append(dict(observation=before,action=action));traces.append(dict(action=action,actual=actual))
            aborted=False
            while not env.current['done']:
                before=env.current['observation'];parsed,text,_=rt.draft(env,past,row,seed,False)
                if not parsed['valid']:aborted=True;break
                actual=rt.step(env,parsed['action']);past.append(dict(observation=before,action=parsed['action']))
                append(rt.out/'utility_traces.jsonl',dict(case_id=case['case_id'],branch=branch,formal_step=env.steps,action=parsed['action'],actual=actual))
                traces.append(dict(action=parsed['action'],actual=actual))
            outputs[branch]=dict(valid=True,success=env.current['success'],score=env.current['score'],formal_steps=env.steps,
                format_abort=aborted,review=review,simulation=sim,traces=traces,continuation='frozen_actor_never_wm',remaining_budget_at_intervention=50-case['step'])
        finally:env.close()
    valid=all(outputs.get(k,{}).get('valid') for k in ['skip','call'])
    consistency_error=None
    if valid and outputs['skip']['traces'][0]['action']==outputs['call']['traces'][0]['action'] and outputs['skip']['traces']!=outputs['call']['traces']:
        valid=False;consistency_error='Unchanged intervention yielded unequal fixed-policy continuation; utility remains Unknown'
    delta=int(outputs['call']['success'])-int(outputs['skip']['success']) if valid else None
    result=dict(case_id=case['case_id'],task_id=row['task_id'],goal_group=row['goal_group'],role=case['role'],step=case['step'],
        feature_file=case['feature_file'],state_sha256=case['state_sha256'],seed=seed,valid=valid,delta_return=delta,
        label='unknown' if delta is None else {-1:'harm',0:'neutral',1:'help'}[delta],branches=outputs,
        feature_available_at_ns=case['feature_available_at_ns'],label_observed_at_ns=time.time_ns(),consistency_error=consistency_error,
        limitation='Single continuation seed; estimates this fixed protocol, not calibrated universal utility')
    append(rt.out/'utility_labels.jsonl',result)
    return result
