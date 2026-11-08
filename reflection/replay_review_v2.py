"""Versioned compact Reviewer. Strict JSON validation, not grammar-constrained decoding."""
import json
from internal_signals.decision import GUIDANCE
from reflection.replay_review import review_messages as original_messages,parse_review as original_parse

VERSION='compact-json-conservative-v2'
REASONS={'KEEP':{'trial_supported','no_grounded_alternative'},'REPLACE':{'trial_failed','trial_harmful'}}
SCHEMA={'type':'object','additionalProperties':False,'required':['decision','final_action','reason_code'],
    'properties':{'decision':{'enum':['KEEP','REPLACE']},'final_action':{'type':'string','minLength':1,'maxLength':256},
                  'reason_code':{'enum':sorted(set.union(*REASONS.values()))}}}
INSTRUCTIONS='''Review one candidate using public feedback from an isolated trial. The real environment has NOT executed the draft.
Output ONLY one compact JSON object with keys decision, final_action, reason_code. No prose, explanation, markdown or reasoning.
Example of format only: {"decision":"KEEP","final_action":"<exact draft>","reason_code":"no_grounded_alternative"}
KEEP when the trial supports the draft without observed harm. Zero score gain alone is not failure.
REPLACE only when public trial evidence shows failure or harm AND one alternative is grounded in the CURRENT observation or committed history. Otherwise KEEP, even when the draft failed.
Use exact known names and one native command; do not invent tools, task answers or unobserved properties. Do not repeat a completed subgoal visible in history.
Trial results are NOT committed. A numbered choice from trial feedback cannot be executed in the current environment unless the CURRENT observation already requests it.
Action acceptance is not goal correctness. Do not infer conductivity, identity, temperature or task completion from names or a zero reward.
For KEEP copy the draft exactly and use reason_code trial_supported or no_grounded_alternative.
For REPLACE use reason_code trial_failed or trial_harmful. final_action must be a nonempty single-line command (at most256 characters).
'''


def review_messages(goal,observation,history,draft,simulation):
    if simulation.get('status')!='ok' or simulation.get('prediction_available') is not True:
        raise ValueError('Unverified simulation cannot reach Reviewer')
    public={k:simulation[k] for k in ('predicted_observation','predicted_reward','predicted_done','predicted_score')}
    payload=dict(goal=goal,current_observation=observation,past_committed_history=history[-2:],draft_action=draft,
                 isolated_trial_feedback_not_committed=public)
    return [dict(role='user',content=GUIDANCE+INSTRUCTIONS+'\n'+json.dumps(payload,ensure_ascii=False))]


def unique_object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate_json_key')
        result[k]=v
    return result


def parse_review(text,draft):
    def invalid(error):return dict(decision='INVALID',final_action=draft,error=error,fallback='keep_draft',reviewer_version=VERSION)
    try:obj=json.loads(text,object_pairs_hook=unique_object)
    except (ValueError,TypeError):return invalid('invalid_or_incomplete_json')
    if not isinstance(obj,dict) or set(obj)!=set(SCHEMA['required']):return invalid('json_schema_keys')
    if any(type(v) is not str for v in obj.values()):return invalid('json_schema_types')
    decision,action,reason=obj['decision'],obj['final_action'],obj['reason_code']
    if decision not in REASONS or reason not in REASONS[decision]:return invalid('inconsistent_decision_reason_code')
    if not action.strip() or action!=action.strip() or len(action)>256 or any(ord(c)<32 for c in action):return invalid('non_single_action_string')
    if decision=='KEEP' and action!=draft:return invalid('keep_changed_draft')
    return dict(decision=decision,final_action=action,reason_code=reason,reason=reason,changed=action!=draft,
                action_validity='unknown',semantic_utility='unknown',reviewer_version=VERSION)


class ReviewerProtocol:
    def __init__(self,version='original'):
        if version not in ('original',VERSION):raise ValueError('Unknown reviewer version')
        self.version=version;self.max_new_tokens=512 if version=='original' else 192
    def messages(self,*args):return original_messages(*args) if self.version=='original' else review_messages(*args)
    def parse(self,text,draft):return original_parse(text,draft) if self.version=='original' else parse_review(text,draft)
    def generate(self,actor,messages,seed):
        from types import SimpleNamespace
        from gates.generation import unscored_generate
        # The shared Actor/model/config remain unchanged. Only this Reviewer call
        # receives its independently versioned cap; no retry or extra forward.
        proxy=SimpleNamespace(config=dict(actor.config,max_new_tokens=self.max_new_tokens),model=actor.model,
            tokenizer=actor.tokenizer,device=actor.device,render=actor.render,sync=actor.sync,request_seed=seed)
        text,meta=unscored_generate(proxy,messages,'review')
        meta.update(role='review',shared_model_id=actor.config['model_id'],shared_model_revision=actor.config['model_revision'],
            config_sha256=actor.fingerprint,reviewer_version=self.version,reviewer_max_new_tokens=self.max_new_tokens,
            json_schema_guidance_only=self.version!='original',grammar_constrained_decoding=False)
        return text,meta
