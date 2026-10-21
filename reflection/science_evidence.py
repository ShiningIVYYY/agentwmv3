"""Conservative ScienceWorld intervention ablations; no simulator-label lookups.

v1: evidence-gated criticism and success preservation.
v2: additionally compare both actions from the same real state.
v3: additionally retain source-labelled, historical observation excerpts.
v3-obs: v3 with all privileged transition metadata hidden from selection.
"""
import json
import re

from .agentwm import SCIENCE_REPLAN_REGEX, brief_text, extract_object, parse_replan
from .controller import valid_replacement
from worldmodel.backends import PredictionError

PROTOCOLS = tuple('science-evidence-' + x for x in ('v1', 'v2', 'v3', 'v3-obs'))
ISSUES = ('none', 'invalid_input', 'terminal_failure', 'harmful_change', 'loop')
REVIEW_SCHEMA = dict(type='object', additionalProperties=False, properties=dict(
    decision=dict(type='string', enum=['keep', 'replan']),
    issue=dict(type='string', enum=list(ISSUES)),
    evidence_source=dict(type='string', enum=['prediction', 'history', 'none']),
    evidence=brief_text(220), reason=brief_text(220)),
    required=['decision', 'issue', 'evidence_source', 'evidence', 'reason'])
PAIR_SCHEMA = dict(type='object', additionalProperties=False, properties=dict(
    choice=dict(type='string', enum=['original', 'alternative', 'uncertain']),
    fixes_original_issue=dict(type='boolean'), introduces_risk=dict(type='boolean'),
    original_evidence=brief_text(220), alternative_evidence=brief_text(220), reason=brief_text(220)),
    required=['choice', 'fixes_original_issue', 'introduces_risk',
              'original_evidence', 'alternative_evidence', 'reason'])
RULES = """You are reviewing an action for the CURRENT REAL ScienceWorld state.
Forecasts describe counterfactual outcomes, never events that have already happened.
Default to KEEP. A zero reward, unchanged score, or done=False is not a defect.
Preserve useful setup: taking containers, connecting a circuit, heating/cooling,
measuring, reading, and moving required objects may take many steps without reward.
Only request replanning for explicit invalid input, terminal failure, a destructive
state change, or a verified repeated information/navigation loop. A different
plausible plan is not evidence that the original action is wrong.
When the forecast explicitly establishes success, preserve the action.
Quote evidence verbatim from the supplied prediction or executed history.
If the evidence is ambiguous, absent, or inconsistent with your reason, KEEP.
All historical excerpts are past observations, not guaranteed current facts.
Learned Word2World forecasts can be wrong. Missing metadata means UNKNOWN.
"""
INVALID = re.compile(r"no known action matches|(?:it's|its) not clear how to|not clear how to go|"
                     r"not sure how to|don't know how to|unknown action|invalid action", re.I)
DESTRUCTIVE = re.compile(r'\b(destroyed|damaged|burned up|broken|irreversibly|died|dead)\b', re.I)
INFORMATION = re.compile(r'^(look around|inventory|task|look at .+|look in .+|go to .+)$', re.I)
NOOP = re.compile(r'\balready (?:open|closed|connected|activated|deactivated)\b', re.I)


def norm(text):
    return ' '.join(str(text).split()).casefold()


def quoted(quote, source):
    return isinstance(quote, str) and len(norm(quote)) >= 12 and norm(quote) in norm(source)


def options(protocol):
    return dict(compare=protocol != 'science-evidence-v1',
                memory=protocol in ('science-evidence-v3', 'science-evidence-v3-obs'),
                metadata=protocol != 'science-evidence-v3-obs')


def visible(prediction, backend, metadata=True):
    value = dict(observation=prediction.get('observation', ''),
        available=prediction.get('prediction_available', True),
        complete=prediction.get('observation_complete', True), source=backend)
    if metadata and backend == 'oracle':
        value.update({k: prediction.get(k) for k in ('reward', 'done', 'success', 'score')})
    elif metadata and backend == 'word2world':
        value['predicted_success'] = prediction.get('predicted_success')
    return value


def success(prediction, backend, metadata=True):
    p = visible(prediction, backend, metadata)
    return bool(p['available'] and p['complete'] and
        ((backend == 'oracle' and p.get('done') is True and p.get('success') is True and p.get('score') == 100)
         or (backend == 'word2world' and p.get('predicted_success') is True and not INVALID.search(p['observation']))))


def loop_evidence(action, steps):
    # Do not call repeated waits/heating/connection/focus a loop: hidden physical
    # processes can advance even when their text observations are identical.
    if not INFORMATION.fullmatch(action) or len(steps) < 3:
        return False
    recent = steps[-3:]
    return all(norm(s['action']) == norm(action) for s in recent) and len(
        {norm(s['next_observation']) for s in recent}) == 1


def defect(action, prediction, backend, steps, metadata=True):
    p = visible(prediction, backend, metadata)
    if not p['available'] or not p['complete']:
        return None
    if backend == 'oracle' and p.get('done') is True and p.get('success') is False:
        return 'terminal_failure'
    if INVALID.search(p['observation']):
        return 'invalid_input'
    if loop_evidence(action, steps) and norm(p['observation']) == norm(steps[-1]['next_observation']):
        return 'loop'
    return None


def facts(steps):
    """Exact committed excerpts, labelled by step; never infer task completion."""
    entries = []
    for i, s in enumerate(steps):
        action, obs = s['action'], s['next_observation']
        if INVALID.search(obs):
            continue
        selected = []
        if re.match(r'^(use |read |connect |disconnect |focus on |move |pick up |pour |mix )', action):
            selected = obs.splitlines()[:2]
        elif re.search(r'temperature|degrees celsius|light bulb.*(?:on|off)', obs, re.I):
            selected = [line for line in obs.splitlines()
                        if re.search(r'temperature|degrees celsius|light bulb.*(?:on|off)', line, re.I)][:2]
        for line in selected:
            line = line.strip()
            if line:
                entries.append(dict(step=i+1, action=action[:160], observed_excerpt=line[:220],
                                    excerpt_truncated=len(line) > 220))
    # Pin early evidence and retain later updates. They remain historical, so
    # disconnected circuits or moved objects are never asserted still present.
    chosen = entries[:4] + entries[4:][-8:]
    return list({json.dumps(e, sort_keys=True): e for e in chosen}.values())


def history(steps, keep):
    start = len(steps)-keep
    return [dict(step=start+i+1, action=s['action'], observation=s['observation'],
                 next_observation=s['next_observation']) for i, s in enumerate(steps[start:])] if keep else []


def messages(controller, observation, steps, action, prediction, opt, instruction, extra=None):
    # Required goal/current state/forecast are never silently cut. On overflow
    # remove whole history pairs, then historical memory, then object hints.
    memory = facts(steps) if opt['memory'] else []
    for use_memory, use_hints in ((True, True), (False, True), (False, False)):
        for keep in range(min(5, len(steps)), -1, -1):
            data = dict(task=controller.task, current_observation=observation,
                original_action=action, original_forecast=visible(prediction, controller.wm.backend, opt['metadata']),
                executed_history=history(steps, keep),
                historical_evidence=memory if use_memory else [])
            if use_hints:
                data['action_templates_and_objects'] = getattr(controller, 'action_information', '')
            if extra:
                data.update(extra)
            result = [dict(role='user', content=RULES+'\n'+instruction+'\nINPUT:\n'+json.dumps(data, ensure_ascii=False))]
            if controller.agent.count(result) <= 4096:
                return result, keep, bool(memory and use_memory)
    return None, 0, False


def judge_critique(data, action, prediction, backend, steps, metadata):
    if set(data) != set(REVIEW_SCHEMA['required']) or data.get('decision') not in ('keep', 'replan') or data.get('issue') not in ISSUES:
        raise PredictionError('Malformed evidence review')
    if data['decision'] == 'keep':
        return None, None
    if data['issue'] == 'none' or data['evidence_source'] not in ('prediction', 'history'):
        return None, 'unsupported_or_inconsistent_review'
    source = json.dumps(visible(prediction, backend, metadata), ensure_ascii=False) if data['evidence_source']=='prediction' else json.dumps(history(steps, min(5,len(steps))), ensure_ascii=False)
    # Match both JSON and its text contents, allowing literal quotes/backslashes.
    source += '\n'+prediction.get('observation','') if data['evidence_source']=='prediction' else '\n'+'\n'.join(s['next_observation'] for s in steps[-5:])
    if not quoted(data['evidence'], source):
        return None, 'ungrounded_review_quote'
    known = defect(action, prediction, backend, steps, metadata)
    if data['issue'] == known:
        return known, None
    if data['issue']=='harmful_change' and data['evidence_source']=='prediction' and DESTRUCTIVE.search(data['evidence']):
        return 'harmful_change', None
    return None, 'unsupported_or_inconsistent_review'


def candidate_rejection(action, p, original, backend, steps, metadata):
    v = visible(p, backend, metadata)
    if not v['available'] or not v['complete']:
        return 'missing_or_partial_alternative'
    bad = defect(action, p, backend, steps, metadata)
    if bad:
        return 'alternative_'+bad
    if NOOP.search(v['observation']):
        return 'alternative_noop'
    if backend=='oracle' and metadata and not original.get('done') and p.get('score') is not None and original.get('score') is not None and p['score'] < original['score']:
        return 'alternative_score_loss'
    return None


def select(controller, original, observation, steps, actions, step_number):
    if controller.env != 'scienceworld':
        raise ValueError('Evidence protocols require ScienceWorld')
    opt = options(controller.review_protocol)
    reviews, predictions, cache, policy_events = [], [], {}, []
    chosen, attempts, blocked = original, 0, 0
    budget_skip, defect_kind, safe_alternative = False, None, None

    def forecast(action, number):
        if action not in cache:
            controller.recorder('predict_start', step_number, dict(round=number, candidate=action))
            cache[action] = dict(action=action, **controller.wm.predict(action))
            predictions.append(cache[action])
            controller.recorder('prediction', step_number, dict(round=number,candidate=action,prediction=cache[action]))
        return cache[action]

    p = forecast(original, 1)
    if success(p, controller.wm.backend, opt['metadata']):
        policy_events.append('oracle_success_preserved' if controller.wm.backend=='oracle' else 'learned_success_prediction_preserved')
    elif not p.get('prediction_available',True) or not p.get('observation_complete',True):
        policy_events.append('unknown_forecast_keep')
    else:
        defect_kind = defect(original,p,controller.wm.backend,steps,opt['metadata'])
        for number in range(1,controller.max_rounds+1):
            attempts = number
            r = dict(round=number,candidate=original,generation={},output='',messages=[],
                history_length=0,preview_clipped=False,error_kind=None,guard=None,
                decision=dict(decision='keep',action=original,reason='No supported defect'))
            reviews.append(r)
            try:
                if defect_kind is None:
                    ms,keep,mem=messages(controller,observation,steps,original,p,opt,
                        'Return only JSON: decision (keep/replan), issue (none/invalid_input/terminal_failure/harmful_change/loop), evidence_source (prediction/history/none), evidence (verbatim quote), reason.')
                    if ms is None:
                        budget_skip=True; policy_events.append('review_budget_keep'); break
                    output,generation=controller.agent.generate(ms,controller.task_id,step_number,response_schema=REVIEW_SCHEMA)
                    r.update(messages=ms,output=output,generation=generation,history_length=keep,memory_used=mem)
                    controller.recorder('review_output',step_number,dict(round=number,candidate=original,messages=ms,output=output,generation=generation))
                    data=extract_object(output); r['critique']=data
                    defect_kind,rejection=judge_critique(data,original,p,controller.wm.backend,steps,opt['metadata'])
                    if rejection:
                        r['guard']=rejection; blocked+=1
                    if defect_kind is None:
                        break
                else:
                    r['automated_evidence']=defect_kind
                extra=dict(verified_original_issue=defect_kind,
                    rejected_alternatives=[v.get('proposed_action') for v in reviews[:-1] if v.get('proposed_action')])
                ms,keep,mem=messages(controller,observation,steps,original,p,opt,
                    'Generate one replacement for the CURRENT REAL state that fixes the verified issue. Do not restart completed preparation. Use <think>brief reason</think> then a newline and <action>one action</action>.',extra)
                if ms is None:
                    budget_skip=True; policy_events.append('replan_budget_keep'); break
                output,generation=controller.agent.generate(ms,controller.task_id,step_number,response_regex=SCIENCE_REPLAN_REGEX)
                r['replan']=dict(messages=ms,output=output,generation=generation,history_length=keep,memory_used=mem)
                controller.recorder('replan_output',step_number,dict(round=number,candidate=original,**r['replan']))
                decision=parse_replan(output,original); alternative=decision['action'];r['proposed_action']=alternative
                if alternative==original or alternative in cache:
                    r['guard']='duplicate_alternative';blocked+=1;continue
                if not valid_replacement('scienceworld',alternative,actions) or alternative in ('reset task',) or alternative.startswith('teleport '):
                    r['error_kind']='invalid_action';raise PredictionError('Replacement fails current-state action syntax check')
                alt=forecast(alternative,number)
                rejection=candidate_rejection(alternative,alt,p,controller.wm.backend,steps,opt['metadata'])
                if rejection:
                    r['guard']=rejection;blocked+=1;continue
                safe_alternative=(alternative,decision,r)
                adopt=not opt['compare'] or success(alt,controller.wm.backend,opt['metadata'])
                if opt['compare'] and not adopt:
                    ms,keep,mem=messages(controller,observation,steps,original,p,opt,
                        'Compare both actions from the SAME current real state. Choose alternative only if it fixes the verified original issue, has direct supporting evidence, and introduces no new risk. Otherwise choose original or uncertain. Mere plausibility is insufficient. Return JSON: choice (original/alternative/uncertain), fixes_original_issue (boolean), introduces_risk (boolean), original_evidence (verbatim quote), alternative_evidence (verbatim quote), reason.',
                        dict(verified_original_issue=defect_kind,alternative_action=alternative,alternative_forecast=visible(alt,controller.wm.backend,opt['metadata'])))
                    if ms is None:
                        budget_skip=True;r['guard']='comparison_budget';blocked+=1;break
                    output,generation=controller.agent.generate(ms,controller.task_id,step_number,response_schema=PAIR_SCHEMA)
                    data=extract_object(output)
                    if set(data)!=set(PAIR_SCHEMA['required']):
                        raise PredictionError('Malformed pair comparison')
                    r['comparison']=dict(messages=ms,output=output,generation=generation,assessment=data)
                    controller.recorder('comparison_output',step_number,dict(round=number,candidate=original,**r['comparison']))
                    originals=json.dumps(visible(p,controller.wm.backend,opt['metadata']),ensure_ascii=False)+'\n'+p['observation']
                    alternatives=json.dumps(visible(alt,controller.wm.backend,opt['metadata']),ensure_ascii=False)+'\n'+alt['observation']
                    adopt=(data['choice']=='alternative' and data['fixes_original_issue'] is True and data['introduces_risk'] is False
                        and quoted(data['original_evidence'],originals) and quoted(data['alternative_evidence'],alternatives))
                    if not adopt:
                        r['guard']='comparison_not_supported';blocked+=1
                if adopt:
                    chosen=alternative;r['decision']=decision;break
                # Spend further rounds only on explicit defects, never solicit
                # additional votes until an otherwise harmless action loses.
                if defect_kind not in ('invalid_input','terminal_failure','loop'):
                    break
            except (PredictionError,ValueError,KeyError,TypeError) as error:
                r['error_kind']=r['error_kind'] or 'invalid_format'
                r['decision']=dict(decision='invalid',action=original,reason=str(error));blocked+=1
        if chosen==original and defect_kind=='terminal_failure':
            if not safe_alternative and original!='look around':
                # Never execute a known fatal action merely because all three
                # model replans were bad. This extra forecast is not a fourth
                # reflection round and is committed only after verification.
                fallback=forecast('look around',controller.max_rounds)
                if (fallback.get('prediction_available',True) and fallback.get('observation_complete',True)
                    and defect('look around',fallback,controller.wm.backend,steps,opt['metadata']) not in ('terminal_failure','invalid_input')):
                    r=reviews[-1]
                    safe_alternative=('look around',dict(decision='replace',action='look around',reason='Verified nonterminal observation fallback'),r)
                    r['fallback_action']='look around'
            if safe_alternative:
                chosen,decision,r=safe_alternative
                r['decision']=dict(decision,reason='Avoid verified terminal failure; '+decision['reason'])
                policy_events.append('safe_fallback_from_verified_terminal_failure')
    final=forecast(chosen,controller.max_rounds)
    if predictions[-1]['action']!=chosen:
        predictions.append(dict(final,cached=True,seconds=0,usage={}))
    return chosen,dict(backend=controller.wm.backend,review_protocol=controller.review_protocol,
        selection_policy=controller.review_protocol,initial_action=original,final_action=chosen,
        reviews=reviews,predictions=predictions,rounds=attempts,replacements=int(chosen!=original),
        invalid_reviews=sum(r['decision']['decision']=='invalid' for r in reviews),
        hit_round_limit=attempts==controller.max_rounds and chosen==original and bool(defect_kind),
        budget_skip=budget_skip,guard_blocked_replacements=blocked,original_defect=defect_kind,
        policy_events=policy_events,transition_metadata_visible=opt['metadata'],
        memory_evidence=facts(steps) if opt['memory'] else [])
