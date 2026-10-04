"""Use locally copied AgentWM templates, with a separate ScienceWorld replan.

No original-repository import or filesystem access is performed at runtime.
Only Oracle predictions provide simulator reward/done/success/score metadata.
"""
import json
import re

from worldmodel.backends import PredictionError
from .agentwm_alfworld_prompts import build_reflection_messages
from .agentwm_webshop_prompts import WEBSHOP_REFLECTION_TEMPLATE
from .agentwm_scienceworld_prompts import (
    SCIWORLD_REFLECTION_TEMPLATE_CRITIQUE_REPLAN_V1,
    SCIWORLD_GUARDED_REPLAN_TEMPLATE_V1,
)
from .native import parse_native_review

PROTOCOL = 'agentwm-copied-v1'
POLICY = 'agentwm-direct/science-critique-guarded-replan-v1'
REVIEW_SCHEMAS = {
    'alfworld': dict(type='object', additionalProperties=False,
        properties=dict(assessment=dict(type='string'), action=dict(type='string'),
                        change=dict(type='boolean'), reason=dict(type='string')),
        required=['assessment', 'action', 'change', 'reason']),
    'scienceworld': dict(type='object', additionalProperties=False,
        properties=dict(assessment=dict(type='string'), execute_proposed=dict(type='boolean'),
            needs_replan=dict(type='boolean'), suggested_action=dict(type='string'), reason=dict(type='string')),
        required=['assessment', 'execute_proposed', 'needs_replan', 'suggested_action', 'reason']),
}
SCIENCE_SCHEMA = ('activate OBJ; close OBJ; connect OBJ to OBJ; deactivate OBJ; '
    'disconnect OBJ; drop OBJ; dunk OBJ in OBJ; eat OBJ; flush OBJ; focus on OBJ; '
    'go OBJ; inventory; look around; look at OBJ; look in OBJ; mix OBJ; move OBJ to OBJ; '
    'open OBJ; pick up OBJ; pour OBJ in OBJ; put down OBJ; read OBJ; reset task; task; '
    'teleport OBJ; use OBJ on OBJ; wait; wait1')


def extract_object(output):
    # AgentWM also extracts JSON from explanatory text/code fences. Never infer
    # a JSON decision from a native action or invent a missing model response.
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    while position < len(output):
        start = output.find('{', position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(output[start:])
        except ValueError:
            position = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position = start + end
    if len(objects) != 1:
        raise PredictionError('AgentWM review requires one explicit JSON object')
    return objects[0]


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise PredictionError('AgentWM review requires a boolean decision')


def parse_direct(output, candidate):
    data = extract_object(output)
    if not {'assessment', 'action', 'change', 'reason'} <= set(data):
        raise PredictionError('AgentWM review requires assessment/action/change/reason')
    changed = boolean(data['change'])
    if not all(isinstance(data[k], str) for k in ('assessment', 'action', 'reason')):
        raise PredictionError('AgentWM action and assessment must be text')
    if not changed:
        # Matches AgentWM's reflect(): change=false always preserves proposal.
        action = candidate
    else:
        action = data['action'].strip()
        if not action:
            raise PredictionError('AgentWM replacement is empty')
    return dict(decision='replace' if changed and action != candidate else 'keep',
                action=action, reason=data['reason']), data


def parse_critique(output):
    data = extract_object(output)
    if not {'assessment', 'execute_proposed', 'needs_replan', 'suggested_action', 'reason'} <= set(data):
        raise PredictionError('ScienceWorld critique is missing decision fields')
    if not all(isinstance(data[k], str) for k in ('assessment', 'suggested_action', 'reason')):
        raise PredictionError('ScienceWorld critique text fields must be strings')
    data['execute_proposed'] = boolean(data['execute_proposed'])
    data['needs_replan'] = boolean(data['needs_replan']) and not data['execute_proposed']
    return data


def parse_replan(output, candidate):
    # The copied replan prompt requests XML. The EE ScienceWorld checkpoint is
    # trained on Thought/Action; accept either explicitly delimited single action.
    if '<action>' in output:
        matches = re.findall(r'<action>([^<>\r\n]+)</action>', output)
        if len(matches) != 1 or output.count('<action>') != 1 or output.count('</action>') != 1:
            raise PredictionError('ScienceWorld replan requires one action block')
        if output.strip().startswith(('{', '[', '```')) or 'Action:' in output:
            raise PredictionError('ScienceWorld replan contains conflicting formats')
        suffix = output.split('</action>', 1)[1].strip()
        if suffix:
            raise PredictionError('ScienceWorld replan has text after its final action')
        action = matches[0].strip()
        reason = output.split('<action>', 1)[0].replace('<think>', '').replace('</think>', '').strip()
        if not action:
            raise PredictionError('ScienceWorld replan action is empty')
        return dict(decision='keep' if action == candidate else 'replace', action=action, reason=reason)
    return parse_native_review(output, candidate, 'scienceworld')


def feedback(prediction, backend):
    if backend == 'oracle':
        return {name: prediction.get(name) for name in ('reward', 'done', 'success', 'score')}
    # [SUCCESS] is a learned-model output, never a real-environment lookup.
    complete = prediction.get('prediction_available', True) and prediction.get('observation_complete', True)
    return dict(reward=None, done=None, score=None,
                success=prediction.get('predicted_success') if complete else None)


def source_note(prediction, backend):
    note = ''
    if backend == 'word2world':
        note = ('World-model source: learned Word2World checkpoint. In the copied template, '
                '"oracle" is a legacy label; this forecast can be inaccurate. '
                'None means UNKNOWN metadata, not zero, false, or failure.\n')
    if not prediction.get('prediction_available', True):
        note += 'The next observation is UNKNOWN; no usable forecast was returned.\n'
    elif not prediction.get('observation_complete', True):
        note += 'The forecast is PARTIAL; omitted outcomes are unknown.\n'
    return note


def build_messages(controller, candidate, prediction, observation, steps, actions, keep, reviews, critique=None):
    recent = steps[-keep:] if keep else []
    meta = feedback(prediction, controller.wm.backend)
    preview = prediction['observation'] if prediction.get('prediction_available', True) else 'UNKNOWN'
    if controller.env == 'alfworld':
        messages = build_reflection_messages(controller.task, observation,
            [(s['action'], s['next_observation']) for s in recent], candidate, preview,
            meta['reward'], meta['done'], meta['success'], actions)
    else:
        history = '\n'.join('Step %d Observation: %s\nStep %d Action: %s' %
            (i + 1, s['observation'], i + 1, s['action']) for i, s in enumerate(recent)) or '(empty)'
        values = dict(task_description=controller.task, current_observation=observation,
            proposed_action=candidate, predicted_next_observation=preview,
            predicted_reward=meta['reward'], predicted_done=meta['done'],
            predicted_success=meta['success'], predicted_score=meta['score'])
        if controller.env == 'webshop':
            values.update(recent_history=history, available_actions='\n'.join(repr(a) + ',' for a in actions))
            text = WEBSHOP_REFLECTION_TEMPLATE.format(**values)
        else:
            values.update(action_history=history, available_actions=(
                'ScienceWorld action templates: ' + SCIENCE_SCHEMA + '\n'
                'Current available grounded actions and ambiguity choices:\n' + '\n'.join(actions)))
            if critique is None:
                text = SCIWORLD_REFLECTION_TEMPLATE_CRITIQUE_REPLAN_V1.format(**values)
            else:
                values.update(reflection_assessment=critique['assessment'],
                    reflection_reason=critique['reason'], reflection_suggested_action=critique['suggested_action'])
                text = SCIWORLD_GUARDED_REPLAN_TEMPLATE_V1.format(**values)
        messages = [dict(role='user', content=text)]
    # Templates above are unmodified local copies. These transport notes make
    # incomplete/learned feedback honest without rewriting the copied rules.
    messages[-1]['content'] = source_note(prediction, controller.wm.backend) + messages[-1]['content']
    failures = [r for r in reviews if r['decision']['decision'] == 'invalid']
    if failures:
        messages[-1]['content'] += '\nPrevious rejected review/replan: ' + failures[-1]['decision']['reason']
    return messages


def bounded_messages(controller, candidate, prediction, observation, steps, actions, reviews, critique=None):
    # AgentWM uses the recent five steps. Remove whole older pairs first.
    for keep in range(min(5, len(steps)), -1, -1):
        messages = build_messages(controller, candidate, prediction, observation, steps, actions, keep, reviews, critique)
        if controller.agent.count(messages) <= 4096:
            return messages, keep, False
    clipped = dict(prediction, observation='', observation_complete=False)
    required = build_messages(controller, candidate, clipped, observation, steps, actions, 0, reviews, critique)
    if controller.agent.count(required) > 4096:
        return None, 0, False
    low, high = 0, len(prediction['observation'])
    while low < high:
        middle = (low + high + 1) // 2
        clipped['observation'] = prediction['observation'][:middle]
        messages = build_messages(controller, candidate, clipped, observation, steps, actions, 0, reviews, critique)
        if controller.agent.count(messages) <= 4096:
            low = middle
        else:
            high = middle - 1
    clipped['observation'] = prediction['observation'][:low]
    return build_messages(controller, candidate, clipped, observation, steps, actions, 0, reviews, critique), 0, True


def reject_replan(action, prediction, backend):
    if not prediction.get('prediction_available', True) or not prediction.get('observation_complete', True):
        return 'replan_prediction_missing_or_partial'
    text = prediction['observation'].lower()
    if any(s in text for s in ('no known action matches', 'its not clear how to', "it's not clear how to",
                              'not clear how to go', 'not sure how to', "don't know how to", 'unknown action', 'invalid action')):
        return 'predicted_invalid_action'
    if backend == 'oracle' and action.lower().startswith('focus on') and prediction.get('done') and not prediction.get('success') and prediction.get('score', 0) <= -100:
        return 'predicted_terminal_bad_focus'
    return None


def select(controller, original, observation, steps, actions, step_number):
    from .controller import valid_replacement
    reviews, predictions, cached = [], [], {}
    candidate, budget_skip, replacements = original, False, 0
    verified = set()

    def forecast(action, number):
        if action not in cached:
            controller.recorder('predict_start', step_number, dict(round=number, candidate=action))
            value = dict(action=action, **controller.wm.predict(action))
            cached[action] = value
            predictions.append(value)
            controller.recorder('prediction', step_number, dict(round=number, candidate=action, prediction=value))
        return cached[action]

    for number in range(1, controller.max_rounds + 1):
        prediction = forecast(candidate, number)
        verified.add(candidate)
        messages, keep, clipped = bounded_messages(controller, candidate, prediction, observation, steps, actions, reviews)
        if messages is None:
            budget_skip = True
            break
        output, generation = controller.agent.generate(messages, controller.task_id, step_number,
            response_schema=REVIEW_SCHEMAS.get(controller.env))
        controller.recorder('review_output', step_number, dict(round=number, candidate=candidate,
            messages=messages, output=output, generation=generation))
        record = dict(round=number, candidate=candidate, messages=messages, output=output,
                      generation=generation, history_length=keep, preview_clipped=clipped,
                      error_kind=None, guard=None)
        stop = False
        try:
            if controller.env == 'scienceworld':
                critique = parse_critique(output)
                record['critique'] = critique
                decision = dict(decision='keep', action=candidate, reason=critique['reason'])
                if not critique['needs_replan']:
                    stop = True
                else:
                    replan_messages, replan_keep, replan_clipped = bounded_messages(
                        controller, candidate, prediction, observation, steps, actions, reviews, critique)
                    if replan_messages is None:
                        record['guard'] = 'replan_budget_skip'
                        budget_skip, stop = True, True
                    else:
                        replan_output, replan_generation = controller.agent.generate(replan_messages, controller.task_id, step_number)
                        record['replan'] = dict(messages=replan_messages, output=replan_output,
                            generation=replan_generation, history_length=replan_keep, preview_clipped=replan_clipped)
                        controller.recorder('replan_output', step_number, dict(round=number,
                            candidate=candidate, **record['replan']))
                        decision = parse_replan(replan_output, candidate)
                        if decision['decision'] == 'keep':
                            stop = True
            elif controller.env == 'alfworld':
                decision, assessment = parse_direct(output, candidate)
                record['assessment'] = assessment
            else:
                decision = parse_native_review(output, candidate, 'webshop')
            if decision['decision'] == 'replace':
                record['proposed_action'] = decision['action']
                if not valid_replacement(controller.env, decision['action'], actions):
                    record['error_kind'] = 'invalid_action'
                    raise PredictionError('Replacement is not executable in the actual current state')
                if decision['action'] in verified:
                    record['guard'], stop = 'oscillation_detected', True
                    decision = dict(decision='keep', action=candidate, reason=record['guard'])
                elif controller.env == 'scienceworld':
                    alternative = forecast(decision['action'], number)
                    rejection = reject_replan(decision['action'], alternative, controller.wm.backend)
                    if rejection:
                        record['guard'] = rejection
                        raise PredictionError(rejection)
                    replacements += 1
                    candidate, stop = decision['action'], True
                    # Preserve EE ScienceWorld's native committed history while
                    # retaining the original XML completion in replan.output.
                    record['execution_output'] = 'Thought: ' + decision['reason'] + '\nAction: ' + candidate
                else:
                    replacements += 1
                    candidate = decision['action']
            else:
                stop = True
        except PredictionError as error:
            record['error_kind'] = record['error_kind'] or ('invalid_prediction' if record.get('guard') else 'invalid_format')
            decision = dict(decision='invalid', action=candidate, reason=str(error))
        record['decision'] = decision
        reviews.append(record)
        if stop:
            break
    final = forecast(candidate, controller.max_rounds)
    if predictions[-1]['action'] != candidate:
        predictions.append(dict(final, cached=True, seconds=0, usage={}))
    return candidate, dict(backend=controller.wm.backend, review_protocol=PROTOCOL,
        selection_policy=POLICY, initial_action=original, final_action=candidate,
        reviews=reviews, predictions=predictions, rounds=len(reviews), budget_skip=budget_skip,
        replacements=replacements, invalid_reviews=sum(r['decision']['decision']=='invalid' for r in reviews),
        hit_round_limit=len(reviews)==controller.max_rounds and not stop)
