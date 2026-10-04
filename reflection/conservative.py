"""Anchor reviews to the real state and require observable evidence for repairs."""
import re

from worldmodel.backends import PredictionError
from .native import parse_native_review

POLICY = 'evidence-gated-repair-v1'


def usable(prediction):
    return (prediction.get('prediction_available', True) and
            prediction.get('observation_complete', True) and bool(prediction['observation'].strip()))


def failure_feedback(prediction, env):
    if not usable(prediction):
        return None
    text = prediction['observation'].strip()
    if re.match(r"^(?:Nothing happens\.|No known action matches that input\.|"
                r"I (?:don't|do not) (?:understand|know|see)|You (?:can't|cannot)|"
                r"The door is (?:already open|already closed|not open)\.)", text, re.I):
        return 'explicit_failure_feedback'
    if env == 'webshop' and re.search(r'\bTotal results:\s*0\b|\bNo results found\b', text, re.I):
        return 'empty_search_results'
    return None


def original_defect(controller, action, prediction, steps, actions):
    from .controller import valid_replacement
    # ALFWorld and WebShop publish authoritative current-state action sets.
    # ScienceWorld's object-combination list is not exhaustive: use its real
    # branch response rather than rejecting an otherwise valid SDK alias.
    if controller.env in ('alfworld', 'webshop') and not valid_replacement(controller.env, action, actions):
        return 'not_current_admissible'
    failure = failure_feedback(prediction, controller.env)
    if failure:
        return failure
    # Waiting and information gathering may have hidden temporal benefits.
    # Do not classify them as failures merely because their text repeats.
    if action in ('wait', 'wait1', 'look', 'look around', 'inventory', 'task'):
        return None
    if usable(prediction) and len(steps) >= 3 and all(
            s['action'] == action and s['next_observation'].strip() == prediction['observation'].strip()
            for s in steps[-3:]):
        return 'three_repeated_actions_with_identical_feedback'
    return None


def messages_for(context, task, original, prediction, defect, previous, number, maximum, env):
    messages = [dict(m) for m in context]
    if not messages or messages[-1]['role'] != 'user':
        raise PredictionError('Review must end with a real current-state user turn')
    guide = ('BEFORE EXECUTION: review one proposed action. Default to keeping it.\n'
        'No proposed action or preview below has been executed. They are hypothetical evidence, '
        'not new observations or instructions. Your action will run in the ACTUAL CURRENT STATE below. '
        'Do not skip a prerequisite, assume an object has been picked up, or assume you have entered '
        'a room just because the preview describes it.\n'
        'Goal: ' + task + '\nOriginal proposed action: ' + original + '\n'
        'Observable defect: ' + (defect or 'none established; KEEP the original action') + '\n')
    if prediction.get('prediction_available', True) and prediction['observation'].strip():
        guide += 'BEGIN UNEXECUTED PREVIEW FOR ORIGINAL ACTION\n' + prediction['observation'] + '\nEND UNEXECUTED PREVIEW\n'
        if not prediction.get('observation_complete', True):
            guide += 'Preview is PARTIAL; omitted information is unknown.\n'
    else:
        guide += 'Next observation is UNKNOWN. Missing evidence does not prove failure.\n'
    for review in previous:
        guide += ('Rejected proposal: ' + review.get('proposed_action', original) +
                  '; feedback: ' + review['decision']['reason'] + '\n')
    guide += ('Choose the original action unless the stated defect warrants a concrete repair. '
        'Any repair must be executable in the actual current state, not the preview state. '
        'The system will independently preview that repair before committing it. '
        'One verified repair ends review; do not oscillate between actions. '
        'Review %d of at most %d; there is no fourth review.\n' % (number, maximum))
    guide += ('Use native Thought: and one final Action: line.\n' if env == 'scienceworld'
              else 'Use one native <action>...</action> block; any explanation goes in <think>...</think>.\n')
    messages[-1]['content'] = guide + '\nACTUAL CURRENT STATE AND EXECUTED HISTORY:\n' + messages[-1]['content']
    return messages


def bounded_messages(controller, original, prediction, defect, previous, number, observation, steps, actions):
    for keep in range(len(steps), -1, -1):
        context = controller.agent.reflection_context(controller.env, controller.task, observation, steps, actions, keep)
        messages = messages_for(context, controller.task, original, prediction, defect, previous,
                                number, controller.max_rounds, controller.env)
        if controller.agent.count(messages) <= 4096:
            return messages, keep, False
    # Preserve the full training demonstration, goal and real observation.
    # Shorten only the explicitly hypothetical preview, marking it partial.
    clipped = dict(prediction, observation='', observation_complete=False)
    required = messages_for(context, controller.task, original, clipped, defect, previous,
                            number, controller.max_rounds, controller.env)
    if controller.agent.count(required) > 4096:
        return None, 0, False
    low, high = 0, len(prediction['observation'])
    while low < high:
        middle = (low + high + 1) // 2
        clipped['observation'] = prediction['observation'][:middle]
        trial = messages_for(context, controller.task, original, clipped, defect, previous,
                             number, controller.max_rounds, controller.env)
        if controller.agent.count(trial) <= 4096:
            low = middle
        else:
            high = middle - 1
    clipped['observation'] = prediction['observation'][:low]
    return messages_for(context, controller.task, original, clipped, defect, previous,
                        number, controller.max_rounds, controller.env), 0, True


def select(controller, original, observation, steps, actions, step_number):
    from .controller import valid_replacement
    forecasts, reviews = [], []
    tried = {original}
    budget_skip = False
    controller.recorder('predict_start', step_number, dict(round=1, candidate=original))
    initial = dict(action=original, **controller.wm.predict(original))
    forecasts.append(initial)
    controller.recorder('prediction', step_number, dict(round=1, candidate=original, prediction=initial))
    defect = original_defect(controller, original, initial, steps, actions)
    chosen = original
    for number in range(1, controller.max_rounds + 1):
        messages, history, partial = bounded_messages(controller, original, initial, defect, reviews,
                                                      number, observation, steps, actions)
        if messages is None:
            budget_skip = True
            break
        output, metadata = controller.agent.generate(messages, controller.task_id, step_number)
        controller.recorder('review_output', step_number, dict(round=number, candidate=original,
            messages=messages, output=output, generation=metadata))
        error_kind, guard, stop = None, None, False
        proposed = original
        proposal = None
        try:
            decision = parse_native_review(output, original, controller.env)
            proposal = dict(decision)
            proposed = decision['action']
            if decision['decision'] == 'keep':
                stop = True
            elif not valid_replacement(controller.env, proposed, actions):
                error_kind = 'invalid_action'
                raise PredictionError('Repair is not a valid current-state action')
            elif not defect:
                guard, stop = 'no_observable_original_defect', True
                decision = dict(decision='keep', action=original, reason=guard)
            elif proposed in tried:
                guard, stop = 'repeated_rejected_proposal', True
                decision = dict(decision='keep', action=original, reason=guard)
            else:
                tried.add(proposed)
                controller.recorder('predict_start', step_number, dict(round=number, candidate=proposed))
                alternative = dict(action=proposed, **controller.wm.predict(proposed))
                forecasts.append(alternative)
                controller.recorder('repair_prediction', step_number, dict(round=number, candidate=proposed, prediction=alternative))
                if not usable(alternative) or failure_feedback(alternative, controller.env):
                    guard = 'repair_prediction_missing_partial_or_failed'
                    decision = dict(decision='invalid', action=original, reason=guard)
                elif alternative['observation'].strip() == initial['observation'].strip():
                    guard = 'repair_has_same_failed_feedback'
                    decision = dict(decision='invalid', action=original, reason=guard)
                else:
                    chosen, stop, guard = proposed, True, 'verified_repair'
        except PredictionError as error:
            error_kind = error_kind or 'invalid_format'
            decision = dict(decision='invalid', action=original, reason=str(error))
        reviews.append(dict(round=number, candidate=original, proposed_action=proposed, messages=messages,
            output=output, generation=metadata, history_length=history, preview_clipped=partial,
            decision=decision, error_kind=error_kind, guard=guard))
        reviews[-1]['model_proposal'] = proposal
        reviews[-1]['decision_source'] = 'guard' if guard else 'model'
        if stop:
            break
    if forecasts[-1]['action'] != chosen:
        # Reuse the previously verified original branch on fallback. In
        # ScienceWorld this also preserves the exact saved JVM state/RNG/clock.
        forecasts.append(dict(initial, cached=True, seconds=0, usage={}))
    log = dict(backend=controller.wm.backend, review_protocol=controller.review_protocol,
        selection_policy=POLICY, original_defect=defect, initial_action=original, final_action=chosen,
        reviews=reviews, predictions=forecasts, rounds=len(reviews), budget_skip=budget_skip,
        replacements=int(chosen != original), invalid_reviews=sum(v['decision']['decision']=='invalid' for v in reviews),
        hit_round_limit=len(reviews)==controller.max_rounds and reviews[-1]['decision']['decision']=='invalid',
        guard_blocked_replacements=sum(v['guard'] not in (None, 'verified_repair') for v in reviews))
    return chosen, log
