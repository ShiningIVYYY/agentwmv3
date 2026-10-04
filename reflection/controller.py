import json
from pathlib import Path
import re

from worldmodel.backends import PredictionError
from .native import native_messages, parse_native_review

PROMPT = (Path(__file__).with_name('prompt.txt')).read_text()


def parse_review(output, candidate):
    try:
        review = json.loads(output)
    except (ValueError, TypeError) as e:
        raise PredictionError('Reflection must return one JSON object: ' + str(e))
    if not isinstance(review, dict) or set(review) != {'decision', 'action', 'reason'}:
        raise PredictionError('Reflection requires exactly decision, action, reason')
    if review['decision'] not in ('keep', 'replace'):
        raise PredictionError('Invalid reflection decision')
    if not isinstance(review['action'], str) or not isinstance(review['reason'], str):
        raise PredictionError('Reflection action/reason must be strings')
    if review['decision'] == 'keep' and review['action'] != candidate:
        raise PredictionError('Keep decision changed the candidate action')
    if review['decision'] == 'replace' and (not review['action'].strip() or review['action'] == candidate):
        raise PredictionError('Replace decision requires a different concrete action')
    return review


def valid_replacement(env, action, actions):
    if action != action.strip() or '\n' in action or re.search(r'</?(?:action|think)>', action):
        return False
    if env == 'webshop':
        return action.lower() in {a.lower() for a in actions if '<your query>' not in a} or bool(
            any(a.startswith('search[') for a in actions) and re.fullmatch(r'search\[[^\[\]]+\]', action))
    if env == 'alfworld':
        return action in actions
    if action in actions:
        return True
    if action.isdigit():
        return False
    # The SDK list is not an exhaustive grammar: retain the published templates,
    # including connect/disconnect, without a new whitelist of object combinations.
    return bool(re.fullmatch(r'(?:activate|close|connect|deactivate|disconnect|drop|dunk|eat|flush|focus on|go|'
                             r'look at|look in|mix|move|open|pick up|pour|put down|read|teleport|use) .+|'
                             r'inventory|look around|reset task|task|wait|wait1', action))


class ReflectionController:
    def __init__(self, wm, agent, env, task, task_id, max_rounds=3, recorder=None, review_protocol='json'):
        if max_rounds not in (1, 2, 3):
            raise ValueError('Reflection rounds must be between 1 and 3')
        self.wm, self.agent, self.env = wm, agent, env
        self.task, self.task_id, self.max_rounds = task, task_id, max_rounds
        self.recorder = recorder or (lambda *args: None)
        if review_protocol not in ('json', 'ee-native-action-v1', 'ee-conservative-v2', 'agentwm-copied-v1'):
            raise ValueError('Unknown reflection review protocol')
        self.review_protocol = review_protocol

    def select(self, candidate, observation, steps, actions, step_number):
        if self.review_protocol == 'agentwm-copied-v1':
            from .agentwm import select
            return select(self, candidate, observation, steps, actions, step_number)
        if self.review_protocol == 'ee-conservative-v2':
            from .conservative import select
            return select(self, candidate, observation, steps, actions, step_number)
        initial = candidate
        reviews = []
        predictions = []
        for number in range(1, self.max_rounds + 1):
            self.recorder('predict_start', step_number, dict(round=number, candidate=candidate))
            prediction = self.wm.predict(candidate)
            self.recorder('prediction', step_number, dict(round=number, candidate=candidate, prediction=prediction))
            predictions.append(dict(action=candidate, **prediction))
            # Only the common observation field is visible to the reviewer.
            # Neither privileged oracle rewards nor hidden state enter the prompt.
            for keep in range(len(steps), -1, -1):
                data = dict(task=self.task, current_observation=observation,
                    executed_history=[dict(observation=s['observation'], action=s['action'],
                                           next_observation=s['next_observation']) for s in steps[-keep:]] if keep else [],
                    action_information=actions if self.env != 'scienceworld' else (
                        'ScienceWorld action templates: activate OBJ; close OBJ; connect OBJ to OBJ; '
                        'deactivate OBJ; disconnect OBJ; dunk OBJ in OBJ; eat OBJ; flush OBJ; focus on OBJ; '
                        'go OBJ; inventory; look around; look at OBJ; look in OBJ; mix OBJ; move OBJ to OBJ; '
                        'open OBJ; pick up OBJ; pour OBJ in OBJ; put down OBJ; read OBJ; reset task; task; '
                        'teleport OBJ; use OBJ on OBJ; wait; wait1. If the CURRENT observation asks '
                        'for a numbered ambiguity choice, its listed numbers are also valid actions.'),
                    candidate_action=candidate, predicted_next_observation=prediction['observation'],
                    predicted_observation_complete=prediction.get('observation_complete', True),
                    previous_reviews=[dict(candidate=r['candidate'], **r['decision']) for r in reviews],
                    round=number, max_rounds=self.max_rounds)
                prompt = PROMPT
                if prediction.get('prediction_available') is False:
                    data['prediction_available'] = False
                    prompt = prompt.replace('Use the hypothetical next observation as evidence.',
                        'Use the actual current observation and executed history as evidence.')
                    prompt = prompt.replace(
                        'If predicted_observation_complete is false, the prediction hit its token limit.\n'
                        'Treat it as partial evidence; do not assume omitted objects or outcomes are absent.',
                        'The world model returned no usable next observation for this candidate.\n'
                        'The next state is UNKNOWN. Do not infer that nothing happens.\n'
                        'Use the actual current state, executed history and action information to review it.')
                if self.review_protocol == 'ee-native-action-v1':
                    context = self.agent.reflection_context(self.env,self.task,observation,steps,actions,keep)
                    messages = native_messages(context,data,self.env)
                else:
                    messages = [dict(role='user', content=prompt + '\n\nINPUT DATA:\n' +
                                     json.dumps(data, ensure_ascii=False))]
                if self.agent.count(messages) <= 4096:
                    break
            else:
                raise PredictionError('Required reflection fields exceed the Agent 4096-token budget')
            output, metadata = self.agent.generate(messages, self.task_id, step_number)
            self.recorder('review_output', step_number, dict(round=number, candidate=candidate,
                messages=messages, output=output, generation=metadata))
            error_kind = 'invalid_format'
            try:
                decision = parse_native_review(output,candidate,self.env) if self.review_protocol == 'ee-native-action-v1' else parse_review(output, candidate)
                error_kind = 'invalid_action'
                if decision['decision'] == 'replace' and not valid_replacement(self.env, decision['action'], actions):
                    raise PredictionError('Reflection proposed an invalid replacement action')
            except PredictionError as error:
                # Invalid MODEL decisions consume a round, not an infrastructure
                # retry or a fabricated action. Keep the current candidate and
                # show the validation error on the next bounded review.
                decision = dict(decision='invalid', action=candidate, reason=str(error))
            reviews.append(dict(round=number, candidate=candidate, messages=messages, output=output,
                                decision=decision, history_length=keep, generation=metadata,
                                error_kind=error_kind if decision['decision']=='invalid' else None))
            if decision['decision'] == 'keep':
                break
            if decision['decision'] == 'invalid':
                continue
            candidate = decision['action']
        if predictions[-1]['action'] != candidate:
            # A replacement on round three is predicted once, but no fourth Agent review.
            predictions.append(dict(action=candidate, **self.wm.predict(candidate)))
            self.recorder('final_prediction', step_number, predictions[-1])
        return candidate, dict(backend=self.wm.backend, review_protocol=self.review_protocol, initial_action=initial, final_action=candidate,
            reviews=reviews, predictions=predictions, rounds=len(reviews),
            replacements=sum(r['decision']['decision'] == 'replace' for r in reviews),
            invalid_reviews=sum(r['decision']['decision'] == 'invalid' for r in reviews),
            hit_round_limit=len(reviews) == self.max_rounds and reviews[-1]['decision']['decision'] != 'keep')

    def commit(self, action, actual, log):
        self.wm.commit(action, actual)
        prediction = log['predictions'][-1]
        log['actual'] = actual
        log['prediction_exact_match'] = prediction['observation'].strip() == actual['observation'].strip()
        return log
