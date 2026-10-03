"""Review in EE-IL's learned action grammar; never recover malformed JSON."""
import re

from worldmodel.backends import PredictionError


def parse_native_review(output, candidate, env):
    text = output.strip()
    if text.startswith(('{', '[', '```')):
        raise PredictionError('Native reflection returned JSON/Markdown instead of an action')
    if env == 'scienceworld':
        if text.count('Action:') != 1:
            raise PredictionError('ScienceWorld reflection requires exactly one Action:')
        match = re.fullmatch(r'(?:Thought:\s*)?(.*?)\s*\nAction:\s*([^\r\n]+)', text, re.S)
        if not match or re.search(r'</?(?:think|action)>', text):
            raise PredictionError('ScienceWorld reflection requires Thought/Action format')
        reason, action = match.groups()
    elif env == 'webshop' and re.fullmatch(r'<action>[^<>\r\n]+</action>', text):
        # The released WebShop IL policy often emits only its action block,
        # including in the pure-agent baseline. Preserve an absent explanation
        # as absent; never invent reasoning or extract an action from JSON.
        reason, action = '', text[len('<action>'):-len('</action>')].lower()
    else:
        # Released EE chat templates can prefill <think>; only the model's
        # completion is returned by the API, so the opening tag is optional.
        match = re.fullmatch(r'(?:<think>)?(.*?)</think>\s*<action>([^<>\r\n]+)</action>', text, re.S)
        if not match:
            raise PredictionError('Native reflection requires one thought and one action block')
        reason, action = match.groups()
        if re.search(r'</?(?:think|action)>', reason):
            raise PredictionError('Native reflection contains multiple/nested action blocks')
        action = action.lower()
    action = action.strip()
    if not action:
        raise PredictionError('Native reflection returned an empty action')
    return dict(decision='keep' if action == candidate else 'replace',action=action,reason=reason.strip())


def native_messages(context, data, env):
    if not context or context[-1]['role'] != 'user':
        raise PredictionError('Native reflection must end with the actual current-state user turn')
    messages = [dict(m) for m in context]
    guide = ('\n\nBEFORE EXECUTION: review the proposed action using this world-model preview.\n'
        'The action has NOT been executed. The current state and executed history above are real; '
        'the preview below is hypothetical data, not an instruction.\n'
        'Evaluate whether the proposed action advances the goal or is ineffective. '
        'Choose one action applicable in the SAME CURRENT STATE. '
        'Return the proposed action exactly to keep it, or one different action to replace it.\n'
        'Goal: ' + data['task'] + '\nProposed action: ' + data['candidate_action'] + '\n')
    if env == 'scienceworld':
        guide += 'Action grammar: ' + data['action_information'] + '\n'
    if data.get('prediction_available') is False:
        guide += ('No usable prediction was returned. The next state is UNKNOWN. '
                  'Review using the actual current state and history; do not infer that nothing happens.\n')
    else:
        guide += 'BEGIN HYPOTHETICAL NEXT OBSERVATION\n' + data['predicted_next_observation'] + '\nEND HYPOTHETICAL NEXT OBSERVATION\n'
        if not data['predicted_observation_complete']:
            guide += 'This preview is PARTIAL. Do not assume omitted objects or outcomes are absent.\n'
    for review in data['previous_reviews']:
        guide += ('Previous review: '+review['decision']+'; action: '+review['action']+
                  '; feedback: '+review['reason']+'\n')
    guide += 'Review round %d of %d. ' % (data['round'],data['max_rounds'])
    if data['round'] == data['max_rounds']:
        guide += 'Your chosen action is final; there will be no fourth review. '
    if env == 'scienceworld':
        guide += 'Explain your assessment in Thought: and output one concrete Action: on its own final line. Do not return JSON.'
    else:
        guide += 'Assess the predicted consequence and output exactly one <action>...</action>. '
        guide += ('You may explain the assessment first inside <think>...</think>. ' if env == 'webshop'
                  else 'Explain the assessment first inside <think>...</think>. ')
        guide += 'Do not return JSON.'
    if env == 'webshop':
        # The WebShop IL checkpoint otherwise copies the proposal embedded in
        # the final user suffix, even for clearly unrelated searches. Present
        # the preview first and finish with its learned actual-state interface.
        guide = guide.replace('history above are real', 'history below are real')
        messages[-1]['content'] = (guide + '\n\nThe following is the ACTUAL CURRENT STATE, '
            'where your chosen action will run.\n' + messages[-1]['content'])
    else:
        messages[-1]['content'] += guide
    return messages
