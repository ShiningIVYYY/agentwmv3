"""Two explicitly different Actor formats; neither receives expert actions."""
import re
from internal_signals.decision import messages as structured_messages


def native_messages(goal,observation,history,source_prompt):
    recent=history[-2:]
    result=[dict(role='user',content=source_prompt['instruction']),dict(role='assistant',content=source_prompt['acknowledgement'])]
    result.append(dict(role='user',content=goal+'\n'+(recent[0]['observation'] if recent else observation)))
    for i,step in enumerate(recent):
        # Only actually executed actions, no fabricated thoughts or expert rollout.
        result.append(dict(role='assistant',content='Action:\n'+step['action']))
        result.append(dict(role='user',content=recent[i+1]['observation'] if i+1<len(recent) else observation))
    return result


def prompt(mode,goal,observation,history,source_prompt):
    if mode=='structured':return structured_messages(goal,observation,history,keep=2)
    if mode=='native_h2':return native_messages(goal,observation,history,source_prompt)
    raise ValueError('Unknown prompt mode')


def parse_action(text):
    matches=list(re.finditer(r'(?m)^Action:[ \t]*(?:\n)?',text))
    if len(matches)!=1:return dict(action=None,error='missing_or_multiple_action_fields')
    action=text[matches[0].end():].strip()
    if not action or '\n' in action or '\r' in action:return dict(action=None,error='empty_or_multiline_action')
    return dict(action=action,error=None)
