"""Versioned short decisions, explicit episode aborts and public-action review guard."""
import hashlib,json
from internal_signals.alfworld_prompts import GUIDANCE
from reflection.replay_review import INSTRUCTIONS,parse_review
from internal_signals.token_nll import FORMAT

SCHEMA='activation-pce-v2'
FORMAT_FAILURE='abort_current_episode_no_retry_no_invented_action'
REVIEW_GUARD='keep_draft_on_invalid_or_nonadmissible_replacement_v2'


def draft_messages(goal,observation,history,commands):
    past='\n\n'.join('Observation: '+r['observation']+'\nAction: '+r['action'] for r in history[-2:])
    return [dict(role='user',content=GUIDANCE+
        'Return exactly THREE lines. No analysis, debate, markdown, or extra fields. Premise and Expected Consequence each use one short clause, at most20 words.\n'
        'Premise: <current observable condition>\nExpected Consequence: <predicted immediate result>\nAction: <one exact ALFWorld command>\n'
        +'Task: '+goal+'\nRecent executed history:\n'+(past or '(none)')+'\nCurrent observation:\n'+observation+
        '\nCURRENT admissible commands:\n'+'\n'.join(commands))]


def parse_draft(text,truncated=False):
    m=FORMAT.fullmatch(text)
    if not m or truncated:return dict(valid=False,error='actor_format_failure',action=None)
    # Read exact named lines; never recover an action from explanatory text.
    ls=text.rstrip('\n').splitlines()
    if len(ls)!=3:return dict(valid=False,error='actor_format_failure',action=None)
    action=ls[2].split(':',1)[1].strip()
    if not action:return dict(valid=False,error='actor_format_failure',action=None)
    return dict(valid=True,action=action,premise=ls[0].split(':',1)[1].strip(),consequence=ls[1].split(':',1)[1].strip())


def review_messages(goal,obs,history,commands,draft,simulation):
    if not simulation.get('prediction_available'):raise ValueError('Unverified trial cannot reach review')
    public={k:simulation[k] for k in ['predicted_observation','predicted_reward','predicted_done','predicted_score']}
    return [dict(role='user',content=GUIDANCE+INSTRUCTIONS+
        '\nReason must be ONE short clause, at most20 words. Output only the three required lines. '
        'Nonterminal zero reward is normal and does not by itself imply action failure. '
        'A replacement must be an exact CURRENT admissible command; do not invent object names.\n'
        +'Goal: '+goal+'\nCurrent Observation: '+obs+'\nPast committed history: '+json.dumps(history[-2:])+
        '\nCURRENT admissible commands: '+json.dumps(commands)+'\nDraft Action: '+draft+
        '\nGround-Truth Simulated Feedback (not committed): '+json.dumps(public))]


def parse_review_guarded(text,draft,commands):
    result=parse_review(text,draft)
    if result['decision']=='REPLACE' and result['final_action'] not in commands:
        return dict(decision='REJECTED',proposed_action=result['final_action'],final_action=draft,error='nonadmissible_replacement',fallback='keep_draft')
    return result


def protocol_hash():
    # Actual source hash guards instructions as well as version labels.
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
