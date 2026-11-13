"""Same decision/review schemas, with ALFWorld-native commands and current public list."""
import json
from internal_signals.decision import STRUCTURE
from reflection.replay_review import INSTRUCTIONS,parse_review

GUIDANCE='''You are an agent in ALFWorld. Complete the task using only current observations and already executed history.
Choose one exact native command from the CURRENT admissible commands. Commands include go to RECEPTACLE, open/close CONTAINER, take OBJECT from RECEPTACLE, put OBJECT in/on RECEPTACLE, clean OBJECT with SINK, heat OBJECT with MICROWAVE, cool OBJECT with FRIDGE, use LAMP, look, inventory. Use exact numbered names. Do not invent objects or task answers.
'''


def draft_messages(goal,observation,history,commands):
    past='\n\n'.join('Observation: '+r['observation']+'\nAction: '+r['action'] for r in history[-2:])
    return [dict(role='user',content=GUIDANCE+STRUCTURE+'\nTask: '+goal+'\nRecent executed history:\n'+(past or '(none)')+
        '\nCurrent observation:\n'+observation+'\nCurrent admissible commands:\n'+'\n'.join(commands))]


def review_messages(goal,observation,history,commands,draft,simulation):
    if simulation.get('prediction_available') is not True or simulation['status']!='ok':raise ValueError('Unverified trial cannot reach Reviewer')
    public={k:simulation[k] for k in ['predicted_observation','predicted_reward','predicted_done','predicted_score']}
    return [dict(role='user',content=GUIDANCE+INSTRUCTIONS+'\nGoal: '+goal+'\nCurrent Observation: '+observation+
        '\nPast committed history: '+json.dumps(history[-2:])+'\nCURRENT admissible commands: '+json.dumps(commands)+
        '\nDraft Action: '+draft+'\nGround-Truth Simulated Feedback (not committed): '+json.dumps(public))]
