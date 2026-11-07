"""One-round public-evidence Reviewer. No hidden snapshots or second simulation."""
import json
import re
from internal_signals.decision import GUIDANCE

PROTOCOL = 'ground-truth-replay-one-round-v1'
INSTRUCTIONS = """Review ONE draft action using the independent simulator's actual one-step feedback.
The trial happened only in a separate instance. The CURRENT environment has NOT executed it.
Decide whether the draft executed successfully, caused harm, and supports the goal. A syntactically accepted action is not necessarily useful.
KEEP unless available evidence justifies a replacement and a single alternative is grounded in the CURRENT pre-action observation, executed history and action rules. Failed trial feedback is not a committed state change.
If no reliable alternative is available, KEEP is allowed. Never invent hidden facts or task answers. Use exact visible object names and one native command. No further simulation, search, retries or multiple actions.
Return exactly three lines:
Decision: KEEP or REPLACE
Reason: <brief evidence-based reason>
Final Action: <one native action, exactly the draft when KEEP>
"""


def review_messages(goal, observation, history, draft, simulation):
    if simulation.get('status') != 'ok' or simulation.get('prediction_available') is not True:
        raise ValueError('Unverified simulation must not reach Reviewer')
    public = {k: simulation[k] for k in ('predicted_observation', 'predicted_reward', 'predicted_done', 'predicted_score')}
    return [dict(role='user', content=GUIDANCE+INSTRUCTIONS+'\nGoal: '+goal+
                 '\nCurrent Observation: '+observation+'\nPast committed history: '+json.dumps(history[-2:])+
                 '\nDraft Action: '+draft+'\nGround-Truth Simulated Feedback (not committed): '+json.dumps(public))]


def parse_review(text, draft):
    match = re.fullmatch(r'Decision: (KEEP|REPLACE)\nReason: ([^\r\n]+)\nFinal Action: ([^\r\n]+)\n?', text)
    if not match:
        return dict(decision='INVALID', final_action=draft, error='review_format_error', fallback='keep_draft')
    decision, reason, action = match.groups()
    action = action.strip()
    if not action or not reason.strip() or (decision == 'KEEP' and action != draft):
        return dict(decision='INVALID', final_action=draft, error='inconsistent_or_empty_review', fallback='keep_draft')
    return dict(decision=decision, reason=reason, final_action=action,
                changed=action != draft, action_validity='unknown', semantic_utility='unknown')
