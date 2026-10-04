from __future__ import annotations

from typing import List, Sequence, Tuple


SYSTEM_PROMPT = """You are a ReAct-style text-game agent for ALFWorld.
Your job is to complete the task by choosing exactly one action from the admissible command list.
Return only valid JSON. Do not add markdown.
"""


FEW_SHOTS = """Example 1
Task: put a clean mug in the cabinet.
Observation: You are in the middle of a room. You see a mug 1, sinkbasin 1, and cabinet 1.
Admissible commands:
1. look
2. take mug 1 from countertop 1
3. go to sinkbasin 1
Answer:
{"thought":"I need to pick up the mug before cleaning it.","action":"take mug 1 from countertop 1","confidence":0.84}

Example 2
Task: examine a lamp.
Observation: You are near a desk lamp 1.
Admissible commands:
1. look
2. use desklamp 1
3. inventory
Answer:
{"thought":"The goal is to examine the lamp by turning it on.","action":"use desklamp 1","confidence":0.90}
"""


def format_transcript(transcript: Sequence[Tuple[str, str]]) -> str:
    if not transcript:
        return "(empty)"
    lines = []
    for idx, (action, observation) in enumerate(transcript, start=1):
        lines.append(f"Step {idx} Action: {action}")
        lines.append(f"Step {idx} Observation: {observation}")
    return "\n".join(lines)


def format_commands(commands: List[str]) -> str:
    return "\n".join(f"{idx}. {command}" for idx, command in enumerate(commands, start=1))


def build_action_messages(
    task_goal: str,
    observation: str,
    transcript: Sequence[Tuple[str, str]],
    admissible_commands: List[str],
) -> List[dict]:
    user = f"""{FEW_SHOTS}

Current task: {task_goal}

Interaction history:
{format_transcript(transcript)}

Current observation:
{observation}

Admissible commands:
{format_commands(admissible_commands)}

Choose exactly one command from the admissible command list.
Return JSON with keys: thought, action, confidence.
Confidence must be a number from 0 to 1.
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_reflection_messages(
    task_goal: str,
    observation: str,
    recent_transcript: Sequence[Tuple[str, str]],
    proposed_action: str,
    predicted_next_observation: str,
    predicted_reward: float | int | None,
    predicted_done: bool | None,
    predicted_success: bool | None,
    admissible_commands: List[str],
) -> List[dict]:
    user = f"""Task goal:
{task_goal}

Current observation:
{observation}

Proposed action:
{proposed_action}

Predicted next observation if the proposed action is executed:
{predicted_next_observation}

Oracle transition metadata:
- reward: {predicted_reward}
- done: {predicted_done}
- success: {predicted_success}

Recent interaction history:
{format_transcript(recent_transcript)}

Admissible commands:
{format_commands(admissible_commands)}

You are evaluating whether the proposed action is a good next step for completing the task.
The world-model prediction above is an oracle transition for the proposed action.

Think about:
1. Does the predicted next observation move the agent closer to the task goal?
2. Does the proposed action gather useful information, acquire a needed object, transform an object correctly, move to a relevant location, or place/use an object in a goal-relevant way?
3. Does the action appear redundant, invalid, low-value, or inconsistent with the current state?
4. Is the agent repeating the same action or cycling through states without progress?
5. Does the oracle metadata show immediate success, failure, or terminal state?
6. Could the action make the task harder or fail, such as placing the wrong object, using the wrong receptacle, losing access to an object, or prematurely finishing?
7. Is there another admissible action that is more directly useful for the current goal?

Set "change" to true only when you intentionally replace the proposed action with a better admissible command.
If "change" is false, the "action" field must exactly copy the proposed action.
If the proposed action leads to success, keep it.
If the proposed action is harmless but not decisive, you may change it only when the predicted next state or recent history clearly supports a better next action.
Choose an action from the admissible command list.

Return only valid JSON:
{{
  "assessment": "briefly explain whether the proposed action helps the goal and any risk or loop you noticed",
  "action": "copy exactly one command from the admissible command list",
  "change": true or false,
  "reason": "brief reason for keeping or changing the action"
}}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
