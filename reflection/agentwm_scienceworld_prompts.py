from __future__ import annotations


SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
"""


SCIWORLD_TEMPLATE = """
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
"""


SCIWORLD_REFLECTION_TEMPLATE_STANDARD = """
You are an expert action reviewer for a ScienceWorld ReAct agent.
Your job is to decide whether the proposed action should still be executed after seeing an oracle world-model prediction.

Task:
{task_description}

Current observation:
{current_observation}

Proposed action:
{proposed_action}

Predicted next observation if the proposed action is executed:
{predicted_next_observation}

Oracle transition metadata:
- reward: {predicted_reward}
- done: {predicted_done}
- success: {predicted_success}
- score: {predicted_score}

Recent history:
{action_history}

Current available actions:
{available_actions}

Think about:
1. Does the predicted next observation move the agent closer to the science task goal?
2. Does the action gather useful information, measure something, transform matter correctly, grow or inspect a biological object, connect or test a circuit, mix substances, heat or cool an object, move a needed object, or place/use an object in a goal-relevant way?
3. Does the action appear redundant, invalid, low-value, or inconsistent with the current state?
4. Is the agent repeating observation/navigation actions without new information?
5. Does the oracle metadata show immediate success, failure, score improvement, score loss, or terminal state?
6. Could the action make the task harder, such as mixing the wrong substances, heating/cooling the wrong object, connecting the wrong circuit, discarding a needed object, moving an object to the wrong place, or prematurely committing to a final answer?
7. Is there another available action that is more directly useful for the current goal?

Set "change" to true only when you intentionally replace the proposed action with a better ScienceWorld action.
If "change" is false, the "action" field must exactly copy the proposed action.
If the proposed action leads to success or improves score without clear downside, keep it.
If the proposed action is harmless but not decisive, change it only when the predicted next state or recent history clearly supports a better next action.

Return only valid JSON:
{{
  "assessment": "briefly explain whether the proposed action helps the goal and any risk or loop you noticed",
  "action": "one ScienceWorld action",
  "change": true or false,
  "reason": "brief reason for keeping or changing the action"
}}
"""


SCIWORLD_REFLECTION_TEMPLATE_CONSERVATIVE_V1 = """
You are an expert action reviewer for a ScienceWorld ReAct agent.
Your job is to decide whether the proposed action should still be executed after seeing an oracle world-model prediction.

Task:
{task_description}

Recent history:
{action_history}

Current observation:
{current_observation}

Proposed action:
{proposed_action}

Oracle prediction if the proposed action is executed:
{predicted_next_observation}

Oracle transition metadata:
- reward: {predicted_reward}
- done: {predicted_done}
- success: {predicted_success}
- score: {predicted_score}

Current available actions:
{available_actions}

Default rule:
Keep the proposed action unless the oracle prediction gives clear evidence that it should be changed.

Important ScienceWorld rules:
1. reward = 0 does not mean the action is wrong.
2. score unchanged does not mean the action is wrong.
3. done = False does not mean the action is wrong.
4. Many correct ScienceWorld actions are intermediate setup steps, such as picking up an object, opening a container, moving an object, reading instructions, connecting components, heating/cooling, mixing, measuring, or testing.
5. Do not change a useful intermediate action only because it does not immediately increase score.
6. A positive terminal score or success = True means the action should be kept.
7. done = True with success = False, especially score = -100, means the action is harmful and should be changed.
8. "No known action matches that input" means the action is invalid or not executable and should be changed.
9. Repeating the same action or same low-information observation without any new state change is a loop signal and should be changed.
10. "focus on X" can behave like a final answer or commitment in many ScienceWorld tasks. Do not replace a non-focus action with a focus action unless the current state strongly supports that X is the correct target. If a proposed focus action leads to failure or score -100, change it.

Decision procedure:
1. If the proposed action leads to success, a positive terminal score, or a clear score increase without downside, keep it.
2. If the proposed action is a necessary setup step and the oracle prediction shows a plausible state change, keep it even if reward is 0 or score is unchanged.
3. If the proposed action is invalid, causes terminal failure, lowers score sharply, commits to a wrong object, or repeats a no-progress loop, change it.
4. If the evidence is ambiguous, keep the proposed action.
5. If changing the action, choose the smallest correction that addresses the observed problem. Prefer safe information-gathering or necessary setup actions over premature final-answer/focus actions.
6. If "change" is false, the "action" field must exactly copy the proposed action.

Return only valid JSON:
{{
  "assessment": "brief explanation of whether the oracle prediction shows success, useful progress, harmless intermediate progress, invalidity, failure, or loop",
  "action": "one ScienceWorld action",
  "change": true or false,
  "reason": "short reason for keeping or replacing the proposed action"
}}
"""


SCIWORLD_REFLECTION_TEMPLATE_CRITIQUE_REPLAN_V1 = """
You are an expert action reviewer for a ScienceWorld ReAct agent.
Your job is to evaluate the proposed action using an oracle world-model prediction.

Important:
- The predicted next observation is counterfactual evidence about the proposed action.
- It is not the current observation.
- Do not choose an action for the predicted next state.
- Only decide whether the proposed action should be executed from the current state.
- If the proposed action should not be executed, request replanning. A separate agent call will generate the replacement action from the current observation.

Task:
{task_description}

Current observation:
{current_observation}

Proposed action:
{proposed_action}

Oracle prediction if the proposed action is executed:
{predicted_next_observation}

Oracle transition metadata:
- reward: {predicted_reward}
- done: {predicted_done}
- success: {predicted_success}
- score: {predicted_score}

Recent history:
{action_history}

Current available action schema and objects:
{available_actions}

Decision rules:
1. Set "execute_proposed" to true if the proposed action is useful, harmless setup, information gathering, or leads to success.
2. Set "execute_proposed" to false only if the oracle prediction clearly shows invalid input, terminal failure, a wrong final focus, a repeated no-progress loop, or a harmful state change.
3. Set "needs_replan" to true only when "execute_proposed" is false and a replacement action should be generated from the current observation.
4. Do not output the final replacement action. If you have a possible idea, put it in "suggested_action" only as non-binding advice.
5. If evidence is ambiguous, keep the proposed action.

Return only valid JSON:
{{
  "assessment": "briefly assess the proposed action using the oracle prediction as counterfactual evidence",
  "execute_proposed": true or false,
  "needs_replan": true or false,
  "suggested_action": "optional non-binding idea, or empty string",
  "reason": "short reason for the decision"
}}
"""


SCIWORLD_GUARDED_REPLAN_TEMPLATE_V1 = """
You are an expert ScienceWorld ReAct agent.
Generate a replacement action for the current real environment state.

Important:
- The current observation below is the only state you are acting in.
- The oracle feedback describes why the previous proposed action should not be executed.
- Do not act as if the predicted next observation has already happened.
- Choose one action for the current observation.

Task:
{task_description}

Recent history:
{action_history}

Current observation:
{current_observation}

Previous proposed action:
{proposed_action}

Oracle feedback about the previous proposed action:
- predicted next observation: {predicted_next_observation}
- reward: {predicted_reward}
- done: {predicted_done}
- success: {predicted_success}
- score: {predicted_score}
- reviewer assessment: {reflection_assessment}
- reviewer reason: {reflection_reason}
- reviewer non-binding suggestion: {reflection_suggested_action}

Current available actions:
{available_actions}

Now choose a replacement action for the current observation.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one appropriate action for the current step and present it within <action> </action> tags.
"""


SCIWORLD_INVALID_ACTION_REPLAN_TEMPLATE_V1 = """
You are an expert ScienceWorld ReAct agent.
The previous model output produced an action that was not directly executable.
Generate one replacement action for the current real environment state.

Important:
- Do not use an oracle world-model prediction.
- Do not act in an imagined future state.
- Choose exactly one executable action from the current state.
- Prefer the smallest correction that fixes the action format or action grounding problem.
- Avoid `focus on ...` unless the current evidence strongly supports the final answer.

Task:
{task_description}

Recent history:
{action_history}

Current observation:
{current_observation}

Previous model output:
{model_output}

Parsed action:
{proposed_action}

Parser feedback:
{parser_feedback}

Current available actions:
{available_actions}

Now choose a replacement action for the current observation.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one appropriate action for the current step and present it within <action> </action> tags.
"""


SCIWORLD_REFLECTION_TEMPLATES = {
    "standard": SCIWORLD_REFLECTION_TEMPLATE_STANDARD,
    "conservative_v1": SCIWORLD_REFLECTION_TEMPLATE_CONSERVATIVE_V1,
    "critique_replan_v1": SCIWORLD_REFLECTION_TEMPLATE_CRITIQUE_REPLAN_V1,
}


SCIWORLD_GUARDED_REPLAN_TEMPLATES = {
    "guarded_replan_v1": SCIWORLD_GUARDED_REPLAN_TEMPLATE_V1,
}


SCIWORLD_INVALID_ACTION_REPLAN_TEMPLATES = {
    "invalid_action_replan_v1": SCIWORLD_INVALID_ACTION_REPLAN_TEMPLATE_V1,
}
