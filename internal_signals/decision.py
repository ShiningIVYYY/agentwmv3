"""Opt-in structured ScienceWorld decision prompt; no future feedback fields."""
import re

SCHEMA = "scienceworld-premise-consequence-action-v1"
GUIDANCE = """You are an agent in ScienceWorld. Complete the current task using only current observations and already executed history.
Use exact object names from observations. Available action forms include: look around; inventory; go ROOM; open OBJ; close OBJ; activate OBJ; deactivate OBJ; look at OBJ; look in OBJ; pick up OBJ; move OBJ to CONTAINER; use TOOL on OBJ; read OBJ; connect OBJ to OBJ; disconnect OBJ; pour OBJ in OBJ; mix OBJ; focus on OBJ; wait.
If the current observation asks for a numbered disambiguation choice, return that number as the action.
"""
STRUCTURE = """Return exactly three lines with no markdown or extra text:
Premise: <one short current condition your action depends on>
Expected Consequence: <one short predicted immediate effect, not observed feedback>
Action: <one concrete ScienceWorld command>
"""
ACTION_ONLY = "Return exactly one line: Action: <one concrete ScienceWorld command>\n"


def messages(task, observation, steps, keep=2, structured=True):
    recent = steps[-keep:] if keep else []
    history = "\n\n".join("Observation: " + s["observation"] + "\nAction: " + s["action"] for s in recent)
    return [{"role": "user", "content": GUIDANCE + (STRUCTURE if structured else ACTION_ONLY) +
             "\nTask: " + task + "\nRecent executed history:\n" + (history or "(none)") +
             "\nCurrent observation:\n" + observation}]


def bounded(agent, task, observation, steps):
    for keep in range(min(len(steps), agent.internal_actor.config["history_length"]), -1, -1):
        result = messages(task, observation, steps, keep)
        if agent.count(result) <= agent.internal_actor.config["max_prompt_tokens"]:
            return result, keep
    raise ValueError("Required goal/current observation exceeds token limit; no silent truncation")


def action_from_text(text):
    # Preserve the existing ScienceWorld last-Action-tail contract.
    parts = text.rsplit("Action:", 1)
    return parts[-1].strip() if len(parts) == 2 else ""


def action_token_indices(text, offsets):
    found = list(re.finditer(r"(?m)^Action: ([^\r\n]+)", text))
    if len(found) != 1:
        return []
    left, right = found[0].span(1)
    return [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
