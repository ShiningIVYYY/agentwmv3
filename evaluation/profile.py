"""Explicit profile validation and official flat-list canonical presentation."""
import re


def canonicalize_observation(text):
    """Copied behavior from pinned official envs.py; numbered/nested rows unchanged."""
    text = str(text)
    if "\n\t\t" in text: return text
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        end = index
        while end < len(lines) and re.match(r"^\t(?:a|an) ", lines[end], re.IGNORECASE): end += 1
        if end > index:
            lines[index:end] = sorted(lines[index:end]); index = end
        else: index += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def validate_profile(p):
    expected = {"benchmark":"scienceworld", "maximum_interaction_turns":20, "history_length":2,
        "prompt_limit":4096, "response_limit":512, "temperature":.4, "top_p":1.0,
        "hf_top_k":0, "task_seed":0, "rollout_seeds":list(range(8)),
        "environment_step_limit":21, "simplifications":"", "metric":"native_final_task_score",
        "observation_format":"canonical-flat-object-lists-v1", "overflow_policy":"left_token_truncation",
        "actor_model":"Qwen/Qwen3-4B", "thinking":False}
    if any(p.get(k)!=v for k,v in expected.items()): raise ValueError("Paper-aligned profile differs from pinned contract")
    if p["test_count"]!=200 or p["test_source_sha256"]!="78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79":
        raise ValueError("Wrong fixed evaluation task identity")
    return p


def native_menu_actions(observation):
    if not observation.startswith("Ambiguous request: Please enter the number for the action you intended"):
        return []
    return re.findall(r"(?m)^(\d+):\t.+$",observation)


def goal_visible(tokenizer,prompt,goal,removed_tokens):
    if not removed_tokens:return True
    start=prompt.find(goal)
    if start<0:return None
    end=start+len(goal)
    offsets=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
    return not any(a<end and b>start for a,b in offsets[:removed_tokens])
