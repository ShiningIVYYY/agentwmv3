"""Actual HTTP receipt capture for the existing Agent/Science loop, no mock data."""
from contextlib import contextmanager
import re
import time
from .splits import digest


def fields(text):
    match = re.fullmatch(r"Premise: ([^\r\n]+)\nExpected Consequence: ([^\r\n]+)\nAction: ([^\r\n]+)\n?", text)
    if not match:
        return "", "", text.rsplit("Action:", 1)[-1].strip() if "Action:" in text else ""
    return match[1], match[2], match[3].strip()


@contextmanager
def receipts(runner):
    original = runner.http
    log = []
    def captured(method, url, **kwargs):
        start = time.time_ns()
        value = original(method, url, **kwargs)  # one real HTTP request, no retries
        if url.endswith(("/reset", "/step")):
            log.append({"endpoint": url.rsplit("/", 1)[-1], "request": kwargs.get("json"),
                        "response": value, "started_at_ns": start, "available_at_ns": time.time_ns()})
        return value
    runner.http = captured
    try:
        yield log
    finally:
        runner.http = original


def case_from_trace(row, run_id, record, step_index, decision_at, receipt, goal, manifest_sha,
                    collector="original-science-loop-frozen-qwen3-no-wm-v1"):
    if receipt["endpoint"] != "step" or receipt["request"]["action"] != record["action"]:
        raise ValueError("Trace action differs from actual environment request")
    response = receipt["response"]
    if response["observation"] != record["next_observation"] or response["score"] != record["score"]:
        raise ValueError("Trace feedback differs from actual HTTP receipt")
    premise, consequence, draft = fields(record["output"])
    if draft != record["action"]:
        raise ValueError("Experience diagnostics must not modify the draft")
    episode = run_id + "/" + row["task_id"]
    fingerprint = digest("\0".join((record["observation"], draft, response["observation"])))
    return {"schema": "experience-case-v1", "case_id": episode + "/step-" + str(step_index),
        "benchmark": "scienceworld", "task_id": row["task_id"],
        "canonical_instance": [row["taskName"], row["variationIdx"]], "goal_group": row["goal_group"],
        "episode_id": episode, "split": row["split"], "goal": goal, "subgoal": None,
        "observation": record["observation"], "premise": premise, "expected_consequence": consequence,
        "draft_action": draft, "executed_action": record["action"],
        "environment_feedback": response["observation"], "reward": response.get("reward"),
        "task_score": response.get("score"), "done": response["done"],
        "success": bool(response["done"] and float(response["score"]) == 100) if response["done"] else None,
        "wm_called": False, "source_trajectory_id": episode, "step_index": step_index,
        "decision_at_ns": decision_at, "outcome_available_at_ns": receipt["available_at_ns"],
        "transition_fingerprint": fingerprint,
        "provenance": {"collector": collector, "oracle_used": False,
            "feedback_source": "actual_AgentGym_HTTP_step_response", "split_sha256": manifest_sha,
            "success_definition": "done and native score == 100; partial success remains unknown"}}
