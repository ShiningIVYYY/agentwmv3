"""Versioned, causal contracts. Outcomes never enter ExternalQuery."""
from dataclasses import asdict, dataclass
from enum import Enum
import math


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self):
        raise TypeError("Tri-state signal requires explicit comparison; Unknown is not False")


@dataclass(frozen=True)
class ExternalQuery:
    benchmark: str
    task_id: str
    episode_id: str
    step_index: int
    available_at_ns: int
    goal: str
    observation: str
    draft_action: str
    premise: str = ""
    expected_consequence: str = ""
    subgoal: str = ""

    def __post_init__(self):
        if self.benchmark != "scienceworld" or self.step_index < 1 or self.available_at_ns <= 0:
            raise ValueError("Invalid causal query")
        for value in (self.task_id, self.episode_id, self.goal, self.observation):
            if not isinstance(value, str) or not value:
                raise ValueError("Missing current observable identity/state")


CASE_KEYS = {"schema", "case_id", "benchmark", "task_id", "canonical_instance", "goal_group",
    "episode_id", "split", "goal", "subgoal", "observation", "premise", "expected_consequence",
    "draft_action", "executed_action", "environment_feedback", "reward", "task_score", "done",
    "success", "wm_called", "source_trajectory_id", "step_index", "decision_at_ns",
    "outcome_available_at_ns", "provenance", "transition_fingerprint"}


def validate_case(case, manifest, construction=True):
    if set(case) != CASE_KEYS:
        raise ValueError("Case schema missing/unknown fields: " + str(set(case) ^ CASE_KEYS))
    if case["schema"] != "experience-case-v1" or case["benchmark"] != "scienceworld":
        raise ValueError("Unknown case schema/domain")
    split = "train" if construction else case["split"]
    allowed = {r["task_id"]: r for r in manifest["partitions"].get(split, [])}
    row = allowed.get(case["task_id"])
    if case["split"] != split or row is None or split == "test":
        raise ValueError("Case source not approved for this purpose")
    if (row["taskName"], row["variationIdx"]) != tuple(case["canonical_instance"]):
        raise ValueError("Canonical instance mismatch")
    from .splits import digest, goal_key
    if goal_key(case["goal"]) != case["goal_group"]:
        raise ValueError("Goal identity mismatch")
    if case["goal_group"] != row["goal_group"]:
        raise ValueError("Goal group mismatch")
    if not (0 < case["decision_at_ns"] < case["outcome_available_at_ns"]) or case["step_index"] < 1:
        raise ValueError("Noncausal step timestamps")
    for name in ("goal", "observation", "environment_feedback", "episode_id", "source_trajectory_id", "case_id"):
        if not isinstance(case[name], str) or not case[name]:
            raise ValueError("Missing case field: " + name)
    for name in ("reward", "task_score"):
        value = case[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Invalid native numeric outcome")
    if not isinstance(case["done"], bool) or not isinstance(case["wm_called"], bool):
        raise ValueError("Invalid outcome/call flag")
    if case["success"] is not None and not isinstance(case["success"], bool):
        raise ValueError("Success must be bool or unknown")
    if case["wm_called"] or case["provenance"].get("oracle_used") is not False:
        raise ValueError("This construction protocol permits only real no-WM non-Oracle collection")
    return case


def query_dict(query):
    return asdict(query)
