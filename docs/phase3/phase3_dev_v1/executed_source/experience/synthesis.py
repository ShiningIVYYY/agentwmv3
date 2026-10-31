"""Frozen narrow predicates; deterministic synthesis from native feedback only."""
import re
import time
from collections import defaultdict
from .schema import validate_case
from .splits import digest

SYNTHESIS_VERSION = "observed-door-rules-v1"


def closed_routes(observation):
    # Only explicit current-room descriptions qualify; old action feedback is
    # not a persistent inferred state. No hidden object/state API is consulted.
    return [(m[1].strip().casefold(), m[2].strip().casefold()) for m in re.finditer(
        r"(?:A|a) ((?:[\w-]+ )*door) to (?:the )?(.+?) \(that is closed\)", observation)]


def applicable(kind, observation, action):
    action = action.strip().casefold()
    for door, room in closed_routes(observation):
        if kind == "closed_route_go" and action in {"go " + room, "go to " + room}:
            return {"door": door, "room": room}
        if kind == "closed_door_open" and action in {"open " + door, "open the " + door,
                "open " + door + " to " + room, "open " + door + " to the " + room}:
            return {"door": door, "room": room}
    return None


def confirms(kind, feedback):
    text = feedback.strip().casefold()
    if kind == "closed_route_go":
        return text == "the door is not open."
    return text == "the door is now open."


def validate_card(card, cases):
    required = {"card_id", "schema", "benchmark", "kind", "card_type", "applicability_conditions",
        "observed_rule", "failure_pattern", "suggested_correction", "supporting_cases", "counterexamples",
        "reliability", "evidence_type", "independent_episodes", "independent_instances",
        "independent_goal_groups", "source_task_ids", "source_trajectory_ids", "available_at_ns",
        "wm_utility", "synthesis_version"}
    if set(card) != required or card["schema"] != "experience-card-v1":
        raise ValueError("Invalid card schema")
    if card["kind"] not in {"closed_route_go", "closed_door_open"}:
        raise ValueError("Unimplemented evidence predicate")
    by_id = {c["case_id"]: c for c in cases}
    support, counter = card["supporting_cases"], card["counterexamples"]
    if not support or set(support) & set(counter) or len(set(support + counter)) != len(support + counter):
        raise ValueError("Missing/duplicate/contradictory evidence IDs")
    for cid in support + counter:
        c = by_id[cid]
        if c["split"] != "train" or not applicable(card["kind"], c["observation"], c["draft_action"]):
            raise ValueError("Unqualified supporting case")
        if confirms(card["kind"], c["environment_feedback"]) != (cid in support):
            raise ValueError("Invented or misclassified native feedback")
    evidence = [by_id[cid] for cid in support]
    counts = {"independent_episodes": len({c["episode_id"] for c in evidence}),
              "independent_instances": len({tuple(c["canonical_instance"]) for c in evidence}),
              "independent_goal_groups": len({c["goal_group"] for c in evidence})}
    reliable = min(counts.values()) >= 2 and not counter
    if any(card[k] != v for k, v in counts.items()) or card["reliability"] != ("supported" if reliable else "limited"):
        raise ValueError("Inflated independence/reliability")
    all_evidence = [by_id[cid] for cid in support + counter]
    if card["source_task_ids"] != sorted({c["task_id"] for c in all_evidence}):
        raise ValueError("Card source identity mismatch")
    if card["source_trajectory_ids"] != sorted({c["source_trajectory_id"] for c in all_evidence}):
        raise ValueError("Card trajectory identity mismatch")
    if card["available_at_ns"] < max(c["outcome_available_at_ns"] for c in all_evidence):
        raise ValueError("Card predates its outcomes")
    if card["wm_utility"] != "unknown" or card["evidence_type"] != "observed_fact":
        raise ValueError("Unsupported WM utility/causal claim")
    return card


def synthesize(cases, manifest):
    start = time.perf_counter()
    unique, ids, seen = [], set(), set()
    for case in cases:
        validate_case(case, manifest)
        if case["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(case["case_id"])
        # Across cloned episodes, the same source/step is the same evidence.
        key = (case["source_trajectory_id"], case["step_index"])
        if key in seen:
            raise ValueError("Duplicated source step")
        seen.add(key)
        unique.append(case)
    cards = []
    for kind in ("closed_route_go", "closed_door_open"):
        matching = [c for c in unique if applicable(kind, c["observation"], c["draft_action"])
                    and c["draft_action"] == c["executed_action"]]
        support = [c for c in matching if confirms(kind, c["environment_feedback"])]
        counter = [c for c in matching if not confirms(kind, c["environment_feedback"])]
        if not support:
            continue
        # Duplicate transitions remain traceable but do not inflate independence.
        episodes = len({c["episode_id"] for c in support})
        instances = len({tuple(c["canonical_instance"]) for c in support})
        goals = len({c["goal_group"] for c in support})
        actor_support = [c for c in support if c["provenance"].get("collector") == "original-science-loop-frozen-qwen3-no-wm-v1"]
        actor_pattern = kind == "closed_route_go" and len({c["task_id"] for c in actor_support}) >= 2
        card = {"schema": "experience-card-v1", "benchmark": "scienceworld", "kind": kind,
            "card_id": "exp_" + digest(SYNTHESIS_VERSION + "|" + kind + "|" + "|".join(sorted(c["case_id"] for c in matching)))[:20],
            "card_type": ["environment_rule", "agent_failure_pattern"] if actor_pattern else ["environment_rule"],
            "applicability_conditions": {"predicate": kind, "requires": "current observation explicitly names a closed door to the drafted destination, or the exact door being opened"},
            "observed_rule": ("In recorded matching states, going through the explicitly closed route returned 'The door is not open.'."
                              if kind == "closed_route_go" else "In recorded matching states, opening the explicitly visible closed door returned 'The door is now open.'."),
            "failure_pattern": "Repeated Actor proposals through observed closed doors" if actor_pattern else None,
            "suggested_correction": None,  # No cross-step causal correction claim.
            "supporting_cases": sorted(c["case_id"] for c in support),
            "counterexamples": sorted(c["case_id"] for c in counter),
            "reliability": "supported" if min(episodes, instances, goals) >= 2 and not counter else "limited",
            "evidence_type": "observed_fact", "independent_episodes": episodes,
            "independent_instances": instances, "independent_goal_groups": goals,
            "source_task_ids": sorted({c["task_id"] for c in matching}),
            "source_trajectory_ids": sorted({c["source_trajectory_id"] for c in matching}),
            "available_at_ns": max(c["outcome_available_at_ns"] for c in matching),
            "wm_utility": "unknown", "synthesis_version": SYNTHESIS_VERSION}
        cards.append(validate_card(card, unique))
    return cards, {"method": SYNTHESIS_VERSION, "seconds": time.perf_counter() - start,
        "input_cases": len(cases), "cards": len(cards), "model_calls": 0, "tokens": 0,
        "extra_forward_passes": 0, "wm_utility": "unknown_no_call_skip_controls"}
