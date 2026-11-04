"""Frozen BM25 retrieval plus conservative, evidence-backed external signals."""
from collections import Counter
from dataclasses import asdict
import json
import math
import re
import time
from typing import Protocol
from .schema import ExternalQuery, Truth, validate_case
from .splits import digest, validate_manifest
from .synthesis import applicable, validate_card


class ExternalSignalProvider(Protocol):
    def evaluate(self, query: ExternalQuery) -> dict: ...


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def card_text(card):
    return " ".join(str(card[k]) for k in ("kind", "observed_rule", "failure_pattern", "applicability_conditions"))


def bm25(cards, query, top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not cards:
        return []
    docs = [Counter(tokens(card_text(c))) for c in cards]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / len(docs)
    q = set(tokens(query))
    results = []
    for card, doc, length in zip(cards, docs, lengths):
        score = 0.0
        for term in q:
            tf = doc[term]
            if tf:
                df = sum(term in d for d in docs)
                idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / avg))
        if score > 0:
            results.append((card, score))
    return sorted(results, key=lambda pair: (-pair[1], pair[0]["card_id"]))[:top_k]


class ExperienceMemory:
    """No online writes. No model calls. Similarity never implies reliability."""
    def __init__(self, snapshot, manifest, top_k=5):
        load_start = time.perf_counter()
        validate_manifest(manifest, require_native=True)
        if snapshot["schema"] != "experience-snapshot-v1":
            raise ValueError("Unknown memory schema")
        if snapshot["manifest_sha256"] != digest(json.dumps(manifest, sort_keys=True)):
            raise ValueError("Snapshot split identity changed")
        self.cases = snapshot["cases"]
        for case in self.cases:
            validate_case(case, manifest)
        if len({c["case_id"] for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case ID in snapshot")
        self.cards = snapshot["cards"]
        if len({c["card_id"] for c in self.cards}) != len(self.cards):
            raise ValueError("Duplicate card ID")
        for card in self.cards:
            validate_card(card, self.cases)
        self.built_at = snapshot["built_at_ns"]
        if any(c["available_at_ns"] > self.built_at for c in self.cards):
            raise ValueError("Snapshot predates evidence")
        self.top_k = top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Invalid Top-K")
        self.snapshot_id = snapshot["snapshot_id"]
        self.serialized_bytes = len(json.dumps(snapshot).encode())
        self.load_seconds = time.perf_counter() - load_start

    def evaluate(self, query):
        if not isinstance(query, ExternalQuery):
            raise TypeError("Only explicit causal ExternalQuery is allowed")
        start = time.perf_counter()
        cpu_start = time.process_time()
        # Filter before computing IDF: future/source-excluded cards must not
        # affect even the retrieval ranking.
        eligible = [c for c in self.cards if self.built_at <= query.available_at_ns
                    and c["available_at_ns"] <= query.available_at_ns
                    and query.task_id not in c["source_task_ids"]
                    and not any(x["episode_id"] == query.episode_id for x in self.cases
                                if x["case_id"] in c["supporting_cases"] + c["counterexamples"])]
        text = " ".join((query.goal, query.subgoal, query.observation, query.draft_action))
        ranked = bm25(eligible, text, self.top_k)
        hits = [{"experience_id": c["card_id"], "bm25_score": score,
                 "applicable": applicable(c["kind"], query.observation, query.draft_action) is not None,
                 "reliability": c["reliability"], "source_ids": c["supporting_cases"] + c["counterexamples"]}
                for c, score in ranked]
        retrieval_latency = time.perf_counter() - start
        retrieval_cpu = time.process_time() - cpu_start
        judge_start = time.perf_counter()
        judge_cpu_start = time.process_time()
        relevant = [c for c, _ in ranked if applicable(c["kind"], query.observation, query.draft_action)]
        trusted = [c for c in relevant if c["reliability"] == "supported" and not c["counterexamples"]]
        uncertain = [c for c in relevant if c not in trusted]
        coverage, conflict, sufficient = Truth.UNKNOWN, Truth.UNKNOWN, Truth.UNKNOWN
        reason = "No reliable applicable evidence; absence is Unknown, not no-conflict."
        if trusted and not uncertain:
            coverage = Truth.TRUE
            kinds = {c["kind"] for c in trusted}
            if kinds == {"closed_route_go"}:
                conflict, sufficient = Truth.TRUE, Truth.FALSE
                reason = "Draft movement contradicts supported closed-route precondition; action remains unchanged and unresolved."
            elif kinds == {"closed_door_open"}:
                conflict = Truth.UNKNOWN
                sufficient = Truth.UNKNOWN
                # Narrow confirmation only, requires the Agent to actually
                # predict the effect supported by the evidence, not just match
                # an action string or arbitrary remote goal.
                consequence = query.expected_consequence.casefold().strip().rstrip(".")
                bindings = applicable("closed_door_open", query.observation, query.draft_action)
                allowed = {"the door will open", "the door will be open", "the door is now open",
                           "the " + bindings["door"] + " to the " + bindings["room"] + " will open"}
                premise = query.premise.casefold().strip().rstrip(".")
                premises = {"the door is closed", "the " + bindings["door"] + " to the " + bindings["room"] + " is closed"}
                if consequence in allowed and premise in premises:
                    conflict, sufficient = Truth.FALSE, Truth.TRUE
                    reason = "Supported visible-door opening and exact immediate-effect claim; sufficiency covers this transition only."
                else:
                    reason = "Observed action has support, but the Agent consequence is outside the verified claim scope."
            elif kinds == {"room_inspection_look"}:
                binding = applicable("room_inspection_look", query.observation, query.draft_action)
                room = binding["room"]
                premises = {"the agent is in the " + room, "i am in the " + room, "the current room is the " + room}
                effects = {"the current room will be observed", "the room description will be visible", "i will see the current room"}
                if query.premise.casefold().strip().rstrip(".") in premises and query.expected_consequence.casefold().strip().rstrip(".") in effects:
                    conflict, sufficient = Truth.FALSE, Truth.TRUE
                    reason = "Supported current-room inspection with exact observable premise/immediate-effect claim; no task-utility claim."
                else:
                    reason = "Room-inspection evidence applies to this action/state, but the Agent premise/effect is not fully confirmed."
        elif uncertain:
            reason = "Applicable evidence is limited or has counterexamples; all judgments remain Unknown."
        evidence = sorted({cid for c in trusted for cid in c["supporting_cases"]}) if not uncertain else []
        return {"schema": "external-signals-v1", "coverage": coverage.value, "conflict": conflict.value,
            "sufficiency": sufficient.value, "reliability": "supported" if trusted and not uncertain else "unknown",
            "matched_experience_ids": [c["card_id"] for c in relevant],
            "supporting_evidence_ids": evidence, "reason": reason,
            "retrieval_latency": retrieval_latency, "judge_latency": time.perf_counter() - judge_start,
            "retrieval_cpu_seconds": retrieval_cpu, "judge_cpu_seconds": time.process_time() - judge_cpu_start,
            "additional_tokens": 0, "retrieval_calls": 1, "judge_model_calls": 0,
            "extra_forward_passes": 0, "retrieval_method": "bm25-k1=1.5-b=0.75-v1",
            "judge_method": "conservative-observed-door-v1", "snapshot_id": self.snapshot_id,
            "query_input_sha256": digest(json.dumps(asdict(query), sort_keys=True)),
            "retrieved_cards": hits, "memory_serialized_bytes": self.serialized_bytes,
            "query_at_ns": query.available_at_ns, "source_scope": "training_only_frozen_snapshot",
            "wm_utility": "unknown", "action_modified": False,
            "sufficiency_scope": "explicitly_supported_immediate_transition_only"}
