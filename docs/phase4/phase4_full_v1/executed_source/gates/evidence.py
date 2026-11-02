"""Bind external claims to current query and validated training evidence, never BM25 score."""
from dataclasses import asdict
from experience.splits import digest
from experience.synthesis import applicable
import json
from .core import ExternalInput, truth


def resolve_external(raw, query, memory):
    if raw is None: return ExternalInput(diagnostics=("external_not_computed",))
    try:
        coverage, conflict, sufficient = (truth(raw.get(k)) for k in ("coverage", "conflict", "sufficiency"))
        cards = {c["card_id"]: c for c in memory.cards}
        matched = tuple(raw.get("matched_experience_ids", []))
        evidence = tuple(raw.get("supporting_evidence_ids", []))
        trusted = [cards[cid] for cid in matched]
        expected = sorted({sid for c in trusted for sid in c["supporting_cases"]})
        valid = (raw["snapshot_id"] == memory.snapshot_id and
            raw["query_input_sha256"] == digest(json.dumps(asdict(query), sort_keys=True)) and
            memory.built_at <= query.available_at_ns and bool(trusted) and bool(evidence) and
            sorted(set(evidence)) == expected and len(set(evidence)) == len(evidence) and
            all(c["reliability"] == "supported" and not c["counterexamples"] and
                c["available_at_ns"] <= query.available_at_ns and query.task_id not in c["source_task_ids"] and
                applicable(c["kind"], query.observation, query.draft_action) for c in trusted))
        diagnostics = [] if valid else ["unverified_or_inapplicable_external_evidence"]
        status = "unknown"
        if valid and coverage == "true" and raw["reliability"] == "supported":
            if conflict == "true":
                corrections = {c["suggested_correction"] for c in trusted if c["suggested_correction"]}
                status = "correctable_conflict" if len(corrections) == 1 else "unresolved_conflict"
            elif conflict == "false" and sufficient == "true" and raw.get("sufficiency_scope") == "explicitly_supported_immediate_transition_only":
                status = "confirmed_action"
        return ExternalInput(coverage, conflict, sufficient, raw.get("reliability", "unknown"), evidence,
                             matched, bool(valid), status, tuple(diagnostics))
    except (KeyError, TypeError, ValueError):
        return ExternalInput(diagnostics=("malformed_external_evidence_unknown",))
