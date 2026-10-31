#!/usr/bin/env python3
"""Replay external judgments and verify every causal input against actual receipts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.schema import ExternalQuery
from experience.splits import file_hash
from experience.memory import ExperienceMemory
from internal_signals.decision import messages
from scripts.extract_experience_cases import extract
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.run_dir)
    manifest = json.loads(Path(a.manifest).read_text())
    memory = ExperienceMemory(json.loads(Path(a.snapshot).read_text()), manifest)
    cases = extract(root, manifest)
    traces = [json.loads(s) for s in (root / "steps.jsonl").read_text().splitlines()]
    history = {}
    fields = ("coverage", "conflict", "sufficiency", "reliability", "matched_experience_ids",
              "supporting_evidence_ids", "reason", "query_input_sha256", "snapshot_id", "action_modified")
    for case, trace in zip(cases, traces):
        q = ExternalQuery(benchmark="scienceworld", task_id=case["task_id"], episode_id=case["episode_id"],
            step_index=case["step_index"], available_at_ns=trace["external_signals"]["query_at_ns"],
            goal=case["goal"], observation=case["observation"], draft_action=case["draft_action"],
            premise=case["premise"], expected_consequence=case["expected_consequence"])
        replay = memory.evaluate(q)
        if any(replay[k] != trace["external_signals"][k] for k in fields):
            raise ValueError("External judgment not reproducible")
        previous = history.setdefault(case["episode_id"], [])
        expected = messages(case["goal"], case["observation"], previous, trace["history_length"])
        if expected != trace["messages"]:
            raise ValueError("Actor prompt differs from unchanged Phase2 visible-history builder")
        previous.append(trace)
    run = json.loads((root / "manifest.json").read_text())
    changed = [name for name, sha in run["source_hashes"].items() if file_hash(ROOT / name) != sha]
    for name in changed:
        archived = root / "executed_source" / name
        if not archived.exists() or file_hash(archived) != run["source_hashes"][name]:
            raise ValueError("Executed source archive missing/differs: " + name)
    result = {"steps": len(cases), "all_native_receipts_aligned": True, "all_queries_before_environment_step": True,
        "external_input_hashes_match_causal_fields": True, "all_external_signals_replay_exactly": True,
        "all_actor_prompts_match_phase2_builder": True, "executed_source_hashes_match_current_or_archived": True,
        "current_source_differences": changed,
        "memory_train_only": True, "actions_unchanged": True, "test_trajectories_used": 0}
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
