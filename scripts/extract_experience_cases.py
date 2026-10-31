#!/usr/bin/env python3
"""Re-extract and causally audit actual saved HTTP-backed trajectories."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.collection import case_from_trace
from experience.schema import ExternalQuery, validate_case
from experience.splits import digest, validate_manifest


def extract(run_dir, manifest):
    validate_manifest(manifest)
    root = Path(run_dir)
    read = lambda name: [json.loads(s) for s in (root / name).read_text().splitlines()]
    originals, traces, receipts = read("cases.jsonl"), read("steps.jsonl"), read("receipts.jsonl")
    if not len(originals) == len(traces) == len(receipts):
        raise ValueError("Incomplete source trace/receipt archive")
    manifest_sha = digest(json.dumps(manifest, sort_keys=True))
    run = json.loads((root / "manifest.json").read_text())
    if run["split_sha256"] != manifest_sha:
        raise ValueError("Source run used a different split")
    rows = {r["task_id"]: r for r in run["tasks"]}
    result = []
    for original, trace, receipt in zip(originals, traces, receipts):
        row = rows[original["task_id"]]
        case = case_from_trace(row, root.name, trace, original["step_index"], original["decision_at_ns"],
            receipt, original["goal"], manifest_sha, run.get("collector", "original-science-loop-frozen-qwen3-no-wm-v1"))
        if original != case:
            raise ValueError("Extracted case differs from actual source trace/receipt")
        validate_case(case, manifest, construction=case["split"] == "train")
        query_at = trace.get("external_signals", {}).get("query_at_ns")
        if query_at is not None:
            q = ExternalQuery(benchmark="scienceworld", task_id=case["task_id"], episode_id=case["episode_id"],
                step_index=case["step_index"], available_at_ns=query_at, goal=case["goal"],
                observation=case["observation"], draft_action=case["draft_action"], premise=case["premise"],
                expected_consequence=case["expected_consequence"])
            if trace["external_signals"]["query_input_sha256"] != digest(json.dumps(asdict(q), sort_keys=True)):
                raise ValueError("External query was not the exact causal current input")
            if not case["decision_at_ns"] <= query_at < receipt["started_at_ns"] < case["outcome_available_at_ns"]:
                raise ValueError("Query/outcome order leakage")
        result.append(case)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("New extracted archive required")
    cases = extract(a.run_dir, json.loads(Path(a.manifest).read_text()))
    out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(json.dumps({"validated_cases": len(cases), "receipt_alignment": True, "causal_input_hashes": True}))
