#!/usr/bin/env python3
"""Validate cases and construct a frozen training-only evidence snapshot."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.splits import digest, validate_manifest
from experience.synthesis import synthesize
from experience.memory import ExperienceMemory


def build(cases, manifest):
    validate_manifest(manifest)
    cards, cost = synthesize(cases, manifest)
    snapshot = {"schema": "experience-snapshot-v1", "manifest_sha256": digest(json.dumps(manifest, sort_keys=True)),
        "built_at_ns": time.time_ns(), "cases": cases, "cards": cards, "synthesis_cost": cost}
    snapshot["snapshot_id"] = "memory_" + digest(json.dumps(snapshot, sort_keys=True))[:24]
    ExperienceMemory(snapshot, manifest)
    return snapshot


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Never overwrite a frozen memory snapshot")
    cases = [json.loads(s) for s in Path(a.cases).read_text().splitlines()]
    snapshot = build(cases, json.loads(Path(a.manifest).read_text()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "synthesis_cost": snapshot["synthesis_cost"], "cards": snapshot["cards"]}, indent=2))
