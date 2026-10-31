#!/usr/bin/env python3
"""Read simulator task identity ONLY. No policy actions, trajectories or metrics."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.splits import file_hash
from scripts.run_audited_paper_eval import http, atomic

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--science-url", default="http://127.0.0.1:36001")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("New output required")
    manifest = json.loads(Path(a.manifest).read_text())
    rows = manifest["partitions"]["test"] + sum(manifest["diagnostic_selection"].values(), [])
    records = []
    start = time.perf_counter()
    eid = http("POST", a.science_url + "/create")["id"]
    try:
        for row in rows:
            r = http("POST", a.science_url + "/reset", json={"id": eid, "data_idx": int(row["task_id"].rsplit("_", 1)[1])})
            if (r["task_name"], r["var_num"]) != (row["taskName"], row["variationIdx"]):
                raise ValueError("Native task mapping changed")
            # Explicit allowlist. Observation/score/reward/done are discarded.
            records.append({"task_id": row["task_id"], "task_name": r["task_name"],
                "variation_id": r["var_num"], "goal": r["task_description"]})
            atomic(out, {"purpose": "identity_only_no_test_outcome_use", "source_manifest_sha256": file_hash(a.manifest),
                "records": records, "complete": False, "reset_calls": len(records),
                "seconds": time.perf_counter() - start, "policy_actions": 0, "model_calls": 0})
    finally:
        http("POST", a.science_url + "/close", json={"id": eid})
    data = json.loads(out.read_text()); data["complete"] = True
    atomic(out, data)
    print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2))
