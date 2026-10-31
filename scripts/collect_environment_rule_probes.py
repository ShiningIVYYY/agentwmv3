#!/usr/bin/env python3
"""Small real controlled training probes, explicitly NOT Qwen rollouts/metrics.

Exercises closed-route failure, real re-observation, then door opening. Every
outcome is from the actual service; no oracle, model, correction or test tasks.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_audited_paper_eval import http, atomic
from experience.collection import case_from_trace
from experience.schema import validate_case
from experience.splits import digest, file_hash, goal_key, validate_manifest
from experience.synthesis import closed_routes

COLLECTOR = "controlled-real-environment-door-probe-v1-not-actor"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--science-url", default="http://127.0.0.1:36001")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists(): raise ValueError("New probe Run ID required")
    m = json.loads(Path(a.manifest).read_text()); validate_manifest(m)
    rows = m["diagnostic_selection"]["train"]
    out.mkdir(parents=True)
    sha = digest(json.dumps(m, sort_keys=True))
    atomic(out / "manifest.json", {"collector": COLLECTOR, "tasks": rows, "split_sha256": sha,
        "script_sha256": file_hash(__file__), "purpose": "environment_rule_evidence_not_actor_performance",
        "selection": "first two selected training instances with explicitly visible closed route", "max_episodes": 2})
    total, episodes, seconds = 0, 0, 0.0
    for row in rows:
        if episodes == 2: break
        eid = http("POST", a.science_url + "/create")["id"]
        try:
            rst = http("POST", a.science_url + "/reset", json={"id": eid, "data_idx": row["index"]})
            if (rst["task_name"], rst["var_num"]) != (row["taskName"], row["variationIdx"]):
                raise ValueError("Native mapping mismatch")
            goal = rst["task_description"]
            if goal_key(goal) != row["goal_group"]: raise ValueError("Native goal mismatch")
            routes = closed_routes(rst["observation"])
            if not routes: continue
            door, room = routes[0]
            plans = [("go " + room, "The door is closed.", "The movement is expected to be blocked."),
                     ("look around", "The agent is in the current room.", "The room description will be visible."),
                     ("open " + door + " to the " + room, "The door is closed.", "The door will open.")]
            for index, (action, premise, consequence) in enumerate(plans, 1):
                decision_at = time.time_ns()
                text = "Premise: " + premise + "\nExpected Consequence: " + consequence + "\nAction: " + action
                before = rst["observation"]
                start = time.time_ns()
                req = {"id": eid, "action": action}
                rst = http("POST", a.science_url + "/step", json=req)
                end = time.time_ns(); seconds += (end - start) / 1e9
                receipt = {"endpoint": "step", "request": req, "response": rst,
                           "started_at_ns": start, "available_at_ns": end}
                record = {"output": text, "action": action, "observation": before,
                          "next_observation": rst["observation"], "score": rst["score"]}
                case = case_from_trace(row, out.name, record, index, decision_at, receipt, goal, sha, COLLECTOR)
                validate_case(case, m)
                for name, item in [("cases.jsonl", case), ("steps.jsonl", record), ("receipts.jsonl", receipt)]:
                    with (out / name).open("a") as stream: stream.write(json.dumps(item) + "\n")
                total += 1
            episodes += 1
        finally:
            http("POST", a.science_url + "/close", json={"id": eid})
    atomic(out / "summary.json", {"episodes": episodes, "real_environment_steps": total,
        "model_calls": 0, "tokens": 0, "environment_step_seconds": seconds,
        "purpose": "controlled_environment_facts_only_not_natural_conflict_or_benchmark_performance"})
    print((out / "summary.json").read_text())
