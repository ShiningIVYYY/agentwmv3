#!/usr/bin/env python3
"""Small training-only extension: current-room look-around facts, no dev feedback."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experience.schema import validate_case
from experience.splits import digest,file_hash
from experience.synthesis import applicable,confirms,validate_card
from experience.memory import ExperienceMemory


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",default="docs/phase3/memory_snapshot.json")
    p.add_argument("--manifest",default="docs/phase3/split_manifest.json");p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError("Frozen memory output already exists")
    start=time.perf_counter();m=json.loads(Path(a.manifest).read_text());base=json.loads(Path(a.base).read_text())
    ExperienceMemory(base,m);cases=base["cases"]
    for c in cases:validate_case(c,m)
    matching=[c for c in cases if c["draft_action"]==c["executed_action"] and applicable("room_inspection_look",c["observation"],c["draft_action"])]
    support=[c for c in matching if confirms("room_inspection_look",c["environment_feedback"],c["observation"])]
    counter=[c for c in matching if not confirms("room_inspection_look",c["environment_feedback"],c["observation"])]
    independent={"independent_episodes":len({c["episode_id"] for c in support}),
        "independent_instances":len({tuple(c["canonical_instance"]) for c in support}),
        "independent_goal_groups":len({c["goal_group"] for c in support})}
    snapshot=copy.deepcopy(base)
    if support:
        card={"schema":"experience-card-v1","card_id":"exp_"+digest("room-inspection-v1|"+"|".join(sorted(c["case_id"] for c in matching)))[:20],
            "benchmark":"scienceworld","kind":"room_inspection_look","card_type":["environment_rule"],
            "applicability_conditions":{"predicate":"room_inspection_look","requires":"current observation explicitly names room; exact look around draft"},
            "observed_rule":"In recorded states with an explicit current-room description, look around returned a description of the same room.",
            "failure_pattern":None,"suggested_correction":None,"supporting_cases":sorted(c["case_id"] for c in support),
            "counterexamples":sorted(c["case_id"] for c in counter),"reliability":"supported" if min(independent.values())>=2 and not counter else "limited",
            "evidence_type":"observed_fact",**independent,"source_task_ids":sorted({c["task_id"] for c in matching}),
            "source_trajectory_ids":sorted({c["source_trajectory_id"] for c in matching}),
            "available_at_ns":max(c["outcome_available_at_ns"] for c in matching),"wm_utility":"unknown","synthesis_version":"room-inspection-training-only-v1"}
        validate_card(card,cases);snapshot["cards"].append(card)
    snapshot["built_at_ns"]=time.time_ns();snapshot["base_snapshot_sha256"]=file_hash(a.base)
    snapshot["synthesis_cost"]={"phase5_extension_seconds":time.perf_counter()-start,"new_cases":0,
        "new_model_calls":0,"new_tokens":0,"support_independence":independent,"counterexamples":len(counter),
        "source":"existing54 approved training cases only; no dev/test outcomes"}
    del snapshot["snapshot_id"];snapshot["snapshot_id"]="memory_"+digest(json.dumps(snapshot,sort_keys=True))[:24]
    ExperienceMemory(snapshot,m)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(snapshot,indent=2)+"\n")
    print(json.dumps({"cards":len(snapshot["cards"]),"snapshot_id":snapshot["snapshot_id"],"extension":snapshot["synthesis_cost"]},indent=2))
