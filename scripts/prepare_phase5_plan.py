#!/usr/bin/env python3
"""Freeze independent goal groups BEFORE collecting new development outcomes."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experience.splits import digest,file_hash,validate_manifest


def prepare(manifest):
    validate_manifest(manifest,require_native=True)
    groups=defaultdict(list)
    for r in manifest["partitions"]["development"]:groups[r["goal_group"]].append(r)
    exposed_ids={"sciworld_1466","sciworld_146","sciworld_3055","sciworld_1709","sciworld_349"}
    exposed={g for g,rows in groups.items() if any(r["task_id"] in exposed_ids for r in rows)}
    ranked=sorted(groups,key=lambda g:digest("agentwmv3-phase5-groups-v1|"+g))
    validation=[];validation_groups=set()
    for category,count in [("measure",2),("test_conductivity",2),("find",1)]:
        choices=[g for g in ranked if g not in exposed and g not in validation_groups and groups[g][0]["category"]==category]
        for g in choices[:count]:
            validation_groups.add(g)
            validation.append(min(groups[g],key=lambda r:digest("agentwmv3-phase5-task-v1|"+r["task_id"])))
    if len(validation)!=5:raise ValueError("Insufficient disjoint development groups")
    calibration_groups=set(exposed)
    for category in ("chemistry_mix","lifespan"):
        calibration_groups.update(g for g in ranked if groups[g][0]["category"]==category)
    for g in ranked:
        if len(calibration_groups)>=30:break
        if g not in validation_groups:calibration_groups.add(g)
    calibration=[min(groups[g],key=lambda r:digest("agentwmv3-phase5-task-v1|"+r["task_id"])) for g in sorted(calibration_groups)]
    assert not calibration_groups & validation_groups
    return {"schema":"phase5-frozen-development-plan-v1","calibration_tasks":calibration,
        "validation_tasks":validation,"calibration_max_steps":10,"episode_max_steps":20,"rollout_seed":0,
        "policies":["never","always","internal","full"],"prior_exposed_goal_groups":sorted(exposed),
        "reserved_uncollected_goal_groups":sorted(set(groups)-calibration_groups-validation_groups),
        "calibration_target":"30 goal groups, up to300 real steps; distribution only; not outcome-selected",
        "validation_scope":"5 independent goal groups in Measure/Test-Cond/Find; Chem-Mix/Lifespan not fully episode-validated",
        "limitation":"Chem-Mix and Lifespan each have only one available dev goal group; assigned calibration to avoid cross-split goal leakage",
        "category_counts":{"calibration":dict(Counter(r["category"] for r in calibration)),
                           "validation":dict(Counter(r["category"] for r in validation))},
        "budget":{"never":0,"always":20,"internal":5,"full":5},
        "budget_selection":"preset independently of NLL quantiles/outcomes; actual forecasts separately capped at4*invocations",
        "memory":"unchanged training-only Phase3 snapshot; no dev insertion",
        "no_final_test_rollouts":True}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError("Frozen plan must use a new output")
    plan=prepare(json.loads(Path(a.manifest).read_text()));plan["manifest_sha256"]=file_hash(a.manifest)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps({"calibration":len(plan["calibration_tasks"]),"validation":len(plan["validation_tasks"]),"categories":plan["category_counts"]},indent=2))
