#!/usr/bin/env python3
"""Distribution-only calibration on frozen goal-disjoint development trajectories."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_gate import quantile,correlation,ranks
from experience.splits import file_hash
from gates.core import RuleGate,GateInput,fingerprint


def calibrate(root,plan,base):
    root=Path(root);run=json.loads((root/"manifest.json").read_text())
    expected={r["task_id"]:r for r in plan["calibration_tasks"]}
    forbidden={r["task_id"] for r in plan["validation_tasks"]}
    if run["signature"]["mode"]!="calibration":raise ValueError("Only completed calibration collection is allowed")
    if not (root/"summary.json").exists():raise ValueError("Incomplete collection")
    gate=RuleGate(base);points=[];seen=set();duplicate_states=set();states=set()
    for file in sorted(root.glob("calibration_actor_only/*/attempt-*/steps.jsonl")):
        task=file.parents[1].name
        accepted=json.loads((file.parents[1]/"episode.json").read_text())
        if accepted["episode_path"]!=str(file.parent.relative_to(root)):continue
        if task not in expected or task in forbidden:raise ValueError("Calibration split leakage")
        for line in file.read_text().splitlines():
            t=json.loads(line);key=(task,t["step_index"])
            if key in seen:raise ValueError("Duplicate episode step")
            seen.add(key)
            if t["evaluation_profile_sha256"]!=run["signature"]["profile_sha256"] or t["seed_metadata"]["namespace"]!="eval_request_seed_v1":
                raise ValueError("Mixed profile or seed path")
            if t["actor_metadata"]["config_sha256"]!=base["internal_signature"]["config_sha256"]:raise ValueError("Mixed Actor config")
            local,glob,missing,errors=gate._internal(t["internal_signals"])
            state=fingerprint({"task":task,"prompt":t["generation_record"]["prompt_ids"],"response":t["generation_record"]["response_ids"]})
            if state in states:duplicate_states.add(state)
            states.add(state)
            points.append({"task_id":task,"goal_group":expected[task]["goal_group"],"category":expected[task]["category"],
                "step_index":t["step_index"],"local":local,"global":glob,"missing_span":missing,"diagnostics":errors,
                "draft_format_valid":t["decision_format_valid"]})
    if set(expected)!={p["task_id"] for p in points}:raise ValueError("Missing calibration instances")
    local=[p["local"] for p in points if p["local"] is not None];glob=[p["global"] for p in points if p["global"] is not None]
    pairs=[p for p in points if p["local"] is not None and p["global"] is not None]
    config=dict(base,policy="full",tau_local=quantile(local,.75),tau_global=quantile(glob,.75),
        calibration_status="distribution_calibrated_development_only",evaluation_profile_sha256=run["signature"]["profile_sha256"])
    config.pop("wm_invocation_budget_per_episode",None);config.pop("wm_forecast_budget_per_episode",None)
    config["budget_configuration_source"]="separate frozen development plan; not selected by NLL/outcomes"
    config["calibrated_at_ns"]=time.time_ns()
    sensitivity=[]
    for q in [.5,.6,.7,.75,.8,.85,.9,.95]:
        tl,tg=quantile(local,q),quantile(glob,q)
        # Missing-required signals request verification; this is not an outcome label.
        requested=sum(not p["draft_format_valid"] or p["global"] is None or (p["local"] is None and p["global"]>tg) or
            (p["local"] is not None and p["local"]>tl) or (p["global"] is not None and p["global"]>tg) for p in points)
        sensitivity.append({"quantile":q,"tau_local":tl,"tau_global":tg,"internal_requested":requested,
                            "fraction_before_external_and_budget":requested/len(points)})
    loo=[]
    for task in sorted(expected):
        subset=[p for p in points if p["task_id"]!=task]
        loo.append({"omitted_task":task,"local":quantile([p["local"] for p in subset if p["local"] is not None],.75),
                    "global":quantile([p["global"] for p in subset if p["global"] is not None],.75)})
    outliers={}
    for name,values in [("local",local),("global",glob)]:
        a,b=quantile(values,.25),quantile(values,.75);iqr=b-a
        outliers[name]={"below_or_above_1_5_iqr":sum(x<a-1.5*iqr or x>b+1.5*iqr for x in values),"removed":0,
            "min":min(values),"max":max(values),"median":quantile(values,.5),"q75":quantile(values,.75),"q90":quantile(values,.9)}
    report={"schema":"phase5-distribution-calibration-v1","steps":len(points),"task_ids":len(expected),
        "goal_groups":len({p["goal_group"] for p in points}),"missing_steps":sum(p["missing_span"] for p in points),
        "unique_prompt_response_states":len(states),"repeated_prompt_response_states":len(duplicate_states),
        "repeated_steps_retained":"real per-step workload; dependence disclosed, not independent evidence",
        "pearson":correlation([p["local"] for p in pairs],[p["global"] for p in pairs]),
        "spearman":correlation(ranks([p["local"] for p in pairs]),ranks([p["global"] for p in pairs])),
        "quantile_method":"linear (n-1)*q; q=.75 fixed before data; strict >",
        "tau_local":config["tau_local"],"tau_global":config["tau_global"],"sensitivity":sensitivity,"outliers":outliers,
        "leave_one_episode_out_ranges":{"local":[min(x["local"] for x in loo),max(x["local"] for x in loo)],
                                        "global":[min(x["global"] for x in loo),max(x["global"] for x in loo)]},
        "outcomes_read_for_selection":False,"test_data_used":False,"validation_goal_groups_disjoint":True,
        "status":"distribution_calibrated; stability/generalization/scientific_utility_not_established",
        "threshold_frozen_at_ns":config["calibrated_at_ns"],
        "source_manifest_sha256":file_hash(root/"manifest.json"),"profile_sha256":run["signature"]["profile_sha256"],
        "budget_not_selected_from_distribution":True,"points":points}
    return config,report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--plan",required=True)
    p.add_argument("--base-config",default="configs/gates/provisional_full_v1.json");p.add_argument("--output-config",required=True);p.add_argument("--output-report",required=True);a=p.parse_args()
    if Path(a.output_config).exists() or Path(a.output_report).exists():raise ValueError("New frozen calibration outputs required")
    start=time.perf_counter();cfg,report=calibrate(a.run_dir,json.loads(Path(a.plan).read_text()),json.loads(Path(a.base_config).read_text()))
    report["analysis_seconds"]=time.perf_counter()-start
    for path,value in [(a.output_config,cfg),(a.output_report,report)]:Path(path).write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="points"},indent=2))
