#!/usr/bin/env python3
"""Forensic Phase4 audit + real, separately labelled same-reset development probes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_audited_paper_eval import http
from experience.splits import file_hash,goal_key
from evaluation.profile import canonicalize_observation

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--science-url",default="http://127.0.0.1:36051");p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output)
    if out.exists():raise ValueError("New audit output required")
    manifest=json.loads((root/"docs/phase3/split_manifest.json").read_text());allowed={r["task_id"]:r for r in manifest["partitions"]["development"]}
    source=root/"outputs/phase4_full_v1/steps.jsonl"
    old=[json.loads(s) for s in source.read_text().splitlines()]
    cases=[t for t in old if t["gate_decision"]["call_wm"]]
    probes=[]
    choices={"sciworld_1709":["go kitchen","focus on glass jar","focus on glass cup"],
             "sciworld_349":["focus on thermometer","focus on substance called air","focus on air","focus on bee hive"]}
    for trace in cases:
        task=trace["task_id"];row=allowed[task]
        for action in choices[task]:
            eid=http("POST",a.science_url+"/create")["id"]
            try:
                reset=http("POST",a.science_url+"/reset",json={"id":eid,"data_idx":row["index"]})
                if goal_key(reset["task_description"])!=row["goal_group"]:raise ValueError("Wrong native development identity")
                same=canonicalize_observation(trace["observation"])==reset["observation"]
                acts=http("GET",a.science_url+"/admissible_actions",params={"id":eid})["admissible_actions"]
                feedback=http("POST",a.science_url+"/step",json={"id":eid,"action":action})
                probes.append({"task_id":task,"action":action,"same_canonical_initial_observation":same,
                    "exact_sdk_admissible":action in acts,"actual_feedback":feedback,
                    "purpose":"controlled_real_development_action_grounding_probe_not_natural_agent_performance"})
            finally:http("POST",a.science_url+"/close",json={"id":eid})
    records=[]
    for trace in cases:
        private=root/"outputs/phase4_full_v1"/("wm_context_"+trace["task_id"])/"context.json"
        context=json.loads(private.read_text())
        predictions=trace["reflection"]["predictions"]
        records.append({"task_id":trace["task_id"],"goal":trace["goal"],"current_observation":trace["observation"],
            "draft_decision":trace["draft_output"],"executed_history":[],"gate":trace["gate_decision"],
            "wm_initialization_path":str(private),"wm_initialization_sha256":file_hash(private),
            "wm_request_schema":{"system":"exact saved initialization (all-room info, WM-only)","user":"candidate action only",
                                 "temperature":0,"top_p":1,"max_tokens":512},
            "predictions":predictions,"reviews":trace["reflection"]["reviews"],
            "final_action":trace["action"],"actual_final_feedback":trace["next_observation"],
            "sdk_focus_actions":[x for x in trace["admissible_actions"] if x.startswith("focus")],
            "format_or_parser_mismatch":False,
            "finding":"WM predicted final failure correctly; JSON critic chose unavailable or unrecognized object, including contradictory justifications; final prediction executes by fixed protocol"})
    result={"source_trace_sha256":file_hash(source),"cases":records,"native_probes":probes,
        "probes_use_approved_development_only":True,"probes_not_inserted_into_memory":True,
        "classification":{"original_two_failures":"Reflection grounding/decision failure, not JSON parsing or WM forecast mismatch",
            "glass_jar":"not present in bathroom; no known native action for absent referent",
            "substance_called_air":"descriptive phrase is not the native referent air",
            "model_switch_causality":"not established; no controlled Qwen2.5 Actor comparison performed"}}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"cases":len(records),"controlled_native_probes":len(probes),"probes":[[x["task_id"],x["action"],x["actual_feedback"]["observation"]] for x in probes]},indent=2))
