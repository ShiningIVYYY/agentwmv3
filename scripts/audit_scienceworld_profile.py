#!/usr/bin/env python3
"""Metadata-only pinned identity/SDK/protocol audit. No final-test rollout."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.profile import validate_profile
from experience.splits import file_hash,digest
from scripts.run_audited_paper_eval import http

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference-root",required=True);p.add_argument("--test-ids",required=True)
    p.add_argument("--science-python",required=True);p.add_argument("--science-url",default="http://127.0.0.1:36051")
    p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];reference=Path(a.reference_root)
    profile=validate_profile(json.loads((root/"configs/evaluation/paper_aligned_v1/scienceworld.json").read_text()))
    commit=subprocess.check_output(["git","-C",str(reference),"rev-parse","HEAD"],text=True).strip()
    if commit!=profile["official_code_revision"]:raise ValueError("Unpinned official source")
    if file_hash(a.test_ids)!=profile["test_source_sha256"]:raise ValueError("Wrong200 task source")
    spec=importlib.util.spec_from_file_location("protocol",reference/"agent_system/scienceworld_protocol.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads((root/"docs/phase3/split_manifest.json").read_text())
    ids=[r["item_id"] for r in json.loads(Path(a.test_ids).read_text())]
    if ids!=[r["task_id"] for r in manifest["partitions"]["test"]]:raise ValueError("Test order/identity mismatch")
    identities=[]
    for row,item in zip(manifest["partitions"]["test"],ids):
        resolved=module.resolve_item(item)
        if (resolved["task_name"],resolved["variation_id"])!=(row["taskName"],row["variationIdx"]):raise ValueError("Task mapping mismatch")
        identities.append({"task_id":item,"task_name":row["taskName"],"variation_id":row["variationIdx"],"goal_sha256":row["goal_sha256"]})
    code='import json;from scienceworld import ScienceWorldEnv;e=ScienceWorldEnv();x=[(k,n,e.get_max_variations(n)) for k,n in e.tasks.items() if k not in {"5-1","5-2","9-1","9-2","9-3","10-1","10-2"}];e.close();print(json.dumps(x))'
    catalog=json.loads(subprocess.check_output([a.science_python,"-c",code],text=True))
    if catalog!=[list(r[:3]) for r in module.TASK_CATALOG]:raise ValueError("Current SDK differs from pinned catalog")
    live=http("GET",a.science_url+"/profile")
    if live["envStepLimit"]!=21 or live["horizon"]!=20:raise ValueError("Wrong current environment limit")
    old=json.loads((root/"docs/phase3/native_goal_identity.json").read_text())
    native={r["task_id"]:r["goal"] for r in old["records"]}
    if not all(digest(native[r["task_id"]])==r["goal_sha256"] for r in identities):raise ValueError("Native goal identity mismatch")
    result={"paper":"https://arxiv.org/pdf/2610.19789","source_commit":commit,"ordered_tasks":len(ids),
        "task_ids_order_and_variations_match":True,"all_prior_native_goal_hashes_match":True,
        "current_sdk_catalog_matches":True,"live_profile":live,
        "formal_parameters":profile,"test_interaction_rollouts":0,
        "jar_identity":"same recorded deployment JAR; official independent JAR hash not published/attested",
        "identities":identities}
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v for k,v in result.items() if k not in ("identities","formal_parameters")},indent=2))
