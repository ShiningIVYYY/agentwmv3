#!/usr/bin/env python3
"""Bounded development collection/full episodes through the EXISTING Science.run loop."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_audited_paper_eval as runner
from experience.collection import receipts
from experience.memory import ExperienceMemory
from experience.splits import file_hash,goal_key,validate_manifest
from gates.core import RuleGate,fingerprint
from gates.generation import GateActorAdapter
from gates.runtime import GateSession
from gates.worldmodel_setup import ExistingWorldModelSetup
from evaluation.profile import validate_profile,goal_visible
from evaluation.seeds import request_seed
from evaluation.metrics import episode_metrics,costs,aggregate
from evaluation.observer import observe_wm


def main():
    p=argparse.ArgumentParser()
    for key in ("plan","model-dir","wm-checkpoint","science-python","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--manifest",default="docs/phase3/split_manifest.json")
    p.add_argument("--snapshot",default="docs/phase3/memory_snapshot.json")
    p.add_argument("--profile",default="configs/evaluation/paper_aligned_v1/scienceworld.json")
    p.add_argument("--threshold-config",default="configs/gates/provisional_full_v1.json")
    p.add_argument("--mode",choices=["calibration","episodes","canary"],required=True)
    p.add_argument("--science-url",default="http://127.0.0.1:36051");p.add_argument("--wm-url",default="http://127.0.0.1:18212")
    p.add_argument("--device",default="cuda:0");p.add_argument("--rollout-seed",type=int,default=0)
    p.add_argument("--resume",action="store_true");a=p.parse_args();began=time.perf_counter()
    profile=validate_profile(json.loads(Path(a.profile).read_text()));profile_sha=fingerprint(profile)
    if a.rollout_seed not in profile["rollout_seeds"]:raise ValueError("Unregistered rollout seed")
    plan=json.loads(Path(a.plan).read_text());manifest=json.loads(Path(a.manifest).read_text());validate_manifest(manifest,require_native=True)
    if plan["manifest_sha256"]!=file_hash(a.manifest):raise ValueError("Frozen development plan identity changed")
    allowed={r["task_id"]:r for r in manifest["partitions"]["development"]}
    calibration=a.mode in ("calibration","canary")
    rows=plan["calibration_tasks"] if calibration else plan["validation_tasks"]
    if a.mode=="canary":rows=rows[:1]
    if any(r!=allowed.get(r["task_id"]) for r in rows):raise ValueError("Only approved Development Tasks may run")
    if len(rows)>30 or len({r["goal_group"] for r in rows})!=len(rows):raise ValueError("Bounded, distinct development goal groups required")
    config=json.loads(Path(a.threshold_config).read_text())
    if not calibration and config.get("evaluation_profile_sha256")!=profile_sha:
        raise ValueError("Threshold calibration belongs to a different sampling/profile path")
    limit=2 if a.mode=="canary" else plan["calibration_max_steps"] if calibration else 20
    policies=["calibration_actor_only"] if calibration else plan["policies"]
    out=Path(a.output).resolve()
    if out.exists() and not a.resume:raise ValueError("New run ID or explicit episode-boundary --resume required")
    out.mkdir(parents=True,exist_ok=True)
    sources=[*sorted((ROOT/"evaluation").glob("*.py")),*sorted((ROOT/"gates").glob("*.py")),
        *sorted((ROOT/"internal_signals").glob("*.py")),*sorted((ROOT/"experience").glob("*.py")),
        *sorted((ROOT/"worldmodel").glob("*.py")),ROOT/"reflection/controller.py",ROOT/"reflection/prompt.txt",
        ROOT/"scripts/run_audited_paper_eval.py",Path(__file__).resolve()]
    signature={"profile_sha256":profile_sha,"plan_sha256":file_hash(a.plan),"threshold_sha256":file_hash(a.threshold_config),
        "memory_sha256":file_hash(a.snapshot),"mode":a.mode,"seed":a.rollout_seed,
        "source_hashes":{str(f.relative_to(ROOT)):file_hash(f) for f in sources}}
    if (out/"manifest.json").exists():
        old=json.loads((out/"manifest.json").read_text())
        if old["signature"]!=signature:raise ValueError("Resume configuration/source changed; use a new run")
    else:
        runner.atomic(out/"manifest.json",{"signature":signature,"tasks":rows,"policies":policies,"max_steps":limit,
            "git_base":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
            "purpose":"development_functional_validation_not_final_test","complete_episode":not calibration,
            "calibration_nll_forced":calibration})
        for source in sources:
            destination=out/"executed_source"/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    actual=runner.http("GET",a.science_url+"/profile")
    if actual["envStepLimit"]!=21 or actual["horizon"]!=20 or actual["scienceworld_version"]!="1.2.3" or actual["jar_sha256"]!="f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7":
        raise ValueError("Wrong owned environment profile/version")
    wmconfig=json.loads((ROOT/"configs/worldmodel/scienceworld.json").read_text())
    wmconfig.update(checkpoint=str(Path(a.wm_checkpoint).resolve()),source_dir=str(ROOT/"external/Word2World"),url=a.wm_url)
    runner.atomic(out/"wm_config.json",wmconfig)
    args=SimpleNamespace(internal_signals_config=str(ROOT/"configs/internal_signals/scienceworld_token_nll_v1.json"),
        tokenizer_path=a.model_dir,internal_device=a.device,science_url=a.science_url,max_steps=limit,max_tokens=512,
        disable_thinking=True,wm_backend="word2world",wm_config=str(out/"wm_config.json"),
        reflection_review_protocol="json",reflection_max_rounds=3,science_context=None,science_demo=None)
    actor=runner.Agent(args,out);actor.evaluation_profile=profile;actor.internal_actor.evaluation_left_truncation=True
    model_ready=time.perf_counter()
    memory=ExperienceMemory(json.loads(Path(a.snapshot).read_text()),manifest)
    summaries=[]
    for policy in policies:
        for row in rows:
            episode_root=out/policy/row["task_id"]
            if (episode_root/"episode.json").exists():
                saved=json.loads((episode_root/"episode.json").read_text())
                if saved["configuration_signature"]!=fingerprint(signature):raise ValueError("Completed episode signature differs")
                summaries.append(saved);continue
            episode_root.mkdir(parents=True,exist_ok=True)
            attempt=len(list(episode_root.glob("attempt-*")))+1
            if attempt>2:raise ValueError("At most one explicit infrastructure retry; inspect retained failures")
            ep=episode_root/("attempt-"+str(attempt));ep.mkdir()
            actor.out=ep;actor.total_calls=0
            gate_policy="never" if calibration else policy
            c=dict(config,policy=gate_policy,run_id=out.name+"/"+policy+"/"+row["task_id"])
            budget=0 if calibration else plan["budget"][policy]
            c.update(wm_invocation_budget_per_episode=budget,wm_forecast_budget_per_episode=4*budget)
            gate=RuleGate(c);adapter=GateActorAdapter(actor,gate.needs_internal or calibration)
            # Rebind original Agent.generate rather than accumulating wrappers.
            adapter.original=runner.Agent.generate.__get__(actor,runner.Agent)
            counters={}
            def seeds(task,step,role):
                key=(task,step);sub=counters.get(key,0)
                if role=="review":counters[key]=sub+1
                return {"request_seed":request_seed(task,step-1,a.rollout_seed,role=role,subcall=sub),
                        "namespace":"eval_request_seed_v1" if role=="draft" else "agentwmv3_review_seed_v1",
                        "rollout_seed":a.rollout_seed,"task_seed":0,"turn_index":step-1,"subcall":sub if role=="review" else 0,
                        "evaluation_profile_sha256":profile_sha}
            adapter.seed_provider=seeds;actor.generate=adapter.generate
            setup=ExistingWorldModelSetup(a.science_python,ROOT,ep)
            actor.wm_context_provider,actor.wm_token_counter=setup.context,setup.count
            active=[None]
            def event(stage,data):
                if stage=="gate_decision_before_wm":active[0]=data["decision_id"]
                with (ep/"gate_events.jsonl").open("a") as stream:stream.write(json.dumps({"stage":stage,"logged_at_ns":time.time_ns(),**data})+"\n")
            session=GateSession(c,actor,adapter,memory if gate.needs_external else None,event);actor.gate_session=session
            traces=[];episode_start=time.perf_counter()
            with receipts(runner,ep/"native_receipts.jsonl") as log,observe_wm(a.wm_url,ep/"wm_http_private.jsonl",lambda:active[0]) as requests:
                original=actor.generate
                def generated(messages,task_id,step,**kwargs):
                    reset=next(x for x in reversed(log) if x["endpoint"]=="reset")
                    if goal_key(reset["response"]["task_description"])!=row["goal_group"]:raise ValueError("Native goal differs")
                    return original(messages,task_id,step,**kwargs)
                actor.generate=generated
                def trace(task_id,record):
                    receipt=log[-1];reset=next(x for x in reversed(log) if x["endpoint"]=="reset")
                    if receipt["request"]["action"]!=record["action"] or receipt["response"]["observation"]!=record["next_observation"]:
                        raise ValueError("Executed action/feedback mismatch")
                    if receipt["response"]["actor_interaction_steps"]!=len(traces)+1:raise ValueError("Duplicate/missing native step")
                    d=record["gate_decision"]
                    if not d["decided_at_ns"]<receipt["started_at_ns"]:raise ValueError("Future input leakage")
                    observed=[x for x in requests if x["decision_id"]==d["decision_id"]]
                    if len(observed)!=record["gate_execution"]["wm_calls"]:raise ValueError("Gate/HTTP forecast counts differ")
                    if not d["call_wm"] and (observed or record["action"]!=record["gate_query"]["draft_action"]):raise ValueError("Skip invoked WM or changed action")
                    value={"task_id":task_id,"step_index":len(traces)+1,"goal":reset["response"]["task_description"],
                        "rollout_seed":a.rollout_seed,"policy":policy,"stage":a.mode,"native_receipt":receipt,
                        "evaluation_profile_sha256":profile_sha,"forced_calibration_nll":calibration,**record}
                    generated_record=record["generation_record"]
                    value["goal_fully_visible_in_model_input"]=goal_visible(actor.internal_actor.tokenizer,
                        generated_record["prompt_text"],value["goal"],generated_record.get("left_truncated_prompt_tokens",0))
                    traces.append(value)
                    with (ep/"steps.jsonl").open("a") as stream:stream.write(json.dumps(value)+"\n")
                    runner.atomic(out/"progress.json",{"state":"running","policy":policy,"task_id":task_id,
                        "episode_step":len(traces),"completed_episodes":len(summaries)})
                actor.trace=trace
                try:
                    result=runner.Science(args).run(row,actor)
                except Exception as error:
                    runner.atomic(ep/"failure.json",{"error":repr(error),"steps_completed":len(traces),"actual_model_calls":adapter.calls})
                    raise
                finally:session.close()
                native=next(x["response"] for x in reversed(log) if x["endpoint"] in ("step","reset"))
                cost=costs(traces,adapter.calls,requests,time.perf_counter()-episode_start)
                if calibration:
                    metric={"calibration_steps":len(traces),"not_a_complete_evaluation_episode":True}
                else:metric=episode_metrics(result,native,20)
                summary={"task_id":row["task_id"],"goal_group":row["goal_group"],"category":row["category"],
                    "policy":policy,"rollout_seed":a.rollout_seed,"metrics":metric,"costs":cost,
                    "configuration_signature":fingerprint(signature),"attempt":attempt,
                    "initial_observation_sha256":fingerprint(next(x["response"]["observation"] for x in log if x["endpoint"]=="reset")),
                    "first_draft_response_ids":traces[0]["generation_record"]["response_ids"] if traces else [],
                    "episode_path":str(ep.relative_to(out)),"wm_context_costs":setup.costs}
                runner.atomic(ep/"actor_calls.json",adapter.calls)
                runner.atomic(ep/"wm_requests_public.json",[{k:v for k,v in x.items() if k not in ("request","response")} for x in requests])
                runner.atomic(ep/"result.json",{k:v for k,v in result.items() if k!="steps"})
                runner.atomic(episode_root/"episode.json",summary);summaries.append(summary)
                runner.atomic(out/"episodes.json",summaries)
    final={"mode":a.mode,"completed_episodes":len(summaries),"actor_load_and_hash_seconds":model_ready-began,
        "whole_run_seconds":time.perf_counter()-began,"snapshot_unchanged":file_hash(a.snapshot)==signature["memory_sha256"],
        "outcome_based_threshold_selection":False}
    if not calibration:final["policies"]={p:aggregate([r for r in summaries if r["policy"]==p]) for p in policies}
    runner.atomic(out/"summary.json",final);runner.atomic(out/"progress.json",{"state":"finished","completed_episodes":len(summaries)})
    print(json.dumps(final,indent=2))


if __name__=="__main__":main()
