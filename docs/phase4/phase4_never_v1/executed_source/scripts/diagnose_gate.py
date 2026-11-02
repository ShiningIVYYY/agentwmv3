#!/usr/bin/env python3
"""Small development-only real learned-WM Call/Skip diagnostic. No efficacy evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_audited_paper_eval as runner
from experience.collection import receipts
from experience.memory import ExperienceMemory
from experience.splits import file_hash,goal_key,validate_manifest
from gates.core import POLICIES,RuleGate,fingerprint
from gates.generation import GateActorAdapter,unscored_generate
from gates.runtime import GateSession
from gates.worldmodel_setup import ExistingWorldModelSetup


def main():
    p=argparse.ArgumentParser()
    for k in ("manifest","snapshot","gate-config","model-dir","wm-checkpoint","science-python","output"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--policy",choices=POLICIES,default="full");p.add_argument("--task-id",action="append",required=True)
    p.add_argument("--steps",type=int,default=5);p.add_argument("--device",default="cuda:0")
    p.add_argument("--science-url",default="http://127.0.0.1:36001");p.add_argument("--wm-url",default="http://127.0.0.1:18212")
    p.add_argument("--allow-provisional",action="store_true");p.add_argument("--validate-unscored-parity",action="store_true")
    a=p.parse_args();start=time.perf_counter()
    if not 1<=a.steps<=5 or len(a.task_id)>3 or len(set(a.task_id))!=len(a.task_id):raise ValueError("Small diagnostic only")
    m=json.loads(Path(a.manifest).read_text());validate_manifest(m,require_native=True)
    allowed={r["task_id"]:r for r in m["partitions"]["development"]}
    rows=[allowed[t] for t in a.task_id]  # No test/train ID can enter this diagnostic.
    config=json.loads(Path(a.gate_config).read_text());config["policy"]=a.policy
    if config["calibration_status"]!="provisional_functional_only" or not a.allow_provisional:
        raise ValueError("Explicit provisional diagnostic acceptance required; no validated threshold claim")
    out=Path(a.output).resolve()
    if out.exists():raise ValueError("New Run ID/output required; no implicit retries")
    out.mkdir(parents=True);config["run_id"]=out.name
    gate=RuleGate(config)
    memory=ExperienceMemory(json.loads(Path(a.snapshot).read_text()),m) if gate.needs_external else None
    wmconfig=json.loads((ROOT/"configs/worldmodel/scienceworld.json").read_text())
    wmconfig.update(checkpoint=str(Path(a.wm_checkpoint).resolve()),source_dir=str(ROOT/"external/Word2World"),url=a.wm_url,gpu=6)
    runner.atomic(out/"wm_config.json",wmconfig);runner.atomic(out/"gate_config.json",config)
    runtime_code='import json,importlib.metadata;from scienceworld.constants import JAR_PATH;import hashlib;print(json.dumps({"scienceworld":importlib.metadata.version("scienceworld"),"jar_sha256":hashlib.sha256(open(JAR_PATH,"rb").read()).hexdigest()}))'
    runtime=json.loads(subprocess.check_output([a.science_python,"-c",runtime_code],text=True))
    if runtime!={"scienceworld":"1.2.3","jar_sha256":"f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7"}:raise ValueError("Simulator differs")
    args=SimpleNamespace(internal_signals_config=str(ROOT/"configs/internal_signals/scienceworld_token_nll_v1.json"),
        tokenizer_path=a.model_dir,internal_device=a.device,science_url=a.science_url,max_steps=a.steps,max_tokens=512,
        disable_thinking=True,wm_backend="word2world",wm_config=str(out/"wm_config.json"),
        reflection_review_protocol="json",reflection_max_rounds=3,science_context=None,science_demo=None)
    sources=[*sorted((ROOT/"gates").glob("*.py")),*sorted((ROOT/"internal_signals").glob("*.py")),
        *sorted((ROOT/"experience").glob("*.py")),ROOT/"scripts/run_audited_paper_eval.py",ROOT/"worldmodel/runtime.py",Path(__file__).resolve()]
    runner.atomic(out/"manifest.json",{"purpose":"targeted_previously_exposed_development_functional_smoke_not_benchmark",
        "tasks":rows,"max_steps":a.steps,"environment":runtime,"gate_config":config,
        "source_hashes":{str(f.relative_to(ROOT)):file_hash(f) for f in sources},
        "input_hashes":{k:file_hash(getattr(a,k)) for k in ("manifest","snapshot","gate_config")},
        "git_base":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        "paper_protocol_sha256":file_hash(ROOT/"docs/PAPER_ALIGNED_EVAL.md"),
        "internal_config_sha256":file_hash(args.internal_signals_config),"no_experience_updates":True})
    actor=runner.Agent(args,out);adapter=GateActorAdapter(actor,gate.needs_internal);actor.generate=adapter.generate
    setup=ExistingWorldModelSetup(a.science_python,ROOT,out)
    actor.wm_token_counter,actor.wm_context_provider=setup.count,setup.context
    def event(stage,data):
        with (out/"gate_events.jsonl").open("a") as f:f.write(json.dumps({"stage":stage,"logged_at_ns":time.time_ns(),**data})+"\n")
    session=GateSession(config,actor,adapter,memory,event);actor.gate_session=session
    original=actor.generate;traces=[];validation=[]
    with receipts(runner,out/"native_receipts.jsonl") as log:
        def generate(messages,task_id,step,**kwargs):
            reset=next(r for r in reversed(log) if r["endpoint"]=="reset")
            if goal_key(reset["response"]["task_description"])!=allowed[task_id]["goal_group"]:raise ValueError("Native development identity mismatch")
            text,meta=original(messages,task_id,step,**kwargs)
            if a.validate_unscored_parity and not validation and not adapter.reviewing:
                other,othermeta=unscored_generate(actor.internal_actor,messages,"validation")
                same=(text==other and meta["generation_record"]["response_ids"]==othermeta["generation_record"]["response_ids"])
                validation.append({"same_prompt_and_sampling":True,"response_ids_identical":same,"extra_actor_calls":1,
                                   "usage":othermeta["usage"],"cost":othermeta["internal_cost"]})
                runner.atomic(out/"unscored_parity.json",validation)
                if not same:raise ValueError("Scored/unscored generation diverged; cannot claim prompt/sampling parity")
            return text,meta
        actor.generate=generate
        def trace(task_id,record):
            receipt=log[-1];reset=next(r for r in reversed(log) if r["endpoint"]=="reset")
            if receipt["request"]["action"]!=record["action"] or receipt["response"]["observation"]!=record["next_observation"]:
                raise ValueError("Native action/feedback mismatch")
            d=record["gate_decision"];execution=record["gate_execution"]
            if not d["decided_at_ns"]<receipt["started_at_ns"]:raise ValueError("Future action leakage")
            if d["call_wm"]!=(execution["wm_calls"]>0):raise ValueError("Gate and actual WM branch differ")
            if execution["branch"]=="call" and not d["decided_at_ns"]<execution["started_at_ns"]<receipt["started_at_ns"]:
                raise ValueError("WM influenced current Gate decision")
            saved={"task_id":task_id,"step_index":1+sum(x["task_id"]==task_id for x in traces),
                "goal":reset["response"]["task_description"],"native_reward":receipt["response"].get("reward"),
                "native_done":receipt["response"]["done"],"native_receipt":receipt,**record}
            traces.append(saved)
            with (out/"steps.jsonl").open("a") as f:f.write(json.dumps(saved)+"\n")
            runner.atomic(out/"progress.json",{"state":"running","completed_steps":len(traces),"task_id":task_id})
        actor.trace=trace
        try:
            for row in rows:runner.atomic(out/(row["task_id"]+".json"),runner.Science(args).run(row,actor))
        finally:session.close()
    predictions=[p for t in traces for p in t.get("reflection",{}).get("predictions",[]) if not p.get("cached",False)]
    usage=[c["usage"] for c in adapter.calls]+[(p.get("usage") or {}) for p in predictions]
    actor_seconds=sum(c["cost"]["generation_seconds"] for c in adapter.calls)
    internal_seconds=sum(c["cost"]["extraction_seconds"] for c in adapter.calls)
    external_gate_seconds=sum(t["signal_and_gate_seconds"] for t in traces)
    wm_seconds=sum(p["seconds"] for p in predictions)
    summary={"purpose":"engineering_only_no_task_benefit_claim","policy":a.policy,"steps":len(traces),
        "gate_call_steps":sum(t["gate_decision"]["call_wm"] for t in traces),
        "gate_skip_steps":sum(not t["gate_decision"]["call_wm"] for t in traces),
        "budget_blocked_steps":sum(t["gate_decision"]["verification_unavailable"] for t in traces),
        "actual_wm_calls":len(predictions),"draft_actor_calls":sum(c["role"]=="draft" for c in adapter.calls),
        "review_actor_calls":sum(c["role"]=="review" for c in adapter.calls),"total_online_model_calls":len(adapter.calls)+len(predictions),
        "online_usage":{k:sum(u.get(k,0) for u in usage) for k in ("prompt_tokens","completion_tokens","total_tokens")},
        "draft_usage":{k:sum(c["usage"][k] for c in adapter.calls if c["role"]=="draft") for k in ("prompt_tokens","completion_tokens","total_tokens")},
        "review_usage":{k:sum(c["usage"][k] for c in adapter.calls if c["role"]=="review") for k in ("prompt_tokens","completion_tokens","total_tokens")},
        "wm_usage":{k:sum((p.get("usage") or {}).get(k,0) for p in predictions) for k in ("prompt_tokens","completion_tokens","total_tokens")},
        "actor_generation_seconds":actor_seconds,"internal_extraction_seconds":internal_seconds,
        "internal_extraction_calls":sum(c["role"]=="draft" for c in adapter.calls) if gate.needs_internal else 0,
        "retrieval_calls":sum(t.get("external_signals",{}).get("retrieval_calls",0) for t in traces),
        "retrieval_seconds":sum(t.get("external_signals",{}).get("retrieval_latency",0) for t in traces),
        "external_judge_seconds":sum(t.get("external_signals",{}).get("judge_latency",0) for t in traces),
        "external_judge_model_calls":0,"gate_rule_seconds":sum(t["gate_decision"]["gate_latency"] for t in traces),
        "external_evidence_gate_total_seconds":external_gate_seconds,"wm_prediction_seconds":wm_seconds,
        "action_review_generation_seconds":sum(c["cost"]["generation_seconds"] for c in adapter.calls if c["role"]=="review"),
        "total_instrumented_inference_seconds":actor_seconds+internal_seconds+external_gate_seconds+wm_seconds,
        "wm_context_setup_seconds":sum(c["seconds"] for c in setup.costs),"whole_driver_seconds":time.perf_counter()-start,
        "max_actor_peak_allocated_bytes":max(c["cost"]["peak_allocated_bytes"] for c in adapter.calls),
        "validation_extra_calls":sum(v["extra_actor_calls"] for v in validation),"validation_cost":validation,
        "offline_memory_synthesis":"unchanged Phase3 snapshot; offline Actor collection/probes/synthesis reported separately",
        "calibration_status":config["calibration_status"],"memory_correction":False,"extra_backbone_forward_passes":0,
        "missing_span_count":sum(t.get("internal_signals",{}).get("missing_span",False) for t in traces),
        "invalid_draft_format_count":sum(not t["decision_format_valid"] for t in traces),"memory_snapshot_unchanged":file_hash(a.snapshot)}
    runner.atomic(out/"summary.json",summary);runner.atomic(out/"actor_calls.json",adapter.calls)
    runner.atomic(out/"progress.json",{"state":"finished","steps":len(traces)})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
