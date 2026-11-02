#!/usr/bin/env python3
"""Causal/evidence/cost replay of actual saved traces; never counterfactual task scores."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gates.core import ExternalInput,GateInput,RuleGate,fingerprint
from gates.evidence import resolve_external
from experience.schema import ExternalQuery
from experience.memory import ExperienceMemory
from experience.splits import file_hash,validate_manifest
from internal_signals.decision import messages


def audit(run,manifest,snapshot):
    root=Path(run);m=json.loads(Path(manifest).read_text());validate_manifest(m,require_native=True)
    memory=ExperienceMemory(json.loads(Path(snapshot).read_text()),m)
    config=json.loads((root/"gate_config.json").read_text());gate=RuleGate(config)
    traces=[json.loads(s) for s in (root/"steps.jsonl").read_text().splitlines()]
    allowed={r["task_id"] for r in m["partitions"]["development"]}
    before={};history={};calls=skips=0
    for t in traces:
        if t["task_id"] not in allowed:raise ValueError("Non-development rollout")
        saved=t["gate_decision"];raw=t["gate_input"];raw=dict(raw)
        raw["external"]=ExternalInput(**raw["external"])
        ctx=GateInput(**raw)
        replay=gate.decide(ctx)
        for k,v in saved.items():
            if k in replay and k!="gate_latency" and replay[k]!=v:raise ValueError("Gate replay differs: "+k)
        if saved["input_sha256"]!=fingerprint(asdict(ctx)):raise ValueError("Gate input hash changed")
        q=ExternalQuery(**t["gate_query"])
        if gate.needs_external:
            external=memory.evaluate(q)
            for k in ("coverage","conflict","sufficiency","matched_experience_ids","supporting_evidence_ids","query_input_sha256"):
                if external[k]!=t["external_signals"][k]:raise ValueError("External replay mismatch")
            resolved=asdict(resolve_external(external,q,memory))
            expected=asdict(ctx.external)
            if json.dumps(resolved,sort_keys=True)!=json.dumps(expected,sort_keys=True):raise ValueError("Evidence binding changed")
        if gate.needs_internal:
            if t["internal_signals"]["backend"]!=config["internal_signature"]["backend"]:raise ValueError("Internal scoring changed")
        elif "internal_signals" in t:raise ValueError("Unused baseline computed internal scores")
        if not gate.needs_external and "external_signals" in t:raise ValueError("Unused baseline retrieved experience")
        receipt=t["native_receipt"]
        if t["action"]!=receipt["request"]["action"] or t["next_observation"]!=receipt["response"]["observation"]:
            raise ValueError("Native receipt/action/feedback differ")
        if not q.available_at_ns<=saved["decided_at_ns"]<receipt["started_at_ns"]<receipt["available_at_ns"]:
            raise ValueError("Current/future feedback entered Gate")
        steps=history.setdefault(t["task_id"],[])
        if messages(t["goal"],t["observation"],steps,t["history_length"])!=t["messages"]:raise ValueError("Actor prompt changed")
        steps.append(t)
        execution=t["gate_execution"];n=execution["wm_calls"]
        if saved["call_wm"]:
            calls+=1
            if not 1<=n<=4 or len(t["reflection"]["predictions"])!=n:raise ValueError("Wrong forecast count")
            if not saved["decided_at_ns"]<execution["started_at_ns"]<execution["finished_at_ns"]<receipt["started_at_ns"]:
                raise ValueError("WM results influenced Gate or outcome preceded prediction")
        else:
            skips+=1
            if n or "reflection" in t:raise ValueError("Skipped step made a WM/review call")
            if t["action"]!=q.draft_action:raise ValueError("Skip modified original action")
        remaining=before.setdefault(t["task_id"],(config["wm_invocation_budget_per_episode"],config["wm_forecast_budget_per_episode"]))
        if (saved["budget_remaining"],saved["forecast_budget_remaining"])!=remaining:raise ValueError("Budget state drift")
        after=(remaining[0]-int(saved["call_wm"]),remaining[1]-n)
        if min(after)<0:raise ValueError("Budget overshot")
        if (execution["budget_after"]["invocations_remaining"],execution["budget_after"]["forecasts_remaining"])!=after:
            raise ValueError("Incorrect reservation settlement")
        before[t["task_id"]]=after
    run_manifest=json.loads((root/"manifest.json").read_text());changes=[]
    for name,sha in run_manifest["source_hashes"].items():
        if file_hash(ROOT/name)!=sha:
            archived=root/"executed_source"/name
            if not archived.exists() or file_hash(archived)!=sha:raise ValueError("Executed source unavailable: "+name)
            changes.append(name)
    return {"steps":len(traces),"call_steps":calls,"skip_steps":skips,"gate_replay_exact":True,
        "evidence_binding_replay_exact":True,"actor_prompt_unchanged":True,"native_receipts_match":True,
        "all_gate_decisions_before_wm_and_environment":True,"no_skip_predictions_or_reviews":True,
        "budgets_no_overshoot":True,"source_changes_with_archived_executed_versions":changes,
        "no_task_benefit_or_counterfactual_performance_claim":True}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--snapshot",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    result=audit(a.run_dir,a.manifest,a.snapshot);Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
