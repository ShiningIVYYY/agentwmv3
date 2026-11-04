"""Post-run diagnostics. Never feed outcome labels back into Gate calibration."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import statistics
import time
from evaluation.metrics import aggregate
from experience.schema import ExternalQuery
from experience.memory import ExperienceMemory


def explicit_failure(feedback):
    text=feedback.strip().casefold()
    return ("no known action matches" in text or text.startswith("unknown action") or
            text=="the door is not open." or text.startswith("i'm not sure how to use") or
            text=="the door is not something that can be activated.")


def completed_traces(root):
    root=Path(root)
    episodes=json.loads((root/"episodes.json").read_text())
    for episode in episodes:
        for line in (root/episode["episode_path"]/"steps.jsonl").read_text().splitlines():
            yield episode,json.loads(line)


def internal_diagnostic(calibration,threshold):
    rows=[t for e,t in completed_traces(calibration)]
    valid=[t for t in rows if t["internal_signals"]["local_uncertainty"] is not None]
    groups={"above_local_threshold":[],"at_or_below_local_threshold":[]}
    for t in valid:
        key="above_local_threshold" if t["internal_signals"]["local_uncertainty"]>threshold["tau_local"] else "at_or_below_local_threshold"
        if t["action"]!=t["gate_query"]["draft_action"]:raise ValueError("Actor-only diagnostic revised its draft")
        groups[key].append(explicit_failure(t["next_observation"]))
    return {"source":"scored Actor-only calibration trajectories; all drafts actually executed", "steps":len(rows),
        "outcome_analysis_after_threshold_freeze":True,"used_to_select_threshold":False,
        "label_definition":"explicit native failure text only; other feedback is NOT proven correct action",
        "groups":{k:{"n":len(v),"explicit_failures":sum(v),"failure_fraction":sum(v)/len(v) if v else None} for k,v in groups.items()},
        "interpretation":"in-sample, episode-dependent preliminary association, not calibrated error detection"}


def external_diagnostic(calibration,evaluation,memory,base_memory,threshold):
    shadow=[];natural_sufficient=[]
    for episode,t in completed_traces(calibration):
        q=ExternalQuery(**t["gate_query"])
        # The old snapshot existed before these real decisions. New look rule
        # is retrospective only here; never pretend it was online at that time.
        old=base_memory.evaluate(q)
        retrospective=memory.evaluate(replace(q,available_at_ns=time.time_ns()))
        low=t["internal_signals"]["local_uncertainty"] is not None and t["internal_signals"]["local_uncertainty"]<=threshold["tau_local"]
        if old["conflict"]=="true" and low:
            shadow.append({"task_id":t["task_id"],"step_index":t["step_index"],"action":t["action"],
                "local":t["internal_signals"]["local_uncertainty"],"actual_feedback":t["next_observation"],
                "signals":old,"real_native_explicit_failure":explicit_failure(t["next_observation"]),
                "analysis_only_not_actual_gate_call":True})
        if retrospective["sufficiency"]=="true":
            natural_sufficient.append({"task_id":t["task_id"],"step_index":t["step_index"],"decision":t["draft_output"],
                "actual_feedback":t["next_observation"],"signals":retrospective,"retrospective_new_snapshot":True})
    live=[t for e,t in completed_traces(evaluation) if t["policy"]=="full"]
    return {"old_snapshot_low_nll_conflict_shadow_cases":shadow,"new_snapshot_natural_sufficiency_retro_cases":natural_sufficient,
        "live_full_steps":len(live),"live_full_coverage_true":sum(t["external_signals"]["coverage"]=="true" for t in live),
        "live_full_conflict_true":sum(t["external_signals"]["conflict"]=="true" for t in live),
        "live_full_sufficiency_true":sum(t["external_signals"]["sufficiency"]=="true" for t in live),
        "memory_correction_and_sufficiency_skip_remained_disabled":True,
        "unknown_not_assumed_no_conflict":True,"new_memory_not_backdated_into_calibration":True}


def wm_utility(evaluation):
    rows=[]
    for episode,t in completed_traces(evaluation):
        if not t["gate_decision"]["call_wm"]:continue
        predictions=[p for p in t["reflection"]["predictions"] if p["action"]==t["action"]]
        final=predictions[-1] if predictions else None
        rows.append({"task_id":t["task_id"],"policy":t["policy"],"step_index":t["step_index"],
            "draft":t["gate_query"]["draft_action"],"executed":t["action"],"changed":t["action"]!=t["gate_query"]["draft_action"],
            "actual_feedback":t["next_observation"],"actual_final_explicit_failure":explicit_failure(t["next_observation"]),
            "forecast_available":bool(final and final.get("prediction_available",True) and final.get("observation_complete",True)),
            "final_forecast_text":final["observation"] if final else None,
            "forecast_exact_text_match":final["observation"].strip()==t["next_observation"].strip() if final else None,
            "draft_outcome_observed":t["action"]==t["gate_query"]["draft_action"]})
    return {"calls":len(rows),"changed_actions":sum(r["changed"] for r in rows),
        "revised_actions_with_real_failure_feedback":sum(r["changed"] and r["actual_final_explicit_failure"] for r in rows),
        "unavailable_or_incomplete_final_predictions":sum(not r["forecast_available"] for r in rows),
        "no_same_state_call_skip_counterfactual_rollout":True,
        "causal_wm_benefit_or_harm_not_established":True,"rows":rows}
