"""Native instance metrics and complete model-usage accounting, without reward shaping."""
import math


def episode_metrics(result, native_final, max_steps=20):
    score=float(result["score"])
    if not math.isfinite(score):raise ValueError("Nonfinite native Task Score")
    n=len(result["steps"])
    native_done=bool(native_final.get("native_done",native_final["done"]))
    horizon=(n>=max_steps and not native_done)
    if not native_done and n<max_steps:raise ValueError("Incomplete episode cannot enter evaluation aggregate")
    return {"task_score":score,"success_score_ge100":score>=100,"environment_steps":n,
        "native_terminated":native_done,"horizon_truncated":horizon,
        "termination_reason":"native_terminal" if native_done else "interaction_horizon"}


def costs(traces,actor_calls,wm_requests,wall_seconds):
    models=[c["usage"] for c in actor_calls]+[r["usage"] for r in wm_requests]
    if any(not isinstance(u,dict) or any(k not in u for k in ("prompt_tokens","completion_tokens","total_tokens")) for u in models):
        raise ValueError("Missing actual usage; cannot fabricate zero model costs")
    role=lambda name:[c for c in actor_calls if c["role"]==name]
    usage=lambda xs:{k:sum(x[k] for x in xs) for k in ("prompt_tokens","completion_tokens","total_tokens")}
    draft,review=role("draft"),role("review")
    return {"agent_calls":len(actor_calls),"draft_calls":len(draft),"reflection_revision_calls":len(review),
        "wm_calls":len(wm_requests),"total_model_calls":len(models),"judge_calls":0,"router_model_calls":0,
        "draft_usage":usage([c["usage"] for c in draft]),"reflection_revision_usage":usage([c["usage"] for c in review]),
        "wm_usage":usage([r["usage"] for r in wm_requests]),"total_usage":usage(models),
        "agent_generation_seconds":sum(c["cost"]["generation_seconds"] for c in actor_calls),
        "reflection_revision_generation_seconds":sum(c["cost"]["generation_seconds"] for c in review),
        "internal_signal_seconds":sum(c["cost"]["extraction_seconds"] for c in actor_calls),
        "retrieval_calls":sum(t.get("external_signals",{}).get("retrieval_calls",0) for t in traces),
        "retrieval_seconds":sum(t.get("external_signals",{}).get("retrieval_latency",0) for t in traces),
        "external_judge_seconds":sum(t.get("external_signals",{}).get("judge_latency",0) for t in traces),
        "gate_rule_seconds":sum(t["gate_decision"]["gate_latency"] for t in traces),
        "signals_evidence_gate_seconds":sum(t["signal_and_gate_seconds"] for t in traces),
        "wm_prediction_http_seconds":sum(r["seconds"] for r in wm_requests),
        "wm_context_setup_seconds":sum(t["gate_execution"].get("wm_context_setup_seconds",0) for t in traces),
        "episode_total_latency_seconds":wall_seconds,"extra_backbone_forward_passes":0,
        "gate_call_steps":sum(t["gate_decision"]["call_wm"] for t in traces),
        "gate_skip_steps":sum(not t["gate_decision"]["call_wm"] for t in traces),
        "budget_blocked_steps":sum(t["gate_decision"]["verification_unavailable"] for t in traces),
        "estimated_monetary_cost":None,"pricing_status":"no_reliable_tariff",
        "actor_peak_allocated_bytes":max((c["cost"]["peak_allocated_bytes"] or 0 for c in actor_calls),default=0)}


def aggregate(rows):
    if not rows:raise ValueError("No complete episodes")
    keys=[(r["task_id"],r["rollout_seed"]) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError("Duplicate task/seed results")
    n=len(rows)
    return {"instances":n,"mean_task_score":sum(r["metrics"]["task_score"] for r in rows)/n,
        "success_rate":sum(r["metrics"]["success_score_ge100"] for r in rows)/n,
        "average_environment_steps":sum(r["metrics"]["environment_steps"] for r in rows)/n,
        "total_environment_steps":sum(r["metrics"]["environment_steps"] for r in rows),
        "total_online_model_calls":sum(r["costs"]["total_model_calls"] for r in rows),
        "total_tokens":sum(r["costs"]["total_usage"]["total_tokens"] for r in rows),
        "total_wm_calls":sum(r["costs"]["wm_calls"] for r in rows),
        "total_latency_seconds":sum(r["costs"]["episode_total_latency_seconds"] for r in rows)}
