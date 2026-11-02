#!/usr/bin/env python3
"""Development-only descriptive quantiles; never reads rewards/success to choose rules."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gates.core import RuleGate, fingerprint
from experience.splits import file_hash, goal_key, validate_manifest


def quantile(values, q):
    if not values or not 0 <= q <= 1: raise ValueError("Nonempty data and quantile in [0,1] required")
    x = sorted(values); p = (len(x) - 1) * q; lo = int(p); hi = min(lo + 1, len(x) - 1)
    return x[lo] + (p - lo) * (x[hi] - x[lo])


def correlation(x, y):
    if len(x) < 2: return None
    a, b = statistics.mean(x), statistics.mean(y)
    num = sum((i-a)*(j-b) for i,j in zip(x,y))
    den = math.sqrt(sum((i-a)**2 for i in x) * sum((j-b)**2 for j in y))
    return num / den if den else None


def ranks(values):
    result = []
    ordered = sorted(values)
    for x in values:
        positions = [i+1 for i,v in enumerate(ordered) if v == x]
        result.append(statistics.mean(positions))
    return result


def calibrate(paths, manifest, actor_config, q=.75):
    validate_manifest(manifest, require_native=True)
    allowed = {r["task_id"]: r for r in manifest["partitions"]["development"]}
    test_ids = {r["task_id"] for r in manifest["partitions"]["test"]}
    signature = {k: actor_config[k] for k in ("backend", "dtype", "model_revision")}
    signature.update(method="fixed_field_mean_token_nll_v1", scoring_path="raw_generation_logits_before_temperature",
                     score_unit="nats_per_token", config_sha256=fingerprint(actor_config))
    base = {"schema": "training-free-gate-v1", "policy": "full", "tau_local": 0, "tau_global": 0,
        "internal_signature": signature, "allow_confirmed_action_skip": False, "memory_correction": False,
        "missing_policy": "global_then_call", "wm_invocation_budget_per_episode": 1,
        "wm_forecast_budget_per_episode": 4, "max_forecasts_per_invocation": 4,
        "review_protocol": "json", "reflection_max_rounds": 3, "budget_unit": "verification_steps_plus_actual_forecast_cap",
        "calibration_status": "provisional_functional_only", "run_id": "set_by_runtime"}
    gate = RuleGate(base)
    seen, points, rejected = set(), [], []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            task = row["task_id"]
            if task in test_ids or task not in allowed: raise ValueError("Only approved Development IDs may calibrate")
            # Explicit causal data allowlist. Outcomes/score/success are not read.
            message = row["messages"][0]["content"]
            goal = message.split("\nTask: ", 1)[1].split("\nRecent executed history:", 1)[0]
            if goal_key(goal) != allowed[task]["goal_group"]: raise ValueError("Development goal identity mismatch")
            if row["actor_metadata"]["config_sha256"] != signature["config_sha256"]: raise ValueError("Mixed generation configs")
            local, glob, _, errors = gate._internal(row["internal_signals"])
            key = fingerprint({"task_id": task, "prompt_ids": row["generation_record"]["prompt_ids"],
                               "response_ids": row["generation_record"]["response_ids"]})
            if key in seen: raise ValueError("Duplicate calibration decision")
            seen.add(key)
            if local is None or glob is None:
                rejected.append({"task_id": task, "errors": errors}); continue
            points.append({"task_id": task, "goal_group": allowed[task]["goal_group"], "local": local, "global": glob})
    if not points: raise ValueError("No valid development signals")
    local = [p["local"] for p in points]; glob = [p["global"] for p in points]
    tl, tg = quantile(local,q), quantile(glob,q)
    base.update(tau_local=tl,tau_global=tg)
    report = {"schema": "gate-calibration-report-v1", "status": "provisional_functional_only",
        "not_stable_calibration": True, "selection_uses_task_success_or_reward": False,
        "quantile": q, "quantile_definition": "linear interpolation at (n-1)*q; call on strict >",
        "valid_steps": len(points), "independent_task_ids": len({p["task_id"] for p in points}),
        "independent_goal_groups": len({p["goal_group"] for p in points}),
        "includes_previously_exposed_targeted_diagnostics": True,
        "minimum_future_diagnostic_target": {"steps":300,"tasks":30,"note":"data volume target alone is not scientific validation"},
        "invalid_or_missing": rejected, "tau_local": tl, "tau_global": tg,
        "local_quantiles": {str(v):quantile(local,v) for v in [0,.25,.5,.75,.9,1]},
        "global_quantiles": {str(v):quantile(glob,v) for v in [0,.25,.5,.75,.9,1]},
        "pearson_local_global": correlation(local,glob), "spearman_local_global":correlation(ranks(local),ranks(glob)),
        "illustrative_internal_union_calls":sum(a>tl or b>tg for a,b in zip(local,glob)),
        "budget_selection":"preset one invocation/four forecast slots per diagnostic episode; no outcome optimization",
        "future_predeclared_invocation_budget_grid":[0,1,2], "source_sha256":{str(p):file_hash(p) for p in paths},
        "internal_signature":signature, "points":points}
    return base,report


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--trace",action="append",required=True)
    p.add_argument("--manifest",required=True);p.add_argument("--actor-config",required=True)
    p.add_argument("--quantile",type=float,default=.75);p.add_argument("--allow-provisional",action="store_true")
    p.add_argument("--output-config",required=True);p.add_argument("--output-report",required=True);a=p.parse_args()
    if not a.allow_provisional: raise ValueError("Current corpus is diagnostic only; explicit --allow-provisional required")
    if Path(a.output_config).exists() or Path(a.output_report).exists(): raise ValueError("New calibration outputs required")
    config,report=calibrate(a.trace,json.loads(Path(a.manifest).read_text()),json.loads(Path(a.actor_config).read_text()),a.quantile)
    for path,value in [(a.output_config,config),(a.output_report,report)]:
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("points","source_sha256")},indent=2))
