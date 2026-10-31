#!/usr/bin/env python3
"""15-step development diagnostic reusing the existing Science.run/Agent interface."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_audited_paper_eval import Agent, Science, atomic
from internal_signals.decision import messages as decision_messages
from internal_signals.spanuq_adapter import sha256
from internal_signals.token_nll import teacher_forced_nll


def prepare_tasks(args):
    protocol_file = Path(args.protocol_file).resolve()
    repo = protocol_file.parents[1]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != "35a44f5c5647b1785ad23ddf0ffbd196b695bf86":
        raise ValueError("Unpinned source task catalog")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", str(protocol_file)]).returncode:
        raise ValueError("Modified reference task catalog")
    spec = importlib.util.spec_from_file_location("diagnostic_catalog", protocol_file)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    source = {}
    for split, path in [("train", args.train_ids), ("test", args.test_ids)]:
        if sha256(path) != protocol.SOURCE_FILES[split][2]:
            raise ValueError("Source split SHA mismatch")
        source[split] = [row["item_id"] for row in json.loads(Path(path).read_text())]
    tests = set(source["test"])
    allowed = set(source["train"]) - tests - {"sciworld_4509"}
    rows = []
    for family in ["use-thermometer", "test-conductivity", "find-living-thing"]:
        candidates = [item for item in allowed if protocol.resolve_item(item)["task_name"] == family]
        chosen = min(candidates, key=lambda item: hashlib.sha256(
            ("agentwmv3-phase2-diagnostic-v1|" + item).encode()).hexdigest())
        task = protocol.resolve_item(chosen)
        rows.append({"task_id": chosen, "index": int(chosen.rsplit("_", 1)[1]),
                     "taskName": task["task_name"], "variationIdx": task["variation_id"],
                     "split": "diagnostic-development", "category": task["category"]})
    assert not {r["task_id"] for r in rows} & tests
    return rows, {"source_revision": protocol.DATA_REVISION, "source_repo": protocol.DATA_REPO,
                  "train_sha256": sha256(args.train_ids), "test_sha256": sha256(args.test_ids),
                  "train_eval_overlap": len(set(source["train"]) & tests),
                  "reference_commit": commit, "catalog_sha256": sha256(protocol_file),
                  "selection": "minimum SHA256(agentwmv3-phase2-diagnostic-v1|id) per named family",
                  "excluded_demo": "sciworld_4509", "reserved_for_future": "development, never untouched validation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--internal-config", default=str(ROOT / "configs/internal_signals/scienceworld_token_nll_v1.json"))
    p.add_argument("--protocol-file", required=True)
    p.add_argument("--train-ids", required=True)
    p.add_argument("--test-ids", required=True)
    p.add_argument("--science-url", default="http://127.0.0.1:36001")
    p.add_argument("--science-python", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("Use a new Run ID/output directory; no ambiguous diagnostic resume")
    out.mkdir(parents=True)
    rows, provenance = prepare_tasks(args)
    atomic(out / "manifest.json", {"tasks": rows, **provenance})
    runtime_code = 'import json,importlib.metadata;from scienceworld.constants import JAR_PATH;import hashlib;print(json.dumps({"scienceworld":importlib.metadata.version("scienceworld"),"jar_sha256":hashlib.sha256(open(JAR_PATH,"rb").read()).hexdigest()}))'
    runtime = json.loads(subprocess.check_output([args.science_python, "-c", runtime_code], text=True))
    if runtime["scienceworld"] != "1.2.3":
        raise ValueError("Diagnostic requires inspected ScienceWorld package version")
    runtime.update(protocol="existing AgentGym HTTP; own sessions; no environment package changes")
    config = SimpleNamespace(internal_signals_config=args.internal_config,
        tokenizer_path=args.model_dir, internal_device=args.device,
        science_url=args.science_url, max_steps=5, max_tokens=512, disable_thinking=True,
        wm_backend="none", science_context=None, science_demo=None)
    actor = Agent(config, out)
    env = Science(config)
    audit = {"raw_generation_repeat": None, "teacher_forced_diagnostic": None}
    original_generate = actor.generate
    first = True

    def generate(messages, task_id, step, **kwargs):
        nonlocal first
        text, meta = original_generate(messages, task_id, step, **kwargs)
        if first:
            first = False
            repeated, other = original_generate(messages, task_id, step, **kwargs)
            a, b = meta["generation_record"], other["generation_record"]
            if a["response_ids"] != b["response_ids"] or text != repeated:
                raise ValueError("Pinned raw-generation path was not reproducible")
            delta = max(abs(x - y) for x, y in zip(a["token_nll"], b["token_nll"]))
            audit["raw_generation_repeat"] = {
                "task_id": task_id, "step": step, "max_abs_nll": delta,
                "response_ids_equal": True, "validation_additional_actor_calls": 1,
                "validation_generation_cost": other["internal_cost"]}
            if delta != 0:
                raise ValueError("Repeat generation token NLL differs")
            actor.internal_actor.sync()
            start = time.perf_counter()
            rescored = teacher_forced_nll(actor.internal_actor.model, a["prompt_ids"], a["response_ids"])
            actor.internal_actor.sync()
            diffs = [abs(x - y) for x, y in zip(a["token_nll"], rescored.tolist())]
            audit["teacher_forced_diagnostic"] = {
                "task_id": task_id, "max_abs_nll": max(diffs), "mean_abs_nll": statistics.mean(diffs),
                "extra_forward_seconds": time.perf_counter() - start,
                "extra_backbone_forward_passes": 1,
                "forward_tokens": len(a["prompt_ids"]) + len(a["response_ids"]),
                "not_used_for_internal_signals": True, "rescored_token_nll": rescored.tolist()}
            atomic(out / "numerical_validation.json", audit)
        return text, meta

    actor.generate = generate
    traces = []

    def trace(task_id, record):
        step = {"task_id": task_id, "step": 1 + sum(x["task_id"] == task_id for x in traces), **record}
        # Counterfactual action-only prompt token count; no extra baseline inference.
        history = [x for x in traces if x["task_id"] == task_id]
        keep = record["history_length"]
        original = decision_messages(goals[task_id], record["observation"], history, keep, structured=False)
        step["internal_cost"]["action_only_prompt_tokens_same_context"] = actor.count(original)
        step["internal_cost"]["structured_prompt_extra_tokens_same_context"] = record["prompt_tokens"] - actor.count(original)
        traces.append(step)
        with (out / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(step, ensure_ascii=False) + "\n")
            stream.flush()
        atomic(out / "progress.json", {"completed_steps": len(traces), "task_id": task_id, "state": "running"})

    # Capture the true goal from the exact messages, without a second environment reset.
    goals = {}
    bounded_generate = actor.generate

    def generated(messages, task_id, step, **kwargs):
        content = messages[0]["content"]
        goals[task_id] = content.split("\nTask: ", 1)[1].split("\nRecent executed history:", 1)[0]
        return bounded_generate(messages, task_id, step, **kwargs)

    actor.generate = generated
    actor.trace = trace
    for row in rows:
        result = env.run(row, actor)  # Original environment action loop, no WM.
        atomic(out / (row["task_id"] + ".json"), result)
    signals = [x["internal_signals"] for x in traces]
    numeric = [x["local_uncertainty"] for x in signals if x["local_uncertainty"] is not None]
    summary = {
        "run_id": out.name, "purpose": "development_functional_diagnostic_not_benchmark",
        "steps": len(traces), "actor_calls_including_repeat_validation": actor.total_calls,
        "missing_span_rate": statistics.mean(s["missing_span"] for s in signals),
        "format_failure_rate": statistics.mean(not s["details"].get("format_valid", False) for s in signals),
        "alignment_or_numeric_error_count": sum(any(e.startswith("alignment_or_numeric_error") for e in s["errors"]) for s in signals),
        "valid_action_format_rate": statistics.mean(x["valid_format"] for x in traces),
        "native_no_known_action_matches_count": sum("no known action matches" in x["next_observation"].lower() for x in traces),
        "local_nll_min": min(numeric) if numeric else None, "local_nll_max": max(numeric) if numeric else None,
        "mean_generation_seconds": statistics.mean(x["internal_cost"]["generation_seconds"] for x in traces),
        "mean_extraction_seconds": statistics.mean(s["extraction_latency"] for s in signals),
        "mean_non_action_output_tokens": statistics.mean(x["internal_cost"]["non_action_response_tokens_including_labels"] for x in traces),
        "mean_structured_prompt_extra_tokens_same_context": statistics.mean(x["internal_cost"]["structured_prompt_extra_tokens_same_context"] for x in traces),
        "max_peak_allocated_bytes": max(x["internal_cost"]["peak_allocated_bytes"] for x in traces),
        "online_extra_backbone_forward_passes": 0,
        "numerical_validation": audit, "environment_metadata": runtime,
        "internal_config": actor.internal_actor.config, "provenance": provenance,
        "phase3_started": False, "threshold_selection": False,
        "input_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [ROOT / "scripts/run_audited_paper_eval.py", Path(__file__).resolve(),
             *sorted((ROOT / "internal_signals").glob("*.py"))]}}
    if not 10 <= len(traces) <= 20:
        summary["step_count_acceptance"] = False
    else:
        summary["step_count_acceptance"] = True
    atomic(out / "summary.json", summary)
    atomic(out / "progress.json", {"state": "finished", "completed_steps": len(traces), "summary": "summary.json"})
    print(json.dumps(summary, indent=2))
    return 0 if summary["step_count_acceptance"] and not summary["alignment_or_numeric_error_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
