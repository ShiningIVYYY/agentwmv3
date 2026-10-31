#!/usr/bin/env python3
"""Small real train/dev collection. No test, gate, correction or threshold search."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_audited_paper_eval as runner
from experience.collection import case_from_trace, receipts
from experience.schema import ExternalQuery, validate_case
from experience.splits import digest, file_hash, goal_key, validate_manifest
from experience.memory import ExperienceMemory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["train", "development"], required=True)
    p.add_argument("--snapshot")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--science-url", default="http://127.0.0.1:36001")
    p.add_argument("--science-python", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not 1 <= a.steps <= 8:
        raise ValueError("Small diagnostic only (1..8 steps)")
    if (a.split == "development") != bool(a.snapshot):
        raise ValueError("Train collects without memory; dev requires frozen training snapshot")
    out = Path(a.output)
    if out.exists():
        raise ValueError("New Run ID required; no implicit retries/resume")
    manifest = json.loads(Path(a.manifest).read_text())
    validate_manifest(manifest)
    rows = manifest["diagnostic_selection"][a.split]
    if not rows or len(rows) > 6 or any(r not in manifest["partitions"][a.split] for r in rows):
        raise ValueError("Invalid diagnostic selection")
    manifest_sha = digest(json.dumps(manifest, sort_keys=True))
    memory = ExperienceMemory(json.loads(Path(a.snapshot).read_text()), manifest, a.top_k) if a.snapshot else None
    runtime_code = 'import json,importlib.metadata;from scienceworld.constants import JAR_PATH;import hashlib;print(json.dumps({"scienceworld":importlib.metadata.version("scienceworld"),"jar_sha256":hashlib.sha256(open(JAR_PATH,"rb").read()).hexdigest()}))'
    runtime = json.loads(subprocess.check_output([a.science_python, "-c", runtime_code], text=True))
    if runtime != {"scienceworld": "1.2.3", "jar_sha256": "f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7"}:
        raise ValueError("Unverified simulator environment")
    out.mkdir(parents=True)
    config_path = ROOT / "configs/internal_signals/scienceworld_token_nll_v1.json"
    config = SimpleNamespace(internal_signals_config=str(config_path), tokenizer_path=a.model_dir,
        internal_device=a.device, science_url=a.science_url, max_steps=a.steps, max_tokens=512,
        disable_thinking=True, wm_backend="none", science_context=None, science_demo=None)
    source_hashes = {str(f.relative_to(ROOT)): file_hash(f) for f in
        [*sorted((ROOT / "experience").glob("*.py")), *sorted((ROOT / "internal_signals").glob("*.py")),
         Path(__file__).resolve(), ROOT / "scripts/run_audited_paper_eval.py", config_path,
         ROOT / "docs/PAPER_ALIGNED_EVAL.md"]}
    runner.atomic(out / "manifest.json", {"purpose": "functional_diagnostic_not_benchmark",
        "split_sha256": manifest_sha, "split_manifest_file_sha256": file_hash(a.manifest), "tasks": rows,
        "runtime": runtime, "max_steps": a.steps, "memory_snapshot": memory.snapshot_id if memory else None,
        "source_hashes": source_hashes, "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    actor, env = runner.Agent(config, out), runner.Science(config)
    actor.external_signal_provider = memory
    actor.external_episode_prefix = out.name
    cases, traces = [], []
    original_generate = actor.generate
    decisions = {}
    def generate(messages, task_id, step, **kwargs):
        reset = next(r for r in reversed(log) if r["endpoint"] == "reset")
        row = next(r for r in rows if r["task_id"] == task_id)
        if goal_key(reset["response"]["task_description"]) != row["goal_group"]:
            raise ValueError("Native goal identity differs before Actor generation")
        decisions[(task_id, step)] = time.time_ns()
        return original_generate(messages, task_id, step, **kwargs)
    actor.generate = generate
    with receipts(runner) as log:
        current = None
        def trace(task_id, record):
            index = 1 + sum(c["task_id"] == task_id for c in cases)
            reset = next(r for r in reversed(log) if r["endpoint"] == "reset")
            goal = reset["response"]["task_description"]
            case = case_from_trace(current, out.name, record, index, decisions[(task_id, index)], log[-1], goal, manifest_sha)
            validate_case(case, manifest, construction=a.split == "train")
            # Every external query happened before the environment action HTTP request.
            if memory and record["external_signals"]["query_at_ns"] >= log[-1]["started_at_ns"]:
                raise ValueError("External signal used after-action information")
            cases.append(case)
            saved = {"case_id": case["case_id"], "task_id": task_id, "step_index": index, **record}
            traces.append(saved)
            for name, value in (("cases.jsonl", case), ("steps.jsonl", saved), ("receipts.jsonl", log[-1])):
                with (out / name).open("a") as stream:
                    stream.write(json.dumps(value) + "\n")
            runner.atomic(out / "progress.json", {"state": "running", "steps": len(cases), "task_id": task_id})
        actor.trace = trace
        for current in rows:
            result = env.run(current, actor)
            runner.atomic(out / (current["task_id"] + ".json"), result)
    signals = [r["external_signals"] for r in traces] if memory else []
    summary = {"purpose": "functional_only_no_gate_or_threshold_selection", "steps": len(cases),
        "episodes": len(rows), "actor_calls": actor.total_calls, "memory_snapshot": memory.snapshot_id if memory else None,
        "internal_backend": "hf_generation_raw_bf16_v1", "internal_config_sha256": file_hash(config_path),
        "missing_span_count": sum(r["internal_signals"]["missing_span"] for r in traces),
        "all_drafts_unchanged": all(c["draft_action"] == c["executed_action"] for c in cases),
        "max_peak_allocated_bytes": max(r["internal_cost"]["peak_allocated_bytes"] for r in traces),
        "generation_seconds": sum(r["internal_cost"]["generation_seconds"] for r in traces),
        "internal_extraction_seconds": sum(r["internal_signals"]["extraction_latency"] for r in traces),
        "generated_tokens_including_eos": sum(r["internal_cost"]["output_tokens_including_eos"] for r in traces),
        "retrieval_seconds": sum(s["retrieval_latency"] for s in signals),
        "judge_seconds": sum(s["judge_latency"] for s in signals), "judge_model_calls": 0, "additional_tokens": 0,
        "external_signal_counts": {name: {v: sum(s[name] == v for s in signals) for v in ("true", "false", "unknown")}
                                   for name in ("coverage", "conflict", "sufficiency")},
        "environment_step_seconds": sum(r["available_at_ns"] - r["started_at_ns"] for r in log if r["endpoint"] == "step") / 1e9,
        "extra_model_forward_passes": 0, "wm_calls": 0, "oracle_calls": 0, "memory_correction_calls": 0}
    runner.atomic(out / "summary.json", summary)
    runner.atomic(out / "progress.json", {"state": "finished", "steps": len(cases)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
