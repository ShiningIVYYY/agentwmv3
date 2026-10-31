#!/usr/bin/env python3
"""Audit saved real decision traces without invoking a model or environment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.decision import messages
from internal_signals.token_nll import FORMAT


def audit(directory):
    folder = Path(directory)
    rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    history = {}
    archived_goals = json.loads((folder / "goals.json").read_text()) if (folder / "goals.json").exists() else None
    for row in rows:
        task_id = row["task_id"]
        goal = archived_goals[task_id] if archived_goals is not None else json.loads((folder / (task_id + ".json")).read_text())["task_description"]
        prior = history.setdefault(task_id, [])
        expected = messages(goal, row["observation"], prior, row["history_length"])
        if expected != row["messages"] or expected != row["generation_record"]["messages"]:
            raise ValueError("Prompt differs from current/past observable-only construction")
        if row["generation_record"]["response"] != row["output"]:
            raise ValueError("Scored and executed draft text differ")
        signal = row["internal_signals"]
        record = row["generation_record"]
        if signal["backend"] != "hf_generation_raw_bf16_v1" or signal["dtype"] != "bfloat16":
            raise ValueError("Mixed backend/precision")
        if signal["missing_span"]:
            raise ValueError("This accepted diagnostic unexpectedly contains missing fields")
        values = record["token_nll"]
        if len(values) != len(record["response_ids"]) or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("Invalid actual token NLL")
        if not math.isclose(signal["global_uncertainty"], math.fsum(values) / len(values), abs_tol=1e-12):
            raise ValueError("Global reduction mismatch")
        by_field = {}
        parsed = FORMAT.fullmatch(row["output"])
        if parsed is None:
            raise ValueError("Archived decision does not match fixed format")
        for span in signal["details"]["field_spans"]:
            raw = parsed[span["field"]]
            left, right = parsed.span(span["field"])
            left += len(raw) - len(raw.lstrip())
            right -= len(raw) - len(raw.rstrip())
            indices = [i for i, (a, b) in enumerate(signal["details"]["offsets"]) if a < right and b > left]
            if (left, right) != (span["char_start"], span["char_end"]) or indices != span["token_indices"]:
                raise ValueError("Field/token boundary mapping mismatch")
            if span["text"] != row["output"][span["char_start"]:span["char_end"]]:
                raise ValueError("Span text/character alignment mismatch")
            by_field[span["field"]] = math.fsum(values[i] for i in span["token_indices"]) / len(span["token_indices"])
            if not math.isclose(by_field[span["field"]], span["uncertainty"], abs_tol=1e-12):
                raise ValueError("Field reduction mismatch")
        if signal["local_uncertainty"] != max(by_field.values()):
            raise ValueError("Local max mismatch")
        prior.append(row)
    return {"steps": len(rows), "causal_prompt_construction": True, "field_and_global_arithmetic": True,
            "uniform_backend_and_dtype": True, "field_bounds_and_token_indices": True, "future_feedback_used_in_generation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))
