#!/usr/bin/env python3
"""Metadata-only audit and frozen split creation; no model or simulator calls."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.splits import prepare, select_diagnostic

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("train-ids", "test-ids", "catalog", "goal-metadata", "test-metadata", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--native-goal-metadata")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("Refuse to overwrite a frozen split")
    result = prepare(args.train_ids, args.test_ids, args.catalog, args.goal_metadata, args.test_metadata, args.native_goal_metadata)
    result["diagnostic_selection"] = {s: select_diagnostic(result, s, 2 if s == "train" else 1)
                                      for s in ("train", "development")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit": result["audit"], "diagnostic_selection": result["diagnostic_selection"]}, indent=2))
