#!/usr/bin/env python3
"""Exit 2 if the real published weights cannot strictly load. Never filters state keys."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.spanuq_adapter import CompatibilityError, strict_probe_load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        _, _, _, report = strict_probe_load(args.source, args.checkpoint_dir)
        status = 0
    except CompatibilityError as error:
        report = {"error": str(error), **error.report}
        status = 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensor_shapes"}, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
