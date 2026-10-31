#!/usr/bin/env python3
"""Read-only tensor/schema comparison. Does not repair or mutate the checkpoint."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from internal_signals.spanuq_adapter import (
    official_classes, translate_released_config, compare_state, sha256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_cls, model_cls = official_classes(args.source)
    checkpoint = Path(args.checkpoint_dir)
    before = sha256(checkpoint / "best_model.pt")
    raw = json.loads((checkpoint / "model_config.json").read_text())
    config, cap, aliases = translate_released_config(raw, config_cls)
    model = model_cls(**config.model_kwargs())
    state = torch.load(checkpoint / "best_model.pt", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    pe = state["token_encoder.pos_enc.pe"]
    module = importlib.import_module("spanuq.model")
    reference = module.SinusoidalPositionalEncoding(pe.shape[-1], max_len=pe.shape[1]).pe
    report = {
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": sha256(checkpoint / "best_model.pt"),
        "tensor_count": len(state), "source_tensor_count": len(expected),
        "aliases": aliases, **compare_state(expected, state),
        "common_matching_shapes": sum(k in state and v.shape == state[k].shape for k, v in expected.items()),
        "all_checkpoint_tensors_finite": all(bool(torch.isfinite(v).all()) for v in state.values()),
        "complete_comparison": [
            {"key": key, "source_shape": list(expected[key].shape) if key in expected else None,
             "checkpoint_shape": list(state[key].shape) if key in state else None,
             "source_dtype": str(expected[key].dtype) if key in expected else None,
             "checkpoint_dtype": str(state[key].dtype) if key in state else None}
            for key in sorted(set(expected) | set(state))],
        "pe_comparison": {
            "max_absolute_error_full_16384": float((pe - reference).abs().max()),
            "max_absolute_error_source_prefix_4096": float((pe[:, :4096] - expected["token_encoder.pos_enc.pe"]).abs().max()),
            "unchanged_checkpoint": True},
        "conditioner_tensor_statistics": {
            k: {"shape": list(v.shape), "finite": bool(torch.isfinite(v).all()), "norm": float(v.norm())}
            for k, v in state.items() if k.startswith("u_conditioner.")}}
    if report["checkpoint_sha256_after"] != before:
        raise RuntimeError("Checkpoint changed during read-only audit")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "complete_comparison"}, indent=2))


if __name__ == "__main__":
    main()
