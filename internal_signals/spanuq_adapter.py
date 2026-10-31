"""Strict official SpanUQ loader; refuses incomplete/unknown architectures."""
import dataclasses
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

SOURCE_COMMIT = "778b1c1e3eeeda66233fe6b426f9303f4c7f76a3"
PROBE_SHA256 = "78a203415037cb6645385d27b14d4207d184d522cc2c33af07098f2f2254fb12"


class CompatibilityError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_classes(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise CompatibilityError("Official source commit differs from pinned version")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if dirty:
        raise CompatibilityError("Official source has tracked modifications")
    if "spanuq" in sys.modules:
        location = Path(sys.modules["spanuq"].__file__).resolve()
        if source not in location.parents:
            raise CompatibilityError("A different spanuq package is already imported")
    sys.path.insert(0, str(source))
    config_cls = importlib.import_module("spanuq.config").SpanUQConfig
    model_cls = importlib.import_module("spanuq.model").SpanUQ_DETR
    return config_cls, model_cls


def translate_released_config(raw, config_cls):
    """Explicit, recorded legacy aliases. No unknown field is dropped."""
    values = dict(raw)
    changes = {}
    for old, new in (("mob_k", "mob_n_components"),
                     ("seq_agg_mode", "seq_aggregation")):
        if old in values:
            if new in values:
                raise CompatibilityError("Duplicate legacy and canonical field: " + old)
            values[new] = values.pop(old)
            changes[old] = {"canonical_field": new, "value": values[new]}
    # This published field controls input validation; it is NOT the PE length.
    response_cap = values.pop("max_seq_len", None)
    if response_cap is None or type(response_cap) is not int or response_cap < 1:
        raise CompatibilityError("Missing/invalid explicit response max_seq_len")
    changes["max_seq_len"] = {"adapter_response_token_cap": response_cap}
    known = {item.name for item in dataclasses.fields(config_cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise CompatibilityError("Unknown checkpoint config fields: " + repr(unknown))
    if values.get("seq_aggregation") != "importance":
        raise CompatibilityError("Only the published importance aggregation was audited")
    return config_cls(**values), response_cap, changes


def compare_state(expected, actual):
    return {
        "missing_keys": sorted(set(expected) - set(actual)),
        "unexpected_keys": sorted(set(actual) - set(expected)),
        "shape_mismatches": [
            {"key": key, "source_shape": list(expected[key].shape),
             "checkpoint_shape": list(actual[key].shape)}
            for key in sorted(set(expected) & set(actual))
            if expected[key].shape != actual[key].shape],
    }


def strict_probe_load(source, checkpoint_dir):
    config_cls, model_cls = official_classes(source)
    directory = Path(checkpoint_dir)
    raw = json.loads((directory / "model_config.json").read_text())
    report = {"source_commit": SOURCE_COMMIT,
              "checkpoint_sha256": sha256(directory / "best_model.pt")}
    if report["checkpoint_sha256"] != PROBE_SHA256:
        raise CompatibilityError("Probe SHA256 differs from pinned official LFS object", report)
    try:
        config_cls(**raw)
        report["unmodified_config_error"] = None
    except TypeError as error:
        report["unmodified_config_error"] = str(error)
    config, response_cap, aliases = translate_released_config(raw, config_cls)
    report["explicit_config_translation"] = aliases
    state = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
        report["checkpoint_wrapper"] = "model_state_dict"
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CompatibilityError("Expected an all-tensor state_dict", report)
    model = model_cls(**config.model_kwargs())
    report.update(compare_state(model.state_dict(), state))
    report["tensor_count"] = len(state)
    report["tensor_shapes"] = {key: list(value.shape) for key, value in state.items()}
    # Always exercise strict loading, preserving its actual exception as evidence.
    try:
        result = model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        report["strict_load_passed"] = False
        report["strict_error"] = str(error)
        raise CompatibilityError("Published SpanUQ architecture cannot strictly load this checkpoint", report) from error
    if result.missing_keys or result.unexpected_keys:
        raise CompatibilityError("Strict load returned unresolved keys", report)
    report["strict_load_passed"] = True
    model.eval().requires_grad_(False)
    temperature = float(json.loads((directory / "temperature.json").read_text())["T"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise CompatibilityError("Invalid released calibration temperature", report)
    return model, response_cap, temperature, report


class SpanUQAdapter:
    """Available only after the actual official checkpoint passes strict load.

    Current released 4B checkpoint fails construction. No fallback scores exist.
    """
    def __init__(self, source, checkpoint_dir, device="cpu"):
        self.probe, self.response_cap, self.temperature, self.metadata = strict_probe_load(
            source, checkpoint_dir)
        self.probe.to(device=device, dtype=torch.float32)
        self.device = device

    @torch.inference_mode()
    def score(self, extraction):
        if extraction.metadata.get("decoder_blocks") != [21, 22, 23]:
            raise ValueError("Extraction layers differ from the checkpoint")
        if extraction.metadata.get("layer_convention") != "zero_based_decoder_block_output":
            raise ValueError("Extraction uses a different layer convention")
        layers = extraction.hidden_states
        if not layers or not 0 < layers[0].shape[1] <= self.response_cap:
            raise ValueError("Empty or overlong response")
        inputs = [layer.to(self.device, dtype=torch.float32) for layer in layers]
        mask = torch.ones(inputs[0].shape[:2], dtype=torch.bool, device=self.device)
        # Reuse official decoder; native sequence score is read separately.
        spans = self.probe.predict_spans(inputs, mask)[0]
        output = self.probe(inputs, mask)
        sequence = float(output.seq_uncertainty[0])
        if not math.isfinite(sequence) or not 0 <= sequence <= 1:
            raise ValueError("Invalid native sequence uncertainty")
        result = []
        for span in spans:
            start, end, raw = span["start"], span["end"], span["u"]
            if not 0 <= start <= end < len(extraction.offsets):
                raise ValueError("Official span is outside response tokens")
            if not math.isfinite(raw) or not 0 <= raw <= 1:
                raise ValueError("Invalid official uncertainty")
            p = torch.tensor(raw, dtype=torch.float64).clamp(1e-7, 1 - 1e-7)
            calibrated = float(torch.sigmoid(torch.logit(p) / self.temperature))
            left, right = extraction.offsets[start][0], extraction.offsets[end][1]
            result.append({**span, "char_start": left, "char_end": right,
                           "text": extraction.response[left:right],
                           "raw_uncertainty": raw, "uncertainty": calibrated})
        return {"detected_spans": result, "native_sequence_uncertainty": sequence,
                "missing_span": not bool(result), "probe_forward_passes": 2,
                "metadata": self.metadata}
