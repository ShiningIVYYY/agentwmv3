"""Pure, versioned Gate interface. No environment, model or future outcome input."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import time

POLICIES = ("never", "always", "local", "global", "internal", "experience", "full")
REQUIREMENTS = {p: (p in ("local", "global", "internal", "full"), p in ("experience", "full")) for p in POLICIES}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def truth(value):
    if value is True or value == "true": return "true"
    if value is False or value == "false": return "false"
    if value is None or value == "unknown": return "unknown"
    raise ValueError("Explicit true/false/unknown required")


@dataclass(frozen=True)
class ExternalInput:
    coverage: str = "unknown"
    conflict: str = "unknown"
    sufficiency: str = "unknown"
    reliability: str = "unknown"
    evidence_ids: tuple = ()
    matched_experience_ids: tuple = ()
    evidence_verified: bool = False
    action_status: str = "unknown"
    diagnostics: tuple = ()

    def __post_init__(self):
        for name in ("coverage", "conflict", "sufficiency"):
            if getattr(self, name) not in ("true", "false", "unknown"):
                raise ValueError("Invalid external tri-state")
        if self.action_status not in ("unknown", "confirmed_action", "correctable_conflict", "unresolved_conflict"):
            raise ValueError("Invalid action resolution status")


@dataclass(frozen=True)
class GateInput:
    decision_id: str
    remaining_wm_budget: int
    internal: dict | None = None
    external: ExternalInput = field(default_factory=ExternalInput)
    remaining_forecast_budget: int | None = None
    draft_action_valid: bool = True

    def __post_init__(self):
        for v in (self.remaining_wm_budget, self.remaining_forecast_budget):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValueError("Budget is a nonnegative integer, not a truth value")
        if not self.decision_id:
            raise ValueError("Decision identity required")


class RuleGate:
    def __init__(self, config, policy=None):
        self.config = dict(config)
        self.policy = policy or config["policy"]
        if self.policy not in POLICIES: raise ValueError("Unknown Gate")
        for name in ("tau_local", "tau_global"):
            v = config[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("Invalid uncertainty threshold")
        if not isinstance(config["allow_confirmed_action_skip"], bool):
            raise ValueError("Confirmation policy must be explicit")
        if config["memory_correction"] is not False or config["missing_policy"] != "global_then_call":
            raise ValueError("Phase4 correction is disabled; missing fallback is fixed")
        if config["max_forecasts_per_invocation"] != 4:
            raise ValueError("Current unchanged JSON/three-review-round protocol reserves four forecasts")
        self.needs_internal, self.needs_external = REQUIREMENTS[self.policy]

    def _internal(self, signal):
        if signal is None: return None, None, True, ["internal_missing"]
        signature = self.config["internal_signature"]
        for k in ("backend", "dtype", "model_revision"):
            if signal.get(k) != signature[k]:
                raise ValueError("Threshold scoring signature mismatch: " + k)
        details = signal.get("details", {})
        for k in ("method", "scoring_path", "score_unit"):
            if details.get(k) != signature[k]:
                raise ValueError("Threshold scoring path mismatch: " + k)
        valid_number = lambda x: not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
        errors = signal.get("errors", [])
        fatal = any(str(e).startswith(("alignment_or_numeric_error", "generation_truncated")) for e in errors)
        local = signal.get("local_uncertainty")
        glob = signal.get("global_uncertainty")
        missing = signal.get("missing_span") is not False or signal.get("status") != "ok"
        if fatal: local = glob = None
        if missing or not valid_number(local): local = None
        if not valid_number(glob): glob = None
        diagnostics = []
        if local is None: diagnostics.append("local_missing_or_invalid")
        if glob is None: diagnostics.append("global_missing_or_invalid")
        return local, glob, missing or fatal, diagnostics

    def decide(self, context: GateInput):
        if not isinstance(context, GateInput): raise TypeError("GateInput required; outcomes are not accepted")
        start = time.perf_counter()
        triggers, diagnostics = [], []
        local = glob = None
        missing = False
        if self.needs_internal:
            local, glob, missing, diagnostics = self._internal(context.internal)
        ext = context.external if self.needs_external else ExternalInput()
        diagnostics += list(ext.diagnostics)
        reliable = ext.evidence_verified and ext.reliability == "supported" and ext.coverage == "true" and bool(ext.evidence_ids)
        conflict = reliable and ext.conflict == "true"
        confirmed = (reliable and ext.conflict == "false" and ext.sufficiency == "true"
                     and ext.action_status == "confirmed_action" and context.draft_action_valid)
        if ext.conflict == "true" and ext.sufficiency == "true":
            diagnostics.append("conflict_sufficiency_inconsistent_conflict_wins")
        confirmed_skip = confirmed and self.config["allow_confirmed_action_skip"]
        if confirmed and not confirmed_skip: diagnostics.append("confirmed_skip_disabled_pending_real_validation")
        if self.policy == "never":
            requested, reason = False, "never_policy"
        elif self.policy == "always":
            requested, reason = True, "always_policy"
            triggers.append("always")
        else:
            if conflict: triggers.append("reliable_" + ext.action_status)
            if self.needs_internal:
                use_local = self.policy in ("local", "internal", "full")
                use_global = self.policy in ("global", "internal", "full")
                if use_local and local is not None and local > self.config["tau_local"]:
                    triggers.append("high_local_uncertainty")
                if use_global and glob is not None and glob > self.config["tau_global"]:
                    triggers.append("high_global_uncertainty")
                if use_local and local is None:
                    if glob is not None:
                        diagnostics.append("missing_local_global_fallback")
                        if glob > self.config["tau_global"] and "high_global_uncertainty" not in triggers:
                            triggers.append("high_global_fallback")
                    else: triggers.append("missing_internal_verification_required")
                if use_global and glob is None:
                    triggers.append("missing_global_verification_required")
                if not context.draft_action_valid: triggers.append("invalid_draft_format")
            if conflict:
                requested, reason = True, "reliable_action_conflict_requires_verification_correction_off"
            elif self.policy == "full" and confirmed_skip:
                requested, reason = False, "evidence_confirms_unchanged_draft"
            else:
                requested = bool(triggers)
                reason = "internal_trigger" if requested else "no_active_trigger"
                if self.needs_external and not reliable:
                    diagnostics.append("external_unknown_internal_fallback" if self.needs_internal else "external_unknown_abstention")
        budget_ok = context.remaining_wm_budget > 0 and (
            context.remaining_forecast_budget is None or context.remaining_forecast_budget >= self.config["max_forecasts_per_invocation"])
        call = requested and budget_ok
        if requested and not budget_ok: reason = "budget_exhausted_execute_unverified_draft"
        return {"schema": "gate-decision-v1", "decision_id": context.decision_id, "policy": self.policy,
            "call_wm": call, "verification_requested": requested, "verification_unavailable": requested and not budget_ok,
            "reason": reason, "triggered_signals": triggers, "diagnostics": diagnostics,
            "local_uncertainty": local, "global_uncertainty": glob, "missing_span": missing,
            "experience_coverage": ext.coverage, "experience_conflict": ext.conflict,
            "experience_sufficiency": ext.sufficiency, "experience_action_status": ext.action_status,
            "experience_evidence_ids": list(ext.evidence_ids), "evidence_verified": ext.evidence_verified,
            "confirmed_action": confirmed, "memory_correction_enabled": False,
            "budget_remaining": context.remaining_wm_budget, "forecast_budget_remaining": context.remaining_forecast_budget,
            "tau_local": self.config["tau_local"], "tau_global": self.config["tau_global"],
            "gate_latency": time.perf_counter() - start,
            "additional_cost": {"model_calls": 0, "tokens": 0, "forward_passes": 0},
            "signal_requirements": {"internal": self.needs_internal, "external": self.needs_external}}
