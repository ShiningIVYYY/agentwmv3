"""Read-only pre-step decisions and dispatch into the EXISTING WM/reflection flow."""
from dataclasses import asdict
import copy
import time
from experience.collection import fields
from experience.schema import ExternalQuery
from worldmodel import runtime as wm_runtime
from .budget import BudgetLedger
from .core import GateInput, RuleGate, fingerprint
from .evidence import resolve_external


class GateSession:
    def __init__(self, config, agent, adapter, memory=None, recorder=None):
        self.config, self.agent, self.adapter, self.memory = config, agent, adapter, memory
        self.gate = RuleGate(config)
        if self.gate.needs_external and memory is None: raise ValueError("Selected Gate requires a frozen memory")
        if agent.internal_actor.fingerprint != config["internal_signature"]["config_sha256"]:
            raise ValueError("Actor/config differs from threshold configuration")
        self.record = recorder or (lambda event, data: None)
        self.controller = None
        self.used = set()

    def start_episode(self, task_id, factory):
        if self.controller is not None: self.controller.wm.close()
        self.controller = None
        self.factory, self.task_id = factory, task_id
        self.budget = BudgetLedger(self.config["wm_invocation_budget_per_episode"], self.config["wm_forecast_budget_per_episode"])
        self.history, self.used = [], set()

    def prepare(self, text, metadata, task, observation, task_id, step):
        started = time.perf_counter()
        premise, consequence, action = fields(text)
        query = ExternalQuery(benchmark="scienceworld", task_id=task_id,
            episode_id=self.config["run_id"] + "/" + task_id, step_index=step,
            available_at_ns=time.time_ns(), goal=task, observation=observation,
            draft_action=action, premise=premise, expected_consequence=consequence)
        raw = None
        if self.gate.needs_external:
            raw = self.memory.evaluate(query)
        external = resolve_external(raw, query, self.memory) if self.gate.needs_external else None
        kwargs = {"external": external} if external is not None else {}
        context = GateInput(decision_id=query.episode_id + "/" + str(step),
            remaining_wm_budget=self.budget.invocations, remaining_forecast_budget=self.budget.forecasts,
            internal=metadata.get("internal_signals"), draft_action_valid=metadata["decision_format_valid"], **kwargs)
        decision = self.gate.decide(context)
        decision["decided_at_ns"] = time.time_ns()
        decision["input_sha256"] = fingerprint(asdict(context))
        self.pending = copy.deepcopy(decision)
        self.record("gate_decision_before_wm", copy.deepcopy(decision))
        result = {"draft_output": text, "gate_decision": decision, "gate_query": asdict(query),
                  "resolved_external": asdict(context.external), "gate_input": asdict(context),
                  "signal_and_gate_seconds": time.perf_counter() - started}
        if raw is not None: result["external_signals"] = raw
        return result

    def select(self, candidate, output, observation, steps, actions, step_number):
        d = self.pending
        identity = d["decision_id"]
        if identity in self.used: raise ValueError("Duplicate Gate dispatch")
        self.used.add(identity)
        if not d["call_wm"]:
            self.record("skip_wm", {"decision_id": identity})
            return candidate, output, {"gate_execution": {"branch": "skip", "wm_calls": 0,
                "review_calls": 0, "started_at_ns": time.time_ns(), "seconds": 0,
                "budget_after": self.budget.snapshot(), "draft_unchanged": True}}
        self.budget.reserve(identity, self.config["max_forecasts_per_invocation"])
        started_ns, start = time.time_ns(), time.perf_counter()
        context_seconds = 0
        if self.controller is None:
            begin = time.perf_counter()
            self.controller = self.factory()
            if self.controller is None or self.controller.wm.backend != "word2world" or self.controller.max_rounds != 3 or self.controller.review_protocol != "json":
                raise ValueError("Phase4 uses unchanged learned Word2World/JSON/three-round review only")
            for action, actual in self.history: self.controller.wm.commit(action, actual)
            context_seconds = time.perf_counter() - begin
        self.adapter.reviewing = True
        try:
            chosen, effective, log = wm_runtime.select(self.controller, candidate, output, observation, steps, actions, step_number)
        finally:
            self.adapter.reviewing = False
        reflection = log["reflection"]
        count = sum(not p.get("cached", False) for p in reflection["predictions"])
        self.budget.settle(identity, count)
        log["gate_execution"] = {"branch": "call", "wm_calls": count, "review_calls": len(reflection["reviews"]),
            "started_at_ns": started_ns, "finished_at_ns": time.time_ns(), "seconds": time.perf_counter() - start,
            "wm_context_setup_seconds": context_seconds, "budget_after": self.budget.snapshot(),
            "draft_unchanged": chosen == candidate, "preserved_protocol": "word2world/json/3-rounds/final-replacement-prediction"}
        self.record("call_wm_completed", {"decision_id": identity, **log["gate_execution"]})
        return chosen, effective, log

    def commit(self, action, actual, metadata):
        self.history.append((action, copy.deepcopy(actual)))
        if self.controller is not None:
            if metadata["gate_execution"]["branch"] == "call":
                wm_runtime.commit(self.controller, action, actual, metadata)
            else:
                # A skipped verification still teacher-forces its real transition.
                # No predict/review is invoked on this path.
                self.controller.wm.commit(action, actual)
        metadata["gate_execution"]["actual_committed_at_ns"] = time.time_ns()

    def close(self):
        if self.controller is not None: self.controller.wm.close()
