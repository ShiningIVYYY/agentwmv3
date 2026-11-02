#!/usr/bin/env python3
"""CPU-only mocks/fixtures below are never scientific WM/environment results."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gates.core import ExternalInput,GateInput,RuleGate,POLICIES,fingerprint
from gates.budget import BudgetLedger
from gates.runtime import GateSession
from gates.evidence import resolve_external
from scripts.calibrate_gate import calibrate,quantile,correlation,ranks
from reflection.controller import ReflectionController


CONFIG={"policy":"full","tau_local":.2,"tau_global":.1,"allow_confirmed_action_skip":False,
    "memory_correction":False,"missing_policy":"global_then_call","max_forecasts_per_invocation":4,
    "wm_invocation_budget_per_episode":1,"wm_forecast_budget_per_episode":4,"run_id":"unit_fixture",
    "internal_signature":{"backend":"hf_generation_raw_bf16_v1","dtype":"bfloat16","model_revision":"unit_revision",
        "method":"fixed_field_mean_token_nll_v1","scoring_path":"raw_generation_logits_before_temperature",
        "score_unit":"nats_per_token","config_sha256":"unit_config"}}


def internal(local=.05,glob=.02,missing=False):
    return {"local_uncertainty":local,"global_uncertainty":glob,"missing_span":missing,
        "status":"partial" if missing else "ok","errors":[],
        **{k:CONFIG["internal_signature"][k] for k in ("backend","dtype","model_revision")},
        "details":{k:CONFIG["internal_signature"][k] for k in ("method","scoring_path","score_unit")}}


def conflict(status="unresolved_conflict",suff="false"):
    return ExternalInput("true","true",suff,"supported",("case-unit",),("card-unit",),True,status)


def context(**kwargs):
    return GateInput(**dict({"decision_id":"episode/1","remaining_wm_budget":1,"remaining_forecast_budget":4,
                             "internal":internal()},**kwargs))


class GateTests(unittest.TestCase):
    def decide(self,ctx=None,policy="full",**config):
        return RuleGate(dict(CONFIG,**config),policy).decide(ctx or context())

    def test_seven_baselines_share_interface(self):
        for p in POLICIES:
            d=self.decide(policy=p)
            for key in ("call_wm","reason","triggered_signals","budget_remaining","gate_latency","additional_cost"):
                self.assertIn(key,d)
            self.assertEqual(d["policy"],p)

    def test_never_ignores_missing_inputs(self):self.assertFalse(self.decide(context(internal=None),"never")["call_wm"])
    def test_always_ignores_missing_inputs(self):self.assertTrue(self.decide(context(internal=None),"always")["call_wm"])
    def test_never_has_no_signal_requirements(self):self.assertEqual(self.decide(policy="never")["signal_requirements"],{"internal":False,"external":False})
    def test_always_has_no_signal_requirements(self):self.assertEqual(self.decide(policy="always")["signal_requirements"],{"internal":False,"external":False})
    def test_local_high(self):self.assertTrue(self.decide(context(internal=internal(.3,.02)),"local")["call_wm"])
    def test_global_high(self):self.assertTrue(self.decide(context(internal=internal(.05,.2)),"global")["call_wm"])
    def test_local_ignores_valid_global_high(self):self.assertFalse(self.decide(context(internal=internal(.05,.2)),"local")["call_wm"])
    def test_global_ignores_local_high(self):self.assertFalse(self.decide(context(internal=internal(.3,.02)),"global")["call_wm"])
    def test_combined_union(self):self.assertTrue(self.decide(context(internal=internal(.3,.02)),"internal")["call_wm"])
    def test_strict_threshold_equality(self):self.assertFalse(self.decide(context(internal=internal(.2,.1)))["call_wm"])
    def test_conflict_overrides_low_internal(self):self.assertTrue(self.decide(context(external=conflict()))["call_wm"])
    def test_correctable_still_calls(self):self.assertTrue(self.decide(context(external=conflict("correctable_conflict")))["call_wm"])
    def test_conflict_overrides_sufficiency(self):
        d=self.decide(context(external=conflict(suff="true")),allow_confirmed_action_skip=True)
        self.assertTrue(d["call_wm"]);self.assertIn("conflict_sufficiency_inconsistent_conflict_wins",d["diagnostics"])
    def test_unsupported_conflict_not_treated_as_no_conflict(self):
        d=self.decide(context(external=replace(conflict(),evidence_verified=False)))
        self.assertFalse(d["call_wm"]);self.assertEqual(d["experience_conflict"],"true")
        self.assertIn("external_unknown_internal_fallback",d["diagnostics"])
    def test_no_evidence_cannot_confirm(self):
        e=ExternalInput("true","false","true","supported",(),(),True,"confirmed_action")
        self.assertTrue(self.decide(context(internal=internal(.3,.2),external=e),allow_confirmed_action_skip=True)["call_wm"])
    def test_confirmed_disabled_by_default(self):
        e=replace(conflict(),conflict="false",sufficiency="true",action_status="confirmed_action")
        self.assertTrue(self.decide(context(internal=internal(.3,.2),external=e))["call_wm"])
    def test_explicit_confirmed_current_action_skip(self):
        e=replace(conflict(),conflict="false",sufficiency="true",action_status="confirmed_action")
        d=self.decide(context(internal=internal(.3,.2),external=e),allow_confirmed_action_skip=True)
        self.assertFalse(d["call_wm"]);self.assertEqual(d["reason"],"evidence_confirms_unchanged_draft")
    def test_invalid_draft_not_confirmed(self):
        e=replace(conflict(),conflict="false",sufficiency="true",action_status="confirmed_action")
        self.assertTrue(self.decide(context(external=e,draft_action_valid=False),allow_confirmed_action_skip=True)["call_wm"])
    def test_experience_unknown_abstains(self):
        d=self.decide(context(internal=None),"experience");self.assertFalse(d["call_wm"])
        self.assertEqual(d["experience_conflict"],"unknown")
    def test_experience_conflict(self):self.assertTrue(self.decide(context(internal=None,external=conflict()),"experience")["call_wm"])
    def test_internal_baseline_does_not_use_conflict(self):self.assertFalse(self.decide(context(external=conflict()),"internal")["call_wm"])
    def test_unknown_falls_back_internal(self):self.assertTrue(self.decide(context(internal=internal(.3,.02)))["call_wm"])
    def test_both_missing_calls(self):self.assertTrue(self.decide(context(internal=None))["call_wm"])
    def test_local_missing_global_high_calls(self):self.assertTrue(self.decide(context(internal=internal(None,.2,True)))["call_wm"])
    def test_local_missing_global_low_records_fallback(self):
        d=self.decide(context(internal=internal(None,.02,True)));self.assertFalse(d["call_wm"])
        self.assertTrue(d["missing_span"]);self.assertIn("missing_local_global_fallback",d["diagnostics"])
    def test_global_invalid_requires_call(self):self.assertTrue(self.decide(context(internal=internal(.05,float("nan"))))["call_wm"])
    def test_fatal_alignment_ignores_apparent_low_values(self):
        s=internal();s["errors"]=["alignment_or_numeric_error: unit"]
        self.assertTrue(self.decide(context(internal=s))["call_wm"])
    def test_wrong_backend_rejected(self):
        s=internal();s["backend"]="other"
        with self.assertRaises(ValueError):self.decide(context(internal=s))
    def test_wrong_precision_rejected(self):
        s=internal();s["dtype"]="float32"
        with self.assertRaises(ValueError):self.decide(context(internal=s))
    def test_wrong_revision_rejected(self):
        s=internal();s["model_revision"]="other"
        with self.assertRaises(ValueError):self.decide(context(internal=s))
    def test_wrong_path_rejected(self):
        s=internal();s["details"]["scoring_path"]="teacher_forced"
        with self.assertRaises(ValueError):self.decide(context(internal=s))
    def test_budget_exhaustion(self):
        d=self.decide(context(remaining_wm_budget=0,external=conflict()))
        self.assertFalse(d["call_wm"]);self.assertTrue(d["verification_unavailable"])
    def test_forecast_reservation_cap(self):self.assertFalse(self.decide(context(remaining_forecast_budget=3),"always")["call_wm"])
    def test_invalid_budget_types(self):
        for x in (-1,True,.5,None):
            with self.assertRaises(ValueError):context(remaining_wm_budget=x)
    def test_zero_is_a_valid_low_score(self):
        d=self.decide(context(internal=internal(0,0)));self.assertFalse(d["call_wm"])
        self.assertEqual(d["signal_validity"],"valid")
    def test_no_future_input(self):
        with self.assertRaises(TypeError):context(environment_feedback="future")
    def test_no_training_or_model_cost(self):self.assertEqual(self.decide()["additional_cost"],{"model_calls":0,"tokens":0,"forward_passes":0})
    def test_invalid_thresholds(self):
        for x in (-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):RuleGate(dict(CONFIG,tau_local=x))
    def test_gate_does_not_mutate_input(self):
        c=context();before=copy.deepcopy(c);self.decide(c);self.assertEqual(c,before)
    def test_correction_cannot_be_silently_enabled(self):
        with self.assertRaises(ValueError):RuleGate(dict(CONFIG,memory_correction=True))
    def test_no_silent_depth_change(self):
        with self.assertRaises(ValueError):RuleGate(dict(CONFIG,max_forecasts_per_invocation=2))


class BudgetTests(unittest.TestCase):
    def test_reserve_and_release_unused(self):
        b=BudgetLedger(2,8);b.reserve("a",4);self.assertEqual((b.invocations,b.forecasts),(1,4))
        b.settle("a",1);self.assertEqual((b.invocations,b.forecasts),(1,7))
    def test_duplicate_call_rejected(self):
        b=BudgetLedger(2,8);b.reserve("a",4)
        with self.assertRaises(ValueError):b.reserve("a",4)
    def test_failed_prediction_keeps_reservation(self):
        b=BudgetLedger(1,4);b.reserve("a",4);self.assertEqual(b.snapshot()["unsettled_reservations"],{"a":4})
    def test_no_forecast_overshoot(self):
        b=BudgetLedger(1,4);b.reserve("a",4)
        with self.assertRaises(ValueError):b.settle("a",5)
    def test_no_budget_inflation(self):
        b=BudgetLedger(0,4)
        with self.assertRaises(ValueError):b.reserve("a",4)
        self.assertEqual(b.forecasts,4)


class CalibrationTests(unittest.TestCase):
    def test_linear_quantile(self):self.assertEqual(quantile([0,1,2,3],.75),2.25)
    def test_constant_correlation_is_unknown(self):self.assertIsNone(correlation([1,1],[2,3]))
    def test_tied_ranks(self):self.assertEqual(ranks([1,1,3]),[1.5,1.5,3])
    def test_real_dev_tool_provisional_and_no_labels(self):
        actor=json.loads((ROOT/"configs/internal_signals/scienceworld_token_nll_v1.json").read_text())
        manifest=json.loads((ROOT/"docs/phase3/split_manifest.json").read_text())
        cfg,report=calibrate([ROOT/"docs/phase3/phase3_dev_v1/steps.jsonl",ROOT/"docs/phase3/phase3_targeted_dev_v1/steps.jsonl"],manifest,actor)
        self.assertEqual(report["valid_steps"],25);self.assertTrue(report["not_stable_calibration"])
        self.assertFalse(report["selection_uses_task_success_or_reward"]);self.assertFalse(cfg["allow_confirmed_action_skip"])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        from experience.memory import ExperienceMemory
        from experience.schema import ExternalQuery
        from experience.collection import fields
        import time
        self.memory=ExperienceMemory(json.loads((ROOT/"docs/phase3/memory_snapshot.json").read_text()),
                                     json.loads((ROOT/"docs/phase3/split_manifest.json").read_text()))
        trace=json.loads((ROOT/"docs/phase3/phase3_targeted_dev_v1/steps.jsonl").read_text().splitlines()[0])
        premise,consequence,action=fields(trace["output"])
        case=json.loads((ROOT/"docs/phase3/phase3_targeted_dev_v1/cases.jsonl").read_text().splitlines()[0])
        self.query=ExternalQuery("scienceworld",case["task_id"],"offline-evidence-unit",1,time.time_ns(),case["goal"],
                                 trace["observation"],action,premise,consequence)
        self.raw=self.memory.evaluate(self.query)
    def test_actual_supported_conflict_binding(self):
        e=resolve_external(self.raw,self.query,self.memory);self.assertTrue(e.evidence_verified)
        self.assertEqual(e.action_status,"unresolved_conflict")
    def test_fake_case_id_cannot_trigger(self):
        self.raw["supporting_evidence_ids"]=["invented"]
        self.assertFalse(resolve_external(self.raw,self.query,self.memory).evidence_verified)
    def test_other_query_cannot_reuse_evidence(self):
        self.assertFalse(resolve_external(self.raw,replace(self.query,draft_action="look around"),self.memory).evidence_verified)
    def test_bm25_does_not_prove_applicability(self):
        self.raw["retrieved_cards"][0]["bm25_score"]=100000000
        self.assertFalse(resolve_external(self.raw,replace(self.query,observation="No door identity"),self.memory).evidence_verified)
    def test_sufficiency_cannot_relabel_movement_as_confirmed(self):
        self.raw.update(conflict="false",sufficiency="true")
        e=resolve_external(self.raw,self.query,self.memory)
        self.assertNotEqual(e.action_status,"confirmed_action")
    def test_malformed_truth_stays_unknown(self):
        self.raw["conflict"]="maybe";e=resolve_external(self.raw,self.query,self.memory)
        self.assertEqual(e.conflict,"unknown");self.assertFalse(e.evidence_verified)
    def test_future_evidence_does_not_bind(self):
        q=replace(self.query,available_at_ns=self.memory.built_at-1)
        self.assertFalse(resolve_external(self.raw,q,self.memory).evidence_verified)


class DispatchTests(unittest.TestCase):
    def setup_session(self,policy="full"):
        class FakeWM:
            backend="word2world"
            def __init__(self):self.calls=0;self.history=[]
            def predict(self,action):
                self.calls+=1
                return {"backend":"word2world","observation":"unit predicted text","seconds":0,"usage":{}}
            def commit(self,action,actual):self.history.append((action,actual))
            def close(self):pass
        agent=SimpleNamespace(internal_actor=SimpleNamespace(fingerprint="unit_config"),count=lambda x:1)
        agent.generate=lambda messages,task,step: (json.dumps({"decision":"keep","action":"look around","reason":"unit fixture"}),{})
        wm=FakeWM();adapter=SimpleNamespace(reviewing=False)
        cfg=dict(CONFIG,policy=policy)
        mem=SimpleNamespace() if policy=="full" else None
        session=GateSession(cfg,agent,adapter,mem)
        created=[]
        def factory():created.append(True);return ReflectionController(wm,agent,"scienceworld","unit task","unit_task",3)
        session.start_episode("unit_task",factory)
        return session,wm,created
    def test_skip_invokes_neither_factory_nor_predict(self):
        s,wm,created=self.setup_session("never");s.pending=RuleGate(dict(CONFIG,policy="never")).decide(context())
        a,_,m=s.select("look around","Action: look around","unit",[],[],1);s.commit(a,{"observation":"unit real","wm_observation":"unit real"},m)
        self.assertEqual(wm.calls,0);self.assertEqual(created,[])
    def test_call_uses_real_controller_algorithm_mock_models_only(self):
        s,wm,created=self.setup_session("always");s.pending=RuleGate(dict(CONFIG,policy="always")).decide(context())
        a,_,m=s.select("look around","Action: look around","unit",[],[],1)
        self.assertEqual(wm.calls,1);self.assertEqual(m["gate_execution"]["branch"],"call")
        s.commit(a,{"observation":"unit real","wm_observation":"unit real"},m);self.assertEqual(len(wm.history),1)
    def test_skip_after_call_commits_real_history_without_predict(self):
        s,wm,_=self.setup_session("always");s.pending=RuleGate(dict(CONFIG,policy="always")).decide(context())
        a,_,m=s.select("look around","Action: look around","unit",[],[],1);s.commit(a,{"observation":"a","wm_observation":"a"},m)
        s.pending=RuleGate(dict(CONFIG,policy="always")).decide(context(decision_id="episode/2",remaining_wm_budget=0))
        a,_,m=s.select("wait","Action: wait","unit",[],[],2);s.commit(a,{"observation":"b","wm_observation":"b"},m)
        self.assertEqual(wm.calls,1);self.assertEqual([a for a,_ in wm.history],["look around","wait"])
    def test_skip_before_first_call_history_is_replayed_to_wm(self):
        s,wm,_=self.setup_session("always");s.pending=RuleGate(dict(CONFIG,policy="never")).decide(context())
        a,_,m=s.select("wait","Action: wait","unit",[],[],1);s.commit(a,{"wm_observation":"a"},m)
        s.pending=RuleGate(dict(CONFIG,policy="always")).decide(context(decision_id="episode/2"))
        s.select("look around","Action: look around","unit",[],[],2)
        self.assertEqual(wm.history,[("wait",{"wm_observation":"a"})])
    def test_duplicate_dispatch_rejected(self):
        s,_,_=self.setup_session("never");s.pending=RuleGate(dict(CONFIG,policy="never")).decide(context())
        s.select("wait","Action: wait","unit",[],[],1)
        with self.assertRaises(ValueError):s.select("wait","Action: wait","unit",[],[],1)


if __name__=="__main__":unittest.main(verbosity=2)
