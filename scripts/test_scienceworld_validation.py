#!/usr/bin/env python3
"""Integration-contract CPU tests. Synthetic unit outcomes are not experiment data."""
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.profile import canonicalize_observation,validate_profile,native_menu_actions
from evaluation.seeds import request_seed
from evaluation.metrics import episode_metrics,costs,aggregate
from evaluation.observer import observe_wm
from scripts.prepare_phase5_plan import prepare
from internal_signals.hidden_states import response_alignment
from transformers import AutoTokenizer


class ProfileTests(unittest.TestCase):
    def test_config_matches_contract(self):validate_profile(json.loads((ROOT/"configs/evaluation/paper_aligned_v1/scienceworld.json").read_text()))
    def test_wrong_horizon_rejected(self):
        p=json.loads((ROOT/"configs/evaluation/paper_aligned_v1/scienceworld.json").read_text());p["maximum_interaction_turns"]=100
        with self.assertRaises(ValueError):validate_profile(p)
    def test_flat_sort(self):self.assertEqual(canonicalize_observation("room\n\ta zebra\n\tan apple\n"),"room\n\ta zebra\n\tan apple\n")
    def test_flat_sort_reversed(self):self.assertEqual(canonicalize_observation("room\n\tan apple\n\ta zebra"),"room\n\ta zebra\n\tan apple")
    def test_nested_unchanged(self):
        text="room\n\tan apple\n\ta box\n\t\tinside"
        self.assertEqual(canonicalize_observation(text),text)
    def test_numeric_menu_preserved(self):
        text="Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n0:\tsame label\n1:\tsame label\n"
        self.assertEqual(canonicalize_observation(text),text);self.assertEqual(native_menu_actions(text),["0","1"])
    def test_numbers_outside_menu_not_actions(self):self.assertEqual(native_menu_actions("0:\tnot a native menu"),[])
    def test_official_canonical_function_equivalent(self):
        # Independent pinned official pure function, no Ray worker instantiation.
        import ast
        p=Path("/data1/shn/tmp/paper-aligned-audit/official/agent_system/environments/env_package/scienceworld/envs.py")
        if not p.exists():self.skipTest("Supply pinned reference source for cross-source check")
        tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="canonicalize_observation")
        ns={};exec("import re\n"+ast.unparse(node),ns)
        for t in ["room\n\tzebra\n\ta x","room\n\tan z\n\ta a\n","0:\tx\n1:\ty","r\n\t\tnested"]:
            self.assertEqual(canonicalize_observation(t),ns["canonicalize_observation"](t))


class SeedTests(unittest.TestCase):
    def test_deterministic(self):self.assertEqual(request_seed("sciworld_1",0),request_seed("sciworld_1",0))
    def test_rollout_changes(self):self.assertNotEqual(request_seed("sciworld_1",0,0),request_seed("sciworld_1",0,1))
    def test_turn_changes(self):self.assertNotEqual(request_seed("sciworld_1",0),request_seed("sciworld_1",1))
    def test_review_does_not_shift_draft(self):
        original=request_seed("sciworld_1",1)
        for i in range(3):request_seed("sciworld_1",0,role="review",subcall=i)
        self.assertEqual(original,request_seed("sciworld_1",1))
    def test_review_namespace_distinct(self):self.assertNotEqual(request_seed("sciworld_1",0),request_seed("sciworld_1",0,role="review"))
    def test_seed_range(self):self.assertTrue(0<=request_seed("sciworld_1",0)<2**63)
    def test_pinned_official_function_equivalence(self):
        p=Path("/data1/shn/tmp/paper-aligned-audit/official/agent_system/evaluation/seeding.py")
        if not p.exists():self.skipTest("Pinned reference source not supplied")
        spec=importlib.util.spec_from_file_location("official_seeds",p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for seed in range(8):
            for turn in (0,1,19):
                self.assertEqual(request_seed("sciworld_742",turn,seed),module.derive_request_seed(benchmark="scienceworld",task_seed=0,rollout_seed=seed,task_key="sciworld_742",turn_index=turn))


class ObserverTests(unittest.TestCase):
    def test_requests_keyword_url_is_preserved(self):
        import requests
        response=type("Response",(),{"json":lambda self:{"usage":{"prompt_tokens":1}},"status_code":200})()
        with tempfile.TemporaryDirectory() as directory,patch.object(requests.sessions.Session,"request",return_value=response) as original:
            with observe_wm("http://wm-unit",Path(directory)/"wire.jsonl",lambda:"unit") as rows:
                self.assertIs(requests.get("http://environment-unit/create"),response)
                self.assertEqual(rows,[])
                self.assertIs(requests.post("http://wm-unit/v1/chat/completions",json={"model":"unit"}),response)
                self.assertEqual(len(rows),1)
            self.assertIs(requests.sessions.Session.request,original)


class PartitionTests(unittest.TestCase):
    def setUp(self):self.m=json.loads((ROOT/"docs/phase3/split_manifest.json").read_text());self.p=prepare(self.m)
    def test_30_distinct_calibration_groups(self):self.assertEqual(len({r["goal_group"] for r in self.p["calibration_tasks"]}),30)
    def test_five_validation_groups(self):self.assertEqual(len({r["goal_group"] for r in self.p["validation_tasks"]}),5)
    def test_goal_groups_disjoint(self):self.assertFalse({r["goal_group"] for r in self.p["calibration_tasks"]}&{r["goal_group"] for r in self.p["validation_tasks"]})
    def test_no_test_ids(self):
        ids={r["task_id"] for r in self.p["calibration_tasks"]+self.p["validation_tasks"]}
        self.assertFalse(ids&{r["task_id"] for r in self.m["partitions"]["test"]})
    def test_exposed_groups_not_validation(self):self.assertFalse(set(self.p["prior_exposed_goal_groups"])&{r["goal_group"] for r in self.p["validation_tasks"]})


class ScoreTests(unittest.TestCase):
    def result(self,score=0,n=20):return {"score":score,"steps":[{}]*n}
    def test_horizon_complete(self):
        m=episode_metrics(self.result(),{"done":True,"native_done":False});self.assertTrue(m["horizon_truncated"])
    def test_native_terminal(self):self.assertTrue(episode_metrics(self.result(50,2),{"done":True,"native_done":True})["native_terminated"])
    def test_negative_score_not_clipped(self):self.assertEqual(episode_metrics(self.result(-100,1),{"done":True})["task_score"],-100)
    def test_raw_score_not_divided(self):self.assertEqual(episode_metrics(self.result(50),{"done":True})["task_score"],50)
    def test_success_native_score(self):self.assertTrue(episode_metrics(self.result(100,2),{"done":True})["success_score_ge100"])
    def test_partial_not_evaluation(self):
        with self.assertRaises(ValueError):episode_metrics(self.result(0,5),{"done":False})
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):episode_metrics(self.result(float("nan")),{"done":True})
    def test_no_missing_usage_as_zero(self):
        with self.assertRaises(ValueError):costs([],[],[{"usage":None}],1)
    def test_zero_calls_valid(self):self.assertEqual(costs([],[],[],1)["total_model_calls"],0)


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/"checkpoints/qwen3-4b",local_files_only=True)
    def test_left_suffix_response_tokens_only(self):
        tok=self.tokenizer;prompt=tok.apply_chat_template([{"role":"user","content":"old text "*50+"Write one sentence."}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        response="Premise: The door is closed.\nExpected Consequence: The door will open.\nAction: open door"
        full=tok(prompt+response,add_special_tokens=False);prefix=tok(prompt,add_special_tokens=False)["input_ids"]
        ids=full["input_ids"][len(prefix):];_,offsets=response_alignment(tok,prompt,prefix[10:],ids,left_truncated_tokens=10)
        self.assertEqual(len(offsets),len(ids));self.assertTrue(all(a>=0 and b<=len(response) for a,b in offsets))
    def test_left_suffix_with_unicode_boundary(self):
        tok=self.tokenizer;prompt=tok.apply_chat_template([{"role":"user","content":"测试α🙂"*40}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tok(prompt,add_special_tokens=False)["input_ids"];response="Premise: The air is visible."
        ids=tok(prompt+response,add_special_tokens=False)["input_ids"][len(prefix):]
        for cut in [1,7,11,30]:self.assertEqual(response_alignment(tok,prompt,prefix[cut:],ids,left_truncated_tokens=cut)[0],response)
    def test_wrong_removed_count_fails(self):
        tok=self.tokenizer;prompt="hello ";prefix=tok(prompt,add_special_tokens=False)["input_ids"]
        with self.assertRaises(ValueError):response_alignment(tok,prompt,prefix,[1],left_truncated_tokens=1)
    def test_phase2_golden_signals_preserved(self):
        from internal_signals.provider import TokenNLLProvider
        config=json.loads((ROOT/"configs/internal_signals/scienceworld_token_nll_v1.json").read_text())
        provider=TokenNLLProvider(self.tokenizer,config)
        for line in (ROOT/"docs/phase2/steps.jsonl").read_text().splitlines():
            row=json.loads(line);signal=provider.extract(row["generation_record"]).to_dict()
            for key in ("local_uncertainty","global_uncertainty","premise_uncertainty","consequence_uncertainty","missing_span"):
                self.assertEqual(signal[key],row["internal_signals"][key])


class InspectionRuleTests(unittest.TestCase):
    def test_current_room_required(self):
        from experience.synthesis import applicable
        self.assertIsNone(applicable("room_inspection_look","No known action matches that input.","look around"))
    def test_only_actual_room_reading_confirms(self):
        from experience.synthesis import confirms
        self.assertTrue(confirms("room_inspection_look","This room is called the bathroom.\nmore","This room is called the bathroom.\nold"))
        self.assertFalse(confirms("room_inspection_look","This room is called the kitchen.","This room is called the bathroom."))
    def test_other_actions_do_not_match(self):
        from experience.synthesis import applicable
        self.assertIsNone(applicable("room_inspection_look","This room is called the bathroom.","look at thermometer"))


if __name__=="__main__":unittest.main(verbosity=2)
