#!/usr/bin/env python3
"""CPU fixtures test engineering contracts only; never scientific results."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import Ledger,ResourceLimit,request_seed,run_episode,verify_frozen,verify_model_assets
from scripts.run_phase57_replay import request_seed as old_seed
from scripts.prepare_independent_dev_eval import select
from scripts.run_independent_dev import schedule
from experience.splits import goal_key


class FakeActor:
    config=dict(max_prompt_tokens=4096,max_new_tokens=512)
    def count(self,prompt):return 4
    def generate_role(self,prompt,role,seed,scored):
        if scored:raise AssertionError('Never should not compute NLL')
        return 'Premise: observable fixture\nExpected Consequence: inspect\nAction: look around',dict(
            usage=dict(prompt_tokens=4,completion_tokens=3,total_tokens=7),decision_format_valid=True,
            internal_cost=dict(generation_seconds=0,extraction_seconds=0,additional_backbone_forward_passes=0))


class FakeEnv:
    last=None
    def __init__(self,spec):
        FakeEnv.last=self;self.spec=spec;self.goal='fixture';self.closed=False;self.capture_count=0;self.pid=0
        self.simulator={'profile':'fixture'};self.current=dict(observation='fixture',raw_observation='fixture',reward=0,done=False,score=0,moves=0)
        self.initial=self.capture()
    def capture(self):
        self.capture_count+=1;return dict(transition=dict(self.current))
    def step(self,action):
        self.capture();self.current=dict(observation='observed',raw_observation='observed',reward=75,done=True,score=75,moves=1)
        self.capture();return dict(self.current,action=action)
    def close(self):self.closed=True


class Contracts(unittest.TestCase):
    def ledger(self,p,**limits):
        return Ledger(Path(p)/'ledger.json',Path(p)/'STOP',time.time()+60,
            dict(tokens=limits.get('tokens',10000),model_calls=10,transition_queries=100,audit_capture_bundles=10000))

    def test_offline_assets_match_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);(root/'docs/phase57_replay').mkdir(parents=True);(root/'model').mkdir()
            (root/'model/asset').write_bytes(b'unit fixture')
            expected=dict(model_id='fixture',revision='fixture',files={'asset':dict(bytes=12,sha256=hashlib.sha256(b'unit fixture').hexdigest())})
            (root/'docs/phase57_replay/assets_manifest.json').write_text(json.dumps(expected))
            self.assertEqual(verify_model_assets(root,root/'model')['status'],'all_pinned_assets_match')
            (root/'model/asset').write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_model_assets(root,root/'model')

    def test_seed_zero_preserves_frozen_scheme(self):
        for role in ('draft','review'):
            self.assertEqual(request_seed('fixture',7,role,0),old_seed('fixture',7,role))
            self.assertNotEqual(request_seed('fixture',7,role,0),request_seed('fixture',7,role,1))

    def test_seed_not_policy_or_scheduler_dependent(self):
        self.assertEqual(request_seed('fixture',1,'draft',1),request_seed('fixture',1,'draft',1))
        self.assertNotEqual(request_seed('fixture',1,'draft',1),request_seed('fixture',1,'review',1))

    def test_full_schedule_and_round1_reuse(self):
        p=dict(tasks=[dict(task_id='fixture'+str(i)) for i in range(12)],rollout_seeds=[0,1],round1_task_ids=['fixture'+str(i) for i in range(5)])
        s=schedule(p)
        self.assertEqual(len(s),24);self.assertEqual(sum(g['round']==1 for g in s),5)
        self.assertEqual(len({(g['task_id'],g['seed']) for g in s}),24)
        self.assertEqual([sum(g['worker']==i for g in s) for i in range(2)],[12,12])

    def test_token_reservation_prevents_call_and_charges_receipt(self):
        with tempfile.TemporaryDirectory() as p:
            l=self.ledger(p,tokens=20)
            with self.assertRaises(ResourceLimit):l.begin_model(21,{})
            self.assertIsNone(l.state['pending_model'])
            l.begin_model(20,{});l.finish_model(dict(usage=dict(total_tokens=7)))
            self.assertEqual(l.state['tokens'],7);self.assertEqual(l.state['model_calls'],1)
            with self.assertRaises(ResourceLimit):l.begin_model(20,{})

    def test_transition_reservations_release_unused(self):
        with tempfile.TemporaryDirectory() as p:
            l=self.ledger(p);l.begin_transition(20,{});l.finish_transition(2)
            self.assertEqual(l.state['transition_queries'],2)
            self.assertIsNone(l.state['pending_transition'])

    def test_interrupted_receipt_blocks_unsafe_resume(self):
        with tempfile.TemporaryDirectory() as p:
            l=self.ledger(p);l.begin_model(10,{})
            with self.assertRaisesRegex(ResourceLimit,'unknown receipt'):self.ledger(p)

    def test_stop_deadline_and_batch_caps(self):
        with tempfile.TemporaryDirectory() as p:
            l=self.ledger(p);l.deadline=time.time()-1
            with self.assertRaises(ResourceLimit):l.check('tokens',1)
            l.deadline=time.time()+60;Path(p,'STOP').write_text('stop')
            with self.assertRaises(ResourceLimit):l.check('tokens',1)

    def test_outcome_blind_selection_and_two_variants(self):
        rows=[dict(task_id=f'fixture{family}-{goal}-{v}',taskName='family'+str(family),goal_group=f'{family}-{goal}')
            for family in range(5) for goal in range(2) for v in range(3)]
        result=select(rows)
        again=select(list(reversed(rows)))
        self.assertEqual(result,again);self.assertEqual(len(result),12)
        self.assertEqual(len({r['taskName'] for r in result[:5]}),5)
        self.assertTrue(all(sum(r['goal_group']==g for r in result)<=2 for g in {r['goal_group'] for r in result}))

    def test_frozen_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as p:
            Path(p,'changed.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen method changed'):
                verify_frozen(Path(p),dict(method_hashes={'changed.py':'incorrect'}))

    def run_fixture(self,p,actor):
        profile=json.loads((ROOT/'configs/evaluation/phase59_independent_dev_v1.json').read_text())
        gate=json.loads((ROOT/profile['gate_config']).read_text())
        row=dict(task_id='sciworld_999999',taskName='fixture',variationIdx=0,goal_group=goal_key('fixture'),category='unit_fixture')
        memory=type('Memory',(),{'evaluate':lambda *a:(_ for _ in ()).throw(AssertionError('Unused Memory'))})()
        with patch('evaluation.independent_dev.ReplayScienceEnvironment',FakeEnv):
            return run_episode(actor,memory,row,'never',0,profile,gate,Path(p)/'episode',self.ledger(p))

    def test_episode_score_termination_cost_and_unused_modules(self):
        with tempfile.TemporaryDirectory() as p:
            result=self.run_fixture(p,FakeActor())
            self.assertEqual(result['task_score'],75);self.assertFalse(result['success'])
            self.assertEqual(result['formal_environment_actions'],1);self.assertEqual(result['lookahead_attempts'],0)
            self.assertEqual(result['total_tokens'],7);self.assertTrue(FakeEnv.last.closed)
            state=json.loads(Path(p,'ledger.json').read_text())
            self.assertEqual(state['transition_queries'],2);self.assertEqual(state['model_calls'],1)

    def test_exception_closes_environment_and_preserves_partial(self):
        actor=FakeActor();actor.generate_role=lambda *args:(_ for _ in ()).throw(RuntimeError('fixture interruption'))
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):self.run_fixture(p,actor)
            self.assertTrue(FakeEnv.last.closed);self.assertFalse(Path(p,'episode/complete.json').exists())
            self.assertIsNotNone(json.loads(Path(p,'ledger.json').read_text())['pending_model'])

if __name__=='__main__':unittest.main()
