"""Regression contracts for actual parallel failure; MOCK success is unit-only."""
import ast,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_alfworld_episode import validate,check_globals
from scripts.run_alfworld_parallel import ReceiptLedger,ResourceLimit,verify
from worldmodel.alfworld_environment import fingerprint
import evaluation.alfworld_episode as ep


class ParallelContracts(unittest.TestCase):
    def test_original_episode_ast(self):
        digest=validate((ROOT/'scripts/run_alfworld_pilot.py').read_text(),(ROOT/'evaluation/alfworld_episode.py').read_text())
        self.assertEqual(digest,json.loads((ROOT/'docs/alfworld_pilot/parallel_episode_identity.json').read_text())['episode_ast_sha256'])
    def test_all_referenced_globals_bound(self):self.assertTrue(check_globals((ROOT/'evaluation/alfworld_episode.py').read_text(),vars(ep)))
    def test_regression_catches_missing_asdict(self):
        ns=dict(vars(ep));ns.pop('asdict')
        with self.assertRaises(ValueError):check_globals((ROOT/'evaluation/alfworld_episode.py').read_text(),ns)
    def test_actual_receipt_persisted_after_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ledger=ReceiptLedger(p/'ledger.json',p/'STOP',time.time()+60,dict(tokens=100,model_calls=10,transition_queries=10,audit_capture_bundles=10))
            (p/'STOP').write_text('stop');ledger.captures(3)
            self.assertEqual(json.loads((p/'ledger.json').read_text())['audit_capture_bundles'],3)
            with self.assertRaises(ResourceLimit):ledger.begin_model(2,{})
    def test_complete_trace_serialization_mock_only(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);row=dict(task_id='UNIT_ONLY',goal_group='unit',family='unit',game='unit')
            initial=dict(observation='unit observation',admissible_commands=['look'],reward=0.,score=0.,success=False,done=False,truncated=False,moves=0,formal_steps=0,private_state_sha256='UNIT_ONLY',action=None)
            class UnitEnv:
                def __init__(self,*_):self.goal='unit goal';self.initial=initial;self.current=dict(initial);self.capture_count=1;self.pid=-1;self.steps=0
                def step(self,action):
                    self.steps+=1;self.capture_count+=1;self.current=dict(initial,action=action,success=True,score=100.,reward=1.,done=True,formal_steps=1);return self.current
                def close(self):pass
            class UnitActor:
                def count(self,_):return 1
                def generate_role(self,*_):return 'Premise: Unit condition.\nExpected Consequence: Unit effect.\nAction: look',dict(decision_format_valid=True,usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))
            class UnusedMemory:
                def evaluate(self,_):raise AssertionError('Never must not retrieve')
            source=Path('outputs/alfworld_five_policy_v1');old=json.loads((ROOT/source/'manifest.json').read_text());p=old['profile'];gate=json.loads((ROOT/source/'gate_config.json').read_text())
            ledger=ReceiptLedger(out/'ledger.json',out/'STOP',time.time()+60,dict(tokens=1000,model_calls=10,transition_queries=10,audit_capture_bundles=10))
            original=ep.AlfEnvironment;ep.AlfEnvironment=UnitEnv
            try:
                result=ep.run_episode(row=row,policy='never',seed=0,actor=UnitActor(),memory=UnusedMemory(),gate_config=gate,p=p,out=out,
                    a=SimpleNamespace(env_python='UNIT_ONLY',data_root='UNIT_ONLY'),ledger=ledger,native_goals={'UNIT_ONLY':dict(goal='unit goal',initial_sha256=fingerprint(initial))},episodes=[])
            finally:ep.AlfEnvironment=original
            trace=json.loads((out/'episodes/UNIT_ONLY/never/steps.jsonl').read_text())
            self.assertEqual(trace['resolved_external']['coverage'],'unknown');self.assertEqual(result['formal_steps'],1)
            # The TemporaryDirectory (including this MOCK success) is deleted and
            # never exported or counted as an environment/model experiment.


if __name__=='__main__':unittest.main()
