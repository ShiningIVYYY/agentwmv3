"""Synthetic guard tests only; no simulated scientific outcomes or model metrics."""
import copy,json,sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from internal_signals.linear_utility_training import readiness,train_head
PROFILE=json.loads((ROOT/'configs/evaluation/linear_utility_v1.json').read_text())

def fixture():
    rs=[]
    for role,n,p,groups in [('probe_train',500,50,20),('probe_validation',200,20,12),('gate_calibration',60,5,12)]:
        rs += [dict(role=role,case_id=role+str(i),goal_group=role+str(i%groups),utility_help=int(i<p)) for i in range(n)]
    return rs

class Guards(unittest.TestCase):
    def test_full_predeclared_targets(self):self.assertTrue(readiness(fixture(),'utility_help',PROFILE)['ready'])
    def test_five_positives_insufficient(self):
        rs=fixture()
        for r in rs:
            if r['role']=='probe_train':r['utility_help']=int(int(r['case_id'].replace('probe_train',''))<5)
        self.assertFalse(readiness(rs,'utility_help',PROFILE)['ready'])
    def test_unknown_not_negative(self):
        rs=fixture()
        for r in rs:
            if r['role']=='probe_train':r['utility_help']=None
        self.assertEqual(readiness(rs,'utility_help',PROFILE)['counts']['probe_train']['negative'],0)
    def test_same_goal_cross_role_rejected(self):
        rs=fixture();rs[500]['goal_group']=rs[0]['goal_group']
        with self.assertRaises(ValueError):readiness(rs,'utility_help',PROFILE)
    def test_duplicate_state_not_independent(self):
        rs=fixture();rs.append(copy.deepcopy(rs[0]))
        with self.assertRaises(ValueError):readiness(rs,'utility_help',PROFILE)
    def test_insufficient_does_not_call_optimizer(self):
        with tempfile.TemporaryDirectory() as d,patch('internal_signals.linear_utility_training._fit_head',side_effect=AssertionError('must not train')):
            result=train_head(ROOT,d,[],'utility_help',PROFILE['layers'],PROFILE)
            self.assertEqual(result['status'],'insufficient_labels');self.assertFalse(result['deployment_allowed'])
    def test_harm_head_not_fabricated(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):train_head(ROOT,d,[],'utility_harm',PROFILE['layers'],PROFILE)
if __name__=='__main__':unittest.main()
