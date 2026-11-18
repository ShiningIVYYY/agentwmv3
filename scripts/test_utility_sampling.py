"""Synthetic unit checks are not task-effectiveness evidence."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.utility_sampling import select_cases,aggregate,public_flags
from scripts.run_utility_sampling import select_pair,mine_training_candidates
import copy


class Sampling(unittest.TestCase):
    def cases(self):return [dict(case_id=str(i),step=i,observation=str(i),commands=['look'],draft_action='look',past=[],sampling={'weight':1+i%4}) for i in range(50)]
    def test_temporal_coverage(self):
        s=select_cases(self.cases(),'test');self.assertEqual(len(s),6)
        self.assertEqual({tuple(x['window']) for x in s},{(0,9),(10,29),(30,49)})
    def test_selection_does_not_use_future(self):
        a=self.cases();b=self.cases()
        for x in b:x['future_success']=True;x['actual']={'reward':100}
        self.assertEqual(select_cases(a,'test'),select_cases(b,'test'))
    def test_inclusion_probability_sums_two(self):
        # Full equal-weight pool: sampling two distinct records gives2/N inclusion.
        a=self.cases()[:10]
        for x in a:x['sampling']['weight']=1
        self.assertTrue(all(abs(s['inclusion_probability']-.2)<1e-8 for s in select_cases(a,'test')))
    def test_one_case(self):self.assertEqual(select_cases(self.cases()[:1],'x')[0]['inclusion_probability'],1.)
    def test_duplicate_contexts(self):
        a=self.cases()[:10]
        for c in a:c['observation']='same'
        self.assertEqual(len(select_cases(a,'x')),1)
    def test_stable_help(self):self.assertEqual(aggregate([1,1,1,0,0],True)['label'],'help')
    def test_seed_sensitive_not_positive(self):self.assertIsNone(aggregate([1,0,0],True)['utility_help'])
    def test_conflicting_not_positive(self):self.assertIsNone(aggregate([1,1,-1],True)['utility_help'])
    def test_harm(self):self.assertEqual(aggregate([-1,-1,-1,0,0],True)['utility_harm'],1)
    def test_no_fake_unrun_seeds(self):self.assertIsNone(aggregate([0],False,False)['utility_help'])
    def test_verified_unchanged_is_not_action_correct(self):
        x=aggregate([0],False,True);self.assertEqual(x['label'],'neutral');self.assertNotIn('action_correct',x)
    def test_external_unknown_not_conflict(self):
        x=public_flags('look',['look'],'room',[],dict(conflict='unknown',supporting_evidence_ids=[],reliability='unknown'),.1,.1,dict(tau_local=.5,tau_global=.5))
        self.assertFalse(x['flags']['verified_external_conflict'])
    def pair(self):
        n=self.cases()
        for x in n:x.update(case_id='N'+x['case_id'],state_sha256=str(x['step']),actual={'step':x['step']},executed_action='look')
        a=copy.deepcopy(n)
        for x in a:x['case_id']='A'+str(x['step'])
        a[5]['executed_action']='inventory'
        return n,a
    def test_shared_prefix_difference_is_candidate_not_help(self):
        n,a=self.pair();s,f=select_pair(n,a,'probe_train','x')
        self.assertTrue(f['exact_common_state']);self.assertIn('N5',[x['case_id'] for x in s]);self.assertNotIn('utility_help',f)
    def test_heldout_has_no_outcome_mining(self):
        n,a=self.pair();s,f=select_pair(n,a,'probe_validation','x')
        before=copy.deepcopy(s);after=mine_training_candidates(s,n,a,'probe_validation',False,True)
        self.assertEqual(before,after);self.assertTrue(all(not x['selection_uses_future_outcomes'] for x in after))
    def test_goal_roles_preserved(self):
        import json
        d=json.loads((ROOT/'docs/activation_v1/split_manifest.json').read_text());seen=set()
        for role,rs in d['partitions'].items():
            groups={r['goal_group'] for r in rs};self.assertFalse(seen&groups);seen|=groups
        self.assertFalse(seen&{r['goal_group'] for r in d['final_test']})


if __name__=='__main__':unittest.main()
