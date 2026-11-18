"""Regression contracts; synthetic unit inputs never become benchmark labels."""
import ast,itertools,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.utility_labels_v3 import aggregate,relabel
from scripts.run_utility_expansion import feasibility,audit_records


class Expansion(unittest.TestCase):
    def test_five_mixed_regressions(self):
        for d in [[0,0,1,1,0],[1,0,0,-1,0],[0,1,-1,0,0],[0,0,-1,0,-1]]:
            r=aggregate(d,True);self.assertEqual(r['label'],'uncertain');self.assertIsNone(r['utility_help'])
    def test_only_all_zero_is_neutral(self):
        for n in [3,5]:
            for ds in itertools.product([-1,0,1],repeat=n):
                r=aggregate(list(ds),True)
                self.assertEqual(r['label']=='neutral',all(d==0 for d in ds))
    def test_positive_standard_unchanged(self):
        for ds in [[0,1,1,1,1],[1,1,0,1,1],[1,0,1,0,1]]:self.assertEqual(aggregate(ds,True)['label'],'help')
    def test_negative_veto(self):self.assertIsNone(aggregate([1,1,1,1,-1],True)['utility_help'])
    def test_incomplete_not_negative(self):self.assertIsNone(aggregate([1,0],True)['utility_help'])
    def test_unchanged_not_action_correct(self):
        r=aggregate([0],False,True);self.assertEqual(r['label'],'neutral');self.assertNotIn('action_correct',r)
    def test_corrected_copy_does_not_mutate_original(self):
        old=dict(label='neutral',deltas=[0,0,1,1,0],changed=True,utility_help=0,utility_harm=0)
        result=relabel(old);self.assertEqual(old['label'],'neutral');self.assertEqual(result['label'],'uncertain');self.assertEqual(result['legacy_label'],'neutral')
    def test_no_threshold_lowering(self):
        ls=[dict(role='probe_train',label='help',task_id=str(i%3)) for i in range(4)]
        self.assertFalse(feasibility(ls)['expand']);ls.append(dict(role='probe_train',label='help',task_id='new'))
        self.assertTrue(feasibility(ls)['expand'])
    def test_heldout_cannot_trigger_training_expansion(self):
        ls=[dict(role='probe_validation',label='help',task_id=str(i)) for i in range(20)]
        self.assertFalse(feasibility(ls)['expand'])
    def test_old_roles_unchanged(self):
        p=json.loads((ROOT/'docs/activation_v1/split_manifest.json').read_text());s=json.loads((ROOT/'docs/utility_expansion_v3/split_manifest.json').read_text())
        self.assertEqual(s['partitions']['probe_train'][:36],p['partitions']['probe_train'])
        for role in ['probe_validation','gate_calibration','independent_dev']:self.assertEqual(p['partitions'][role],s['partitions'][role])
    def test_new_goals_disjoint(self):
        p=json.loads((ROOT/'docs/activation_v1/split_manifest.json').read_text());s=json.loads((ROOT/'docs/utility_expansion_v3/split_manifest.json').read_text())
        ids=set(s['expansion_train_task_ids']);new=[r for r in s['partitions']['probe_train'] if r['task_id'] in ids]
        self.assertEqual(len(new),24)
        prior={r['goal_group'] for rs in p['partitions'].values() for r in rs}|set(p['excluded_goal_groups'])|{r['goal_group'] for r in p['final_test']}
        self.assertFalse(prior&{r['goal_group'] for r in new});self.assertTrue(all(r['source_split']=='train' for r in new))
    def test_counterfactual_execution_unchanged(self):
        def tree(path):
            t=ast.parse((ROOT/path).read_text());return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='label_state'))
        self.assertEqual(tree('scripts/run_utility_sampling.py'),tree('evaluation/utility_labeling_v3.py'))
    def test_old_helpful_candidates_preserved(self):
        p=json.loads((ROOT/'docs/utility_expansion_v3/prior_corpus.json').read_text())
        self.assertTrue(p['old_help_preserved']);self.assertEqual(p['corrected_label_counts']['help'],4)
        self.assertEqual(len(p['corrections']),5)
    def test_future_time_is_rejected(self):
        split=dict(partitions={'probe_train':[dict(task_id='x',goal_group='g')],'probe_validation':[],'gate_calibration':[]})
        c=dict(case_id='c',task_id='x',role='probe_train',goal_group='g',feature_available_at_ns=2,label_observed_at_ns=1)
        with self.assertRaises(ValueError):audit_records([c],[],split)
    def test_role_reassignment_is_rejected(self):
        split=dict(partitions={'probe_train':[],'probe_validation':[dict(task_id='x',goal_group='g')],'gate_calibration':[]})
        c=dict(case_id='c',task_id='x',role='probe_train',goal_group='g',feature_available_at_ns=1,label_observed_at_ns=2)
        with self.assertRaises(ValueError):audit_records([c],[],split)


if __name__=='__main__':unittest.main()
