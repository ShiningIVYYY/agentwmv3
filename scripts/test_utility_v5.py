"""Synthetic selection guards plus actual prior-corpus/provenance audits."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.utility_sampling_v5 import context_key,changed_candidates,validation_selection
from evaluation.utility_labels_v3 import aggregate
from scripts.run_utility_expansion_v5 import prior,tasks_for
from scripts.run_utility_expansion import audit_records
from internal_signals.linear_utility_training import readiness
from scripts.prepare_utility_v5 import allocate_roles


def fixture():
    return [dict(case_id=str(i),goal_group='UNIT',step=i,source_policy='always',feature_valid=True,
        draft_action='look',executed_action='inventory',metadata={'generation_record':{'prompt_text':str(i)}}) for i in range(50)]


class SelectionGuards(unittest.TestCase):
    def test_train_enrichment_forbidden_in_validation(self):
        with self.assertRaises(ValueError):changed_candidates(fixture(),'probe_validation','UNIT',set())
    def test_train_enrichment_deduplicates_contexts(self):
        cs=fixture();cs[1]['metadata']=copy.deepcopy(cs[0]['metadata'])
        chosen=changed_candidates(cs,'probe_train','UNIT',{context_key(cs[0])},20)
        self.assertFalse({'0','1'}&{s['case_id'] for s in chosen})
    def test_enriched_state_is_not_a_positive_label(self):
        chosen=changed_candidates(fixture(),'probe_train','UNIT',set())
        self.assertTrue(all('utility_help' not in s and s['selection_uses_future_outcomes'] for s in chosen))
    def test_validation_ignores_post_action_outcomes(self):
        cs=fixture();before=validation_selection(cs,'UNIT')
        for c in cs:c.update(executed_action='made_up_UNIT_ONLY',actual={'success':True},review={'decision':'REPLACE'},utility_help=1)
        self.assertEqual(validation_selection(cs,'UNIT'),before)
    def test_uniform_probability_and_count(self):
        s=validation_selection(fixture(),'UNIT');self.assertEqual(len(s),12)
        self.assertTrue(all(x['inclusion_probability']==4/x['pool_size'] and not x['selection_uses_future_outcomes'] for x in s))
    def test_missing_features_excluded(self):
        cs=fixture()
        for c in cs:c['feature_valid']=False
        self.assertEqual(validation_selection(cs,'UNIT'),[]);self.assertEqual(changed_candidates(cs,'probe_train','UNIT',set()),[])
    def test_validation_context_not_counted_again_in_later_window(self):
        cs=fixture()
        for c in cs:c['metadata']['generation_record']['prompt_text']='same context'
        self.assertEqual(len(validation_selection(cs,'UNIT')),1)
    def test_label_standard_unchanged(self):
        self.assertEqual(aggregate([1,1,1,0,0],True)['label'],'help')
        self.assertEqual(aggregate([1,1,1,0,-1],True)['label'],'uncertain')
        self.assertEqual(aggregate([0,0,0,0,1],True)['label'],'uncertain')
    def test_role_allocation_cannot_alias_family_cycle(self):
        selected=[dict(family=f,goal_group=f+str(i)) for f,n in [('a',53),('b',31),('c',53)] for i in range(n)]
        allocation=allocate_roles(selected)
        self.assertEqual(len(allocation['probe_train']),80);self.assertEqual(len(allocation['probe_validation']),40)
        for role in ['probe_train','probe_validation']:self.assertEqual({r['family'] for r in allocation[role]},{'a','b','c'})


class RegisteredCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split=json.loads((ROOT/'docs/utility_expansion_v5/split_manifest.json').read_text())
        cls.parent=json.loads((ROOT/'docs/recovery_v4/split_manifest.json').read_text())
        cls.cs,cls.ls,cls.ps=prior(verify=True)
    def test_all_prior_roles_preserved(self):
        for role,rs in self.parent['partitions'].items():self.assertEqual(self.split['partitions'][role][:len(rs)],rs)
        self.assertEqual(self.parent['final_test'],self.split['final_test'])
    def test_new_fixed_denominators(self):
        for phase,n in [('densify_train',100),('new_train',80),('new_validation',40)]:self.assertEqual(len(tasks_for(phase,self.split)),n)
        self.assertEqual(len(self.split['fresh_independent_dev_task_ids']),12)
    def test_no_cross_role_or_final_test_groups(self):
        seen=set()
        for rs in self.split['partitions'].values():
            gs={r['goal_group'] for r in rs};self.assertEqual(len(gs),len(rs));self.assertFalse(gs&seen);seen|=gs
        self.assertFalse(seen&{r['goal_group'] for r in self.split['final_test']})
    def test_all_real_prior_data_retained(self):
        self.assertEqual(len(self.cs),9628);self.assertEqual(len(self.ls),782)
        self.assertEqual(sum(l['label']=='help' and l['role']=='probe_train' for l in self.ls),10)
        self.assertTrue(audit_records(self.cs,self.ls,self.split)['no_future_features'])
    def test_old_small_data_does_not_trigger_training(self):
        p=json.loads((ROOT/'configs/evaluation/linear_utility_v1.json').read_text())
        r=readiness(self.ls,'utility_help',p);self.assertFalse(r['ready'])
        self.assertEqual(r['counts']['probe_train']['positive'],10)
    def test_new_budget_does_not_erase_prior_spending(self):
        r=json.loads((ROOT/'docs/utility_expansion_v5/prior_corpus.json').read_text())
        self.assertEqual(r['known_cumulative_cost']['tokens'],48699305)
        self.assertEqual(r['conservative_cumulative_cost']['tokens']-r['known_cumulative_cost']['tokens'],1890)


if __name__=='__main__':unittest.main()
