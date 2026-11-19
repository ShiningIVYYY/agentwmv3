"""Real-record regression plus clearly separated synthetic unit boundary tests."""
import copy,json,math,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from internal_signals.actual_token_alignment import response_alignment,field_masks
from internal_signals.hidden_states import response_alignment as legacy_alignment
from internal_signals.actual_token_nll import ActualTokenNLLProvider
from evaluation.recovery_collection import select_pair,mine_candidates


class Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
        cls.r=json.loads((ROOT/'docs/recovery_v4/failing_generation_record.json').read_text())
        cls.cfg=json.loads((ROOT/'configs/internal_signals/qwen35_9b_activation_v1.json').read_text())
    def test_real_failure_reproduces_legacy(self):
        with self.assertRaises(ValueError):legacy_alignment(self.t,self.r['prompt_text'],self.r['prompt_ids'],self.r['response_ids'])
    def test_real_actual_ids_supported_without_replacement(self):
        r=copy.deepcopy(self.r);text,offsets=response_alignment(self.t,r['prompt_text'],r['prompt_ids'],r['response_ids'])
        self.assertEqual(text,r['response']);self.assertEqual(len(offsets),len(r['response_ids']));self.assertEqual(r,self.r)
        self.assertEqual(offsets[0][0],0);self.assertEqual(offsets[-1][1],len(text))
        self.assertTrue(all(offsets[i][1]==offsets[i+1][0] for i in range(len(offsets)-1)))
    def test_canonical_path_identical(self):
        r=self.r;ids=self.t(r['response'],add_special_tokens=False)['input_ids']
        self.assertEqual(response_alignment(self.t,r['prompt_text'],r['prompt_ids'],ids),legacy_alignment(self.t,r['prompt_text'],r['prompt_ids'],ids))
    def test_real_nll_uses_original_denominator(self):
        s=ActualTokenNLLProvider(self.t,self.cfg).extract(self.r)
        self.assertEqual(s.status,'ok');self.assertEqual(s.global_uncertainty,math.fsum(self.r['token_nll'])/len(self.r['response_ids']))
        self.assertFalse(s.missing_span)
    def test_prompt_tamper_rejected(self):
        with self.assertRaises(ValueError):response_alignment(self.t,self.r['prompt_text'],[0]+self.r['prompt_ids'][1:],self.r['response_ids'])
    def test_special_token_rejected(self):
        with self.assertRaises(ValueError):response_alignment(self.t,self.r['prompt_text'],self.r['prompt_ids'],self.r['response_ids']+[self.t.eos_token_id])
    def test_truncation_is_missing_not_zero(self):
        r=copy.deepcopy(self.r);r['truncated']=True;s=ActualTokenNLLProvider(self.t,self.cfg).extract(r)
        self.assertTrue(s.missing_span);self.assertIsNone(s.local_uncertainty);self.assertIsNone(s.global_uncertainty)
    def test_invalid_nll_length_fails_closed(self):
        r=copy.deepcopy(self.r);r['token_nll']=r['token_nll'][:-1];s=ActualTokenNLLProvider(self.t,self.cfg).extract(r)
        self.assertTrue(s.missing_span);self.assertIsNone(s.local_uncertainty)
    def test_actual_field_indices_within_real_ids(self):
        from types import SimpleNamespace
        m=field_masks(SimpleNamespace(tokenizer=self.t),{'generation_record':self.r})
        self.assertEqual(set(m),{'premise','consequence','action'});self.assertTrue(all(m.values()))
        self.assertTrue(all(0<=i<len(self.r['response_ids']) for ids in m.values() for i in ids))
    def test_unsupported_noncanonical_unicode_fails_closed(self):
        class Stub:
            all_special_ids=[]
            def __call__(self,s,**kw):return {'input_ids':[1] if s=='P' else [1,3,4],'offset_mapping':[]}
            def decode(self,ids,**kw):return 'é'
        with self.assertRaises(ValueError):response_alignment(Stub(),'P',[1],[2])
    def test_quarantined_state_not_selected(self):
        cs=[dict(case_id=str(i),step=i,feature_valid=i!=0,observation=str(i),commands=['look'],draft_action='look',past=[],sampling={'weight':1},
            executed_action='look',state_sha256=str(i),actual={'x':i}) for i in range(3)]
        selected,_=select_pair(cs,copy.deepcopy(cs),'probe_train','UNIT_ONLY')
        selected=mine_candidates(selected,cs,copy.deepcopy(cs),'probe_train',False,True)
        self.assertNotIn('0',[s['case_id'] for s in selected])
    def test_fixed_goal_roles_and_count(self):
        s=json.loads((ROOT/'docs/recovery_v4/split_manifest.json').read_text());old=json.loads((ROOT/'docs/utility_expansion_v3/split_manifest.json').read_text())
        self.assertEqual(len(s['partitions']['probe_train']),100);self.assertEqual(s['partitions']['probe_train'][:60],old['partitions']['probe_train'])
        for role in ['probe_validation','gate_calibration','independent_dev']:self.assertEqual(s['partitions'][role],old['partitions'][role])
        seen=set()
        for rs in s['partitions'].values():
            g={r['goal_group'] for r in rs};self.assertFalse(seen&g);seen|=g
        self.assertFalse(seen&{r['goal_group'] for r in s['final_test']})
    def test_all_incomplete_tasks_restarted_not_chosen_by_score(self):
        s=json.loads((ROOT/'docs/recovery_v4/split_manifest.json').read_text());r=json.loads((ROOT/'docs/recovery_v4/recovery_receipt.json').read_text())
        self.assertEqual(len(r['completed_task_ids']),11);self.assertEqual(len(r['remaining_original_task_ids']),13)
        self.assertFalse(set(r['completed_task_ids'])&set(r['remaining_original_task_ids']))
        self.assertEqual(set(r['completed_task_ids'])|set(r['remaining_original_task_ids']),set(s['expansion_train_task_ids']))
    def test_unknown_cost_not_fabricated_or_budget_reset(self):
        r=json.loads((ROOT/'docs/recovery_v4/recovery_receipt.json').read_text());c=json.loads((ROOT/'configs/evaluation/recovery_v4.json').read_text())
        self.assertEqual(len(r['unreceipted_requests']),2);self.assertEqual(sum(u['max_tokens'] for u in r['unreceipted_requests']),1890)
        self.assertTrue(all(u['actual_tokens'] is None for u in r['unreceipted_requests']))
        self.assertEqual(c['global_limits']['tokens']+r['budget_charge_upper_bound']['tokens']+c.get('auxiliary_cost',{}).get('tokens',0),72000000)
        self.assertEqual(c['recovery_limits']['tokens']+r['budget_charge_upper_bound']['tokens']+c.get('auxiliary_cost',{}).get('tokens',0),18000000)
        self.assertEqual(c['original_recovery_deadline'],r['original_deadline_unix'])


if __name__=='__main__':unittest.main()
