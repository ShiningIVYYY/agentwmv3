"""Resource-boundary unit tests and actual retained-data audits."""
import json,sys,unittest,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.gpu_phase_resources import candidates,acquire
from scripts.recover_utility_v5 import prior,base,CONFIG
from scripts.run_utility_expansion import audit_records
from internal_signals.linear_utility_training import readiness


class ResourceGuards(unittest.TestCase):
    def test_only_empty_large_cards(self):
        self.assertEqual(candidates('0, 24000, 8\n1, 22001, 1500\n2, 21000, 8\n3, 24000, 0'),[0,3])
    def test_wait_has_real_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):acquire(4,0,Path(d),'UNIT_ONLY')
    def test_invalid_gpu_bounds_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):acquire(4,0,Path(d),'UNIT_ONLY',5)


class Retention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/CONFIG).read_text());cls.receipt=json.loads((ROOT/cls.cfg['recovery_receipt']).read_text())
        cls.cs,cls.ls,cls.ps=prior(verify=True);cls.split=json.loads((ROOT/cls.cfg['split']).read_text())
    def test_no_old_completed_task_rerun(self):
        self.assertEqual(len(self.receipt['completed_densification']),100)
        for phase,n in [('new_train',80),('new_validation',40)]:self.assertEqual(len(base.tasks_for(phase,self.split)),n)
    def test_old_labels_and_unknowns_retained(self):
        self.assertEqual(len(self.cs),9628);self.assertEqual(len(self.ls),1076)
        self.assertEqual(sum(l['role']=='probe_train' and l['label']=='help' for l in self.ls),16)
        self.assertEqual(sum(l['label']=='unknown' for l in self.ls),140)
        self.assertTrue(all(l['utility_help'] is None for l in self.ls if l['label']=='unknown'))
        self.assertTrue(audit_records(self.cs,self.ls,self.split)['no_future_features'])
    def test_no_new_budget_or_deadline(self):
        old=json.loads((ROOT/'configs/evaluation/utility_expansion_v5.json').read_text())
        for k,v in old['global_limits'].items():
            self.assertEqual(self.cfg['global_limits'][k]+self.receipt['prior_actual_cost'][k]+self.receipt['preflight_cost'][k],v)
        self.assertEqual(self.cfg['original_deadline_unix'],json.loads((ROOT/'outputs/utility_expansion_v5/manifest.json').read_text())['deadline_unix'])
    def test_insufficient_data_still_cannot_train(self):
        profile=json.loads((ROOT/self.cfg['probe_training_profile']).read_text());r=readiness(self.ls,'utility_help',profile)
        self.assertFalse(r['ready']);self.assertEqual(r['counts']['probe_train']['positive'],16)
    def test_failed_reload_did_not_issue_requests(self):
        self.assertEqual(self.receipt['new_train_requests'],0);self.assertEqual(self.receipt['unknown_new_outputs'],0)
        self.assertIn('compute',self.receipt['new_train_failed_load_compute'])
    def test_adaptive_schedule_preserves_fixed_cohort(self):
        self.assertEqual(self.cfg['minimum_gpus'],1);self.assertEqual(len(self.cfg['gpus']),4)
        self.assertEqual(self.cfg['phase_order'],['new_validation','new_train'])


if __name__=='__main__':unittest.main()
