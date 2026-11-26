"""Actual retained-corpus checks and synthetic scheduling-only regression."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import resume_v5_remaining as mod
from experience.splits import file_hash
from evaluation.independent_dev import atomic
from scripts.run_utility_expansion import audit_records


class SourceFirst(unittest.TestCase):
    def test_every_source_precedes_any_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';out=run/'new_train/worker0';out.mkdir(parents=True)
            cfg={'gpus':[0,1,3,4]};atomic(root/'config.json',cfg)
            phase=run/'new_train_runtime_config.json';atomic(phase,{'gpus':[6]})
            atomic(run/'new_train_gpu_assignment.json',dict(gpus=[6],config_sha256=file_hash(phase)))
            atomic(run/'manifest.json',{'source_hashes':{}});events=[]
            rt=SimpleNamespace(ledger=SimpleNamespace(state={'tokens':0,'model_calls':0}))
            def fake_sources(*_):
                for i in range(2):
                    row={'task_id':'UNIT_ONLY_'+str(i)};events.append('source'+str(i))
                    mod.base.label_state(rt,row,{'case_id':str(i)},{},[],{})
                    mod.base.append(out/'completed_tasks.jsonl',dict(task_id=row['task_id'],selected=1))
                mod.base.atomic(out/'complete.json',dict(ledger=rt.ledger.state))
            def fake_label(*args):events.append('label'+args[2]['case_id'])
            with patch.object(mod,'ROOT',root),patch.object(mod,'CONFIG','config.json'),patch.object(mod.base,'worker',fake_sources),patch.object(mod,'label_state',fake_label):
                mod.worker(run,'new_train',0)
            self.assertEqual(events,['source0','source1','label0','label1'])
            self.assertTrue((out/'source_pass_complete.json').exists())
            self.assertEqual(len((out/'completed_tasks.jsonl').read_text().splitlines()),2)


class RealRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/mod.CONFIG).read_text());cls.r=json.loads((ROOT/cls.cfg['recovery_receipt']).read_text())
        cls.cs,cls.ls,cls.ps=mod.prior(verify=True);cls.split=json.loads((ROOT/cls.cfg['split']).read_text())
    def test_only_fixed_remaining_tasks(self):
        self.assertEqual(len(mod.remaining_tasks('new_validation',self.split)),2)
        self.assertEqual(len(mod.remaining_tasks('new_train',self.split)),80)
        self.assertEqual(len(self.r['completed_validation']),38)
    def test_known_positive_labels_retained(self):
        self.assertEqual(sum(l['label']=='help' and l['role']=='probe_train' for l in self.ls),16)
        self.assertEqual(sum(l['label']=='help' and l['role']=='probe_validation' for l in self.ls),8)
        self.assertTrue(audit_records(self.cs,self.ls,self.split)['no_future_features'])
    def test_partial_always_excluded_complete_never_retained(self):
        ids=set(self.r['remaining_validation_ids']);cs=[c for c in self.cs if c['task_id'] in ids]
        self.assertTrue(cs);self.assertEqual({c['source_policy'] for c in cs},{'never'})
    def test_original_budget_and_deadline(self):
        original=json.loads((ROOT/'configs/evaluation/utility_expansion_v5.json').read_text())
        pre=json.loads((ROOT/'outputs/utility_expansion_v5_preflight/complete.json').read_text())
        for k,v in original['global_limits'].items():self.assertEqual(self.cfg['global_limits'][k]+self.r['prior_actual_cost'][k]+pre.get(k,0),v)
        self.assertEqual(self.cfg['original_deadline_unix'],self.r['old_deadline_unix'])
    def test_whole_remaining_source_pass_funded(self):
        left=self.cfg['phase_limits']['new_validation']
        self.assertGreaterEqual(left['model_calls'],200);self.assertGreaterEqual(left['tokens'],512000)
        for n in [1,2,3,4]:
            tasks=(80+n-1)//n
            self.assertGreaterEqual(self.cfg['phase_limits']['new_train']['tokens']//n,tasks*150*2560)
            self.assertGreaterEqual(self.cfg['phase_limits']['new_train']['model_calls']//n,tasks*150)


if __name__=='__main__':unittest.main()
