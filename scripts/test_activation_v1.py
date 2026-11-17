"""UNIT_ONLY synthetic tensors; never reported as scientific task results."""
import sys,unittest,torch,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from internal_signals.activation_protocol import parse_draft,parse_review_guarded
from internal_signals.activation_probe import ActivationProbe,pad_batch,binary_metrics
from internal_signals.activation_capture import ActivationCapture
from internal_signals.activation_training import Ensemble,train_head


class Contracts(unittest.TestCase):
    def test_valid_three_fields(self):
        x=parse_draft('Premise: The door is closed.\nExpected Consequence: It opens.\nAction: open fridge 1')
        self.assertTrue(x['valid']);self.assertEqual(x['action'],'open fridge 1')
    def test_no_action_invented(self):
        self.assertIsNone(parse_draft('Premise: I might go to fridge 1.')['action'])
    def test_truncated_rejected(self):
        self.assertFalse(parse_draft('Premise: A.\nExpected Consequence: B.\nAction: look',True)['valid'])
    def test_nonadmissible_replacement_guard(self):
        r=parse_review_guarded('Decision: REPLACE\nReason: Try it.\nFinal Action: invent object','look',['look'])
        self.assertEqual(r['decision'],'REJECTED');self.assertEqual(r['final_action'],'look')
    def test_padding_invariance(self):
        torch.manual_seed(0)
        for kind in ['linear','mlp','ema']:
            model=ActivationProbe(8,kind,width=8).eval();v=torch.randn(3,8)
            a,m=pad_batch([v]);b,n=pad_batch([v,torch.randn(8,8)])
            torch.testing.assert_close(model(a,m)[0],model(b,n)[0],rtol=1e-5,atol=1e-6)
    def test_stream_batch_equivalence(self):
        torch.manual_seed(2);model=ActivationProbe(8,'ema',width=8).eval();v=torch.randn(20,8);state=None
        for h in v:out,state=model.stream_step(h,state)
        a,m=pad_batch([v]);torch.testing.assert_close(out,model(a,m)[0],rtol=1e-5,atol=1e-6)
    def test_training_updates_only_probe(self):
        model=ActivationProbe(8,'mlp',width=8);initial=model.head.weight.detach().clone();a,m=pad_batch([torch.randn(3,8),torch.randn(5,8)])
        opt=torch.optim.AdamW(model.parameters(),lr=.01);loss=model(a,m).square().mean();loss.backward();opt.step()
        self.assertFalse(torch.equal(initial,model.head.weight))
    def test_metric_ties(self):
        r=binary_metrics([0,1],[.5,.5]);self.assertAlmostEqual(r['auroc'],.5);self.assertAlmostEqual(r['average_precision'],.5)
    def test_metric_perfect(self):
        r=binary_metrics([0,1],[.1,.9]);self.assertEqual(r['auroc'],1.);self.assertEqual(r['average_precision'],1.)
    def test_single_class_not_fake_auc(self):self.assertIsNone(binary_metrics([0,0],[.1,.2])['auroc'])
    def test_future_fields_rejected(self):
        capture=ActivationCapture.__new__(ActivationCapture)
        with self.assertRaises(ValueError):capture.extract(dict(prompt_ids=[1],response_ids=[2],prompt_text='a',response='b',future_success=True))
    def test_stream_training_rejected(self):
        model=ActivationProbe(8,'ema',width=8).train()
        with self.assertRaises(ValueError):model.stream_step(torch.randn(8))
    def test_strict_checkpoint_and_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe.pt';model=ActivationProbe(8,'linear')
            data=dict(state_dict=model.state_dict(),contract={'test':'UNIT_ONLY'},layer=1,target='test',dim=8,kind='linear')
            torch.save(data,p);ensemble=Ensemble(d,['probe.pt'])
            with self.assertRaises(ValueError):ensemble.score({'1':torch.randn(3,8)},{'test':'changed'})
            data['state_dict']['unexpected']=torch.zeros(1);torch.save(data,p)
            with self.assertRaises(RuntimeError):Ensemble(d,['probe.pt'])
    def test_sparse_labels_never_create_fake_head(self):
        with tempfile.TemporaryDirectory() as d:
            records=[dict(task_id=str(i),role='probe_train',risk=0) for i in range(6)]
            result=train_head(d,d,records,'risk',[1])
            self.assertEqual(result['status'],'insufficient_labels');self.assertEqual(list(Path(d).glob('*.pt')),[])


if __name__=='__main__':unittest.main()
