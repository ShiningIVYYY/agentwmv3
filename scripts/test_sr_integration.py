"""CPU contracts with actual tokenizer; fabricated values here are unit inputs only."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from internal_signals.structured_decode import FormatMachine
from internal_signals.hidden_states import response_alignment
from internal_signals.sr_alignment import generated_alignment
from internal_signals.token_nll import FORMAT
from internal_signals.decision import messages
from reflection.replay_review_v2 import ReviewerProtocol,parse_review
from scripts.prepare_sr_isolation import goal_from_dialogue
from experience.splits import file_hash


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/ee_scienceworld_sr/sr',local_files_only=True)
    def machine(self,kind='draft',draft='look around'):
        m=FormatMachine(self.tok,1,512 if kind=='draft' else 192,kind,draft);m.test_ids=[];return m
    def feed(self,m,ids):
        for i in ids:
            self.assertIn(i,m.allowed(len(m.test_ids)));m.consume(i);m.test_ids.append(i)
    def literal(self,m):
        while m.state=='literal':self.feed(m,[m.pending[0]])
    def draft(self):
        m=self.machine()
        for i,text in enumerate([' The door is closed.',' The door will open.',' open door to hallway']):
            self.literal(m);self.feed(m,self.tok.encode(text,add_special_tokens=False))
            self.feed(m,m.nl if i<2 else [m.eos])
        return m
    def test_three_real_fields(self):
        m=self.draft();self.assertEqual(m.state,'finished');self.assertTrue(FORMAT.fullmatch(self.tok.decode(m.test_ids[:-1])))
    def test_no_early_eos(self):
        m=self.machine();self.literal(m);self.assertNotIn(m.eos,m.allowed(10))
    def test_no_fake_feedback_or_second_action_line(self):
        m=self.machine();self.literal(m);self.assertTrue(all('\n' not in m.pieces[i] for i in m.allowed(10)))
    def test_next_field_requires_nonempty_value(self):
        m=self.machine();self.literal(m);self.assertNotIn(m.nl[0],m.allowed(10))
    def test_actual_generated_token_alignment(self):
        m=self.draft();ids=m.test_ids[:-1];prompt=self.tok.apply_chat_template(messages('goal','observation',[],2),tokenize=False,add_generation_prompt=True)
        pids=self.tok(prompt,add_special_tokens=False)['input_ids'];record=dict(prompt_text=prompt,prompt_ids=pids,response_ids=ids,response=self.tok.decode(ids))
        text,offsets,canonical=generated_alignment(self.tok,record)
        self.assertEqual(len(offsets),len(ids));self.assertTrue(FORMAT.fullmatch(text))
        self.assertFalse(canonical) # '.\\n' split by constrained delimiters, disclosed.
    def test_alignment_drift_fails_closed(self):
        m=self.draft();record=dict(prompt_text='hello',prompt_ids=[1],response_ids=m.test_ids[:-1],response='wrong')
        with self.assertRaises(ValueError):generated_alignment(self.tok,record)
    def test_no_completed_flag_on_truncation(self):
        m=self.machine()
        with self.assertRaises(ValueError):m.verify_finished([])
    def test_budget_closure(self):
        m=self.machine();self.literal(m);self.feed(m,self.tok.encode(' I',add_special_tokens=False));self.assertEqual(m.allowed(511),m.nl)
    def test_mask_preserves_raw_logits(self):
        m=self.machine();scores=torch.zeros((1,152064));before=scores.clone();result=m(torch.tensor([[1]]),scores)
        self.assertTrue(torch.equal(scores,before));self.assertTrue(torch.isfinite(scores).all());self.assertEqual(torch.isfinite(result).sum(),1)
    def test_keep_exact_draft(self):
        m=self.machine('review','open door to hallway');self.literal(m);self.feed(m,self.tok.encode('KEEP',add_special_tokens=False));self.literal(m)
        self.feed(m,self.tok.encode('no_grounded_alternative',add_special_tokens=False));self.literal(m);self.feed(m,[m.eos])
        text=self.tok.decode(m.test_ids[:-1]);self.assertEqual(parse_review(text,'open door to hallway')['decision'],'KEEP')
    def test_replace_json(self):
        m=self.machine('review');self.literal(m);self.feed(m,self.tok.encode('REPLACE',add_special_tokens=False));self.literal(m)
        self.feed(m,self.tok.encode('open door',add_special_tokens=False));self.feed(m,m.quote);self.literal(m)
        self.feed(m,self.tok.encode('trial_failed',add_special_tokens=False));self.literal(m);self.feed(m,[m.eos])
        self.assertEqual(parse_review(self.tok.decode(m.test_ids[:-1]),'look around')['decision'],'REPLACE')
    def test_invalid_json_not_silently_accepted(self):self.assertEqual(parse_review('{','look around')['decision'],'INVALID')
    def test_review_settings_are_original_fixed_settings(self):self.assertEqual(ReviewerProtocol('compact-json-conservative-v2').max_new_tokens,192)


class IsolationTests(unittest.TestCase):
    def test_outside_boundary(self):
        self.assertEqual(goal_from_dialogue({'messages':[{}, {},{'role':'user','content':'Your task is to wait.\nThis outside location is called the outside. history'}]}),'Your task is to wait.')
    def test_unknown_goal_boundary_rejected(self):
        with self.assertRaises(ValueError):goal_from_dialogue({'messages':[{}, {},{'role':'user','content':'Your task is to wait. arbitrary'}]})
    def test_selected_goals_not_public_training_or_each_other(self):
        m=json.loads((ROOT/'docs/sr_integration/isolation_manifest.json').read_text());a={r['goal_group'] for r in m['calibration_tasks']};b={r['goal_group'] for r in m['evaluation_tasks']}
        self.assertFalse(a&b);self.assertFalse((a|b)&set(m['published_source_goal_keys']));self.assertFalse(m['complete_checkpoint_training_isolation_claimed'])
    def test_two_training_sources_fully_audited(self):
        m=json.loads((ROOT/'docs/sr_integration/isolation_manifest.json').read_text());self.assertEqual([s['rows'] for s in m['sources']],[2038,35065])
    def test_old_method_and_protocol_protected(self):
        p=json.loads((ROOT/'configs/evaluation/sr_isolated_integration_v1.json').read_text())
        for name,sha in p['protected_hashes'].items():self.assertEqual(file_hash(ROOT/name),sha,name)
    def test_benchmark_limits_equal_9b(self):
        p=json.loads((ROOT/'configs/evaluation/sr_isolated_integration_v1.json').read_text());b=json.loads((ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json').read_text())
        for k in ['maximum_interaction_turns','history_length','prompt_limit','response_limit','temperature','top_p','top_k','rollout_seeds','wm_invocation_budgets','engine','memory_snapshot']:
            self.assertEqual(p[k],b[k],k)


if __name__=='__main__':unittest.main()
