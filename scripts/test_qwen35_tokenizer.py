#!/usr/bin/env python3
"""Real pinned tokenizer tests. Synthetic NLL fixtures are unit tests only."""
import argparse
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from internal_signals.hidden_states import response_alignment
from internal_signals.provider import TokenNLLProvider


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TOKENIZER
        self.config = json.loads((ROOT/'configs/internal_signals/qwen35_9b_replay_v1.json').read_text())
        self.prompt = TOKENIZER.apply_chat_template([dict(role='user',content='Decide using current observation only.')],
                           tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.ids = TOKENIZER(self.prompt,add_special_tokens=False)['input_ids']
        self.text = 'Premise: The door is closed.\nExpected Consequence: Opening reveals the route.\nAction: open door to kitchen'
        self.response_ids = TOKENIZER(self.text,add_special_tokens=False)['input_ids']

    def record(self):
        return dict(backend=self.config['backend'],dtype='bfloat16',model_revision=self.config['model_revision'],
                    prompt_text=self.prompt,prompt_ids=self.ids,response_ids=self.response_ids,response=self.text,
                    token_nll=[.4]*len(self.response_ids),generation_tokens=len(self.response_ids)+1,
                    truncated=False,reduction_seconds=0.)

    def test_non_thinking_prompt_and_official_eos(self):
        self.assertTrue(self.prompt.endswith('<think>\n\n</think>\n\n'))
        self.assertEqual(TOKENIZER.eos_token_id,248046)
        self.assertIn(TOKENIZER.eos_token_id,self.config['eos_token_ids'])
        self.assertIn(248044,self.config['eos_token_ids'])

    def test_exact_chat_response_alignment(self):
        text, offsets = response_alignment(TOKENIZER,self.prompt,self.ids,self.response_ids)
        self.assertEqual(text,self.text)
        self.assertEqual(len(offsets),len(self.response_ids))
        self.assertEqual(offsets[0][0],0)

    def test_fixed_fields_and_global(self):
        value = TokenNLLProvider(TOKENIZER,self.config).extract(self.record()).to_dict()
        self.assertEqual(value['status'],'ok')
        self.assertAlmostEqual(value['local_uncertainty'],.4)
        self.assertAlmostEqual(value['global_uncertainty'],.4)
        self.assertEqual(len(value['details']['field_spans']),2)

    def test_special_eos_is_not_content(self):
        record=self.record()
        record['response_ids']=record['response_ids']+[TOKENIZER.eos_token_id]
        record['token_nll']=record['token_nll']+[.4]
        record['response']=TOKENIZER.decode(record['response_ids'],skip_special_tokens=False)
        value=TokenNLLProvider(TOKENIZER,self.config).extract(record).to_dict()
        self.assertIsNone(value['global_uncertainty'])
        self.assertTrue(value['missing_span'])

    def test_wrong_prompt_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            response_alignment(TOKENIZER,self.prompt,self.ids[:-1],self.response_ids)

    def test_missing_fields_are_not_zero(self):
        record=self.record();record['response']='Action: look around'
        record['response_ids']=TOKENIZER(record['response'],add_special_tokens=False)['input_ids']
        record['token_nll']=[.4]*len(record['response_ids'])
        value=TokenNLLProvider(TOKENIZER,self.config).extract(record).to_dict()
        self.assertIsNone(value['local_uncertainty'])
        self.assertTrue(value['missing_span'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True)
    args,remaining=parser.parse_known_args()
    TOKENIZER=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    unittest.main(argv=[sys.argv[0]]+remaining)
