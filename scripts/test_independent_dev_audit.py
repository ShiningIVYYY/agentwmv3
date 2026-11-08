#!/usr/bin/env python3
"""Seed-schema regression; unit metadata fixtures are not experiment results."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_independent_dev import recorded_seed

class SeedSchema(unittest.TestCase):
    def test_real_unscored_schema(self):
        self.assertEqual(recorded_seed({'generation_record':{'request_seed':7}}),7)
    def test_real_scored_schema(self):
        self.assertEqual(recorded_seed({'generation_record':{'token_nll':[.2]},'actor_metadata':{'generation_seed':9}}),9)
    def test_same_duplicate_fields_allowed(self):
        self.assertEqual(recorded_seed({'generation_record':{'request_seed':0},'actor_metadata':{'generation_seed':0}}),0)
    def test_conflicting_fields_rejected(self):
        with self.assertRaises(ValueError):recorded_seed({'generation_record':{'request_seed':1},'actor_metadata':{'generation_seed':2}})
    def test_missing_not_filled_from_expectation(self):
        with self.assertRaises(ValueError):recorded_seed({'generation_record':{}})
    def test_wrong_types_rejected(self):
        for value in [None,True,'7',7.0]:
            with self.assertRaises(ValueError):recorded_seed({'generation_record':{'request_seed':value}})

if __name__=='__main__':unittest.main()
