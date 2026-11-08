#!/usr/bin/env python3
"""Diagnostic classifier/statistic fixtures; not experiment data."""
import csv
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_reviewer_repair import mean,utility,validity,refined_feedback_status,csv_write
from scripts.audit_phase57_replay import feedback_status

class Analysis(unittest.TestCase):
    def test_explicit_connector_error_refined_without_changing_legacy(self):
        text='Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)'
        self.assertEqual(feedback_status(text),'unknown');self.assertEqual(refined_feedback_status(text),'explicit_rejection')
        self.assertEqual(validity(text)[1],'electrical_contact_point_precondition')
    def test_unsupported_use_refined(self):self.assertEqual(validity("I'm not sure how to use the switch.")[1],'unsupported_tool_use')
    def test_description_not_assumed_valid_or_task_correct(self):
        self.assertEqual(validity('A aluminum foil'),('unknown','not_classifiable_from_public_text'))
    def test_accepted_feedback_not_task_correctness(self):self.assertEqual(validity('You focus on the painting.')[1],'accepted_execution_not_semantic_correctness')
    def test_zero_not_missing(self):self.assertEqual(mean([0,0]),0);self.assertIsNone(mean([]))
    def test_signed_utility_not_clipped(self):self.assertEqual(utility([-100,0,5]),dict(positive=1,zero=1,negative=1,mean=-95/3))
    def test_csv_lf_and_quoted_text_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';csv_write(p,[dict(action='object,"x"',score=-100)])
            self.assertNotIn(b'\r',p.read_bytes())
            with p.open() as f:self.assertEqual(next(csv.DictReader(f))['action'],'object,"x"')

if __name__=='__main__':unittest.main()
