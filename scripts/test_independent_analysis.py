#!/usr/bin/env python3
"""Statistical fixtures are unit tests, not research outcomes."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_independent_dev import interval,write_csv,rejection_reason,analyze
from scripts.diagnose_independent_dev import ranks,correlation


class Analysis(unittest.TestCase):
    plan=dict(bootstrap_seed=202659,bootstrap_resamples=2000,confidence=.95)

    def test_cluster_resampling_keeps_instance_weights(self):
        rows=[dict(goal_group='a',score=0),dict(goal_group='a',score=100),dict(goal_group='b',score=100)]
        result=interval(rows,'score',self.plan)
        self.assertEqual(result['low'],50);self.assertEqual(result['high'],100)

    def test_one_cluster_weighted_mean_not_category_mean(self):
        rows=[dict(goal_group='a',score=v) for v in [0,0,100]]
        result=interval(rows,'score',self.plan)
        self.assertAlmostEqual(result['low'],100/3);self.assertAlmostEqual(result['high'],100/3)

    def test_negative_scores_not_clipped_and_seed_reproducible(self):
        rows=[dict(goal_group='a',score=-100),dict(goal_group='b',score=100)]
        a=interval(rows,'score',self.plan);b=interval(list(reversed(rows)),'score',self.plan)
        self.assertEqual(a,b);self.assertEqual(a['low'],-100);self.assertEqual(a['high'],100)

    def test_csv_retains_quoted_multiline_actions(self):
        row=dict(action='open door,\n"kitchen"',unknown=None,score=-100)
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'raw.csv';write_csv(path,[row])
            with path.open() as f:back=list(csv.DictReader(f))
            self.assertEqual(back[0]['action'],row['action']);self.assertEqual(back[0]['score'],'-100')
            self.assertEqual(back[0]['unknown'],'')

    def test_unknown_not_labeled_invalid(self):
        self.assertEqual(rejection_reason('The door is already open.',0),'unknown')
        self.assertEqual(rejection_reason('No known action matches that input.',0),'syntax_or_reference_resolution')
        self.assertEqual(rejection_reason('The door is not open.',0),'state_precondition')
        self.assertEqual(rejection_reason('You focus on an object.',-100),'accepted_execution_but_task_failure')

    def test_spearman_average_tied_ranks(self):
        self.assertEqual(ranks([4,1,4,2]).tolist(),[2.5,0.,2.5,1.])
        self.assertAlmostEqual(correlation(ranks([1,2,2]),ranks([3,2,2])),-1.)

    def test_constant_correlation_is_unknown(self):
        self.assertIsNone(correlation([1,1],[2,3]))
        self.assertIsNone(correlation([],[]))

    def test_partial_run_not_analyzed_as_complete(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(ValueError,'All96 completed'):analyze(p,Path(p)/'results')
            self.assertFalse(Path(p,'results').exists())

if __name__=='__main__':unittest.main()
