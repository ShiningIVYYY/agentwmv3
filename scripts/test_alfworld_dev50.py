"""CPU-only scheduling/split contracts. Synthetic fixtures are UNIT_ONLY."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.alfworld_dev50 import choose_tasks, schedule, maxima, spent
from scripts.analyze_alfworld_dev50 import wilson


class Contracts(unittest.TestCase):
    def fixture(self):
        rows=[dict(task_id=f'UNIT_ONLY-{f}-{i}',family=f'f{f}',goal_group=f'g{f}-{i}',game=f'f{f}/{i}',source_split='train') for f in range(6) for i in range(15)]
        old={k:[] for k in ['final_valid_seen','memory_tasks','calibration_tasks','evaluation_tasks']}
        old['evaluation_tasks']=[rows[0]]
        return rows,old

    def test_deterministic_selection_and_counts(self):
        rows,old=self.fixture()
        a,v=choose_tasks(rows,old,[])
        b,w=choose_tasks(list(reversed(rows)),old,[])
        self.assertEqual((a,v),(b,w))
        self.assertEqual((len(a),len(v)),(50,6))
        self.assertEqual(len({r['goal_group'] for r in a+v}),56)

    def test_excludes_prior_goals(self):
        rows,old=self.fixture()
        preliminary=[rows[1]]
        a,v=choose_tasks(rows,old,preliminary)
        self.assertNotIn(rows[0]['goal_group'],{r['goal_group'] for r in a+v})
        self.assertNotIn(rows[1]['goal_group'],{r['goal_group'] for r in a+v})

    def test_never_selects_valid_seen_goal(self):
        rows,old=self.fixture()
        old['final_valid_seen']=[rows[2]]
        a,v=choose_tasks(rows,old,[])
        self.assertNotIn(rows[2]['goal_group'],{r['goal_group'] for r in a+v})

    def test_250_unique_pairs_and_gpu_bound(self):
        rows,old=self.fixture()
        a,_=choose_tasks(rows,old,[])
        policies=['never','always','internal','experience','full']
        entries=schedule({'evaluation_tasks':a},policies,[0],{},4)
        self.assertEqual(len(entries),250)
        self.assertEqual(len({(e['task_id'],e['policy'],e['seed']) for e in entries}),250)
        self.assertEqual({e['worker'] for e in entries},{0,1,2,3})
        with self.assertRaises(ValueError):schedule({'evaluation_tasks':a},policies,[0],{},5)

    def test_resume_skips_complete_only(self):
        rows,old=self.fixture()
        a,_=choose_tasks(rows,old,[])
        done={(a[0]['task_id'],'never',0):True}
        entries=schedule({'evaluation_tasks':a},['never','always','internal','experience','full'],[0],done,1)
        self.assertEqual(len(entries),249)
        self.assertFalse(any((e['task_id'],e['policy'],e['seed']) in done for e in entries))

    def test_true_query_maximum_includes_replay(self):
        p=dict(maximum_interaction_turns=50,wm_invocation_budgets=dict(always=50,never=0,internal=5))
        self.assertEqual(maxima({'policy':'always'},p)['transition_queries'],1376)
        self.assertEqual(maxima({'policy':'always'},p)['audit_capture_bundles'],1476)
        self.assertEqual(maxima({'policy':'internal'},p)['transition_queries'],296)
        self.assertEqual(maxima({'policy':'never'},p)['model_calls'],50)

    def test_pending_receipt_blocks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'attempt000/worker0/ledger.json'
            p.parent.mkdir(parents=True)
            value=dict(tokens=17,model_calls=1,transition_queries=2,audit_capture_bundles=2,pending_model=None,pending_transition=None)
            p.write_text(json.dumps(value))
            self.assertEqual(spent(Path(d))['tokens'],17)
            value['pending_model']={'UNIT_ONLY':'unknown receipt'}
            p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):spent(Path(d))

    def test_descriptive_interval_bounds(self):
        self.assertAlmostEqual(wilson(0,50)[0],0.,places=12)
        self.assertLess(wilson(0,50)[1],.08)
        self.assertAlmostEqual(wilson(50,50)[1],1.,places=12)


if __name__=='__main__':unittest.main()
