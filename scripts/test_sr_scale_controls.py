"""Protect benchmark controls when expanding SR inference concurrency."""
from pathlib import Path
import subprocess
import sys
import unittest

import test_sr_experiments as fixture


class ScalingTests(unittest.TestCase):
    def test_expansion_changes_full_job_only(self):
        matrix = fixture.SRTests().matrix(Path('/tmp/sr-scaling-test'))
        key = 'oracle/scienceworld_ee_sr'
        job = matrix['jobs'][key]
        before = fixture.sr.command(matrix, key)
        job['preflight_config'] = dict(job['config'])
        job['config']['url'] = 'http://router'
        job['workers'] = 12
        after = fixture.sr.command(matrix, key)
        for flag in ('--max-steps', '--max-tokens', '--model', '--manifest',
                     '--reflection-max-rounds', '--reflection-review-protocol'):
            self.assertEqual(before[before.index(flag)+1], after[after.index(flag)+1])
        self.assertEqual(after[after.index('--workers')+1], '12')
        self.assertEqual(after[after.index('--url')+1], 'http://router')
        smoke = fixture.sr.command(matrix, key, True)
        self.assertEqual(smoke[smoke.index('--workers')+1], '4')
        self.assertEqual(smoke[smoke.index('--url')+1], job['preflight_config']['url'])
        alf = fixture.sr.command(matrix, 'oracle/alfworld_ee_sr')
        self.assertEqual(alf[alf.index('--workers')+1], '4')

    def test_native_cli_accepts_expanded_worker_counts(self):
        script = Path(__file__).with_name('run_audited_paper_eval.py')
        for workers in (12, 20):
            result = subprocess.run([sys.executable, str(script), '--workers',
                str(workers), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
