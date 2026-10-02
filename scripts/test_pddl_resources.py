#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmodel.resources import PddlLibraryPool


class PoolTests(unittest.TestCase):
    def test_live_and_preview_engines_have_exclusive_ownership_and_are_reused(self):
        loader = Mock(side_effect=lambda: Mock())
        pool = PddlLibraryPool(loader)
        live = pool.acquire()
        for _ in range(100):
            preview = pool.acquire()
            self.assertIsNot(preview,live)
            pool.release(preview)
        self.assertEqual(loader.call_count,2)
        self.assertEqual(preview.cleanup.call_count,100)
        live.cleanup.assert_not_called()
        pool.release(live)
        self.assertFalse(pool.active)

    def test_exhaustion_and_double_release_fail_before_native_calls(self):
        pool = PddlLibraryPool(Mock(side_effect=lambda: Mock()))
        a,b = pool.acquire(),pool.acquire()
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            pool.acquire()
        pool.release(b)
        with self.assertRaisesRegex(RuntimeError,'ownership'):
            pool.release(b)
        self.assertEqual(b.cleanup.call_count,1)
        self.assertIs(pool.acquire(),b)

    def test_failed_native_cleanup_does_not_offer_the_engine_for_reuse(self):
        library = Mock()
        library.cleanup.side_effect = RuntimeError('native cleanup failed')
        pool = PddlLibraryPool(lambda:library)
        pool.acquire()
        with self.assertRaisesRegex(RuntimeError,'native cleanup'):
            pool.release(library)
        self.assertIs(pool.active[id(library)],library)
        self.assertFalse(pool.idle)


if __name__ == '__main__':
    unittest.main()
