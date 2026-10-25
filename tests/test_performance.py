import json
from pathlib import Path
import tempfile
import unittest

from scene1.performance import CycleTimer


class CycleTimerTests(unittest.TestCase):
    def test_cycle_includes_recording_and_wait(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'timing.jsonl'
            times = iter([0., .02, .03, .08, .1])
            timer = CycleTimer(path, True, lambda: next(times))
            timer.begin(1, 10.)
            timer.mark('compute', phase='carry')
            timer.mark('send')
            timer.mark('diagnostics')
            timer.finish(sim_end_s=10.05)
            timer.close()
            row = json.loads(path.read_text())
            self.assertAlmostEqual(row['cycle_s'], .1)
            self.assertAlmostEqual(sum(row['spans_s'].values()), .1)
            self.assertAlmostEqual(row['spans_s']['diagnostics'], .05)
            self.assertEqual(row['phase'], 'carry')

    def test_exception_tail_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'timing.jsonl'
            times = iter([0., .2])
            timer = CycleTimer(path, True, lambda: next(times))
            timer.begin(9, 1.)
            timer.close()
            self.assertTrue(json.loads(path.read_text())['incomplete'])

    def test_disabled_does_not_read_clock_or_create_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'timing.jsonl'
            timer = CycleTimer(path, False, lambda: self.fail('clock read'))
            timer.begin(1, 1.)
            timer.mark('x')
            timer.close()
            self.assertFalse(path.exists())
