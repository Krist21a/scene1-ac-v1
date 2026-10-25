import threading
import unittest
from scene1.parallel_read import CycleRead


class ParallelReadTests(unittest.TestCase):
    def test_read_overlaps_independent_work_but_is_joined_in_same_cycle(self):
        entered, release = threading.Event(), threading.Event()
        reader = CycleRead()
        def read(value):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('Test worker was not released')
            return value
        try:
            reader.begin(read, 42)
            self.assertTrue(entered.wait(2))
            with self.assertRaises(RuntimeError):
                reader.begin(lambda: 'must not queue another frame')
            release.set()
            self.assertEqual(reader.result(), 42)
            with self.assertRaises(RuntimeError):
                reader.result()
            reader.begin(lambda: 43)
            self.assertEqual(reader.result(), 43)
        finally:
            release.set()
            reader.close()

    def test_sensor_errors_are_preserved_and_never_reuse_an_old_frame(self):
        reader = CycleRead()
        def fail():
            raise ValueError('RGB-D RGB is stale')
        try:
            reader.begin(fail)
            with self.assertRaisesRegex(ValueError, 'RGB-D RGB is stale'):
                reader.result()
            reader.begin(lambda: {'stamp': 2.})
            self.assertEqual(reader.result(), {'stamp': 2.})
        finally:
            reader.close()
