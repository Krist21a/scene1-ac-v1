from types import SimpleNamespace
import unittest
import numpy as np
from scene1.sdk_limits import CachedArmLimits


class SDKLimitsTests(unittest.TestCase):
    def fixture(self):
        self.calls = 0
        self.params = {'/run_id': 'epoch-a', '/humanoid_description': '<robot/>'}
        def original():
            self.calls += 1
            return [-2.]*14, [2.]*14
        self.original = original
        self.info = SimpleNamespace(get_arm_joint_limits=original)
        self.limits = np.tile([-2., 2.], (14, 1))

    def test_cached_bounds_are_exact_independent_and_restorable(self):
        self.fixture()
        cache = CachedArmLimits(self.info, self.params.__getitem__, self.limits)
        lo, hi = self.info.get_arm_joint_limits()
        lo[0] = -100.
        self.assertEqual(self.info.get_arm_joint_limits(), ([-2.]*14, [2.]*14))
        self.assertEqual(self.calls, 1)
        cache.close()
        self.assertIs(self.info.get_arm_joint_limits, self.original)

    def test_changed_or_deleted_parameter_stops_before_returning_limits(self):
        for key in ('/run_id', '/humanoid_description'):
            self.fixture()
            cache = CachedArmLimits(self.info, self.params.__getitem__, self.limits)
            self.params[key] += '-changed'
            with self.assertRaises(RuntimeError):
                self.info.get_arm_joint_limits()
            del self.params[key]
            with self.assertRaises(KeyError):
                self.info.get_arm_joint_limits()
            cache.close()

    def test_sdk_fallback_cannot_be_cached(self):
        self.fixture()
        self.limits[0, 0] = -1.9
        with self.assertRaises(ValueError):
            CachedArmLimits(self.info, self.params.__getitem__, self.limits)
        self.assertIs(self.info.get_arm_joint_limits, self.original)
