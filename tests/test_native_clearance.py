"""Native SAT must preserve original decisions, including historical failures."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scene1.fast_components import boxes_overlap_native
import test_fast_clearance
import test_clearance
import test_retreat_projection
import test_retreat_fixed_constraint
import test_fresh_command_sweep


class NativeFallbackTests(unittest.TestCase):
    def test_float32_retains_reference_precision(self):
        self.assertIsNone(boxes_overlap_native(np.zeros(3, dtype=np.float32),
            np.eye(3), np.ones(3), np.zeros(3), np.eye(3), np.ones(3)))

    def test_speed_seed_two_blocked_release_is_still_rejected(self):
        from scene1.kinematics import UrdfArmKinematics
        from scene1.command_sweep import sampled_sweep
        fixtures = Path(__file__).resolve().parent / 'fixtures'
        frame = json.loads((fixtures / 'speed_release_blocked_s2_20261025.json').read_text())
        kin = UrdfArmKinematics(description=(fixtures / 'release_robot.urdf').read_text())
        q = np.array(frame['measured'])[8:15]
        results = []
        for enabled in (False, True):
            with patch('scene1.clearance.FAST_CLEARANCE', enabled):
                results.append(sampled_sweep(kin, q, q, frame['target'], frame['plane']))
        self.assertEqual(results[0], results[1])
        self.assertFalse(results[1]['clear'])
        self.assertEqual(results[1]['first_blocked_fraction'], 0.)

    def test_exact_contact_requests_reference_arithmetic(self):
        self.assertIsNone(boxes_overlap_native(np.zeros(3), np.eye(3), np.full(3, .02),
                          np.array([.04, 0., 0.]), np.eye(3), np.full(3, .02)))

    def test_invalid_numbers_request_reference_validation(self):
        self.assertIsNone(boxes_overlap_native(np.full(3, np.nan), np.eye(3), np.ones(3),
                          np.zeros(3), np.eye(3), np.ones(3)))


def load_tests(loader, tests, pattern):
    for module in (test_fast_clearance, test_clearance, test_retreat_projection,
                   test_retreat_fixed_constraint, test_fresh_command_sweep):
        for original in vars(module).values():
            if not isinstance(original, type) or not issubclass(original, unittest.TestCase):
                continue
            def setup(self, parent=original):
                parent.setUp(self)
                context = patch('scene1.clearance.FAST_CLEARANCE', True)
                context.start()
                self.addCleanup(context.stop)
            native = type('Native' + original.__name__, (original,), {'setUp': setup})
            tests.addTests(loader.loadTestsFromTestCase(native))
    return tests
