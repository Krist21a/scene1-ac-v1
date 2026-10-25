import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scene1.visual_cycle import VisualCycle


class LinearArm:
    limits = [np.tile([-3., 3.], (7, 1)) for _ in range(2)]

    def matrix(self, q, side):
        result = np.eye(4)
        result[:3, 3] = q[:3]
        result[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return result


class SpeedProfileTests(unittest.TestCase):
    def cycle(self, profiles):
        low = np.r_[[-3.]*7, 0., [-3.]*7, 0.]
        high = np.r_[[3.]*7, 1., [3.]*7, 1.]
        cycle = VisualCycle(dict(servo_profiles=profiles), LinearArm(), low, high, np.zeros(16))
        cycle.command = np.zeros(16)
        return cycle

    def test_profile_changes_only_selected_motion_not_arrival_error_or_caps(self):
        goal = np.eye(4); goal[0, 3] = .02
        base, fast = self.cycle({}), self.cycle({'transport': {'gain_scale': 3.}})
        for cycle in (base, fast):
            cycle._stage('transport')
        expected = base._servo(np.zeros(16), 1, goal)
        actual = fast._servo(np.zeros(16), 1, goal)
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(actual[:2], expected[:2])
        self.assertGreater(fast.command[8], base.command[8])
        self.assertLessEqual(abs(fast.command[8]), .006)
        for cycle in (base, fast):
            cycle.command[:] = 0.
            cycle._stage('left_insert')
            cycle._servo(np.zeros(16), 1, goal)
        np.testing.assert_array_equal(fast.command, base.command)

    def test_loaded_receiver_is_not_accelerated_during_giver_retreat(self):
        goal = np.eye(4); goal[0, 3] = .02
        base = self.cycle({})
        fast = self.cycle({'right_retreat': {'gain_scale': 3.}})
        for cycle in (base, fast):
            cycle._stage('right_retreat')
            cycle._servo(np.zeros(16), 0, goal)
        np.testing.assert_array_equal(base.command, fast.command)

    def test_invalid_gain_rejected(self):
        for gain in (0., -1., float('nan'), float('inf'), 3.01):
            cycle = self.cycle({'transport': {'gain_scale': gain}})
            cycle._stage('transport')
            with self.assertRaises(ValueError):
                cycle._servo(np.zeros(16), 1, np.eye(4))
