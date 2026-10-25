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

    def home_cycle(self, gain):
        cycle = self.cycle({'return_home': {'gain_scale': gain}})
        cycle.command[:] = .1
        cycle.side, cycle.total_steps = 1, 0
        cycle.active = True
        cycle.goal = [None, None]
        cycle.rim = dict(center=np.zeros(2), rim_z=0.)
        cycle.geometry = dict(center=np.zeros(3), rotation=Rotation.identity(), extent=np.ones(3))
        cycle._stage('return_home')
        return cycle

    def test_home_gain_keeps_step_limit_and_actual_feedback_completion(self):
        measured = np.full(16, .1)
        base, fast = self.home_cycle(1.), self.home_cycle(3.)
        a, _ = base.update(measured, {'stamp': 1.}, None)
        b, _ = fast.update(measured, {'stamp': 1.}, None)
        self.assertAlmostEqual(a[0], .094)
        self.assertAlmostEqual(b[0], .088)
        self.assertFalse(fast.done)
        self.assertEqual(b[7], 0.)
        self.assertEqual(b[15], 0.)
        for i in range(5):
            fast.update(np.zeros(16), {'stamp': 2.+i}, None)
            self.assertEqual(fast.done, i == 4)

    def test_home_profile_rejects_invalid_gain(self):
        for gain in (0., -1., float('nan'), float('inf'), 3.01):
            with self.assertRaises(ValueError):
                self.home_cycle(gain).update(np.full(16, .1), {'stamp': 1.}, None)
