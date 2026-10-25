import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_residual as original


class ReleaseOpenHoldTests(unittest.TestCase):
    setUp = original.ResidualTests.setUp

    def fixture(self):
        self.a.wrist_config = dict(release_cartesian_open_hold=True,
            release_open_hold_position_gain=2., release_open_dwell_steps=12)
        self.a._begin('short_retreat', 1, self.raw.copy(),
            release_hold_position=self.measured[8:11].copy(),
            release_hold_rotation=Rotation.identity())

    def test_stationary_feedback_keeps_pose_and_original_twelve_step_dwell(self):
        self.fixture()
        for index in range(12):
            result, info = self.a._pending_action(self.measured, None)
            np.testing.assert_array_equal(result, self.raw)
            self.assertEqual(info['phase_steps'], index + 1)
            self.assertEqual(info['release_pose_hold']['position_error_m'], 0.)
            self.assertIsNone(self.a.pending.get('goal'))

    def test_downward_drift_requests_upward_correction_without_moving_other_arm(self):
        self.fixture()
        self.measured[10] -= .02
        before = self.a.offset.copy()
        result, info = self.a._pending_action(self.measured, None)
        self.assertGreater(result[10], self.raw[10])
        self.assertLessEqual(np.max(abs(self.a.offset-before)),
                             self.a.config['max_joint_step_rad'] + 1e-12)
        np.testing.assert_array_equal(result[:8], self.raw[:8])
        self.assertEqual(result[15], 0.)
        self.assertAlmostEqual(info['release_pose_hold']['position_error_m'], .02)

    def test_recovering_velocity_reduces_correction(self):
        self.fixture()
        self.measured[10] -= .01
        slow, _ = self.a._pending_action(self.measured, None)
        self.a.offset[:] = 0.
        self.a.pending['hold_last_position'] = self.measured[8:11] - [0., 0., .005]
        moving, _ = self.a._pending_action(self.measured, None)
        self.assertGreater(moving[10], self.raw[10])
        self.assertLess(moving[10], slow[10])

    def test_joint_limit_is_enforced_during_hold(self):
        self.fixture()
        self.a.high[10] = self.raw[10]
        self.measured[10] -= .02
        for _ in range(12):
            result, _ = self.a._pending_action(self.measured, None)
            self.assertLessEqual(result[10], self.a.high[10])
            self.assertTrue(np.all(result >= self.a.low))
            self.assertTrue(np.all(result <= self.a.high))

    def test_gripper_open_timeout_is_unchanged(self):
        self.fixture()
        self.a.pending['steps'] = 39
        self.measured[15] = .2
        with self.assertRaisesRegex(RuntimeError, 'gripper did not open'):
            self.a._pending_action(self.measured, None)

    def test_invalid_hold_gain_rejected_before_command(self):
        for value in [0., -1., float('nan'), float('inf')]:
            self.fixture()
            self.a.wrist_config['release_open_hold_position_gain'] = value
            with self.assertRaisesRegex(ValueError, 'positive and finite'):
                self.a._pending_action(self.measured, None)

    def test_shorter_minimum_dwell_does_not_override_measured_gripper_gate(self):
        self.fixture()
        self.a.wrist_config['release_open_dwell_steps'] = 4
        self.a.pending['steps'] = 4
        self.measured[15] = .11
        result, info = self.a._pending_action(self.measured, None)
        self.assertEqual(result[15], 0.)
        self.assertIn('release_pose_hold', info)
        self.assertIsNone(self.a.pending.get('goal'))

    def test_ready_gripper_still_requires_current_parcel_observation(self):
        self.fixture()
        self.a.wrist_config.update(release_open_dwell_steps=4,
            release_geometry_reobserve=True, max_release_geometry_wait_steps=20,
            max_short_retreat_steps=140)
        self.a.weighing_target = np.array([.31, -.2, -.08])
        self.a.pending['steps'] = 4
        self.measured[15] = .03
        result, info = self.a._pending_action(self.measured, None)
        self.assertEqual(info['state'], 'waiting_release_geometry')
        self.assertIsNone(self.a.pending.get('goal'))
        for _ in range(18):
            self.a._pending_action(self.measured, None)
        with self.assertRaisesRegex(RuntimeError, 'observation remained missing'):
            self.a._pending_action(self.measured, None)


if __name__ == '__main__':
    unittest.main()
