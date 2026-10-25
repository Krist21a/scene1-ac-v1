import unittest
import numpy as np
import test_residual as original


class GraspMotionTests(unittest.TestCase):
    setUp = original.ResidualTests.setUp

    def begin(self):
        self.a.wrist_config['grasp_motion_settle'] = True
        target = self.snapshot['candidates'][0]
        target['grasp_goal_xy'] = self.measured[8:10].tolist()
        self.a._begin('grasp', 1, self.raw, target=target, plane=-.08,
                      grasp_z=0., wrist_verified=False)

    def test_close_waits_for_motion_to_settle(self):
        self.begin()
        action, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertEqual(action[15], 0.)
        self.measured[10] -= .0028
        action, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertFalse(info['grasp_motion_ready'])
        self.assertEqual(action[15], 0.)
        np.testing.assert_allclose(info['grasp_velocity_brake_m'], [0.,0.,.0056])
        for _ in range(2):
            action, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertTrue(info['grasp_motion_ready'])
        self.assertEqual(action[15], .98)

    def test_brake_opposes_motion_with_original_increment_bound(self):
        self.begin()
        self.a.pending['grasp_last_position'] = self.measured[8:11]-[0.,0.,.004]
        before = self.a.offset.copy()
        action, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertLess(action[10], self.raw[10])
        self.assertLessEqual(max(abs(self.a.offset-before)), self.a.config['max_joint_step_rad'])
        self.assertEqual(action[15], 0.)

    def test_motion_gate_does_not_extend_original_budget(self):
        self.begin()
        self.a.pending['steps'] = self.a.config['max_grasp_steps']-1
        with self.assertRaisesRegex(RuntimeError, 'gripper kept open'):
            self.a._pending_action(self.measured, self.snapshot)


if __name__ == '__main__': unittest.main()
