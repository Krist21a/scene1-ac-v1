import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.residual import ResidualAssist
from scene1.retreat_certificate import certify_retreat_execution


class CoupledArm:
    def matrix(self, q, side):
        result = np.eye(4)
        result[:3, 3] = q[:3]
        result[:3, :3] = Rotation.from_euler('y', 4*q[2]).as_matrix()
        return result


class RetreatCertificateTests(unittest.TestCase):
    def setUp(self):
        self.q = np.zeros(16)
        self.q[8:11] = [.3, -.2, .15]
        self.assist = ResidualAssist(dict(max_translation_m=.02,
            max_joint_offset_rad=.5235987756, max_joint_step_rad=.012, gain=.2),
            CoupledArm(), np.full(16, -3.), np.full(16, 3.))
        self.assist.wrist_config = dict(max_regrasp_joint_offset_rad=.5235987756,
                                       release_retreat_ready_m=.014)
        self.assist.weighing_target = np.array([.7, -.2, .1])
        self.assist.observe_command(self.q)
        self.target = dict(position_base_m=[.7, -.2, .15], top_z_base_m=.2,
                           extent_xy_m=[.1, .04])
        self.kw = dict(assist=self.assist, base=self.q, measured=self.q,
                       goal=self.q[8:11]+[-.06, 0., .07], target=self.target, plane=.1,
                       remaining_steps=128, motion_config=dict(alpha=1., max_velocity=.6, period=.1))

    def test_executor_can_certify_clear_path_above_added_eight_degree_gate(self):
        offset = self.assist.offset.copy()
        command = self.assist.last_sent.copy()
        result = certify_retreat_execution(**self.kw)
        self.assertTrue(result['certified'], result['reason'])
        self.assertGreater(result['orientation_residual_degrees'], 8.)
        self.assertIsNone(result['orientation_tolerance_degrees'])
        self.assertLessEqual(result['position_residual_m'], .014)
        np.testing.assert_array_equal(self.assist.offset, offset)
        np.testing.assert_array_equal(self.assist.last_sent, command)
        np.testing.assert_array_equal(self.kw['base'], self.q)

    def test_collision_is_rejected_even_if_goal_is_reachable(self):
        self.kw['target'] = dict(position_base_m=self.q[8:11], top_z_base_m=.25,
                                 extent_xy_m=[.3, .3])
        result = certify_retreat_execution(**self.kw)
        self.assertFalse(result['certified'])
        self.assertFalse(result['nominal_records'][0]['filtered_sweep']['clear'])

    def test_pending_filtered_reference_is_not_ignored(self):
        self.kw['target'] = dict(position_base_m=[.3, -.2, -.02], top_z_base_m=.04,
                                 extent_xy_m=[.3, .3])
        self.kw['plane'] = -.08
        command = self.q.copy(); command[10] = -.05
        self.assist.observe_command(command)
        result = certify_retreat_execution(**self.kw)
        self.assertFalse(result['certified'])
        self.assertTrue(result['nominal_records'][0]['proposed_sweep']['clear'])
        self.assertFalse(result['nominal_records'][0]['filtered_sweep']['clear'])

    def test_original_remaining_phase_budget_is_not_extended(self):
        self.kw['remaining_steps'] = 2
        result = certify_retreat_execution(**self.kw)
        self.assertFalse(result['certified'])
        self.assertEqual(result['nominal_steps'], 2)

    def test_physical_bounds_still_reject_unreachable_goal(self):
        self.assist.high[10] = self.q[10]+.002
        result = certify_retreat_execution(**self.kw)
        self.assertFalse(result['certified'])
        self.assertLessEqual(result['feasibility_joint_rad'][2], self.assist.high[10])

    def test_missing_filter_state_does_not_invent_certificate(self):
        self.assist.last_sent = None
        with self.assertRaisesRegex(RuntimeError, 'filter state'):
            certify_retreat_execution(**self.kw)


if __name__ == '__main__':
    unittest.main()
