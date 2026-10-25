import copy
import unittest
import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from scene1.residual import ResidualAssist
from scene1.release_retreat import axis_retreat_goal
from scene1.command_sweep import command_sweep_audit


class LinearArm:
    def __init__(self, axis=(-1., 0., 0.)):
        self.axis = np.asarray(axis)

    def matrix(self, q, side):
        m = np.eye(4); m[:3, 3] = q[:3]
        y = np.array([0., 1., 0.])
        m[:3, :3] = np.column_stack((np.cross(y, self.axis), y, self.axis))
        return m


class AxisRetreatTests(unittest.TestCase):
    def setUp(self):
        self.q = np.zeros(16); self.q[8:11] = [.3, -.2, 0.]
        self.parcel = dict(position_base_m=[.34, -.2, 0.], extent_xy_m=[.1, .04],
                           top_z_base_m=.05)
        self.kw = dict(kinematics=LinearArm(), measured=self.q, reference=self.q.copy(),
                       low=np.full(16, -4.), high=np.full(16, 4.), offset=np.zeros(16),
                       cap=.5235987756, unwinding=False,
                       snapshot=dict(candidates=[self.parcel], stamp=5.),
                       pad_xy=np.array([.34, -.2]), extent=[.1, .04, .1],
                       distance=.06, rise=.07, tolerance=.014)

    def test_goal_uses_measured_axis_and_keeps_original_reference(self):
        before = self.kw['reference'].copy()
        goal, audit = axis_retreat_goal(**self.kw)
        np.testing.assert_allclose(goal, [.24, -.2, .07])
        np.testing.assert_array_equal(before, self.kw['reference'])
        self.assertLess(audit['position_residual_m'], 1e-6)
        self.assertGreater(audit['away_dot_m'], 0.)

    def test_degenerate_projection_is_rejected(self):
        self.kw['kinematics'] = LinearArm((0., 0., 1.))
        with self.assertRaisesRegex(RuntimeError, 'degenerate'):
            axis_retreat_goal(**self.kw)

    def test_axis_toward_parcel_is_rejected_without_flipping_it(self):
        self.kw['kinematics'] = LinearArm((1., 0., 0.))
        with self.assertRaisesRegex(RuntimeError, 'does not point away'):
            axis_retreat_goal(**self.kw)

    def test_missing_and_ambiguous_observation_rejected(self):
        for candidates in [[], [self.parcel, copy.deepcopy(self.parcel)]]:
            with self.subTest(candidates=len(candidates)):
                self.kw['snapshot']['candidates'] = candidates
                with self.assertRaisesRegex(RuntimeError, 'one currently observed'):
                    axis_retreat_goal(**self.kw)

    def test_original_correction_budget_cannot_be_rebased(self):
        self.kw['cap'] = .005
        with self.assertRaisesRegex(RuntimeError, 'not certified'):
            axis_retreat_goal(**self.kw)

    def test_physical_limits_cannot_be_relaxed(self):
        self.kw['low'][8] = .295
        with self.assertRaisesRegex(RuntimeError, 'not certified'):
            axis_retreat_goal(**self.kw)

    def test_weighted_optimum_can_miss_a_goal_inside_both_original_tolerances(self):
        class CoupledArm(LinearArm):
            def matrix(self, q, side):
                m = super().matrix(q, side)
                m[:3, :3] = m[:3, :3] @ Rotation.from_rotvec([0., 2.8*(.3-q[0]), 0.]).as_matrix()
                return m
        self.kw['kinematics'] = CoupledArm()
        goal, audit = axis_retreat_goal(**self.kw)
        np.testing.assert_allclose(goal, [.24, -.2, .07])
        self.assertGreater(audit['initial_weighted_result']['orientation_residual_degrees'], 8.)
        self.assertEqual(audit['solver_method'], 'explicit_same_orientation_constraint')
        self.assertTrue(audit['certified'])
        self.assertLessEqual(audit['position_residual_m'], .014)
        self.assertLessEqual(audit['orientation_residual_degrees'], 8.)
        self.assertEqual(audit['correction_cap_rad'], self.kw['cap'])

    def test_failed_certificate_keeps_original_bounds_and_diagnostics(self):
        self.kw['cap'] = .005
        with self.assertRaises(RuntimeError) as caught:
            axis_retreat_goal(**self.kw)
        audit = caught.exception.retreat_preflight_audit
        self.assertFalse(audit['certified'])
        self.assertEqual(audit['position_tolerance_m'], .014)
        self.assertEqual(audit['orientation_tolerance_degrees'], 8.)
        np.testing.assert_allclose(audit['lower_joint_rad'], self.q[8:15]-.005)
        np.testing.assert_allclose(audit['upper_joint_rad'], self.q[8:15]+.005)

    def test_pending_reference_collision_is_visible_without_new_increment(self):
        # The tool is safely above the box but its pending target is inside it.
        arm = LinearArm((0., 0., 1.))
        measured = np.zeros(16); measured[8:11] = [.34, -.2, .2]
        sent = measured.copy(); sent[10] = -.02
        parcel = dict(self.parcel, extent_xy_m=[.2, .2])
        report = command_sweep_audit(arm, measured, sent, sent, measured,
                    dict(target=parcel, plane=-.05, stage='level',
                         increment_end_rad=measured[8:15].tolist()))
        self.assertTrue(report['legacy_increment']['clear'])
        self.assertFalse(report['filtered']['clear'])
        self.assertTrue(report['previous']['clear'])
        self.assertEqual(report['margin_m'], .003)

    def test_complete_reference_blocks_even_zero_increment_and_keeps_stop_budget(self):
        arm = LinearArm((0., 0., 1.))
        q = np.zeros(16); q[8:11] = [.34, -.2, .2]
        base = q.copy(); base[10] = -.02
        assist = ResidualAssist(dict(max_translation_m=.02, max_joint_offset_rad=.5235987756,
            max_joint_step_rad=.012, gain=.2), arm, np.full(16, -4.), np.full(16, 4.))
        assist.wrist_config = dict(command_sweep_enforce=True)
        target = dict(self.parcel, extent_xy_m=[.2, .2])
        assist.pending = dict(kind='pregrasp', base=base, steps=1, stable=0,
            target=target, plane=-.05, grasp_rotation=Rotation.identity(),
            regrasp_stage='level', view_position=q[8:11].copy())
        poses = SimpleNamespace(right_pose=SimpleNamespace(pos_xyz=q[8:11], quat_xyzw=[0,0,0,1]))
        for i in range(19):
            _, info = assist._regrasp_action(q, None, poses, {})
            self.assertFalse(info['open_fingers_clear'])
            self.assertEqual(assist.pending['blocked_steps'], i+1)
            self.assertEqual(assist.pending['stable'], 0)
            np.testing.assert_array_equal(assist.offset, np.zeros(16))
        with self.assertRaisesRegex(RuntimeError, 'Open finger sweep'):
            assist._regrasp_action(q, None, poses, {})


if __name__ == '__main__':
    unittest.main()
