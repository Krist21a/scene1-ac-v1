import json
from pathlib import Path
import unittest
import numpy as np
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
from scene1.retreat_projection import FingerSeparation, project_retreat_reference


class RetreatProjectionTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).resolve().parent/'fixtures'
        self.data = json.loads((fixture/'retreat_blocked_reference_20261023.json').read_text())
        self.kin = UrdfArmKinematics(description=(fixture/'release_robot.urdf').read_text())
        f = self.data
        self.low = np.r_[self.kin.limits[0][:, 0], 0., self.kin.limits[1][:, 0], 0.]
        self.high = np.r_[self.kin.limits[0][:, 1], 1., self.kin.limits[1][:, 1], 1.]
        self.kw = dict(kinematics=self.kin, measured=np.array(f['measured']),
            proposed=np.array(f['proposed']), reference=np.array(f['previous']),
            previous=np.array(f['previous']), lower=self.low, upper=self.high,
            step=.006, target=f['target'], plane=f['plane'],
            motion_config=dict(alpha=1., max_velocity=.6, period=.1))

    def test_actual_rejected_endpoint_has_clear_progressing_step_in_same_bounds(self):
        chosen, audit = project_retreat_reference(**self.kw)
        self.assertFalse(audit['original']['filtered']['clear'])
        self.assertTrue(audit['final']['filtered']['clear'])
        self.assertTrue(audit['final']['proposed']['clear'])
        self.assertFalse(audit['final']['bounds_clipped'])
        self.assertLessEqual(max(abs(chosen[8:15]-self.kw['reference'][8:15])), .0060000001)
        self.assertTrue(np.all(chosen >= self.low) and np.all(chosen <= self.high))
        goal = np.array(self.data['goal'])
        error = lambda q: np.linalg.norm(self.kin.matrix(q[8:15], 1)[:3, 3]-goal)
        self.assertLess(error(chosen), error(self.kw['previous']))
        np.testing.assert_array_equal(chosen[:8], self.kw['proposed'][:8])

    def test_original_clear_command_is_unchanged(self):
        self.kw['proposed'] = self.kw['previous'].copy()
        chosen, audit = project_retreat_reference(**self.kw)
        self.assertFalse(audit['applied'])
        np.testing.assert_array_equal(chosen, self.kw['proposed'])

    def test_already_blocked_measured_pose_is_never_projected_into_acceptance(self):
        self.kw['measured'] = self.kw['proposed'].copy()
        with self.assertRaisesRegex(RuntimeError, 'already blocked measured pose'):
            project_retreat_reference(**self.kw)

    def test_constraint_sign_agrees_with_original_geometry_near_failure(self):
        separation = FingerSeparation(self.data['target'], self.data['plane'])
        rng = np.random.RandomState(13)
        q = self.kw['measured'][8:15]
        for sample in np.vstack((q, self.kw['proposed'][8:15],
                                 q+rng.uniform(-.2, .2, (200, 7)))):
            pose = self.kin.matrix(sample, 1)
            self.assertEqual(bool(np.all(separation.gaps(pose) > 0.)),
                open_fingers_clear(pose, self.data['target'], self.data['plane']))

    def test_no_available_safe_step_is_rejected(self):
        blocked = self.kw['proposed'].copy()
        self.kw.update(reference=blocked, step=1e-8)
        with self.assertRaisesRegex(RuntimeError, 'No swept-clear retreat correction'):
            project_retreat_reference(**self.kw)

    def test_failed_projection_does_not_accumulate_unsent_correction(self):
        from scene1.residual import ResidualAssist
        assist = ResidualAssist(dict(max_translation_m=.02, max_joint_step_rad=.012,
            max_joint_offset_rad=.5235987756, gain=.2), self.kin, self.low, self.high)
        assist.wrist_config = dict(release_sweep_projection=True,
                                   max_regrasp_joint_offset_rad=2*np.pi)
        assist.weighing_target = np.array(self.data['target']['position_base_m'])
        base = np.array(self.data['base'])
        assist.offset = self.kw['reference']-base
        before = assist.offset.copy()
        assist.observe_command(self.kw['previous'])
        with self.assertRaisesRegex(RuntimeError, 'already blocked measured pose'):
            assist._retreat_correct(base, self.kw['proposed'], [.0, .0, .02],
                self.data['target'], self.data['plane'], self.kw['motion_config'])
        np.testing.assert_array_equal(assist.offset, before)


if __name__ == '__main__':
    unittest.main()
