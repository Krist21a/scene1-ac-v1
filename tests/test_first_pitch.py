import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import test_residual as original

class FirstPitchTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp
    def test_known_visual_orientation_is_level_and_axis_sign_invariant(self):
        self.a.wrist_config['first_grasp_pitch_degrees']=15.
        pose=SimpleNamespace(quat_xyzw=Rotation.from_euler('y',-80,degrees=True).as_quat())
        a=self.a._initial_grasp_rotation(dict(major_axis_xy=[1.,0.]),pose).as_matrix()
        b=self.a._initial_grasp_rotation(dict(major_axis_xy=[-1.,0.]),pose).as_matrix()
        np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertAlmostEqual(a[2,0],0.)
        self.assertAlmostEqual(np.rad2deg(np.arcsin(a[2,2])),15.)
        np.testing.assert_allclose(a.T@a,np.eye(3),atol=1e-12)
    def test_default_preserves_parent(self):
        self.assertIsNone(self.a._initial_grasp_rotation({},None))
    def test_initial_pitch_does_not_change_existing_gain_or_allow_close_before_alignment(self):
        self.a.wrist_config.update(first_grasp_gain_scale=.35,first_grasp_pitch_degrees=15.)
        target=self.snapshot['candidates'][0]
        self.a._begin('grasp',1,self.raw,target=target,plane=-.08,grasp_z=0.,wrist_verified=False)
        with patch.object(self.a,'_correct',return_value=self.raw.copy()) as correct:
            action,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(correct.call_args.kwargs['gain_scale'],.35)
            self.assertEqual(action[15],0.)
            self.assertGreater(info['initial_grasp_orientation_error_degrees'],2.)
            np.testing.assert_array_equal(correct.call_args.args[3],np.zeros(3))
    def test_later_visual_grasp_keeps_its_existing_orientation_and_gain(self):
        self.a.wrist_config.update(first_grasp_gain_scale=.35,first_grasp_pitch_degrees=15.)
        self.a._begin('grasp',1,self.raw,target=self.snapshot['candidates'][0],plane=-.08,
            grasp_z=0.,wrist_verified=False,grasp_rotation=Rotation.identity())
        with patch.object(self.a,'_correct',return_value=self.raw.copy()) as correct:
            _,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(correct.call_args.kwargs['gain_scale'],1.)
            self.assertNotIn('initial_grasp_pitch_degrees',info)
    def test_bad_pitch_or_axis_rejected(self):
        pose=SimpleNamespace(quat_xyzw=[0,0,0,1])
        for pitch in [0,18,float('nan')]:
            self.a.wrist_config['first_grasp_pitch_degrees']=pitch
            with self.assertRaises(ValueError):self.a._initial_grasp_rotation(dict(major_axis_xy=[1,0]),pose)
        self.a.wrist_config['first_grasp_pitch_degrees']=15.
        with self.assertRaises(RuntimeError):self.a._initial_grasp_rotation(dict(major_axis_xy=[0,0]),pose)
    def test_original_grasp_timeout_still_applies(self):
        self.a.wrist_config['first_grasp_pitch_degrees']=15.
        self.a._begin('grasp',1,self.raw,target=self.snapshot['candidates'][0],plane=-.08,grasp_z=0.,wrist_verified=False)
        self.a.pending['steps']=self.a.config['max_grasp_steps']-1
        with self.assertRaisesRegex(RuntimeError,'gripper kept open'):self.a._pending_action(self.measured,self.snapshot)
