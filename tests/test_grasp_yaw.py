import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_residual as original
from scene1.residual import ResidualAssist
from scene1.retreat_projection import FingerSeparation


class GraspYawTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp

    def test_parcel_axis_sign_does_not_reverse_gripper(self):
        pose=SimpleNamespace(quat_xyzw=Rotation.from_euler('z',80,degrees=True).as_quat())
        for major in [[1,0],[-1,0]]:
            self.assertAlmostEqual(np.rad2deg(self.a._grasp_yaw_error(dict(major_axis_xy=major),pose)),10.)

    def test_degenerate_axis_rejected(self):
        pose=SimpleNamespace(quat_xyzw=Rotation.identity().as_quat())
        with self.assertRaisesRegex(RuntimeError,'valid parcel'):
            self.a._grasp_yaw_error(dict(major_axis_xy=[0,0]),pose)

    def test_measured_yaw_must_align_before_close_with_original_bounds(self):
        self.a.wrist_config['grasp_align_parcel_yaw']=True
        self.measured[13]=np.deg2rad(80);self.raw=self.measured.copy()
        target=self.snapshot['candidates'][0];target['grasp_goal_xy']=self.measured[8:10].tolist()
        self.a._begin('grasp',1,self.raw,target=target,plane=-.08,grasp_z=0.,wrist_verified=False)
        closed=False
        for _ in range(self.a.config['max_grasp_steps']):
            before=self.a.offset.copy()
            action,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertLessEqual(max(abs(self.a.offset-before)),self.a.config['max_joint_step_rad']+1e-10)
            if abs(info['grasp_yaw_error_degrees'])>2:self.assertEqual(action[15],0.)
            self.measured=action.copy()
            if action[15]>.9:
                self.assertLessEqual(abs(info['grasp_yaw_error_degrees']),2.)
                closed=True;break
        self.assertTrue(closed)

    def test_centering_preserves_depth_and_height(self):
        self.a.wrist_config['grasp_align_parcel_yaw']=True
        target=dict(position_base_m=[.4,-.2,0.],oriented_center_base_m=[.405,-.208,0.],major_axis_xy=[1.,0.])
        point=np.array([.37,-.19,.02]);centered=self.a._center_grasp_transverse(target,point)
        np.testing.assert_allclose(centered,[.37,-.208,.02])

    def test_lateral_miscentering_still_blocks_close_at_correct_yaw(self):
        self.a.wrist_config['grasp_align_parcel_yaw']=True
        self.measured[13]=np.pi/2;self.measured[9]=-.192;self.raw=self.measured.copy()
        target=self.snapshot['candidates'][0]
        self.a._begin('grasp',1,self.raw,target=target,plane=-.08,grasp_z=0.,wrist_verified=False)
        action,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertGreater(abs(info['grasp_transverse_error_m']),.002)
        self.assertEqual(action[15],0.)

    def test_recorded_drift_retains_clearance_after_frame_alignment(self):
        f=json.loads((Path(__file__).parent/'fixtures/grasp_yaw_clearance_20261024.json').read_text())
        separation=FingerSeparation(f['target'],f['plane'])
        # Counterfactual orientation screen, not contact/dynamics evidence.
        self.a.wrist_config['grasp_align_parcel_yaw']=True
        release=np.array(f['records'][2]['pose']);failed=np.array(f['records'][-1]['pose'])
        yaw=self.a._grasp_yaw_error(f['target'],SimpleNamespace(quat_xyzw=Rotation.from_matrix(release[:3,:3]).as_quat()))
        aligned=release.copy();aligned[:3,:3]=Rotation.from_rotvec([0,0,yaw]).as_matrix()@release[:3,:3]
        aligned[:3,3]=self.a._center_grasp_transverse(f['target'],release[:3,3])
        self.assertGreater(min(separation.gaps(aligned)),.008)
        # Apply exactly the old measured translation and rotation drift, without
        # recentering after release or changing the original parcel envelope.
        aligned[:3,3]+=failed[:3,3]-release[:3,3]
        aligned[:3,:3]=(failed[:3,:3]@release[:3,:3].T)@aligned[:3,:3]
        self.assertLess(min(separation.gaps(failed)),0.)
        self.assertGreater(min(separation.gaps(aligned)),0.)


if __name__=='__main__':unittest.main()
