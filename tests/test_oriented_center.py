import copy
import json
from pathlib import Path
import unittest
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scene1.rgbd import metric_candidates, oriented_footprint
from scene1.clearance import open_fingers_clear
from scene1.retreat_projection import FingerSeparation
from scene1.kinematics import UrdfArmKinematics
from scene1.table_reference import move_target


class OrientedCenterTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/'tests/fixtures/release_obb_center_20261023.json').read_text())
        config=yaml.safe_load((root/'configs/visual_cycle_four_parcels.yaml').read_text())
        with np.load(root/'tests/fixtures/release_obb_center_20261023.npz') as raw:
            candidates,plane=metric_candidates(raw['rgb'],raw['depth_m'],raw['intrinsic'],
                raw['camera_to_base'],config['cartesian_alignment'],stride=2,spatial_components=True)
        old=data['failed']['rgbd_candidates'][0]
        target=min(candidates,key=lambda c:np.linalg.norm(np.array(c['position_base_m'])-old['position_base_m']))
        kin=UrdfArmKinematics(description=(root/'tests/fixtures/release_robot.urdf').read_text())
        pose=kin.matrix(np.array(data['failed']['observation_state'])[8:15],1)
        return old,target,plane,pose

    def test_retained_false_block_is_center_mismatch_without_smaller_envelope(self):
        old,target,plane,pose=self.fixture()
        self.assertEqual(old['points'],target['points'])
        for key in ['position_base_m','extent_xy_m','oriented_extent_m','major_axis_xy','top_z_base_m']:
            np.testing.assert_allclose(target[key],old[key],atol=1e-12)
        self.assertFalse(open_fingers_clear(pose,old,plane))
        self.assertTrue(open_fingers_clear(pose,target,plane))
        shift=np.array(target['oriented_center_base_m'])-target['position_base_m']
        self.assertGreater(np.linalg.norm(shift[:2]),.013)
        self.assertGreater(min(FingerSeparation(target,plane).gaps(pose)),.009)

    def test_footprint_center_and_dimensions_are_invariant_under_rigid_frame_change(self):
        rng=np.random.RandomState(17)
        points=np.r_[np.column_stack((rng.uniform(-.055,.055,400),np.full(400,-.02))),
                     rng.uniform([-.055,-.02],[.055,.02],(300,2))]
        _,extent,center=oriented_footprint(points)
        for angle in [19.,47.,125.]:
            rotation=Rotation.from_euler('z',angle,degrees=True).as_matrix()[:2,:2]
            translation=np.array([.42,-.55])
            _,new_extent,new_center=oriented_footprint(points@rotation.T+translation)
            np.testing.assert_allclose(new_extent,extent,atol=1e-12)
            np.testing.assert_allclose(new_center,rotation@center+translation,atol=1e-12)

    def test_table_motion_transports_both_centers_and_preserves_collision_result(self):
        old,target,plane,pose=self.fixture()
        transform=np.eye(4)
        transform[:3,:3]=Rotation.from_euler('z',1.2,degrees=True).as_matrix()
        transform[:3,3]=[.01,-.003,.002]
        moved=move_target(target,transform)
        for key in ['position_base_m','oriented_center_base_m']:
            np.testing.assert_allclose(moved[key],(transform@np.r_[target[key],1.])[:3])
        self.assertEqual(open_fingers_clear(transform@pose,moved,plane+.002),
                         open_fingers_clear(pose,target,plane))
        np.testing.assert_array_equal(moved['oriented_extent_m'],target['oriented_extent_m'])

    def test_actual_overlap_is_still_rejected_with_oriented_center(self):
        target=dict(position_base_m=[10.,10.,.04],oriented_center_base_m=[0.,0.,.04],
                    top_z_base_m=.08,major_axis_xy=[1.,0.],extent_xy_m=[.12,.04])
        pose=np.eye(4);pose[:3,:3]=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
        pose[:3,3]=[-.04,.025,.04]
        self.assertFalse(open_fingers_clear(pose,target,0.))
        self.assertLessEqual(min(FingerSeparation(target,0.).gaps(pose)),0.)


if __name__=='__main__':unittest.main()
