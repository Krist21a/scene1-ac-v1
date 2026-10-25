import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import yaml

from run_scene1 import read_manipulation_rgbd
from scene1.rgbd import metric_candidates
from scene1.residual import ResidualAssist


class ReleaseGeometryTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root/'configs/visual_cycle_four_parcels.yaml').read_text())
        with np.load(root/'tests/fixtures/release_geometry_missing_20261023.npz') as data:
            snapshot = {key:data[key].copy() for key in ['rgb','depth_m','intrinsic','camera_to_base']}
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
            snapshot['rgb'], snapshot['depth_m'], snapshot['intrinsic'], snapshot['camera_to_base'],
            config['cartesian_alignment'], stride=2)
        snapshot.update(stamp=1., candidate_spatial_components=False)
        aligner = SimpleNamespace(pending=dict(kind='short_retreat'),
                                  weighing_target=np.array([.41698218,-.56280556,-.08331076]))
        return snapshot, config, aligner

    def near(self, snapshot, aligner):
        return [c for c in snapshot['candidates'] if np.linalg.norm(
            np.array(c['position_base_m'])[:2]-aligner.weighing_target[:2]) < .12]

    def test_real_missing_frame_recovers_existing_depth_points_without_envelope_relaxation(self):
        snapshot, config, aligner = self.fixture()
        self.assertEqual(self.near(snapshot, aligner), [])
        before = copy.deepcopy(snapshot['candidates'])
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
                                          aligner, object(), None, config)
        found = self.near(recovered, aligner)
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0]['oriented_extent_m'], [.11027356,.04382974], atol=1e-7)
        self.assertEqual(found[0]['points'], 212)
        self.assertTrue(recovered['candidate_spatial_components'])
        self.assertEqual(snapshot['candidates'], before)
        np.testing.assert_array_equal(snapshot['depth_m'], recovered['depth_m'])

    def test_existing_candidate_is_not_replaced_to_obtain_smaller_geometry(self):
        snapshot, config, aligner = self.fixture()
        snapshot['candidates'].append(dict(position_base_m=aligner.weighing_target.tolist(),
                                           extent_xy_m=[.16,.09], top_z_base_m=.03))
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
                                          aligner, object(), None, config)
        self.assertIs(recovered, snapshot)

    def test_missing_parcel_pixels_do_not_generate_a_parcel(self):
        snapshot, config, aligner = self.fixture()
        snapshot['depth_m'][200:260,390:450] = 0.
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
                                          aligner, object(), None, config)
        self.assertEqual(self.near(recovered, aligner), [])

    def test_real_accepted_xy_merge_is_separated_before_any_sweep_decision(self):
        snapshot, config, aligner = self.fixture()
        root = Path(__file__).resolve().parent/'fixtures'
        with np.load(root/'release_geometry_merged_20261023.npz') as data:
            for key in ['rgb','depth_m','intrinsic','camera_to_base']:
                snapshot[key] = data[key].copy()
        calls = []
        def read(**kw):
            calls.append(kw)
            out = snapshot.copy()
            out['candidates'], out['table_z_base_m'] = metric_candidates(
                out['rgb'],out['depth_m'],out['intrinsic'],out['camera_to_base'],
                config['cartesian_alignment'],stride=kw['candidate_stride'],
                spatial_components=kw['spatial_components'])
            out['candidate_spatial_components'] = kw['spatial_components']
            return out
        merged = self.near(read(candidate_stride=2,spatial_components=False),aligner)[0]
        recovered = read_manipulation_rgbd(SimpleNamespace(read=read),aligner,object(),None,config)
        found = self.near(recovered,aligner)
        self.assertEqual(calls[-1],dict(candidate_stride=2,spatial_components=True))
        self.assertEqual(len(found),1)
        self.assertGreater(merged['oriented_extent_m'][0],.17)
        np.testing.assert_allclose(found[0]['oriented_extent_m'],[.1089392044,.0452723379],atol=1e-8)
        self.assertEqual(found[0]['points'],201)
        np.testing.assert_array_equal(snapshot['depth_m'],recovered['depth_m'])


class HoldGeometryTests(unittest.TestCase):
    def setUp(self):
        class Arm:
            def fk(self, q):
                return SimpleNamespace(right_pose=SimpleNamespace(pos_xyz=q[8:11],quat_xyzw=[0,0,0,1]))
        self.q=np.zeros(16);self.q[8:11]=[.3,-.2,.1]
        self.assist=ResidualAssist(dict(max_joint_offset_rad=.5),Arm(),np.full(16,-3.),np.full(16,3.))
        self.assist.weighing_target=np.array([.34,-.2,0.])
        self.assist.wrist_config=dict(release_geometry_reobserve=True, max_release_geometry_wait_steps=3,
            max_short_retreat_steps=140, release_retreat_direction='measured_tool_axis',
            release_short_retreat_m=.06, release_clearance_rise_m=.07)
        self.assist.pending=dict(kind='short_retreat',side=1,base=self.q.copy(),steps=12,stable=0,
                                 surface={},extent=[.11,.04,.1])
        self.missing=dict(candidates=[],table_z_base_m=0.,stamp=1.)

    def test_temporary_missing_geometry_keeps_same_open_command_and_does_not_generate_goal(self):
        with patch('scene1.residual.axis_retreat_goal') as goal:
            action, info=self.assist._pending_action(self.q,self.missing)
        goal.assert_not_called()
        np.testing.assert_array_equal(action,self.q)
        np.testing.assert_array_equal(self.assist.offset,np.zeros(16))
        self.assertEqual(info['state'],'waiting_release_geometry')
        self.assertEqual(self.assist.pending['steps'],13)
        self.assertNotIn('goal',self.assist.pending)

    def test_reappearing_geometry_resumes_with_current_observation_without_resetting_budget(self):
        self.assist._pending_action(self.q,self.missing)
        current=dict(self.missing,stamp=2.,candidates=[dict(position_base_m=[.34,-.2,.05])])
        self.assist._correct=Mock(return_value=self.q.copy())
        with patch('scene1.residual.axis_retreat_goal', return_value=(self.q[8:11]+[-.06,0.,.07],{})) as goal:
            self.assist._pending_action(self.q,current)
        self.assertEqual(goal.call_args.args[8]['stamp'],2.)
        self.assertEqual(self.assist.pending['steps'],14)

    def test_missing_and_ambiguous_observations_have_finite_wait_without_moving(self):
        for candidates in [[],[dict(position_base_m=[.34,-.2,.05])]*2]:
            with self.subTest(count=len(candidates)):
                self.setUp();snapshot=dict(self.missing,candidates=candidates)
                for _ in range(2):
                    action,_=self.assist._pending_action(self.q,snapshot)
                    np.testing.assert_array_equal(action,self.q)
                with self.assertRaisesRegex(RuntimeError,'no retreat sent'):
                    self.assist._pending_action(self.q,snapshot)

    def test_overall_phase_budget_remains_binding(self):
        self.assist.wrist_config['max_short_retreat_steps']=13
        with self.assertRaisesRegex(RuntimeError,'no retreat sent'):
            self.assist._pending_action(self.q,self.missing)


if __name__ == '__main__':
    unittest.main()
