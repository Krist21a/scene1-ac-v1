import json
from pathlib import Path
import unittest
import numpy as np
import yaml
from run_scene1 import make_residual_assist
from scene1.kinematics import UrdfArmKinematics
from scene1.rgbd import metric_candidates


class ParcelControllerContextTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.config=yaml.safe_load((self.root/'configs/visual_cycle_four_parcels.yaml').read_text())
        self.kinematics=UrdfArmKinematics(description=(self.root/'tests/fixtures/release_robot.urdf').read_text())
        self.low=np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0]
        self.high=np.r_[np.full(7,3.14),1,np.full(7,3.14),1]

    def test_each_parcel_gets_fresh_filter_contract_without_reusing_previous_state(self):
        previous=[]
        for index in range(4):
            assist=make_residual_assist(self.config,self.kinematics,self.low,self.high,10.)
            self.assertEqual(assist.retreat_motion_config,dict(alpha=1.,max_velocity=.6,period=.1))
            self.assertIsNone(assist.pending)
            self.assertIsNone(assist.last_sent)
            np.testing.assert_array_equal(assist.offset,np.zeros(16))
            self.assertTrue(all(assist.retreat_motion_config is not a.retreat_motion_config for a in previous))
            assist.observe_command(np.zeros(16)+index*.01)
            previous.append(assist)

    def test_retained_second_parcel_release_executes_certificate_after_fresh_construction(self):
        fixtures=self.root/'tests/fixtures'
        data=json.loads((fixtures/'release_second_context_20261023.json').read_text())
        with np.load(fixtures/'release_second_context_20261023.npz') as raw:
            snapshot={key:raw[key].copy() for key in ['rgb','depth_m','intrinsic','camera_to_base']}
        snapshot['candidates'],snapshot['table_z_base_m']=metric_candidates(
            snapshot['rgb'],snapshot['depth_m'],snapshot['intrinsic'],snapshot['camera_to_base'],
            self.config['cartesian_alignment'],stride=2,spatial_components=True)
        snapshot['stamp']=data['failed']['rgbd_stamp']
        assist=make_residual_assist(self.config,self.kinematics,self.low,self.high,10.)
        assist.wrist_config=self.config['wrist_assist']
        assist.weighing_target=np.r_[data['pad_xy'],snapshot['table_z_base_m']]
        base=np.array(data['release']['proposed_action'])
        assist.offset=np.array(data['last']['proposed_action'])-base
        assist.observe_command(data['last']['sent_action'])
        assist._begin('short_retreat',1,base,steps_before_release=0,
                      surface={},extent=np.array([.11,.04,.1]))
        assist.pending['steps']=12
        measured=np.array(data['failed']['observation_state'])
        before=assist.offset.copy()
        command,info=assist._pending_action(measured,snapshot)
        certificate=info['retreat_goal_check']
        self.assertTrue(certificate['certified'],certificate['reason'])
        self.assertEqual(certificate['remaining_steps'],128)
        self.assertEqual(certificate['nominal_filter']['period'],.1)
        self.assertTrue(info['retreat_sweep_required'])
        self.assertEqual(assist.command_sweep_context['stage'],'short_retreat')
        self.assertLessEqual(np.max(abs(assist.offset-before)),.0060000001)
        self.assertTrue(np.all(command>=assist.low) and np.all(command<=assist.high))


if __name__=='__main__':unittest.main()
