import json
from pathlib import Path
import unittest
import numpy as np
import yaml
from run_scene1 import make_residual_assist
from scene1.kinematics import UrdfArmKinematics
from scene1.retreat_projection import FingerSeparation, project_retreat_reference


class FixedConstraintRegression(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.f=json.loads((root/'tests/fixtures/retreat_fixed_constraint_20261023.json').read_text())
        self.k=UrdfArmKinematics(description=(root/'tests/fixtures/release_robot.urdf').read_text())
        self.cfg=yaml.safe_load((root/'configs/visual_cycle_four_parcels.yaml').read_text())
        self.low=np.r_[self.k.limits[0][:,0],0,self.k.limits[1][:,0],0]
        self.high=np.r_[self.k.limits[0][:,1],1,self.k.limits[1][:,1],1]

    def test_recorded_sub_picometre_roundoff_keeps_original_full_sweeps(self):
        f=self.f;r=f['blocked_segment'];obs=np.array(r['measured'])
        gap=min(FingerSeparation(f['target'],f['plane']).gaps(self.k.matrix(obs[8:15],1)))
        self.assertGreater(gap,0.)
        self.assertLess(gap,1e-6)
        self.assertLess(1e-6-gap,1e-12)
        chosen,audit=project_retreat_reference(self.k,obs,np.array(r['proposed']),
            np.array(r['reference']),np.array(r['previous']),self.low,self.high,.006,
            f['target'],f['plane'],dict(alpha=1.,max_velocity=.6,period=.1))
        self.assertFalse(audit['original']['proposed']['clear'])
        self.assertTrue(audit['final']['proposed']['clear'])
        self.assertTrue(audit['final']['filtered']['clear'])
        self.assertFalse(audit['final']['bounds_clipped'])
        self.assertLessEqual(max(abs(chosen[8:15]-np.array(r['reference'])[8:15])),.0060000001)

    def test_second_parcel_nominal_executor_finishes_original_goal_and_budget(self):
        f=self.f;base=np.array(f['release']['proposed_action']);prev=f['previous']
        a=make_residual_assist(self.cfg,self.k,self.low,self.high,10.)
        a.wrist_config=self.cfg['wrist_assist']
        a.weighing_target=np.array(f['release']['cartesian_alignment']['weighing_surface']['position_base_m'])
        a.offset=np.array(prev['proposed_action'])-base;a.observe_command(prev['sent_action'])
        a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=4
        _,info=a._pending_action(np.array(f['failed']['observation_state']),dict(
            candidates=f['failed']['rgbd_candidates'],table_z_base_m=f['plane'],stamp=f['failed']['rgbd_stamp']))
        cert=info['retreat_goal_check']
        self.assertTrue(cert['certified'])
        self.assertLessEqual(cert['nominal_steps'],136)
        self.assertLessEqual(cert['position_residual_m'],.014)
        self.assertEqual(cert['horizontal_distance_m'],.06)
        self.assertEqual(cert['rise_m'],.07)
        for row in cert['nominal_records']:
            self.assertTrue(row['proposed_sweep']['clear'])
            self.assertTrue(row['filtered_sweep']['clear'])
            self.assertFalse(row['motion']['bounds_clipped'])


if __name__=='__main__':unittest.main()
