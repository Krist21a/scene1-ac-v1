import copy
import json
from pathlib import Path
import unittest
import numpy as np
import yaml
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.command_sweep import fresh_command_sweep_audit, retreat_pending_hold_allowed


class PendingHoldTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).parent
        self.f=json.loads((p/'fixtures/fresh_pending_hold_20261024.json').read_text())
        self.k=UrdfArmKinematics(description=(p/'fixtures/release_robot.urdf').read_text())
        lo=np.r_[self.k.limits[0][:,0],0.,self.k.limits[1][:,0],0.]
        hi=np.r_[self.k.limits[0][:,1],1.,self.k.limits[1][:,1],1.]
        config=yaml.safe_load((p.parent/'config.yaml').read_text())
        self.a=ResidualAssist(config['visual_residual'],self.k,lo,hi)
        self.previous=np.array(self.f['previous_sent_action'])
        self.proposed=np.array(self.f['proposed_action'])
        self.fresh=np.array(self.f['sweep']['feedback']['state'])
        self.context=dict(stage='short_retreat',target=self.f['sweep']['target'],plane=self.f['sweep']['plane_m'])
        self.a._begin('short_retreat',1,self.previous)
        self.a.pending['steps']=23
        self.a.offset[8:15]=self.proposed[8:15]-self.previous[8:15]

    def audit(self, measured, proposed):
        return fresh_command_sweep_audit(self.k,lambda:measured,
            self.f['observation_state'],proposed,proposed,self.previous,self.context)

    def test_recorded_new_reference_blocked_previous_clear_and_full_hold_rechecked(self):
        audit=self.audit(self.fresh,self.proposed)
        self.assertTrue(retreat_pending_hold_allowed(audit))
        held=self.a.hold_retreat_reference(self.previous)
        np.testing.assert_array_equal(held,self.previous)
        np.testing.assert_allclose(self.a.offset[8:15],0.)
        self.assertEqual(self.a.pending['steps'],23)
        final=self.audit(self.fresh,held)
        self.assertTrue(all(final[k]['clear'] for k in ['proposed','filtered','previous']))

    def test_already_blocked_measured_pose_cannot_be_held(self):
        blocked=self.fresh.copy()
        blocked[8:15]+=.875*(self.proposed[8:15]-blocked[8:15])
        audit=self.audit(blocked,self.proposed)
        self.assertFalse(audit['previous']['clear'])
        self.assertFalse(retreat_pending_hold_allowed(audit))

    def test_stale_feedback_or_other_phase_does_not_authorize_hold(self):
        for change in ['feedback','stage']:
            audit=copy.deepcopy(self.f['sweep'])
            if change=='feedback':audit.pop('feedback')
            else:audit['stage']='grasp_yaw'
            self.assertFalse(retreat_pending_hold_allowed(audit))

    def test_bounds_and_phase_remain_enforced(self):
        bad=self.previous.copy();bad[8]=self.a.high[8]+.001
        with self.assertRaisesRegex(RuntimeError,'physical command bounds'):
            self.a.hold_retreat_reference(bad)
        self.a.pending['kind']='grasp'
        with self.assertRaisesRegex(RuntimeError,'active retreat'):
            self.a.hold_retreat_reference(self.previous)


if __name__=='__main__':unittest.main()
