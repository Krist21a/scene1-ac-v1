import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle
from scene1.first_transport import begin_first_transport

class FirstTransportTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).parent
        self.f=json.loads((p/'fixtures/first_transport_20261024.json').read_text())
        self.k=UrdfArmKinematics(description=(p/'fixtures/release_robot.urdf').read_text())
        config=yaml.safe_load((p.parent/'configs/visual_cycle_four_parcels.yaml').read_text())
        lo=np.r_[self.k.limits[0][:,0],0.,self.k.limits[1][:,0],0.]
        hi=np.r_[self.k.limits[0][:,1],1.,self.k.limits[1][:,1],1.]
        self.measured=np.array(self.f['contact']['observation_state']);self.sent=np.array(self.f['contact']['sent_action'])
        self.cycle=VisualCycle(config['visual_cycle'],self.k,lo,hi,self.sent)
        self.assist=SimpleNamespace(first_grasp_complete=True,regrasp_complete=False,pending=None,
            geometry=[None,dict(center=np.array([0.,0.,.02]),rotation=Rotation.identity(),extent=np.array([.11,.04,.10]))])
        self.snapshot=dict(surfaces=copy.deepcopy(self.f['contact']['rgbd_surfaces']))
    def begin(self):
        return begin_first_transport(self.cycle,self.assist,{},self.f['grasp'],self.snapshot,self.measured,self.sent)
    def test_confirmed_contact_preserves_all_sent_commands_and_existing_lift(self):
        pickup,pad=self.begin();np.testing.assert_array_equal(pickup.motion.command,self.sent)
        expected=self.k.matrix(self.measured[8:15],1);expected[2,3]+=.16
        np.testing.assert_array_equal(pickup.motion.goal[1],expected)
        self.assertEqual(pickup.stage,'lift');self.assertEqual(pickup.motion.config['stage_timeout_steps'],500)
        self.assertEqual(pickup.completed_parcels,0);self.assertEqual(pickup.pose_frames,0)
    def test_unconfirmed_contact_cannot_start(self):
        self.assist.first_grasp_complete=False
        with self.assertRaisesRegex(RuntimeError,'confirmed first contact'):self.begin()
    def test_missing_and_ambiguous_support_cannot_start(self):
        original=copy.deepcopy(self.snapshot)
        self.snapshot['surfaces']=[]
        with self.assertRaisesRegex(RuntimeError,'unambiguous'):self.begin()
        self.snapshot=original;self.snapshot['surfaces'].append(copy.deepcopy(self.snapshot['surfaces'][-1]))
        with self.assertRaisesRegex(RuntimeError,'unambiguous'):self.begin()
    def test_lost_grip_cannot_start(self):
        self.measured[15]=0.
        with self.assertRaisesRegex(RuntimeError,'contact'):self.begin()
    def test_later_parcel_cannot_restart_first_transport(self):
        self.cycle.completed=1
        with self.assertRaisesRegex(RuntimeError,'confirmed first contact'):self.begin()
    def test_preserves_geometry_without_aliasing(self):
        pickup,pad=self.begin();pickup.motion.geometry['center'][0]=2
        self.assertEqual(self.assist.geometry[1]['center'][0],0.)
