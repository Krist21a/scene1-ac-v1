import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import yaml
from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle
from scene1.residual import ResidualAssist
from scene1.first_pickup import begin_first_pickup

class FirstPickupTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).parent;self.f=json.loads((p/'fixtures/first_transport_20261024.json').read_text())
        self.k=UrdfArmKinematics(description=(p/'fixtures/release_robot.urdf').read_text())
        self.cfg=yaml.safe_load((p.parent/'configs/visual_cycle_four_parcels.yaml').read_text())
        lo=np.r_[self.k.limits[0][:,0],0.,self.k.limits[1][:,0],0.];hi=np.r_[self.k.limits[0][:,1],1.,self.k.limits[1][:,1],1.]
        self.measured=np.array(self.f['contact']['observation_state']);self.sent=np.array(self.f['contact']['sent_action']);self.measured[[7,15]]=0.;self.sent[[7,15]]=0.
        self.cycle=VisualCycle(self.cfg['visual_cycle'],self.k,lo,hi,self.sent)
        self.assist=ResidualAssist(self.cfg['visual_residual'],self.k,lo,hi);self.assist.wrist_config=self.cfg['wrist_assist']
        self.assist.approach[1]=dict(target=self.f['grasp']['target'],plane=self.f['grasp']['plane'],aligned=True)
        self.snapshot=dict(surfaces=self.f['contact']['rgbd_surfaces'],candidates=[self.f['grasp']['target']])
    def begin(self):
        with patch('scene1.first_pickup.table_hull',return_value=None),patch('scene1.first_pickup.table_boundary',return_value=None),patch('scene1.first_pickup.roof_edges',return_value=None):
            return begin_first_pickup(self.cycle,self.assist,self.snapshot,self.measured,self.sent)
    def test_transition_keeps_sent_reference_and_contact_is_not_yet_confirmed(self):
        pickup,_=self.begin();np.testing.assert_array_equal(pickup.motion.command,self.sent)
        self.assertEqual(pickup.stage,'first_clear');self.assertIsNone(pickup.table_reference);self.assertFalse(self.assist.first_grasp_complete)
        self.assertIsNone(self.assist.geometry[1]);self.assertEqual(pickup.motion.config['stage_timeout_steps'],500)
    def test_orientation_and_observed_height_with_no_extra_initial_insertion(self):
        pickup,_=self.begin();g=pickup._target_goal(self.measured,self.assist)
        self.assertAlmostEqual(g[2,0],0.);self.assertAlmostEqual(np.rad2deg(np.arcsin(g[2,2])),15.)
        self.assertAlmostEqual(g[2,3],(pickup.plane+pickup.target['top_z_base_m'])/2)
        self.assertEqual(pickup.target['extra_insertion_m'],0.)
    def test_first_overrides_do_not_mutate_later_pickup_config(self):
        before=copy.deepcopy(self.cycle.config);pickup,_=self.begin()
        self.assertEqual(self.cycle.config,before);self.assertEqual(self.cycle.config['pickup_pitch_offsets_degrees'][0],0.)
        self.assertEqual(pickup.motion.config['pickup_grasp_gain_scale'],.35)
    def test_unaligned_context_is_rejected(self):
        self.assist.approach[1]['aligned']=False
        with self.assertRaisesRegex(RuntimeError,'confirmed initial pregrasp'):self.begin()
    def test_closed_gripper_is_rejected(self):
        self.measured[15]=.5
        with self.assertRaisesRegex(RuntimeError,'empty'):self.begin()
    def test_missing_pad_is_rejected(self):
        self.snapshot['surfaces']=[]
        with self.assertRaisesRegex(RuntimeError,'unambiguous'):self.begin()

    def test_unstable_head_does_not_anchor_table_or_move_arm(self):
        pickup,_=self.begin();pickup._stage('select');snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.eligible_targets',return_value=[self.f['grasp']['target']]):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        np.testing.assert_array_equal(command,self.sent)
        self.assertEqual(pickup.stage,'select');self.assertIsNone(pickup.table_reference)
        self.assertFalse(info['selection_view_ready']);self.assertFalse(hasattr(pickup,'anchor_target'))
    def test_three_fresh_stable_views_anchor_before_clear(self):
        pickup,_=self.begin();pickup._stage('select');snapshot=dict(self.snapshot,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.eligible_targets',return_value=[self.f['grasp']['target']]),patch.object(pickup,'_selection_view_ready',return_value=True),patch('scene1.visual_pickup.table_hull',return_value=None),patch('scene1.visual_pickup.table_boundary',return_value=None),patch('scene1.visual_pickup.roof_edges',return_value=None):
            for stamp in [1.,2.]:
                snapshot['stamp']=stamp;pickup.update(self.measured,snapshot,self.assist,None)
                self.assertEqual(pickup.stage,'select');self.assertFalse(hasattr(pickup,'anchor_target'))
            snapshot['stamp']=3.;pickup.update(self.measured,snapshot,self.assist,None)
        self.assertEqual(pickup.stage,'clear');self.assertTrue(hasattr(pickup,'anchor_target'))

    def test_visual_clear_exports_full_pending_sweep_context(self):
        pickup,_=self.begin();pickup._stage('clear');pickup.motion.goal[1]=self.k.matrix(self.measured[8:15],1)
        snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch.object(pickup.motion,'_arrived',return_value=(False,[])):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        self.assertTrue(info['visual_pickup_sweep_required'])
        self.assertEqual(self.assist.command_sweep_context['stage'],'visual_pickup_clear')
        self.assertEqual(self.assist.command_sweep_context['plane'],pickup.plane)
        self.assertIsNot(self.assist.command_sweep_context['target'],pickup.target)

    def test_initial_clear_keeps_xy_orientation_and_existing_clear_height(self):
        pickup,_=self.begin();pose=self.k.matrix(self.measured[8:15],1);goal=pickup.motion.goal[1]
        np.testing.assert_array_equal(goal[:3,:3],pose[:3,:3]);np.testing.assert_array_equal(goal[:2,3],pose[:2,3])
        self.assertAlmostEqual(goal[2,3],max(pose[2,3],self.f['grasp']['target']['top_z_base_m']+.12))
    def test_first_selector_does_not_switch_to_distant_visible_parcel(self):
        pickup,_=self.begin();pickup._stage('select');other=copy.deepcopy(self.f['grasp']['target']);other['position_base_m'][0]+=.15
        snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.eligible_targets',return_value=[other]),patch.object(pickup,'_selection_view_ready',return_value=True):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        np.testing.assert_array_equal(command,self.sent);self.assertEqual(info['first_target_matches'],0)
        self.assertEqual(pickup.stage,'select');self.assertFalse(hasattr(pickup,'anchor_target'))
    def test_first_clear_exports_final_send_sweep_and_starts_bounded_reobservation(self):
        pickup,_=self.begin();snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        self.assertEqual(pickup.stage,'first_back');self.assertTrue(info['visual_pickup_sweep_required'])
        self.assertEqual(self.assist.command_sweep_context['stage'],'visual_pickup_first_clear')
        self.assertIsNone(pickup.table_reference)

    def test_reobservation_keeps_height_rotation_and_moves_60mm_away(self):
        pickup,_=self.begin();pose=self.k.matrix(self.measured[8:15],1)
        snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])):
            pickup.update(self.measured,snapshot,self.assist,None)
        goal=pickup.motion.goal[1];np.testing.assert_array_equal(goal[:3,:3],pose[:3,:3])
        self.assertEqual(goal[2,3],pose[2,3]);self.assertAlmostEqual(np.linalg.norm(goal[:2,3]-pose[:2,3]),.06)
        self.assertGreater((goal[:2,3]-pose[:2,3])@(pose[:2,3]-np.asarray(pickup.target['position_base_m'])[:2]),0.)
    def test_reobservation_finishes_before_selection_and_has_full_sweep(self):
        pickup,_=self.begin();pickup._stage('first_back');snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])),patch('scene1.visual_pickup.cloud',return_value=(np.tile([.6,.3,.2],(160,1)),np.tile([60,150,150],(160,1)))):
            _,info=pickup.update(self.measured,snapshot,self.assist,None)
        self.assertEqual(pickup.stage,'observe_bin');self.assertEqual(self.assist.command_sweep_context['stage'],'visual_pickup_first_back')
        self.assertTrue(info['visual_pickup_sweep_required'])

    def test_sensor_hold_still_checks_inflight_reference(self):
        pickup,_=self.begin();command,info=pickup.update(self.measured,None,self.assist,None)
        np.testing.assert_array_equal(command,self.sent)
        self.assertTrue(info['visual_pickup_sweep_required'])
        self.assertEqual(self.assist.command_sweep_context['stage'],'visual_pickup_first_clear')
    def test_unverified_table_hold_still_checks_inflight_reference(self):
        pickup,_=self.begin();pickup._stage('descend');pickup.table_reference={}
        snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.table_hull',return_value=None),patch('scene1.visual_pickup.table_boundary',return_value=None):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        self.assertTrue(info['table_reference_hold']);self.assertTrue(info['visual_pickup_sweep_required'])
        np.testing.assert_array_equal(command,self.sent)

    def test_bin_observation_holds_arms_and_does_not_accept_partial_wall(self):
        pickup,_=self.begin();pickup._stage('observe_bin');pickup.bin_view_goal=np.array([.6,.3,.2]);pickup.bin_frames=0;pickup.bin_previous=None
        snapshot=dict(self.snapshot,stamp=1.,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.bin_rim',return_value=None),patch.object(pickup,'_selection_view_ready',return_value=True):
            command,info=pickup.update(self.measured,snapshot,self.assist,None)
        np.testing.assert_array_equal(command,self.sent);self.assertIsNone(pickup.rim)
        self.assertFalse(info['bin_full_boundary_observed']);self.assertEqual(pickup.stage,'observe_bin')
        np.testing.assert_array_equal(pickup.camera_target(self.measured,self.assist),pickup.bin_view_goal)
    def test_bin_requires_three_fresh_stable_full_boundaries(self):
        pickup,_=self.begin();pickup._stage('observe_bin');pickup.bin_frames=0;pickup.bin_previous=None
        rim=dict(center=np.array([.6,.3]),rim_z=.2,corners=np.array([[.4,.1],[.8,.1],[.8,.5],[.4,.5]]),stamp=1.,plane_z=-.08)
        snapshot=dict(self.snapshot,table_z_base_m=self.f['grasp']['plane'])
        with patch('scene1.visual_pickup.bin_rim',return_value=rim),patch.object(pickup,'_selection_view_ready',return_value=True):
            for stamp in [1.,1.,2.]:
                snapshot['stamp']=stamp;pickup.update(self.measured,snapshot,self.assist,None)
                self.assertEqual(pickup.stage,'observe_bin');self.assertIsNone(pickup.rim)
            snapshot['stamp']=3.;pickup.update(self.measured,snapshot,self.assist,None)
        self.assertEqual(pickup.stage,'select');self.assertIsNot(pickup.rim,rim)
