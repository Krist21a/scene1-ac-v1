import unittest
import numpy as np
import test_residual as original

class ReleaseReferenceTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp

    def fixture(self):
        self.a.wrist_config['release_measured_reference_hold']=True
        self.a.config['max_placement_joint_offset_rad']=.5235987756
        reference=self.raw.copy()
        accepted=self.raw.copy();accepted[8:15]+=.15;accepted[15]=.98
        self.a.geometry[1]=dict(extent=np.array([.1,.04,.1]))
        self.a._view_ready=lambda *args:True
        self.a._begin('weighing',1,accepted,stage='release',reference_base=reference.copy(),
                      requested_open=0.,surface={},hold_grip=.98)
        return reference,accepted

    def test_transfer_to_measured_pose_retains_original_reference_and_inactive_hold(self):
        reference,accepted=self.fixture()
        self.measured[8:15]=reference[8:15]+.05
        result,info=self.a._pending_action(self.measured,self.snapshot)
        np.testing.assert_array_equal(result[8:15],self.measured[8:15])
        np.testing.assert_array_equal(result[:8],accepted[:8])
        np.testing.assert_array_equal(self.a.pending['base'][8:15],reference[8:15])
        np.testing.assert_allclose(self.a.offset[8:15],result[8:15]-reference[8:15])
        self.assertEqual(result[15],0.)
        self.assertFalse(any(info['release_reference_transfer']['bound_active']))
        for _ in range(12):
            held,_=self.a._pending_action(self.measured,self.snapshot)
            np.testing.assert_allclose(held,result)

    def test_transfer_cannot_reset_or_enlarge_existing_offset_budget(self):
        reference,accepted=self.fixture()
        self.measured[11]=reference[11]+.7
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertAlmostEqual(result[11]-reference[11],.5235987756)
        self.assertTrue(info['release_reference_transfer']['bound_active'][3])
        self.assertAlmostEqual(info['release_reference_transfer']['remaining_target_error_rad'][3],.5235987756-.7)
        np.testing.assert_array_equal(self.a.pending['base'][8:15],reference[8:15])

    def test_invalid_physical_feedback_is_not_projected_into_success(self):
        self.fixture();self.measured[8]=self.a.high[8]+.01
        with self.assertRaisesRegex(RuntimeError,'Measured release pose is outside'):
            self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(self.a.pending['kind'],'weighing')

    def test_original_release_view_check_still_blocks_opening(self):
        _,accepted=self.fixture();self.a._view_ready=lambda *args:False
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(info['state'],'waiting_release_view')
        self.assertEqual(result[15],accepted[15])
        self.assertNotIn('release_reference_transfer',info)

    def test_visual_release_can_hold_measured_pose_beyond_old_act_budget(self):
        reference,accepted=self.fixture()
        self.a.wrist_config['release_reference_mode']='measured_physical'
        self.measured[11]=reference[11]+.7
        result,info=self.a._pending_action(self.measured,self.snapshot)
        np.testing.assert_array_equal(result[8:15],self.measured[8:15])
        np.testing.assert_array_equal(self.a.pending['base'][8:15],self.measured[8:15])
        np.testing.assert_array_equal(self.a.offset[8:15],np.zeros(7))
        self.assertFalse(self.a.unwinding_placement)
        self.assertIsNone(info['release_reference_transfer']['correction_cap_rad'])
        self.assertFalse(info['phase_reference_hold']['active_arm_reference_unchanged'])
        self.assertEqual(info['phase_reference_hold']['active_reference_source'],'measured_physical_release')
        for _ in range(12):
            held,_=self.a._pending_action(self.measured,self.snapshot)
            np.testing.assert_allclose(held,result)

    def test_visual_release_retains_physical_bound_rejection(self):
        self.fixture();self.a.wrist_config['release_reference_mode']='measured_physical'
        self.measured[8]=self.a.high[8]+.01
        with self.assertRaisesRegex(RuntimeError,'Measured release pose is outside'):
            self.a._pending_action(self.measured,self.snapshot)

    def test_visual_release_target_still_passes_original_motion_rate_limit(self):
        from scene1.motion import MotionFilter
        reference,accepted=self.fixture()
        self.a.wrist_config['release_reference_mode']='measured_physical'
        self.measured[11]=reference[11]+.7
        result,_=self.a._pending_action(self.measured,self.snapshot)
        motion=MotionFilter(self.a.low,self.a.high,1.,.6);motion.reset(accepted)
        sent,audit=motion.apply(result,.1)
        self.assertTrue(audit['velocity_limited'])
        self.assertFalse(audit['bounds_clipped'])
        self.assertLessEqual(np.max(abs(sent[8:15]-accepted[8:15])),.0600000001)
