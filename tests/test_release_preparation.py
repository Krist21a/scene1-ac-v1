import unittest
import numpy as np
import test_residual as original
import test_release_reference_transfer as transfer
from scene1.motion import MotionFilter


class PreparationTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp
    fixture=transfer.ReleaseReferenceTests.fixture

    def setup_preparation(self):
        _,accepted=self.fixture()
        self.a.pending['stage']='correct'
        self.a.wrist_config.update(release_reference_mode='mpc_continuous',release_prepare_before_open=True)
        self.a.observe_command(accepted)
        self.measured[12:15]=accepted[12:15]+np.array([.14,-.05,-.336])
        return accepted

    def test_handoff_finishes_before_opening_and_release_has_no_jump(self):
        accepted=self.setup_preparation();motion=MotionFilter(self.a.low,self.a.high,1.,.6);motion.reset(accepted)
        for i in range(20):
            self.a.pending['steps']+=1
            result=self.a._prepare_release_reference(self.measured,True,accepted,{})
            self.assertEqual(result[15],.98)
            sent,limits=motion.apply(result,.1)
            self.assertFalse(limits['bounds_clipped'])
            self.a.observe_command(sent)
            if self.a.pending.get('reference_prepared'):break
            self.assertEqual(self.a.pending['stage'],'correct')
        self.assertGreaterEqual(i,6)
        self.assertTrue(self.a.pending['reference_prepared'])
        self.assertEqual(self.a.pending['stage'],'release')
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(result[15],0.)
        np.testing.assert_array_equal(result[8:15],sent[8:15])
        np.testing.assert_array_equal(info['release_reference_transfer']['requested_change_rad'],np.zeros(7))

    def test_support_loss_prevents_opening_even_with_matching_feedback(self):
        accepted=self.setup_preparation()
        for _ in range(10):
            result=self.a._prepare_release_reference(self.measured,False,accepted,{})
            self.a.observe_command(result)
        self.assertFalse(self.a.pending.get('reference_prepared',False))
        self.assertEqual(self.a.pending['stage'],'correct')
        self.assertEqual(result[15],.98)

    def test_same_original_budget_expires_with_gripper_closed(self):
        accepted=self.setup_preparation();self.a.pending['steps']=self.a.config['max_release_steps']
        with self.assertRaisesRegex(RuntimeError,'original weighing budget'):
            self.a._prepare_release_reference(self.measured,True,accepted,{})
        self.assertEqual(self.a.pending['stage'],'correct')
        self.assertFalse(self.a.opening[1])

    def test_large_feedback_movement_restarts_stability(self):
        accepted=self.setup_preparation()
        for _ in range(3):
            result=self.a._prepare_release_reference(self.measured,True,accepted,{})
            self.a.observe_command(result)
        self.measured[14]+=.01
        info={};self.a._prepare_release_reference(self.measured,True,accepted,info)
        self.assertEqual(info['release_preparation']['stable_steps'],0)

    def test_invalid_target_is_rejected_before_opening(self):
        accepted=self.setup_preparation();self.measured[14]=self.a.high[14]+.01
        with self.assertRaisesRegex(RuntimeError,'physical bounds'):
            self.a._prepare_release_reference(self.measured,True,accepted,{})
        self.assertFalse(self.a.opening[1])


if __name__=='__main__':unittest.main()
