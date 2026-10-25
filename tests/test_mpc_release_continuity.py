import unittest
import numpy as np
import test_residual as original
import test_release_reference_transfer as transfer_tests
from scene1.motion import MotionFilter


class MpcReleaseContinuityTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp
    fixture=transfer_tests.ReleaseReferenceTests.fixture

    def test_preserves_actual_sent_mpc_joints_and_transfers_only_wrists(self):
        reference,accepted=self.fixture()
        self.a.wrist_config['release_reference_mode']='mpc_continuous'
        sent=accepted.copy();sent[8:12]-=.04
        self.a.observe_command(sent)
        self.measured[8:15]=reference[8:15]+.05
        result,info=self.a._pending_action(self.measured,self.snapshot)
        np.testing.assert_array_equal(result[8:12],sent[8:12])
        np.testing.assert_array_equal(result[12:15],self.measured[12:15])
        np.testing.assert_array_equal(self.a.pending['base'][8:15],result[8:15])
        np.testing.assert_array_equal(self.a.offset[8:15],np.zeros(7))
        transfer=info['release_reference_transfer']
        self.assertEqual(transfer['preserved_action_indices'],[8,9,10,11])
        np.testing.assert_array_equal(transfer['requested_change_rad'][:4],np.zeros(4))
        self.assertFalse(any(transfer['bound_active']))
        self.assertIsNone(transfer['correction_cap_rad'])

    def test_missing_previous_command_is_rejected_before_release(self):
        self.fixture();self.a.wrist_config['release_reference_mode']='mpc_continuous'
        self.a.last_sent=None
        with self.assertRaisesRegex(RuntimeError,'actual previous sent'):
            self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(self.a.pending['kind'],'weighing')

    def test_original_rate_and_physical_limits_still_apply(self):
        reference,accepted=self.fixture();self.a.wrist_config['release_reference_mode']='mpc_continuous'
        self.a.observe_command(accepted);self.measured[13]=reference[13]+.7
        result,_=self.a._pending_action(self.measured,self.snapshot)
        motion=MotionFilter(self.a.low,self.a.high,1.,.6);motion.reset(accepted)
        sent,audit=motion.apply(result,.1)
        np.testing.assert_array_equal(sent[8:12],accepted[8:12])
        self.assertTrue(audit['velocity_limited'])
        self.assertFalse(audit['bounds_clipped'])
        self.assertLessEqual(np.max(abs(sent[8:15]-accepted[8:15])),.0600000001)

    def test_out_of_bounds_previous_reference_is_rejected(self):
        self.fixture();self.a.wrist_config['release_reference_mode']='mpc_continuous'
        self.a.last_sent=self.raw.copy();self.a.last_sent[8]=self.a.high[8]+.01
        with self.assertRaisesRegex(RuntimeError,'Previous MPC release reference'):
            self.a._pending_action(self.measured,self.snapshot)
