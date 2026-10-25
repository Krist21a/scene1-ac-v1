import unittest
from unittest.mock import patch
import numpy as np
import test_residual as original

class ForwardGateTests(unittest.TestCase):
    setUp = original.ResidualTests.setUp
    _insert_fixture = original.ResidualTests._insert_fixture

    def fixture(self, error):
        self._insert_fixture([.335,-.2,0],dict(correction_base_xy=[0,.006],centering_error_px=error))
        self.a.wrist_config['regrasp_center_before_advance'] = True

    def test_off_center_view_retains_lateral_control_without_forward_demand(self):
        self.fixture(15)
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertAlmostEqual(result[8],self.measured[8])
        self.assertGreater(result[9],self.measured[9])
        self.assertEqual(result[15],0)
        self.assertFalse(info['regrasp_forward_gate']['centered'])

    def test_centered_view_advances_inside_original_step_cap(self):
        self.fixture(0)
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertGreater(result[8],self.measured[8])
        self.assertLessEqual(np.max(abs(result[8:15]-self.raw[8:15])),.0048+1e-9)
        self.assertTrue(info['open_fingers_clear'])
        self.assertEqual(result[15],0)

    def test_off_center_moving_hand_keeps_derivative_braking(self):
        self.fixture(15)
        self.a.pending['last_position']=self.measured[8:11]-[.001,0,0]
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertLess(result[8],self.measured[8])
        self.assertEqual(result[15],0)

    def test_original_sweep_stop_remains_active(self):
        self.fixture(0)
        with patch('scene1.residual.open_fingers_clear',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Open finger sweep intersects'):
                for _ in range(20):
                    self.a._pending_action(self.measured,self.snapshot)

    def test_admissible_off_deadband_view_can_advance_and_recenter(self):
        self.fixture(9)
        result, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertGreater(result[8], self.measured[8])
        self.assertGreater(result[9], self.measured[9])
        self.assertTrue(info['regrasp_forward_gate']['centered'])
        self.assertEqual(result[15], 0)

    def test_original_admission_boundary_is_inclusive(self):
        self.fixture(10)
        result, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertGreater(result[8], self.measured[8])
        self.assertTrue(info['regrasp_forward_gate']['centered'])

    def test_outside_original_admission_still_blocks(self):
        self.fixture(10.001)
        result, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertAlmostEqual(result[8], self.measured[8])
        self.assertFalse(info['regrasp_forward_gate']['centered'])

    def test_runtime_admission_threshold_is_respected(self):
        self.fixture(9)
        self.a.wrist_config['regrasp_aligned_error_px'] = 8
        result, info = self.a._pending_action(self.measured, self.snapshot)
        self.assertAlmostEqual(result[8], self.measured[8])
        self.assertFalse(info['regrasp_forward_gate']['centered'])
