import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import test_regrasp_forward_gate as gates

class InsertPitchTests(unittest.TestCase):
    setUp=gates.ForwardGateTests.setUp
    _insert_fixture=gates.ForwardGateTests._insert_fixture
    fixture=gates.ForwardGateTests.fixture

    def test_only_insert_inside_original_orientation_gate_uses_pitch_weight(self):
        self.fixture(0);self.a.wrist_config['regrasp_insert_pitch_weight']=.3
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertTrue(info['insert_pitch_task']['relaxed'])
        self.assertEqual(info['insert_pitch_task']['weight'],.3)
        self.assertLessEqual(max(abs(result[8:15]-self.raw[8:15])),.0048+1e-12)
        self.assertEqual(result[15],0.)
        self.a.pending['regrasp_stage']='align'
        self.a.pending['stand_off_position']=self.measured[8:11].copy()
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertFalse(info['insert_pitch_task']['relaxed'])

    def test_outside_orientation_gate_restores_full_task_and_withholds_close(self):
        self.fixture(0);self.a.wrist_config['regrasp_insert_pitch_weight']=.3
        self.a.pending['grasp_rotation']=Rotation.from_rotvec([.2,0,0])*self.a.pending['grasp_rotation']
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertFalse(info['insert_pitch_task']['relaxed'])
        self.assertEqual(result[15],0.)
        self.assertEqual(info['stable_steps'],0)

    def test_original_view_and_sweep_gates_still_apply(self):
        self.fixture(15);self.a.wrist_config['regrasp_insert_pitch_weight']=.3
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertFalse(info['regrasp_forward_gate']['centered'])
        with patch('scene1.residual.open_fingers_clear',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Open finger sweep intersects'):
                for _ in range(20):self.a._pending_action(self.measured,self.snapshot)

    def test_invalid_weight_rejected(self):
        for weight in [0.,-1.,1.1,float('nan')]:
            self.fixture(0);self.a.wrist_config['regrasp_insert_pitch_weight']=weight
            with self.assertRaisesRegex(ValueError,'finite'):
                self.a._pending_action(self.measured,self.snapshot)
