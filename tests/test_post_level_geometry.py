import copy
import unittest
from unittest.mock import patch
import numpy as np
from scene1.settlement import SettlementMonitor
import test_residual as original


class PostLevelGeometryTests(unittest.TestCase):
    setUp = original.ResidualTests.setUp
    _insert_fixture = original.ResidualTests._insert_fixture

    def fixture(self):
        self._insert_fixture([.3, -.2, 0.], dict(correction_base_xy=[0.,0.], centering_error_px=0.))
        self.a.wrist_config['regrasp_reobserve_after_level'] = True
        p=self.a.pending
        p.update(regrasp_stage='reobserve', reobserve_hold=self.raw.copy(),
                 reobserve_steps=0, reobserve_monitor=SettlementMonitor(),
                 surface={}, extent=np.array([.1,.04,.1]), steps=123)
        target=copy.deepcopy(p['target']);target['position_base_m'][1]-=.02
        return dict(target=target, center_xy_m=target['position_base_m'][:2],
                    top_height_m=.1, tilt_degrees=2., footprint_margin_m=.02,
                    support_height_ok=True, stamp=0.)

    def test_refresh_requires_new_support_frames_and_preserves_reference_budget(self):
        observed=self.fixture();base=self.a.pending['base'].copy();offset=self.a.offset.copy()
        for i in range(5):
            observed['stamp']=1.+i*.3
            with patch('scene1.residual.resting_observation',return_value=copy.deepcopy(observed)):
                result,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(result[15],0.)
            np.testing.assert_array_equal(self.a.pending['base'],base)
            np.testing.assert_array_equal(self.a.offset,offset)
            if i<4:self.assertNotIn('post_level_geometry_refresh',info)
        self.assertTrue(info['post_level_geometry_refresh']['support_reconfirmed'])
        self.assertAlmostEqual(info['post_level_geometry_refresh']['xy_shift_m'],.02)
        self.assertEqual(self.a.pending['steps'],128)
        self.assertEqual(self.a.pending['regrasp_stage'],'level')
        self.assertTrue(self.a.pending['post_level_geometry_verified'])
        self.assertNotIn('grasp_rotation',self.a.pending)
        self.assertEqual(self.a.approach[1]['target']['position_base_m'],observed['target']['position_base_m'])

    def test_duplicate_frames_cannot_refresh(self):
        observed=self.fixture();observed['stamp']=1.
        with patch('scene1.residual.resting_observation',return_value=observed):
            for _ in range(6):
                result,info=self.a._pending_action(self.measured,self.snapshot)
                self.assertEqual(result[15],0.)
                self.assertNotIn('post_level_geometry_refresh',info)
        self.assertEqual(self.a.pending['regrasp_stage'],'reobserve')

    def test_original_footprint_requirement_is_not_weakened(self):
        observed=self.fixture();observed['footprint_margin_m']=.009
        for i in range(6):
            observed['stamp']=1.+i*.3
            with patch('scene1.residual.resting_observation',return_value=copy.deepcopy(observed)):
                result,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(info['post_level_settlement']['reason'],'unsupported_or_outside_pad')
            self.assertEqual(result[15],0.)
            self.assertNotIn('post_level_geometry_refresh',info)

    def test_missing_geometry_keeps_original_hundred_step_support_budget(self):
        self.fixture()
        with patch('scene1.residual.resting_observation',return_value=None):
            for _ in range(99):
                result,info=self.a._pending_action(self.measured,self.snapshot)
                self.assertEqual(result[15],0.)
            with self.assertRaisesRegex(RuntimeError,'Post-level parcel support'):
                self.a._pending_action(self.measured,self.snapshot)

    def test_overall_regrasp_budget_is_not_reset(self):
        self.fixture();self.a.pending['steps']=899
        with patch('scene1.residual.resting_observation',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'within allowed joint correction'):
                self.a._pending_action(self.measured,self.snapshot)

    def test_confirmed_geometry_is_not_requested_again_after_releveling(self):
        self.fixture();p=self.a.pending
        p['regrasp_stage']='level';p['post_level_geometry_verified']=True
        p['stable']=0
        for _ in range(3):
            result,info=self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(result[15],0.)
        self.assertEqual(p['regrasp_stage'],'align')
