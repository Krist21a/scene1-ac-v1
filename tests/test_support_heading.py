import unittest
import numpy as np
import test_residual as original


class SupportHeadingTests(unittest.TestCase):
    setUp = original.ResidualTests.setUp
    weighing_fixture = original.ResidualTests.weighing_fixture
    aim_camera = original.ResidualTests.aim_camera

    def begin(self, position=None, verified=True):
        self.weighing_fixture(yaw=22, position=position)
        self.a.config['placement_heading_mode'] = 'support_footprint'
        self.a._observe_box = lambda *args: dict(points=80,bottom_base_m=-.08,
            orientation_confirmed=verified,heading_error_degrees=22.,tilt_degrees=0.)
        self.a.weighing_target = np.array([.4,-.5,-.08])
        self.a._begin('weighing',1,self.raw,surface=self.snapshot['surfaces'][0],
            stage='correct',hold_grip=.98,reference_base=self.raw.copy(),level_ready=True)

    def test_supported_diagonal_pose_can_settle_without_act_yaw(self):
        self.begin()
        for _ in range(4):
            self.aim_camera()
            result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertTrue(info['support_footprint_fit'])
        self.assertTrue(info['heading_descent_ok'])
        self.assertEqual(self.a.pending['stage'],'release')
        self.assertGreater(result[15],.9)
        self.assertAlmostEqual(self.a.config['placement_margin_m'],.025)

    def test_near_edge_pose_still_withholds_release(self):
        self.begin(position=[.46,-.5,0.])
        for _ in range(4):
            self.aim_camera()
            result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertFalse(info['support_footprint_fit'])
        self.assertFalse(info['heading_descent_ok'])
        self.assertNotEqual(self.a.pending['stage'],'release')
        self.assertGreater(result[15],.9)

    def test_missing_support_polygon_does_not_authorize_descent(self):
        self.begin();self.a.pending['surface'].pop('footprint_xy_m')
        self.aim_camera()
        result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertFalse(info['heading_descent_ok'])
        self.assertGreater(result[15],.9)

    def test_orientation_still_requires_visual_confirmation(self):
        self.begin(verified=False)
        for _ in range(4):
            self.aim_camera()
            result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertNotEqual(self.a.pending['stage'],'release')
        self.assertEqual(info['stable_steps'],0)


if __name__ == '__main__':unittest.main()
