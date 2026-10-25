from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import test_residual as residual_tests
import test_settlement as settlement_tests
from test_native_projection import NativeProjectionTests as FrameFixtures
from scene1.settlement import sampled_geometry

AVAILABLE = (Path(__file__).resolve().parents[1]/'scene1/_components.so').exists()


@unittest.skipUnless(AVAILABLE, 'Optional native kernel is not built')
class NativeSampleTests(unittest.TestCase):
    def test_sampled_geometry_keeps_all_pixels_depths_colors_and_order(self):
        for snapshot, _ in FrameFixtures().frames():
            with patch('scene1.settlement.FAST_PROJECTION', False):
                expected = sampled_geometry(snapshot)
            with patch('scene1.settlement.FAST_PROJECTION', True):
                actual = sampled_geometry(snapshot)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-15)
            np.testing.assert_array_equal(actual[2], expected[2])


@unittest.skipUnless(AVAILABLE, 'Optional native kernel is not built')
class NativeHeldGeometryTests(residual_tests.ResidualTests):
    def setUp(self):
        super().setUp()
        toggle = patch('scene1.residual.FAST_PROJECTION', True)
        toggle.start()
        self.addCleanup(toggle.stop)


@unittest.skipUnless(AVAILABLE, 'Optional native kernel is not built')
class NativeSettlementTests(settlement_tests.SettlementTests):
    def setUp(self):
        toggle = patch('scene1.settlement.FAST_PROJECTION', True)
        toggle.start()
        self.addCleanup(toggle.stop)


def load_tests(loader, tests, pattern):
    # Select original tests exercising held RGB-D geometry, not unrelated ACT logic.
    names = ('test_top_face_bias_does_not_move_estimated_box_center_up',
             'test_rgbd_roof_recovers_level_box_bottom_and_normal',
             'test_rgbd_roof_detects_tilt_and_rejects_incomplete_top',
             'test_roof_fit_separates_top_from_side_face_and_raised_label',
             'test_unexpected_roof_orientation_is_not_used_to_steer_the_arm',
             'test_roof_fit_over_forty_five_degrees_cannot_bypass_consistency_check',
             'test_partial_or_ambiguous_components_do_not_pull_the_held_box_center')
    return unittest.TestSuite([loader.loadTestsFromTestCase(NativeSampleTests),
                              loader.loadTestsFromTestCase(NativeSettlementTests),
                              unittest.TestSuite(NativeHeldGeometryTests(name) for name in names)])
