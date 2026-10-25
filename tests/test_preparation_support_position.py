import unittest
import numpy as np
import test_support_heading as support

class PreparationSupportPositionTests(unittest.TestCase):
    setUp=support.SupportHeadingTests.setUp
    weighing_fixture=support.SupportHeadingTests.weighing_fixture
    aim_camera=support.SupportHeadingTests.aim_camera
    begin=support.SupportHeadingTests.begin

    def run_case(self, prepared, position, verified=True):
        self.begin(position=position,verified=verified)
        self.a.wrist_config.update(release_prepare_before_open=True,release_prepare_support_footprint=True)
        if prepared:self.a.pending['release_prepare_target']=self.raw.copy()
        captured=[]
        self.a._prepare_release_reference=lambda measured,ready,result,info: (captured.append(ready) or result)
        for _ in range(4):
            self.aim_camera();result,info=self.a._pending_action(self.measured,self.snapshot)
        return result,info,captured

    def test_after_entry_supported_center_offset_does_not_freeze_preparation(self):
        result,info,captured=self.run_case(True,[.418,-.5,0.])
        self.assertFalse(info['original_centering_passed'])
        self.assertTrue(info['support_footprint_fit'])
        self.assertEqual(info['support_position_rule'],'prepared_footprint')
        self.assertTrue(captured[-1])
        self.assertAlmostEqual(self.a.config['placement_margin_m'],.025)

    def test_entry_still_requires_original_centering(self):
        result,info,captured=self.run_case(False,[.418,-.5,0.])
        self.assertFalse(info['original_centering_passed'])
        self.assertEqual(captured,[])
        self.assertNotEqual(self.a.pending['stage'],'release')

    def test_actual_edge_and_unverified_orientation_still_block(self):
        for position,verified in [([.46,-.5,0.],True),([.418,-.5,0.],False)]:
            self.setUp();result,info,captured=self.run_case(True,position,verified)
            self.assertFalse(captured[-1])
            self.assertGreater(result[15],.9)
