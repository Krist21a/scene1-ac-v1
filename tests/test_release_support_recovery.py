import unittest
import numpy as np
import test_support_heading as support

class SupportRecoveryTests(unittest.TestCase):
    setUp=support.SupportHeadingTests.setUp
    weighing_fixture=support.SupportHeadingTests.weighing_fixture
    aim_camera=support.SupportHeadingTests.aim_camera
    begin=support.SupportHeadingTests.begin

    def fixture(self,verified=True):
        self.begin(position=[.46,-.5,0.],verified=verified)
        self.a.wrist_config.update(release_prepare_before_open=True,release_prepare_recover_support=True)
        target=self.raw.copy();target[15]=.98
        self.a.pending.update(release_prepare_target=target,release_prepare_stable=0,release_prepare_quiet=2,steps=100)
        self.a.observe_command(target)

    def test_lost_footprint_returns_to_correction_without_jump_or_budget_reset(self):
        self.fixture();before=self.a.last_sent.copy()
        for _ in range(3):
            self.aim_camera();result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(info['state'],'recovering_release_support')
        self.assertEqual(self.a.pending['steps'],103)
        self.assertNotIn('release_prepare_target',self.a.pending)
        np.testing.assert_array_equal(result[8:15],before[8:15])
        self.assertGreater(result[15],.9)
        self.assertAlmostEqual(info['support_recovery']['original_margin_m'],.025)
        self.aim_camera();result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertNotIn('release_preparation',info)
        self.assertEqual(self.a.pending['steps'],104)
        self.assertGreater(result[15],.9)

    def test_unverified_view_cannot_start_recovery_motion(self):
        self.fixture(verified=False)
        for _ in range(5):
            self.aim_camera();result,info=self.a._pending_action(self.measured,self.snapshot)
        self.assertIn('release_prepare_target',self.a.pending)
        self.assertNotIn('support_recovery',info)
        self.assertGreater(result[15],.9)

    def test_original_budget_cannot_be_extended_by_recovery(self):
        self.fixture();self.a.pending['release_support_lost_frames']=2
        self.a.pending['steps']=self.a.config['max_release_steps']-1
        self.aim_camera()
        with self.assertRaisesRegex(RuntimeError,'original weighing budget'):
            self.a._pending_action(self.measured,self.snapshot)
