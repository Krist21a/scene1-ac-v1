import unittest
import numpy as np
import test_release_open_hold as opening

class SoftOpenTests(unittest.TestCase):
    setUp=opening.ReleaseOpenHoldTests.setUp
    fixture=opening.ReleaseOpenHoldTests.fixture

    def test_progressive_command_preserves_hold_and_waits_for_feedback(self):
        self.fixture();self.a.wrist_config.update(release_gripper_open_step=.08,release_open_dwell_steps=4)
        previous=self.raw.copy();previous[15]=.98;self.a.observe_command(previous)
        self.measured[15]=.5
        for i in range(13):
            result,info=self.a._pending_action(self.measured,None)
            self.assertAlmostEqual(result[15],max(0.,.98-.08*(i+1)))
            self.assertIn('release_pose_hold',info)
            self.assertIsNone(self.a.pending.get('goal'))
            self.assertLessEqual(previous[15]-result[15],.08+1e-12)
            np.testing.assert_array_equal(result[8:15],previous[8:15])
            self.a.observe_command(result);previous=result.copy()
        self.assertEqual(result[15],0.)
        self.assertEqual(self.a.pending['steps'],13)

    def test_measured_open_does_not_skip_command_ramp(self):
        self.fixture();self.a.wrist_config.update(release_gripper_open_step=.08,release_open_dwell_steps=4)
        previous=self.raw.copy();previous[15]=.5;self.a.observe_command(previous)
        self.a.pending['steps']=5;self.measured[15]=.01
        result,info=self.a._pending_action(self.measured,None)
        self.assertAlmostEqual(result[15],.42)
        self.assertIn('release_pose_hold',info)

    def test_original_open_timeout_retained(self):
        self.fixture();self.a.wrist_config['release_gripper_open_step']=.08
        self.a.observe_command(self.raw);self.a.pending['steps']=39;self.measured[15]=.5
        with self.assertRaisesRegex(RuntimeError,'gripper did not open'):
            self.a._pending_action(self.measured,None)

    def test_invalid_step_rejected(self):
        for step in [0.,-1.,2.,float('nan')]:
            self.fixture();self.a.wrist_config['release_gripper_open_step']=step
            with self.assertRaisesRegex(ValueError,'finite'):
                self.a._release_open_command()
