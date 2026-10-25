import unittest
import numpy as np
import test_release_preparation as preparation

class ArmPreparationTests(unittest.TestCase):
    setUp=preparation.PreparationTests.setUp
    fixture=preparation.PreparationTests.fixture
    setup_preparation=preparation.PreparationTests.setup_preparation

    def setup_arm(self):
        accepted=self.setup_preparation()
        self.measured[8:15]=accepted[8:15]
        self.measured[10]+=.151
        self.a.wrist_config.update(release_prepare_whole_arm=True,release_prepare_adapt_settled=True)
        return accepted

    def test_recorded_proximal_bias_cannot_open_with_only_wrist_settled(self):
        accepted=self.setup_arm();previous=accepted.copy()
        for i in range(100):
            self.a.pending['steps']+=1
            info={};result=self.a._prepare_release_reference(self.measured,True,accepted,info)
            self.assertLessEqual(max(abs(result[8:15]-previous[8:15])),self.a.config['max_joint_step_rad']+1e-12)
            self.assertEqual(result[15],.98)
            self.a.observe_command(result);previous=result.copy()
            if self.a.pending.get('reference_prepared'):break
        self.assertGreater(i,10)
        self.assertTrue(self.a.pending['reference_prepared'])
        self.assertLessEqual(max(abs(result[8:15]-self.measured[8:15])),.015)

    def test_proximal_motion_or_support_loss_prevents_adaptation_and_opening(self):
        for support,drift in [(False,0.),(True,.01)]:
            self.setUp();accepted=self.setup_arm()
            for i in range(8):
                self.measured[8]+=drift
                result=self.a._prepare_release_reference(self.measured,support,accepted,{})
                np.testing.assert_array_equal(result[8:15],accepted[8:15])
                self.a.observe_command(result)
            self.assertFalse(self.a.pending.get('reference_prepared',False))

    def test_original_timeout_still_applies_to_proximal_bias(self):
        accepted=self.setup_arm();self.a.pending['steps']=self.a.config['max_release_steps']
        with self.assertRaisesRegex(RuntimeError,'original weighing budget'):
            self.a._prepare_release_reference(self.measured,True,accepted,{})
