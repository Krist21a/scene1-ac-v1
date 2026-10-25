import unittest
import numpy as np
import test_release_preparation as preparation


class SettledPreparationTests(unittest.TestCase):
    setUp=preparation.PreparationTests.setUp
    fixture=preparation.PreparationTests.fixture
    setup_preparation=preparation.PreparationTests.setup_preparation

    def test_static_contact_shift_is_aligned_in_original_steps_before_opening(self):
        accepted=self.setup_preparation()
        self.a.wrist_config['release_prepare_adapt_settled']=True
        result=self.a._prepare_release_reference(self.measured,True,accepted,{})
        self.a.observe_command(result)
        # The measured settled wrist shifted by the recorded 0.048rad while
        # the previous snapshot stayed fixed. No plant prediction is implied.
        self.measured[12]+=.048
        adjusted=0
        for _ in range(30):
            self.a.pending['steps']+=1
            info={};result=self.a._prepare_release_reference(self.measured,True,accepted,info)
            delta=np.array(info['release_preparation']['settled_target_adjustment_rad'])
            self.assertLessEqual(max(abs(delta)),self.a.config['max_joint_step_rad']+1e-12)
            adjusted+=bool(np.any(delta))
            self.assertEqual(result[15],.98)
            self.a.observe_command(result)
            if self.a.pending.get('reference_prepared'):break
        self.assertGreaterEqual(adjusted,3)
        self.assertTrue(self.a.pending['reference_prepared'])
        self.assertLessEqual(max(abs(result[12:15]-self.measured[12:15])),.015)

    def test_target_does_not_chase_unsettled_wrist_or_lost_support(self):
        for support,movement in [(False,0.),(True,.01)]:
            self.setUp();accepted=self.setup_preparation()
            self.a.wrist_config['release_prepare_adapt_settled']=True
            result=self.a._prepare_release_reference(self.measured,True,accepted,{})
            before=result.copy();self.a.observe_command(result);self.measured[12]+=.048
            for _ in range(10):
                self.measured[13]+=movement
                info={};result=self.a._prepare_release_reference(self.measured,support,accepted,info)
                np.testing.assert_array_equal(result,before)
            self.assertFalse(self.a.pending.get('reference_prepared',False))

    def test_adaptation_cannot_extend_weighing_timeout(self):
        accepted=self.setup_preparation();self.a.wrist_config['release_prepare_adapt_settled']=True
        self.a.pending['steps']=self.a.config['max_release_steps']
        with self.assertRaisesRegex(RuntimeError,'original weighing budget'):
            self.a._prepare_release_reference(self.measured,False,accepted,{})


if __name__=='__main__':unittest.main()
