import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import test_residual as original


class FirstGraspGainTests(unittest.TestCase):
    setUp=original.ResidualTests.setUp

    def test_scaled_increment_and_original_bounds(self):
        base=self.raw.copy()
        full=self.a._correct(base,self.measured,1,[.001,0.,0.])
        self.a.offset[:]=0.
        low=self.a._correct(base,self.measured,1,[.001,0.,0.],gain_scale=.35)
        np.testing.assert_allclose(low-base,.35*(full-base),atol=1e-10)
        self.assertLessEqual(max(abs(low-base)),self.a.config['max_joint_step_rad'])

    def test_scale_not_stacked_on_later_visual_grasp(self):
        self.a.wrist_config['first_grasp_gain_scale']=.35
        target=self.snapshot['candidates'][0]
        for later, expected in [(False,.35),(True,1.)]:
            kwargs={'grasp_rotation':Rotation.identity()} if later else {}
            self.a._begin('grasp',1,self.raw,target=target,plane=-.08,grasp_z=0.,
                wrist_verified=False,**kwargs)
            with patch.object(self.a,'_correct',return_value=self.raw.copy()) as correct:
                self.a._pending_action(self.measured,self.snapshot)
                self.assertEqual(correct.call_args.kwargs['gain_scale'],expected)

    def test_invalid_scale_rejected(self):
        for scale in [0.,-.1,1.1,float('nan'),float('inf')]:
            with self.assertRaisesRegex(ValueError,'Correction gain scale'):
                self.a._correct(self.raw,self.measured,1,[0.,0.,0.],gain_scale=scale)


if __name__=='__main__':unittest.main()
