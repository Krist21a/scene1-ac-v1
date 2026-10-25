import unittest
import numpy as np
from analyze_reference import stats


class ReferenceAnalysisTests(unittest.TestCase):
    def test_known_velocity_delay_and_no_static_plateau(self):
        t = np.arange(0, 8, .01)
        ref = np.repeat((.1*np.sin(2*np.pi*.8*t))[:,None], 14, axis=1)
        q = np.repeat((.1*np.sin(2*np.pi*.8*(t-.12)))[:,None], 14, axis=1)
        result = stats(ref,q,np.gradient(q,.01,axis=0),np.gradient(ref,.01,axis=0),.01)
        self.assertIsNone(result['wrist']['static_bias_rad'])
        for r in result['wrist']['approximate_dynamic_lag_per_joint']:
            self.assertAlmostEqual(r['sim_lag_s'],.12,places=2)
            self.assertGreater(r['velocity_correlation'],.99)

    def test_static_bias_is_distinguished_from_dynamic_delay(self):
        ref = np.zeros((300,14)); q = ref+.07; v = ref.copy()
        result = stats(ref,q,v,v,.01)
        np.testing.assert_allclose(result['proximal']['static_bias_rad'],.07)
        self.assertEqual(result['wrist']['approximate_dynamic_lag_per_joint'],[None]*6)
        self.assertLess(max(result['wrist']['static_position_std_rad']),1e-12)


if __name__ == '__main__':unittest.main()
