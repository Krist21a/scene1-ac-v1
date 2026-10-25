import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_residual as original_tests


class PhaseReferenceHoldTests(unittest.TestCase):
    setUp=original_tests.ResidualTests.setUp
    def test_release_retains_valid_inactive_hold_and_original_active_budget(self):
        reference=self.raw.copy();reference[4]=-4.01;reference[7]=-.002
        accepted=self.raw.copy();accepted[4]=-4.;accepted[8]+=.12;accepted[15]=.98
        self.a.geometry[1]=dict(extent=np.array([.1,.04,.1]))
        self.a._view_ready=lambda *args:True
        self.a._begin('weighing',1,accepted,stage='release',reference_base=reference.copy(),
                      requested_open=0.,surface={},hold_grip=.98)
        action,info=self.a._pending_action(self.measured,self.snapshot)
        np.testing.assert_array_equal(self.a.pending['base'][8:15],reference[8:15])
        np.testing.assert_array_equal(self.a.pending['base'][:8]+self.a.offset[:8],action[:8])
        np.testing.assert_allclose(self.a.offset[8:15],action[8:15]-reference[8:15],atol=0,rtol=0)
        self.assertEqual(info['phase_reference_hold']['frozen_act_reference'][4],-4.01)
        self.assertEqual(info['phase_reference_hold']['frozen_act_reference'][7],-.002)
        for _ in range(12):
            held,_=self.a._pending_action(self.measured,self.snapshot)
            np.testing.assert_allclose(held,action,atol=1e-15,rtol=0)
            self.assertTrue(np.all(held>=self.a.low) and np.all(held<=self.a.high))

    def test_invalid_release_is_rejected_without_projecting_or_hiding_it(self):
        bad=self.raw.copy();bad[4]=-4.01
        self.a.geometry[1]=dict(extent=np.array([.1,.04,.1]))
        self.a._view_ready=lambda *args:True
        self.a._begin('weighing',1,bad,stage='release',reference_base=self.raw.copy(),
                      requested_open=0.,surface={},hold_grip=.98)
        with self.assertRaisesRegex(RuntimeError,'release command is outside'):
            self.a._pending_action(self.measured,self.snapshot)


if __name__=='__main__':unittest.main()
