import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scene1.kinematics import UrdfArmKinematics
from scene1.retreat_projection import FingerSeparation, project_retreat_reference
import test_retreat_projection
import test_retreat_fixed_constraint
import test_retreat_certificate


class NativeConstraintFKTests(unittest.TestCase):
    def test_fused_gaps_match_reference_for_random_poses_and_blocked_fixture(self):
        fixture = test_retreat_projection.RetreatProjectionTests()
        fixture.setUp()
        kin = fixture.kin
        sep = FingerSeparation(fixture.kw['target'], fixture.kw['plane'])
        geometry = np.r_[sep.rotation.ravel(), sep.center, sep.half, sep.finger_half]
        rng = np.random.default_rng(91265)
        low, high = kin.limits[1].T
        for q in (fixture.kw['measured'][8:15], fixture.kw['proposed'][8:15],
                  *rng.uniform(low, high, (500, 7))):
            np.testing.assert_allclose(kin.constraint_gaps(q, 1, geometry),
                                       sep.gaps(kin.matrix(q, 1)), rtol=0., atol=2e-12)

    def test_both_arm_chains_and_tool_offsets_match_reference(self):
        description = (Path(__file__).parent / 'fixtures/release_robot.urdf').read_text()
        rng = np.random.default_rng(5291025)
        for tool in ((0., 0., -.1488), (.02, -.01, -.12)):
            kin = UrdfArmKinematics(description=description, tool_xyz=tool)
            for side in (0, 1):
                low, high = kin.limits[side].T
                for q in (low, high, (low + high) / 2, *rng.uniform(low, high, (200, 7))):
                    np.testing.assert_allclose(kin.constraint_matrix(q, side), kin.matrix(q, side),
                                               rtol=0., atol=3e-15)

    def test_optimizer_cannot_bypass_original_sweep_with_bad_fk(self):
        fixture = test_retreat_projection.RetreatProjectionTests()
        fixture.setUp()
        kin = fixture.kin
        def falsely_clear(q, side, geometry):
            return np.ones(2)
        with patch('scene1.retreat_projection.FAST_CONSTRAINTS', True), \
             patch.object(kin, 'constraint_gaps', side_effect=falsely_clear):
            with self.assertRaisesRegex(RuntimeError, 'No swept-clear retreat correction'):
                project_retreat_reference(**fixture.kw)


def load_tests(loader, tests, pattern):
    for module in (test_retreat_projection, test_retreat_fixed_constraint, test_retreat_certificate):
        for original in vars(module).values():
            if not isinstance(original, type) or not issubclass(original, unittest.TestCase):
                continue
            def setup(self, parent=original):
                parent.setUp(self)
                context = patch('scene1.retreat_projection.FAST_CONSTRAINTS', True)
                context.start()
                self.addCleanup(context.stop)
            native = type('NativeFK' + original.__name__, (original,), {'setUp': setup})
            tests.addTests(loader.loadTestsFromTestCase(native))
    return tests
