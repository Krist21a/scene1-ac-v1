import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.kinematics import UrdfArmKinematics


def fixture():
    xml = '<robot name="mixed_axes">'
    axes = ['1 0 0', '0 1 0', '0 0 1', '1 1 0', '0 1 1', '1 0 1', '0 0 1']
    for side in ('l', 'r'):
        for i, axis in enumerate(axes, 1):
            parent = 'base_link' if i == 1 else f'zarm_{side}{i-1}_link'
            xml += (f'<joint name="zarm_{side}{i}_joint" type="revolute">'
                    f'<parent link="{parent}"/><child link="zarm_{side}{i}_link"/>'
                    f'<origin xyz=".03 -.02 -.1" rpy=".1 -.2 .3"/><axis xyz="{axis}"/>'
                    '<limit lower="-2" upper="2"/></joint>')
    return xml+'</robot>'


class JacobianTests(unittest.TestCase):
    def test_base_and_tool_frame_jacobians_match_central_differences(self):
        model = UrdfArmKinematics(description=fixture(), tool_xyz=(.02, -.03, -.1488))
        rng = np.random.default_rng(105)
        for side in (0, 1):
            for q in rng.uniform(-1.5, 1.5, (40, 7)):
                pose, jac = model.matrix_and_jacobian(q, side)
                np.testing.assert_allclose(pose, model.matrix(q, side), atol=1e-14)
                for j in range(7):
                    offset = np.eye(7)[j]*1e-6
                    plus, minus = model.matrix(q+offset, side), model.matrix(q-offset, side)
                    angular = (Rotation.from_matrix(plus[:3, :3]) *
                               Rotation.from_matrix(minus[:3, :3]).inv()).as_rotvec()/2e-6
                    numeric = np.r_[(plus[:3, 3]-minus[:3, 3])/2e-6, angular]
                    np.testing.assert_allclose(jac[:, j], numeric, atol=2e-9)
                    local = pose[:3, :3].T @ jac[3:, j]
                    numeric_local = (Rotation.from_matrix(minus[:3, :3]).inv() *
                                     Rotation.from_matrix(plus[:3, :3])).as_rotvec()/2e-6
                    np.testing.assert_allclose(local, numeric_local, atol=2e-9)

    def test_cache_is_exact_bounded_and_returns_independent_arrays(self):
        model = UrdfArmKinematics(description=fixture(), fast=True)
        q = np.zeros(7)
        original = model.matrix(q, 0)
        model.matrix(q, 0)[:] = 999.
        np.testing.assert_array_equal(model.matrix(q, 0), original)
        q[0] += 1e-8
        self.assertGreater(np.max(abs(model.matrix(q, 0)-original)), 1e-10)
        for x in np.linspace(-1., 1., 300):
            model.matrix(np.full(7, x), 1)
        self.assertLessEqual(model._cached_matrix.cache_info().currsize, 256)
