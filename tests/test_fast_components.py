from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scene1.rgbd import spatial_groups


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'scene1/_components.so').exists(),
                     'Optional native kernel is not built')
class FastComponentsTests(unittest.TestCase):
    def test_exact_membership_and_order_against_reference(self):
        rng = np.random.default_rng(109)
        sets = [np.empty((0, 3)), np.zeros((1, 3)), np.zeros((20, 3))]
        sets += [rng.normal(size=(100, 3))*scale for scale in (.005, .02, .1)]
        sets += [np.vstack((rng.normal(size=(50, 3))*.002,
                           rng.normal(size=(50, 3))*.002 + [.1, 0, 0]))]
        for points in sets:
            with patch('scene1.rgbd.FAST_COMPONENTS', False):
                expected = spatial_groups(points)
            with patch('scene1.rgbd.FAST_COMPONENTS', True):
                actual = spatial_groups(points)
            self.assertEqual([a.tolist() for a in actual], [e.tolist() for e in expected])

    def test_dense_real_bin_keeps_exact_rim(self):
        import json
        from scene1.visual_cycle import bin_rim
        directory = Path(__file__).parent/'fixtures'
        fixture = json.loads((directory/'bin_rim_outlier_v53.json').read_text())
        with np.load(directory/'bin_rim_outlier_v53.npz') as raw:
            for step in (60, 80, 100):
                row = next(x for x in fixture['frames'] if x['step'] == step)
                snap = dict(row, **{k: raw[f'frame_{step}_'+k] for k in
                            ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')})
                with patch('scene1.rgbd.FAST_COMPONENTS', False):
                    expected = bin_rim(snap, stride=1)
                with patch('scene1.rgbd.FAST_COMPONENTS', True):
                    actual = bin_rim(snap, stride=1)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key])

    def test_invalid_graph_rejected_before_native_call(self):
        from scene1.fast_components import component_labels
        for pairs in ([[-1, 0]], [[0, 2]]):
            with self.assertRaises(ValueError):
                component_labels(pairs, 2)
