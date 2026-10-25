import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scene1.fast_components import project_grid
from scene1.approach_edges import roof_edges


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'scene1/_components.so').exists(),
                     'Optional native kernel is not built')
class NativeProjectionTests(unittest.TestCase):
    def frames(self):
        root = Path(__file__).parent/'fixtures'
        for path in sorted(root.glob('approach_edges_*.npz')):
            with np.load(path) as data:
                for key in data.files:
                    if not key.endswith('_target'):
                        continue
                    name = key[:-len('_target')]
                    yield ({k: data[name+'_'+k] for k in
                            ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')},
                           json.loads(str(data[key])))

    def compare(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.compare(actual[key], expected[key])
        elif isinstance(expected, (bool, int, str)) or expected is None:
            self.assertEqual(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)

    def test_real_roof_edges_keep_identical_support_and_motion_decisions(self):
        for snapshot, target in self.frames():
            for shift in (0., .014):
                snapshot['camera_to_base'] = snapshot['camera_to_base'].copy()
                snapshot['camera_to_base'][0, 3] += shift
                with patch('scene1.approach_edges.FAST_PROJECTION', False):
                    expected = roof_edges(snapshot, target)
                with patch('scene1.approach_edges.FAST_PROJECTION', True):
                    actual = roof_edges(snapshot, target)
                self.compare(actual, expected)

    def test_organized_projection_keeps_invalid_samples_and_grid_positions(self):
        snapshot, _ = next(self.frames())
        depth = snapshot['depth_m'].copy()
        depth.flat[:6] = [np.nan, np.inf, 0., -.1, .1, 3.]
        k, m = snapshot['intrinsic'], snapshot['camera_to_base']
        for stride in (1, 2, 3, 4):
            v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
            z = depth[v, u]
            with np.errstate(invalid='ignore'):
                camera = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=-1)
                expected = camera @ m[:3, :3].T+m[:3, 3]
            actual = project_grid(depth, k, m, stride)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
