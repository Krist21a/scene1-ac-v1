from pathlib import Path
import json
import unittest
from unittest.mock import patch
import numpy as np
from scene1.visual_cycle import cloud
from scene1.visual_pickup import table_hull
from scene1.table_reference import table_frame, table_boundary


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'scene1/_components.so').exists(),
                     'Optional native kernel is not built')
class NativeCloudTests(unittest.TestCase):
    def snapshots(self):
        root = Path(__file__).parent/'fixtures'
        fixture = json.loads((root/'bin_rim_outlier_v53.json').read_text())
        with np.load(root/'bin_rim_outlier_v53.npz') as data:
            for step in (60, 80, 100):
                row = next(x for x in fixture['frames'] if x['step'] == step)
                yield dict(row, **{k: data[f'frame_{step}_'+k] for k in
                    ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')})

    def test_dense_and_sparse_points_preserve_selection_order_and_coordinates(self):
        for snap in self.snapshots():
            for stride in (1, 2, 3, 4):
                with patch('scene1.visual_cycle.FAST_CLOUD', False):
                    expected, colors = cloud(snap, stride)
                with patch('scene1.visual_cycle.FAST_CLOUD', True):
                    actual, actual_colors = cloud(snap, stride)
                np.testing.assert_array_equal(actual_colors, colors)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)

    def test_real_table_geometry_retains_original_decisions_and_float32_hull(self):
        for snap in self.snapshots():
            with patch('scene1.visual_cycle.FAST_CLOUD', False):
                expected = table_hull(snap, 1)
                ref = table_frame(expected, snap['table_z_base_m'], table_boundary(snap))
            with patch('scene1.visual_cycle.FAST_CLOUD', True):
                actual = table_hull(snap, 1)
                result = table_frame(actual, snap['table_z_base_m'], table_boundary(snap))
            np.testing.assert_array_equal(actual, expected)
            if ref is None:
                self.assertIsNone(result)
            else:
                for key in ref:
                    np.testing.assert_array_equal(result[key], ref[key])

    def test_invalid_depth_values_have_identical_filtering(self):
        snap = next(self.snapshots())
        snap['depth_m'] = snap['depth_m'].copy()
        snap['depth_m'].flat[:6] = [float('nan'), float('inf'), 0., .1, 2.5, -1.]
        with patch('scene1.visual_cycle.FAST_CLOUD', False):
            expected, colors = cloud(snap, 1)
        with patch('scene1.visual_cycle.FAST_CLOUD', True):
            actual, actual_colors = cloud(snap, 1)
        np.testing.assert_array_equal(actual_colors, colors)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
