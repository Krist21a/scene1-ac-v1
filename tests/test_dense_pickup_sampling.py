import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import yaml

from run_scene1 import read_manipulation_rgbd
from scene1.rgbd import metric_candidates
from scene1.residual import ResidualAssist


class DensePickupSamplingTests(unittest.TestCase):
    def test_sampling_is_selected_before_geometry_and_keeps_other_stages(self):
        config = dict(performance=dict(dense_pickup_candidates=True))
        for stage in ('select', 'clear', 'hover', 'descend', 'lift', 'carry', 'local_grasp'):
            scene = SimpleNamespace(read=Mock(return_value=None))
            pickup = SimpleNamespace(active=True, stage=stage)
            read_manipulation_rgbd(scene, object(), None, pickup, config)
            stride = 2 if stage in ('select', 'clear', 'hover', 'descend') else 4
            scene.read.assert_called_once_with(candidate_stride=stride, spatial_components=False)

    def test_recorded_occlusion_dense_estimate_matches_full_resolution_and_keeps_motion_guard(self):
        root = Path(__file__).parent/'fixtures'
        snapshot = dict(np.load(root/'approach_sampling_s4_20261026.npz'))
        target = json.loads((root/'approach_sampling_s4_20261026.json').read_text())['target']
        config = yaml.safe_load((root.parent.parent/'configs/visual_cycle_four_parcels.yaml').read_text())['cartesian_alignment']
        def observe(stride, shift=0.):
            matrix = snapshot['camera_to_base'].copy(); matrix[0,3] += shift
            candidates, plane = metric_candidates(snapshot['rgb'], snapshot['depth_m'],
                snapshot['intrinsic'], matrix, config, stride=stride)
            out = dict(snapshot, candidates=candidates, table_z_base_m=plane)
            nearest = min(candidates, key=lambda c: np.linalg.norm(
                np.asarray(c['position_base_m'])[:2]-np.asarray(target['position_base_m'])[:2]))
            return ResidualAssist._parcel_displacement(out, copy.deepcopy(target)), nearest
        sparse, _ = observe(4)
        dense, candidate = observe(2)
        full, full_candidate = observe(1)
        self.assertGreater(sparse, .01)
        self.assertLess(dense, .01)
        self.assertLess(full, .01)
        np.testing.assert_allclose(candidate['position_base_m'][:2],
                                   full_candidate['position_base_m'][:2], atol=.0002)
        self.assertGreater(observe(2, .014)[0], .01)
        self.assertGreater(observe(2, -.020)[0], .01)
