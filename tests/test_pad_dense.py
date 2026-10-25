import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from scene1.rgbd import surface_candidates
from scene1.visual_pickup import visible_pad
from run_scene1 import read_manipulation_rgbd


class DensePadTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).parent / 'fixtures'
        self.meta = json.loads((fixtures / 'pad_dense_seed3.json').read_text())
        with np.load(fixtures / 'pad_dense_seed3.npz') as data:
            self.snapshot = dict(data)
        self.snapshot['table_z_base_m'] = self.meta['plane']
        self.snapshot['surfaces'] = self.surfaces(4)

    def surfaces(self, stride):
        s = self.snapshot
        return surface_candidates(s['rgb'], s['depth_m'], s['intrinsic'],
                                  s['camera_to_base'], self.meta['plane'],
                                  self.meta['config'], stride=stride)

    def test_recorded_occlusion_passes_original_area_and_edge_gates_with_dense_samples(self):
        reference = copy.deepcopy(self.meta['reference'])
        self.assertFalse(visible_pad(reference, dict(surfaces=self.surfaces(4))))
        self.assertTrue(visible_pad(reference, dict(surfaces=self.surfaces(2))))
        self.assertEqual(reference, self.meta['reference'])

    def test_only_active_pad_observation_resamples_same_snapshot(self):
        for stage, active in [('observe_pad', True), ('observe_pad', False), ('carry', True)]:
            with self.subTest(stage=stage, active=active):
                source = self.snapshot
                result = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: source),
                    SimpleNamespace(pending=None), None,
                    SimpleNamespace(stage=stage, active=active),
                    dict(cartesian_alignment=self.meta['config']))
                if active and stage == 'observe_pad':
                    self.assertEqual(result['surface_sampling_stride'], 2)
                    self.assertIs(result['depth_m'], source['depth_m'])
                    self.assertTrue(visible_pad(self.meta['reference'], result))
                else:
                    self.assertIs(result, source)
                self.assertNotIn('surface_sampling_stride', source)

    def test_dense_observation_cannot_pass_wrong_height_or_outside_reference(self):
        surfaces = self.surfaces(2)
        for case in ['height', 'outside']:
            changed = copy.deepcopy(surfaces)
            for surface in changed:
                if case == 'height':
                    surface['position_base_m'][2] += .03
                else:
                    surface['footprint_xy_m'] = (np.asarray(surface['footprint_xy_m']) + [.1, 0.]).tolist()
            self.assertFalse(visible_pad(self.meta['reference'], dict(surfaces=changed)))

    def test_invalid_sampling_rejects(self):
        with self.assertRaises(ValueError):
            self.surfaces(3)
