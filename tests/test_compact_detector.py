from pathlib import Path
import unittest
import numpy as np
import yaml
from scene1.vision import ParcelDetector


class CompactDetectorTests(unittest.TestCase):
    def test_real_images_keep_all_detections_tracking_and_input_bytes(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root/'configs/visual_cycle_four_parcels.yaml').read_text())['vision']
        with np.load(root/'tests/fixtures/bin_rim_outlier_v53.npz') as data:
            frames = [data[f'frame_{step}_rgb'] for step in (60, 80, 100, 120)]
        for table in (True, False):
            settings = dict(config, table_segmentation=table)
            original = ParcelDetector(settings)
            compact = ParcelDetector(dict(settings, compact_regions=True))
            for frame in frames*3:
                before = frame.copy()
                self.assertEqual(original.detect(frame), compact.detect(frame))
                np.testing.assert_array_equal(frame, before)

    def test_float_chw_still_uses_original_validation(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root/'configs/visual_cycle_four_parcels.yaml').read_text())['vision']
        original = ParcelDetector(config)
        compact = ParcelDetector(dict(config, compact_regions=True))
        frame = np.zeros((3, 480, 848), dtype=np.float32)
        self.assertEqual(original.detect(frame), compact.detect(frame))
        frame[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compact.detect(frame)
