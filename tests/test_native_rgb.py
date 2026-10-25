from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scene1.vision import rgb_image


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'scene1/_components.so').exists(),
                     'Optional native kernel is not built')
class NativeRGBTests(unittest.TestCase):
    def test_exact_output_for_pixels_random_values_and_noncontiguous_fallback(self):
        rng = np.random.default_rng(110)
        frames = [np.tile(np.arange(256, dtype=np.float32)/255, (3, 10, 1)),
                  rng.random((3, 17, 19), dtype=np.float32),
                  rng.random((1, 3, 17, 19), dtype=np.float32),
                  rng.random((3, 17, 19)),
                  rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)]
        frames += [frames[1][:, :, ::-1], frames[1].transpose(1, 2, 0)]
        for frame in frames:
            with patch('scene1.vision.FAST_IMAGE_CONVERSION', False):
                expected = rgb_image(frame)
            with patch('scene1.vision.FAST_IMAGE_CONVERSION', True):
                actual = rgb_image(frame)
            np.testing.assert_array_equal(actual, expected)

    def test_unusual_values_preserve_original_validation(self):
        for value in (-.001, float('nan'), float('inf'), 1.0001, 1.001, 1.002):
            frame = np.full((3, 4, 5), value, dtype=np.float32)
            with patch('scene1.vision.FAST_IMAGE_CONVERSION', False):
                try:
                    expected = rgb_image(frame)
                except ValueError:
                    expected = None
            with patch('scene1.vision.FAST_IMAGE_CONVERSION', True):
                if expected is None:
                    with self.assertRaises(ValueError):
                        rgb_image(frame)
                else:
                    np.testing.assert_array_equal(rgb_image(frame), expected)
