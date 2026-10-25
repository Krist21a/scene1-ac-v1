import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from scene1.snapshots import write_sensor_snapshot


def members(source):
    with zipfile.ZipFile(source) as z:
        return {name: z.read(name) for name in z.namelist()}


class SnapshotEncodingTests(unittest.TestCase):
    def test_members_match_numpy_for_numeric_layouts_and_special_values(self):
        payload = dict(rgb=np.arange(120, dtype=np.uint8).reshape(5, 8, 3),
            depth_m=np.array([[0., -0., np.nan, np.inf, -np.inf]], dtype=np.float32),
            intrinsic=np.asfortranarray(np.eye(3)),
            view=np.arange(30, dtype='>f8').reshape(5, 6)[:, ::2],
            scalar=np.array(1.25), empty=np.empty((0, 3)))
        original = io.BytesIO(); np.savez_compressed(original, **payload)
        expected = members(original)
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'sensors.npz'
            write_sensor_snapshot(target, payload)
            self.assertEqual(members(target), expected)
            self.assertEqual(list(Path(d).iterdir()), [target])
            with np.load(target, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(payload))

    def test_real_rgbd_fixture_members_are_unchanged(self):
        fixture = Path(__file__).parent/'fixtures/pickup_height_v61.npz'
        with np.load(fixture, allow_pickle=False) as saved:
            payload = {k: saved[k] for k in saved.files}
        original = io.BytesIO(); np.savez_compressed(original, **payload)
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'sensors.npz'
            write_sensor_snapshot(target, payload)
            self.assertEqual(members(target), members(original))

    def test_encoding_failure_preserves_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sensors.npz'; path.write_bytes(b'prior evidence')
            with patch('numpy.lib.format.write_array', side_effect=OSError('disk write failed')):
                with self.assertRaises(OSError):write_sensor_snapshot(path, dict(rgb=np.zeros(3)))
            self.assertEqual(path.read_bytes(), b'prior evidence')
            self.assertEqual(list(Path(d).iterdir()), [path])

    def test_commit_failure_preserves_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sensors.npz'; path.write_bytes(b'prior evidence')
            with patch.object(Path, 'replace', side_effect=OSError('rename failed')):
                with self.assertRaises(OSError):write_sensor_snapshot(path, dict(rgb=np.zeros(3)))
            self.assertEqual(path.read_bytes(), b'prior evidence')
            self.assertEqual(list(Path(d).iterdir()), [path])

    def test_invalid_level_names_and_objects_do_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sensors.npz'
            for level in (-1, 10, True, 1.5):
                with self.assertRaises(ValueError):write_sensor_snapshot(path, dict(rgb=np.zeros(3)), compression_level=level)
            for payload in ({'../bad':np.zeros(3)}, {'rgb':np.array([{}],dtype=object)}):
                with self.assertRaises(ValueError):write_sensor_snapshot(path, payload)
            self.assertEqual(list(Path(d).iterdir()), [])
