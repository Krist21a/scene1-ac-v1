"""Lossless numeric sensor archives with bounded synchronous encoding work."""
from pathlib import Path
import zipfile

import numpy as np


def write_sensor_snapshot(path, payload, *, compression_level=1):
    """Keep each NumPy member byte-identical; change only DEFLATE effort.

    Finish the archive before replacing the destination, so a failed write
    cannot present a partial snapshot as usable evidence. No frames are queued
    or dropped, and this function returns only after the whole write finishes.
    """
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError('Snapshot compression level must be an integer from 0 to 9')
    arrays = {name: np.asanyarray(value) for name, value in payload.items()}
    if any(not name or '/' in name or '\\' in name for name in arrays):
        raise ValueError('Sensor snapshot names must be simple nonempty names')
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError('Sensor snapshots must contain numeric arrays, not pickled objects')
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compression_level) as archive:
            for name, array in arrays.items():
                with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
