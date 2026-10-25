"""Optional exact native version of the existing ordered union-find loop."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def native_union():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_union_pairs
    array = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [array, ctypes.c_int64, array, ctypes.c_int64]
    function.restype = None
    return function


def component_labels(pairs, n):
    pairs = np.ascontiguousarray(pairs, dtype=np.int64).reshape(-1, 2)
    if n < 0 or (pairs.size and (pairs.min() < 0 or pairs.max() >= n)):
        raise ValueError('Invalid component graph')
    parent = np.arange(n, dtype=np.int64)
    native_union()(pairs, len(pairs), parent, n)
    return parent
