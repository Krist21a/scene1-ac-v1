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


@lru_cache(maxsize=1)
def native_rgb():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_rgb_chw
    function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
                         ctypes.c_int64,
                         np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')]
    function.restype = ctypes.c_int
    return function


def convert_rgb_chw(image):
    if (image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3
            or not image.flags.c_contiguous or not image.size):
        return None
    output = np.empty((*image.shape[1:], 3), dtype=np.uint8)
    return output if native_rgb()(image, image.shape[1]*image.shape[2], output) == 0 else None


@lru_cache(maxsize=1)
def native_cloud():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_cloud
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    bytes_ = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    function.argtypes = [doubles, bytes_, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         doubles, doubles, doubles, bytes_]
    function.restype = ctypes.c_int64
    return function


def project_cloud(depth, hsv, intrinsic, matrix, stride):
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    hsv = np.ascontiguousarray(hsv, dtype=np.uint8)
    k = np.ascontiguousarray(intrinsic, dtype=np.float64)
    m = np.ascontiguousarray(matrix, dtype=np.float64)
    if (depth.ndim != 2 or hsv.shape != (*depth.shape, 3) or k.shape != (3, 3)
            or m.shape != (4, 4) or type(stride) is not int or stride < 1):
        raise ValueError('Invalid native point-cloud dimensions or stride')
    h, w = depth.shape
    size = ((h+stride-1)//stride)*((w+stride-1)//stride)
    points = np.empty((size, 3), dtype=np.float64)
    colors = np.empty((size, 3), dtype=np.uint8)
    count = native_cloud()(depth, hsv, h, w, stride, k, m, points, colors)
    return points[:count], colors[:count]


@lru_cache(maxsize=1)
def native_grid():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_project_grid
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    function.argtypes = [doubles, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         doubles, doubles, doubles]
    function.restype = None
    return function


def project_grid(depth, intrinsic, matrix, stride=1):
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    k = np.ascontiguousarray(intrinsic, dtype=np.float64)
    m = np.ascontiguousarray(matrix, dtype=np.float64)
    if (depth.ndim != 2 or k.shape != (3, 3) or m.shape != (4, 4)
            or type(stride) is not int or stride < 1):
        raise ValueError('Invalid organized projection dimensions or stride')
    h, w = depth.shape
    points = np.empty(((h+stride-1)//stride, (w+stride-1)//stride, 3), dtype=np.float64)
    native_grid()(depth, h, w, stride, k, m, points)
    return points
