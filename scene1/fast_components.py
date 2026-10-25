"""Optional exact native version of the existing ordered union-find loop."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def native_fk():
    # A bounded, tiny chain calculation: retain the GIL to avoid a thread
    # handoff for each optimizer finite-difference sample.
    library = ctypes.PyDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_fk_chain
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integers = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [doubles, doubles, integers, ctypes.c_int64, doubles, doubles, doubles]
    function.restype = None
    return function


@lru_cache(maxsize=1)
def native_constraint_gaps():
    library = ctypes.PyDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_fk_finger_gaps
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integers = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [doubles, doubles, integers, ctypes.c_int64, doubles, doubles, doubles, doubles]
    function.restype = None
    return function


@lru_cache(maxsize=1)
def native_boxes():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_boxes_overlap
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    function.argtypes = [doubles] * 6
    function.restype = ctypes.c_int
    return function


def boxes_overlap_native(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    arrays = [np.asarray(x) for x in
              (center_a, rotation_a, half_a, center_b, rotation_b, half_b)]
    if [a.shape for a in arrays] != [(3,), (3, 3), (3,), (3,), (3, 3), (3,)]:
        return None
    # Runtime FK and parcel envelopes are float64. Preserve the original
    # arithmetic for callers using float32 or other numeric representations.
    if any(a.dtype != np.float64 for a in arrays):
        return None
    arrays = [np.ascontiguousarray(a) for a in arrays]
    result = native_boxes()(*arrays)
    return None if result < 0 else bool(result)


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


def sampled_geometry(snapshot, stride=2):
    import cv2
    depth = snapshot['depth_m']
    points = project_grid(depth, snapshot['intrinsic'], snapshot['camera_to_base'], stride)
    hsv = cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV)
    return (depth[::stride, ::stride].reshape(-1), points.reshape(-1, 3),
            hsv[::stride, ::stride].reshape(-1, 3))


@lru_cache(maxsize=1)
def native_roof():
    library = ctypes.CDLL(str(Path(__file__).with_name('_components.so')))
    function = library.scene1_roof_mask
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    bytes_ = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    function.argtypes = [doubles, bytes_, ctypes.c_int64, ctypes.c_int64,
                         doubles, doubles, doubles, doubles, doubles, ctypes.c_double,
                         doubles, bytes_, bytes_]
    function.restype = None
    return function


def roof_masks(depth, hsv, intrinsic, matrix, center, axes, size, top):
    depth, k, m, center, axes, size = [np.ascontiguousarray(a, dtype=np.float64)
                                     for a in (depth, intrinsic, matrix, center, axes, size)]
    hsv = np.ascontiguousarray(hsv, dtype=np.uint8)
    if (depth.ndim != 2 or hsv.shape != (*depth.shape, 3) or k.shape != (3, 3)
            or m.shape != (4, 4) or center.shape != (2,) or axes.shape != (2, 2) or size.shape != (2,)):
        raise ValueError('Invalid roof-mask dimensions')
    h, w = depth.shape
    local = np.empty((h, w, 2), dtype=np.float64)
    roof = np.empty((h, w), dtype=np.uint8)
    lower = np.empty_like(roof)
    native_roof()(depth, hsv, h, w, k, m, center, axes, size, float(top), local, roof, lower)
    return local, roof.astype(bool), lower.astype(bool)
