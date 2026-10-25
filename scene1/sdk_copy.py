"""Local numeric-vector copy acceleration; no SDK files or subscriptions change."""
import copy
from functools import update_wrapper
from types import FunctionType


def numeric_deepcopy(value, memo=None):
    # ROS float arrays deserialize as flat tuples. deepcopy already returns
    # those tuples unchanged, but walks every element in Python first.
    # Lists still receive independent storage; anything else uses deepcopy.
    if memo is None and type(value) in (tuple, list):
        snapshot = value if type(value) is tuple else value.copy()
        if all(type(item) in (float, int, bool) for item in snapshot):
            return snapshot
    return copy.deepcopy(value, memo)


class _CopyProxy:
    deepcopy = staticmethod(numeric_deepcopy)

    def __getattr__(self, name):
        return getattr(copy, name)


class SensorNumericCopies:
    """Clone only the two SDK callbacks with a private copy-module binding."""
    names = ('_sensors_data_raw_callback', '_sensors_data_raw_shm_callback')

    def __init__(self, state_class):
        if getattr(getattr(state_class, '_instance', None), '_initialized', False):
            raise RuntimeError('Sensor copy acceleration must precede SDK subscriptions')
        self.state_class = state_class
        self.originals = {}
        replacements = {}
        for name in self.names:
            original = getattr(state_class, name)
            if not isinstance(original, FunctionType) or original.__globals__.get('copy') is not copy:
                raise RuntimeError('Unsupported SDK sensor callback copy binding')
            namespace = dict(original.__globals__, copy=_CopyProxy())
            function = FunctionType(original.__code__, namespace, original.__name__,
                                    original.__defaults__, original.__closure__)
            function.__kwdefaults__ = original.__kwdefaults__
            update_wrapper(function, original)
            self.originals[name] = original
            replacements[name] = function
        for name, function in replacements.items():
            setattr(state_class, name, function)

    def close(self):
        for name, function in self.originals.items():
            setattr(self.state_class, name, function)
