"""Cache static SDK limits while checking ROS parameter epochs before each command."""
import numpy as np


class CachedArmLimits:
    def __init__(self, info, get_cached_param, physical_limits):
        self.info = info
        self.read = get_cached_param
        self.original = info.get_arm_joint_limits
        self.run_id = self.read('/run_id')
        self.description = self.read('/humanoid_description')
        if not self.run_id or not self.description:
            raise ValueError('Arm limit cache requires a live ROS epoch and URDF')
        low, high = self.original()
        expected = np.asarray(physical_limits)
        if (expected.shape != (14, 2) or not np.isfinite(expected).all() or
                not np.array_equal(np.column_stack((low, high)), expected)):
            raise ValueError('SDK limits do not match the physical URDF; cache not installed')
        self.low, self.high = tuple(low), tuple(high)
        self._check_epoch()
        info.get_arm_joint_limits = self.get

    def _check_epoch(self):
        # get_param_cached subscribes to server updates, including deletion;
        # a change is an error, never an excuse to use a stale bound.
        if (self.read('/run_id') != self.run_id or
                self.read('/humanoid_description') != self.description):
            raise RuntimeError('ROS epoch or robot description changed during arm-limit caching')

    def get(self):
        self._check_epoch()
        return list(self.low), list(self.high)

    def close(self):
        self.info.get_arm_joint_limits = self.original
