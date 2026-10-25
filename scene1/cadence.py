"""Higher command cadence only for explicitly selected non-contact movements."""
import math


class MotionCadence:
    CYCLE_STAGES = frozenset(('transport', 'right_retreat', 'left_clear',
                             'bin_transport', 'left_retreat', 'return_home'))
    PICKUP_STAGES = frozenset(('clear', 'first_clear', 'first_back', 'hover'))

    def __init__(self, env, make_rate, hz):
        self.env, self.make_rate = env, make_rate
        self.base_hz = float(env.ros_rate)
        self.hz = float(hz)
        if (self.base_hz != 10. or not math.isfinite(self.hz)
                or not self.base_hz <= self.hz <= 20. or env.direct_to_wbc):
            raise ValueError('Motion cadence requires the original 10 Hz trajectory mode and 10–20 Hz')
        self.original_rate = env.rate
        self.current_hz = self.base_hz

    def select(self, cycle, pickup):
        moving = (pickup is not None and pickup.active and pickup.stage in self.PICKUP_STAGES)
        moving = moving or (cycle is not None and cycle.active and cycle.stage in self.CYCLE_STAGES)
        hz = self.hz if moving else self.base_hz
        if hz != self.current_hz:
            self.env.rate = self.make_rate(hz)
            self.current_hz = hz
        period = 1./hz
        for controller in (cycle, getattr(pickup, 'motion', None)):
            if controller is not None:
                controller.control_period_s = period
        return period

    def close(self):
        self.env.rate = self.original_rate
