"""Opt-in motion cadence with unchanged final contact/evidence dwell gates."""
import math
import numpy as np


class MotionCadence:
    CYCLE_STAGES = frozenset(('transport', 'right_retreat', 'left_clear',
                             'bin_transport', 'left_retreat', 'return_home'))
    PICKUP_STAGES = frozenset(('clear', 'first_clear', 'first_back', 'hover'))
    ALLOWED_CYCLE_STAGES = CYCLE_STAGES | frozenset(('lift', 'left_approach'))
    ALLOWED_PICKUP_STAGES = PICKUP_STAGES | frozenset(('lift', 'carry'))

    def __init__(self, env, make_rate, hz, cycle_stages=None, cycle_rates=None,
                 pickup_stages=None, pickup_rates=None, guarded_motion_hz=None):
        self.env, self.make_rate = env, make_rate
        self.base_hz = float(env.ros_rate)
        self.hz = float(hz)
        if (self.base_hz != 10. or not math.isfinite(self.hz)
                or not self.base_hz <= self.hz <= 25. or env.direct_to_wbc):
            raise ValueError('Motion cadence requires the original 10 Hz trajectory mode and 10–25 Hz')
        self.original_rate = env.rate
        self.current_hz = self.base_hz
        self.guarded_hz = None if guarded_motion_hz is None else float(guarded_motion_hz)
        if self.guarded_hz is not None and (not math.isfinite(self.guarded_hz)
                or not self.base_hz <= self.guarded_hz <= min(self.hz, 20.)):
            raise ValueError('Guarded motion cadence must stay within configured rates')
        self.cycle_stages = self.CYCLE_STAGES if cycle_stages is None else frozenset(cycle_stages)
        allowed_cycle = self.ALLOWED_CYCLE_STAGES | (frozenset(('left_insert',)) if self.guarded_hz else frozenset())
        if not self.cycle_stages <= allowed_cycle:
            raise ValueError('Fast cadence cannot include contact/evidence stages')
        self.cycle_rates = dict(cycle_rates or {})
        if (not self.cycle_rates.keys() <= self.cycle_stages or any(
                not math.isfinite(float(rate)) or not self.base_hz <= float(rate) <= self.hz
                for rate in self.cycle_rates.values())):
            raise ValueError('Per-stage cadence must stay between the original and configured rates')
        self.pickup_stages = self.PICKUP_STAGES if pickup_stages is None else frozenset(pickup_stages)
        self.pickup_rates = dict(pickup_rates or {})
        allowed_pickup = self.ALLOWED_PICKUP_STAGES | (frozenset(('descend','prepare_weighing')) if self.guarded_hz else frozenset())
        if not self.pickup_stages <= allowed_pickup:
            raise ValueError('Fast pickup cadence cannot include contact/evidence stages')
        if (not self.pickup_rates.keys() <= self.pickup_stages or any(
                not math.isfinite(float(rate)) or not self.base_hz <= float(rate) <= self.hz
                for rate in self.pickup_rates.values())):
            raise ValueError('Per-stage pickup cadence must stay between the original and configured rates')

    def select(self, cycle, pickup, measured=None, assist=None):
        moving = (pickup is not None and pickup.active and pickup.stage in self.pickup_stages)
        hz = float(self.pickup_rates.get(pickup.stage, self.hz)) if moving else self.base_hz
        if moving and pickup.stage in ('descend','prepare_weighing'):
            if not self.guarded_hz or pickup.stage not in pickup.motion.config.get('servo_time_normalization_stages',[]):
                raise ValueError('Guarded pickup motion requires feedback-time damping')
            hz = self.guarded_hz
        pending = getattr(assist, 'pending', None)
        if self.guarded_hz and pending is not None:
            regrasp = (pending['kind'] == 'pregrasp' and getattr(assist, 'weighing_target', None) is not None
                and 'support_z_base_m' in pending.get('target', {})
                and pending.get('regrasp_stage', 'level') in ('level', 'align', 'descend', 'insert'))
            if regrasp or pending['kind'] == 'short_retreat':
                hz = self.guarded_hz
        if cycle is not None and cycle.active and cycle.stage in self.cycle_stages:
            allowed = True
            if (cycle.stage == 'return_home'
                    and getattr(cycle, 'config', {}).get('return_home_settle', False)):
                allowed = measured is not None and not cycle.settle_home(measured)
            if cycle.stage == 'left_approach':
                # Wrist edge confirmation begins within 40 mm. Restore 10 Hz
                # at 80 mm, before that region, retaining its original dwell.
                allowed = False
                if measured is not None and cycle.goal[0] is not None:
                    tool = cycle.k.matrix(np.asarray(measured)[:7], 0)
                    allowed = np.linalg.norm(tool[:3, 3]-cycle.goal[0][:3, 3]) > .08
                if self.guarded_hz and cycle.config.get('guarded_receiver_approach', False):
                    if not cycle.config.get('servo_profiles',{}).get('left_approach',{}).get('time_normalize',False):
                        raise ValueError('Guarded receiver approach requires feedback-time damping')
                    # VisualCycle requires >=3 fresh centered observations
                    # spanning >=.2 s, including at the faster approach rate.
                    allowed = True
            if allowed:
                hz = float(self.cycle_rates.get(cycle.stage, self.hz))
                if cycle.stage == 'left_approach' and self.guarded_hz:
                    hz = min(hz, self.guarded_hz)
                if cycle.stage == 'left_insert':
                    if 'left_insert' not in cycle.config.get('servo_time_normalization_stages', []):
                        raise ValueError('Fast insertion requires consumed-feedback damping normalization')
                    hz = self.guarded_hz
        if hz != self.current_hz:
            self.env.rate = self.make_rate(hz)
            self.current_hz = hz
        period = 1./hz
        for controller in (cycle, getattr(pickup, 'motion', None)):
            if controller is not None:
                controller.control_period_s = period
        if assist is not None:
            assist.control_period_s = period
        return period

    def close(self):
        self.env.rate = self.original_rate
