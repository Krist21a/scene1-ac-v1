from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scene1.cadence import MotionCadence
import test_speed_profiles as profiles


class MotionCadenceTests(unittest.TestCase):
    def test_home_settling_latches_original_rate_and_resets_next_cycle(self):
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=None)
        cadence = MotionCadence(env, lambda hz: hz, 20)
        cycle = profiles.SpeedProfileTests().cycle({})
        cycle.config['return_home_settle'] = True
        cycle.active = True; cycle._stage('return_home')
        measured = cycle.home.copy(); measured[0] += .3
        self.assertEqual(cadence.select(cycle, None, measured), .05)
        measured[0] -= .15
        self.assertEqual(cadence.select(cycle, None, measured), .1)
        measured[0] += .15
        self.assertEqual(cadence.select(cycle, None, measured), .1)
        cycle._stage('return_home')
        self.assertEqual(cadence.select(cycle, None, measured), .05)

    def test_pickup_lift_override_keeps_contact_and_observation_at_ten_hz(self):
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=None)
        cadence = MotionCadence(env, lambda hz: hz, 20,
            pickup_stages=['hover', 'lift'], pickup_rates={'lift': 15})
        pickup = SimpleNamespace(active=True, stage='lift', motion=SimpleNamespace())
        self.assertAlmostEqual(cadence.select(None, pickup), 1./15)
        self.assertAlmostEqual(pickup.motion.control_period_s, 1./15)
        self.assertEqual(env.ros_rate, 10)
        for stage in ('descend', 'local_grasp', 'local_weighing', 'carry', 'observe_pad'):
            pickup.stage = stage
            self.assertEqual(cadence.select(None, pickup), .1)

    def test_pickup_override_rejects_contact_stages_and_excess_rates(self):
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=None)
        for stages, rates in ((['descend'], {}), (['lift'], {'lift': 21}),
                              (['lift'], {'lift': 9}), (['lift'], {'carry': 15})):
            with self.assertRaises(ValueError):
                MotionCadence(env, lambda hz: hz, 20, pickup_stages=stages, pickup_rates=rates)

    def test_contact_and_policy_stages_keep_original_rate_and_observation_window(self):
        original = object()
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=original)
        cadence = MotionCadence(env, lambda hz: ('rate', hz), 20)
        cycle = SimpleNamespace(active=True, stage='transport')
        self.assertEqual(cadence.select(cycle, None), .05)
        self.assertEqual(env.rate, ('rate', 20))
        self.assertEqual(env.ros_rate, 10)
        for stage in ('left_insert', 'close_left', 'release_right', 'release_left', 'verify_bin', 'lift'):
            cycle.stage = stage
            self.assertEqual(cadence.select(cycle, None), .1)
        cycle.active = False
        pickup = SimpleNamespace(active=True, stage='hover', motion=SimpleNamespace())
        self.assertEqual(cadence.select(cycle, pickup), .05)
        for stage in ('descend', 'local_grasp', 'carry', 'lift', 'local_weighing', 'observe_pad'):
            pickup.stage = stage
            self.assertEqual(cadence.select(cycle, pickup), .1)
        self.assertEqual(cadence.select(None, None), .1)
        cadence.close()
        self.assertIs(env.rate, original)

    def test_invalid_frequency_and_direct_wbc_rejected(self):
        for hz in (0, 9, 21, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                MotionCadence(SimpleNamespace(ros_rate=10, direct_to_wbc=False), None, hz)
        with self.assertRaises(ValueError):
            MotionCadence(SimpleNamespace(ros_rate=10, direct_to_wbc=True), None, 20)

    def test_stage_override_only_narrows_permitted_movements(self):
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=None)
        with self.assertRaises(ValueError):
            MotionCadence(env, lambda hz: hz, 20, ['release_left'])
        cadence = MotionCadence(env, lambda hz: hz, 20, ['return_home'])
        cycle = SimpleNamespace(active=True, stage='bin_transport')
        self.assertEqual(cadence.select(cycle, None), .1)
        cycle.stage = 'return_home'
        self.assertEqual(cadence.select(cycle, None), .05)

    def test_receiver_approach_restores_original_cadence_before_visual_contact_region(self):
        env = SimpleNamespace(ros_rate=10, direct_to_wbc=False, rate=None)
        cadence = MotionCadence(env, lambda hz: hz, 20, ['lift', 'left_approach'], {'lift':15})
        cycle = profiles.SpeedProfileTests().cycle({})
        cycle.active = True
        cycle._stage('lift')
        self.assertAlmostEqual(cadence.select(cycle, None), 1./15)
        cycle._stage('left_approach');cycle.goal = [np.eye(4), None]
        self.assertEqual(cadence.select(cycle, None), .1)
        measured = np.zeros(16);measured[0] = .10
        self.assertEqual(cadence.select(cycle, None, measured), .05)
        measured[0] = .079
        self.assertEqual(cadence.select(cycle, None, measured), .1)
        for rates in ({'lift': 21}, {'lift': 9}, {'release_left': 15}):
            with self.assertRaises(ValueError):
                MotionCadence(env, lambda hz: hz, 20, ['lift'], rates)

    def cycle(self):
        cycle = profiles.SpeedProfileTests().cycle({})
        cycle._stage('transport')
        cycle.goal = [np.eye(4), None]
        cycle.control_period_s = .05
        return cycle

    def test_fast_arrival_requires_original_elapsed_dwell(self):
        cycle = self.cycle()
        for now, expected in ((0., False), (.05, False), (.10, False), (.15, False), (.20, False), (.26, True)):
            with patch('scene1.visual_cycle.time.monotonic', return_value=now):
                arrived, _ = cycle._arrived(np.zeros(16))
            self.assertEqual(arrived, expected)

    def test_fast_arrival_does_not_relax_actual_movement_speed(self):
        cycle = self.cycle()
        measured = np.zeros(16)
        with patch('scene1.visual_cycle.time.monotonic', return_value=0.):
            cycle._arrived(measured)
        # 3 mm in 50 ms exceeds the original 4 mm / 100 ms limit.
        measured[0] = .003
        with patch('scene1.visual_cycle.time.monotonic', return_value=.05):
            cycle._arrived(measured)
        self.assertEqual(cycle.stable, 0)

    def test_fast_home_keeps_four_tenths_second_dwell(self):
        cycle = profiles.SpeedProfileTests().home_cycle(2.)
        cycle.control_period_s = .05
        for now in (0., .05, .10, .15, .20, .25, .30, .35):
            with patch('scene1.visual_cycle.time.monotonic', return_value=now):
                cycle.update(np.zeros(16), {'stamp': now}, None)
            self.assertFalse(cycle.done)
        with patch('scene1.visual_cycle.time.monotonic', return_value=.41):
            cycle.update(np.zeros(16), {'stamp': .41}, None)
        self.assertTrue(cycle.done)
