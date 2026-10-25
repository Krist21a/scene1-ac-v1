from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scene1.cadence import MotionCadence
import test_speed_profiles as profiles


class MotionCadenceTests(unittest.TestCase):
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
