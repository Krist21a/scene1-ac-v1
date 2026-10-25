"""Real failed retreat feedback: change the solve, retain arrival and budgets."""
import json
from pathlib import Path
import unittest

import numpy as np

from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle


class HoldingPosePriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent/'fixtures/holding_servo_v68.json').read_text())
        cls.k = UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def cycle(self, case, enabled=False, stage='right_retreat'):
        config = dict(self.fixture['config'], holding_pose_priority=enabled)
        a = VisualCycle(config, self.k, np.full(16, -10.), np.full(16, 10.), case['previous_command'])
        a._stage(stage)
        a.command = np.array(case['previous_command'])
        a.goal = [np.array(g) for g in case['goal']]
        a.servo_precision = [True, True]
        a.servo_previous = [self.k.matrix(np.array(case['previous_measured'])[start:start+7], side)
                            for side, start in enumerate((0, 8))]
        a.total_steps = 0
        return a

    def test_disabled_reproduces_recorded_two_arm_commands(self):
        for case in self.fixture['cases']:
            a = self.cycle(case)
            a._arrived(np.array(case['measured']))
            np.testing.assert_array_equal(a.command, case['sent_command'])

    def test_only_loaded_left_retreat_solve_changes(self):
        case = self.fixture['cases'][3]
        for stage in ('lift', 'transport', 'left_approach', 'left_insert', 'release_right',
                      'right_retreat', 'left_clear', 'bin_transport', 'left_retreat'):
            a = self.cycle(case, True, stage)
            for side in (0, 1):
                a._servo(np.array(case['measured']), side, a.goal[side])
            np.testing.assert_array_equal(a.command[8:], case['sent_command'][8:])
            if stage != 'right_retreat':
                np.testing.assert_array_equal(a.command, case['sent_command'])
            else:
                self.assertGreater(np.max(abs(a.command[:7]-case['sent_command'][:7])), 1e-6)

    def test_recorded_unsafe_orientation_does_not_pass_with_priority(self):
        case = self.fixture['cases'][-1]
        for enabled in (False, True):
            a = self.cycle(case, enabled)
            for _ in range(4):
                arrived, errors = a._arrived(np.array(case['measured']))
                self.assertFalse(arrived)
                self.assertGreater(errors[0]['angle_rad'], .14)
                self.assertLess(errors[0]['position_m'], .012)

    def model(self, step, enabled):
        case = next(c for c in self.fixture['cases'] if c['step'] == step)
        a = self.cycle(case, enabled)
        a.command = np.array(case['sent_command'])
        a.goal[1] = None  # Isolate the held hand; no claim about robot dynamics.
        a.servo_previous[0] = self.k.matrix(np.array(case['measured'])[:7], 0)
        bias = np.array(case['after_measured'])-a.command
        goal = a.goal[0].copy()
        arrived = False
        for _ in range(500-case['phase_steps']):
            before = a.command.copy()
            arrived, errors = a._arrived(a.command+bias)
            self.assertLessEqual(max(abs(a.command-before)), .006+1e-12)
            self.assertTrue(np.all(a.command[:7] >= a.low[:7]))
            self.assertTrue(np.all(a.command[:7] <= a.high[:7]))
            np.testing.assert_array_equal(a.command[7:], before[7:])
            np.testing.assert_array_equal(a.goal[0], goal)
            if arrived:
                self.assertLess(errors[0]['position_m'], .012)
                self.assertLess(errors[0]['angle_rad'], .14)
                break
        return arrived

    def test_fixed_bias_late_states_converge_with_original_remaining_budget(self):
        for step in (12742, 12842):
            self.assertFalse(self.model(step, False))
            self.assertTrue(self.model(step, True))

    def test_model_still_refuses_nonconvergent_and_exhausted_states(self):
        self.assertFalse(self.model(12642, True))
        self.assertFalse(self.model(12892, True))

    def test_original_timeout_keeps_last_gripper_command(self):
        case = self.fixture['cases'][-1]
        a = self.cycle(case, True)
        a.command = np.array(case['sent_command'])
        a.stage_steps = 500
        before = a.command.copy()
        with self.assertRaisesRegex(RuntimeError, 'timed out at right_retreat'):
            a.update(case['measured'], None, None)
        np.testing.assert_array_equal(a.command, before)


if __name__ == '__main__':
    unittest.main()
