"""Saved failed preparation states: preserve task gates and bounded commands."""
import json
from pathlib import Path
import unittest
import numpy as np

from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle


class ReserveCommandBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent/'fixtures/weighing_reserve_servo_v66.json').read_text())
        cls.k = UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def cycle(self, case):
        a = VisualCycle(self.fixture['config'], self.k, np.full(16, -10.), np.full(16, 10.), case['previous_command'])
        a._stage('prepare_weighing')
        a.command = np.array(case['previous_command'])
        a.goal = [None, np.array(case['goal'])]
        a.servo_precision[1] = True
        if case['previous_measured'] is not None:
            a.servo_previous[1] = self.k.matrix(np.array(case['previous_measured'])[8:15], 1)
        return a

    def test_disabled_reproduces_nine_actual_commands_exactly(self):
        for case in self.fixture['cases']:
            a = self.cycle(case)
            a._servo(np.array(case['measured']), 1, a.goal[1], np.array(case['reference']),
                     posture_gain=.3, coupled_pose_request=True)
            np.testing.assert_array_equal(a.command, case['sent_command'])

    def test_saved_states_approach_existing_band_without_larger_steps(self):
        for case in self.fixture['cases']:
            a = self.cycle(case)
            reference = np.array(case['reference'])
            original_reference = reference.copy()
            old = a.command.copy()
            a._servo(np.array(case['measured']), 1, a.goal[1], reference,
                     posture_gain=.3, coupled_pose_request=True, reserve_joint_bound=True)
            self.assertLessEqual(max(abs(a.command-old)), .006+1e-12)
            self.assertLessEqual(abs(a.command[12]-reference[4]),
                                max(abs(old[12]-reference[4]), np.deg2rad(2.5))+1e-12)
            self.assertTrue(np.all(a.command[8:15] >= a.low[8:15]))
            self.assertTrue(np.all(a.command[8:15] <= a.high[8:15]))
            self.assertEqual(a.command[15], old[15])
            np.testing.assert_array_equal(reference, original_reference)
            np.testing.assert_array_equal(a.goal[1], case['goal'])

    def test_fixed_bias_case_can_reach_original_pose_and_reference_within_remaining_budget(self):
        case = next(c for c in self.fixture['cases'] if c['step'] == 7250)
        for enabled in (False, True):
            a = self.cycle(case)
            bias = np.array(case['measured'])-case['previous_command']
            ready = False
            for _ in range(case['remaining_prepare_steps']):
                measured = a.command+bias
                arrived, errors = a._arrived(measured, position_limit=.008, angle_limit=np.deg2rad(10),
                    posture_targets=[None, np.array(case['reference'])], posture_gain=.3,
                    coupled_pose_request=True, reserve_joint_bound=enabled)
                ready = arrived and abs(a.command[12]-case['reference'][4]) <= np.deg2rad(3)
                if ready:
                    break
            self.assertEqual(bool(ready), enabled)
            if enabled:
                self.assertLess(errors[0]['position_m'], .008)
                self.assertLess(errors[0]['angle_rad'], np.deg2rad(10))

    def test_joint_band_alone_cannot_bypass_actual_pose_arrival(self):
        case = self.fixture['cases'][0]
        a = self.cycle(case)
        a.command[12] = case['reference'][4]
        a.goal[1][0, 3] += .2
        for _ in range(4):
            arrived, errors = a._arrived(np.array(case['measured']), position_limit=.008,
                angle_limit=np.deg2rad(10), posture_targets=[None, np.array(case['reference'])],
                posture_gain=.3, coupled_pose_request=True, reserve_joint_bound=True)
            self.assertFalse(arrived)
            self.assertGreater(errors[0]['position_m'], .008)

    def test_missing_invalid_or_wrong_arm_reference_rejected(self):
        case = self.fixture['cases'][0]
        for reference, side in ((None, 1), (np.zeros(6), 1), (np.full(7, np.nan), 1),
                                (np.full(7, 10.), 1), (np.array(case['reference']), 0)):
            a = self.cycle(case)
            before = a.command.copy()
            with self.assertRaises(ValueError):
                a._servo(np.array(case['measured']), side, a.goal[1], reference, reserve_joint_bound=True)
            np.testing.assert_array_equal(a.command, before)


if __name__ == '__main__':
    unittest.main()
