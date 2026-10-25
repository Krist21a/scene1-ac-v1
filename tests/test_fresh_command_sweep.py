import json
from pathlib import Path
import numpy as np
import unittest
from scene1.command_sweep import command_sweep_audit, fresh_command_sweep_audit
from scene1.kinematics import UrdfArmKinematics

FIXTURES = Path(__file__).parent/'fixtures'


def setup_case():
    x = json.loads((FIXTURES/'stale_retreat_20261023.json').read_text())
    k = UrdfArmKinematics(description=(FIXTURES/'release_robot.urdf').read_text())
    context = dict(target=x['target'], plane=x['plane'], stage='short_retreat')
    return x, k, context


def real_old_state_passes_but_send_time_feedback_blocks():
    x, k, context = setup_case()
    old = command_sweep_audit(k,x['planning_observation'],x['proposed'],x['filtered'],x['previous'],context)
    assert old['filtered']['clear']
    calls = []
    def refresh():
        calls.append(True)
        return x['latest_feedback']
    fresh = fresh_command_sweep_audit(k,refresh,x['planning_observation'],x['proposed'],x['filtered'],x['previous'],context)
    assert calls == [True]
    assert not fresh['filtered']['clear']
    assert fresh['filtered']['first_blocked_fraction'] == 0.
    assert fresh['margin_m'] == old['margin_m'] == .003
    assert fresh['feedback']['max_joint_change_from_planning_rad'] > .06
    assert fresh['feedback']['state'] == x['latest_feedback']


def same_fresh_state_preserves_original_decision_and_command():
    x, k, context = setup_case()
    sent = np.array(x['filtered'])
    before = sent.copy()
    fresh = fresh_command_sweep_audit(k,lambda:x['planning_observation'],x['planning_observation'],x['proposed'],sent,x['previous'],context)
    assert fresh['filtered']['clear']
    assert fresh['feedback']['max_joint_change_from_planning_rad'] == 0.
    np.testing.assert_array_equal(sent,before)


class FreshSweepTest(unittest.TestCase):
    def test_real_stale_feedback_regression(self):
        real_old_state_passes_but_send_time_feedback_blocks()

    def test_unchanged_fresh_feedback(self):
        same_fresh_state_preserves_original_decision_and_command()

    def check_invalid(self, invalid):
        x, k, context = setup_case()
        with self.assertRaisesRegex(ValueError, 'finite complete'):
            fresh_command_sweep_audit(k,lambda:invalid,x['planning_observation'],x['proposed'],x['filtered'],x['previous'],context)

    def test_incomplete_feedback(self):
        self.check_invalid([0.]*15)

    def test_nonfinite_feedback(self):
        self.check_invalid([float('nan')]*16)
