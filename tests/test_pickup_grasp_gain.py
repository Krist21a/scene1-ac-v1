"""Actual v65 grasp feedback and the unchanged original closure/timeout gates."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup


class PickupGraspGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent/'fixtures/pickup_close_v65.json').read_text())
        cls.k = UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def setup_case(self, case, scale=None, stage='local_grasp'):
        assist = ResidualAssist(case['config'].copy(), self.k, np.full(16, -10.), np.full(16, 10.))
        assist.wrist_config = case['wrist_config'].copy()
        base = np.array(case['base'])
        assist._begin('grasp', 1, base, target=copy.deepcopy(case['target']),
            plane=case['plane'], grasp_z=case['goal'][2], wrist_verified=False,
            grasp_rotation=Rotation.from_matrix(case['rotation']))
        config = dict(pickup_grasp_z_tolerance_m=.005)
        if scale is not None:
            config['pickup_grasp_gain_scale'] = scale
        motion = VisualCycle(config, self.k, np.full(16, -10.), np.full(16, 10.), base)
        pickup = VisualPickup(motion, {}, [0., 0., 0.], base, base)
        pickup._stage(stage)
        return pickup, assist

    def update(self, pickup, assist, frame):
        # Replay the original pending solver on recorded feedback. This checks
        # implementation/gates, not the physical response to changed commands.
        with patch.object(assist, '_remember'), patch.object(assist, 'apply',
                side_effect=lambda _, measured, snapshot: assist._pending_action(measured, None)):
            return pickup.update(np.array(frame['measured']), dict(stamp=frame['step']), assist)

    def test_unscaled_reproduces_seven_real_sequences_and_original_failure(self):
        for case in self.fixture['cases']:
            for scale in (None, 1.):
                pickup, assist = self.setup_case(case, scale)
                for frame in case['frames']:
                    self.update(pickup, assist, frame)
                    np.testing.assert_allclose(pickup.motion.command, frame['sent'], rtol=0., atol=2e-10)
                if case['failure']:
                    failed = case['failure']
                    with self.assertRaisesRegex(RuntimeError, 'allowed small correction'):
                        self.update(pickup, assist, dict(step=failed['step'], measured=failed['observation_state']))
                    self.assertEqual(assist.pending['steps'], 45)

    def test_reduced_gain_preserves_actual_closure_and_timeout_conditions(self):
        for case in self.fixture['cases']:
            pickup, assist = self.setup_case(case, .8)
            original = assist.config
            for frame in case['frames']:
                self.update(pickup, assist, frame)
                self.assertEqual(pickup.motion.command[15], frame['sent'][15])
                self.assertIs(assist.config, original)
            if case['failure']:
                failed = case['failure']
                with self.assertRaisesRegex(RuntimeError, 'allowed small correction'):
                    self.update(pickup, assist, dict(step=failed['step'], measured=failed['observation_state']))
                self.assertEqual(assist.pending['steps'], 45)
                self.assertEqual(pickup.motion.command[15], 0.)
                self.assertIs(assist.config, original)

    def test_scope_only_later_local_grasp_and_restore_on_exception(self):
        case = self.fixture['cases'][0]
        for stage in ('local_grasp', 'local_weighing'):
            for fail in (False, True):
                pickup, assist = self.setup_case(case, .8, stage)
                original = assist.config
                before = copy.deepcopy(assist.pending)
                def apply(*args):
                    expected = dict(original, gain=original['gain']*.8) if stage == 'local_grasp' else original
                    self.assertEqual(assist.config, expected)
                    if fail:
                        raise RuntimeError('pending solver failure')
                    return np.array(case['base']), dict(state='small_grasp')
                with patch.object(assist, '_remember'), patch.object(assist, 'apply', side_effect=apply):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'pending solver failure'):
                            pickup.update(np.array(case['base']), dict(stamp=1.), assist)
                    else:
                        pickup.update(np.array(case['base']), dict(stamp=1.), assist)
                self.assertIs(assist.config, original)
                self.assertEqual(assist.pending['steps'], before['steps'])
                np.testing.assert_array_equal(assist.pending['base'], before['base'])

    def test_invalid_or_increased_gain_is_rejected_before_control(self):
        for scale in (0., -.1, 1.0001, float('nan'), float('inf')):
            pickup, assist = self.setup_case(self.fixture['cases'][0], scale)
            original = assist.config
            with patch.object(assist, '_remember'), patch.object(assist, 'apply') as apply:
                with self.assertRaisesRegex(ValueError, 'grasp gain scale'):
                    pickup.update(np.array(self.fixture['cases'][0]['base']), dict(stamp=1.), assist)
                apply.assert_not_called()
            self.assertIs(assist.config, original)


if __name__ == '__main__':
    unittest.main()
