import json
from pathlib import Path
import unittest
import numpy as np
import test_release_preparation as preparation

class ReleaseQuietTests(unittest.TestCase):
    setUp=preparation.PreparationTests.setUp
    fixture=preparation.PreparationTests.fixture

    def recorded(self,limit):
        _,accepted=self.fixture()
        p=Path(__file__).parent
        self.f=json.loads((p/'fixtures/release_moving_20261024.json').read_text())
        target=np.array(self.f['rows'][-1]['sent_action'])
        self.a.wrist_config.update(release_prepare_whole_arm=True,
            release_prepare_motion_limit_rad=limit)
        self.a.pending.update(stage='correct',release_prepare_target=target.copy(),
            release_prepare_previous_wrist=np.array(self.f['previous'])[8:15],
            release_prepare_stable=0,steps=100)
        self.a.observe_command(target)
        return target

    def replay(self,limit):
        target=self.recorded(limit)
        for r in self.f['rows']:
            self.a.pending['steps']+=1
            self.measured=np.array(r['observation_state'])
            info={};result=self.a._prepare_release_reference(self.measured,True,target,info)
            self.a.observe_command(result)
        return result,info

    def test_recorded_moving_hand_passes_old_gate_but_cannot_open_under_new_gate(self):
        self.replay(.003)
        self.assertTrue(self.a.pending.get('reference_prepared'))
        self.setUp();result,info=self.replay(.0003)
        self.assertFalse(self.a.pending.get('reference_prepared',False))
        self.assertEqual(result[15],.98)
        self.assertGreater(info['release_preparation']['settled_movement_rad'],.0003)
        self.assertEqual(self.a.pending['steps'],103)

    def test_same_recorded_target_can_release_after_feedback_actually_stops(self):
        target,info=self.replay(.0003)
        for _ in range(3):
            self.a.pending['steps']+=1
            result=self.a._prepare_release_reference(self.measured,True,target,{})
            self.a.observe_command(result)
        self.assertTrue(self.a.pending['reference_prepared'])
        np.testing.assert_array_equal(result[8:15],target[8:15])
        self.assertEqual(result[15],.98)

    def test_motion_does_not_extend_original_release_budget(self):
        target=self.recorded(.0003)
        self.a.pending['steps']=self.a.config['max_release_steps']
        self.measured=np.array(self.f['rows'][0]['observation_state'])
        with self.assertRaisesRegex(RuntimeError,'original weighing budget'):
            self.a._prepare_release_reference(self.measured,True,target,{})
