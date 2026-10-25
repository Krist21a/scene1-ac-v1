"""Exercise the delivered certificate on retained release snapshots."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--urdf', type=Path, required=True)
a = ap.parse_args()
policy = a.root / 'candidate-policy-executor'
sys.path.insert(0, str(policy))
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.release_retreat import axis_retreat_goal, RetreatPreflightError
from scene1.retreat_certificate import certify_retreat_execution
config = yaml.safe_load((policy / 'configs/visual_cycle_four_parcels.yaml').read_text())
k = UrdfArmKinematics(description=a.urdf.read_text())
low = np.r_[np.full(7, -3.14), 0., np.full(7, -3.14), 0.]
high = np.r_[np.full(7, 3.14), 1., np.full(7, 3.14), 1.]
output = []
for fixture in json.loads((a.root / 'executor-regression-inputs.json').read_text()):
    last, failed = fixture['last_sent_row'], fixture['failed']
    prior = failed['retreat_preflight_audit']
    assist = ResidualAssist(config['visual_residual'], k, low, high)
    assist.wrist_config = config['wrist_assist']
    assist.weighing_target = np.asarray(prior['observed_parcel']['position_base_m'])
    assist.observe_command(last['sent_action'])
    base = np.array(last['proposed_action']); base[8:15] = prior['original_reference_rad']
    assist.offset = np.array(last['proposed_action']) - base
    assist.unwinding_placement = bool(np.max(abs(assist.offset[8:15])) > config['visual_residual']['max_joint_offset_rad'])
    measured = np.asarray(failed['observation_state'])
    previous_offset = assist.offset.copy()
    motion_config = dict(alpha=config['motion']['smoothing_alpha'],
                        max_velocity=config['cartesian_alignment']['approach_joint_velocity_rad_s'], period=.1)
    def validator(goal, parcel):
        return certify_retreat_execution(assist, base, measured, goal, parcel,
            last['rgbd_table_z_base_m'], config['wrist_assist']['max_short_retreat_steps']-
            last['cartesian_alignment']['phase_steps'], motion_config)
    start = time.perf_counter()
    try:
        goal, audit = axis_retreat_goal(k, measured, base, assist.low, assist.high, assist.offset,
            config['wrist_assist']['max_regrasp_joint_offset_rad'], assist.unwinding_placement,
            dict(candidates=[prior['observed_parcel']], stamp=prior['observation_stamp']),
            assist.weighing_target[:2], [.11, .04, .1], .06, .07, .014, execution_validator=validator)
        assert np.allclose(goal, prior['goal_base_m'], atol=1e-10)
    except RetreatPreflightError as error:
        audit = error.retreat_preflight_audit
    np.testing.assert_array_equal(assist.offset, previous_offset)
    expected = 'original' not in fixture['label']
    assert audit['certified'] == expected, fixture['label']
    output.append(dict(label=fixture['label'], expected_accepted=expected,
                       preflight_wall_s=time.perf_counter()-start, audit=audit))
(a.root / 'certificate-replay.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps([dict(label=x['label'], accepted=x['audit']['certified'],
                      nominal_steps=x['audit']['nominal_steps'],
                      reason=x['audit']['reason'], preflight_wall_s=x['preflight_wall_s']) for x in output], indent=2))
