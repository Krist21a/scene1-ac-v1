"""Replay the unmodified local retreat executor; no ROS or robot commands.

Perfect tracking and a fixed first-order response are sensitivity models only.
They are not dynamic validation or permission to relax a collision condition.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--urdf', type=Path, required=True)
    ap.add_argument('--fixtures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--offset-mode', choices=['original_act', 'physical_only', 'measured_phase'], default='original_act')
    ap.add_argument('--period', type=float, default=.1)
    a = ap.parse_args()
    sys.path.insert(0, str(a.policy))
    from scene1.kinematics import UrdfArmKinematics
    from scene1.residual import ResidualAssist
    from scene1.motion import MotionFilter
    from scene1.command_sweep import sampled_sweep
    from scene1.clearance import open_fingers_clear
    config = yaml.safe_load((a.policy / 'configs/visual_cycle_four_parcels.yaml').read_text())
    k = UrdfArmKinematics(description=a.urdf.read_text())
    low = np.r_[np.full(7, -3.14), 0., np.full(7, -3.14), 0.]
    high = np.r_[np.full(7, 3.14), 1., np.full(7, 3.14), 1.]
    output = []
    for fixture in json.loads(a.fixtures.read_text()):
        last, failed = fixture['last_sent_row'], fixture['failed']
        audit = failed['retreat_preflight_audit']
        goal = np.array(audit['goal_base_m'])
        start_pose = k.matrix(np.array(failed['observation_state'])[8:15], 1)
        start_rotation = Rotation.from_matrix(start_pose[:3, :3])
        target = audit['observed_parcel']
        plane = last['rgbd_table_z_base_m']
        for response in [1., .25, .1]:
            assist = ResidualAssist(config['visual_residual'], k, low, high)
            assist.wrist_config = dict(config['wrist_assist'])
            # _correct only tests whether this field is present, not its value.
            assist.weighing_target = np.array(target['position_base_m'])
            base = np.array(last['proposed_action'])
            base[8:15] = audit['original_reference_rad']
            assist.offset = np.array(last['proposed_action']) - base
            if a.offset_mode == 'measured_phase':
                base[8:15] = np.array(failed['observation_state'])[8:15]
                assist.offset[8:15] = 0.
            if a.offset_mode != 'original_act':
                # Offline emulation of nonbinding ACT offset constraints, while
                # the unmodified executor still enforces the exact physical box.
                assist.wrist_config['max_regrasp_joint_offset_rad'] = float(max(
                    np.max(abs(assist.low[8:15] - base[8:15])),
                    np.max(abs(assist.high[8:15] - base[8:15]))) + 1e-6)
            assist.unwinding_placement = True
            motion = MotionFilter(assist.low, assist.high, config['motion']['smoothing_alpha'],
                                  config['cartesian_alignment']['approach_joint_velocity_rad_s'])
            motion.reset(last['sent_action'])
            measured = np.array(failed['observation_state'])
            previous_position = None
            stable = 0
            rows = []
            success = False
            error_message = None
            initial_phase_step = last['cartesian_alignment']['phase_steps']
            for phase_step in range(initial_phase_step + 1, config['wrist_assist']['max_short_retreat_steps'] + 1):
                pose = k.matrix(measured[8:15], 1)
                position = pose[:3, 3]
                error = goal - position
                velocity = np.zeros(3) if previous_position is None else position - previous_position
                movement = np.inf if previous_position is None else np.linalg.norm(velocity)
                try:
                    proposed = assist._correct(base, measured, 1, error - 1.5 * velocity, step_scale=.5)
                    proposed[15] = 0.
                    sent, info = motion.apply(proposed, a.period)
                except Exception as exc:
                    error_message = repr(exc)
                    break
                sweep = sampled_sweep(k, measured[8:15], sent[8:15], target, plane)
                proposed_sweep = sampled_sweep(k, measured[8:15], proposed[8:15], target, plane)
                angular = float(np.rad2deg((start_rotation.inv() * Rotation.from_matrix(pose[:3, :3])).magnitude()))
                stable = stable + 1 if np.linalg.norm(error) <= config['wrist_assist']['release_retreat_ready_m'] and movement <= .004 else 0
                rows.append(dict(phase_step=phase_step, position_error_m=float(np.linalg.norm(error)),
                                 cumulative_rotation_deg=angular, measured_joint_rad=measured[8:15].tolist(),
                                 sent_joint_rad=sent[8:15].tolist(), offset_rad=assist.offset[8:15].tolist(),
                                 measured_clear=open_fingers_clear(pose, target, plane),
                                 filtered_sweep=sweep, proposed_sweep=proposed_sweep,
                                 motion=info, stable=stable))
                previous_position = position.copy()
                measured[:7] += response * (sent[:7] - measured[:7])
                measured[8:15] += response * (sent[8:15] - measured[8:15])
                measured[15] = 0.
                if stable >= 3:
                    success = True
                    break
            output.append(dict(label=fixture['label'], feedback_response_fraction=response,
                               offset_mode=a.offset_mode, filter_dt_s=a.period,
                               initial_phase_step=initial_phase_step,
                               original_max_phase_steps=config['wrist_assist']['max_short_retreat_steps'],
                               reached_original_position_and_dwell=success, error=error_message,
                               steps=len(rows), final_position_error_m=rows[-1]['position_error_m'] if rows else None,
                               max_rotation_deg=max((r['cumulative_rotation_deg'] for r in rows), default=None),
                               blocked_filtered_sweeps=sum(not r['filtered_sweep']['clear'] for r in rows),
                               blocked_proposed_sweeps=sum(not r['proposed_sweep']['clear'] for r in rows),
                               target_clips=sum(r['motion']['bounds_clipped'] for r in rows), rows=rows))
    report = dict(cases=output, commands_sent=False,
                  limitations=['Kinematic executor replay, not a dynamics/contact simulation',
                               'Frozen observed parcel geometry; no new perception or scene truth',
                               'Feedback response fractions are sensitivity assumptions, not identified plant models',
                               'Diagnostic only: bypasses the candidate preflight offline without changing the live policy'])
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'rows'} for r in output], indent=2))


if __name__ == '__main__':
    main()
