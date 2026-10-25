"""Nominal executable retreat certificate using the actual local corrector.

This is a kinematic preflight, not a prediction of contact dynamics. Live
filtered-command sweeps and the existing execution timeout remain mandatory.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from .command_sweep import sampled_sweep
from .motion import MotionFilter


def certify_retreat_execution(assist, base, measured, goal, target, plane,
                              remaining_steps, motion_config):
    shadow = copy.copy(assist)
    shadow.offset = assist.offset.copy()
    measured = np.asarray(measured, float).copy()
    base = np.asarray(base, float).copy()
    goal = np.asarray(goal, float)
    if remaining_steps < 1 or motion_config is None or assist.last_sent is None:
        raise RuntimeError('Release retreat certificate lacks remaining budget or filter state')
    motion = MotionFilter(assist.low, assist.high, motion_config['alpha'], motion_config['max_velocity'])
    motion.reset(assist.last_sent)
    period = motion_config['period']
    tolerance = assist.wrist_config.get('release_retreat_ready_m', .014)
    start_rotation = Rotation.from_matrix(assist.kinematics.matrix(measured[8:15], 1)[:3, :3])
    previous_position = None
    stable = 0
    records = []
    reason = 'nominal executor did not settle within remaining original steps'
    certified = False
    max_rotation = 0.
    for index in range(remaining_steps):
        pose = assist.kinematics.matrix(measured[8:15], 1)
        position = pose[:3, 3]
        error = goal - position
        velocity = np.zeros(3) if previous_position is None else position - previous_position
        movement = np.inf if previous_position is None else float(np.linalg.norm(velocity))
        proposed, projection = shadow._retreat_correct(base, measured, error - 1.5 * velocity,
                                                       target, plane, motion_config)
        proposed[15] = 0.
        sent, motion_info = motion.apply(proposed, period)
        if projection is not None:
            # _retreat_correct already checked these exact endpoints using
            # the same measured state and filter reset to shadow.last_sent.
            checked = projection['final' if projection['applied'] else 'original']
            complete, filtered = checked['proposed'], checked['filtered']
        else:
            complete = sampled_sweep(assist.kinematics, measured[8:15], proposed[8:15], target, plane)
            filtered = sampled_sweep(assist.kinematics, measured[8:15], sent[8:15], target, plane)
        rotation = float(np.rad2deg((start_rotation.inv() * Rotation.from_matrix(pose[:3, :3])).magnitude()))
        max_rotation = max(max_rotation, rotation)
        stable = stable + 1 if np.linalg.norm(error) <= tolerance and movement <= .004 else 0
        records.append(dict(index=index, error_m=float(np.linalg.norm(error)),
                            cumulative_rotation_degrees=rotation, proposed_sweep=complete,
                            filtered_sweep=filtered, motion=motion_info, stable_steps=stable,
                            projection=projection if projection and projection['applied'] else None))
        if motion_info['bounds_clipped'] or not complete['clear'] or not filtered['clear']:
            reason = 'nominal executor sweep or physical command bounds rejected'
            break
        previous_position = position.copy()
        measured = sent.copy()
        shadow.observe_command(sent)
        if stable >= 3:
            certified = True
            reason = 'nominal bounded executor reached original position and dwell criteria'
            break
    # Certify the final command explicitly, rather than assuming its FK equals
    # the pose evaluated before the last command.
    final = assist.kinematics.matrix(measured[8:15], 1)
    residual = float(np.linalg.norm(final[:3, 3] - goal))
    certified = certified and residual <= tolerance
    return dict(certified=bool(certified), reason=reason,
                feasibility_joint_rad=measured[8:15].tolist(), position_residual_m=residual,
                orientation_residual_degrees=float(np.rad2deg((start_rotation.inv() *
                    Rotation.from_matrix(final[:3, :3])).magnitude())),
                orientation_tolerance_degrees=None,
                orientation_criterion='unchanged per-step executor plus full sampled finger sweeps; no added cumulative angle gate',
                solver_method='nominal_incremental_executor_sweep', remaining_steps=remaining_steps,
                nominal_steps=len(records), max_cumulative_rotation_degrees=max_rotation,
                nominal_filter=dict(motion_config), nominal_records=records,
                limitation='Perfect-tracking kinematic preflight only; live sweep and original timeout remain active')
