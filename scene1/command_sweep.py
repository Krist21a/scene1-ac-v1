"""Audit pending and filtered references against the unchanged parcel envelope."""
import numpy as np
import time
from .clearance import open_fingers_clear


def sampled_sweep(kinematics, q, target_q, target, plane):
    q, target_q = np.asarray(q, float), np.asarray(target_q, float)
    if (q.shape != (7,) or target_q.shape != (7,)
            or not np.isfinite(np.r_[q, target_q]).all()):
        raise ValueError('Sweep requires finite seven-joint vectors')
    # Include the original measured/midpoint/endpoint tests and subdivide long
    # pending references. Sampling does not prove a continuous swept volume.
    intervals = max(2, int(np.ceil(np.max(np.abs(target_q-q)) / .012)))
    intervals += intervals % 2
    for i in range(intervals + 1):
        fraction = i / intervals
        if not open_fingers_clear(kinematics.matrix(q+fraction*(target_q-q), 1), target, plane):
            return dict(clear=False, first_blocked_fraction=fraction, samples=intervals+1)
    return dict(clear=True, first_blocked_fraction=None, samples=intervals+1)


def command_sweep_audit(kinematics, measured, proposed, sent, previous, context):
    q = np.asarray(measured)[8:15]
    target, plane = context['target'], context['plane']
    output = dict(stage=context['stage'], target=target, plane_m=float(plane),
                  margin_m=.003, interpretation='Sampled open inner fingers only; no whole robot or dynamics proof')
    for key, action in [('proposed', proposed), ('filtered', sent), ('previous', previous)]:
        joints = np.asarray(action)[8:15]
        output[key] = sampled_sweep(kinematics, q, joints, target, plane)
        output[key]['max_pending_reference_rad'] = float(np.max(np.abs(joints-q)))
    if context.get('increment_end_rad') is not None:
        endpoint = np.asarray(context['increment_end_rad'])
        output['legacy_increment'] = dict(clear=all(open_fingers_clear(
            kinematics.matrix(q+a*(endpoint-q), 1), target, plane) for a in (0., .5, 1.)))
    return output


def fresh_command_sweep_audit(kinematics, read_feedback, planning_measured,
                              proposed, sent, previous, context):
    """Refresh SDK feedback after planning, immediately before the final guard.

    A recent background ROS callback does not refresh an observation already
    copied by the planner. Retain the distinct planning and guard states.
    """
    capture_started = time.monotonic()
    measured = np.array(read_feedback(), dtype=float, copy=True).reshape(-1)
    captured = time.monotonic()
    if measured.shape != (16,) or not np.isfinite(measured).all():
        raise ValueError('Fresh sweep requires finite complete joint/gripper feedback')
    output = command_sweep_audit(kinematics, measured, proposed, sent, previous, context)
    output['feedback'] = dict(source='fresh_sdk_get_obs_after_planning',
        state=measured.tolist(), capture_started_monotonic_s=capture_started,
        captured_monotonic_s=captured, checked_monotonic_s=time.monotonic(),
        max_joint_change_from_planning_rad=float(np.max(np.abs(
            measured[8:15]-np.asarray(planning_measured)[8:15]))))
    return output


def retreat_pending_hold_allowed(audit):
    """Only a fresh, fully clear previous sweep can replace a blocked increment."""
    return (audit.get('stage') == 'short_retreat'
            and audit.get('feedback', {}).get('source') == 'fresh_sdk_get_obs_after_planning'
            and not audit['filtered']['clear'] and audit['previous']['clear'])
