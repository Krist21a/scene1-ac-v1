"""Release retreat preflight using only current RGB-D and measured robot FK."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


class RetreatPreflightError(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.retreat_preflight_audit = audit


def axis_retreat_goal(kinematics, measured, reference, low, high, offset,
                      cap, unwinding, snapshot, pad_xy, extent, distance,
                      rise, tolerance, execution_validator=None):
    """Reject ambiguity or an uncertified goal; never change limits.

    The bounded solves are feasibility checks, not a replacement command
    or a new correction reference. The existing per-step controller executes
    the accepted Cartesian goal with its original limits and time budget.
    """
    q = np.asarray(measured, float)[8:15]
    pose = kinematics.matrix(q, 1)
    axis = pose[:2, 2].copy()
    projection = float(np.linalg.norm(axis))
    if not np.isfinite(pose).all() or projection < 1e-3:
        raise RuntimeError('Release retreat rejected: degenerate measured tool-axis projection')
    direction = axis / projection
    if not all(np.isfinite(x) and x > 0 for x in (distance, rise, tolerance, cap)):
        raise ValueError('Release retreat distances, tolerance and cap must be positive and finite')
    candidates = [] if snapshot is None else snapshot.get('candidates', [])
    near = [c for c in candidates if np.linalg.norm(
        np.asarray(c['position_base_m'], float)[:2] - pad_xy) < .12]
    if len(near) != 1:
        raise RuntimeError('Release retreat rejected: expected one currently observed parcel on pad')
    parcel = near[0]
    center = np.asarray(parcel.get('oriented_center_base_m', parcel['position_base_m']), float)
    size = np.asarray(parcel.get('oriented_extent_m', parcel['extent_xy_m']), float)
    if (center.shape != (3,) or size.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(size).all() or np.any(size <= 0)):
        raise RuntimeError('Release retreat rejected: invalid observed parcel geometry')
    separation = pose[:2, 3] - center[:2]
    away = float(direction @ separation)
    if not np.isfinite(away) or away <= 0:
        raise RuntimeError('Release retreat rejected: measured tool axis does not point away from parcel')
    goal = pose[:3, 3] + np.r_[direction * distance, rise]
    reference = np.asarray(reference, float)[8:15]
    offset = np.asarray(offset, float)[8:15]
    lower_offset, upper_offset = np.full(7, -cap), np.full(7, cap)
    if unwinding:
        lower_offset = np.minimum(lower_offset, offset)
        upper_offset = np.maximum(upper_offset, offset)
    lower = np.maximum(np.asarray(low)[8:15], reference + lower_offset)
    upper = np.minimum(np.asarray(high)[8:15], reference + upper_offset)
    if np.any(lower > upper) or not np.isfinite(np.r_[lower, upper]).all():
        raise RuntimeError('Release retreat rejected: no feasible original joint/correction bounds')
    if execution_validator is not None:
        certificate = execution_validator(goal, parcel)
        audit = dict(direction_source='measured_tool_positive_z_horizontal_projection',
                     projection_norm=projection, direction_xy=direction.tolist(),
                     origin_base_m=pose[:3, 3].tolist(), goal_base_m=goal.tolist(),
                     observed_parcel=parcel, observation_stamp=snapshot.get('stamp'),
                     away_dot_m=away, horizontal_distance_m=float(distance), rise_m=float(rise),
                     lower_joint_rad=lower.tolist(), upper_joint_rad=upper.tolist(),
                     original_reference_rad=reference.tolist(), correction_cap_rad=float(cap),
                     position_tolerance_m=float(tolerance), **certificate)
        solution = np.asarray(certificate['feasibility_joint_rad'])
        certified = (certificate['certified'] and np.isfinite(solution).all()
                     and np.all(solution >= lower-1e-10) and np.all(solution <= upper+1e-10)
                     and np.isfinite(certificate['position_residual_m'])
                     and certificate['position_residual_m'] <= tolerance)
        audit['certified'] = bool(certified)
        if not certified:
            raise RetreatPreflightError('Release retreat rejected: ' + certificate['reason'], audit)
        return goal, audit
    rotation = Rotation.from_matrix(pose[:3, :3])
    free = upper - lower > 1e-10
    solved = (lower + upper) / 2

    def residual(x):
        joints = solved.copy(); joints[free] = x
        mat = kinematics.matrix(joints, 1)
        angle = (rotation.inv() * Rotation.from_matrix(mat[:3, :3])).as_rotvec()
        return np.r_[mat[:3, 3] - goal, .15 * angle]

    if np.any(free):
        solution = least_squares(residual, np.clip(q[free], lower[free], upper[free]),
                                 bounds=(lower[free], upper[free]), max_nfev=180,
                                 ftol=1e-8, xtol=1e-8, gtol=1e-8)
        solved[free] = solution.x
    error = residual(solved[free])
    position_error = float(np.linalg.norm(error[:3]))
    angle_error = float(np.linalg.norm(error[3:]) / .15)
    initial = dict(position_residual_m=position_error,
                   orientation_residual_degrees=float(np.rad2deg(angle_error)))
    method = 'original_weighted_least_squares'
    fallback = None
    angle_limit = np.deg2rad(8)
    if np.any(free) and (position_error > tolerance or angle_error > angle_limit):
        # A weighted objective is not a feasibility test for two independent
        # tolerances. Retain the exact same box and orientation acceptance,
        # and explicitly constrain orientation while minimizing position error.
        # The tiny solver margin prevents roundoff at the unchanged 8-degree
        # acceptance boundary. No solution is used as an execution reference.
        def objective(x):
            e = residual(x)[:3]
            return float(e @ e)

        def orientation_slack(x):
            angular = residual(x)[3:] / .15
            return (angle_limit - 1e-6)**2 - float(angular @ angular)

        result = minimize(objective, solved[free], method='SLSQP',
                          bounds=list(zip(lower[free], upper[free])),
                          constraints=[dict(type='ineq', fun=orientation_slack)],
                          options=dict(maxiter=180, ftol=1e-12))
        solved[free] = np.clip(result.x, lower[free], upper[free])
        error = residual(solved[free])
        position_error = float(np.linalg.norm(error[:3]))
        angle_error = float(np.linalg.norm(error[3:]) / .15)
        method = 'explicit_same_orientation_constraint'
        fallback = dict(solver_success=bool(result.success), message=str(result.message),
                        iterations=int(result.nit))
    audit = dict(direction_source='measured_tool_positive_z_horizontal_projection',
                     projection_norm=projection, direction_xy=direction.tolist(),
                     origin_base_m=pose[:3, 3].tolist(), goal_base_m=goal.tolist(),
                     observed_parcel=parcel, observation_stamp=snapshot.get('stamp'),
                     away_dot_m=away, horizontal_distance_m=float(distance), rise_m=float(rise),
                     feasibility_joint_rad=solved.tolist(), lower_joint_rad=lower.tolist(),
                     upper_joint_rad=upper.tolist(), position_residual_m=position_error,
                     orientation_residual_degrees=float(np.rad2deg(angle_error)),
                     original_reference_rad=reference.tolist(), correction_cap_rad=float(cap),
                     solver_method=method, initial_weighted_result=initial, fallback=fallback,
                     position_tolerance_m=float(tolerance), orientation_tolerance_degrees=8.)
    certified = (np.isfinite(solved).all() and np.isfinite(error).all()
                 and np.all(solved >= lower) and np.all(solved <= upper)
                 and position_error <= tolerance and angle_error <= angle_limit)
    audit['certified'] = bool(certified)
    if not certified:
        raise RetreatPreflightError('Release retreat rejected: goal not certified within original bounds '
                         f'(position={position_error:.6f}m, angle={np.rad2deg(angle_error):.3f}deg)', audit)
    return goal, audit
