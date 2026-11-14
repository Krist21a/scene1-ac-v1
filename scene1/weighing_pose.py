"""Bounded weighing waypoints from observed support and robot feedback."""
import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def pose_fit_problem(kinematics, target, rotation, side):
    """Pose residual and exact geometric-Jacobian derivative for nominal IK."""
    cached_q = cached_pose = cached_jacobian = None

    def evaluate(q):
        nonlocal cached_q, cached_pose, cached_jacobian
        if cached_q is None or not np.array_equal(q, cached_q):
            cached_pose, cached_jacobian = kinematics.matrix_and_jacobian(q, side)
            cached_q = q.copy()
        angle = (rotation*Rotation.from_matrix(cached_pose[:3, :3]).inv()).as_rotvec()
        return cached_pose, cached_jacobian, angle

    def residual(q):
        pose, _, angle = evaluate(q)
        return np.r_[pose[:3, 3]-target, .12*angle]

    def jacobian(q):
        _, geometric, angle = evaluate(q)
        x, y, z = angle
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        theta = np.linalg.norm(angle)
        coefficient = (1./12.+theta*theta/720. if theta < 1e-4 else
                       (1.-.5*theta/np.tan(.5*theta))/(theta*theta))
        right_inverse = np.eye(3)+.5*skew+coefficient*(skew@skew)
        return np.vstack((geometric[:3], -.12*right_inverse@geometric[3:]))

    return residual, jacobian


def support_margin(center, rotation, extent, footprint):
    corners = np.asarray(center)+rotation.apply([[x,y,z] for x in (-extent[0]/2,extent[0]/2)
        for y in (-extent[1]/2,extent[1]/2) for z in (-extent[2]/2,extent[2]/2)])
    hull = np.asarray(footprint, np.float32)
    return min(cv2.pointPolygonTest(hull, tuple(map(float,p[:2])), True) for p in corners)


def supported_candidate(candidates):
    """Prefer support among already-qualified, similarly reachable waypoints.

    A tilted roof can move horizontally as the parcel rests upright. Do not
    spend substantial support margin on a negligible IK residual improvement.
    All original feasibility/support filters run before this ranking.
    """
    if not candidates:
        return None
    error = min(c['predicted_error'] for c in candidates)
    near = [c for c in candidates if c['predicted_error'] <= error+max(.1*error, 1e-8)]
    margin = max(c['support_margin_m'] for c in near)
    # Millimetric boundary noise should not choose a worse pose when the
    # available support is effectively the same.
    return min((c for c in near if c['support_margin_m'] >= margin-.002),
               key=lambda c: c['predicted_error'])


def regrasp_posture_reserve(kinematics, measured, command, goal):
    """A bounded posture preference at the SAME Cartesian weighing goal.

    This is only a local model. The caller must physically servo to it before
    the original weighing controller captures its unchanged reference/budget.
    """
    measured, command = np.asarray(measured), np.asarray(command)
    bias = measured[8:15]-command[8:15]
    limits = kinematics.limits[1]
    low = np.maximum(limits[:, 0], limits[:, 0]-bias)
    high = np.minimum(limits[:, 1], limits[:, 1]-bias)
    if np.any(high-low < 1e-5):
        return None
    rotation = Rotation.from_matrix(goal[:3, :3])
    preferred = limits[4, 1]-np.deg2rad(25)
    def task_error(q):
        pose = kinematics.matrix(q+bias, 1)
        return np.r_[pose[:3, 3]-goal[:3, 3], .12*(rotation*
            Rotation.from_matrix(pose[:3, :3]).inv()).as_rotvec()]
    fit = least_squares(lambda q: np.r_[task_error(q), .06*(q[4]-preferred)],
        np.clip(command[8:15], low+1e-8, high-1e-8), bounds=(low, high), max_nfev=200)
    def qualified(solution):
        error = task_error(solution.x)
        return (solution.success and np.linalg.norm(error[:3]) <= .008
                and np.linalg.norm(error[3:])/.12 <= np.deg2rad(9.3)
                and limits[4, 1]-solution.x[4] >= np.deg2rad(15))
    refined = False
    if not qualified(fit):
        # The soft 25-degree preference can sacrifice a feasible task pose.
        # Refine the SAME goal with the existing minimum reserve as a bound;
        # a weak movement cost resolves the remaining arm redundancy.
        reserve_high = high.copy()
        reserve_high[4] = min(reserve_high[4], limits[4, 1]-np.deg2rad(15))
        if np.any(reserve_high-low < 1e-5):
            return None
        fit = least_squares(lambda q: np.r_[task_error(q), .001*(q-command[8:15])],
            np.clip(fit.x, low+1e-8, reserve_high-1e-8),
            bounds=(low, reserve_high), max_nfev=200)
        refined = True
    error = task_error(fit.x)
    position = float(np.linalg.norm(error[:3]))
    angle = float(np.linalg.norm(error[3:])/.12)
    reserve = float(limits[4, 1]-fit.x[4])
    # Leave prediction margin below the original actual 10-degree gate.
    # Do not trade an already valid weighing pose for a marginal model fit.
    if not fit.success or position > .008 or angle > np.deg2rad(9.3) or reserve < np.deg2rad(15):
        return None
    return dict(reference=fit.x, predicted_position_m=position,
                predicted_angle_degrees=float(np.rad2deg(angle)),
                wrist_reserve_degrees=float(np.rad2deg(reserve)),
                same_goal_refined=refined)


def alternate_weighing_pose(kinematics, measured, command, geometry, goal, surface,
                           cycle=None, wrist_config=None, diagnostics=None):
    """Select a nearby supported Cartesian goal, never an executable joint action.

    Current tracking bias is only a local feasibility estimate. The caller must
    still servo with measured feedback and verify the actual parcel pose before
    handing over to the original placement controller.
    """
    extent = np.asarray(geometry['extent'])
    relative = geometry['rotation']
    center = goal[:3,3]+Rotation.from_matrix(goal[:3,:3]).apply(geometry['center'])
    box = Rotation.from_matrix(goal[:3,:3])*relative
    yaw = np.arctan2(box.as_matrix()[1,0],box.as_matrix()[0,0])
    heading = np.round(yaw/(np.pi/2))*(np.pi/2)
    q = np.asarray(measured)[8:15]
    bias = q-np.asarray(command)[8:15]
    limits = kinematics.limits[1]
    low = np.maximum(limits[:,0], limits[:,0]+bias)
    high = np.minimum(limits[:,1], limits[:,1]+bias)
    if np.any(high-low < 1e-5):
        return None
    seed = np.clip(q, low+1e-6, high-1e-6)
    def fit(target, rotation):
        def objective(joints):
            pose = kinematics.matrix(joints, 1)
            return np.r_[pose[:3,3]-target, .12*(rotation*
                Rotation.from_matrix(pose[:3,:3]).inv()).as_rotvec()]
        jacobian = '2-point'
        if cycle is not None:
            objective, jacobian = pose_fit_problem(kinematics, target, rotation, 1)
        solution = least_squares(objective, seed, jac=jacobian, bounds=(low,high), max_nfev=70)
        error = objective(solution.x)
        return float(np.linalg.norm(error)), float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])/.12)
    baseline = fit(goal[:3,3], Rotation.from_matrix(goal[:3,:3]))[0]
    candidates = []
    footprint = surface['footprint_xy_m']
    support_reserve = (float(cycle.config.get('weighing_support_reserve_m', 0.))
                       if cycle is not None else 0.)
    if not np.isfinite(support_reserve) or not 0. <= support_reserve <= .015:
        raise ValueError('Weighing support planning reserve must be within 0–15 mm')
    minimum_support = .025+support_reserve
    if diagnostics is not None:
        diagnostics['minimum_support_margin_m'] = minimum_support
        diagnostics['support_planning_reserve_m'] = support_reserve
    for dx,dy in ((0.,0.),(-.01,.03),(0.,.03),(.01,.03),(-.01,0.),(.01,0.),(-.01,-.03),(0.,-.03),(.01,-.03)):
        destination = center+np.array([dx,dy,0.])
        for offset in (-10.,0.,10.):
            candidate_yaw = float(np.clip(yaw+np.deg2rad(offset),
                heading-np.deg2rad(10.), heading+np.deg2rad(10.)))
            box_rotation = Rotation.from_euler('z', candidate_yaw)
            # Stay within ten degrees of the existing goal and the established
            # heading range; no large rotation above the weighing pad.
            if (box_rotation*box.inv()).magnitude() > np.deg2rad(10.01):
                continue
            margin = support_margin(destination,box_rotation,extent,footprint)
            # Retain the original 25 mm live gate and leave additional room
            # for measured pose/attachment changes after the nominal search.
            # A rejected placement must not win merely through joint reserve.
            if margin < minimum_support:
                continue
            tool_rotation = box_rotation*relative.inv()
            xyz = destination-tool_rotation.apply(geometry['center'])
            score, position, angle = fit(xyz,tool_rotation)
            # The legacy search was only a rescue for a poor placement fit.
            # Joint planning ranks the same spatial candidates by the next
            # skill, using stricter absolute pose errors rather than requiring
            # a 20% improvement over an already near-zero placement residual.
            qualified_pose = (position <= .008 and angle <= np.deg2rad(5)
                              if cycle is not None else position <= .01 and score < baseline*.8)
            if not qualified_pose:
                continue
            candidates.append(dict(center=destination, rotation=box_rotation, predicted_error=score,
                predicted_position_m=position, predicted_angle_rad=angle,
                baseline_error=baseline, support_margin_m=margin,
                shift_xy_m=[dx,dy], yaw_offset_degrees=float(np.rad2deg(candidate_yaw-yaw))))
    if cycle is not None:
        if diagnostics is not None:
            diagnostics['supported_pose_candidates'] = len(candidates)
        qualified = []
        for candidate in candidates:
            rejection = {}
            reserve = weighing_cycle_reserve(cycle, measured, command, geometry,
                                             candidate, surface, wrist_config, rejection)
            if reserve is not None:
                candidate.update(reserve)
                qualified.append(candidate)
            elif diagnostics is not None:
                reasons = diagnostics.setdefault('rejections', {})
                reason = rejection.get('stage', 'unknown')
                reasons[reason] = reasons.get(reason, 0)+1
                diagnostics.setdefault('rejection_examples', {}).setdefault(reason,
                    rejection.get('last_fit', {}))
        candidates = qualified
        if diagnostics is not None:
            diagnostics['cycle_qualified_candidates'] = len(candidates)
        # Python's stable minimum preserves enumeration order on exact ties.
        best = min(candidates, key=lambda c: (-c['regrasp_joint_reserve_rad'],
                   c['insertion_joint_path_rad']), default=None)
    else:
        best = supported_candidate(candidates)
    if best is not None:
        minimum = min(candidates, key=lambda c: c['predicted_error'])
        best.update(minimum_predicted_error=minimum['predicted_error'],
                    minimum_error_support_margin_m=minimum['support_margin_m'],
                    support_priority=cycle is None, joint_cycle_priority=cycle is not None)
    return best


def weighing_cycle_reserve(cycle, measured, command, geometry, candidate, surface, wrist,
                          rejection):
    """Conservative nominal path check, not permission to execute predicted joints.

    Only existing placement candidates reach this function. Nominal open-finger
    sweeps add a 5 mm margin; execution retains its original 3 mm gate. This does not claim
    whole-robot collision certification; live guards and settled reobservation
    remain mandatory, and all predicted joints are discarded after ranking.
    """
    import copy
    from .command_sweep import sampled_sweep
    k = cycle.k
    measured, command = np.asarray(measured), np.asarray(command)
    extent = np.asarray(geometry['extent'])
    center = np.asarray(candidate['center']).copy()
    support = float(surface['position_base_m'][2])
    center[2] = support+extent[2]/2
    box = candidate['rotation']
    rotation = box*geometry['rotation'].inv()
    release = np.eye(4)
    release[:3, :3] = rotation.as_matrix()
    release[:3, 3] = center-rotation.apply(geometry['center'])
    major = box.as_matrix()[:2, 0]
    target = dict(position_base_m=center.tolist(), oriented_center_base_m=center.tolist(),
                  major_axis_xy=major.tolist(), oriented_extent_m=extent[:2].tolist(),
                  extent_xy_m=extent[:2].tolist(), top_z_base_m=float(support+extent[2]))
    joints = measured[8:15].copy()
    reserve, path = [], 0.
    planning_margin = .005
    refinements = []

    def fit(goal, initial, side=1):
        bounds = k.limits[side]
        # These are future nominal waypoints, across support/contact changes.
        # The loaded preparation's instantaneous tracking error is NOT a
        # calibrated bias for the unloaded retreat or the other arm. Use the
        # physical bounds here; only the immediate placement fit above uses
        # the currently observed bias. Execution still checks actual feedback.
        low, high = bounds[:, 0], bounds[:, 1]
        if np.any(high-low < 1e-5):
            return None
        desired = Rotation.from_matrix(goal[:3, :3])
        residual, jacobian = pose_fit_problem(k, goal[:3, 3], desired, side)
        seed = np.clip(initial, low+1e-8, high-1e-8)
        evaluations = 0
        for budget in (12, 58):
            result = least_squares(residual, seed, jac=jacobian,
                                   bounds=(low, high), max_nfev=budget)
            evaluations += result.nfev
            error = residual(result.x)
            if (np.isfinite(result.x).all() and np.isfinite(error).all()
                    and np.linalg.norm(error[:3]) <= .004
                    and np.linalg.norm(error[3:])/.12 <= np.deg2rad(5)):
                break
            seed = np.clip(result.x, low+1e-8, high-1e-8)
        weighted_fit = dict(success=bool(result.success), evaluations=int(evaluations),
            position_m=float(np.linalg.norm(error[:3])),
            angle_degrees=float(np.rad2deg(np.linalg.norm(error[3:])/.12)))
        refined = False
        if (cycle.config.get('weighing_feasibility_refine', False)
                and np.isfinite(result.x).all() and np.isfinite(error).all()
                and (np.linalg.norm(error[:3]) > .004
                     or np.linalg.norm(error[3:])/.12 > np.deg2rad(5))):
            # Weighted pose error can trade position against an unused angle
            # allowance. Check feasibility with the SAME independent limits;
            # predicted joints are still discarded after candidate ranking.
            angle_limit = np.deg2rad(5)-1e-6
            def objective(q):
                position = residual(q)[:3]
                return float(position@position)/.004**2
            def objective_jac(q):
                return 2.*jacobian(q)[:3].T@residual(q)[:3]/.004**2
            def angle_slack(q):
                angle = residual(q)[3:]/.12
                return 1.-float(angle@angle)/angle_limit**2
            def angle_jac(q):
                return -2.*(residual(q)[3:]/.12)@(jacobian(q)[3:]/.12)/angle_limit**2
            result = minimize(objective, np.clip(result.x, low+1e-8, high-1e-8),
                jac=objective_jac, method='SLSQP', bounds=list(zip(low, high)),
                constraints=[dict(type='ineq', fun=angle_slack, jac=angle_jac)],
                options=dict(maxiter=60, ftol=1e-10))
            evaluations += result.nfev
            refined = True
            error = residual(result.x)
            refinements.append(dict(stage=rejection.get('stage'), weighted=weighted_fit,
                success=bool(result.success), iterations=int(result.nit),
                position_m=float(np.linalg.norm(error[:3])),
                angle_degrees=float(np.rad2deg(np.linalg.norm(error[3:])/.12))))
            rejection['feasibility_refinements'] = refinements
        rejection['last_fit'] = dict(success=bool(result.success),
            evaluations=int(evaluations),
            feasibility_refined=refined, weighted_fit=weighted_fit,
            position_m=float(np.linalg.norm(error[:3])),
            angle_degrees=float(np.rad2deg(np.linalg.norm(error[3:])/.12)))
        # Seven joints leave a redundant pose coordinate. A bounded solver
        # may reach its evaluation budget after satisfying every Cartesian
        # tolerance without meeting its gradient stopping criterion. Judge
        # this discarded nominal waypoint by FK and physical bounds; retain
        # the optimizer status in diagnostics, not as a reachability claim.
        if (not np.isfinite(result.x).all() or not np.isfinite(error).all()
                or np.any(result.x < low) or np.any(result.x > high)
                or np.linalg.norm(error[:3]) > .004
                or np.linalg.norm(error[3:])/.12 > np.deg2rad(5)):
            return None
        reserve.append(float(np.min(np.minimum(result.x-low, high-result.x))))
        return result.x

    rejection['stage'] = 'release_reachability'
    joints = fit(release, joints)
    if joints is None:
        return None
    retreat = release.copy()
    outward = release[:2, 2].copy()
    rejection['stage'] = 'retreat_direction'
    if np.linalg.norm(outward) < .2 or outward@(release[:2, 3]-center[:2]) <= 0:
        return None
    outward /= np.linalg.norm(outward)
    retreat[:2, 3] += outward*wrist.get('release_short_retreat_m', .035)
    retreat[2, 3] += wrist.get('release_clearance_rise_m', .012)
    forward = major if major@(center[:2]-retreat[:2, 3]) >= 0 else -major
    closing = np.r_[-forward[1], forward[0], 0.]
    pitch = np.deg2rad(wrist.get('regrasp_pitch_degrees', 12.5))
    z_axis = -np.r_[forward*np.cos(pitch), -np.sin(pitch)]
    level = retreat.copy()
    level[:3, :3] = np.column_stack((closing, np.cross(z_axis, closing), z_axis))
    align = level.copy()
    align[:2, 3] = center[:2]-forward*(extent[0]/2+.015)
    descend = align.copy(); descend[2, 3] = center[2]
    insert = descend.copy()
    insert[:2, 3] = center[:2]-forward*max(0., extent[0]/2-.02)
    for name, waypoint in zip(('retreat', 'level', 'align', 'descend', 'insert'),
                             (retreat, level, align, descend, insert)):
        rejection['stage'] = name+'_reachability'
        q = fit(waypoint, joints)
        if q is None:
            return None
        rejection['stage'] = name+'_sweep'
        if not sampled_sweep(k, joints, q, target, support, margin=planning_margin)['clear']:
            return None
        path += float(np.linalg.norm(q-joints))
        joints = q
    local_reserve = min(reserve)
    # Predict the new attachment only for handover reachability. Execution
    # reconstructs it from actual contact and measured FK as before.
    tool = k.matrix(joints, 1)
    tool_rotation = Rotation.from_matrix(tool[:3, :3])
    shadow = copy.copy(cycle)
    shadow.side = 1
    shadow.geometry = dict(center=tool_rotation.inv().apply(center-tool[:3, 3]),
                           rotation=tool_rotation.inv()*box, extent=extent)
    lift = tool.copy(); lift[2, 3] += cycle.config.get('lift_m', .16)
    rejection['stage'] = 'lift_reachability'
    joints = fit(lift, joints)
    if joints is None:
        return None
    rejection['stage'] = 'handover_reachability'
    try:
        handover = shadow._handover_goal()
    except RuntimeError:
        return None
    joints = fit(handover, joints)
    if joints is None:
        return None
    nominal = measured.copy(); nominal[8:15] = joints
    receiver = shadow._receiver_goal(nominal, .08)
    rejection['stage'] = 'receiver_reachability'
    receiver_joints = fit(receiver, measured[:7], side=0)
    if receiver_joints is None:
        return None
    # Enclose the predicted (slightly tilted) held parcel in a horizontal-axis
    # box before applying the SAME public finger geometry to the receiver.
    # This is conservative for tilt and is still not whole-robot certification.
    held_center, held_rotation, held_extent = shadow.world_box(nominal)
    axis = held_rotation.as_matrix()[:2, 0]
    if np.linalg.norm(axis) < .2:
        rejection['stage'] = 'handover_parcel_axis'
        return None
    axis = axis/np.linalg.norm(axis)
    upright = np.column_stack(([*axis, 0.], [-axis[1], axis[0], 0.], [0., 0., 1.]))
    enclosing = np.abs(upright.T@held_rotation.as_matrix())@held_extent
    bottom = float(held_center[2]-enclosing[2]/2)
    held_target = dict(position_base_m=held_center.tolist(),
        oriented_center_base_m=held_center.tolist(), major_axis_xy=axis.tolist(),
        extent_xy_m=enclosing[:2].tolist(), oriented_extent_m=enclosing[:2].tolist(),
        top_z_base_m=float(held_center[2]+enclosing[2]/2))
    rejection['stage'] = 'receiver_approach_sweep'
    if not sampled_sweep(k, measured[:7], receiver_joints, held_target, bottom,
                         side=0, margin=planning_margin)['clear']:
        return None
    rejection['stage'] = 'receiver_insert_reachability'
    inserted = fit(shadow._receiver_goal(nominal, 0.), receiver_joints, side=0)
    if inserted is None:
        return None
    rejection['stage'] = 'receiver_insert_sweep'
    if not sampled_sweep(k, receiver_joints, inserted, held_target, bottom,
                         side=0, margin=planning_margin)['clear']:
        return None
    return dict(regrasp_joint_reserve_rad=local_reserve, insertion_joint_path_rad=path,
                cycle_reachability_checked=True, local_finger_sweeps_checked=True,
                cycle_feasibility_refinements=refinements,
                planning_margin_m=planning_margin, receiver_finger_sweeps_checked=True,
                collision_scope='sampled giver/receiver open fingers versus predicted parcel; live guards retained')
