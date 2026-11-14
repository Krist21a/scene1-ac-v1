"""Plan distinct drop slots relative to a visually observed bin rim."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def validate_layout(config):
    count = config.get('max_parcels', 1)
    offsets = np.asarray(config.get('bin_lateral_offsets_m', [0.]), float)
    depth = np.asarray(config.get('bin_depth_offsets_m', [0.]*count), float)
    angles = np.asarray(config.get('bin_tool_rpy_degrees', [70., 5., -99.]), float)
    if count not in (1, 2, 3, 4) or offsets.shape != (count,) or not np.isfinite(offsets).all():
        raise ValueError('Visual cycle needs one finite bin offset per parcel')
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError('Bin tool orientation must contain three finite angles')
    if depth.shape != (count,) or not np.isfinite(depth).all():
        raise ValueError('Visual cycle needs one finite bin depth offset per parcel')
    mode = config.get('bin_layout_mode', 'line')
    if mode not in ('line', 'flat_two_stacks', 'flat_grid'):
        raise ValueError('Unknown bin layout mode')
    if mode == 'flat_two_stacks':
        if (count != 4 or not np.allclose(offsets[:2], offsets[2:], atol=1e-9, rtol=0)
                or not np.allclose(depth[:2], depth[2:], atol=1e-9, rtol=0)
                or abs(offsets[1]-offsets[0]) < .12):
            raise ValueError('Flat stacking requires two repeated locations separated by at least 120 mm')
    elif mode == 'flat_grid':
        centers = np.column_stack((depth, offsets))
        if count != 4 or min(np.linalg.norm(a-b) for i,a in enumerate(centers) for b in centers[i+1:]) < .12:
            raise ValueError('Flat grid requires four distinct locations separated by at least 120 mm')
    elif count > 1 and min(abs(a-b) for i, a in enumerate(offsets) for b in offsets[i+1:]) < .06:
        raise ValueError('Bin slots must be separated by at least 60 mm')
    return offsets, angles


def slot_axis(rim):
    edges = np.roll(rim['corners'], -1, axis=0)-rim['corners']
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    lateral = edges[np.argmax(abs(edges[:, 1]))]
    return -lateral if lateral[1] < 0 else lateral


def slot_points(points, rim, config, index):
    """Exclude neighboring parcels that become visible after hand withdrawal.

    Distinct drop slots own disjoint lateral regions. This association is in
    addition to before/after subtraction, never an object count by itself.
    """
    offsets, _ = validate_layout(config)
    if not 0 <= index < len(offsets):
        raise ValueError('No remaining bin slot')
    if len(offsets) == 1 or not len(points):
        return points
    lateral = (points[:, :2]-(rim['center']+[.025, 0.]))@slot_axis(rim)
    if config.get('bin_layout_mode') == 'flat_grid':
        axis = slot_axis(rim)
        depth = (points[:, :2]-(rim['center']+[.025, 0.]))@np.array([axis[1], -axis[0]])
        centers = np.column_stack((config['bin_depth_offsets_m'], offsets))
        distances = np.linalg.norm(np.column_stack((depth,lateral))[:,None,:]-centers[None,:,:],axis=2)
        others = np.min(np.delete(distances,index,axis=1),axis=1)
        return points[distances[:,index]+.002 < others]
    if config.get('bin_layout_mode') == 'flat_two_stacks':
        # Each stack owns a disjoint region. Repeated placement additionally
        # requires new 3D surface height in VisualCycle, never region count alone.
        offsets = offsets[:2]
        index %= 2
    distances = abs(lateral[:, None]-offsets)
    others = np.min(np.delete(distances, index, axis=1), axis=1)
    # Exclude the shared boundary rather than assigning its noise to a slot.
    return points[distances[:, index]+.002 < others]


def empty_retreat_posture(kinematics, measured, goal, rim, low, high):
    """Find a bounded empty-hand posture for the unchanged withdrawal goal.

    A straight elbow can leave a local Cartesian solver at a stationary point.
    The returned posture is only a bounded servo preference, never a command.
    Its nominal joint path must keep both pinch frame and wrist above the
    existing rim-clearance gate; actual arrival still belongs to VisualCycle.
    """
    physical = np.asarray(kinematics.limits[0])
    lower = np.maximum(np.asarray(low)[:7], physical[:, 0])+1e-6
    upper = np.minimum(np.asarray(high)[:7], physical[:, 1])-1e-6
    start = np.asarray(measured)[:7]
    seed = np.clip(start, lower, upper)
    desired = Rotation.from_matrix(goal[:3, :3])

    def residual(q):
        pose = kinematics.matrix(q, 0)
        return np.r_[pose[:3, 3]-goal[:3, 3],
            .12*(desired*Rotation.from_matrix(pose[:3, :3]).inv()).as_rotvec()]

    seeds = [seed]
    for elbow in (-.6, -1.2):
        bent = seed.copy()
        bend = max(0., seed[3]-elbow)
        bent[[1, 3, 5]] += [bend/2, -bend, -bend/2]
        seeds.append(np.clip(bent, lower, upper))
    fits = []
    attempts = []
    for initial in seeds:
        fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=160)
        error = residual(fit.x)
        position, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])/.12)
        attempts.append(dict(position_error_m=position, angle_error_rad=angle))
        if position <= .004 and angle <= np.deg2rad(5.):
            fits.append((float(np.linalg.norm(fit.x-start)), fit.x.copy(), position, angle))
    info = dict(passed=False, position_error_m=position, angle_error_rad=angle,
                source='bounded IK preference for original empty-hand retreat', ik_attempts=attempts)
    if not fits:
        return None, dict(info, reason='nominal pose did not converge')
    _, target, position, angle = min(fits, key=lambda item: item[0])
    info.update(position_error_m=position, angle_error_rad=angle)
    # Sum of downstream link lengths bounds translation per joint radian.
    # Using the full chain is conservative for both wrist and pinch frame.
    reach_bound = sum(float(np.linalg.norm(origin[:3, 3]))
                      for origin, _, _ in kinematics.chains[0])+float(np.linalg.norm(kinematics.tool_xyz))
    count = max(2, int(np.ceil(reach_bound*np.sum(abs(target-start))/.005))+1)
    minimum = float('inf')
    for fraction in np.linspace(0., 1., count):
        pose = kinematics.matrix(start+fraction*(target-start), 0)
        wrist = pose[:3, 3]-pose[:3, :3]@kinematics.tool_xyz
        minimum = min(minimum, float(min(pose[2, 3], wrist[2])-rim['rim_z']))
    info.update(path_samples=count, maximum_nominal_translation_step_m=.005,
                minimum_tcp_wrist_rim_clearance_m=minimum, required_rim_clearance_m=.08)
    if minimum <= .08:
        return None, dict(info, reason='nominal empty-hand path lacks original rim clearance')
    info.update(passed=True, bounded_joint_target=target.tolist())
    return target.copy(), info


def plan_bin_drop(rim, geometry, config, index, kinematics=None, measured=None, stack_center_xy=None):
    """Prefer a deeper observable drop only within the existing pose gates."""
    advance = float(config.get('bin_view_depth_search_m', 0.))
    if not np.isfinite(advance) or not 0. <= advance <= .02:
        raise ValueError('Bin viewing depth search must stay within 0–20 mm')
    if not advance:
        return _plan_bin_drop(rim, geometry, config, index, kinematics, measured, stack_center_xy)
    offsets, _ = validate_layout(config)
    if not 0 <= index < len(offsets) or kinematics is None or measured is None:
        raise ValueError('Bin viewing depth search requires a current bounded arm pose')
    if config.get('bin_follow_confirmed_stack', False):
        raise ValueError('Depth search cannot override an observed stack support location')
    if (config.get('bin_layout_mode') != 'flat_two_stacks'
            or config.get('bin_confirmation_mode') != 'independent_entry'):
        raise ValueError('Viewing depth search requires the independently confirmed two-slot layout')
    original = list(config.get('bin_depth_offsets_m', [0.]*len(offsets)))
    rejected = []
    # These are discarded Cartesian candidates. Every candidate goes through
    # the unchanged full-box margin, bounded flat IK and arm reach checks.
    # Prefer the farthest qualified location to expose more surface behind
    # the near wall; a nominal plan never substitutes for actual bin evidence.
    for delta in np.linspace(advance, 0., int(np.ceil(advance/.005))+1):
        candidate = dict(config)
        candidate['bin_depth_offsets_m'] = original.copy()
        for slot in (index % 2, index % 2+2):
            candidate['bin_depth_offsets_m'][slot] += float(delta)
        try:
            goal, info = _plan_bin_drop(rim, geometry, candidate, index,
                                       kinematics, measured, stack_center_xy)
        except RuntimeError as error:
            rejected.append(dict(advance_m=float(delta), reason=str(error)))
            if delta == 0.:
                raise
            continue
        info['view_depth_planning'] = dict(original_depth_m=original[index],
            maximum_advance_m=advance, chosen_advance_m=float(delta), rejected=rejected,
            evidence_required='Original independent new-surface confirmation after release')
        return goal, info
    raise RuntimeError('No original-gate bin drop candidate')


def _plan_bin_drop(rim, geometry, config, index, kinematics=None, measured=None, stack_center_xy=None):
    offsets, angles = validate_layout(config)
    if not 0 <= index < len(offsets):
        raise ValueError('No remaining bin slot')
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    extent = np.asarray(geometry['extent'])
    lateral = slot_axis(rim)
    stacking = config.get('bin_layout_mode') == 'flat_two_stacks'
    flat = stacking or config.get('bin_layout_mode') == 'flat_grid'
    flat_candidates = []
    if flat:
        if kinematics is None or measured is None or extent[1] > min(extent[0], extent[2]):
            raise RuntimeError('Flat stacking requires current kinematics and an observed narrow box axis')
        object_offset = geometry['rotation'].inv().apply(geometry['center'])
        major = np.array([lateral[1], -lateral[0], 0.]) * (1. if object_offset[0] >= 0 else -1.)
        for sign in (1., -1.):
            narrow = np.array([0., 0., sign])
            box = Rotation.from_matrix(np.column_stack((major, narrow, np.cross(major, narrow))))
            flat_candidates.append(box*geometry['rotation'].inv())
        rotation = flat_candidates[0]
    box_rotation = rotation*geometry['rotation']
    height = abs(box_rotation.as_matrix()[2])@extent
    release_clearance = float(config.get('bin_release_clearance_m', .065))
    if not np.isfinite(release_clearance) or not .05 <= release_clearance <= .065:
        raise ValueError('Bin release planning clearance must be within 50-65 mm')
    center = np.r_[rim['center'], rim['rim_z']+release_clearance+height/2]
    center[0] += .025
    center[:2] += offsets[index]*lateral
    # Depth is perpendicular to the slot axis, so a nearer reachable waypoint
    # cannot move a parcel into a neighboring slot's evidence region.
    depth = config.get('bin_depth_offsets_m', [0.]*len(offsets))[index]
    center[:2] += depth*np.array([lateral[1], -lateral[0]])
    stack_clearance_shift = 0.
    if stacking and index >= 2 and config.get('bin_follow_confirmed_stack', False):
        reference = np.asarray(stack_center_xy, float)
        if reference.shape != (2,) or not np.isfinite(reference).all() or np.linalg.norm(reference-center[:2]) > .065:
            raise RuntimeError('Missing or displaced observed stack support')
        center[:2] = reference
        if config.get('bin_stack_clearance_adjust', False):
            # Select a nearby feasible drop before solving IK. Never reduce
            # the original all-corner margin; retain overlap with the observed
            # support by limiting this planning displacement to 20 mm.
            corner_offsets = box_rotation.apply([[a,b,c] for a in (-extent[0]/2,extent[0]/2)
                for b in (-extent[1]/2,extent[1]/2) for c in (-extent[2]/2,extent[2]/2)])
            direction = np.asarray(rim['center'])-reference
            direction /= max(float(np.linalg.norm(direction)),1e-9)
            for shift in np.linspace(0.,.020,21):
                candidate_xy = reference+shift*direction
                margin = min(cv2.pointPolygonTest(rim['corners'], tuple(map(float,p)),True)
                    for p in candidate_xy+corner_offsets[:,:2])
                if margin >= .027:
                    center[:2] = candidate_xy
                    stack_clearance_shift = float(shift)
                    break
            else:
                raise RuntimeError('No nearby stack target preserves full bin clearance')
    planned_ik = None
    if flat:
        limits = np.asarray(kinematics.limits[0])
        seed = np.clip(np.asarray(measured)[:7], limits[:, 0]+1e-6, limits[:, 1]-1e-6)
        feasible = []
        for candidate in flat_candidates:
            tcp = center-candidate.apply(geometry['center'])
            def residual(q):
                pose = kinematics.matrix(q, 0)
                return np.r_[pose[:3, 3]-tcp,
                    .12*(candidate*Rotation.from_matrix(pose[:3, :3]).inv()).as_rotvec()]
            fit = least_squares(residual, seed, bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6), max_nfev=160)
            error = residual(fit.x)
            position, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])/.12)
            if position <= .008 and angle <= .20:
                feasible.append((float(np.linalg.norm(fit.x-seed)), candidate,
                    dict(position_m=position, angle_rad=angle, bounded_joint_target=fit.x.tolist())))
        if not feasible:
            raise RuntimeError('No bounded flat placement pose meets the original arrival gates')
        _, rotation, planned_ik = min(feasible, key=lambda item:item[0])
        box_rotation = rotation*geometry['rotation']
    corners = center+box_rotation.apply([[a, b, c] for a in (-extent[0]/2, extent[0]/2)
        for b in (-extent[1]/2, extent[1]/2) for c in (-extent[2]/2, extent[2]/2)])
    margin = min(cv2.pointPolygonTest(rim['corners'], tuple(map(float, p[:2])), True) for p in corners)
    if margin < .025:
        raise RuntimeError('Observed parcel does not fit its planned bin slot with 25 mm clearance')
    goal = np.eye(4)
    goal[:3, :3] = rotation.as_matrix()
    goal[:3, 3] = center-rotation.apply(geometry['center'])
    reach = {}
    if kinematics is not None:
        # Triangle inequality gives an upper bound independent of joint pose.
        # A target outside it cannot be rescued by waiting or servo gains.
        shoulder = np.eye(4)
        remaining = []
        found = False
        for origin, joint, _ in kinematics.chains[0]:
            if found:
                remaining.append(float(np.linalg.norm(origin[:3, 3])))
            else:
                shoulder = shoulder @ origin
                found = joint == 0
        if not found:
            raise ValueError('Missing left shoulder in kinematic chain')
        upper_bound = sum(remaining)+float(np.linalg.norm(kinematics.tool_xyz))
        radius = float(np.linalg.norm(goal[:3, 3]-shoulder[:3, 3]))
        minimum_error = max(0., radius-upper_bound)
        tolerances = np.asarray(config.get('bin_arrival_position_m', .025))
        tolerance = float(tolerances[index] if tolerances.ndim else tolerances)
        reach = dict(tcp_reach_upper_bound_m=upper_bound, target_radius_m=radius,
                     position_error_lower_bound_m=minimum_error)
        if minimum_error > tolerance+1e-6:
            raise RuntimeError('Planned bin target is outside the arm reach bound: '+str(reach))
    return goal, dict(slot=index, lateral_offset_m=float(offsets[index]),
                      stack_clearance_shift_m=stack_clearance_shift,
                      depth_offset_m=float(depth),
                      layout_mode=config.get('bin_layout_mode', 'line'),
                      stack=index % 2 if stacking else None, layer=index // 2 if stacking else None,
                      planned_ik=planned_ik,
                      target_center_base_m=center.tolist(), planned_margin_m=float(margin), **reach)
