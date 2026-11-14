"""Preserve confirmed pregrasp orientation through the existing visual approach."""
import copy
import numpy as np
from .visual_pickup import VisualPickup, grasp_template, grasp_plan, table_hull
from .table_reference import table_frame, table_boundary
from .approach_edges import roof_edges


def begin_first_pickup(cycle, assist, snapshot, measured, command):
    context = assist.approach[1]
    if (cycle.completed != 0 or cycle.active or cycle.done or assist.first_grasp_complete
            or assist.regrasp_complete or assist.pending is not None
            or context is None or not context.get('aligned')):
        raise RuntimeError('First visual pickup requires confirmed initial pregrasp')
    if snapshot is None:
        raise RuntimeError('First visual pickup requires fresh RGB-D')
    surfaces = [s for s in snapshot.get('surfaces', []) if not s.get('partial', True)
                and s.get('points',0) >= 60 and min(s['extent_xy_m']) >= .145]
    if len(surfaces) != 1:
        raise RuntimeError('First visual pickup requires one unambiguous complete observed pad')
    target, plane = copy.deepcopy(context['target']), context['plane']
    pose = assist.kinematics.fk(np.asarray(measured)).right_pose
    point = assist._grasp_goal(target,pose,plane)
    # This template is a prospective approach only. Original contact confirmation
    # still registers the actual held geometry before lift or weighing.
    template = grasp_template(assist._geometry(target,pose,plane),grasp_plan(target,plane,point))
    surface = copy.deepcopy(surfaces[0])
    pickup = VisualPickup(cycle,template,surface['position_base_m'],measured,command)
    pickup.motion.command = np.asarray(command).copy()
    pickup.motion.config = copy.deepcopy(pickup.motion.config)
    pickup.motion.config['pickup_extra_insertion_m'] = 0.  # Retain initial near-face position.
    offsets = list(pickup.motion.config.get('pickup_pitch_offsets_degrees',[0.,2.5,2.5,5.]))
    offsets[0] = assist.wrist_config['first_grasp_pitch_degrees']-assist.wrist_config['regrasp_pitch_degrees']
    pickup.motion.config['pickup_pitch_offsets_degrees'] = offsets
    pickup.target, pickup.plane = target, plane
    # The original selector must observe a stable head view before fixing any
    # table/roof reference. Pregrasp's ACT camera view cannot be reused after turn.
    pickup.surface_reference = surface
    # Clear the already observed parcel before selecting from a new head view.
    # Otherwise its partial roof is excluded and a farther parcel is selected.
    pickup.first_selection_target = np.asarray(target['position_base_m'])[:2].copy()
    pickup.motion.goal[1] = pickup.motion.k.matrix(np.asarray(measured)[8:15], 1)
    pickup.motion.goal[1][2, 3] = max(pickup.motion.goal[1][2, 3],
        max(c['top_z_base_m'] for c in snapshot['candidates'])+.12)
    pickup._stage('first_clear')
    pickup.first_visual_pickup = True
    return pickup,np.asarray(surface['position_base_m']).copy()


def begin_traditional_pickup(cycle, assist, snapshot, measured, command):
    """Seed the shared visual selector from observed geometry, without ACT motion.

    The prospective template is replaced by contact-confirmed geometry in the
    runner. Selection stability, sweeps, close and lift remain in VisualPickup.
    """
    from types import SimpleNamespace
    from .visual_pickup import eligible_targets
    if cycle.completed or cycle.active or cycle.done or assist.first_grasp_complete:
        raise RuntimeError('Traditional bootstrap requires an untouched first parcel')
    if snapshot is None or not snapshot.get('candidates'):
        raise RuntimeError('Traditional bootstrap requires fresh parcel geometry')
    surfaces = [s for s in snapshot.get('surfaces', []) if not s.get('partial', True)
                and s.get('points', 0) >= 60 and min(s['extent_xy_m']) >= .145]
    if len(surfaces) != 1:
        raise RuntimeError('Traditional bootstrap requires one unambiguous complete pad')
    pose = assist.kinematics.fk(np.asarray(measured)).right_pose
    surface = copy.deepcopy(surfaces[0])
    pad = np.asarray(surface['position_base_m']).copy()
    plane = snapshot['table_z_base_m']
    ordered = sorted(snapshot['candidates'], key=lambda c:
        np.linalg.norm(assist._grasp_goal(c, pose)[:2]-pose.pos_xyz[:2]))
    if cycle.config.get('traditional_template_widest_visible', False):
        # The nearest initial image component can be a shortened visible end.
        # Use the most complete visible long axis for the prospective size
        # reference; this does not select a parcel for the shared selector.
        # Stable sorting retains the old ordering when visible lengths tie.
        ordered.sort(key=lambda c: -float(c['oriented_extent_m'][0]))
    for observed in ordered:
        target = copy.deepcopy(observed)
        rotation = assist._initial_grasp_rotation(target, pose)
        if rotation is None:
            raise RuntimeError('Traditional bootstrap requires configured grasp orientation')
        prospective = SimpleNamespace(pos_xyz=np.asarray(pose.pos_xyz), quat_xyzw=rotation.as_quat())
        point = assist._grasp_goal(target, prospective, plane)
        prospective.pos_xyz = point
        plan = grasp_plan(target, plane, point)
        template = grasp_template(assist._geometry(target, prospective, plane), plan)
        candidates = eligible_targets(snapshot, template, pad, cycle.rim)
        if any(np.linalg.norm(np.asarray(c['position_base_m'])[:2]-
                              np.asarray(target['position_base_m'])[:2]) < .012 for c in candidates):
            break
    else:
        raise RuntimeError('No initial target passed shared visual eligibility checks')
    pickup = VisualPickup(cycle, template, pad, measured, command)
    pickup.surface_reference = surface
    pickup.bootstrap_template_source = dict(
        strategy=('widest_visible' if cycle.config.get('traditional_template_widest_visible', False)
                  else 'nearest_initial'),
        position_base_m=target['position_base_m'], extent_m=template['extent'],
        actual_target_selected=False)
    # This first image supplies only a prospective size/attachment template.
    # Let the SAME stable-view selector used by parcels 2–4 choose the target.
    # Pinning an initially partial observation here can select a farther parcel
    # even after its nearer neighbor becomes fully visible, obstructing descent.
    return pickup, pad
