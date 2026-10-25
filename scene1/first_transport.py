"""Enter the existing visual lift after the original first contact checks."""
import copy
import numpy as np
from .visual_pickup import VisualPickup


def begin_first_transport(cycle, assist, template, grasp_context, snapshot, measured, command):
    if (cycle.completed != 0 or cycle.active or cycle.done or not assist.first_grasp_complete
            or assist.regrasp_complete or assist.pending is not None):
        raise RuntimeError('First visual transport requires original confirmed first contact')
    if template is None or grasp_context is None or snapshot is None:
        raise RuntimeError('First visual transport requires observed grasp and support context')
    surfaces = [s for s in snapshot.get('surfaces', []) if not s.get('partial', True)
                and s.get('points',0) >= 60 and min(s['extent_xy_m']) >= .145]
    if len(surfaces) != 1:
        raise RuntimeError('First visual transport requires one unambiguous complete observed pad')
    surface = copy.deepcopy(surfaces[0])
    geometry = copy.deepcopy(assist.geometry[1])
    if geometry is None:
        raise RuntimeError('First visual transport requires settled object-to-hand geometry')
    goal = cycle.k.matrix(np.asarray(measured)[8:15], 1)
    goal[2,3] += .16  # Same lift as VisualPickup.local_grasp, with the same observed lift gate.
    resume = dict(target=copy.deepcopy(grasp_context['target']), plane=grasp_context['plane'],
                  geometry=geometry, goal=goal, surface=surface, stage='lift')
    pickup = VisualPickup(cycle,template,surface['position_base_m'],measured,command,resume)
    pickup.motion.command = np.asarray(command).copy()
    pickup.pose_frames = pickup.unverified_pose = 0
    pickup.first_transport = True
    return pickup, np.asarray(surface['position_base_m']).copy()
