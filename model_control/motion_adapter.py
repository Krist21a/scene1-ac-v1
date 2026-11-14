"""Bounded Cartesian primitives for model-selected scene1 actions.

Only the servo and geometry checks are reused; VisualCycle.update and
ResidualAssist.apply (the task policies) are never called here.
"""
import copy
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.motion import action_vector, JOINTS
from scene1.visual_cycle import VisualCycle, bin_rim, track_bin_rim, inside_bin
from scene1.residual import ResidualAssist
from scene1.command_sweep import sampled_sweep
from scene1.clearance import boxes_overlap
from scene1.table_reference import table_boundary, table_frame, table_motion
from scene1.approach_edges import roof_edges, roof_edge_motion


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class ObservationBlocked(RuntimeError):
    """A fresh view may resolve this guard; never publish the rejected target."""


class MotionAdapter:
    def __init__(self, config, kinematics, low, high, initial):
        self.config, self.k = config, kinematics
        self.low, self.high = action_vector(low), action_vector(high)
        for side, start in enumerate((0, 8)):
            self.low[start:start+7] = np.maximum(self.low[start:start+7], self.k.limits[side][:, 0])
            self.high[start:start+7] = np.minimum(self.high[start:start+7], self.k.limits[side][:, 1])
        self.command = action_vector(initial)
        self.servo = VisualCycle(config['visual_cycle'], kinematics, self.low, self.high, initial)
        self.servo.command = self.command.copy()
        self.servo._stage('model_control')
        self.servo.control_period_s = .1
        self.geometry = [None, None]
        self.payload_seen = [None, None]
        self.payload_lost = [0, 0]
        self.observers = [ResidualAssist(config['visual_residual'], kinematics, self.low, self.high) for _ in range(2)]
        for observer in self.observers:
            observer.wrist_config = dict(config.get('wrist_assist', {}), image_aligned_box_pose=False)
        self.active = None
        self.rim = None
        self.support = None
        self.last_snapshot_stamp = None
        self.last_feedback = None
        self.frozen_steps = 0
        self.last_release = None
        # Opening breaks the rigid attachment before the fingers are fully
        # open. Keep this transaction across stop/timeout; only opening may
        # resume, with exactly the arm reference certified at authorization.
        self.releases = [None, None]
        self.selected_targets = [None, None]
        self.table_polygon = None
        self.payload_visual_time = [None, None]
        self.payload_snapshot_stamp = [None, None]
        self.rim_seen_at = self.table_seen_at = self.support_seen_at = None
        self.support_table_anchor = None
        self.table_anchor = None
        self.table_anchor_polygon = None
        self.table_anchor_halfspaces = None
        self.table_halfspaces = None
        self.table_transform = np.eye(4)
        self.table_tracking = None
        self.support_reference = None

    @staticmethod
    def _table_polygon(frame):
        if frame is None or frame.get('partial') or frame.get('edges') is None:
            return None
        center, rotation, size = frame['center'][:2], frame['rotation'], frame['size']
        polygon = np.asarray([center+rotation@np.array([x,y]) for x,y in
            ((-size[0]/2,-size[1]/2),(size[0]/2,-size[1]/2),
             (size[0]/2,size[1]/2),(-size[0]/2,size[1]/2))], np.float32)
        # Opposite side-edge endpoints must not impersonate a cropped front
        # edge. Require the near edge and at least two other observed sides.
        # Unseen sides stay unbounded/occupied in collision checks.
        points, normals = frame['edges']['points'], frame['edges']['normals']
        halfspaces, certified = [], []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            direction = (b-a)/np.linalg.norm(b-a)
            normal = np.array([-direction[1], direction[0]])
            selected = points[(abs((points-a)@normal) < .008) & (abs(normals@normal) >= .95)]
            if len(selected) < 6 or np.ptp(selected@direction) < max(.075, .08*np.linalg.norm(b-a)):
                certified.append(False)
                continue
            certified.append(True)
            if (center-a)@normal < 0:
                normal = -normal
            halfspaces.append((a, normal))
        nearest = int(np.argmin([abs(a@np.array([-(b-a)[1], (b-a)[0]]))/np.linalg.norm(b-a)
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0))]))
        if len(halfspaces) < 3 or not certified[nearest]:
            return None
        return polygon, np.asarray(halfspaces)

    def _table_contains(self, point, margin):
        if self.table_halfspaces is None:
            return True  # Unknown extent cannot authorize below-plane motion.
        return bool(np.all(np.sum((np.asarray(point)-self.table_halfspaces[:,0])*self.table_halfspaces[:,1],axis=1) >= margin))

    def sent(self, command):
        self.command = action_vector(command)
        self.servo.command = self.command.copy()

    def update_scene(self, snapshot):
        if snapshot is None:
            raise RuntimeError('Fresh RGB-D is required')
        if self.last_snapshot_stamp == snapshot['stamp']:
            return False
        self.last_snapshot_stamp = snapshot['stamp']
        boundary = table_boundary(snapshot, fit_tilt=True)
        if len(boundary) >= 30:
            hull = cv2.convexHull(np.asarray(boundary, np.float32)).reshape(-1, 2)
            frame = table_frame(hull, snapshot['table_z_base_m'], boundary)
            if self.table_anchor is None:
                certified = self._table_polygon(frame)
                if certified is not None:
                    polygon, halfspaces = certified
                    self.table_anchor = copy.deepcopy(frame)
                    self.table_anchor_polygon = polygon
                    self.table_anchor_halfspaces = halfspaces
                    self.table_halfspaces = halfspaces.copy()
                    self.table_polygon = polygon.copy()
                    self.table_seen_at = time.monotonic()
                    self.table_tracking = dict(reference_mode='observed_depth_edges',
                        observed_sides=len(halfspaces), unseen_sides='occupied_halfspace')
            motion = None if self.table_anchor is None else table_motion(
                dict(self.table_anchor, partial=True), frame, self.table_transform)
            if motion is not None:
                self.table_transform = motion['transform']
                self.table_polygon = (self.table_anchor_polygon@self.table_transform[:2,:2].T+
                                      self.table_transform[:2,3]).astype(np.float32)
                self.table_halfspaces = self.table_anchor_halfspaces.copy()
                self.table_halfspaces[:,0] = (self.table_anchor_halfspaces[:,0]@self.table_transform[:2,:2].T+
                                             self.table_transform[:2,3])
                self.table_halfspaces[:,1] = self.table_anchor_halfspaces[:,1]@self.table_transform[:2,:2].T
                self.table_seen_at = time.monotonic()
                self.table_tracking = motion
                self.table_tracking.update(observed_sides=len(self.table_halfspaces), unseen_sides='occupied_halfspace')
                if self.support_table_anchor is not None and self.support_reference is not None:
                    old_polygon, old_plane = self.support_table_anchor
                    distances = np.linalg.norm(self.table_polygon[:,None,:]-old_polygon[None,:,:],axis=2)
                    # Bound each fresh reference change, not displacement from
                    # the original robot pose. Geometry is still transformed
                    # from the immutable selection, never integrated framewise.
                    if (max(distances.min(axis=1)) <= .01
                            and abs(snapshot['table_z_base_m']-old_plane) <= .005):
                        source, reference_transform = self.support_reference
                        relative = self.table_transform@np.linalg.inv(reference_transform)
                        self.support = copy.deepcopy(source)
                        self.support['position_base_m'] = (relative@np.r_[source['position_base_m'],1.])[:3].tolist()
                        self.support['footprint_xy_m'] = (np.asarray(source['footprint_xy_m'])@
                            relative[:2,:2].T+relative[:2,3]).tolist()
                        self.support_seen_at = time.monotonic()
                        self.support_table_anchor = (self.table_polygon.copy(), float(snapshot['table_z_base_m']))
        new_rim = bin_rim(snapshot) if self.rim is None else track_bin_rim(snapshot, self.rim)
        if new_rim is not None:
            self.rim = new_rim
            self.rim_seen_at = time.monotonic()
        return True

    def world_box(self, side, measured):
        g = self.geometry[side]
        if g is None:
            return None
        tool = self.k.matrix(measured[side*8:side*8+7], side)
        rotation = Rotation.from_matrix(tool[:3, :3])
        return tool[:3, 3]+rotation.apply(g['center']), rotation*g['rotation'], np.asarray(g['extent'])

    @staticmethod
    def corners(box):
        center, rotation, extent = box
        return center+rotation.apply([[x, y, z] for x in (-extent[0]/2, extent[0]/2)
                                     for y in (-extent[1]/2, extent[1]/2)
                                     for z in (-extent[2]/2, extent[2]/2)])

    def observe_payload(self, side, measured, snapshot):
        if self.geometry[side] is None:
            return None
        if self.payload_snapshot_stamp[side] == snapshot['stamp']:
            return self.payload_seen[side]
        self.payload_snapshot_stamp[side] = snapshot['stamp']
        image_joints = np.asarray(snapshot.get('rgb_joint_positions_rad'), dtype=float)
        if image_joints.shape != (29,) or not np.isfinite(image_joints).all():
            raise RuntimeError('Payload observation requires RGB-aligned v52 joint history')
        image_state = np.r_[image_joints[13:20], measured[7], image_joints[20:27], measured[15]]
        poses = self.k.fk(image_state)
        pose = poses.left_pose if side == 0 else poses.right_pose
        observer = self.observers[side]
        observer.geometry[1] = self.geometry[side]
        evidence = observer._observe_box(snapshot, SimpleNamespace(right_pose=pose))
        self.geometry[side] = observer.geometry[1]
        self.payload_seen[side] = evidence
        if evidence is not None:
            evidence['image_pose'] = dict(source='rgb_aligned_joint_history', side=('left','right')[side],
                joint_stamp_s=snapshot['rgb_joint_stamp_s'], rgb_stamp_s=snapshot['stamp'])
            self.payload_visual_time[side] = time.monotonic()
        return evidence

    def describe(self, measured, snapshot):
        self.update_scene(snapshot)
        payloads = []
        for side in (0, 1):
            if self.geometry[side] is not None and self.releases[side] is None:
                self.observe_payload(side, measured, snapshot)
            box = self.world_box(side, measured)
            payloads.append(None if box is None else dict(center_base_m=box[0], quaternion_xyzw=box[1].as_quat(),
                extent_m=box[2], corners_base_m=self.corners(box), visual_evidence=self.payload_seen[side]))
        return plain(dict(candidates=snapshot.get('candidates', []), surfaces=snapshot.get('surfaces', []),
                          bin_rim=self.rim, table_polygon=self.table_polygon, payloads=payloads, last_release=self.last_release,
                          support=self.support, releases=self.releases, table_tracking=self.table_tracking,
                          table_halfspaces=self.table_halfspaces,
                          table_evidence_age_s=None if self.table_seen_at is None else time.monotonic()-self.table_seen_at,
                          support_evidence_age_s=None if self.support_seen_at is None else time.monotonic()-self.support_seen_at,
                          selected_targets=self.selected_targets, active=self.active is not None))

    def begin(self, request, measured, snapshot):
        measured = action_vector(measured)
        self.update_scene(snapshot)
        for held in (0,1):
            if self.geometry[held] is not None and self.releases[held] is None:
                self.observe_payload(held, measured, snapshot)
            elif self.geometry[held] is None and .15 < measured[held*8+7] < .85:
                raise RuntimeError('Unregistered closed gripper: cannot infer payload after restart')
        if request.get('side') not in ('left', 'right'):
            raise ValueError('side must be left or right')
        side = ('left', 'right').index(request['side'])
        context = request.get('context', 'free')
        if context not in ('free', 'grasp', 'transport', 'place', 'bin', 'handover'):
            raise ValueError('Unknown action context')
        p = dict(request=copy.deepcopy(request), side=side, context=context,
                 started=time.monotonic(), stable=0, previous=None, previous_grip=None,
                 last_stamp=None, steps=0, initial=measured.copy())
        if any(item is not None for item in self.releases):
            release = self.releases[side]
            if (release is None or request['op'] != 'set_gripper' or
                    not float(request.get('value', 1.)) < .15 or
                    context != release['context']):
                raise RuntimeError('Release in progress: only same-hand/context opening may resume')
            if not np.array_equal(self.command[JOINTS], release['reference'][JOINTS]):
                raise RuntimeError('Arm reference changed during release; cannot resume')
            if any(key in request for key in ('target_index', 'surface_index', 'reuse_target', 'target_observation_id')):
                raise ValueError('Resume release without replacing target or support evidence')
            p['release_authorized'] = True
        reuse = request.get('reuse_target', False)
        if type(reuse) is not bool:
            raise ValueError('reuse_target must be boolean')
        if reuse and 'target_index' in request:
            raise ValueError('Choose target_index or reuse_target, not both')
        if reuse:
            selected = self.selected_targets[side]
            if selected is None:
                raise ValueError('No previous model-selected target for this hand')
            expected = request.get('target_observation_id')
            if expected is not None and expected != selected['observation_id']:
                raise ValueError('target_observation_id differs from the selected target identity')
            for key in ('target','plane','roof_reference','target_seen_at'):
                p[key] = copy.deepcopy(selected[key])
            p['target_observation_id'] = selected['observation_id']
        elif 'target_observation_id' in request:
            raise ValueError('target_observation_id requires reuse_target=true')
        if 'target_index' in request:
            index = request['target_index']
            if type(index) is not int or not 0 <= index < len(snapshot['candidates']):
                raise ValueError('target_index outside this observation')
            p['target'] = copy.deepcopy(snapshot['candidates'][index])
            p['plane'] = float(p['target'].get('support_z_base_m', snapshot['table_z_base_m']))
            p['roof_reference'] = roof_edges(snapshot, p['target'])
            p['target_seen_at'] = time.monotonic()
            p['target_observation_id'] = request.get('observation_id')
        if 'surface_index' in request:
            index = request['surface_index']
            if type(index) is not int or not 0 <= index < len(snapshot['surfaces']):
                raise ValueError('surface_index outside this observation')
            surface = copy.deepcopy(snapshot['surfaces'][index])
            if surface.get('partial', True):
                raise RuntimeError('Selected support footprint is partial')
            self.support = surface
            self.support_seen_at = time.monotonic()
            self.support_table_anchor = None if self.table_polygon is None else (self.table_polygon.copy(),float(snapshot['table_z_base_m']))
            self.support_reference = (copy.deepcopy(surface), self.table_transform.copy())
        if context == 'place' and self.support is None:
            raise ValueError('place requires a previously observed surface_index')
        if context == 'transport' and self.geometry[side] is None:
            raise RuntimeError('transport requires confirmed payload geometry')
        if context == 'bin' and self.rim is None:
            raise RuntimeError('No observed bin rim')
        if request['op'] == 'move_ee':
            position_tolerance = float(request.get('position_tolerance_m', .002 if context == 'place' else .012))
            angle_tolerance = float(request.get('angle_tolerance_rad', .08 if context == 'place' else .14))
            if not np.isfinite([position_tolerance,angle_tolerance]).all() or not .001 <= position_tolerance <= .012 or not .02 <= angle_tolerance <= .14:
                raise ValueError('Arrival tolerances may only tighten original gates: position [0.001,0.012] m, angle [0.02,0.14] rad')
            p['position_tolerance_m'], p['angle_tolerance_rad'] = position_tolerance, angle_tolerance
            position = np.asarray(request['position_m'], dtype=float)
            quat = np.asarray(request['quaternion_xyzw'], dtype=float)
            if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[position, quat]).all():
                raise ValueError('Invalid finite pose')
            if abs(np.linalg.norm(quat)-1) > 1e-3:
                raise ValueError('Quaternion must be normalized')
            p['goal'] = np.eye(4)
            p['goal'][:3, :3] = Rotation.from_quat(quat).as_matrix()
            p['goal'][:3, 3] = position
            reference = request.get('reference', 'base')
            if reference not in ('base', 'table'):
                raise ValueError('reference must be base or table')
            p['reference'] = reference
            if reference == 'table':
                if self.table_seen_at is None or time.monotonic()-self.table_seen_at > 5.:
                    raise ValueError('table reference requires fresh verified depth edges')
                p['table_goal'] = np.linalg.inv(self.table_transform) @ p['goal']
                p['goal_table_polygon'] = self.table_polygon.copy()
                p['goal_table_transform'] = self.table_transform.copy()
            current = self.k.matrix(measured[side*8:side*8+7], side)
            if np.linalg.norm(position-current[:3, 3]) > .65:
                raise ValueError('Model waypoint farther than 65 cm; split it into observed segments')
            if context == 'grasp' and 'target' not in p:
                raise ValueError('grasp requires target_index from the current observation')
            self.servo._stage('model_control')
            self.servo.command = self.command.copy()
        elif request['op'] == 'set_gripper':
            value = float(request['value'])
            if not np.isfinite(value) or not self.low[side*8+7] <= value <= self.high[side*8+7]:
                raise ValueError('Gripper value outside native SDK limits')
            if value > .85:
                if context not in ('grasp','handover'):
                    raise ValueError('Closing requires grasp or handover context')
                if self.geometry[side] is not None:
                    raise RuntimeError('Already holding a payload')
                if context == 'handover':
                    giver = 1-side
                    box = self.world_box(giver, measured)
                    if box is None:
                        raise RuntimeError('Handover requires another hand holding the payload')
                    p['handover_box'] = box
                elif 'target' not in p:
                    raise ValueError('Closing requires an observed target_index')
                self._grasp_envelope(p, measured)
            elif value < .15 and self.geometry[side] is not None:
                if not p.get('release_authorized'):
                    self.observe_payload(side, measured, snapshot)
                    self._release_check(p, measured)
            else:
                if value >= .15:
                    raise ValueError('Use open <0.15 or close >0.85, not ambiguous intermediate targets')
            p['value'] = value
        else:
            raise ValueError('Only move_ee/set_gripper are primitives')
        if 'target' in p:
            self.selected_targets[side] = dict(observation_id=p['target_observation_id'],
                **{key:copy.deepcopy(p[key]) for key in ('target','plane','roof_reference','target_seen_at')})
        self.active = p
        return dict(status='running', context=context, side=request['side'],
                    target_observation_id=p.get('target_observation_id'))

    def _grasp_envelope(self, p, measured):
        side = p['side']
        pose = self.k.fk(measured)
        pose = pose.left_pose if side == 0 else pose.right_pose
        if 'handover_box' in p:
            center, rotation, extent = p['handover_box']
            local = rotation.inv().apply(np.asarray(pose.pos_xyz)-center)
            if np.any(abs(local) > extent/2+.012):
                raise RuntimeError('Receiving TCP is outside the held parcel envelope')
            return
        target = p['target']
        major = np.asarray(target.get('major_axis_xy', [1., 0.]), float)
        major /= np.linalg.norm(major)
        center = np.asarray(target.get('oriented_center_base_m', target['position_base_m']), float).copy()
        height = target['top_z_base_m']-p['plane']
        center[2] = p['plane']+height/2
        size = np.asarray(target.get('oriented_extent_m', target['extent_xy_m']))
        relative = np.asarray(pose.pos_xyz)-center
        if not (abs(relative[:2]@major) <= size[0]/2+.012 and
                abs(relative[:2]@np.array([-major[1], major[0]])) <= size[1]/2+.012 and
                abs(relative[2]) <= height/2+.015):
            raise RuntimeError('TCP outside original observed grasp envelope')
        p['target_geometry'] = self.observers[side]._geometry(target, pose, p['plane'])

    def _release_check(self, p, measured):
        side, context = p['side'], p['context']
        box = self.world_box(side, measured)
        if box is None:
            return
        corners = self.corners(box)
        if context == 'handover':
            if self.geometry[1-side] is None or not .2 < measured[(1-side)*8+7] < .85:
                raise RuntimeError('Receiving hand has no confirmed contact')
            return
        if context == 'bin':
            if self.rim_seen_at is None or time.monotonic()-self.rim_seen_at > 5.:
                raise RuntimeError('Bin rim observation expired; re-observe before release')
            if self.rim is None or not inside_bin(corners, self.rim, .025) or min(corners[:, 2]) <= self.rim['rim_z']+.025:
                raise RuntimeError('Payload does not clear observed bin rim and original 25 mm margin')
            return
        if context != 'place' or self.support is None:
            raise RuntimeError('Held release requires place, bin, or confirmed handover context')
        if self.support_seen_at is None or time.monotonic()-self.support_seen_at > 5.:
            raise RuntimeError('Support and table-reference evidence expired')
        evidence = self.payload_seen[side]
        if not evidence or not evidence.get('orientation_confirmed'):
            raise RuntimeError('Held payload roof is not visually confirmed')
        height = self.support['position_base_m'][2]
        gap = evidence['bottom_base_m']-height
        config = self.config['visual_residual']
        if not -.003 <= gap <= config['max_release_gap_m']:
            raise RuntimeError(f'Observed support gap {gap:.4f} m outside original release gate')
        if evidence['tilt_degrees'] > config['level_tolerance_degrees']:
            raise RuntimeError('Payload tilt exceeds original release gate')
        if not self.observers[side]._fit(*box, self.support):
            raise RuntimeError('Payload footprint lacks original 25 mm support margin')
        bottom = box[0]+box[1].apply([[x, y, -box[2][2]/2]
            for x in (-box[2][0]/2, box[2][0]/2) for y in (-box[2][1]/2, box[2][1]/2)])
        if np.ptp(bottom[:, 2]) > config['max_bottom_height_spread_m']:
            raise RuntimeError('Payload bottom spread exceeds original release gate')

    def _guard(self, p, measured, command, snapshot):
        if np.any(command < self.low) or np.any(command > self.high):
            raise RuntimeError('Command outside original physical limits')
        for checked_side in (0,1):
            start = checked_side*8
            key = 'max_loaded_tracking_compensation_rad' if self.geometry[checked_side] is not None else 'max_tracking_compensation_rad'
            cap = self.config['cartesian_alignment'][key]
            if np.max(abs(command[start:start+7]-measured[start:start+7])) > cap:
                raise RuntimeError('Pending command exceeds original tracking compensation bound')
        max_step = self.config['visual_cycle'].get('max_joint_step_rad', .012)
        if np.max(abs(command[JOINTS]-self.command[JOINTS])) > max_step+1e-8:
            raise RuntimeError('Command exceeds original joint increment')
        side = p['side']; start = side*8
        q, goal = measured[start:start+7], command[start:start+7]
        plane = snapshot['table_z_base_m']
        if not np.isfinite(plane):
            raise RuntimeError('Observed support plane missing')
        release = self.releases[side]
        if release is not None:
            grip = start+7
            unchanged = np.arange(16) != grip
            if (p['request']['op'] != 'set_gripper' or not p.get('release_authorized') or
                    p['context'] != release['context'] or p['value'] >= .15 or
                    not np.array_equal(command[unchanged], release['reference'][unchanged]) or
                    command[grip] > self.command[grip]+1e-9):
                raise RuntimeError('Authorized release may only open fingers while holding arm references')
            # All release gates and the held sweep were checked before opening.
            # The parcel now settles independently; propagating its old rigid
            # attachment with the unloading wrist creates a fictitious sweep.
            # Joint limits, tracking bounds, increment and fresh sensor checks
            # still apply; no Cartesian integration is allowed in this phase.
            return
        attached_index = None
        # Associate the current image once against measured payload pose.
        # A future sweep pose must not turn the attached parcel into an obstacle.
        observed_box = self.world_box(side, measured)
        if observed_box is not None:
            observed_corners = self.corners(observed_box)
            # Exempt at most one uniquely associated attached component.
            # Nearby unrelated/ambiguous components remain obstacles.
            associations = []
            expected_top = max(observed_corners[:, 2])
            segmentation_ceiling = plane+self.config['cartesian_alignment']['max_object_height_m']
            if expected_top > segmentation_ceiling:
                evidence = self.payload_seen[side]
                seen_at = self.payload_visual_time[side]
                if (not evidence or not evidence.get('orientation_confirmed') or
                        seen_at is None or time.monotonic()-seen_at > 5.):
                    raise RuntimeError('Clipped attached candidate requires fresh confirmed payload roof')
                # Global components are height-censored; keep the full payload
                # box for every collision check and only censor this comparison.
                expected_top = segmentation_ceiling
            for index,target in enumerate(snapshot['candidates']):
                distance = np.linalg.norm(np.asarray(target['position_base_m'])[:2]-observed_box[0][:2])
                ratio = np.asarray(target.get('oriented_extent_m',target['extent_xy_m']))/observed_box[2][:2]
                if distance < .04 and np.all((ratio >= .70) & (ratio <= 1.4)) and abs(target['top_z_base_m']-expected_top) < .03:
                    associations.append((float(distance),index))
            associations.sort()
            attached_index = (associations[0][1] if len(associations)==1 or
                (len(associations)>1 and associations[1][0]-associations[0][0] > .02) else None)
        for alpha in (0., .5, 1.):
            state = measured.copy(); state[start:start+7] = q+alpha*(goal-q)
            tool = self.k.matrix(state[start:start+7], side)
            if tool[2, 3] < plane+.003:
                if self.table_polygon is None or self.table_seen_at is None or time.monotonic()-self.table_seen_at > 5.:
                    raise ObservationBlocked('Fresh table boundary unavailable for below-table-height motion')
                if self._table_contains(tool[:2,3], -.06):
                    raise RuntimeError('TCP would penetrate observed tabletop or edge margin')
            box = self.world_box(side, state)
            if box is not None:
                corners = self.corners(box)
                if min(corners[:, 2]) < plane-.003:
                    if self.table_polygon is None or any(self._table_contains(pt[:2], -.003)
                                                          for pt in corners if pt[2] < plane-.003):
                        raise RuntimeError('Payload would penetrate observed table')
                if self.rim is not None and np.linalg.norm(box[0][:2]-self.rim['center']) < .45:
                    polygon = np.asarray(self.rim['corners'], np.float32)
                    overlap = any(cv2.pointPolygonTest(polygon, tuple(pt), True) >= -.025 for pt in corners[:, :2])
                    if overlap and not inside_bin(corners, self.rim, .025) and min(corners[:, 2]) < self.rim['rim_z']+.025:
                        raise RuntimeError('Payload would cross bin wall below rim clearance')
                for index,target in enumerate(snapshot['candidates']):
                    center = np.asarray(target['position_base_m']).copy()
                    if index == attached_index:
                        continue
                    height = target['top_z_base_m']-plane
                    center[2] = plane+height/2
                    axis = np.asarray(target.get('major_axis_xy', [1., 0.])); axis /= np.linalg.norm(axis)
                    rot = np.column_stack(([*axis,0.],[-axis[1],axis[0],0.],[0.,0.,1.]))
                    extent = np.r_[target.get('oriented_extent_m',target['extent_xy_m']),height]
                    if boxes_overlap(box[0],box[1].as_matrix(),box[2]/2+.003,center,rot,extent/2):
                        raise ObservationBlocked('Carried payload intersects observed candidate: '+str(plain(dict(candidate_index=index, attached_index=attached_index, candidate=target, measured_payload=observed_box[0], expected_candidate_top=expected_top, sweep_fraction=alpha))))
        if self.geometry[side] is None and p['request']['op'] == 'move_ee':
            for index,target in enumerate(snapshot['candidates']):
                audit = sampled_sweep(self.k, q, goal, target, plane, side=side)
                if not audit['clear']:
                    import json
                    detail = plain(dict(candidate_index=index,target=target,plane=plane,audit=audit,
                        source_stamp_s=snapshot['stamp'],measured_joints=q,proposed_joints=goal,
                        measured_tcp=self.k.matrix(q,side),proposed_tcp=self.k.matrix(goal,side)))
                    raise RuntimeError('Original open-finger sweep rejected motion: '+json.dumps(detail))
        if 'target' in p and self.geometry[side] is None:
            target = p['target']
            displacement = ResidualAssist._parcel_displacement(snapshot, target)
            if displacement is None:
                motion = roof_edge_motion(p['roof_reference'], roof_edges(snapshot,target))
                displacement = None if motion is None else motion['displacement_m']
            if displacement is not None:
                if displacement > .01:
                    raise RuntimeError('Observed target moved over original 10 mm limit before contact')
                p['target_seen_at'] = time.monotonic()
                if self.selected_targets[side] is not None:
                    self.selected_targets[side]['target_seen_at'] = p['target_seen_at']
            elif p['request']['op'] == 'move_ee' or time.monotonic()-p['target_seen_at'] > 5.:
                raise ObservationBlocked('Target identity/roof evidence unavailable; re-observe before approach')

    def step(self, measured, snapshot):
        p = self.active
        try:
            command, info = self._step(measured, snapshot)
        except ObservationBlocked as exc:
            if p is None:
                raise
            # _servo may have integrated a proposal, but _guard rejected it.
            # Roll that proposal back; runtime continues the last checked hold.
            self.servo.command = self.command.copy()
            p['stable'] = 0
            since = p.setdefault('observation_blocked_since', time.monotonic())
            waiting = time.monotonic()-since
            if waiting >= 5.:
                raise RuntimeError('Observation remained blocked for five seconds: '+str(exc)) from exc
            return self.command.copy(), dict(status='running', steps=p['steps'], context=p['context'],
                phase='waiting_for_clear_observation', holding_reference=True,
                waiting_s=waiting, reason=str(exc))
        if p is not None and 'observation_blocked_since' in p:
            waited = time.monotonic()-p.pop('observation_blocked_since')
            p['observation_hold_s'] = p.get('observation_hold_s', 0.)+waited
            info['resumed_after_observation_hold_s'] = waited
        if p is not None:
            info['observation_hold_s'] = p.get('observation_hold_s', 0.)
        return command, info

    def _step(self, measured, snapshot):
        measured = action_vector(measured)
        fresh = self.update_scene(snapshot)
        p = self.active
        if p is None:
            return self.command.copy(), dict(status='idle')
        p['steps'] += 1
        if self.last_feedback is not None and np.array_equal(self.last_feedback, measured):
            self.frozen_steps += 1
        else:
            self.frozen_steps = 0
        self.last_feedback = measured.copy()
        if self.frozen_steps >= 20 and max(abs(self.command[JOINTS]-measured[JOINTS])) > .01:
            raise RuntimeError('Feedback frozen while command differs')
        for held in (0, 1):
            if self.geometry[held] is not None:
                releasing = p['request']['op'] == 'set_gripper' and p['side'] == held and p['value'] < .15
                self.payload_lost[held] = (self.payload_lost[held]+1 if not releasing and not .2 < measured[held*8+7] < .85 else 0)
                if self.payload_lost[held] >= 4:
                    raise RuntimeError('Original payload-contact check failed')
                if fresh and self.releases[held] is None:
                    self.observe_payload(held, measured, snapshot)
                seen_at = self.payload_visual_time[held]
                if seen_at is not None and time.monotonic()-seen_at > 5. and not releasing:
                    raise RuntimeError('Payload unseen for over five seconds; re-observe before further motion')
        side = p['side']; grip = side*8+7
        command = self.command.copy()
        tool = self.k.matrix(measured[side*8:side*8+7], side)
        movement = np.inf if p['previous'] is None else np.linalg.norm(tool[:3,3]-p['previous'][:3,3])
        angular_motion = np.inf if p['previous'] is None else (Rotation.from_matrix(tool[:3,:3])*Rotation.from_matrix(p['previous'][:3,:3]).inv()).magnitude()
        p['previous'] = tool.copy()
        info = dict(status='running', steps=p['steps'], context=p['context'])
        if p['request']['op'] == 'move_ee':
            if p['reference'] == 'table':
                if self.table_seen_at is None or time.monotonic()-self.table_seen_at > 5.:
                    raise ObservationBlocked('Table-referenced goal requires fresh verified depth edges')
                displacement = np.max(np.linalg.norm(self.table_polygon-p['goal_table_polygon'], axis=1))
                height_change = abs(self.table_transform[2,3]-p['goal_table_transform'][2,3])
                if displacement > .01 or height_change > .005:
                    raise ObservationBlocked('Table-referenced goal change exceeds 10 mm XY or 5 mm height')
                p['goal'] = self.table_transform @ p['table_goal']
                p['goal_table_polygon'] = self.table_polygon.copy()
                p['goal_table_transform'] = self.table_transform.copy()
                info.update(reference='table', goal_position_base_m=p['goal'][:3,3].tolist())
            self.servo.command = self.command.copy()
            pos, angle, _ = self.servo._servo(measured, side, p['goal'], coupled_pose_request=True)
            command = self.servo.command.copy()
            arrived = pos <= p['position_tolerance_m'] and angle <= p['angle_tolerance_rad'] and movement <= .004
            p['stable'] = p['stable']+1 if arrived and fresh else (p['stable'] if arrived else 0)
            info.update(position_error_m=pos, angle_error_rad=angle,
                        position_tolerance_m=p['position_tolerance_m'], angle_tolerance_rad=p['angle_tolerance_rad'])
            if p['stable'] >= 3:
                info['status'] = 'reached'
                command = self.command.copy()
        else:
            value = p['value']
            command[grip] += np.clip(value-command[grip], -.04, .04)
            if value > .85:
                stationary = p['previous_grip'] is not None and abs(measured[grip]-p['previous_grip']) < .025
                contact = .2 < measured[grip] < .85 and stationary and command[grip] >= value-1e-6
                settled = movement <= .004 and angular_motion <= np.deg2rad(1)
                p['stable'] = p['stable']+1 if contact and settled and fresh else (p['stable'] if contact and settled else 0)
                self._grasp_envelope(p, measured)
                if p['stable'] >= 3:
                    pose = self.k.fk(measured); pose = pose.left_pose if side == 0 else pose.right_pose
                    if 'handover_box' in p:
                        center, rotation, extent = self.world_box(1-side, measured)
                        local = Rotation.from_quat(pose.quat_xyzw)
                        self.geometry[side] = dict(center=local.inv().apply(center-pose.pos_xyz),rotation=local.inv()*rotation,extent=extent.copy())
                    else:
                        self.geometry[side] = self.observers[side]._geometry(p['target'], pose, p['plane'])
                    self.payload_visual_time[side] = time.monotonic()
                    info['status'] = 'reached'; info['contact_confirmed'] = True
            else:
                if self.geometry[side] is not None and not p.get('release_authorized'):
                    self._release_check(p, measured)
                    if movement > .004 or angular_motion > np.deg2rad(1):
                        command[grip] = self.command[grip]
                    else:
                        p['stable'] = p['stable']+1 if fresh else p['stable']
                        if p['stable'] < 3:
                            command[grip] = self.command[grip]
                        else:
                            self._guard(p, measured, self.command.copy(), snapshot)
                            self.releases[side] = dict(context=p['context'],
                                command_id=p['request']['command_id'],
                                authorized_stamp=snapshot['stamp'],
                                reference=self.command.copy(),
                                payload_box=self.corners(self.world_box(side, measured)),
                                visual_evidence=copy.deepcopy(self.payload_seen[side]))
                            p['release_authorized'] = True
                if measured[grip] < .15 and abs(command[grip]-value) < 1e-6:
                    info['status'] = 'reached'
                info['release_authorized'] = bool(p.get('release_authorized'))
            p['previous_grip'] = float(measured[grip])
            info['gripper_feedback'] = float(measured[grip])
        self._guard(p, measured, command, snapshot)
        if (info['status'] == 'reached' and p['request']['op'] == 'set_gripper' and
                p['value'] < .15 and self.geometry[side] is not None):
            release = self.releases[side]
            self.last_release = plain(dict(context=p['context'], side=('left','right')[side],
                stamp=snapshot['stamp'], authorized_stamp=release['authorized_stamp'],
                payload_box=release['payload_box'], visual_evidence=release['visual_evidence']))
            self.geometry[side] = None
            self.releases[side] = None
            self.selected_targets[side] = None
        self.sent(command)
        if info['status'] == 'reached':
            self.active = None
        return command, plain(info)
