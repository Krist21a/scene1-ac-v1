"""Bounded visual corrections around the original ACT commands.

Local grasp/release checks freeze policy selection. Their pose corrections
retain the original ACT reference, physical joint limits and per-step limits.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear

from .motion import action_vector
from .settlement import SettlementMonitor, resting_observation, track_resting_roof
from .clearance import open_fingers_clear
from .release_retreat import axis_retreat_goal
from .retreat_certificate import certify_retreat_execution
from .command_sweep import sampled_sweep
from .retreat_projection import project_retreat_reference

FAST_PROJECTION = False


class ResidualAssist:
    def __init__(self, config, kinematics, low, high):
        self.config, self.kinematics = config, kinematics
        self.low, self.high = action_vector(low), action_vector(high)
        if hasattr(kinematics,'limits'):
            self.low=self.low.copy(); self.high=self.high.copy()
            for side,start in enumerate((0,8)):
                self.low[start:start+7]=np.maximum(self.low[start:start+7],kinematics.limits[side][:,0])
                self.high[start:start+7]=np.minimum(self.high[start:start+7],kinematics.limits[side][:,1])
        self.pending = None
        self.held_action = None
        self.last_sent = None
        self.closed = [False, False]
        self.release_ready = [False, False]
        self.approach = [None, None]
        self.geometry = [None, None]
        self.closing_geometry = [None, None]
        self.contact_frames = [0, 0]
        self.closing_frames = [0, 0]
        self.ready_close = [True, True]
        self.opening = [False, False]
        self.previous_fingers = [None, None]
        self.offset = np.zeros(16)
        self.surfaces = []
        self.targets = []
        self.weighing_target = None
        self.release_countdown = 0
        self.last_tool = [None, None]
        self.wrist_projector = None
        self.wrist_config = {}
        self.wrist_detections = {}
        self.wrist_image_wh = {}
        self.wrist_images = {}
        self.handover_camera_ready = False
        self.unwinding_placement = False
        self.regrasp_complete = False
        self.first_grasp_complete = False

    def observe_command(self, command):
        self.last_sent = action_vector(command)

    def observe_wrist(self, detections, image_wh, images=None):
        self.wrist_detections = detections
        self.wrist_image_wh = image_wh
        self.wrist_images = images or {}

    def _wrist_measure(self, side, target_base, measured, regrasp=False, target=None):
        name = ('left', 'right')[side]
        if self.wrist_projector is None or name not in self.wrist_image_wh:
            return None
        if regrasp and self.wrist_config.get('require_edges', False):
            if target is None or name not in self.wrist_images:
                return None
            method = (self.wrist_projector.measure_outline if self.wrist_config.get('use_parcel_outline',False)
                      else self.wrist_projector.measure_edges)
            return method(side, target, measured, self.wrist_images[name])
        inputs = (side, target_base, measured,
                  self.wrist_detections.get(name, []), self.wrist_image_wh[name])
        if regrasp:
            return self.wrist_projector.measure(*inputs,
                max_center_error_px=self.wrist_config.get('max_regrasp_center_error_px'))
        return self.wrist_projector.measure(*inputs)

    @staticmethod
    def _wrist_grasp_target(target):
        # After release the lower half is hidden by the pad and open fingers.
        # Project the visible top face for RGB association.
        point = np.asarray(target['position_base_m'], float).copy()
        point[2] = target['top_z_base_m']
        return point

    def restore_regrasp(self, records, measured, snapshot, arm_command=None,
                        wrist_roll_degrees=0.):
        """Continue a released parcel with the same bounded ACT residual."""
        if arm_command is not None or wrist_roll_degrees:
            raise ValueError('Residual regrasp does not replace the ACT arm path')
        if not records or records[-1]['sent_action'][15] > .25:
            raise RuntimeError('Regrasp source does not have an open right gripper')
        if not any(r['cartesian_alignment'].get('state') == 'weighing_release'
                   for r in records):
            raise RuntimeError('Regrasp source has no confirmed weighing release')
        pad_records = [r['cartesian_alignment']['target_base_m'] for r in records
                       if r['cartesian_alignment'].get('target_base_m') is not None]
        if not pad_records:
            raise RuntimeError('Regrasp source has no observed weighing target')
        pad = np.asarray(pad_records[-1], float)
        surfaces = snapshot.get('surfaces', [])
        if not surfaces:
            raise RuntimeError('Weighing pad is not visible for regrasp')
        surface = min(surfaces, key=lambda s:
            np.linalg.norm(np.asarray(s['position_base_m'])[:2]-pad[:2]))
        if np.linalg.norm(np.asarray(surface['position_base_m'])[:2]-pad[:2]) > .12:
            raise RuntimeError('Observed weighing pad differs from released pad')
        candidates = [c for c in snapshot['candidates'] if
                      np.linalg.norm(np.asarray(c['position_base_m'])[:2]-pad[:2]) < .12]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one parcel on weighing pad, found {len(candidates)}')
        sent = action_vector(records[-1]['sent_action'])
        act = action_vector(records[-1]['act_action'])
        offset = sent[8:15]-act[8:15]
        if np.max(np.abs(offset)) > self.config['max_joint_offset_rad']+.01:
            raise RuntimeError('Previous ACT residual exceeds the regrasp cap')
        self.offset[8:15] = offset
        self.last_sent = sent.copy()
        self.weighing_target = np.asarray(surface['position_base_m'], float)
        base = sent.copy(); base[8:15] -= offset; base[15] = 0.
        self.approach[1] = dict(target=copy.deepcopy(candidates[0]),
                                plane=snapshot['table_z_base_m'], aligned=False)
        self._begin('pregrasp', 1, base, target=copy.deepcopy(candidates[0]),
                    plane=snapshot['table_z_base_m'])

    @staticmethod
    def _pose(poses, side):
        return poses.left_pose if side == 0 else poses.right_pose

    def _remember(self, snapshot):
        if snapshot is None:
            return
        if snapshot['candidates']:
            self.targets = copy.deepcopy(snapshot['candidates'])
        for surface in snapshot.get('surfaces', []):
            if surface.get('partial') or min(surface['extent_xy_m']) < .145:
                continue
            matching = [i for i, old in enumerate(self.surfaces) if np.linalg.norm(
                np.asarray(old['position_base_m'])[:2] - np.asarray(surface['position_base_m'])[:2]) < .05]
            if matching:
                # An occupied pad must not replace its full visible boundary.
                i = matching[0]
                if np.prod(surface['extent_xy_m']) >= .9 * np.prod(self.surfaces[i]['extent_xy_m']):
                    self.surfaces[i] = copy.deepcopy(surface)
            else:
                self.surfaces.append(copy.deepcopy(surface))

    def _geometry(self, target, pose, plane):
        major = np.asarray(target.get('major_axis_xy', [1., 0.]))
        major /= np.linalg.norm(major)
        rotation = Rotation.from_matrix(np.column_stack(([*major, 0], [-major[1], major[0], 0], [0, 0, 1])))
        height = target.get('top_z_base_m', target['position_base_m'][2] + .04) - plane
        center = np.asarray(target['position_base_m']).copy()
        center[2] = plane + height / 2  # visible-point median is biased toward the top face
        tool = Rotation.from_quat(pose.quat_xyzw)
        return dict(center=tool.inv().apply(center - pose.pos_xyz),
                    rotation=tool.inv() * rotation,
                    extent=np.r_[target.get('oriented_extent_m', target.get('extent_xy_m', [.11, .04])), height])

    def _world_geometry(self, side, poses):
        geometry = self.geometry[side]
        pose = self._pose(poses, side)
        rotation = Rotation.from_quat(pose.quat_xyzw)
        return (np.asarray(pose.pos_xyz) + rotation.apply(geometry['center']),
                rotation * geometry['rotation'], geometry['extent'])

    def _grasp_goal(self, target, pose, plane=None):
        point = np.asarray(target['position_base_m']).copy()
        if plane is not None:
            point[2] = (plane + target.get('top_z_base_m', point[2] + .04)) / 2
        if target.get('grasp_goal_xy') is not None:
            point[:2] = target['grasp_goal_xy']
            return self._center_grasp_transverse(target,point)
        forward = -Rotation.from_quat(pose.quat_xyzw).as_matrix()[:2, 2]
        if np.linalg.norm(forward) > .65:
            forward /= np.linalg.norm(forward)
            major = np.asarray(target.get('major_axis_xy', [1., 0.]))
            minor = np.array([-major[1], major[0]])
            extent = target.get('oriented_extent_m', target.get('extent_xy_m', [.11, .04]))
            half = (abs(forward @ major) * extent[0] + abs(forward @ minor) * extent[1]) / 2
            point[:2] -= forward * max(0., half - .02)
        return self._center_grasp_transverse(target,point)

    def _initial_grasp_rotation(self, target, pose):
        pitch = self.wrist_config.get('first_grasp_pitch_degrees')
        if pitch is None:
            return None
        if not np.isfinite(pitch) or not 12.5 <= pitch <= 17.5:
            raise ValueError('Initial grasp pitch must stay within established visual pickup range')
        major = np.asarray(target['major_axis_xy'], float)
        if major.shape != (2,) or not np.isfinite(major).all() or np.linalg.norm(major) < 1e-3:
            raise RuntimeError('Initial grasp orientation requires a valid parcel axis')
        major = major/np.linalg.norm(major)
        current_forward = -Rotation.from_quat(pose.quat_xyzw).as_matrix()[:2,2]
        forward = major if major@current_forward >= 0. else -major
        closing = np.r_[-forward[1],forward[0],0.]
        angle = np.deg2rad(pitch)
        z_axis = -np.r_[forward*np.cos(angle),-np.sin(angle)]
        return Rotation.from_matrix(np.column_stack((closing,np.cross(z_axis,closing),z_axis)))

    def _center_grasp_transverse(self, target, point):
        if self.wrist_config.get('grasp_align_parcel_yaw',False) and 'support_z_base_m' not in target:
            major=np.asarray(target['major_axis_xy'],float)
            if major.shape!=(2,) or not np.isfinite(major).all() or np.linalg.norm(major)<1e-3:
                raise RuntimeError('Grasp centering requires a valid parcel axis')
            minor=np.array([-major[1],major[0]])/np.linalg.norm(major)
            center=np.asarray(target.get('oriented_center_base_m',target['position_base_m']),float)
            point=point.copy()
            point[:2]+=minor*(minor@(center[:2]-point[:2]))
        return point

    @staticmethod
    def _parcel_displacement(snapshot, target):
        if snapshot is None:
            return None
        origin = np.asarray(target['position_base_m'])[:2]
        expected = np.asarray(target.get('oriented_extent_m', target['extent_xy_m']))
        if '_reference_centers_xy' not in target:
            target['_reference_centers_xy'] = [np.asarray(c['position_base_m'])[:2].tolist()
                for c in snapshot['candidates']
                if np.linalg.norm(np.asarray(c['position_base_m'])[:2] - origin) > .08]
        # Fixed parcels share apparent motion when the standing robot's base
        # drifts. Subtract that motion only when two independent references
        # agree; a single moving parcel must still trip the collision guard.
        shifts = []
        for reference in target['_reference_centers_xy']:
            reference = np.asarray(reference)
            matches = [np.asarray(c['position_base_m'])[:2] - reference
                       for c in snapshot['candidates']
                       if np.linalg.norm(np.asarray(c['position_base_m'])[:2] - reference) < .05]
            if matches:
                shifts.append(min(matches, key=np.linalg.norm))
        common_shift = np.zeros(2)
        if len(shifts) >= 2:
            median = np.median(shifts, axis=0)
            if max(np.linalg.norm(shift - median) for shift in shifts) <= .008:
                common_shift = median
        if 'support_z_base_m' in target and 'depth_m' in snapshot:
            roof = track_resting_roof(snapshot,target,origin+common_shift)
            if roof is None:
                return None
            return float(np.linalg.norm(np.asarray(roof['center_xy_m'])-origin-common_shift))
        distances = []
        for candidate in snapshot['candidates']:
            size = np.asarray(candidate.get('oriented_extent_m', candidate['extent_xy_m']))
            ratio = size / expected
            distance = np.linalg.norm(np.asarray(candidate['position_base_m'])[:2]
                                      - origin - common_shift)
            # A hand hiding one end shortens the measured box and shifts its
            # visible centroid. It cannot prove that the box itself moved.
            # A 15% shortened 11 cm box shifts its visible centroid by up
            # to 8 mm, enough to spuriously cross the 10 mm motion guard.
            # Only nearly complete extents can establish centroid motion.
            if distance < .04 and .95 <= ratio[0] <= 1.25 and .9 <= ratio[1] <= 1.3:
                distances.append(distance)
        return min(distances) if distances else None

    def _correct(self, base, measured, side, xyz_error, angular_error=None, level_axis=None,
                 step_scale=1., offset_cap_rad=None, translation_limit_m=None, view_target=None, gain_scale=1., level_axis_weight=.1):
        """Damped local correction, with optional bottom-leveling rotation."""
        start = (0, 8)[side]
        error = np.asarray(xyz_error, float).copy()
        if len(error) == 2:
            error = np.r_[error, 0.]
        limit = (self.config['max_translation_m'] if translation_limit_m is None
                 else translation_limit_m)
        error *= min(1., limit / max(np.linalg.norm(error), 1e-9))
        q = measured[start:start + 7].copy()
        original = self.kinematics.matrix(q, side)
        rotation = Rotation.from_matrix(original[:3, :3])
        jacobian = np.zeros((6, 7))
        for joint in range(7):
            shifted = q.copy(); shifted[joint] += 1e-4
            matrix = self.kinematics.matrix(shifted, side)
            jacobian[:3, joint] = (matrix[:3, 3] - original[:3, 3]) / 1e-4
            jacobian[3:, joint] = .15 * (rotation.inv() * Rotation.from_matrix(matrix[:3, :3])).as_rotvec() / 1e-4
        angular = np.zeros(3) if angular_error is None else np.asarray(angular_error, float).copy()
        angular *= min(1., np.deg2rad(3) / max(np.linalg.norm(angular), 1e-9))
        if view_target is not None:
            wh = self.wrist_image_wh[('left','right')[side]]
            projected = self.wrist_projector.project(side,view_target,measured,wh,allow_outside=True)
            if projected is None:
                raise RuntimeError('Wrist target has invalid depth for observation correction')
            image_jacobian = np.zeros((2,7))
            for joint in range(7):
                shifted = measured.copy(); shifted[start+joint] += 1e-4
                shifted_projection = self.wrist_projector.project(side,view_target,shifted,wh,allow_outside=True)
                if shifted_projection is None:
                    raise RuntimeError('Wrist projection lost while checking local motion')
                image_jacobian[:,joint] = (shifted_projection['uv']-projected['uv']) / projected['focal_px'] / 1e-4
            view_error = (np.array([wh[0]*.5,wh[1]*.62])-projected['uv'])/projected['focal_px']
            view_error *= min(1.,.052/max(np.linalg.norm(view_error),1e-9))
            # Use the actual camera projection derivative. With the camera
            # behind the TCP, wrist rotation also translates the camera; a
            # pure orientation Jacobian gives the wrong near-field response.
            jacobian = np.vstack((jacobian[:3],.15*image_jacobian))
            requested = np.r_[error,.15*view_error]
        elif level_axis is None:
            requested = np.r_[error, .15 * rotation.inv().apply(angular)]
        else:
            axis = np.asarray(level_axis,float).copy(); axis /= np.linalg.norm(axis)
            # Leveling constrains two rotation directions. A weak spin cost
            # permits a small yaw change when it helps respect joint limits.
            if not np.isfinite(level_axis_weight) or not 0. < level_axis_weight <= 1.:
                raise ValueError('Axis weight must be finite in (0,1]')
            projection = np.eye(3) - (1.-level_axis_weight)*np.outer(axis,axis)
            jacobian[3:] = projection @ rotation.as_matrix() @ jacobian[3:]
            requested = np.r_[error,.15*projection@angular]
        offset = self.offset[start:start+7]
        cap = self.config['max_joint_offset_rad'] if offset_cap_rad is None else offset_cap_rad
        if offset_cap_rad is None and side == 1 and self.weighing_target is not None and not self.closed[1]:
            cap = self.wrist_config.get('max_regrasp_joint_offset_rad', cap)
        step = self.config['max_joint_step_rad'] * step_scale
        lower = np.maximum(-step, -cap-offset)
        upper = np.minimum(step, cap-offset)
        if self.unwinding_placement and side == 1 and offset_cap_rad is None:
            # Retain the original ACT reference. Existing placement offsets
            # above the normal cap may only shrink, without a command jump or
            # inventing a fresh correction budget around the corrected pose.
            lower = np.maximum(-step, np.minimum(-cap-offset, 0.))
            upper = np.minimum(step, np.maximum(cap-offset, 0.))
        joint_low=self.low[start:start+7].copy(); joint_high=self.high[start:start+7].copy()
        if hasattr(self.kinematics,'limits'):
            joint_low=np.maximum(joint_low,self.kinematics.limits[side][:,0])
            joint_high=np.minimum(joint_high,self.kinematics.limits[side][:,1])
        lower=np.maximum(lower,joint_low-base[start:start+7]-offset)
        upper=np.minimum(upper,joint_high-base[start:start+7]-offset)
        if np.any(lower > upper+1e-10):
            raise RuntimeError('ACT correction cannot respect physical joint limits within the step cap')
        # Solve with the limits included, so a saturated wrist does not absorb
        # all further requests that could be shared by other arm joints.
        matrix=np.vstack((jacobian,.015*np.eye(7)))
        if not np.isfinite(gain_scale) or not 0. < gain_scale <= 1.:
            raise ValueError('Correction gain scale must be finite and in (0, 1]')
        target=np.r_[self.config['gain']*gain_scale*requested,np.zeros(7)]
        free=upper-lower > 1e-10
        change=(lower+upper)/2
        if np.any(free):
            solution=lsq_linear(matrix[:,free],target-matrix[:,~free]@change[~free],
                bounds=(lower[free],upper[free]),method='bvls',tol=1e-10)
            change[free]=solution.x
        self.offset[start:start+7] += change
        if self.unwinding_placement and np.max(np.abs(self.offset[8:15])) <= cap+1e-9:
            self.unwinding_placement = False
        result = base.copy()
        result[start:start + 7] += self.offset[start:start + 7]
        return np.clip(result, self.low, self.high)

    def _retreat_correct(self, base, measured, error, target, plane, motion_config):
        before = self.offset.copy()
        proposed = self._correct(base, measured, 1, error, step_scale=.5)
        proposed[15] = 0.
        if not self.wrist_config.get('release_sweep_projection', False):
            return proposed, None
        if motion_config is None or self.last_sent is None:
            self.offset = before
            raise RuntimeError('Retreat projection requires actual command/filter context')
        cap = self.wrist_config.get('max_regrasp_joint_offset_rad', self.config['max_joint_offset_rad'])
        lower, upper = self.low.copy(), self.high.copy()
        lower[8:15] = np.maximum(lower[8:15], base[8:15]-cap)
        upper[8:15] = np.minimum(upper[8:15], base[8:15]+cap)
        try:
            chosen, audit = project_retreat_reference(self.kinematics, measured, proposed,
                base+before, self.last_sent, lower, upper, self.config['max_joint_step_rad']*.5,
                target, plane, motion_config)
        except Exception:
            self.offset = before
            raise
        self.offset[8:15] = chosen[8:15]-base[8:15]
        return chosen, audit

    def hold_retreat_reference(self, previous):
        """Discard an unsent increment while preserving the original phase clock."""
        if self.pending is None or self.pending['kind'] != 'short_retreat':
            raise RuntimeError('Safe pending hold is limited to active retreat')
        previous = action_vector(previous)
        if np.any(previous < self.low) or np.any(previous > self.high):
            raise RuntimeError('Pending retreat hold is outside physical command bounds')
        if hasattr(self.kinematics, 'limits'):
            if (np.any(previous[8:15] < self.kinematics.limits[1][:,0])
                    or np.any(previous[8:15] > self.kinematics.limits[1][:,1])):
                raise RuntimeError('Pending retreat hold violates robot joint limits')
        if previous[15] > .1:
            raise RuntimeError('Pending retreat hold requires an open gripper')
        self.offset[8:15] = previous[8:15]-self.pending['base'][8:15]
        return previous.copy()

    def camera_target(self, measured):
        if self.pending is None or self.weighing_target is None:
            return None
        kind = self.pending['kind']
        observing_pad = (kind in ('weighing','short_retreat','settle') or
                         (kind in ('pregrasp','grasp') and
                          'support_z_base_m' in self.pending['target']))
        if not observing_pad:
            return None
        target = self.weighing_target + [0, 0, .06]
        if kind == 'weighing' and self.pending['stage'] in ('scan_left', 'scan_right'):
            toward = self.weighing_target[:2] / np.linalg.norm(self.weighing_target[:2])
            lateral = np.array([-toward[1], toward[0]])
            sign = -1 if self.pending['stage'] == 'scan_left' else 1
            target[:2] += sign * .07 * lateral
        return target

    def _begin(self, kind, side, base, **fields):
        self.held_action = base.copy()
        self.pending = dict(kind=kind, side=side, base=base.copy(), steps=0, stable=0, **fields)

    def _fit(self, center, rotation, extent, surface):
        import cv2
        footprint = surface.get('footprint_xy_m')
        if footprint is None:
            return False
        corners = rotation.apply([[x, y, 0] for x in (-extent[0]/2, extent[0]/2)
                                  for y in (-extent[1]/2, extent[1]/2)])[:, :2] + center[:2]
        return all(cv2.pointPolygonTest(np.asarray(footprint, np.float32), tuple(p), True)
                   >= self.config['placement_margin_m'] for p in corners)

    def _view_ready(self, snapshot, measured):
        if snapshot is None or 'camera_to_base' not in snapshot:
            return False
        matrix = snapshot['camera_to_base']
        direction = self.camera_target(measured) - matrix[:3, 3]
        direction /= np.linalg.norm(direction)
        return float(direction @ matrix[:3, 2]) >= np.cos(.08)

    @staticmethod
    def _roof_normal(group):
        """Fit a high support plane without mixing the visible side faces."""
        roof = group[group[:,2] >= np.percentile(group[:,2],60)]
        if len(roof) < 20:
            return None
        rng = np.random.default_rng(4)
        best = None
        for _ in range(80):
            sample = roof[rng.choice(len(roof),3,replace=False)]
            normal = np.cross(sample[1]-sample[0],sample[2]-sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            if abs(normal[2]) < .70:
                continue
            selected = np.abs((roof-sample[0]) @ normal) <= .003
            support = roof[selected]
            if len(support) < 20 or np.min(np.ptp(support[:,:2],axis=0)) < .015:
                continue
            if np.median(support[:,2]) < np.percentile(group[:,2],90)-.012:
                continue
            if best is None or len(support) > len(best):
                best = support
        if best is None:
            return None
        _,_,vectors = np.linalg.svd(best-best.mean(axis=0),full_matrices=False)
        normal = vectors[-1]
        return normal if normal[2] >= 0 else -normal

    def _observe_box(self, snapshot, poses):
        """Use RGB-D around the held box, excluding the wrist behind it."""
        if snapshot is None or 'depth_m' not in snapshot:
            return None
        import cv2
        center, rotation, extent = self._world_geometry(1, poses)
        if FAST_PROJECTION:
            from .fast_components import sampled_geometry
            z, points, hsv = sampled_geometry(snapshot, 2)
        else:
            depth = snapshot['depth_m']; intrinsic = snapshot['intrinsic']
            v, u = np.mgrid[0:depth.shape[0]:2, 0:depth.shape[1]:2]
            v, u = v.ravel(), u.ravel(); z = depth[v, u]
            camera = np.column_stack(((u-intrinsic[0,2])*z/intrinsic[0,0],
                                      (v-intrinsic[1,2])*z/intrinsic[1,1], z))
            matrix = snapshot['camera_to_base']
            points = camera @ matrix[:3,:3].T + matrix[:3,3]
            hsv = cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV)[v,u]
        local = rotation.inv().apply(points-center)
        green = (hsv[:,0] >= 40) & (hsv[:,0] <= 85) & (hsv[:,1] >= 100)
        selected = (z > .1) & np.isfinite(points).all(axis=1) & ~green & (hsv[:,2] >= 65)
        selected &= np.all(np.abs(local) <= extent/2 + .025, axis=1)
        group = points[selected]; local = local[selected]
        if len(group) < 25:
            return None
        low, high = np.percentile(local, [5,95], axis=0)
        if high[0]-low[0] < extent[0]*.75 or high[1]-low[1] < extent[1]*.60:
            return None  # a partial face must not become a new box center
        shift = rotation.apply(np.r_[(low[:2]+high[:2])/2, 0.])
        position_source = 'held_box_roi'
        components = []
        for candidate in snapshot.get('candidates',[]):
            size = np.asarray(candidate.get('oriented_extent_m',candidate['extent_xy_m']))
            ratio = size/extent[:2]
            distance = np.linalg.norm(np.asarray(candidate['position_base_m'])[:2]-center[:2])
            if distance < .04 and .85 <= ratio[0] <= 1.25 and .80 <= ratio[1] <= 1.35:
                components.append((distance,candidate))
        components.sort(key=lambda item:item[0])
        if components and (len(components)==1 or components[1][0]-components[0][0] > .02):
            shift = np.r_[np.asarray(components[0][1]['position_base_m'])[:2]-center[:2],0.]
            position_source = 'rgbd_component'
        if np.linalg.norm(shift) > .025:
            return None
        pose = poses.right_pose; tool = Rotation.from_quat(pose.quat_xyzw)
        self.geometry[1]['center'] += .3 * tool.inv().apply(shift)
        vertical_height = float(np.abs(rotation.as_matrix()[2]) @ extent)
        bottom = float(np.percentile(group[:,2],95)-vertical_height)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(group[:,:2].T))
        major = eigenvectors[:,-1]
        angle = np.arctan2(major[1], major[0])
        heading_error = abs((angle+np.pi/4) % (np.pi/2)-np.pi/4)
        tilt = None
        normal = self._roof_normal(group)
        if normal is not None:
            tilt = float(np.rad2deg(np.arccos(np.clip(abs(normal[2]),0,1))))
        orientation_confirmed = bool(tilt is not None and eigenvalues[1] >= 2*eigenvalues[0])
        orientation_difference = None
        # Every fitted normal needs the same agreement check. Previously
        # fits above 45 degrees skipped this block but remained confirmed,
        # allowing a mismatched face to steer the loaded wrist.
        if orientation_confirmed:
            normal = normal if normal[2] >= 0 else -normal
            axis = np.r_[major, -float(normal[:2] @ major) / normal[2]]
            axis /= np.linalg.norm(axis)
            if float(axis @ rotation.apply([1, 0, 0])) < 0:
                axis = -axis
            observed = Rotation.from_matrix(np.column_stack((axis, np.cross(normal, axis), normal)))
            angular_shift = (rotation.inv() * observed).as_rotvec()
            orientation_difference = float(np.rad2deg(np.linalg.norm(angular_shift)))
            # Update the grasp transform only from an adequately visible box;
            # limit the change from one RGB-D frame to reject wrist faces.
            if np.linalg.norm(angular_shift) <= np.deg2rad(20):
                rotation = rotation * Rotation.from_rotvec(.3 * angular_shift)
                self.geometry[1]['rotation'] = tool.inv() * rotation
                vertical_height = float(np.abs(rotation.as_matrix()[2]) @ extent)
                bottom = float(np.percentile(group[:,2],95)-vertical_height)
            else:
                orientation_confirmed = False
        return dict(points=len(group), bottom_base_m=bottom,
                    position_source=position_source,
                    roof_normal_base=normal.tolist() if tilt is not None else None,
                    heading_error_degrees=float(np.rad2deg(heading_error)),
                    tilt_degrees=tilt, orientation_confirmed=orientation_confirmed,
                    orientation_difference_degrees=orientation_difference)

    def _regrasp_action(self, measured, snapshot, poses, info):
        p = self.pending; pose = poses.right_pose; target = p['target']
        self.command_sweep_context = dict(target=copy.deepcopy(target), plane=p['plane'],
                                          stage=p.get('regrasp_stage', 'level'))
        position = np.asarray(pose.pos_xyz)
        rotation = Rotation.from_quat(pose.quat_xyzw)
        if p.get('regrasp_stage') == 'reobserve':
            # The parcel can settle further while the open hand levels. Use
            # the SAME support/height/tilt/footprint checks on new frames before
            # translating toward it. Do not chase per-frame centroids or reset
            # the original joint-correction or total regrasp step budget.
            result = p['reobserve_hold'].copy(); result[15] = 0.
            p['reobserve_steps'] += 1
            observed = resting_observation(snapshot, p['surface'], p['extent'], self._roof_normal)
            settled = p['reobserve_monitor'].update(observed)
            info.update(regrasp_stage='reobserve', post_level_settlement=settled,
                        reobserve_steps=p['reobserve_steps'])
            if settled['confirmed']:
                refreshed = copy.deepcopy(observed['target'])
                refreshed['support_z_base_m'] = snapshot['table_z_base_m']
                info['post_level_geometry_refresh'] = dict(
                    previous_target=copy.deepcopy(p['target']), refreshed_target=copy.deepcopy(refreshed),
                    xy_shift_m=float(np.linalg.norm(np.asarray(refreshed['position_base_m'])[:2]
                                                   - np.asarray(p['target']['position_base_m'])[:2])),
                    support_reconfirmed=True, total_phase_steps=p['steps'])
                p['target'] = refreshed
                p['plane'] = snapshot['table_z_base_m']
                self.approach[1] = dict(target=copy.deepcopy(refreshed), plane=p['plane'], aligned=False, frames=0)
                for key in ('grasp_rotation', 'grasp_position', 'stand_off_position'):
                    p.pop(key, None)
                p['post_level_geometry_verified'] = True
                p['regrasp_stage'] = 'level'; p['stable'] = 0
            elif p['reobserve_steps'] >= 100:
                raise RuntimeError('Post-level parcel support could not be reconfirmed; gripper kept open')
            if p['steps'] >= self.wrist_config.get('max_level_regrasp_steps', 900):
                raise RuntimeError('Level regrasp pose could not converge within allowed joint correction')
            return result, info
        if 'grasp_rotation' not in p:
            major = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
            major /= np.linalg.norm(major)
            forward = major if major@(np.asarray(target['position_base_m'])[:2]-position[:2]) >= 0 else -major
            closing = np.r_[-forward[1],forward[0],0.]
            pitch = np.deg2rad(self.wrist_config.get('regrasp_pitch_degrees',12.5))
            z_axis = -np.r_[forward*np.cos(pitch),-np.sin(pitch)]
            p['grasp_rotation'] = Rotation.from_matrix(np.column_stack((closing,np.cross(z_axis,closing),z_axis)))
            size = target.get('oriented_extent_m',target['extent_xy_m'])
            xy = np.asarray(target['position_base_m'])[:2]-forward*max(0.,size[0]/2-.02)
            target['grasp_goal_xy'] = xy.tolist()
            p['grasp_position'] = np.r_[xy,(p['plane']+target['top_z_base_m'])/2]
            # Lower outside the front face, then enter horizontally. Moving
            # over the roof first blocks the head view before the wrist can
            # see enough of the roof to verify it.
            stand_off = np.asarray(target['position_base_m'])[:2]-forward*(size[0]/2+.015)
            p['stand_off_position'] = np.r_[stand_off,p['grasp_position'][2]]
            p['view_position'] = position.copy()
            p['regrasp_stage'] = 'level'
        stage = p['regrasp_stage']
        matrix = rotation.as_matrix()
        finger_tilt = float(np.rad2deg(np.arcsin(np.clip(abs(matrix[2,0]),0,1))))
        pitch = float(np.rad2deg(np.arcsin(np.clip(matrix[2,2],-1,1))))
        angular_error = (p['grasp_rotation']*rotation.inv()).as_rotvec()
        orientation_ok = (finger_tilt <= self.wrist_config.get('regrasp_finger_tilt_degrees',5.)
                          and np.linalg.norm(angular_error) <= np.deg2rad(8.))
        goal = p['view_position'].copy()
        if stage in ('align','descend'):
            goal[:2] = p['stand_off_position'][:2]
        if stage == 'insert':
            goal[:2] = p['grasp_position'][:2]
        if stage in ('descend','insert'):
            goal[2] = p['grasp_position'][2]
        wrist = self._wrist_measure(1,self._wrist_grasp_target(target),measured,regrasp=True,target=target)
        displacement = self._parcel_displacement(snapshot,target)
        # The settled head estimate supplies the initial geometry. During
        # regrasp it may be occluded by the arm; visibility from that camera
        # and its partial-roof centroid are diagnostic only. Wrist evidence
        # is required before entering the parcel and again before closing.
        velocity = position-p['last_position'] if p.get('last_position') is not None else np.zeros(3)
        p['last_position'] = position.copy()
        angular_velocity = ((rotation*p['last_rotation'].inv()).as_rotvec()
                            if p.get('last_rotation') is not None else np.zeros(3))
        p['last_rotation'] = rotation
        error = goal-position
        forward = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
        forward /= np.linalg.norm(forward)
        lateral = np.array([-forward[1],forward[0]])
        if stage in ('descend','insert') and wrist is not None:
            # Wrist centering replaces the stale lateral head goal. Adding
            # them creates a biased equilibrium and repeated small reversals.
            side_error = float(np.asarray(wrist['correction_base_xy']) @ lateral)
            if abs(wrist['centering_error_px']) <= self.wrist_config.get('centering_deadband_px',4.):
                side_error = 0.
            error[:2] = forward*(error[:2]@forward) + lateral*side_error
        # Complete leveling before lateral approach or descent. A temporarily
        # lost orientation freezes translation while the fingers re-level.
        correction = error.copy()
        if stage in ('descend','insert'):
            # Keep lateral alignment while lowering. A distant Z goal must
            # not dominate the bounded translation solve and sweep a finger
            # into the side of the parcel.
            descent_request = self.wrist_config.get('regrasp_descent_request_m',.012)
            correction[2] = (np.clip(error[2],-descent_request,descent_request)
                             if stage == 'insert' or np.linalg.norm(error[:2]) <= .008 else 0.)
        if stage == 'insert':
            # Keep height while entering; forward error must never disable Z.
            advance = np.clip(correction[:2]@forward,-.006,.006)
            if abs(error[2]) > .010:
                advance = 0.
            if self.wrist_config.get('regrasp_center_before_advance', False):
                # Recenter with the current wrist view before adding forward
                # demand; the original derivative brake below still applies.
                # Keep the existing sweep, pose, contact and close checks.
                # Admission uses the existing regrasp tolerance. The tighter
                # centering deadband above remains the lateral control target.
                centered = (wrist is not None and
                    abs(wrist['centering_error_px']) <= self.wrist_config.get('regrasp_aligned_error_px', 10.))
                requested_advance = float(advance)
                if not centered:
                    advance = 0.
                info['regrasp_forward_gate'] = dict(centered=bool(centered),
                    requested_before_gate_m=requested_advance,
                    requested_after_gate_m=float(advance),
                    wrist_error_px=None if wrist is None else float(wrist['centering_error_px']))
            correction[:2] = forward*advance + lateral*(correction[:2]@lateral)
        correction -= 3.*velocity
        if stage != 'level' and not orientation_ok:
            correction = -3.*velocity
        missing_insert_view = stage == 'insert' and wrist is None
        p['missing_insert_view_steps'] = p.get('missing_insert_view_steps',0)+1 if missing_insert_view else 0
        if missing_insert_view:
            correction[:2] = -3.*velocity[:2]
            # Stop forward travel, but recover the known safe height if the
            # arm rose and hid the visible edges behind the fingers.
            if p['missing_insert_view_steps'] >= 40:
                raise RuntimeError('Right wrist lost parcel edges during insertion; gripper kept open')
        previous_offset = self.offset.copy()
        step_scale = .6 if stage == 'descend' and orientation_ok and np.linalg.norm(error[:2]) <= .008 else .4
        pitch_weight = self.wrist_config.get('regrasp_insert_pitch_weight',1.)
        if not np.isfinite(pitch_weight) or not 0. < pitch_weight <= 1.:
            raise ValueError('Insertion pitch weight must be finite in (0,1]')
        pitch_relaxed = stage == 'insert' and orientation_ok and pitch_weight < 1.
        pitch_axis = p['grasp_rotation'].as_matrix()[:,0] if pitch_relaxed else None
        result = self._correct(p['base'],measured,1,correction,angular_error-4.*angular_velocity,
            step_scale=step_scale,level_axis=pitch_axis,level_axis_weight=pitch_weight)
        info['insert_pitch_task'] = dict(relaxed=bool(pitch_relaxed),
            weight=float(pitch_weight) if pitch_relaxed else 1.,
            original_orientation_gate_passed=bool(orientation_ok))
        delta = self.offset[8:15]-previous_offset[8:15]
        self.command_sweep_context['increment_end_rad'] = (measured[8:15]+delta).tolist()
        clear = all(open_fingers_clear(self.kinematics.matrix(measured[8:15]+a*delta,1),target,p['plane'])
                    for a in (0.,.5,1.))
        if self.wrist_config.get('command_sweep_enforce', False):
            info['complete_reference_sweep'] = sampled_sweep(
                self.kinematics, measured[8:15], result[8:15], target, p['plane'])
            clear = clear and info['complete_reference_sweep']['clear']
        if not clear:
            self.offset = previous_offset
            result = p['base'].copy()+self.offset
            p['blocked_steps'] = p.get('blocked_steps',0)+1
            if p['blocked_steps'] >= 20 and not clear:
                raise RuntimeError('Open finger sweep intersects observed parcel; regrasp stopped')
        else:
            p['blocked_steps'] = 0
        result[15] = 0.
        geometry_ready = orientation_ok and clear
        if stage == 'level':
            geometry_ready = geometry_ready and np.linalg.norm(position-p['view_position']) <= .015
        else:
            geometry_ready = geometry_ready and np.linalg.norm(error[:2]) <= .008
        if stage in ('descend','insert'):
            geometry_ready = (geometry_ready and abs(error[2]) <= .010 and wrist is not None
                     and abs(wrist['centering_error_px']) <= self.wrist_config.get('regrasp_aligned_error_px',10.))
        if stage == 'insert' and geometry_ready:
            # Stop changing the command while testing that the measured hand
            # has settled; do not keep injecting corrections into the dwell.
            self.offset = previous_offset
            result = p['base'].copy()+self.offset
            result[15] = 0.
        ready = (geometry_ready and np.linalg.norm(velocity) <= (.0015 if stage == 'insert' else .003)
                 and np.linalg.norm(angular_velocity) <= np.deg2rad(1.))
        p['stable'] = p['stable']+1 if ready else 0
        info.update(regrasp_stage=stage, finger_line_tilt_degrees=finger_tilt,
                    approach_pitch_degrees=pitch, orientation_error_degrees=float(np.rad2deg(np.linalg.norm(angular_error))),
                    angular_movement_degrees=float(np.rad2deg(np.linalg.norm(angular_velocity))),
                    xy_error_m=float(np.linalg.norm(error[:2])), z_error_m=float(error[2]),
                    movement_m=float(np.linalg.norm(velocity)), pose_hold=bool(stage == 'insert' and geometry_ready),
                    parcel_displacement_m=displacement, wrist=wrist, open_fingers_clear=bool(clear),
                    head_roof_visible=displacement is not None,
                    stable_steps=p['stable'], offset_rad=self.offset[8:15].tolist())
        if p['stable'] >= 3:
            if stage == 'level':
                if (self.wrist_config.get('regrasp_reobserve_after_level', False)
                        and not p.get('post_level_geometry_verified', False)):
                    if 'surface' not in p or 'extent' not in p:
                        raise RuntimeError('Missing original support geometry for post-level verification')
                    p['regrasp_stage'] = 'reobserve'
                    p['reobserve_monitor'] = SettlementMonitor()
                    p['reobserve_steps'] = 0
                    p['reobserve_hold'] = result.copy()
                else:
                    p['regrasp_stage'] = 'align'
                p['stable'] = 0
            elif stage == 'align':
                p['regrasp_stage'] = 'descend'; p['stable'] = 0
            elif stage == 'descend':
                p['regrasp_stage'] = 'insert'; p['stable'] = 0
            else:
                self.approach[1] = None
                self.ready_close[1] = self.release_ready[1] = False
                self.contact_frames[1] = self.closing_frames[1] = 0
                result[15] = .98
                self.closing_geometry[1] = self._geometry(target,pose,p['plane'])
                self._begin('grasp',1,p['base'],target=copy.deepcopy(target),plane=p['plane'],
                            grasp_z=float(position[2]),wrist_verified=True,
                            grasp_rotation=p['grasp_rotation'],stage='close_confirm',
                            holding_action=result.copy(),hold_position=position.copy(),hold_rotation=rotation)
                info['state'] = 'regrasp_pose_verified'
        if self.pending is p and p['steps'] >= self.wrist_config.get('max_level_regrasp_steps',900):
            raise RuntimeError('Level regrasp pose could not converge within allowed joint correction')
        return result,info

    @staticmethod
    def _grasp_yaw_error(target, pose):
        major = np.asarray(target['major_axis_xy'], float)
        closing = Rotation.from_quat(pose.quat_xyzw).as_matrix()[:2,0]
        if (major.shape != (2,) or not np.isfinite(major).all()
                or np.linalg.norm(major)<1e-3 or np.linalg.norm(closing)<1e-3):
            raise RuntimeError('Grasp yaw requires valid parcel and finger horizontal axes')
        minor = np.array([-major[1],major[0]])/np.linalg.norm(major)
        closing = closing/np.linalg.norm(closing)
        if minor@closing < 0:minor = -minor
        return float(np.arctan2(closing[0]*minor[1]-closing[1]*minor[0],closing@minor))

    def _prepare_release_reference(self, measured, support_ok, result, info):
        """Settle the configured handoff while gripping; share weighing budget."""
        whole_arm = self.wrist_config.get("release_prepare_whole_arm", False)
        active = slice(8,15) if whole_arm else slice(12,15)
        p = self.pending
        if 'release_prepare_target' not in p:
            if self.last_sent is None:
                raise RuntimeError('Release preparation requires actual sent reference')
            target = result.copy()
            target[8:12] = self.last_sent[8:12]
            target[12:15] = measured[12:15]
            if whole_arm:
                target[8:15] = self.last_sent[8:15]
            target[15] = p['hold_grip']
            if not np.isfinite(target).all() or np.any(target < self.low) or np.any(target > self.high):
                raise RuntimeError('Release preparation target outside physical bounds')
            p['release_prepare_target'] = target
            p['release_prepare_stable'] = 0
        target = p['release_prepare_target']
        sent_error = float(np.max(abs(self.last_sent[8:15]-target[8:15])))
        wrist_error = float(np.max(abs(measured[12:15]-target[12:15])))
        tracking_error = float(np.max(abs(measured[active]-target[active])))
        previous = p.get('release_prepare_previous_wrist')
        wrist_movement = np.inf if previous is None else float(np.max(abs(measured[active]-previous)))
        p['release_prepare_previous_wrist'] = measured[active].copy()
        quiet = support_ok and sent_error <= .005 and wrist_movement <= .003
        p['release_prepare_quiet'] = p.get('release_prepare_quiet',0)+1 if quiet else 0
        adjustment = np.zeros(7 if whole_arm else 3)
        if (self.wrist_config.get('release_prepare_adapt_settled',False)
                and p['release_prepare_quiet'] >= 3 and tracking_error > .015):
            step = self.config['max_joint_step_rad']
            adjustment = np.clip(measured[active]-target[active],-step,step)
            updated = np.clip(target[active]+adjustment,self.low[active],self.high[active])
            adjustment = updated-target[active]
            target[active] = updated
            p['release_prepare_quiet'] = 0
            sent_error = float(np.max(abs(self.last_sent[8:15]-target[8:15])))
            wrist_error = float(np.max(abs(measured[12:15]-target[12:15])))
        tracking_error = float(np.max(abs(measured[active]-target[active])))
        release_motion_limit = self.wrist_config.get('release_prepare_motion_limit_rad',.003)
        if not np.isfinite(release_motion_limit) or not 0. < release_motion_limit <= .003:
            raise ValueError('Release motion limit must be positive and no looser than original')
        ready = support_ok and sent_error <= .005 and tracking_error <= .015 and wrist_movement <= release_motion_limit
        p['release_prepare_stable'] = p['release_prepare_stable']+1 if ready else 0
        self.offset[8:15] = target[8:15]-p['base'][8:15]
        info.update(state='preparing_release_reference', release_preparation=dict(
            target_rad=target[8:15].tolist(), sent_error_rad=sent_error,
            whole_arm=whole_arm, tracking_error_rad=tracking_error,
            settled_movement_rad=wrist_movement,
            release_motion_limit_rad=float(release_motion_limit),
            wrist_error_rad=wrist_error, wrist_movement_rad=wrist_movement,
            original_support_checks_passed=bool(support_ok and info.get('original_centering_passed',True)),
            support_checks_passed=bool(support_ok),
            support_position_rule=info.get('support_position_rule','center_and_footprint'),
            stable_steps=p['release_prepare_stable'],
            settled_target_adjustment_rad=adjustment.tolist(),
            gripper_kept_closed=True, shared_weighing_step_budget=self.config['max_release_steps']))
        if p['release_prepare_stable'] >= 3:
            p['reference_prepared'] = True
            p['stage'] = 'release'
            p['base'] = target.copy()
        elif p['steps'] >= self.config['max_release_steps']:
            raise RuntimeError('Release reference did not settle within original weighing budget; gripper kept closed')
        return target.copy()

    def _release_open_command(self):
        step = self.wrist_config.get('release_gripper_open_step')
        if step is None:
            return 0.
        if not np.isfinite(step) or not 0. < step <= 1.:
            raise ValueError('Release gripper opening step must be finite in (0,1]')
        if self.last_sent is None or not np.isfinite(self.last_sent[15]):
            raise RuntimeError('Progressive release requires actual sent gripper reference')
        return float(max(0., self.last_sent[15]-step))

    def _pending_action(self, measured, snapshot):
        p = self.pending; side = p['side']; grip = (7, 15)[side]
        p['steps'] += 1
        poses = self.kinematics.fk(measured); pose = self._pose(poses, side)
        info = dict(state='small_' + p['kind'], arm=side, phase_steps=p['steps'], strategy='act_residual')
        if p['kind'] == 'short_retreat':
            result = p['base'].copy()+self.offset; result[15] = 0.
            opening_command = self._release_open_command()
            if (p['steps'] <= self.wrist_config.get('release_open_dwell_steps', 12)
                    or measured[15] > .1 or opening_command > 0.):
                if p['steps'] >= 40:
                    raise RuntimeError('Right gripper did not open before retreat')
                if self.wrist_config.get('release_cartesian_open_hold', False):
                    position=np.asarray(pose.pos_xyz)
                    rotation=Rotation.from_quat(pose.quat_xyzw)
                    error=p['release_hold_position']-position
                    angular=(p['release_hold_rotation']*rotation.inv()).as_rotvec()
                    velocity=position-p.get('hold_last_position',position)
                    angular_velocity=(rotation*p.get('hold_last_rotation',rotation).inv()).as_rotvec()
                    position_gain=self.wrist_config.get('release_open_hold_position_gain',1.)
                    if not np.isfinite(position_gain) or position_gain<=0:
                        raise ValueError('Opening hold position gain must be positive and finite')
                    # Compensate measured unload drift while fingers open.
                    # Same incremental corrector, step limit and downstream
                    # MotionFilter; the original open dwell/budget is retained.
                    result=self._correct(p['base'],measured,side,position_gain*error-1.5*velocity,
                                         angular-4.*angular_velocity)
                    result[15]=0.
                    p['hold_last_position']=position.copy()
                    p['hold_last_rotation']=rotation
                    info['release_pose_hold']=dict(position_error_m=float(np.linalg.norm(error)),
                        angular_error_degrees=float(np.rad2deg(np.linalg.norm(angular))),
                        position_gain=float(position_gain),
                        position_goal_m=p['release_hold_position'].tolist(),
                        measured_gripper=float(measured[15]))
                result[15] = opening_command
                info['release_gripper_ramp'] = dict(command=opening_command,
                    previous_command=None if self.last_sent is None else float(self.last_sent[15]),
                    step=self.wrist_config.get('release_gripper_open_step'),
                    original_open_timeout_steps=40)
                return result, info
            if p.get('goal') is None:
                if self.wrist_config.get('release_geometry_reobserve', False):
                    candidates = [] if snapshot is None else snapshot.get('candidates', [])
                    near = [c for c in candidates if np.linalg.norm(
                        np.asarray(c['position_base_m'])[:2]-self.weighing_target[:2]) < .12]
                    valid_plane = snapshot is not None and np.isfinite(snapshot.get('table_z_base_m', np.nan))
                    if len(near) != 1 or not valid_plane:
                        p['geometry_wait_steps'] = p.get('geometry_wait_steps', 0)+1
                        info.update(state='waiting_release_geometry', current_parcel_count=len(near),
                                    geometry_wait_steps=p['geometry_wait_steps'])
                        if (p['geometry_wait_steps'] >= self.wrist_config.get('max_release_geometry_wait_steps', 20)
                                or p['steps'] >= self.wrist_config.get('max_short_retreat_steps', 90)):
                            raise RuntimeError('Release parcel observation remained missing or ambiguous; no retreat sent')
                        return result, info
                if self.wrist_config.get('release_retreat_direction') == 'measured_tool_axis':
                    validator = None
                    if self.wrist_config.get('release_retreat_preflight') == 'executor_sweep':
                        if snapshot is None or not np.isfinite(snapshot.get('table_z_base_m', np.nan)):
                            raise RuntimeError('Release retreat requires an observed parcel plane')
                        validator = lambda goal, parcel: certify_retreat_execution(
                            self, p['base'], measured, goal, parcel, snapshot['table_z_base_m'],
                            self.wrist_config.get('max_short_retreat_steps', 90)-p['steps']+1,
                            getattr(self, 'retreat_motion_config', None))
                    p['goal'], p['retreat_goal_check'] = axis_retreat_goal(
                        self.kinematics, measured, p['base'], self.low, self.high, self.offset,
                        self.wrist_config.get('max_regrasp_joint_offset_rad', self.config['max_joint_offset_rad']),
                        self.unwinding_placement, snapshot, self.weighing_target[:2], p['extent'],
                        self.wrist_config['release_short_retreat_m'],
                        self.wrist_config['release_clearance_rise_m'],
                        self.wrist_config.get('release_retreat_ready_m', .014),
                        execution_validator=validator)
                    if validator is not None:
                        p['retreat_sweep_target'] = copy.deepcopy(p['retreat_goal_check']['observed_parcel'])
                        p['retreat_sweep_plane'] = snapshot['table_z_base_m']
                else:
                    direction = -np.asarray(self.weighing_target[:2], float)
                    direction /= max(np.linalg.norm(direction), 1e-9)
                    p['goal'] = np.asarray(pose.pos_xyz, float).copy()
                    p['goal'][:2] += direction * self.wrist_config.get('release_short_retreat_m', .035)
                    p['goal'][2] += self.wrist_config.get('release_clearance_rise_m', .012)
            if 'retreat_goal_check' in p:
                info['retreat_goal_check'] = p['retreat_goal_check']
            error = p['goal'] - np.asarray(pose.pos_xyz)
            velocity = (np.zeros(3) if p.get('last_position') is None else
                        np.asarray(pose.pos_xyz)-p['last_position'])
            if self.wrist_config.get('release_sweep_projection', False):
                result, projection = self._retreat_correct(p['base'], measured, error-1.5*velocity,
                    p['retreat_sweep_target'], p['retreat_sweep_plane'], self.retreat_motion_config)
                if projection and projection['applied']:
                    info['retreat_projection'] = projection
            else:
                result = self._correct(p['base'], measured, side, error-1.5*velocity, step_scale=.5)
            result[15] = 0.
            if self.wrist_config.get('release_retreat_preflight') == 'executor_sweep':
                self.command_sweep_context = dict(target=p['retreat_sweep_target'],
                    plane=p['retreat_sweep_plane'], stage='short_retreat')
                info['retreat_sweep_required'] = True
            movement = (np.inf if p.get('last_position') is None else
                        np.linalg.norm(np.asarray(pose.pos_xyz)-p['last_position']))
            p['last_position'] = np.asarray(pose.pos_xyz).copy()
            p['stable'] = (p['stable']+1 if np.linalg.norm(error) <=
                           self.wrist_config.get('release_retreat_ready_m', .014)
                           and movement <= .004 else 0)
            info.update(retreat_error_m=float(np.linalg.norm(error)),
                        retreat_movement_m=float(movement), stable_steps=p['stable'])
            if p['stable'] >= 3:
                candidates = [] if snapshot is None else [c for c in snapshot['candidates']
                    if np.linalg.norm(np.asarray(c['position_base_m'])[:2]
                                      -self.weighing_target[:2]) < .12]
                if len(candidates) == 1:
                    target = copy.deepcopy(candidates[0])
                    plane = snapshot['table_z_base_m']
                    self.approach[1] = dict(target=copy.deepcopy(target), plane=plane,
                                            aligned=False, frames=0)
                    self._begin('settle', 1, p['base'], target=target, plane=plane,
                                surface=p['surface'], extent=p['extent'],
                                monitor=SettlementMonitor())
                    info['state'] = 'short_retreat_complete'
            if self.pending is p and p['steps'] >= self.wrist_config.get('max_short_retreat_steps', 90):
                raise RuntimeError('Short release retreat could not settle; right gripper kept open')
            return result, info
        if p['kind'] == 'settle':
            result = p['base'].copy()+self.offset; result[15] = 0.
            observed = resting_observation(snapshot, p['surface'], p['extent'], self._roof_normal)
            settled = p['monitor'].update(observed)
            info['settlement'] = settled
            if settled['confirmed']:
                target = copy.deepcopy(observed['target'])
                target['support_z_base_m'] = snapshot['table_z_base_m']
                self.approach[1] = dict(target=target, plane=p['plane'], aligned=False, frames=0)
                self._begin('pregrasp', 1, p['base'], target=target, plane=p['plane'],
                            surface=copy.deepcopy(p['surface']), extent=np.asarray(p['extent']).copy())
                info['state'] = 'placement_stable'
            elif p['steps'] >= 100:
                raise RuntimeError('Released parcel has no stable support observation; regrasp withheld')
            return result, info
        if p['kind'] == 'pregrasp':
            if self.weighing_target is not None and self.wrist_projector is not None and 'support_z_base_m' in p['target']:
                return self._regrasp_action(measured,snapshot,poses,info)
            point = self._grasp_goal(p['target'], pose, p['plane'])
            error = point[:2] - np.asarray(pose.pos_xyz)[:2]
            require_wrist = self.weighing_target is not None and self.wrist_projector is not None
            wrist_target = self._wrist_grasp_target(p['target'])
            wrist = self._wrist_measure(side, wrist_target, measured, regrasp=True, target=p['target']) if require_wrist else None
            # A matched pad/parcel blob can still sit behind the open fingers.
            # Continue the bounded wrist aim until the projected parcel top
            # rises clear of that lower image band, then settle before descent.
            view_low = (wrist is not None and 'edge_support' not in wrist and
                wrist['projected_uv'][1] >
                self.wrist_image_wh['right'][1] *
                self.wrist_config.get('regrasp_visible_v_fraction', .67))
            wrist_aim = (self.wrist_projector.aim_error(side, wrist_target, measured,
                         vertical_fraction=.62)
                         if require_wrist and (wrist is None or view_low) else None)
            if wrist is not None and p['target'].get('grasp_goal_xy') is None:
                p['target']['grasp_goal_xy'] = point[:2].tolist()
            if wrist is not None:
                error += np.asarray(wrist['correction_base_xy'])
            p['wrist_stable'] = (p.get('wrist_stable', 0) + 1 if wrist is not None
                and abs(wrist['centering_error_px']) <= self.wrist_config['aligned_error_px'] else 0)
            velocity = (np.asarray(pose.pos_xyz)[:2] - p['last_position'][:2]
                        if p.get('last_position') is not None else np.zeros(2))
            if require_wrist and wrist is None:
                # Acquire edges before sweeping the fingers toward the box.
                if 'view_position' not in p:
                    p['view_position'] = np.asarray(pose.pos_xyz).copy()
                correction = p['view_position']-np.asarray(pose.pos_xyz)
                correction[:2] -= 2.*velocity
            else:
                correction = error-2.*velocity
            previous_offset = self.offset.copy()
            initial_rotation = (self._initial_grasp_rotation(p['target'],pose)
                if side == 1 and not require_wrist and 'support_z_base_m' not in p['target'] else None)
            initial_angle = (None if initial_rotation is None else
                (initial_rotation*Rotation.from_quat(pose.quat_xyzw).inv()).as_rotvec())
            grasp_yaw = (self._grasp_yaw_error(p['target'],pose) if side==1 and not require_wrist
                         and self.wrist_config.get('grasp_align_parcel_yaw',False) else None)
            yaw_ready = grasp_yaw is None or abs(grasp_yaw)<=np.deg2rad(2.)
            transverse_error = 0.
            if grasp_yaw is not None:
                major=np.asarray(p['target']['major_axis_xy'],float);major/=np.linalg.norm(major)
                transverse_error=float(np.array([-major[1],major[0]])@error)
                yaw_ready=yaw_ready and abs(transverse_error)<=.002
            if initial_angle is not None:
                yaw_ready = yaw_ready and np.linalg.norm(initial_angle) <= np.deg2rad(2.)
                info.update(initial_grasp_orientation_error_degrees=float(np.rad2deg(np.linalg.norm(initial_angle))),
                    initial_grasp_pitch_degrees=float(self.wrist_config['first_grasp_pitch_degrees']))
            result = self._correct(p['base'], measured, side, correction,
                                   (initial_angle if initial_angle is not None else np.array([0.,0.,grasp_yaw]) if grasp_yaw is not None else
                                    None if wrist_aim is None else wrist_aim['error_base']),
                                   step_scale=.4,
                                   view_target=wrist_target if require_wrist and wrist is None else None)
            if require_wrist and wrist is None:
                # Check the immediate measured-joint sweep before sending it.
                # The TCP is below the camera/hand; retaining 30 mm above the
                # parcel limits fingertip intrusion while changing viewpoint.
                delta = self.offset[8:15]-previous_offset[8:15]
                origin = p['view_position']
                safe = True
                for fraction in (.5,1.):
                    predicted = self.kinematics.matrix(measured[8:15]+fraction*delta,1)[:3,3]
                    if (predicted[2] < min(pose.pos_xyz[2],p['target']['top_z_base_m']+.03)-.001
                            or np.linalg.norm(predicted[:2]-origin[:2]) > .03):
                        safe = False
                if not safe:
                    self.offset = previous_offset
                    result = p['base'].copy()+self.offset
                    info['view_step_blocked'] = True
            result[grip] = 0.
            movement = np.linalg.norm(velocity) if p.get('last_position') is not None else np.inf
            p['last_position'] = np.asarray(pose.pos_xyz).copy()
            # Keep the ACT descent paused while the open fingers move across
            # the box. A box moving this far before contact means it was hit.
            displacement = self._parcel_displacement(snapshot, p['target'])
            if displacement is not None and displacement > .01:
                raise RuntimeError('Parcel moved before grasp alignment; gripper kept open')
            if require_wrist and 'support_z_base_m' in p['target'] and displacement is None:
                self.offset = previous_offset
                result = p['base'].copy()+self.offset; result[grip] = 0.
                p['unseen_roof_steps'] = p.get('unseen_roof_steps',0)+1
                info['view_step_blocked'] = True
                if p['unseen_roof_steps'] >= 20:
                    raise RuntimeError('Resting parcel roof is obscured; observation motion stopped')
            else:
                p['unseen_roof_steps'] = 0
            clearance = pose.pos_xyz[2] - p['target']['top_z_base_m']
            pose_hold = (not require_wrist and np.linalg.norm(error) <= .01
                         and clearance >= .03 and not self.unwinding_placement and yaw_ready)
            if pose_hold:
                # A measured pose can be aligned while still moving. Preserve
                # the previous target during the dwell instead of injecting
                # another correction and restarting the arm's settling motion.
                self.offset = previous_offset
                result = p['base'].copy()+self.offset
                result[grip] = 0.
            aligned = (yaw_ready and np.linalg.norm(error) <= .01 and movement <= .002
                       and clearance >= .03
                       and p.get('unseen_roof_steps',0) == 0
                       and not self.unwinding_placement
                       and (not require_wrist or (not view_low and
                            p['wrist_stable'] >= self.wrist_config['stable_frames'])))
            p['stable'] = p['stable'] + 1 if aligned else 0
            if grasp_yaw is not None:
                info.update(grasp_yaw_error_degrees=float(np.rad2deg(grasp_yaw)),
                            grasp_transverse_error_m=transverse_error,grasp_yaw_sweep_required=True)
                self.command_sweep_context=dict(target=p['target'],plane=p['plane'],stage='pregrasp_yaw')
            info.update(xy_error_m=float(np.linalg.norm(error)), clearance_m=float(clearance), pose_hold=bool(pose_hold),
                        parcel_displacement_m=displacement,
                        lateral_movement_m=float(movement), stable_steps=p['stable'],
                        wrist=wrist, wrist_aim_degrees=None if wrist_aim is None
                        else wrist_aim['angle_degrees'], wrist_stable_steps=p['wrist_stable'],
                        offset_rad=self.offset[8:15].tolist(),
                        observation_only=bool(require_wrist and wrist is None))
            if p['stable'] >= 3:
                self.approach[side]['target'] = p['target']
                self.approach[side]['aligned'] = True
                self.approach[side]['wrist_verified'] = p['wrist_stable'] >= self.wrist_config['stable_frames'] if require_wrist else False
                self.pending = None
                info['state'] = 'pregrasp_aligned'
            elif p['steps'] >= (self.wrist_config.get('max_pregrasp_align_steps', 80)
                                if require_wrist else self.config.get('max_initial_pregrasp_steps', 160)):
                raise RuntimeError('Pregrasp lateral alignment exceeded the allowed small correction')
            return result, info
        if p['kind'] == 'grasp':
            if p.get('stage') == 'close_confirm':
                p['closing_steps'] = p.get('closing_steps', 0) + 1
                error = p['hold_position'] - np.asarray(pose.pos_xyz)
                rotation = Rotation.from_quat(pose.quat_xyzw)
                angular_error = (p['hold_rotation'] * rotation.inv()).as_rotvec()
                # Keep the aligned ACT target after finger contact; chasing
                # the unloaded TCP can twist a parcel resting on the table.
                result = p['holding_action'].copy()
                result[grip] = .98
                movement = np.linalg.norm(np.asarray(pose.pos_xyz)-p.get('previous_hold_position', p['hold_position']))
                angular_motion = (p.get('previous_hold_rotation', p['hold_rotation']).inv()*rotation).magnitude()
                p['previous_hold_position'] = np.asarray(pose.pos_xyz).copy()
                p['previous_hold_rotation'] = rotation
                target = p['target']
                center = np.asarray(target['position_base_m']).copy()
                height = target['top_z_base_m']-p['plane']
                center[2] = p['plane']+height/2
                major = np.asarray(target.get('major_axis_xy',[1.,0.]))
                major /= np.linalg.norm(major)
                relative = np.asarray(pose.pos_xyz)-center
                size = np.asarray(target.get('oriented_extent_m',target['extent_xy_m']))
                in_grasp = (abs(relative[:2]@major) <= size[0]/2+.012
                            and abs(relative[:2]@np.array([-major[1],major[0]])) <= size[1]/2+.012
                            and abs(relative[2]) <= height/2+.015)
                settled = in_grasp and movement <= .004 and angular_motion <= np.deg2rad(1)
                p['hold_stable'] = p.get('hold_stable', 0) + 1 if settled and self.closed[side] else 0
                info.update(holding_error_m=float(np.linalg.norm(error)),
                            contact_envelope_ok=bool(in_grasp),
                            holding_movement_m=float(movement), stable_steps=p['hold_stable'],
                            holding_angle_degrees=float(np.rad2deg(np.linalg.norm(angular_error))))
                if p['hold_stable'] >= 2:
                    # Register the object-to-hand transform at settled contact,
                    # not before the closing linkage and arm load have settled.
                    self.geometry[side] = self._geometry(target,pose,p['plane'])
                    if side == 1 and np.max(np.abs(self.offset[8:15])) > self.config['max_joint_offset_rad']:
                        self.unwinding_placement = True
                    if side == 1:
                        info['contact_visual_box'] = self._observe_box(snapshot,poses)
                        if 'support_z_base_m' in target:
                            self.regrasp_complete = True
                            info['regrasp_contact'] = True
                        else:
                            self.first_grasp_complete = True
                    self.pending = None
                    info['state'] = 'grasp_contact_confirmed'
                elif p['closing_steps'] >= self.config['max_close_settle_steps']:
                    raise RuntimeError('Grasp contact did not settle before ACT lift')
                return result, info
            point = self._grasp_goal(p['target'], pose, p['plane'])
            point[2] = p['grasp_z']
            error = point - np.asarray(pose.pos_xyz)
            wrist = (self._wrist_measure(side, self._wrist_grasp_target(p['target']), measured,
                                         regrasp=True, target=p['target'])
                     if p.get('wrist_verified') else None)
            if wrist is not None:
                error[:2] += np.asarray(wrist['correction_base_xy'])
            angular_error = None
            orientation_ok = True
            initial_rotation = (self._initial_grasp_rotation(p['target'],pose)
                if side == 1 and 'grasp_rotation' not in p and 'support_z_base_m' not in p['target'] else None)
            if initial_rotation is not None:
                rotation = Rotation.from_quat(pose.quat_xyzw)
                angular_error = (initial_rotation*rotation.inv()).as_rotvec()
                orientation_ok = np.linalg.norm(angular_error) <= np.deg2rad(2.)
                info.update(initial_grasp_orientation_error_degrees=float(np.rad2deg(np.linalg.norm(angular_error))),
                    initial_grasp_pitch_degrees=float(self.wrist_config['first_grasp_pitch_degrees']))
            if p.get('grasp_rotation') is not None:
                rotation = Rotation.from_quat(pose.quat_xyzw)
                angular_error = (p['grasp_rotation']*rotation.inv()).as_rotvec()
                finger_tilt = np.rad2deg(np.arcsin(np.clip(abs(rotation.as_matrix()[2,0]),0,1)))
                orientation_ok = (finger_tilt <= self.wrist_config.get('regrasp_finger_tilt_degrees',5.)
                                  and np.linalg.norm(angular_error) <= np.deg2rad(8.))
                info['finger_line_tilt_degrees'] = float(finger_tilt)
            if (side==1 and self.wrist_config.get('grasp_align_parcel_yaw',False)
                    and 'support_z_base_m' not in p['target']):
                yaw = self._grasp_yaw_error(p['target'],pose)
                if angular_error is None:angular_error=np.array([0.,0.,yaw])
                orientation_ok = orientation_ok and abs(yaw)<=np.deg2rad(2.)
                major=np.asarray(p['target']['major_axis_xy'],float);major/=np.linalg.norm(major)
                transverse_error=float(np.array([-major[1],major[0]])@error[:2])
                info.update(grasp_yaw_error_degrees=float(np.rad2deg(yaw)),
                            grasp_transverse_error_m=transverse_error,grasp_yaw_sweep_required=True)
                self.command_sweep_context=dict(target=p['target'],plane=p['plane'],stage='grasp_yaw')
            correction = error if orientation_ok else np.zeros(3)
            motion_ready = True
            if (side == 1 and self.wrist_config.get('grasp_motion_settle', False)
                    and 'support_z_base_m' not in p['target']):
                position = np.asarray(pose.pos_xyz)
                previous = p.get('grasp_last_position')
                velocity = np.zeros(3) if previous is None else position-previous
                movement = float(np.linalg.norm(velocity)) if previous is not None else np.inf
                p['grasp_last_position'] = position.copy()
                # The incremental target otherwise keeps accumulating while
                # the MPC/WBC response is still approaching the grasp point.
                correction = correction-2.*velocity
                motion_ready = movement <= .0015
                info.update(grasp_movement_m=movement, grasp_motion_ready=motion_ready,
                            grasp_velocity_brake_m=(-2.*velocity).tolist())
            first_grasp_scale = (self.wrist_config.get('first_grasp_gain_scale', 1.)
                if side == 1 and 'grasp_rotation' not in p
                and 'support_z_base_m' not in p['target'] else 1.)
            result = self._correct(p['base'], measured, side, correction, angular_error,
                                   gain_scale=first_grasp_scale)
            if first_grasp_scale != 1.:
                info['first_grasp_gain_scale'] = float(first_grasp_scale)
            result[grip] = 0
            arrived = (motion_ready and orientation_ok and np.linalg.norm(error[:2]) <= self.config['grasp_tolerance_m']
                       and abs(error[2]) <= self.config['grasp_z_tolerance_m']
                       and (not p.get('wrist_verified') or (wrist is not None
                            and abs(wrist['centering_error_px']) <= self.wrist_config['aligned_error_px'])))
            if info.get('grasp_yaw_sweep_required'):
                arrived=arrived and abs(info['grasp_transverse_error_m'])<=.002
            p['stable'] = p['stable'] + 1 if arrived else 0
            info.update(xy_error_m=float(np.linalg.norm(error[:2])), z_error_m=float(error[2]),
                        grasp_target_base_m=point.tolist(), stable_steps=p['stable'], wrist=wrist)
            if p['stable'] >= 2:
                result[grip] = .98
                self.closing_geometry[side] = self._geometry(p['target'], pose, p['plane'])
                p['stage'] = 'close_confirm'
                p['holding_action'] = result.copy()
                p['hold_position'] = point.copy()
                p['hold_rotation'] = Rotation.from_quat(pose.quat_xyzw)
            elif p['steps'] >= self.config['max_grasp_steps']:
                raise RuntimeError('ACT grasp needs more than the allowed small correction; gripper kept open')
            return result, info
        if p['kind'] == 'handover':
            result = p['base'].copy()
            result[15] = max(.9, self.last_sent[15])
            receiver_ok = self.closed[0] and .20 < measured[7] < .85
            p['stable'] = p['stable'] + 1 if receiver_ok else 0
            if p['stable'] >= 2:
                result[15] = p['requested_open']
                self.opening[1] = True
                self.closed[1] = False; self.geometry[1] = None
                self.handover_camera_ready = False
                self.pending = None
            elif p['steps'] >= 10:
                raise RuntimeError('Receiving gripper did not confirm contact; giver remains closed')
            return result, info
        if p['kind'] == 'handover_align':
            if self.geometry[1] is None:
                raise RuntimeError('Right-hand parcel pose lost before handover')
            center, _, _ = self._world_geometry(1, poses)
            wrist = self._wrist_measure(0, center, measured)
            correction = np.zeros(2) if wrist is None else np.asarray(wrist['correction_base_xy'])
            result = self._correct(p['base'], measured, 0, correction, step_scale=.4)
            result[7] = 0.
            result[15] = max(.98, p['hold_grip'])
            right_position = np.asarray(poses.right_pose.pos_xyz)
            left_position = np.asarray(poses.left_pose.pos_xyz)
            right_movement = (np.inf if p.get('last_right_position') is None else
                              np.linalg.norm(right_position-p['last_right_position']))
            left_movement = (np.inf if p.get('last_left_position') is None else
                             np.linalg.norm(left_position-p['last_left_position']))
            p['last_right_position'] = right_position.copy()
            p['last_left_position'] = left_position.copy()
            ready = (wrist is not None
                     and abs(wrist['centering_error_px']) <= self.wrist_config['aligned_error_px']
                     and right_movement <= .003 and left_movement <= .003)
            p['stable'] = p['stable']+1 if ready else 0
            info.update(wrist=wrist, stable_steps=p['stable'],
                        giver_movement_m=float(right_movement),
                        receiver_movement_m=float(left_movement))
            if p['stable'] >= self.wrist_config['stable_frames']:
                self.handover_camera_ready = True
                self.pending = None
                info['state'] = 'handover_camera_aligned'
            elif p['steps'] >= self.wrist_config['max_handover_align_steps']:
                raise RuntimeError('Left wrist could not verify handover alignment; right hand kept closed')
            return result, info
        if p['kind'] == 'weighing':
            if p['stage'] == 'release':
                if not self._view_ready(snapshot,measured):
                    if p['steps'] >= self.config['max_release_steps']:
                        raise RuntimeError('Weighing camera lost before release; gripper kept closed')
                    return p['base'].copy(), dict(info,state='waiting_release_view')
                result = p['base'].copy(); result[15] = (self._release_open_command()
                    if self.wrist_config.get('release_gripper_open_step') is not None else p['requested_open'])
                if self.wrist_config.get('release_measured_reference_hold', False):
                    # Unloading can expose a large pending tracking error even
                    # while the visually supported TCP appears stationary.
                    # Transfer toward this observation's measured pose within
                    # the existing placement budget around the ORIGINAL ACT
                    # reference. No new budget or enlarged physical bounds.
                    observed = np.asarray(measured[8:15], float)
                    joint_low = self.low[8:15].copy()
                    joint_high = self.high[8:15].copy()
                    if hasattr(self.kinematics, 'limits'):
                        joint_low = np.maximum(joint_low, self.kinematics.limits[1][:, 0])
                        joint_high = np.minimum(joint_high, self.kinematics.limits[1][:, 1])
                    if (not np.isfinite(observed).all() or
                            np.any(observed < joint_low - 5e-4) or
                            np.any(observed > joint_high + 5e-4)):
                        raise RuntimeError('Measured release pose is outside physical bounds')
                    reference_mode = self.wrist_config.get('release_reference_mode', 'original_act_budget')
                    if reference_mode not in ('original_act_budget', 'measured_physical', 'mpc_continuous'):
                        raise ValueError('Unknown release reference mode')
                    cap = (None if reference_mode in ('measured_physical','mpc_continuous') else self.config.get(
                        'max_placement_joint_offset_rad', self.config['max_joint_offset_rad']))
                    original_reference = p['reference_base'][8:15]
                    lower = joint_low if cap is None else np.maximum(joint_low, original_reference - cap)
                    upper = joint_high if cap is None else np.minimum(joint_high, original_reference + cap)
                    if np.any(lower > upper):
                        raise RuntimeError('Release reference has no feasible original correction budget')
                    prior = result[8:15].copy()
                    bounded = np.minimum(np.maximum(observed, lower), upper)
                    physical_bound_active = bounded != observed
                    if reference_mode == 'mpc_continuous':
                        if self.last_sent is None:
                            raise RuntimeError('MPC release continuity requires the actual previous sent command')
                        prior = self.last_sent[8:15].copy()
                        if (not np.isfinite(prior[:4]).all() or np.any(prior[:4]<joint_low[:4])
                                or np.any(prior[:4]>joint_high[:4])):
                            raise RuntimeError('Previous MPC release reference is outside physical bounds')
                        # The proximal upper reference and current MPC state
                        # have distinct steady values. Preserve the actually
                        # issued proximal reference through unloading; only
                        # the directly referenced wrists transfer to feedback.
                        bounded[:4] = prior[:4]
                        if p.get('reference_prepared', False):
                            # Preparation already settled this reference under
                            # the original support checks. Do not reset it when
                            # opening removes contact.
                            bounded[:] = prior
                    result[8:15] = bounded
                    info['release_reference_transfer'] = dict(
                        previous_command_rad=prior.tolist(), measured_target_rad=observed.tolist(),
                        bounded_target_rad=bounded.tolist(), original_reference_rad=original_reference.tolist(),
                        correction_cap_rad=cap, reference_mode=reference_mode,
                        physical_lower_joint_rad=joint_low.tolist(), physical_upper_joint_rad=joint_high.tolist(),
                        bound_active=physical_bound_active.tolist(),
                        remaining_target_error_rad=(bounded-observed).tolist(),
                        requested_change_rad=(bounded-prior).tolist(),
                        interpretation='Continuous proximal MPC input with measured wrist reference' if reference_mode=='mpc_continuous' else
                            'Physical measured hold with a new visual phase reference' if cap is None else
                            'Original ACT-budget reference construction; both modes use unchanged MotionFilter')
                    if reference_mode == 'mpc_continuous':
                        info['release_reference_transfer']['preserved_action_indices'] = [8,9,10,11]
                        info['release_reference_transfer']['previous_sent_command_rad'] = prior.tolist()
                        if p.get('reference_prepared', False):
                            info['release_reference_transfer'].update(
                                preserved_action_indices=list(range(8,15)),
                                prepared_before_open=True,
                                interpretation='Reference settled under closed grip before opening')
                extent = self.geometry[1]['extent'].copy()
                self.opening[1] = True
                self.closed[1] = False; self.geometry[1] = None
                reference = p['reference_base'].copy(); reference[15] = 0.
                # The inactive left arm must keep the visually accepted release
                # command through retreat, support inspection and regrasp. Its
                # frozen ACT prediction is not a new motion request in these
                # phases. Preserve the right ACT reference and residual budget.
                if np.any(result < self.low) or np.any(result > self.high):
                    raise RuntimeError('Verified release command is outside configured bounds')
                old_inactive_reference = reference[:8].copy()
                reference[:8] = result[:8] - self.offset[:8]
                visual_reference = (self.wrist_config.get('release_reference_mode') in ('measured_physical','mpc_continuous')
                                    and self.wrist_config.get('release_measured_reference_hold', False))
                if visual_reference:
                    # A visual release starts its next phase at the accepted
                    # physical hold, rather than pulling an unloaded wrist
                    # back toward an obsolete ACT prediction. MotionFilter
                    # still limits the transition from the last sent command.
                    reference[8:15] = result[8:15]
                info['phase_reference_hold'] = dict(
                    inactive_action_indices=list(range(8)),
                    frozen_act_reference=old_inactive_reference.tolist(),
                    accepted_hold_command=result[:8].tolist(),
                    retained_inactive_offset=self.offset[:8].tolist(),
                    discarded_reference_delta=(old_inactive_reference-reference[:8]).tolist(),
                    active_arm_reference_unchanged=not visual_reference,
                    active_reference_source=('continuous_mpc_measured_wrist_release'
                        if self.wrist_config.get('release_reference_mode')=='mpc_continuous'
                        else 'measured_physical_release') if visual_reference else 'original_act')
                self.offset[8:15] = result[8:15]-reference[8:15]
                self.unwinding_placement = bool(np.max(np.abs(self.offset[8:15])) > self.config['max_joint_offset_rad'])
                self._begin('short_retreat', 1, reference, surface=copy.deepcopy(p['surface']), extent=extent,
                            release_hold_position=np.asarray(pose.pos_xyz).copy(),
                            release_hold_rotation=Rotation.from_quat(pose.quat_xyzw))
                info['state'] = 'weighing_release'
                info['release_gripper_ramp'] = dict(command=float(result[15]),
                    previous_command=None if self.last_sent is None else float(self.last_sent[15]),
                    step=self.wrist_config.get('release_gripper_open_step'),
                    original_open_timeout_steps=40)
                return result, info
            seen = self._observe_box(snapshot, poses)
            if seen is not None:
                p['last_box_observation'] = seen
                p['last_box_step'] = p['steps']
            info.update(head_scan_stage=p['stage'], visible_box_points=seen['points'] if seen else 0)
            if p['stage'] in ('scan_left', 'scan_right'):
                p['stable'] = p['stable']+1 if self._view_ready(snapshot, measured) else 0
                if p['stable'] >= 2:
                    p['stage'] = 'scan_right' if p['stage'] == 'scan_left' else 'correct'
                    p['stable'] = 0
                elif p['steps'] >= self.config['max_release_steps']:
                    raise RuntimeError('Weighing head scan did not reach a confirmed view')
                result = p['base'].copy() + self.offset
                result[15] = p['hold_grip']
                return np.clip(result,self.low,self.high), info
            center, rotation, extent = self._world_geometry(1, poses)
            error = self.weighing_target[:2] - center[:2]
            previous_center = p.get('last_center_xy')
            center_velocity = (np.zeros(2) if previous_center is None else
                               center[:2] - previous_center)
            p['last_center_xy'] = center[:2].copy()
            if np.linalg.norm(error) > self.config['max_release_error_m']:
                raise RuntimeError('Weighing error exceeds small correction range; keeping parcel grasped')
            corners = rotation.apply([[x,y,z] for x in (-extent[0]/2, extent[0]/2)
                                      for y in (-extent[1]/2, extent[1]/2)
                                      for z in (-extent[2]/2, extent[2]/2)]) + center
            bottom_gap = float(corners[:,2].min() - p['surface']['position_base_m'][2])
            if seen is not None:
                bottom_gap = seen['bottom_base_m']-p['surface']['position_base_m'][2]
            previous_gap = p.get('last_bottom_gap')
            gap_velocity = 0. if previous_gap is None else bottom_gap - previous_gap
            p['last_bottom_gap'] = bottom_gap
            up = rotation.apply([0, 0, 1])
            # The initial box was standing on the table; retain yaw while
            # aligning its support face with the observed horizontal pad.
            normal = seen.get('roof_normal_base') if seen is not None else None
            if normal is not None and seen.get('orientation_confirmed'):
                up = np.asarray(normal, float)
                if up[2] < 0:
                    up = -up
            level_error = np.cross(up, [0., 0., 1.])
            movement = np.linalg.norm(np.asarray(pose.pos_xyz) - p['last_position']) if p.get('last_position') is not None else np.inf
            p['last_position'] = np.asarray(pose.pos_xyz)
            heading = rotation.apply([1,0,0]); yaw = np.arctan2(heading[1], heading[0])
            signed_axis_error = (yaw + np.pi/4) % (np.pi/2) - np.pi/4
            axis_error = abs(signed_axis_error)
            heading_ok = axis_error <= np.deg2rad(
                self.config.get('placement_heading_tolerance_degrees', 15.))
            if not p.get('heading_ready', False):
                p['heading_stable'] = (p.get('heading_stable', 0)+1 if axis_error <=
                    np.deg2rad(self.config.get('placement_heading_settle_degrees', 10.)) else 0)
                if p['heading_stable'] >= 3:
                    p['heading_ready'] = True
            heading_descent_ok = heading_ok and (p.get('heading_ready', False) or
                axis_error <= np.deg2rad(self.config.get('placement_heading_settle_degrees', 10.)))
            support_fit = self._fit(center, rotation, extent, p['surface'])
            heading_mode = self.config.get('placement_heading_mode', 'axis_alignment')
            if heading_mode not in ('axis_alignment', 'support_footprint'):
                raise ValueError('Unknown placement heading mode')
            if heading_mode == 'support_footprint':
                # A supported footprint need not resemble the ACT training
                # yaw. Retain the original observed polygon and 25mm margin.
                # If it does not fit, continue to withhold descent/release.
                heading_ok = heading_descent_ok = support_fit
            observation = p.get('last_box_observation', {})
            vision_confirmed = (p.get('last_box_step', -100) >= p['steps']-5
                                and observation.get('orientation_confirmed', False)
                                and (heading_mode == 'support_footprint' or
                                     observation.get('heading_error_degrees', 0.) <= 18)
                                and observation.get('tilt_degrees', 0.) is not None
                                and observation.get('tilt_degrees', 0.) <= self.config['level_tolerance_degrees'])
            if (self.wrist_config.get('release_prepare_recover_support',False)
                    and 'release_prepare_target' in p):
                lost = (not support_fit and seen is not None
                        and seen.get('orientation_confirmed',False)
                        and self._view_ready(snapshot,measured))
                p['release_support_lost_frames'] = p.get('release_support_lost_frames',0)+1 if lost else 0
                if p['release_support_lost_frames'] >= 3:
                    if p['steps'] >= self.config['max_release_steps']:
                        raise RuntimeError('Support recovery exceeds original weighing budget; gripper kept closed')
                    if self.last_sent is None:
                        raise RuntimeError('Support recovery requires actual sent reference')
                    result = self.last_sent.copy(); result[15] = p['hold_grip']
                    self.offset[8:15] = result[8:15]-p['base'][8:15]
                    for key in list(p):
                        if key.startswith('release_prepare_'):
                            del p[key]
                    p['stable'] = 0
                    p['release_support_lost_frames'] = 0
                    p['release_support_recoveries'] = p.get('release_support_recoveries',0)+1
                    info.update(state='recovering_release_support',support_recovery=dict(
                        reason='observed_footprint_outside_original_margin',
                        original_margin_m=self.config['placement_margin_m'],
                        count=p['release_support_recoveries'],phase_steps=p['steps'],
                        original_budget_steps=self.config['max_release_steps'],
                        continuous_arm_reference=True,gripper_kept_closed=True))
                    return result,info
            bottom_corners = rotation.apply([[x,y,-extent[2]/2]
                for x in (-extent[0]/2, extent[0]/2) for y in (-extent[1]/2, extent[1]/2)]) + center
            bottom_spread = float(np.ptp(bottom_corners[:, 2]))
            flat = (seen is not None and seen.get('orientation_confirmed')
                    and seen.get('tilt_degrees') is not None
                    and seen['tilt_degrees'] <= self.config['level_tolerance_degrees']
                    and bottom_spread <= self.config['max_bottom_height_spread_m'])
            level_settled = (flat and seen['tilt_degrees'] <=
                self.config.get('level_target_degrees', 10.)
                and bottom_spread <= self.config.get('level_target_bottom_spread_m', .023))
            if p.get('level_ready', False) and flat:
                # A low edge may touch first. Once inside the permitted tilt,
                # stop asking the delayed wrist to rotate toward zero.
                level_error = np.zeros(3)
            if not heading_descent_ok:
                # Keep the box clear of the pad while bringing its long side
                # back toward the ACT training orientation.
                level_error -= np.array([0., 0., signed_axis_error])
            level_ok = level_settled
            if not p.get('level_ready', False):
                p['level_stable'] = p.get('level_stable', 0) + 1 if level_ok else 0
                if p['level_stable'] >= 3:
                    p['level_ready'] = True
            elif (seen is not None and seen.get('tilt_degrees') is not None
                  and (seen['tilt_degrees'] > self.config['level_tolerance_degrees'] + 2
                       or bottom_spread > self.config['max_bottom_height_spread_m'] + .005)
                  and bottom_gap > .015):
                p['level_ready'] = False
                p['level_stable'] = 0
            if not heading_descent_ok:
                descent = float(np.clip(.05-bottom_gap, -.02, .02))
            elif p.get('level_ready', False) and flat:
                contact_goal = .5*self.config['max_release_gap_m']
                descent = -min(.04, 2.*max(0., bottom_gap-contact_goal))
            elif p.get('level_ready', False) and bottom_gap <= .008:
                # Hold the low edge just above contact if tilt drifts beyond
                # the release limit; keep correcting roll and pitch first.
                descent = 0.
            elif flat and bottom_gap <= .008:
                descent = 0.
            else:
                # Hold the lowest corner at a safe observed clearance while
                # rotating; leveling alone can otherwise lift the parcel away.
                descent = float(np.clip(.06-bottom_gap, -.02, .02))
            # The loaded wrist responds several frames after a correction.
            # Brake measured movement toward the target before it overshoots.
            descent = float(np.clip(descent - 1.5*np.clip(gap_velocity, -.01, .01), -.04, .02))
            damped_xy = error - 1.5*np.clip(center_velocity, -.01, .01)
            fast_descent = (heading_descent_ok and p.get('level_ready', False)
                            and flat and bottom_gap > self.config['max_release_gap_m'])
            result = (p['release_prepare_target'].copy() if 'release_prepare_target' in p else self._correct(p['base'], measured, 1,
                                   np.r_[damped_xy, descent], level_error,
                                   level_axis=up if heading_descent_ok else None,
                                   step_scale=1. if fast_descent else .5,
                                   translation_limit_m=.04 if fast_descent else None,
                                   offset_cap_rad=self.config.get(
                                       'max_placement_joint_offset_rad', self.config['max_joint_offset_rad'])))
            result[15] = p['hold_grip']
            centered = np.linalg.norm(error) <= self.config['placement_tolerance_m']
            preparation_footprint = (self.wrist_config.get('release_prepare_support_footprint', False)
                                     and 'release_prepare_target' in p)
            position_ready = support_fit if preparation_footprint else centered
            stable = (p.get('level_ready', False) and vision_confirmed
                      and self._view_ready(snapshot, measured) and position_ready
                      and support_fit
                      and -.008 <= bottom_gap <= self.config['max_release_gap_m']
                      and bottom_spread <= self.config['max_bottom_height_spread_m']
                      and movement < .004 and heading_ok)
            p['stable'] = p['stable'] + 1 if stable else 0
            info.update(xy_error_m=float(np.linalg.norm(error)), bottom_gap_m=bottom_gap,
                        bottom_height_spread_m=bottom_spread, visual_box=seen,
                        placement_phase='lower' if p.get('level_ready', False) else 'level',
                        level_stable_steps=p.get('level_stable', 0),
                        target_base_m=self.weighing_target.tolist(),
                        heading_error_degrees=float(np.rad2deg(axis_error)),
                        heading_ready=bool(p.get('heading_ready', False)),
                        heading_mode=heading_mode, support_footprint_fit=bool(support_fit),
                        original_centering_passed=bool(centered),
                        support_position_rule='prepared_footprint' if preparation_footprint else 'center_and_footprint',
                        heading_descent_ok=bool(heading_descent_ok),
                        stable_steps=p['stable'])
            # Keep the pad view through release, settlement and regrasp.
            # The runner restores the ACT view after the local check ends.
            if (self.wrist_config.get('release_prepare_before_open', False)
                    and (p['stable'] >= 3 or 'release_prepare_target' in p)):
                result = self._prepare_release_reference(measured, stable, result, info)
            elif p['stable'] >= 3:
                p['stage'] = 'release'; p['base'] = result.copy()
            elif p['steps'] >= self.config['max_release_steps']:
                raise RuntimeError('ACT release did not pass visual/settlement checks within small correction limits')
            return result, info
        raise RuntimeError('Unknown residual check')

    def apply(self, raw, measured, snapshot):
        raw, measured = action_vector(raw), action_vector(measured)
        raw = raw.copy()
        for side, grip in enumerate((7, 15)):
            if self.opening[side]:
                if raw[grip] <= .25:
                    self.opening[side] = False
                raw[grip] = 0.
        self._remember(snapshot)
        poses = self.kinematics.fk(measured)
        # Confirm actual contact, never equate a close command with grasping.
        for side, grip in enumerate((7,15)):
            previous = self.previous_fingers[side]
            stationary = previous is not None and abs(measured[grip]-previous) < .015
            self.previous_fingers[side] = measured[grip]
            if raw[grip] <= .25 and not self.closed[side]:
                self.ready_close[side] = True
            if self.closing_geometry[side] is not None and not self.closed[side]:
                strong_command = self.last_sent is not None and self.last_sent[grip] >= .9
                contact = .20 < measured[grip] < .85 and stationary and strong_command
                self.contact_frames[side] = self.contact_frames[side] + 1 if contact else 0
                self.closing_frames[side] += 1
                if self.contact_frames[side] >= 2:
                    if side == 0 and self.closed[1]:
                        center, rotation, extent = self._world_geometry(1, poses)
                        tool = poses.left_pose; local = Rotation.from_quat(tool.quat_xyzw)
                        self.closing_geometry[0] = dict(center=local.inv().apply(center-tool.pos_xyz),
                                                       rotation=local.inv()*rotation, extent=extent.copy())
                    self.closed[side] = True; self.geometry[side] = self.closing_geometry[side]
                    self.closing_geometry[side] = None
                elif self.closing_frames[side] >= 12:
                    raise RuntimeError('ACT close has no confirmed finger contact')
        if self.pending is not None:
            return self._pending_action(measured, snapshot)
        result = raw + self.offset
        transport_info = None
        if self.approach[1] is not None and not self.closed[1] and snapshot is not None:
            displacement = self._parcel_displacement(snapshot, self.approach[1]['target'])
            if displacement is not None and displacement > .01:
                raise RuntimeError('Parcel moved during open-hand approach; gripper kept open')
        # Start visual correction while the open right hand is still above
        # the parcel. The ACT descent can otherwise strike its side before
        # the learned close command reaches the later grasp check.
        if (not any(self.closed) and self.ready_close[1] and raw[15] <= .25
                and snapshot is not None and snapshot['candidates']):
            pose = poses.right_pose
            candidates = sorted(snapshot['candidates'], key=lambda c:
                np.linalg.norm(self._grasp_goal(c, pose)[:2] - pose.pos_xyz[:2]))
            target = (self.approach[1]['target'] if self.approach[1] is not None
                      else candidates[0])
            point = self._grasp_goal(target, pose, snapshot['table_z_base_m'])
            lateral = np.linalg.norm(point[:2] - pose.pos_xyz[:2])
            clearance = pose.pos_xyz[2] - target['top_z_base_m']
            distinct = (len(candidates) == 1 or
                np.linalg.norm(self._grasp_goal(candidates[1], pose)[:2] - pose.pos_xyz[:2]) - lateral >= .02)
            already_aligned = self.approach[1] is not None and self.approach[1].get('aligned', False)
            yaw_needs_alignment = (self.wrist_config.get('grasp_align_parcel_yaw',False)
                and abs(self._grasp_yaw_error(target,pose))>np.deg2rad(2.))
            if (.03 < clearance < .10 and lateral < .10 and (.012 < lateral or yaw_needs_alignment)
                    and distinct and not already_aligned):
                if self.approach[1] is None:
                    self.approach[1] = dict(target=copy.deepcopy(target),
                                            plane=snapshot['table_z_base_m'], frames=0)
                base = (self.last_sent.copy() if self.last_sent is not None else raw.copy())
                base[8:15] -= self.offset[8:15]
                base[15] = 0.
                self._begin('pregrasp', 1, base,
                            target=copy.deepcopy(target), plane=snapshot['table_z_base_m'])
                return self._pending_action(measured, snapshot)
        if (self.approach[1] is not None and self.approach[1].get('aligned')
                and not self.closed[1] and raw[15] < .6):
            pose = poses.right_pose
            target = self.approach[1]['target']
            point = self._grasp_goal(target, pose, self.approach[1]['plane'])
            lateral_error = point[:2] - np.asarray(pose.pos_xyz)[:2]
            wrist = (self._wrist_measure(1, self._wrist_grasp_target(target), measured,
                                         regrasp=True, target=target)
                     if self.approach[1].get('wrist_verified') else None)
            if wrist is not None:
                lateral_error += np.asarray(wrist['correction_base_xy'])
            if self.approach[1].get('wrist_verified') and (
                    wrist is None or np.linalg.norm(lateral_error) > .012):
                # The next ACT chunk can turn or sweep the open wrist away
                # from the verified parcel. Re-align above it before descent.
                base = (self.last_sent.copy() if self.last_sent is not None
                        else raw.copy())
                base[8:15] -= self.offset[8:15]
                base[15] = 0.
                self.approach[1]['aligned'] = False
                self._begin('pregrasp', 1, base,
                            target=copy.deepcopy(target),
                            plane=self.approach[1]['plane'])
                return self._pending_action(measured, snapshot)
            velocity = (np.asarray(pose.pos_xyz)[:2] - self.last_tool[1]
                        if self.last_tool[1] is not None else np.zeros(2))
            result = self._correct(raw, measured, 1, lateral_error - 2. * velocity,
                                   step_scale=.4)
            if self.approach[1].get('wrist_verified') and self.last_sent is not None:
                step = self.wrist_config.get('max_open_joint_delta_rad', .025)
                cap = self.wrist_config.get('max_regrasp_joint_offset_rad', self.config['max_joint_offset_rad'])
                lower = np.maximum(self.last_sent[8:15]-step, raw[8:15]-cap)
                upper = np.minimum(self.last_sent[8:15]+step, raw[8:15]+cap)
                if np.any(lower > upper):
                    raise RuntimeError('ACT open-hand jump exceeds wrist-safe correction range')
                result[8:15] = np.clip(result[8:15], lower, upper)
                self.offset[8:15] = result[8:15]-raw[8:15]
            self.last_tool[1] = np.asarray(pose.pos_xyz)[:2].copy()
            result[15] = 0.
            transport_info = dict(state='open_hand_visual_tracking', strategy='act_residual',
                                  xy_error_m=float(np.linalg.norm(lateral_error)),
                                  wrist=wrist,
                                  offset_rad=self.offset.tolist())
        for side, grip in enumerate((7,15)):
            if self.closed[side] and raw[grip] >= .9:
                self.release_ready[side] = True
            # Intervene before gradual opening removes the gripping force.
            release_threshold = min(.85, max(.65, measured[grip] + .15))
            if self.closed[side] and self.release_ready[side] and raw[grip] <= release_threshold:
                if side == 1 and raw[7] >= .55:
                    base = raw.copy(); base[15] = max(.98, self.last_sent[15])
                    base[8:15] = self.last_sent[8:15]
                    self._begin('handover', 1, base, requested_open=0.)
                    return self._pending_action(measured, snapshot)
                if side == 1:
                    if self.regrasp_complete:
                        result[15] = .98
                        continue
                    center, _, _ = self._world_geometry(1, poses)
                    if not self.surfaces:
                        raise RuntimeError('No observed weighing boundary; keeping gripper closed')
                    surface = min(self.surfaces, key=lambda s: np.linalg.norm(np.asarray(s['position_base_m'])[:2] - center[:2]))
                    if np.linalg.norm(np.asarray(surface['position_base_m'])[:2] - center[:2]) > .18:
                        raise RuntimeError('No weighing surface near ACT release')
                    self.weighing_target = np.asarray(surface['position_base_m'])
                    base = raw.copy(); base[15] = max(.98, self.last_sent[15])
                    self._begin('weighing', 1, base, stage='scan_left', surface=surface,
                                reference_base=base.copy(),
                                requested_open=0., hold_grip=float(base[15]))
                    return self._pending_action(measured, snapshot)
                self.closed[side] = False; self.geometry[side] = None
                self.offset[side * 8:side * 8 + 7] = 0
            elif not self.closed[side] and self.closing_geometry[side] is None and self.ready_close[side] and raw[grip] >= self.config['close_event_threshold']:
                if side == 0 and self.closed[1]:
                    if raw[grip] < .6:
                        continue
                    if self.wrist_projector is not None and not self.handover_camera_ready:
                        base = raw.copy(); base[7] = 0.
                        base[8:15] = self.last_sent[8:15] - self.offset[8:15]
                        self._begin('handover_align', 0, base,
                                    hold_grip=float(max(.98, self.last_sent[15])))
                        return self._pending_action(measured, snapshot)
                else:
                    # Let ACT finish its learned descent while keeping the
                    # fingers open. Do not freeze at the first partial close.
                    if raw[grip] < .6:
                        if snapshot is not None and snapshot['candidates'] and self.approach[side] is None:
                            pose = self._pose(poses, side)
                            candidates = sorted(snapshot['candidates'], key=lambda c: np.linalg.norm(self._grasp_goal(c,pose)[:2]-pose.pos_xyz[:2]))
                            if np.linalg.norm(self._grasp_goal(candidates[0],pose)[:2]-pose.pos_xyz[:2]) <= .10:
                                self.approach[side] = dict(target=copy.deepcopy(candidates[0]), plane=snapshot['table_z_base_m'], frames=0)
                        if self.approach[side] is not None:
                            self.approach[side]['frames'] += 1
                            if self.approach[side]['frames'] > 20:
                                raise RuntimeError('ACT approach did not finish before close timeout')
                        result[grip] = 0.
                        continue
                self.ready_close[side] = False
                self.release_ready[side] = False
                self.contact_frames[side] = self.closing_frames[side] = 0
                if side == 0 and self.closed[1]:
                    center, rotation, extent = self._world_geometry(1, poses)
                    tool = self._pose(poses, 0); local = Rotation.from_quat(tool.quat_xyzw)
                    self.closing_geometry[0] = dict(center=local.inv().apply(center-tool.pos_xyz),
                                                   rotation=local.inv()*rotation, extent=extent.copy())
                    continue
                if snapshot is None:
                    raise RuntimeError('No fresh parcel observation at ACT close')
                pose = self._pose(poses, side)
                remembered = self.approach[side]
                candidates = sorted(snapshot['candidates'], key=lambda c: np.linalg.norm(self._grasp_goal(c, pose)[:2] - pose.pos_xyz[:2]))
                if remembered is not None:
                    target = remembered['target']; plane = remembered['plane']
                    wrist_verified = remembered.get('wrist_verified', False)
                elif candidates:
                    target = candidates[0]; plane = snapshot['table_z_base_m']
                    wrist_verified = False
                else:
                    raise RuntimeError('No parcel observation before ACT close')
                distance = np.linalg.norm(self._grasp_goal(target, pose)[:2] - pose.pos_xyz[:2])
                if distance > .10:
                    raise RuntimeError('ACT closing target is outside local visual correction range')
                if remembered is None and len(candidates) > 1 and np.linalg.norm(self._grasp_goal(candidates[1], pose)[:2] - pose.pos_xyz[:2]) - distance < .02:
                    raise RuntimeError('ACT closing target is ambiguous')
                self.approach[side] = None
                # ACT supplies the grasp height. The proxy TCP can lie above
                # the actual fingertips, so only trim this height by 15 mm.
                grasp_z = max(float(self._grasp_goal(target, pose, plane)[2]),
                              float(pose.pos_xyz[2]) - .015)
                self._begin('grasp', side, raw, target=copy.deepcopy(target),
                            plane=plane, grasp_z=grasp_z, wrist_verified=wrist_verified)
                return self._pending_action(measured, snapshot)
            # Decay small grasp corrections while ACT carries the parcel.
            if self.closed[side]:
                result[grip] = max(.98, raw[grip])
                leveling = False
                if side == 1 and raw[7] <= .25 and self.surfaces:
                    center, rotation, extent = self._world_geometry(1,poses)
                    surface = min(self.surfaces,key=lambda s:np.linalg.norm(np.asarray(s['position_base_m'])[:2]-center[:2]))
                    gap = center[2] - float(np.abs(rotation.as_matrix()[2])@extent)/2 - surface['position_base_m'][2]
                    if np.linalg.norm(np.asarray(surface['position_base_m'])[:2]-center[:2]) < .12 and .03 < gap < .25:
                        seen = self._observe_box(snapshot,poses)
                        # Preserve ACT translation while gradually leveling
                        # the visible held parcel before the final low pose.
                        if seen is not None and seen['orientation_confirmed']:
                            normal = np.asarray(seen['roof_normal_base'])
                            if normal[2] < 0:
                                normal = -normal
                            result = self._correct(raw,measured,1,np.zeros(3),np.cross(normal,[0.,0.,1.]),level_axis=normal)
                            result[15] = max(.98,raw[15])
                            leveling = True
                            transport_info = dict(state='act_transport_leveling',strategy='act_residual',
                                visual_box=seen, offset_rad=self.offset.tolist())
                        elif np.any(self.offset[8:15]):
                            leveling = True  # keep the small correction through brief occlusion
                if not leveling:
                    self.offset[side*8:side*8+7] *= .85
        if self.release_countdown:
            self.release_countdown -= 1
            if self.release_countdown == 0 and max(raw[7], raw[15]) <= .25:
                return np.clip(result, self.low, self.high), dict(state='release_dwell', arm=1, strategy='act_residual')
        return np.clip(result, self.low, self.high), transport_info or dict(state='act', strategy='act_residual', offset_rad=self.offset.tolist())
