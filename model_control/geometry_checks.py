"""Pure pose conversion and explicit visual acceptance, with no action selection."""
import copy
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene1.settlement import resting_observation, SettlementMonitor
from scene1.residual import ResidualAssist
from scene1.visual_cycle import VisualCycle, largest_surface, track_bin_rim


def snapshot_from_observation(observation):
    """Rebuild a verifier snapshot from a saved bridge response and PNG/NPZ.

    This only loads recorded sensor artifacts. It does not subscribe to ROS,
    choose a target, mutate the saved observation, or send robot commands.
    """
    observation = observation.get('observation',observation)
    def available(primary, fallback):
        for candidate in (observation.get(primary),observation.get(fallback)):
            if candidate and Path(candidate).is_file():
                return candidate
        raise FileNotFoundError(f'Observation artifact missing: {primary}/{fallback}')
    bgr = cv2.imread(available('host_image','image'),cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Observation image could not be decoded')
    with np.load(available('host_depth','depth'),allow_pickle=False) as data:
        snapshot = {key:np.array(data[key],copy=True) for key in ('depth_m','intrinsic','camera_to_base')}
    if snapshot['depth_m'].shape != bgr.shape[:2]:
        raise ValueError('Observation RGB and depth dimensions differ')
    snapshot.update(rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),
        stamp=float(observation['source_stamp_s']),
        table_z_base_m=float(observation['table_z_base_m']),
        candidates=copy.deepcopy(observation['candidates']),
        surfaces=copy.deepcopy(observation.get('surfaces',[])))
    return snapshot


def normalized_rim(rim):
    result = copy.deepcopy(rim)
    result['corners'] = np.asarray(result['corners'],np.float32)
    result['center'] = np.asarray(result['center'],float)
    return result


def _vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must contain {length} finite values')
    return result


def _rotation(value, label):
    value = _vector(value, 4, label)
    if abs(np.linalg.norm(value)-1.) > 1e-3:
        raise ValueError(f'{label} must be a normalized xyzw quaternion')
    return Rotation.from_quat(value)


def payload_tool_goal(payload, current_tool, desired_center, desired_box_quaternion):
    """Convert a model-selected box pose to a TCP pose using observed attachment.

    payload is one describe().payloads entry; current_tool is observation.left
    or observation.right from the SAME observation. This function neither
    chooses a surface nor moves the robot and makes no reachability assertion.
    """
    box_position = _vector(payload['center_base_m'], 3, 'payload center')
    box_rotation = _rotation(payload['quaternion_xyzw'], 'payload quaternion')
    tool_position = _vector(current_tool['pos_xyz'], 3, 'tool position')
    tool_rotation = _rotation(current_tool['quat_xyzw'], 'tool quaternion')
    desired_center = _vector(desired_center, 3, 'desired box center')
    desired_rotation = _rotation(desired_box_quaternion, 'desired box quaternion')
    local_center = tool_rotation.inv().apply(box_position-tool_position)
    local_rotation = tool_rotation.inv()*box_rotation
    goal_rotation = desired_rotation*local_rotation.inv()
    goal_position = desired_center-goal_rotation.apply(local_center)
    return dict(position_m=goal_position.tolist(), quaternion_xyzw=goal_rotation.as_quat().tolist(),
                desired_payload_center_m=desired_center.tolist(),
                desired_payload_quaternion_xyzw=desired_rotation.as_quat().tolist(),
                source='observed_attachment_geometry', reachability_checked=False)


def support_pose_goal(payload, current_tool, surface, desired_center_xy, desired_yaw_rad,
                      bottom_clearance_m):
    """Convert an explicitly chosen level box pose above an observed support.

    Caller chooses center, heading, and clearance. The observed polygon must
    accommodate the complete payload footprint with the original 25 mm margin.
    """
    xy = _vector(desired_center_xy, 2, 'desired center xy')
    extent = _vector(payload['extent_m'], 3, 'payload extent')
    yaw, clearance = float(desired_yaw_rad), float(bottom_clearance_m)
    if not np.isfinite([yaw,clearance]).all() or not 0. <= clearance <= .3 or np.any(extent <= 0):
        raise ValueError('Invalid yaw, positive extent, or clearance in [0,0.3] m')
    rotation = Rotation.from_euler('z', yaw)
    surface_z = _vector(surface['position_base_m'], 3, 'surface center')[2]
    center = np.r_[xy, surface_z+extent[2]/2+clearance]
    footprint = np.asarray(surface['footprint_xy_m'], dtype=np.float32)
    corners = center+rotation.apply([[x,y,0] for x in (-extent[0]/2,extent[0]/2)
                                     for y in (-extent[1]/2,extent[1]/2)])
    margin = min(cv2.pointPolygonTest(footprint,tuple(p[:2]),True) for p in corners)
    if margin < .025:
        raise ValueError(f'Observed support footprint margin {margin:.4f} m is below 0.025 m')
    return dict(**payload_tool_goal(payload,current_tool,center,rotation.as_quat()),
                support_margin_m=float(margin),requested_bottom_clearance_m=clearance)


class SupportVerifier:
    """Evidence only; initialize after an explicitly recorded supported release."""
    def __init__(self, surface, extent, release_stamp):
        self.surface = copy.deepcopy(surface)
        self.extent = _vector(extent,3,'released payload extent')
        self.release_stamp = float(release_stamp)
        self.monitor = SettlementMonitor()

    def update(self, snapshot, measured):
        measured = _vector(measured,16,'measured state')
        if snapshot['stamp'] <= self.release_stamp:
            return dict(confirmed=False,reason='frame_precedes_release')
        if max(measured[7],measured[15]) >= .15:
            self.monitor.samples.clear()
            return dict(confirmed=False,reason='gripper_not_open')
        observation = resting_observation(snapshot,self.surface,self.extent,ResidualAssist._roof_normal)
        result = self.monitor.update(observation)
        result['interpretation'] = 'observed supported settlement; not official judge task_success'
        return result


class BinVerifier:
    """One fixed pre-release baseline and the unchanged independent-entry gates."""
    def __init__(self, before_snapshot, rim, release_side, kinematics):
        if release_side not in (0,1):
            raise ValueError('release_side must be 0 or 1')
        self.k, self.side = kinematics, release_side
        self.rim = normalized_rim(rim)
        if abs(before_snapshot['stamp']-rim['stamp']) > 1.:
            raise ValueError('Pre-release bin rim observation must be within one second')
        self.before_stamp = float(before_snapshot['stamp'])
        self.before = VisualCycle._bin_points(SimpleNamespace(rim=self.rim),before_snapshot)
        self.samples = []
        self.last_stamp = None
        self.release_stamp = None

    def record_release(self, stamp):
        if self.release_stamp is not None:
            raise RuntimeError('Release already recorded; baseline may not be reset')
        if not np.isfinite(stamp) or stamp <= self.before_stamp:
            raise ValueError('Release must follow the recorded baseline')
        self.release_stamp = float(stamp)

    def update(self, snapshot, measured):
        measured = _vector(measured,16,'measured state')
        if self.release_stamp is None or snapshot['stamp'] <= self.release_stamp:
            return dict(confirmed=False,reason='release_not_recorded_or_frame_precedes_release')
        if self.last_stamp is not None and snapshot['stamp'] <= self.last_stamp:
            return dict(confirmed=False,reason='duplicate_or_old_frame')
        if self.last_stamp is not None and snapshot['stamp']-self.last_stamp > .4:
            self.samples.clear()
        self.last_stamp = float(snapshot['stamp'])
        updated = track_bin_rim(snapshot,self.rim)
        if updated is not None:
            self.rim = updated
        if snapshot['stamp']-self.rim['stamp'] > 1.:
            self.samples.clear()
            return dict(confirmed=False,reason='bin_rim_not_visible')
        start = 8*self.side
        tool = self.k.matrix(measured[start:start+7],self.side)
        wrist = tool[:3,3]-tool[:3,:3]@self.k.tool_xyz
        if max(measured[7],measured[15]) >= .15 or min(tool[2,3],wrist[2]) <= self.rim['rim_z']+.08:
            self.samples.clear()
            return dict(confirmed=False,reason='empty_release_hand_not_clear_of_bin')
        points = VisualCycle._bin_points(SimpleNamespace(rim=self.rim),snapshot)
        if len(points) and len(self.before):
            points = points[cKDTree(self.before).query(points)[0] > .015]
        points = largest_surface(points)
        info = dict(confirmed=False,reason='insufficient_new_surface',new_points=len(points),
                    baseline_stamp=self.before_stamp,release_stamp=self.release_stamp,
                    interpretation='observed independent entry; not official judge task_success')
        if len(points) < 80 or np.count_nonzero(np.ptp(points,axis=0) >= .01) < 2:
            self.samples.clear()
            return info
        self.samples.append((float(snapshot['stamp']),np.median(points,axis=0)))
        self.samples = self.samples[-15:]
        span = float(np.max(np.ptp([p[1] for p in self.samples],axis=0)))
        duration = self.samples[-1][0]-self.samples[0][0]
        confirmed = len(self.samples) >= 5 and duration >= 1. and span <= .015
        info.update(confirmed=bool(confirmed),reason='stable' if confirmed else 'observing',
                    frames=len(self.samples),sensor_span_s=duration,max_center_span_m=span)
        return info
