"""Parcel candidates from observed RGB-D points in the robot base frame.

No simulator truth, object-name lookup or fixed parcel coordinates are used.
The dominant horizontal green surface is estimated from sensor points; object
components are separated in metric XY, including boxes touching in the image.
"""
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import cKDTree

FAST_COMPONENTS = False
SHARED_RGB_DECODER = None


def decode_depth(message):
    if "16UC1" not in message.format:
        raise ValueError(f"Expected metric uint16 compressed depth, got {message.format}")
    magic = b"\x89PNG\r\n\x1a\n"
    offset = bytes(message.data).find(magic)
    if offset < 0:
        raise ValueError("Compressed depth PNG missing")
    depth = cv2.imdecode(np.frombuffer(message.data[offset:], np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Invalid compressed depth image")
    return depth.astype(np.float64) / 1000.0


def spatial_groups(points, max_gap=.02):
    """Split an XY component only at actual gaps between its observed 3D points."""
    pairs = cKDTree(points).query_pairs(max_gap, output_type='ndarray')
    if FAST_COMPONENTS:
        from .fast_components import component_labels
        labels = component_labels(pairs, len(points))
        return [np.flatnonzero(labels == label) for label in np.unique(labels)]
    parent = np.arange(len(points))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a
    labels = np.array([root(i) for i in range(len(points))])
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def oriented_footprint(points_xy):
    """Fit both robust dimensions and midpoint in one PCA coordinate frame."""
    _, axes = np.linalg.eigh(np.cov(np.asarray(points_xy).T))
    major, minor = axes[:, -1], axes[:, 0]
    basis = np.column_stack((major, minor))
    lower, upper = np.percentile(np.asarray(points_xy)@basis, [5, 95], axis=0)
    return major, upper-lower, ((upper+lower)/2)@basis.T


@lru_cache(maxsize=16)
def pixel_grid(height, width, stride):
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    u, v = u.reshape(-1), v.reshape(-1)
    u.setflags(write=False)
    v.setflags(write=False)
    return u, v


def sampled_points(depth, intrinsic, matrix, stride):
    u, v = pixel_grid(*depth.shape, stride)
    z = depth[v, u]
    if FAST_COMPONENTS:
        from .fast_components import project_grid
        points = project_grid(depth, intrinsic, matrix, stride).reshape(-1, 3)
    else:
        camera = np.column_stack(((u-intrinsic[0,2])*z/intrinsic[0,0],
                                  (v-intrinsic[1,2])*z/intrinsic[1,1], z))
        points = camera@matrix[:3,:3].T+matrix[:3,3]
    return u, v, z, points


def metric_candidates(rgb, depth, intrinsic, camera_to_base, config, stride=4,
                      spatial_components=False, hsv=None, samples=None, find_objects=True,
                      component_max_gap_m=.02):
    height, width = depth.shape
    # A 4-pixel grid bounds CPU cost; XY grouping separates overlapping boxes.
    if stride not in (1, 2, 4):
        raise ValueError('Candidate sampling stride must be 1, 2 or 4 pixels')
    if spatial_components and stride != 2:
        raise ValueError('Spatial settlement components require dense 2-pixel sampling')
    if not np.isfinite(component_max_gap_m) or not .012 <= component_max_gap_m <= .02:
        raise ValueError('Component connectivity must be between 12 and 20 mm')
    u, v, z, points = (sampled_points(depth, intrinsic, camera_to_base, stride)
                       if samples is None else samples)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if hsv is None else hsv
    lower, upper = config["table_hsv"]
    green_image = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    green = green_image[v, u] > 0
    valid = np.isfinite(points).all(axis=1) & (z > 0.1) & (z < 3.0)
    valid &= np.all(points >= config["workspace_min"], axis=1)
    valid &= np.all(points <= config["workspace_max"], axis=1)
    table_heights = points[valid & green, 2]
    if len(table_heights) < 100:
        raise ValueError("Insufficient observed green tabletop points")
    histogram, edges = np.histogram(table_heights, bins=np.arange(-0.405, 0.306, 0.005))
    peak = int(np.argmax(histogram))
    plane_z = float(np.median(table_heights[(table_heights >= edges[peak]) & (table_heights < edges[peak + 1])]))
    # Check the fitted surface has real support, not a handful of green pixels.
    if np.count_nonzero(np.abs(table_heights - plane_z) < 0.015) < 100:
        raise ValueError("Observed tabletop height is not stable")
    if not find_objects:
        return [], plane_z
    objects = valid & ~green & (hsv[v, u, 2] >= 60) & (points[:, 2] > plane_z + config["min_object_height_m"])
    objects &= points[:, 2] < plane_z + config["max_object_height_m"]
    selected = points[objects]
    pixels = np.column_stack((u[objects], v[objects]))
    resolution = config["grid_resolution_m"]
    low = np.asarray(config["workspace_min"][:2])
    high = np.asarray(config["workspace_max"][:2])
    grid_wh = np.ceil((high - low) / resolution).astype(int) + 1
    grid = np.zeros((grid_wh[1], grid_wh[0]), np.uint8)
    cells = np.floor((selected[:, :2] - low) / resolution).astype(int)
    if len(cells):
        grid[cells[:, 1], cells[:, 0]] = 255
    # Close holes in sparse depth observations; gaps between parcels are larger
    # than this ~2 cm bridge, unlike gaps in the perspective RGB image.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(grid)
    membership = labels[cells[:, 1], cells[:, 0]] if len(cells) else []
    candidates = []
    members = []
    for label in np.unique(membership):
        if label == 0:
            continue
        indices = np.flatnonzero(np.asarray(membership) == label)
        # The XY closing operation can bridge a resting parcel and fingers
        # that are actually separated in depth. Never fill that 3D gap or
        # infer an unseen face; each child retains all original size checks.
        if spatial_components:
            members.extend(indices[part] for part in spatial_groups(
                selected[indices], max_gap=component_max_gap_m))
        else:
            members.append(indices)
    for member in members:
        group = selected[member]
        image_points = pixels[member]
        if len(group) < config["min_component_points"]:
            continue
        lower, upper = np.percentile(group, [5,95], axis=0)
        extent = upper[:2]-lower[:2]
        major, oriented_extent, oriented_center_xy = oriented_footprint(group[:, :2])
        minor_extent = float(np.min(oriented_extent))
        if (np.max(extent) > config["max_object_extent_m"] or np.min(extent) < 0.015
                or minor_extent > config["max_object_width_m"]):
            continue
        if np.max(extent) < 0.035:
            continue
        # Midpoint of robust XY extents avoids a bias toward the visible face.
        center = (lower+upper)/2
        pmin, pmax = image_points.min(axis=0), image_points.max(axis=0)
        candidates.append({"position_base_m": center.tolist(),
                           # Rotated extents require their own rotated-frame
                           # midpoint. Robust world-axis quantiles generally
                           # have a different midpoint under partial visibility.
                           "oriented_center_base_m": [*oriented_center_xy.tolist(), float(center[2])],
                           "extent_xy_m": extent.tolist(), "points": len(group),
                           "minor_extent_m": minor_extent,
                           "major_axis_xy": major.tolist(),
                           "oriented_extent_m": oriented_extent.tolist(),
                           "top_z_base_m": float(upper[2]),
                           "uv": np.median(image_points, axis=0).tolist(),
                           "bbox": [int(pmin[0]), int(pmin[1]), int(pmax[0] - pmin[0] + 1), int(pmax[1] - pmin[1] + 1)]})
    return candidates, plane_z


class CameraCalibrationEpoch:
    """Fail closed if a calibrated observation source changes its epoch."""
    @staticmethod
    def signature(transform):
        t, q = transform.transform.translation, transform.transform.rotation
        values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        if not np.isfinite(values).all():
            raise ValueError('Camera optical calibration contains nonfinite values')
        return (transform.header.frame_id, transform.child_frame_id, *values)

    def __init__(self, transform, ros_time):
        self.expected = self.signature(transform)
        self.last_ros_time = float(ros_time)
        if not np.isfinite(self.last_ros_time):
            raise ValueError('Camera calibration requires a finite ROS time')
        self.invalid = False

    def check_clock(self, ros_time):
        now = float(ros_time)
        self.invalid |= not np.isfinite(now) or now < self.last_ros_time
        if self.invalid:
            raise RuntimeError('Camera calibration epoch invalidated; use a fresh controller/scene')
        self.last_ros_time = now

    def check_transform(self, transform):
        try:
            same = self.signature(transform) == self.expected
        except ValueError:
            same = False
        self.invalid |= not same
        if self.invalid:
            raise RuntimeError('Camera optical calibration changed after bootstrap')


class RgbdScene:
    def __init__(self, rospy, config):
        import tf2_ros
        from sensor_msgs.msg import CameraInfo, CompressedImage
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
        self.rospy, self.config = rospy, config
        self.lock = threading.Lock()
        self.messages = {}
        self.joint_history = deque(maxlen=1500)
        self.joint_stamps = deque(maxlen=1500)
        self.joint_stamps_ordered = True
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.subscribers = [rospy.Subscriber(config[key], msg_type,
                            lambda msg, kind=kind: self._receive(kind, msg),
                            queue_size=1, buff_size=2**22)
                            for kind, key, msg_type in (("rgb", "rgb_topic", CompressedImage),
                                                      ("depth", "depth_topic", CompressedImage),
                                                      ("info", "info_topic", CameraInfo))]
        self.subscribers.append(rospy.Subscriber("/sensors_data_raw", sensorsData,
                                lambda msg: self._receive("joints", msg), queue_size=1))
        self.description = None
        self.calibration_epoch = None

    def finish_camera_calibration(self):
        """Stop unused dynamic TF updates only after wrist calibration.

        The existing buffer and its static subscription remain intact. Verify
        that the sole subsequent TF query (head mount to optical frame) is
        already exactly available from static TF. Measured head/arm joints and
        public URDF FK continue to provide all moving camera poses.
        """
        if self.calibration_epoch is not None:
            return
        import tf2_ros
        from tf2_msgs.msg import TFMessage
        static_buffer = tf2_ros.Buffer()
        def receive(message):
            for transform in message.transforms:
                static_buffer.set_transform_static(transform, 'camera_calibration_check')
        subscriber = self.rospy.Subscriber('/tf_static', TFMessage, receive, queue_size=100)
        try:
            query = (self.config['camera_mount_frame'], self.config['camera_frame'],
                     self.rospy.Time(0), self.rospy.Duration(.5))
            static = static_buffer.lookup_transform(*query)
            original = self.buffer.lookup_transform(*query)
            if CameraCalibrationEpoch.signature(static) != CameraCalibrationEpoch.signature(original):
                raise RuntimeError('Head optical calibration is not exactly supported by static TF')
            epoch = CameraCalibrationEpoch(original, self.rospy.Time.now().to_sec())
            self.listener.tf_sub.unregister()
            self.calibration_epoch = epoch
        finally:
            subscriber.unregister()

    def camera_mount_matrix(self, message):
        from tf.transformations import euler_matrix, rotation_matrix
        if self.description is None:
            self.description = ET.fromstring(self.rospy.get_param(self.config["description_param"]))
        if not hasattr(self, '_mount_chain'):
            by_child = {joint.find("child").attrib["link"]: joint for joint in self.description.findall("joint")}
            chain = []
            frame = self.config["camera_mount_frame"]
            while frame != self.config["base_frame"]:
                if frame not in by_child or len(chain) > 20:
                    raise ValueError("Camera mount has no URDF chain to the IK base frame")
                joint = by_child[frame]
                chain.append(joint)
                frame = joint.find("parent").attrib["link"]
            self._mount_chain = []
            for joint in reversed(chain):
                origin = joint.find("origin")
                attrs = {} if origin is None else origin.attrib
                transform = euler_matrix(*[float(v) for v in attrs.get("rpy", "0 0 0").split()])
                transform[:3, 3] = [float(v) for v in attrs.get("xyz", "0 0 0").split()]
                moving = joint.attrib["type"] not in ("fixed", "floating")
                axis = [float(v) for v in joint.find("axis").attrib["xyz"].split()] if moving else None
                self._mount_chain.append((transform, joint.attrib["name"], axis))
        q = message.joint_data.joint_q
        yaw_index, pitch_index = self.config["head_joint_indices"]
        values = {"zhead_1_joint": q[yaw_index], "zhead_2_joint": q[pitch_index]}
        # The 14-arm-DOF FK/IK server fixes all non-arm joints. Express camera
        # observations in that same fixed-waist convention, not a moving pelvis
        # frame. Head yaw/pitch remain measured, following the SDK tail indices.
        if not self.config["fixed_ik_waist"]:
            raise ValueError("This aligner requires the arm-only fixed-waist IK convention")
        values["waist_yaw_joint"] = 0.0
        matrix = np.eye(4)
        for transform, name, axis in self._mount_chain:
            matrix = matrix @ transform
            if axis is not None:
                if name not in values:
                    raise ValueError(f"Unmapped moving camera joint {name}")
                matrix = matrix @ rotation_matrix(values[name], axis)
        return matrix

    def _receive(self, kind, message):
        with self.lock:
            self.messages[kind] = (message, time.monotonic())
            if kind == 'joints':
                stamp = message.header.stamp.to_sec()
                if self.joint_stamps and stamp < self.joint_stamps[-1]:
                    self.joint_stamps_ordered = False
                self.joint_stamps.append(stamp)
                self.joint_history.append(message)

    def _nearest_joint(self, stamp):
        # Caller holds the message lock: image and history selection share the
        # same snapshot. Preserve the original earliest-entry tie rule, even
        # for duplicate stamps; unordered histories use the original minimum.
        if not self.joint_stamps_ordered:
            index = min(range(len(self.joint_stamps)), key=lambda i: abs(self.joint_stamps[i]-stamp))
        else:
            from bisect import bisect_left
            right = bisect_left(self.joint_stamps, stamp)
            candidates = (max(0, right-1), min(right, len(self.joint_stamps)-1))
            index = min(candidates, key=lambda i: abs(self.joint_stamps[i]-stamp))
            index = bisect_left(self.joint_stamps, self.joint_stamps[index])
        return self.joint_history[index]

    def head_angles(self):
        with self.lock:
            message = self.messages.get('joints')
        if message is None or time.monotonic() - message[1] > self.config['max_sensor_age_s']:
            raise ValueError('Head joint feedback missing/stale')
        return np.asarray(message[0].joint_data.joint_q)[self.config['head_joint_indices']]

    def read(self, candidate_stride=4, spatial_components=False, find_objects=True, find_surfaces=True,
             component_max_gap_m=.02):
        from tf.transformations import quaternion_matrix
        epoch = getattr(self, 'calibration_epoch', None)
        if epoch is not None:
            epoch.check_clock(self.rospy.Time.now().to_sec())
        with self.lock:
            messages = dict(self.messages)
            joint_message = (self._nearest_joint(messages['rgb'][0].header.stamp.to_sec())
                             if 'rgb' in messages and self.joint_history else None)
        if any(kind not in messages for kind in ("rgb", "depth", "info", "joints")):
            raise ValueError("Waiting for RGB, depth and CameraInfo")
        now = time.monotonic()
        for kind in ("rgb", "depth", "joints"):
            if now - messages[kind][1] > self.config["max_sensor_age_s"]:
                raise ValueError(f"RGB-D {kind} is stale")
        rgb_message, depth_message, info = (messages[kind][0] for kind in ("rgb", "depth", "info"))
        rgb_stamp = rgb_message.header.stamp.to_sec()
        if joint_message is None:
            raise ValueError('Waiting for RGB-aligned joint history')
        joint_dt = abs(joint_message.header.stamp.to_sec() - rgb_stamp)
        if joint_dt > self.config['max_rgb_joint_dt_s']:
            raise ValueError('RGB and head joint timestamps do not match')
        if abs(rgb_message.header.stamp.to_sec() - depth_message.header.stamp.to_sec()) > self.config["max_rgb_depth_dt_s"]:
            raise ValueError("RGB and depth timestamps do not match")
        if info.header.frame_id != self.config["camera_frame"]:
            raise ValueError(f"Unexpected camera optical frame {info.header.frame_id}")
        shared_hit = None
        if SHARED_RGB_DECODER is None:
            bgr = cv2.imdecode(np.frombuffer(rgb_message.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            bgr, shared_hit = SHARED_RGB_DECODER(rgb_message)
        if bgr is None:
            raise ValueError("RGB-D RGB decode failed")
        depth = decode_depth(depth_message)
        if depth.shape != bgr.shape[:2]:
            raise ValueError("Depth is not registered to the RGB image resolution")
        wh = tuple(self.config["image_wh"])
        bgr = cv2.resize(bgr, wh)
        depth = cv2.resize(depth, wh, interpolation=cv2.INTER_NEAREST)
        k = np.asarray(info.K).reshape(3, 3).copy()
        k[0] *= wh[0] / info.width
        k[1] *= wh[1] / info.height
        # Some simulator launches publish the body camera chain only once.
        # Derive that chain from the public URDF and measured joint sensors;
        # retain the official optical calibration from its latched static TF.
        transform = self.buffer.lookup_transform(self.config["camera_mount_frame"], info.header.frame_id,
                                                 self.rospy.Time(0), self.rospy.Duration(0.05))
        if epoch is not None:
            epoch.check_transform(transform)
        q = transform.transform.rotation
        matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
        t = transform.transform.translation
        matrix[:3, 3] = [t.x, t.y, t.z]
        matrix = self.camera_mount_matrix(joint_message) @ matrix
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        samples = sampled_points(depth, k, matrix, candidate_stride)
        grids = {candidate_stride: samples}
        if candidate_stride != 4:
            u4, v4 = pixel_grid(*depth.shape, 4)
            h, w = depth.shape
            factor = 4//candidate_stride
            dense = samples[3].reshape((h+candidate_stride-1)//candidate_stride,
                                       (w+candidate_stride-1)//candidate_stride, 3)
            grids[4] = (u4, v4, depth[v4,u4], dense[::factor,::factor].reshape(-1,3))
        candidates, plane_z = metric_candidates(rgb, depth, k, matrix, self.config,
                                                stride=candidate_stride, spatial_components=spatial_components, hsv=hsv, samples=samples, find_objects=find_objects,
                                                component_max_gap_m=component_max_gap_m)
        return {"candidates": candidates, "table_z_base_m": plane_z, "rgb": rgb, "hsv": hsv, "depth_m": depth,
                "candidate_spatial_components": bool(spatial_components), "sampled_points": grids,
                "component_max_gap_m": float(component_max_gap_m),
                "candidates_evaluated": bool(find_objects), "surfaces_evaluated": bool(find_surfaces),
                "surfaces": (surface_candidates(rgb, depth, k, matrix, plane_z, self.config, hsv=hsv, samples=grids[4]) if find_surfaces else []),
                "intrinsic": k, "camera_to_base": matrix,
                "head_angles_rad": np.asarray(joint_message.joint_data.joint_q)[self.config['head_joint_indices']].tolist(),
                # Keep the arm pose from the same sensor sample as the camera
                # transform. A later servo observation is not the image pose.
                "rgb_joint_positions_rad": list(joint_message.joint_data.joint_q),
                "rgb_joint_stamp_s": joint_message.header.stamp.to_sec(),
                "rgb_joint_dt_s": joint_dt, "shared_rgb_decode_hit": shared_hit,
                "stamp": rgb_message.header.stamp.to_sec()}

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.listener.unregister()


def surface_candidates(rgb, depth, intrinsic, camera_to_base, plane_z, config, stride=4, hsv=None, samples=None):
    """Bright green/yellow planar pad observed on the tabletop."""
    if stride not in (1, 2, 4):
        raise ValueError('Surface stride must be 1, 2 or 4 pixels')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if hsv is None else hsv
    mask = cv2.inRange(hsv, np.array([20, 120, 175], np.uint8), np.array([85, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels = cv2.connectedComponents(mask)
    u, v, z, points = (sampled_points(depth, intrinsic, camera_to_base, stride)
                       if samples is None else samples)
    valid = (z > .1) & (z < 3) & np.isfinite(points).all(axis=1)
    valid &= np.abs(points[:, 2] - plane_z) < .015
    valid &= np.all(points >= config['workspace_min'], axis=1) & np.all(points <= config['workspace_max'], axis=1)
    ids = labels[v, u]
    surfaces = []
    for label in np.unique(ids[valid]):
        if label == 0:
            continue
        group = points[valid & (ids == label)]
        if len(group) < 12:
            continue
        low, high = np.percentile(group[:, :2], [3, 97], axis=0)
        extent = high - low
        if min(extent) < .025 or max(extent) > .30:
            continue
        surfaces.append({'position_base_m': [*((low + high) / 2), plane_z],
                         'extent_xy_m': extent.tolist(), 'points': len(group),
                         'partial': bool(len(group) < 35 or min(extent) < .145),
                         'footprint_xy_m': cv2.convexHull(group[:, :2].astype(np.float32)).reshape(-1, 2).tolist()})
    return surfaces


def snapshot_hsv(snapshot):
    """Reuse the exact conversion attached to this RGB-D observation only."""
    value = snapshot.get('hsv')
    return cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV) if value is None else value
