"""Optional raw short-skill evidence; alignment and expert filtering are offline.

Camera payloads retain their original compressed bytes and source stamps. Arm
and gripper actions are captured at the actual ROS publishers, after publish()
returns, with both ends of the publication interval. No intended policy output
is substituted for a publication. This module never sends a command itself.
"""
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np


class PublishedCommand:
    def __init__(self, original, recorder, kind, scale):
        self.original, self.recorder = original, recorder
        self.kind, self.scale = kind, scale

    def __getattr__(self, key):
        return getattr(self.original, key)

    def publish(self, message):
        before = self.recorder.clock()
        result = self.original.publish(message)
        after = self.recorder.clock()
        try:
            self.recorder.capture(self.kind, dict(
                values=(np.asarray(message.position, float)*self.scale).tolist(),
                names=list(message.name), publish_before=before,
                source_ros_s=message.header.stamp.to_sec()), stamp=after)
        except Exception as exc:
            self.recorder.record_error(exc)
        return result


class GraspSkillRecorder:
    """Record hover-arrival through externally confirmed trial lift, at source rates."""
    def __init__(self, env, output, image_features):
        import rospy
        from sensor_msgs.msg import CompressedImage, JointState
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
        from kuavo_data.common.config_platform import get_arm_joint_slice

        if env.direct_to_wbc or env.which_arm != 'both' or env.eef_type != 'rq2f85':
            raise ValueError('Skill recording requires the original two-arm ROS/gripper publication path')
        arm_control = env.robot._kuavo_core._control.kuavo_arm_control
        if arm_control.arm_collision_enable:
            raise ValueError('Skill recording cannot label an intermediate collision-filter input as sent action')
        self.ros = rospy
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.lock = threading.RLock()
        self.inbox = queue.Queue(maxsize=4096)
        self.ring = deque()
        self.sequence = 0
        self.active = None
        self.segments = []
        self.last_phase = None
        self.dropped = 0
        self.errors = []
        self.last_sample = {}
        self.closed = False
        self.subscribers = []
        self.wrappers = []
        self.arm_slice = get_arm_joint_slice(env.platform_type)
        joint_map = env.obs_buffer.obs_key_map['joint_q']['handle']['params']['slice']
        gripper_map = env.obs_buffer.obs_key_map['gripper']['handle']['params']['slice']
        start, end = self.arm_slice
        if joint_map != [[start, start+7], [start+7, end]] or gripper_map != [[0, 1], [1, 2]]:
            raise ValueError('Skill recorder must match the installed observation joint/gripper order')
        camera_topics = {feature: env.obs_buffer.obs_key_map[feature.split('.')[-1]]['topic']
                         for feature in image_features}
        metadata = dict(schema_version=1, run_id=rospy.get_param('/run_id'),
            image_features=image_features, camera_topics=camera_topics,
            arm_joint_slice=list(self.arm_slice), platform_type=env.platform_type,
            layout='left7, left_gripper, right7, right_gripper',
            joint_unit='radian', gripper_state_scale=1/.8, gripper_command_scale=1/255.,
            arm_command_scale=float(np.pi/180),
            requested_alignment_hz=10., alignment_clock='ROS source clock',
            simulation_clock_recorded_separately=True,
            ros_uses_sim_time=bool(rospy.get_param('/use_sim_time', False)),
            causal_alignment_required=True, future_images_forbidden=True,
            raw_state_max_receipt_hz=100., preroll_receipt_seconds=2.,
            publication_semantics='successful publish() return, not controller application',
            recording_started=self.clock())
        (self.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        self.writer = threading.Thread(target=self._write, name='scene1-skill-evidence', daemon=True)
        self.writer.start()
        try:
            for feature, topic in camera_topics.items():
                self.subscribers.append(rospy.Subscriber(topic, CompressedImage,
                    self._guarded(self._image), feature, queue_size=4, tcp_nodelay=True))
            self.subscribers.append(rospy.Subscriber('/sensors_data_raw', sensorsData,
                self._guarded(self._state), queue_size=4, tcp_nodelay=True))
            self.subscribers.append(rospy.Subscriber('/gripper/state', JointState,
                self._guarded(self._gripper), queue_size=4, tcp_nodelay=True))
            for owner, name, kind, scale in (
                (arm_control, '_pub_ctrl_arm_traj', 'arm_command', np.pi/180),
                (env, 'pub_eef_joint', 'gripper_command', 1/255.)):
                original = getattr(owner, name)
                setattr(owner, name, PublishedCommand(original, self, kind, scale))
                self.wrappers.append((owner, name, original))
        except Exception:
            self.close('recorder_initialization_failed', 0)
            raise

    def clock(self):
        return dict(ros_s=self.ros.Time.now().to_sec(), monotonic_s=time.monotonic(),
                    wall_unix_s=time.time())

    def record_error(self, exc):
        with self.lock:
            if len(self.errors) < 20:
                self.errors.append(type(exc).__name__+': '+str(exc))

    def _guarded(self, callback):
        def receive(*args):
            try:
                callback(*args)
            except Exception as exc:
                self.record_error(exc)
        return receive

    def _enqueue(self, event, payload=None):
        try:
            self.inbox.put_nowait((event, payload))
        except queue.Full:
            self.dropped += 1

    def capture(self, kind, values, payload=None, stamp=None, downsample=False):
        # Recording failures invalidate labels, never change robot actions.
        stamp = stamp or self.clock()
        with self.lock:
            if self.closed:
                return
            now = stamp['monotonic_s']
            if downsample and now-self.last_sample.get(kind, -float('inf')) < .01:
                return
            self.last_sample[kind] = now
            self.sequence += 1
            event = dict(sequence=self.sequence, kind=kind, receipt=stamp, **values)
            self.ring.append((event, payload))
            while self.ring and self.ring[0][0]['receipt']['monotonic_s'] < now-2.:
                self.ring.popleft()
            if self.active is not None:
                self._enqueue(dict(event, segment_id=self.active['id']), payload)

    def _image(self, msg, feature):
        self.capture('image', dict(feature=feature, format=msg.format,
            source_ros_s=msg.header.stamp.to_sec()), payload=bytes(msg.data))

    def _state(self, msg):
        start, end = self.arm_slice
        sim = msg.sensor_time.to_sec() if hasattr(msg.sensor_time, 'to_sec') else float(msg.sensor_time)
        self.capture('arm_state', dict(values=list(msg.joint_data.joint_q[start:end]),
            source_ros_s=msg.header.stamp.to_sec(), sensor_sim_s=sim), downsample=True)

    def _gripper(self, msg):
        self.capture('gripper_state', dict(values=(np.asarray(msg.position)/.8).tolist(),
            names=list(msg.name), source_ros_s=msg.header.stamp.to_sec()), downsample=True)

    def phase(self, info, parcel_index, step, measured):
        stage, transition = info.get('stage'), info.get('transition_to')
        with self.lock:
            if stage == 'hover' and transition == 'descend':
                if self.active is not None:
                    raise RuntimeError('Overlapping grasp skill boundaries')
                self.active = dict(id=len(self.segments), parcel_index=int(parcel_index),
                    start=self.clock(), start_step=step, success=False,
                    start_evidence=dict(next_goal_tcp=info.get('goal_tcp'),
                                        pregrasp_measured_state=np.asarray(measured).tolist(),
                                        arrival_pose_errors=info.get('pose_errors'),
                                        selected_target=info.get('selected_target')))
                self.segments.append(self.active)
                self.last_phase = None
                for event, payload in self.ring:
                    self._enqueue(dict(event, segment_id=self.active['id'], preroll=True), payload)
                self._enqueue(dict(kind='skill_begin', **self.active))
            if self.active is None:
                return
            phase = (stage, info.get('state'), transition)
            if phase != self.last_phase:
                self._enqueue(dict(kind='phase', segment_id=self.active['id'], receipt=self.clock(),
                    step=step, stage=stage, state=info.get('state'), transition_to=transition))
                self.last_phase = phase
            if stage == 'lift' and transition == 'observe_pad':
                evidence = info.get('lift_evidence')
                if not evidence:
                    raise RuntimeError('Skill success requires existing external lift evidence')
                self.active.update(success=True, end=self.clock(), end_step=step,
                                   lift_evidence=evidence)
                self._enqueue(dict(kind='skill_end', **self.active))
                self.active = None

    def _write(self):
        try:
            with (self.output/'events.jsonl').open('x') as events, (self.output/'images.bin').open('xb') as images:
                while True:
                    item = self.inbox.get()
                    if item is None:
                        break
                    event, payload = item
                    if payload is not None:
                        event = dict(event, image_offset=images.tell(), image_bytes=len(payload))
                        images.write(payload)
                    events.write(json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n')
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))

    def close(self, reason, completed_parcels):
        for owner, name, original in self.wrappers:
            setattr(owner, name, original)
        for subscriber in self.subscribers:
            subscriber.unregister()
        with self.lock:
            if self.closed:
                return
            if self.active is not None:
                self.active.update(end=self.clock(), success=False, failure_reason=reason)
                self._enqueue(dict(kind='skill_end', **self.active))
                self.active = None
            self.closed = True
        # Never block shutdown on a failed writer with a full queue.
        deadline = time.monotonic()+30.
        while self.writer.is_alive() and time.monotonic() < deadline:
            try:
                self.inbox.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.writer.join(timeout=max(0., deadline-time.monotonic()))
        if self.writer.is_alive():
            self.errors.append('writer did not finish within 30 seconds')
        report = dict(reason=reason, independently_confirmed_parcels=completed_parcels,
            segments=self.segments, dropped_events=self.dropped, errors=self.errors,
            raw_recording_valid=self.dropped == 0 and not self.errors,
            expert_labels_created=False)
        (self.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
