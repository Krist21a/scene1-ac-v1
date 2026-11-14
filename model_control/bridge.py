"""Local, serialized model/ROS bridge. Run inside the algorithm container.

No task policy runs here. Each task decision is supplied by the model client.
"""
from collections import deque
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socketserver
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serializable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def finite_vector(value, size):
    import numpy as np
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'Expected {size} finite numbers')
    return a


class Bridge:
    def __init__(self, config_path, output):
        import numpy as np
        import rospy
        import rosgraph
        import yaml
        self.rospy = rospy
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.log = (output / 'events.jsonl').open('a', buffering=1)
        self.started = time.monotonic()
        self.log_lock = threading.Lock()
        self.event_seq = 0
        self.feedback = deque(maxlen=512)
        self.session_id = uuid.uuid4().hex
        self.admission = threading.RLock()
        self.closing = False
        self.completed = {}
        self.command_observations = {}
        self.worker = None
        self.runtime = None
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.send_gate = threading.Lock()
        self.results = {}
        self.observations = {}
        self.active = None
        self.progress = {}
        self.active_signature = None
        self.fault = None
        self.sent_motion = False
        self.env = self.scene = self.watchdog = None
        config = yaml.safe_load(Path(config_path).read_text())
        # Raised payloads must remain complete in obstacle segmentation.
        # This expands sensing only; workspace and motion limits are unchanged.
        config['cartesian_alignment']['max_object_height_m'] = .35
        self.config = config
        sys.path.insert(0, config['project_root'])
        from kuavo_deploy.config import load_kuavo_config
        from scene1.kinematics import UrdfArmKinematics
        from scene1.rgbd import RgbdScene
        from run_scene1 import SensorWatchdog
        from scene1.motion import MotionFilter, reset_for_manipulation
        base = load_kuavo_config(config['base_config'])
        if (base.env.inference_env, base.env.control_mode, base.env.which_arm, base.env.only_arm) != ('sim', 'joint', 'both', True):
            raise RuntimeError('Only scene1 simulation with both arms is supported')
        if not rosgraph.is_master_online():
            raise RuntimeError('ROS master unavailable; initialize official scene1 first')
        rospy.init_node('kuavo_deploy', anonymous=True)
        self.check_publishers()
        from model_control.runtime_support import initialize_environment, HoldPublisher
        self.env = initialize_environment(base, rospy)
        topic_map = {e['topic']: e['msg_type'] for e in base.env.obs_key_map.values() if 'topic' in e}
        self.watchdog = SensorWatchdog(rospy, topic_map, config['sensor_timeout_s'])
        self.scene = RgbdScene(rospy, config['cartesian_alignment'])
        self.kin = UrdfArmKinematics(rospy)
        from model_control.self_geometry import RobotSurfaceFilter
        self.surface_filter = RobotSurfaceFilter(self.kin, rospy.get_param("/humanoid_description"), config["cartesian_alignment"])
        deadline = time.monotonic()+config['startup_timeout_s']
        while not self.watchdog.ready():
            if time.monotonic() >= deadline:
                self.watchdog.check()
            time.sleep(.05)
        self.runtime = HoldPublisher(self.env, self.watchdog, hz=10.,
                                     on_sample=lambda measured, sent, tick: self.event('control_sample', measured=measured, sent=sent, monotonic_s=tick))
        self.hold = self.runtime.read()
        from model_control.motion_adapter import MotionAdapter
        self.adapter = MotionAdapter(config, self.kin, self.env.action_space.low,
                                     self.env.action_space.high, self.hold)
        self.event('startup', config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                   urdf_sha256=hashlib.sha256(rospy.get_param('/humanoid_description').encode()).hexdigest(),
                   mode='model_cartesian_primitives', task_success=None,
                   perception=dict(max_object_height_m=.35, component_max_gap_m=.012))

    def event(self, kind, **data):
        with self.log_lock:
            self.event_seq += 1
            record = dict(seq=self.event_seq, kind=kind, wall_time=time.time(), elapsed_s=time.monotonic()-self.started, **data)
            self.log.write(json.dumps(record, allow_nan=False, default=serializable)+'\n')
            if kind in ('command', 'progress', 'result', 'observation_ready', 'observation_failed', 'stop'):
                self.feedback.append(record)

    def check_publishers(self):
        import rosgraph
        standby = {'/arms_ik_node', '/sim_leju_claw_interface'}
        for topic, nodes in rosgraph.Master(self.rospy.get_name()).getSystemState()[0]:
            if topic in ('/kuavo_arm_traj', '/gripper/command'):
                others = [n for n in nodes if n != self.rospy.get_name() and n not in standby and not n.startswith('/challenge_task_scene1')]
                if others:
                    raise RuntimeError(f'Competing command publishers: {topic}: {others}')

    def measured(self):
        return self.runtime.read()

    def fresh_snapshot(self):
        deadline = time.monotonic()+1.
        while True:
            self.runtime.check()
            try:
                started = time.monotonic()
                snapshot = self.scene.read(candidate_stride=2, spatial_components=True, component_max_gap_m=.012)
                captured = time.monotonic()
                snapshot = self.surface_filter.apply(snapshot, self.measured(), self.runtime.reference())
                snapshot["processing_s"] = dict(rgbd=captured-started, self_filter=time.monotonic()-captured)
                return snapshot
            except ValueError as exc:
                if ('timestamps do not match' not in str(exc) and 'joint timestamps' not in str(exc)) or time.monotonic() >= deadline:
                    raise
                time.sleep(.02)

    def observe(self, request):
        import cv2
        import numpy as np
        snap = self.fresh_snapshot()
        measured = self.measured()
        oid = uuid.uuid4().hex
        path = self.output / (oid+'.png')
        if not cv2.imwrite(str(path), cv2.cvtColor(snap['rgb'], cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Could not save observation')
        np.savez_compressed(self.output/(oid+'.npz'), depth_m=snap['depth_m'], intrinsic=snap['intrinsic'], camera_to_base=snap['camera_to_base'], rgb_joint_positions_rad=snap['rgb_joint_positions_rad'], rgb_joint_stamp_s=snap['rgb_joint_stamp_s'])
        self.observations = {oid: (time.monotonic(), measured, snap)}
        poses = self.kin.fk(measured)
        result = dict(session_id=self.session_id, rgb_joint_positions_rad=snap['rgb_joint_positions_rad'], rgb_joint_stamp_s=snap['rgb_joint_stamp_s'], rgb_joint_dt_s=snap['rgb_joint_dt_s'], surfaces=snap['surfaces'], head_angles_rad=snap['head_angles_rad'], observation_id=oid, image=str(path), depth=str(self.output/(oid+'.npz')),
                      source_stamp_s=snap['stamp'], state=measured.tolist(), frame='base_link',
                      left=vars(poses.left_pose), right=vars(poses.right_pose),
                      intrinsic=snap['intrinsic'].tolist(), camera_to_base=snap['camera_to_base'].tolist(),
                      candidates=snap['candidates'], table_z_base_m=snap['table_z_base_m'])
        if 'pixel' in request:
            uv = finite_vector(request['pixel'], 2)
            u, v = map(int, uv)
            h, w = snap['depth_m'].shape
            if not (0 <= u < w and 0 <= v < h):
                raise ValueError('Pixel outside image')
            z = float(snap['depth_m'][v, u])
            if not math.isfinite(z) or z <= 0:
                raise ValueError('No valid depth at pixel')
            k = snap['intrinsic']
            p = np.array([(u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z, 1.])
            result['point_base_m'] = (snap['camera_to_base']@p)[:3].tolist()
        from scene1.visual_cycle import bin_rim
        result['bin_rim'] = bin_rim(snap)
        result['robot_surface_filter'] = snap.get('robot_surface_filter')
        result['unfiltered_candidates'] = snap.get('unfiltered_candidates')
        if hasattr(self, 'adapter'):
            result['motion_state'] = self.adapter.describe(measured, snap)
        self.event('observation', **result)
        return result

    def execute(self, request):
        import numpy as np
        from scene1.motion import JOINTS
        if self.fault:
            raise RuntimeError(self.fault)
        self.runtime.check()
        self.check_publishers()
        if self.cancel.is_set():
            return dict(status='stopped')
        if request['op'] == 'set_head':
            angles = finite_vector(request['angles_rad'], 2)
            with self.send_gate:
                if self.cancel.is_set():
                    return dict(status='stopped')
                self.runtime.set_head(*angles)
            time.sleep(.5)
            return dict(status='commanded', angles_rad=angles.tolist())
        oid = request.get('observation_id')
        if oid not in self.observations:
            raise ValueError('Unknown observation; call observe')
        captured, initial, snapshot = self.observations[oid]
        if time.monotonic()-captured > 60:
            raise ValueError('Observation older than 60 seconds; observe again')
        state = self.measured()
        if np.max(np.abs(state[JOINTS]-initial[JOINTS])) > .03:
            raise RuntimeError('Robot moved since observation; observe again')
        duration = float(request.get('timeout_s', 30))
        if not math.isfinite(duration) or not 0 < duration <= 60:
            raise ValueError('timeout_s must be in (0,60]')
        self.adapter.sent(self.runtime.reference())
        self.adapter.begin(request, state, snapshot)
        deadline = time.monotonic()+duration
        result = dict(status='timeout')
        previous_phase = 'executing'
        while time.monotonic() < deadline:
            if self.cancel.is_set():
                result = dict(status='stopped')
                break
            tick = time.monotonic()
            snap = self.fresh_snapshot()
            observed = time.monotonic()
            state = self.measured()
            command, info = self.adapter.step(state, snap)
            computed = time.monotonic()
            phase = info.get('phase', 'executing')
            progress = dict(phase=phase)
            if 'reason' in info:
                progress['reason'] = info['reason'].split(':', 1)[0]
            for key in ('waiting_s', 'observation_hold_s', 'resumed_after_observation_hold_s'):
                if key in info:
                    progress[key] = info[key]
            with self.log_lock:
                self.progress = progress
            if phase != previous_phase:
                self.event('progress', command_id=request['command_id'], **progress)
                previous_phase = phase
            if self.cancel.is_set():
                result = dict(status='stopped')
                break
            with self.send_gate:
                if self.cancel.is_set():
                    result = dict(status='stopped')
                    break
                self.sent_motion = True
                self.runtime.send(command)
            self.adapter.sent(command)
            self.event('step', measured=state.tolist(), sent=command.tolist(), info=info,
                       timing_s=dict(snapshot=observed-tick, adapter=computed-observed,
                                     publish=time.monotonic()-computed, **snap.get('processing_s', {})))
            result = info
            if info['status'] != 'running':
                break
        else:
            result = dict(status='timeout', last_info=result)
        return result

    def dispatch(self, request):
        # Admission and cancellation share one linearization point. Workers never
        # hold this lock while reading sensors or saving observations.
        with self.admission:
            if self.closing:
                raise RuntimeError('Bridge is closing')
            if request.get('op') not in ('status', 'observe') and request.get('session_id') != self.session_id:
                raise ValueError('session_id missing or stale; call status/observe')
            observing = request.get('op') == 'observe' and self.lock.acquire(blocking=False)
            if request.get('op') == 'observe':
                result = dict(status='busy', active=self.active)
            else:
                result = self._dispatch(request)
        if observing:
            try:
                result = self.observe(request)
            finally:
                self.lock.release()
        if isinstance(result, threading.Event):
            result.wait()
            with self.log_lock:
                result = self.results[request['command_id']][1]
        return dict(result, session_id=self.session_id)

    def _dispatch(self, request):
        op = request.get('op')
        if op == 'command_observation':
            with self.log_lock:
                return self.command_observations.get(request.get('command_id'), dict(status='unknown'))
        if op in ('move_ee', 'set_gripper', 'set_head'):
            cid = request.get('command_id')
            if not isinstance(cid, str) or not 1 <= len(cid) <= 128:
                raise ValueError('command_id is required, 1..128 characters')
            if type(request.get('async', False)) is not bool:
                raise ValueError('async must be boolean')
            signature = json.dumps(request, sort_keys=True, allow_nan=False)
            with self.log_lock:
                entry = self.results.get(cid)
                if entry is not None:
                    if entry[0] != signature:
                        raise ValueError('command_id reused with different parameters')
                    return entry[1]
                if self.active == cid:
                    if self.active_signature != signature:
                        raise ValueError('command_id reused with different parameters')
                    return dict(status='running', command_id=cid)
        if op == 'events':
            after = request.get('after', 0)
            if type(after) is not int or after < 0:
                raise ValueError('after must be a nonnegative integer')
            with self.log_lock:
                return dict(events=[e for e in self.feedback if e['seq'] > after],
                            cursor=self.event_seq, oldest_seq=self.feedback[0]['seq'] if self.feedback else None)
        if op == 'result':
            cid = request.get('command_id')
            with self.log_lock:
                entry = self.results.get(cid)
                return entry[1] if entry else dict(command_id=cid,
                    status='running' if self.active == cid else 'unknown',
                    **(self.progress if self.active == cid else {}))
        if op == 'stop':
            self.cancel.set()
            if self.runtime is not None:
                with self.send_gate:
                    self.runtime.stop()
            self.event('stop', active=self.active)
            return dict(status='stop_requested', active=self.active)
        if op == 'status':
            return dict(active=self.active, fault=self.fault, runtime=self.runtime.snapshot(), mode='model_cartesian_primitives',
                        elapsed_s=time.monotonic()-self.started)
        if not self.lock.acquire(blocking=False):
            return dict(status='busy', active=self.active)
        handed_off = False
        try:
            if op == 'observe':
                return self.observe(request)
            if op not in ('move_ee', 'set_gripper', 'set_head'):
                raise ValueError('Unknown operation')
            cid = request.get('command_id')
            if not isinstance(cid, str) or not 1 <= len(cid) <= 128:
                raise ValueError('command_id is required, 1..128 characters')
            signature = json.dumps(request, sort_keys=True, allow_nan=False)
            if cid in self.results:
                old, result = self.results[cid]
                if old != signature:
                    raise ValueError('command_id reused with different parameters')
                return result
            self.cancel.clear()
            with self.log_lock:
                self.active_signature = signature
                self.active = cid
                self.progress = dict(phase='executing')
            self.sent_motion = False
            self.event('command', request=request)
            self.completed[cid] = threading.Event()
            with self.log_lock:
                self.command_observations[cid] = dict(status='pending', command_id=cid)
            handed_off = True
            self.worker = threading.Thread(target=self.finish_command, args=(request, cid, signature))
            try:
                self.worker.start()
            except Exception:
                handed_off = False
                self.active = None
                raise
            if request.get('async', False):
                return dict(status='accepted', command_id=cid)
            return self.completed[cid]

        finally:
            # finish_command owns the lock once a command is admitted.
            if not handed_off:
                self.lock.release()

    def finish_command(self, request, cid, signature):
        started = time.monotonic()
        try:
            try:
                result = self.execute(request)
            except Exception as exc:
                result = dict(status='rejected_or_failed', error=str(exc))
            try:
                if self.sent_motion and result.get('status') != 'reached':
                    self.runtime.stop()
                    self.adapter.sent(self.runtime.reference())
                self.adapter.active = None
            except Exception as exc:
                self.fault = 'Could not establish reference hold: '+str(exc)
            result.update(command_id=cid, execution_s=time.monotonic()-started,
                          observation_status='pending', session_id=self.session_id)
            with self.log_lock:
                self.results[cid] = (signature, result)
            self.event('result', result=result)
            self.completed[cid].set()
            # Keep the single-operation lock until the final snapshot is saved;
            # result/status/stop remain available throughout this phase.
            try:
                observation = self.observe({})
                response = dict(status='ready', command_id=cid, observation=observation)
                kind = 'observation_ready'
                detail = dict(observation_id=observation['observation_id'])
            except Exception as exc:
                response = dict(status='failed', command_id=cid, error=str(exc))
                kind = 'observation_failed'
                detail = dict(error=str(exc))
            with self.log_lock:
                self.command_observations[cid] = response
            self.event(kind, command_id=cid, **detail)
            return result

        finally:
            self.active = None
            self.lock.release()

    def close(self):
        with self.admission:
            self.closing = True
            self.cancel.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join()
        if self.runtime is not None:
            self.runtime.close()
        for resource in (self.scene, self.watchdog, self.env):
            if resource is not None:
                resource.close()
        self.log.close()


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(70)
        try:
            line = self.rfile.readline(65537)
            if len(line) > 65536:
                raise ValueError('Request too large')
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('Expected JSON object')
            result = self.server.bridge.dispatch(request)
        except Exception as exc:
            result = dict(status='error', error=str(exc))
        self.wfile.write((json.dumps(result, allow_nan=False, default=serializable)+'\n').encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/visual_cycle_four_parcels.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--socket', default='/tmp/scene1-model.sock')
    args = parser.parse_args()
    os.umask(0o077)
    with open('/tmp/scene1-speed-trial.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path(args.socket).exists():
            raise RuntimeError('Socket exists; inspect the existing bridge before removing it')
        bridge = Bridge.__new__(Bridge)
        try:
            bridge.__init__(args.config, args.output)
        except BaseException:
            if hasattr(bridge, 'env'):
                bridge.close()
            raise
        try:
            with Server(args.socket, Handler) as server:
                server.bridge = bridge
                server.timeout = .2
                while not bridge.rospy.is_shutdown():
                    server.handle_request()
        finally:
            bridge.close()
            Path(args.socket).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
