"""Read-only, bounded ROS trace. Raw field names; no object truth topics."""
import argparse
import collections
import json
import queue
import time

import rospy
import rostopic

TOPICS = [
    '/sensors_data_raw', '/joint_cmd', '/kuavo_arm_traj',
    '/humanoid_mpc_arm_commanded', '/humanoid_mpc_target_arm',
    '/humanoid_mpc_observation', '/humanoid/mpc/arm_control_mode',
    '/humanoid_controller/optimizedState_mrt_origin',
    '/humanoid_controller/optimizedInput_mrt_origin',
    '/humanoid_controller/optimizedState_wbc_mrt_origin',
    '/humanoid_controller/optimizedInput_wbc_mrt_origin',
    '/humanoid_controller/wbc_planned_joint_acc',
    '/humanoid_controller/torque',
    '/humanoid_controller/measuredRbdState_/joint_pos',
    '/humanoid_controller/measuredRbdState_/joint_vel',
    '/humanoid_controller/resetting_mpc_state_',
    '/monitor/time_cost/controller_loop_time', '/monitor/time_cost/wbc',
    '/monitor/time_cost/mpc', '/gripper/state',
]


def plain(value):
    if hasattr(value, '__slots__'):
        return {s: plain(getattr(value, s)) for s in value.__slots__}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--duration', type=float, default=900)
    ap.add_argument('--hz', type=float, default=50)
    args = ap.parse_args()
    rospy.init_node('scene1_tracking_readonly', anonymous=True)
    inbox = queue.Queue(maxsize=4096)
    counts = collections.Counter()
    last = {}
    subscribers = []

    def callback(msg, topic):
        now = time.monotonic_ns()
        counts['received:' + topic] += 1
        # Arm publications are sparse events and retained without downsampling.
        interval = 0 if topic in TOPICS[2:5] else int(1e9 / args.hz)
        if now - last.get(topic, 0) < interval:
            return
        last[topic] = now
        row = dict(topic=topic, callback_monotonic_ns=now,
                   receipt_ros_ns=rospy.Time.now().to_nsec(), message=plain(msg))
        try:
            inbox.put_nowait(row)
        except queue.Full:
            counts['dropped:' + topic] += 1

    with open(args.output, 'x', buffering=1024*1024) as f:
        def emit(x):
            f.write(json.dumps(x, separators=(',', ':')) + '\n')
        metadata = dict(kind='metadata', run_id=rospy.get_param('/run_id'),
                        timestamp_semantics='callback arrival, not controller application; unstamped vectors cannot be exactly synchronized',
                        max_hz=args.hz, topics={}, params={k: rospy.get_param(k, None) for k in
                        ['/robot_version', '/taskFile', '/referenceFile', '/is_rl_start', '/use_external_mpc', '/wbc_frequency']})
        for topic in TOPICS:
            cls, _, _ = rostopic.get_topic_class(topic, blocking=False)
            if cls is None:
                metadata['topics'][topic] = {'missing': True}
                continue
            metadata['topics'][topic] = {'type': cls._type, 'md5': cls._md5sum}
            subscribers.append(rospy.Subscriber(topic, cls, callback, topic, queue_size=4, tcp_nodelay=True))
        emit(metadata)
        f.flush()
        print(json.dumps(metadata), flush=True)
        start = last_flush = time.monotonic()
        try:
            while not rospy.is_shutdown() and time.monotonic() - start < args.duration:
                try:
                    emit(inbox.get(timeout=.1))
                except queue.Empty:
                    pass
                if time.monotonic() - last_flush > 1:
                    f.flush()
                    last_flush = time.monotonic()
        finally:
            for sub in subscribers:
                sub.unregister()
            while not inbox.empty():
                emit(inbox.get_nowait())
            emit(dict(kind='counts', counts=dict(counts)))


if __name__ == '__main__':
    main()
