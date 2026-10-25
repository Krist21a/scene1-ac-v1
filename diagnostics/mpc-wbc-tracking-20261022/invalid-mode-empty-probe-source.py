"""Small empty-hand ramps in freshly reset official scene, normal MPC route.

Never use on a held object. No gripper publisher, scene or object state access.
Uses actual current arm pose as a gentle entry, then deterministic fixed poses.
"""
import argparse
import json
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kuavo_msgs.msg import sensorsData
from kuavo_msgs.srv import changeArmCtrlMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rospy.init_node('scene1_empty_tracking_probe', anonymous=True)
    assert rospy.get_param('/robot_version') == 52
    assert not rospy.get_param('/is_rl_start')
    state = [None]
    def cb(msg):
        state[0] = msg
    sub = rospy.Subscriber('/sensors_data_raw', sensorsData, cb, queue_size=1)
    first = rospy.wait_for_message('/sensors_data_raw', sensorsData, timeout=10)
    q0 = np.asarray(first.joint_data.joint_q)[13:27]
    # A new official scene has hanging empty arms. Reject prepared/held poses.
    assert np.max(np.abs(q0)) < .6, q0
    for name, mode in [('/enable_wbc_arm_trajectory_control', 0), ('/humanoid_change_arm_ctrl_mode', 2)]:
        rospy.wait_for_service(name, timeout=10)
        response = rospy.ServiceProxy(name, changeArmCtrlMode)(mode)
        assert response.result, (name, response)
    pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1)
    time.sleep(.5)
    base = np.array([.15, 0, 0, -.4, 0, 0, 0] * 2)
    targets = []
    for name, indices, values in [('proximal', [0, 7], [.15, .15]),
                                   ('distal', [4, 5, 6, 11, 12, 13], [.18, .15, .15, -.18, .15, -.15])]:
        q = base.copy()
        q[indices] += values
        targets.extend([(name + '_out', q), (name + '_return', base.copy())])
    schedule = [('entry', base)] + targets
    rate = rospy.Rate(20)
    previous = q0.copy()
    with open(args.output, 'x') as log:
        log.write(json.dumps(dict(kind='start', run_id=rospy.get_param('/run_id'), initial=q0.tolist(),
                                  probe='fixed empty arm ramps, 2s ramp + 5s hold', task_success=None))+'\n')
        for name, target in schedule:
            start = time.monotonic()
            log.write(json.dumps(dict(kind='segment', name=name, start_monotonic_ns=time.monotonic_ns(),
                                      start_ros_ns=rospy.Time.now().to_nsec(), source=previous.tolist(), target=target.tolist()))+'\n')
            log.flush()
            while time.monotonic() - start < 7:
                elapsed = time.monotonic() - start
                q = previous + min(1., elapsed / 2.) * (target - previous)
                if state[0] is not None:
                    observed = np.asarray(state[0].joint_data.joint_q)[13:27]
                    velocity = np.asarray(state[0].joint_data.joint_v)[13:27]
                    if np.max(abs(observed-q)) > .5 or np.max(abs(velocity)) > 2.:
                        raise RuntimeError('Probe tracking/velocity stop guard')
                m = JointState()
                m.header.stamp = rospy.Time.now()
                m.position = np.rad2deg(q).tolist()
                pub.publish(m)
                log.write(json.dumps(dict(kind='command', monotonic_ns=time.monotonic_ns(), ros_ns=m.header.stamp.to_nsec(), q_rad=q.tolist(), segment=name))+'\n')
                rate.sleep()
            previous = target.copy()
        log.write(json.dumps(dict(kind='complete', monotonic_ns=time.monotonic_ns()))+'\n')
    sub.unregister()


if __name__ == '__main__':
    main()
