"""Read-only robot/PD watchdog; reports to host, does not send control commands.

Time thresholds refer to observed wall-clock duration. This is an experimental
abort guard, not a replacement for native robot protection. No object truth.
"""
import argparse
import json
import math
import threading
import time

import rospy
from kuavo_msgs.msg import sensorsData, jointCmd
from std_msgs.msg import Bool, Float64MultiArray

WRIST = [4, 5, 6, 11, 12, 13]
LIMITS = [66, 75, 57, 75, 14.1, 14.1, 14.1] * 2


class Guard:
    def __init__(self, expect_pd):
        self.expect_pd = expect_pd
        self.lock = threading.Lock()
        self.last = {}
        self.since = {}
        self.counts = {}
        self.peaks = {}
        self.trigger = None

    def condition(self, name, bad, now, value, duration=.3):
        if bad:
            self.counts[name] = self.counts.get(name, 0) + 1
            self.since.setdefault(name, now)
            if now - self.since[name] >= duration and self.trigger is None:
                self.trigger = dict(kind='trigger', reason=name, monotonic_s=now,
                                    start_monotonic_s=self.since[name], duration_s=now-self.since[name], value=value)
        else:
            self.since.pop(name, None)

    def sensor(self, msg):
        with self.lock:
            now = time.monotonic()
            self.last['sensor'] = now
            q = msg.imu_data.quat
            roll = math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y))
            pitch = math.asin(max(-1., min(1., 2*(q.w*q.y-q.z*q.x))))
            tilt = max(abs(roll), abs(pitch))
            self.peaks['tilt_rad'] = max(self.peaks.get('tilt_rad', 0), tilt)
            # Conservative experimental abort at 20 degrees; native limits unchanged.
            self.condition('base_tilt_20deg', tilt > math.radians(20), now, [roll, pitch], .1)
            tau = list(msg.joint_data.joint_torque[13:27])
            vel = list(msg.joint_data.joint_v[13:27])
            if len(tau) != 14 or len(vel) != 14:
                self.condition('invalid_sensor_width', True, now, [len(tau), len(vel)], 0)
                return
            if not all(math.isfinite(v) for v in tau + vel + [roll, pitch]):
                self.condition('nonfinite_sensor', True, now, None, 0)
                return
            ratio = max(abs(v)/b for v, b in zip(tau, LIMITS))
            self.peaks['actuator_torque_ratio'] = max(self.peaks.get('actuator_torque_ratio', 0), ratio)
            self.peaks['arm_velocity_rad_s'] = max(self.peaks.get('arm_velocity_rad_s', 0), max(map(abs, vel)))
            self.condition('actuator_near_limit', ratio >= .99, now, ratio)

    def command(self, msg):
        with self.lock:
            now = time.monotonic()
            self.last['command'] = now
            tau = list(msg.tau[13:27])
            if len(tau) != 14 or not all(map(math.isfinite, tau)):
                self.condition('invalid_command', True, now, len(tau), 0)
                return
            ratio = max(abs(v)/b for v, b in zip(tau, LIMITS))
            self.peaks['command_torque_ratio'] = max(self.peaks.get('command_torque_ratio', 0), ratio)
            self.condition('command_near_limit', ratio >= .99, now, ratio)

    def pd(self, msg):
        with self.lock:
            now = time.monotonic()
            self.last['pd'] = now
            x = list(msg.data)
            if len(x) != 57 or not all(map(math.isfinite, x)):
                self.condition('invalid_pd_diagnostic', True, now, len(x), 0)
                return
            if x[56] <= .5:
                self.condition('pd_added_limit', False, now, None)
                self.condition('pd_total_limit', False, now, None)
                return
            raw = max(abs(x[14+j]) for j in WRIST)
            total = max(abs(x[j]+max(-.5, min(.5, x[14+j]))) for j in WRIST)
            self.peaks['pd_raw_nm'] = max(self.peaks.get('pd_raw_nm', 0), raw)
            self.condition('pd_added_limit', raw > .5+1e-9, now, raw)
            self.condition('pd_total_limit', total > 14.1+1e-9, now, total)

    def stop_signal(self, msg):
        if msg.data:
            with self.lock:
                self.condition('native_stop_robot', True, time.monotonic(), True, 0)

    def velocity_reference(self, msg):
        with self.lock:
            now = time.monotonic()
            self.last['velocity_reference'] = now
            x = list(msg.data)
            if len(x) != 47 or not all(map(math.isfinite, x)):
                self.condition('invalid_velocity_reference', True, now, len(x), 0)
                return
            raw = max(abs(x[14+j]) for j in WRIST)
            self.peaks['wrist_reference_raw_rad_s'] = max(self.peaks.get('wrist_reference_raw_rad_s', 0), raw)
            self.condition('wrist_reference_velocity_clipping', x[43] > .5 and x[44] > .5, now,
                           dict(raw=raw, limit=x[45]))

    def snapshot(self):
        with self.lock:
            return dict(kind='heartbeat', monotonic_s=time.monotonic(), last=dict(self.last),
                        counts=dict(self.counts), peaks=dict(self.peaks), trigger=self.trigger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-pd', action='store_true')
    ap.add_argument('--expect-velocity', action='store_true')
    a = ap.parse_args()
    rospy.init_node('scene1_contact_readonly_guard', anonymous=True)
    guard = Guard(a.expect_pd)
    subs = [rospy.Subscriber('/sensors_data_raw', sensorsData, guard.sensor, queue_size=8, tcp_nodelay=True),
            rospy.Subscriber('/joint_cmd', jointCmd, guard.command, queue_size=8, tcp_nodelay=True),
            rospy.Subscriber('/stop_robot', Bool, guard.stop_signal, queue_size=8)]
    if a.expect_pd:
        subs.append(rospy.Subscriber('/scene1_tracking/wrist_pd', Float64MultiArray, guard.pd, queue_size=8, tcp_nodelay=True))
    if a.expect_velocity:
        subs.append(rospy.Subscriber('/scene1_tracking/wrist_velocity_reference', Float64MultiArray, guard.velocity_reference, queue_size=8, tcp_nodelay=True))
    required = ['sensor', 'command'] + (['pd'] if a.expect_pd else []) + (['velocity_reference'] if a.expect_velocity else [])
    start = time.monotonic()
    ready = False
    while not rospy.is_shutdown():
        snapshot = guard.snapshot()
        now = snapshot['monotonic_s']
        if snapshot['trigger']:
            print(json.dumps(snapshot['trigger']), flush=True)
            return
        if all(now-snapshot['last'].get(k, -1e10) < .5 for k in required):
            if not ready:
                ready = True
                print(json.dumps(dict(kind='ready', monotonic_s=now, topics=required)), flush=True)
        elif ready or now-start > 10:
            print(json.dumps(dict(kind='trigger', reason='observation_stale', monotonic_s=now,
                                  ages={k: now-snapshot['last'].get(k, 0) for k in required})), flush=True)
            return
        print(json.dumps(snapshot), flush=True)
        time.sleep(.1)


if __name__ == '__main__':
    main()
