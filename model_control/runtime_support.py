"""Single wall-clock SDK writer and measured-state cache for model control.

There is no policy here: references originate in the bridge/adapter. A stalled
ROS clock cannot block this writer in rospy.Rate.sleep because it never uses
Env.step. Failure latches; callers must inspect and restart rather than clear it.
"""
import math
import threading
import time

import numpy as np

from scene1.motion import action_vector


def initialize_environment(base, rospy):
    """Create simulation SDK without bag playback, head reset or gripper reset."""
    from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
    from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    if (base.env.inference_env, base.env.control_mode, base.env.which_arm,
            base.env.only_arm, base.env.direct_to_wbc) != ('sim', 'joint', 'both', True, False):
        raise RuntimeError('Model bridge requires simulation, joint/both/only_arm and direct_to_wbc=false')
    env = KuavoSimEnv(base)
    try:
        rospy.wait_for_service('/enable_wbc_arm_trajectory_control', timeout=5)
        service = rospy.ServiceProxy('/enable_wbc_arm_trajectory_control', changeArmCtrlMode)
        request = changeArmCtrlModeRequest()
        request.control_mode = 0
        if not service(request).result:
            raise RuntimeError('MPC arm routing service rejected control_mode=0')
        if not env.robot.set_external_control_arm_mode():
            raise RuntimeError('SDK rejected external arm control mode')
        # Only Env.step needs reset accounting. Initialize it for diagnostic
        # compatibility without resetting head, fingers, or recorded targets.
        env.sleep_time = env.average_sleep_time = 0
        return env
    except BaseException:
        env.close()
        raise


class HoldPublisher:
    """One SDK writer. All reads return copies from that same worker.

    send() waits until one publication; it does not report physical arrival.
    stop() confirms another publication of the last guarded reference, including
    its steady-state tracking compensation and commanded fingers. The bridge
    cancels further adapter integration before calling stop(). This holds the
    last target; it does not promise immediate physical arrest.
    """
    def __init__(self, env, watchdog, hz=10, on_sample=None):
        if not math.isfinite(hz) or not 1 <= hz <= 20:
            raise ValueError('Wall-clock command frequency must be 1..20 Hz')
        if env.direct_to_wbc:
            raise RuntimeError('HoldPublisher supports normal MPC routing only')
        self.env, self.watchdog = env, watchdog
        self.hz, self.on_sample = float(hz), on_sample
        self._condition = threading.Condition()
        self._quit = threading.Event()
        self._ready = threading.Event()
        self._reference = None
        self._measured = None
        self._measured_at = None
        self._head_pending = None
        self._sequence = self._published_sequence = 0
        self._fault = None
        self._closed = False
        deadline = time.monotonic()+max(5., float(watchdog.timeout)+1)
        while not watchdog.ready():
            if time.monotonic() >= deadline or watchdog.rospy.is_shutdown():
                watchdog.check()
                raise RuntimeError('ROS stopped during writer startup')
            time.sleep(.02)
        self._thread = threading.Thread(target=self._run, name='model-sdk-writer', daemon=True)
        self._thread.start()
        if not self._ready.wait(max(5., float(watchdog.timeout)+1)):
            self._quit.set()
            raise RuntimeError('SDK writer did not initialize within startup timeout')
        self.check()

    def _fail(self, exc):
        with self._condition:
            self._fault = str(exc)
            self._condition.notify_all()
        self._ready.set()

    def _run(self):
        period = 1/self.hz
        last_clock = None
        last_clock_change = time.monotonic()
        try:
            while not self._quit.is_set():
                tick = time.monotonic()
                self.watchdog.check()
                clock = self.watchdog.rospy.Time.now().to_sec()
                if last_clock is not None and clock < last_clock:
                    raise RuntimeError('ROS clock moved backwards; scene reset invalidates command session')
                if clock != last_clock:
                    last_clock_change, last_clock = tick, clock
                elif tick-last_clock_change > self.watchdog.timeout:
                    raise RuntimeError('ROS clock stopped; no new action sent')
                measured = action_vector(self.env.get_obs()['observation.state'].detach().cpu().numpy().reshape(-1))
                self.watchdog.check()
                if time.monotonic()-tick > self.watchdog.timeout:
                    raise RuntimeError('SDK feedback read exceeded sensor freshness budget')
                with self._condition:
                    self._measured, self._measured_at = measured, time.monotonic()
                    if self._reference is None:
                        self._reference = measured.copy()
                    action = self._reference.copy()
                    head = self._head_pending
                    self._head_pending = None
                    sequence = self._sequence
                    # Hold the condition through the short publish operations:
                    # stop cannot be acknowledged before an older send finishes.
                    if head is not None:
                        if self.env.robot.control_head(*head) is False:
                            raise RuntimeError('SDK rejected head command')
                    self.env.exec_action(action)
                    self._published_sequence = sequence
                    self._condition.notify_all()
                self._ready.set()
                if self.on_sample is not None:
                    self.on_sample(measured.copy(), action.copy(), time.monotonic())
                self._quit.wait(max(0., period-(time.monotonic()-tick)))
        except BaseException as exc:
            self._fail(exc)

    def check(self):
        with self._condition:
            if self._fault is not None:
                raise RuntimeError('SDK writer fault: '+self._fault)
            if self._closed:
                raise RuntimeError('SDK writer is closed')
            if self._measured_at is not None and time.monotonic()-self._measured_at > self.watchdog.timeout:
                raise RuntimeError('SDK writer feedback cache is stale')

    def read(self):
        self.check()
        with self._condition:
            if self._measured is None:
                raise RuntimeError('SDK feedback not ready')
            return self._measured.copy()

    def reference(self):
        self.check()
        with self._condition:
            return self._reference.copy()

    def _wait(self, sequence, timeout):
        deadline = time.monotonic()+timeout
        while self._published_sequence < sequence:
            self.check()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                self._fault = 'SDK command publication timed out'
                self._quit.set()
                self._condition.notify_all()
                raise RuntimeError(self._fault)
            self._condition.wait(remaining)
        return sequence

    def send(self, action, timeout=2):
        action = action_vector(action)
        if np.any(action < self.env.action_space.low) or np.any(action > self.env.action_space.high):
            raise ValueError('Reference exceeds SDK action limits; refusing silent clipping')
        with self._condition:
            self.check()
            self._reference = action
            self._sequence += 1
            return self._wait(self._sequence, timeout)

    def set_head(self, yaw_rad, pitch_rad, timeout=2):
        # The public SDK and bridge both consume radians (SDK robot.py control_head).
        angles = np.asarray([yaw_rad, pitch_rad], dtype=float)
        if not np.isfinite(angles).all() or np.any(np.abs(np.rad2deg(angles)) > [70., 25.]):
            raise ValueError('Head target exceeds yaw +/-70deg, pitch +/-25deg')
        with self._condition:
            self.check()
            self._head_pending = angles.tolist()
            self._sequence += 1
            return self._wait(self._sequence, timeout)

    def stop(self, timeout=2):
        """Keep the last guarded reference; wait for its next publication.

        Replacing this reference with measured joints discards accumulated
        gravity/tracking compensation and can make a loaded hand sag. The
        caller must cancel new adapter commands before invoking this method.
        """
        with self._condition:
            self.check()
            self._head_pending = None
            self._sequence += 1
            return self._wait(self._sequence, timeout)

    def snapshot(self):
        with self._condition:
            return dict(sequence=self._sequence, published_sequence=self._published_sequence,
                        fault=self._fault, alive=self._thread.is_alive(),
                        feedback_age_s=None if self._measured_at is None else time.monotonic()-self._measured_at,
                        hz=self.hz)

    def close(self):
        if self._closed:
            return
        try:
            if self._thread.is_alive() and self._fault is None:
                self.stop()
        finally:
            self._quit.set()
            self._thread.join(timeout=3)
            self._closed = True
            if self._thread.is_alive():
                raise RuntimeError('SDK writer did not stop; do not close SDK resources underneath it')
