"""Bound only the SDK's latest-value odometry subscription queue.

Sensor histories, MPC/gait callbacks, publishers and the installed SDK files
remain as supplied. The private constructor binding cannot alter other ROS
subscribers created elsewhere in the process.
"""
from functools import update_wrapper
import hashlib
import json
from pathlib import Path
from types import FunctionType


def install():
    import rospy
    from kuavo_humanoid_sdk.kuavo.core.ros import state

    expected = 'b64d8523df6eff85445875ac95cee07dae5b5b21c373f47a4d6b2db9abba7fb9'
    if hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Latest odometry queue requires the verified SDK state implementation')
    cls = state.KuavoRobotStateCore
    if getattr(getattr(cls, '_instance', None), '_initialized', False):
        raise RuntimeError('Odometry queue configuration must precede SDK initialization')
    original = cls.__init__
    if not isinstance(original, FunctionType) or original.__globals__.get('rospy') is not rospy:
        raise RuntimeError('Unsupported SDK constructor ROS binding')

    class ROSBinding:
        def __getattr__(self, name):
            return getattr(rospy, name)

        def Subscriber(self, topic, message, callback, *args, **kwargs):
            if topic != '/odom':
                return rospy.Subscriber(topic, message, callback, *args, **kwargs)
            if args or kwargs or callback.__name__ != '_odom_callback':
                raise RuntimeError('Unexpected SDK odometry subscription signature')
            subscriber = rospy.Subscriber(topic, message, callback, queue_size=1)
            if subscriber.impl.queue_size != 1:
                raise RuntimeError('SDK odometry queue was not configured')
            print('LATEST_SDK_ODOMETRY '+json.dumps(dict(topic=topic, queue_size=1,
                sdk_source_sha256=expected, callback='original latest-value odometry callback')), flush=True)
            return subscriber

    namespace = dict(original.__globals__, rospy=ROSBinding())
    replacement = FunctionType(original.__code__, namespace, original.__name__,
                               original.__defaults__, original.__closure__)
    replacement.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(replacement, original)
    cls.__init__ = replacement
