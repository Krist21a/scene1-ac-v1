"""Share only JPEG decoding of identical ROS messages between two consumers."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np


class SharedRGBDecoder:
    def __init__(self):
        self.frames = OrderedDict()
        self.lock = threading.Lock()

    def decode(self, message):
        # Keep strong references so object IDs cannot be recycled while cached.
        # ROS delivers the same immutable message to subscribers in this process;
        # timestamp equality alone would not establish identical payloads.
        key = id(message)
        with self.lock:
            cached = self.frames.get(key)
            if cached is not None and cached[0] is message:
                self.frames.move_to_end(key)
                return cached[1], True
            bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                bgr.setflags(write=False)
                self.frames[key] = (message, bgr)
                while len(self.frames) > 4:
                    self.frames.popitem(last=False)
            return bgr, False


def install():
    from kuavo_deploy.utils import obs_buffer
    from . import rgbd

    # This adapter preserves the installed callback's decode -> RGB -> resize
    # -> img_preprocess -> append sequence. Refuse a different SDK implementation
    # rather than silently omit any newly introduced preprocessing.
    expected = 'd72848613dd7d03a9fe50e38e95ec7fa6a39f6b3dd8ce83c35cabfa740037aaa'
    if hashlib.sha256(Path(obs_buffer.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Shared RGB decoder requires the verified original observation callback')
    original = obs_buffer.ObsBuffer.rgb_callback
    decoder = SharedRGBDecoder()

    def callback(self, message, key, handle):
        if key != 'head_cam_h':
            return original(self, message, key, handle)
        bgr, _ = decoder.decode(message)
        if bgr is None:
            raise ValueError('Failed to decode compressed image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resize_wh = handle.get('params', {}).get('resize_wh', None)
        if resize_wh:
            rgb = cv2.resize(rgb, resize_wh)
        data = self.img_preprocess(rgb)
        self._append_data(key, data, message.header.stamp.to_sec())

    obs_buffer.ObsBuffer.rgb_callback = callback
    rgbd.SHARED_RGB_DECODER = decoder.decode
