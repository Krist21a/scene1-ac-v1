"""Keep classical camera observations in byte form until ACT needs floats."""
import hashlib
from pathlib import Path

import numpy as np


def install():
    import torch
    from kuavo_deploy.utils import obs_buffer
    from . import vision

    expected = 'd72848613dd7d03a9fe50e38e95ec7fa6a39f6b3dd8ce83c35cabfa740037aaa'
    if hashlib.sha256(Path(obs_buffer.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Byte observations require the verified original image preprocessing')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Original camera normalization requires float32')
    # Reproduce the original to_tensor division and visual detector's float32
    # multiplication/truncation for every possible byte intensity. Applying
    # this small lookup table avoids constructing full floating camera frames
    # for classical vision while retaining its exact quantized pixel values.
    values = torch.arange(256, dtype=torch.float32).div(255).mul(255).to(torch.uint8)
    vision.BYTE_RGB_ROUNDTRIP = values.numpy().copy()
    vision.BYTE_RGB_ROUNDTRIP.setflags(write=False)

    def preprocess(self, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Byte camera observations require decoded RGB uint8')
        # Retain the original [1,3,H,W] shape. A view preserves the decoded
        # HWC allocation; camera consumers do not mutate its pixels.
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    obs_buffer.ObsBuffer.img_preprocess = preprocess
    return dict(storage='uint8 CHW view', sdk_source_sha256=expected,
                visual_roundtrip_lut_sha256=hashlib.sha256(vision.BYTE_RGB_ROUNDTRIP.tobytes()).hexdigest(),
                policy_conversion='float32 / 255, contiguous CHW, only at local ACT inference')
