"""BlurPool — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- anti-aliasing box-filter downsample: the 1-D Pascal-triangle row
  (filt_size 1..7) outer-producted with itself and normalized to a
  (k, k) depthwise kernel, stride downsample with the official
  asymmetric pad [floor((k-1)/2), ceil((k-1)/2)];
- the kernel is a CONSTANT in the official layer (tf.constant, not a
  checkpoint variable) and the layer has NO checkpoint keys; the torch
  implementation therefore keeps the kernel as a plain python array
  (self.a, like the official) and materializes the torch tensor on the
  input's device/dtype per forward. (External A stores it as a
  registered buffer — that would inject a non-official 'k:0' key into
  the Phase 3A Saveable enumeration and break strict official-file
  loads, so it is intentionally not adopted.)

Device/dtype: the convolution runs on the input tensor's device (the
kernel is materialized there); no nn.device dependency, no CUDA calls.

Phase 10B fix (latent Phase 3B defect, no consumer before XSeg): the
channel count is read from the input in its native data format BEFORE
the NHWC boundary permute (official semantics:
``x.shape[nn.conv2d_ch_axis]`` on the raw tensor). The Phase 3B
forward computed it after the permute, which under NHWC indexed the
width axis and broke every NHWC forward (XSeg is the first
BlurPool consumer; the Phase 3B layer test exercised NCHW only).
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
from .LayerBase import LayerBase


class BlurPool(LayerBase):
    def __init__(self, filt_size=3, stride=2, name=None, **kwargs):
        self.filt_size = filt_size
        self.stride = stride

        pad = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad0 = pad[0]
        self.pad1 = pad[1]

        if self.filt_size == 1:
            a = np.array([1.,])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError(f"unsupported BlurPool filt_size {self.filt_size}")

        a = a[:, None] * a[None, :]
        a = a / np.sum(a)
        self.a = a

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # the official kernel is a tf.constant (not a checkpoint
        # variable) -> no parameters/buffers are registered here; the
        # torch kernel is materialized in forward()
        pass

    def forward(self, x):
        # official: the channel count is read from the input in its
        # NATIVE data format (x.shape[nn.conv2d_ch_axis] on the raw
        # tensor) — the NHWC boundary permute below happens AFTER, so
        # the channel axis must be resolved before it (reading it after
        # would index the width axis, breaking every NHWC forward)
        nhwc = (nn.data_format == "NHWC")
        ch = int(x.shape[nn.conv2d_ch_axis])
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()
        k = (
            torch.from_numpy(self.a[None, None, :, :].astype(np.float32))
            .to(device=x.device, dtype=x.dtype)
            .repeat(ch, 1, 1, 1)
        )

        # official asymmetric pad, then VALID depthwise convolution
        x = F.pad(x, (self.pad0, self.pad1, self.pad0, self.pad1),
                  mode="constant", value=0.0)
        x = F.conv2d(x, k, stride=self.stride, padding=0, groups=ch)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : filt_size:{self.filt_size} stride:{self.stride} "
        return r


nn.BlurPool = BlurPool
