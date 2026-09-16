"""InstanceNorm2D — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- per-instance statistics over the spatial axes:
  x_std = reduce_std(x, spatial_axes) + 1e-5 (population std, added
  OUTSIDE the sqrt — the official formula), then
  (x - x_mean) / x_std * weight + bias;
- weight (in_ch,) initialized with glorot_uniform (the official uses
  tf.initializers.glorot_uniform); the torch initializer reproduces
  TF's exact 1-D glorot rule (fan_in = fan_out = n,
  limit = sqrt(6/n)) because torch's xavier_uniform_ needs >=2-D
  inputs and its (1, n) view rule differs from TF's 1-D rule;
- bias (in_ch,) zeros.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


def _glorot_uniform_1d(shape, dtype=None, device=None):
    """TF glorot_uniform for a 1-D shape: for 1-D tensors TF uses
    fan_in = fan_out = number of elements, so limit = sqrt(6/n) and the
    values are uniform in [-limit, limit]. (torch's xavier_uniform_
    needs >=2-D inputs and its fan_avg for a (1, n) view differs from
    TF's 1-D rule, so the exact TF formula is used instead.)"""
    dtype = dtype if dtype is not None else (nn.floatx if nn.floatx is not None else torch.float32)
    device = device if device is not None else (nn.device if nn.device is not None else torch.device("cpu"))
    n = 1
    for s in shape:
        n *= int(s)
    limit = float((6.0 / float(n)) ** 0.5)
    t = torch.empty(shape, device=device, dtype=dtype)
    return torch.nn.init.uniform_(t, -limit, limit)


class InstanceNorm2D(LayerBase):
    def __init__(self, in_ch, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("weight", _glorot_uniform_1d)

        self.bias = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("bias", nn.initializers.zeros)

    def get_weights(self):
        return [self.weight, self.bias]

    def forward(self, x):
        shape = (1, self.in_ch, 1, 1) if nn.data_format == "NCHW" else (1, 1, 1, self.in_ch)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)

        spatial_axes = nn.conv2d_spatial_axes
        x_mean = torch.mean(x, dim=tuple(spatial_axes), keepdim=True)
        # official: population std (unbiased=False) + 1e-5 outside sqrt
        x_std = torch.sqrt(torch.var(x, dim=tuple(spatial_axes), unbiased=False, keepdim=True)) + 1e-5

        x = (x - x_mean) / x_std
        x = x * weight + bias
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} "
        return r


nn.InstanceNorm2D = InstanceNorm2D
