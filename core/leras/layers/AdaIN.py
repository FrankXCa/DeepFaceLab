"""AdaIN — torch implementation of the official leras contract
(Phase 3B).

Adaptive Instance Normalization (official DFL semantics):
- inputs: (x, mlp); x is the feature map, mlp the style vector
  (N, mlp_ch);
- gamma = mlp @ weight1 + bias1, beta = mlp @ weight2 + bias2
  (weight1/weight2 are (mlp_ch, in_ch) — the TF matmul layout, which
  is identical in torch, so no layout conversion is needed);
- x is instance-normalized over the spatial axes
  ((x - mean) / (std + 1e-5), population std), then scaled by gamma
  and shifted by beta;
- default kernel initializer: he_normal (official tf.initializers.
  he_normal = kaiming_normal, fan_in, relu gain — the registered torch
  initializer below reproduces it exactly); bias defaults: zeros.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


def _he_normal(shape, dtype=None, device=None):
    """TF he_normal: normal, std = sqrt(2 / fan_in) (fan_in = number of
    elements in the input axis — for a 2-D (out, in) kernel this is
    `in` — matching torch kaiming_normal_ mode='fan_in' relu)."""
    dtype = dtype if dtype is not None else (nn.floatx if nn.floatx is not None else torch.float32)
    device = device if device is not None else (nn.device if nn.device is not None else torch.device("cpu"))
    t = torch.empty(shape, device=device, dtype=dtype)
    torch.nn.init.kaiming_normal_(t, mode="fan_in", nonlinearity="relu")
    return t


class AdaIN(LayerBase):
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch
        self.mlp_ch = mlp_ch
        self.kernel_initializer = kernel_initializer
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight1 = torch.nn.Parameter(
            torch.empty(self.mlp_ch, self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.bias1 = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.weight2 = torch.nn.Parameter(
            torch.empty(self.mlp_ch, self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.bias2 = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )

        kernel_initializer = self.kernel_initializer
        if kernel_initializer is None:
            # official: tf.initializers.he_normal()
            kernel_initializer = _he_normal
        self.register_param_initializer("weight1", kernel_initializer)
        self.register_param_initializer("weight2", kernel_initializer)
        self.register_param_initializer("bias1", nn.initializers.zeros)
        self.register_param_initializer("bias2", nn.initializers.zeros)

    def get_weights(self):
        return [self.weight1, self.bias1, self.weight2, self.bias2]

    def forward(self, inputs):
        x, mlp = inputs

        gamma = torch.matmul(mlp, self.weight1) + self.bias1.view(1, self.in_ch)
        beta = torch.matmul(mlp, self.weight2) + self.bias2.view(1, self.in_ch)

        shape = (-1, self.in_ch, 1, 1) if nn.data_format == "NCHW" else (-1, 1, 1, self.in_ch)

        spatial_axes = nn.conv2d_spatial_axes
        x_mean = torch.mean(x, dim=tuple(spatial_axes), keepdim=True)
        # official: population std + 1e-5 outside sqrt
        x_std = torch.sqrt(torch.var(x, dim=tuple(spatial_axes), unbiased=False, keepdim=True)) + 1e-5

        x = (x - x_mean) / x_std
        x = x * gamma.view(shape)
        x = x + beta.view(shape)
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} mlp_ch:{self.mlp_ch} "
        return r


nn.AdaIN = AdaIN
