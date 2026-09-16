"""DenseNorm — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- parameter-less RMS-normalization over the last axis:
  x * rsqrt(mean(x^2, axis=-1, keepdims=True) + eps);
- ``eps`` is a constant (not a checkpoint variable) in the official
  layer, so it stays a plain python float here and never appears in
  checkpoints;
- the official layer overrides ``__call__``; the torch equivalent is
  ``forward`` (the LayerBase entry point) — documented adaptation.

Device/dtype: the computation runs on the input's device; no nn.device
dependency (no parameters).
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


class DenseNorm(LayerBase):
    def __init__(self, dense=False, eps=1e-06, dtype=None, name=None, **kwargs):
        self.dense = dense
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype
        # official: tf.constant(eps, dtype, name="epsilon") - NOT a
        # checkpoint variable
        self.eps = float(eps)

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        pass  # no checkpoint variables in the official layer

    def forward(self, x):
        return x * torch.rsqrt(
            torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps
        )

    def __str__(self):
        r = f"{self.__class__.__name__} "
        return r


nn.DenseNorm = DenseNorm
