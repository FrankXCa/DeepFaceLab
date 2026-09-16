"""BatchNorm2D — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- the official layer is explicitly "not for training": the forward
  applies the SAVED running statistics (no moment updates, no training
  mode); the torch forward is the same manual formula
  (x - running_mean) / sqrt(running_var + eps) * weight + bias;
- weight (ones) / bias (zeros) are checkpoint variables;
  running_mean (zeros) / running_var (ZEROs — confirmed against the
  official source, matching External A) are non-trainable checkpoint
  variables -> torch buffers (enumerated in the same order, so the
  official checkpoint keys weight:0/bias:0/running_mean:0/
  running_var:0 come out in the official order);
- eps (default 1e-5) and momentum are stored config (the official
  stores them too; momentum is unused by the inference-only forward).

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


class BatchNorm2D(LayerBase):
    """
    currently not for training
    """

    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, name=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(
            torch.empty(self.dim, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("weight", nn.initializers.ones)

        self.bias = torch.nn.Parameter(
            torch.empty(self.dim, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("bias", nn.initializers.zeros)

        # non-trainable checkpoint variables in the official layer ->
        # torch buffers (kept in the Saveable enumeration on purpose)
        self.register_buffer(
            "running_mean",
            torch.zeros(self.dim, device=nn.device, dtype=self.dtype),
        )
        # official initializes running_var with zeros (inference-only BN)
        self.register_buffer(
            "running_var",
            torch.zeros(self.dim, device=nn.device, dtype=self.dtype),
        )

    def get_weights(self):
        return [self.weight, self.bias, self.running_mean, self.running_var]

    def forward(self, x):
        shape = (1, self.dim, 1, 1) if nn.data_format == "NCHW" else (1, 1, 1, self.dim)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)
        running_mean = self.running_mean.view(shape)
        running_var = self.running_var.view(shape)

        # official inference-only formula (no statistics updates)
        x = (x - running_mean) / torch.sqrt(running_var + self.eps)
        x = x * weight + bias
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : dim:{self.dim} "
        return r


nn.BatchNorm2D = BatchNorm2D
