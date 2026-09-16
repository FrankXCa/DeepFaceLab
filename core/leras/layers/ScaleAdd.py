"""ScaleAdd — torch implementation of the official leras contract
(Phase 3B).

Residual-connection helper used by the official archis:
    inputs = (x0, x1);  y = x0 + x1 * weight
with ``weight`` (ch,) a zero-initialized checkpoint variable (the same
name as in the official layer, so the checkpoint key is ``weight:0``).
Starting from zeros, a fresh ScaleAdd passes x0 through unchanged and
gradually mixes in x1 — the official semantics.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


class ScaleAdd(LayerBase):
    def __init__(self, ch, dtype=None, name=None, **kwargs):
        self.ch = ch
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(
            torch.empty(self.ch, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("weight", nn.initializers.zeros)

    def get_weights(self):
        return [self.weight]

    def forward(self, inputs):
        shape = (1, self.ch, 1, 1) if nn.data_format == "NCHW" else (1, 1, 1, self.ch)
        weight = self.weight.view(shape)

        x0, x1 = inputs
        x = x0 + x1 * weight
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : ch:{self.ch} "
        return r


nn.ScaleAdd = ScaleAdd
