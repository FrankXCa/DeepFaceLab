"""TLU — torch implementation of the official leras contract (Phase 3B).

The official DFL "TLU" (docstring references the FRN paper, but the
implementation is the leaky unit):
    y = max(x, tau)
with ``tau`` (in_ch,) a zero-initialized checkpoint variable — the
torch implementation preserves that parameter name exactly, so the
official checkpoint key is ``tau:0``.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


class TLU(LayerBase):
    def __init__(self, in_ch, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.tau = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("tau", nn.initializers.zeros)

    def get_weights(self):
        return [self.tau]

    def forward(self, x):
        shape = (1, self.in_ch, 1, 1) if nn.data_format == "NCHW" else (1, 1, 1, self.in_ch)
        tau = self.tau.view(shape)
        return torch.maximum(x, tau)

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} "
        return r


nn.TLU = TLU
