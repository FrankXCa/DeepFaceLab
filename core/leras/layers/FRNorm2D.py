"""FRNorm2D — torch implementation of the official leras contract
(Phase 3B).

Filter Response Normalization (arXiv 1911.09737), as implemented by
official DeepFaceLab:
    nu2 = mean(x^2, spatial_axes, keepdims)
    x = x * (1 / sqrt(nu2 + abs(eps))) * weight + bias
(note: the official variant does NOT subtract the mean).

Official behavior preserved:
- weight (in_ch,) initialized with ones; bias (in_ch,) zeros;
- ``eps`` is a (1,)-shaped checkpoint VARIABLE in the official layer
  (constant 1e-6, trainable by TF default) -> a torch Parameter of
  shape (1,) here, so the official `eps:0` checkpoint key is preserved;
- no layout differences (all 1-D channel vectors) -> the inherited
  identity conversion hooks apply.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch

from core.leras import nn
from .LayerBase import LayerBase


class FRNorm2D(LayerBase):
    """
    Tensorflow implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """

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
        self.register_param_initializer("weight", nn.initializers.ones)

        self.bias = torch.nn.Parameter(
            torch.empty(self.in_ch, device=nn.device, dtype=self.dtype)
        )
        self.register_param_initializer("bias", nn.initializers.zeros)

        # official: tf.get_variable("eps", (1,), initializer=constant(1e-6))
        # - a checkpoint variable, created with its value directly
        # (no registered initializer needed; the construction-time
        # value stands, per the Phase 3A lifecycle contract)
        self.eps = torch.nn.Parameter(
            torch.full((1,), 1e-6, device=nn.device, dtype=self.dtype)
        )

    def get_weights(self):
        return [self.weight, self.bias, self.eps]

    def forward(self, x):
        shape = (1, self.in_ch, 1, 1) if nn.data_format == "NCHW" else (1, 1, 1, self.in_ch)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)

        spatial_axes = nn.conv2d_spatial_axes
        nu2 = torch.mean(torch.square(x), dim=tuple(spatial_axes), keepdim=True)
        x = x * (1.0 / torch.sqrt(nu2 + torch.abs(self.eps)))

        return x * weight + bias

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} "
        return r


nn.FRNorm2D = FRNorm2D
