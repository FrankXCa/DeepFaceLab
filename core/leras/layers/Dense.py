"""Dense — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- constructor signature (in_ch, out_ch, use_bias, use_wscale,
  maxout_ch, kernel_initializer, bias_initializer, trainable, dtype,
  **kwargs);
- the weight keeps the OFFICIAL layout (in_ch, out_ch*maxout_ch) in
  torch as well (External A's layout choice, adopted): the official
  TF kernel (in, out) and the torch parameter are then layout-identical,
  so the Phase 4 conversion is the identity; the forward computes
  x @ weight (via F.linear with weight.t()), which is exactly the
  official tf.tensordot(x, weight, axes=1);
- maxout: output is reshaped to (..., out_ch, maxout_ch) and reduced
  with max over the last axis (official);
- use_wscale: fan_in = in_ch, gain 1.0 (official), wscale is a
  non-saved float constant; kernel_initializer=None forces
  random_normal(0, 1.0) (official);
- default kernel initializer: glorot_uniform (the TF get_variable
  default); bias default: zeros.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import torch
import torch.nn.functional as F

from core.leras import nn
from .LayerBase import LayerBase


class Dense(LayerBase):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
        """
        use_wscale          enables weight scale (equalized learning rate)
        maxout_ch           typical 2-4 if you want to enable DenseMaxout behaviour
        """
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # official layout kept in torch: (in_ch, out_ch*maxout_ch)
        out_dim = self.out_ch * (self.maxout_ch if self.maxout_ch > 1 else 1)
        self.weight = torch.nn.Parameter(
            torch.empty(self.in_ch, out_dim, device=nn.device, dtype=self.dtype),
            requires_grad=self.trainable
        )

        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain = 1.0
            fan_in = self.in_ch
            he_std = gain / float(fan_in ** 0.5)  # He init
            # official: tf.constant(he_std) - NOT a checkpoint variable
            self.wscale = he_std
            if kernel_initializer is None:
                # official: forced random_normal(0, 1.0)
                kernel_initializer = nn.initializers.random_normal
        if kernel_initializer is None:
            # TF get_variable default: glorot_uniform
            kernel_initializer = nn.initializers.glorot_uniform
        self.register_param_initializer("weight", kernel_initializer)

        if self.use_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.out_ch, device=nn.device, dtype=self.dtype),
                requires_grad=self.trainable
            )
            bias_initializer = self.bias_initializer
            if bias_initializer is None:
                bias_initializer = nn.initializers.zeros
            self.register_param_initializer("bias", bias_initializer)

    def get_weights(self):
        weights = [self.weight]
        if self.use_bias:
            weights += [self.bias]
        return weights

    # weight/bias keep the official layouts -> identity conversion
    # (inherited from Saveable).

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # x @ weight  (official tf.tensordot(x, weight, axes=1))
        x = F.linear(x, weight.t())

        if self.maxout_ch > 1:
            x = x.view(-1, self.out_ch, self.maxout_ch)
            x = x.max(dim=-1)[0]

        if self.use_bias:
            x = x + self.bias.view(1, self.out_ch)

        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
        return r


nn.Dense = Dense
