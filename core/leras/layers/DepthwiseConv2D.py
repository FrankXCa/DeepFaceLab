"""DepthwiseConv2D — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- constructor signature (in_ch, kernel_size, strides, padding,
  depth_multiplier, dilations, use_bias, use_wscale,
  kernel_initializer, bias_initializer, trainable, dtype, **kwargs);
- padding semantics exactly as official ('SAME' -> int
  ((k-1)*d+1)//2, 'VALID' -> 0, int passthrough; explicit constant pad
  + VALID convolution);
- the official tf.depthwise_conv2d kernel layout is
  (kH, kW, in_ch, depth_multiplier); torch runs the same operation as a
  group convolution: weight (in_ch*depth_multiplier, 1, kH, kW) with
  groups=in_ch, where
  torch[c_in*dm + m, 0, h, w] == official[h, w, c_in, m];
- use_wscale as in the official Conv2D (constant float, not saved;
  random_normal(0,1) forced when no kernel_initializer);
- the official docstring default initializer is CA, but in the official
  source the CA line is commented out (glorot_uniform default is what
  actually runs); a caller passing nn.initializers.ca explicitly will
  hit its documented Phase 3B+ placeholder failure at init_weights.

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
from .LayerBase import LayerBase


class DepthwiseConv2D(LayerBase):
    """
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal
    """

    def __init__(self, in_ch, kernel_size, strides=1, padding='SAME', depth_multiplier=1, dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
        if not isinstance(strides, int):
            raise ValueError("strides must be an int type")
        if not isinstance(dilations, int):
            raise ValueError("dilations must be an int type")
        kernel_size = int(kernel_size)

        if isinstance(padding, str):
            if padding == "SAME":
                padding = ((kernel_size - 1) * dilations + 1) // 2
            elif padding == "VALID":
                padding = 0
            else:
                raise ValueError("Wrong padding type. Should be VALID SAME or INT")
        padding = int(padding)

        self.in_ch = in_ch
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx if nn.floatx is not None else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # torch depthwise layout: (in_ch * depth_multiplier, 1, kH, kW)
        self.weight = torch.nn.Parameter(
            torch.empty(self.in_ch * self.depth_multiplier, 1,
                        self.kernel_size, self.kernel_size,
                        device=nn.device, dtype=self.dtype),
            requires_grad=self.trainable
        )

        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else float(2.0 ** 0.5)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / float(fan_in ** 0.5)  # He init
            # official: tf.constant(he_std) - NOT a checkpoint variable
            self.wscale = he_std
            if kernel_initializer is None:
                # official: forced random_normal(0, 1.0)
                kernel_initializer = nn.initializers.random_normal
        if kernel_initializer is None:
            # official runtime default (the CA line is commented out in
            # the official source): glorot_uniform
            kernel_initializer = nn.initializers.glorot_uniform
        self.register_param_initializer("weight", kernel_initializer)

        if self.use_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.in_ch * self.depth_multiplier,
                            device=nn.device, dtype=self.dtype),
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

    # Phase 4 official checkpoint layout conversion:
    # official TF (kH, kW, in_ch, depth_multiplier) -> torch
    # (in_ch*depth_multiplier, 1, kH, kW), exact element map
    # torch[c_in*dm + m, 0, h, w] == official[h, w, c_in, m]
    def convert_weight_layout(self, value, param):
        if param is self.weight:
            k = self.kernel_size
            in_ch, dm = self.in_ch, self.depth_multiplier
            v = torch.as_tensor(value)
            if tuple(v.shape) != (k, k, in_ch, dm):
                # let the strict loader report the exact mismatch
                return v
            # (kH, kW, in, dm) -> (in, dm, kH, kW) -> (in*dm, 1, kH, kW)
            v = v.permute(2, 3, 0, 1).contiguous().view(in_ch * dm, k, k)
            return v.unsqueeze(1)
        return value

    def convert_weight_to_official(self, value, param):
        if param is self.weight:
            # inverse: torch (in*dm, 1, kH, kW) -> official (kH, kW, in, dm)
            import numpy as _np
            k = self.kernel_size
            in_ch, dm = self.in_ch, self.depth_multiplier
            if value.shape != (in_ch * dm, 1, k, k):
                return value
            return (
                np.transpose(
                    np.asarray(value).reshape(in_ch, dm, k, k), (2, 3, 0, 1)
                )
                .copy()
            )
        return value

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # official: explicit constant pad + VALID depthwise convolution
        p = self.padding
        if p:
            x = F.pad(x, (p, p, p, p), mode="constant", value=0.0)

        x = F.conv2d(x, weight,
                     bias=self.bias if self.use_bias else None,
                     stride=self.strides, padding=0,
                     dilation=self.dilations, groups=self.in_ch)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} depth_multiplier:{self.depth_multiplier} "
        return r


nn.DepthwiseConv2D = DepthwiseConv2D
