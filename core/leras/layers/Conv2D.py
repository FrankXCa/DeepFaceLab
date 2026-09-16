"""Conv2D — torch implementation of the official leras contract (Phase 3B).

Official behavior preserved:
- constructor signature (in_ch, out_ch, kernel_size, strides, padding,
  dilations, use_bias, use_wscale, kernel_initializer, bias_initializer,
  trainable, dtype, **kwargs);
- padding semantics: 'SAME' -> int ((k-1)*d+1)//2, 'VALID' -> 0, int
  passthrough (the official layer performs the SAME->int conversion in
  __init__ itself, so the runtime behavior is exactly this int padding,
  applied symmetrically and followed by a VALID convolution);
- use_wscale: the official wscale is a tf.constant (NOT a checkpoint
  variable) -> stored as a plain python float here as well; when
  wscale is enabled and kernel_initializer is None, initialization is
  forced to random_normal(0, 1.0) (official);
- default kernel initializer (no wscale): the TF get_variable default,
  glorot_uniform (torch: xavier_uniform_ — same distribution);
- weight is stored in torch layout (out_ch, in_ch, kH, kW); the official
  HWIO checkpoint layout (kH, kW, in_ch, out_ch) is converted by
  ``convert_weight_layout`` (Phase 4), explicit axis map
  transpose(3,2,0,1);
- two-phase lifecycle: build_weights() creates the parameters and
  registers the initializers; init_weights() applies them.

Device/dtype: created on nn.device with nn.floatx via the Phase 2
abstraction; no torch.cuda.* call in this module. NHWC inputs are
handled by boundary permutation (official DFL call sites use NCHW).
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
from .LayerBase import LayerBase


class Conv2D(LayerBase):
    """
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal
    """

    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
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
        self.out_ch = out_ch
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
        # torch layout: (out_ch, in_ch, kH, kW)
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size,
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

    # Phase 4 official checkpoint layout conversion:
    # official TF (kH, kW, in_ch, out_ch) -> torch (out_ch, in_ch, kH, kW)
    def convert_weight_layout(self, value, param):
        if param is self.weight:
            # explicit axis map, no element-count heuristics
            return torch.as_tensor(value).permute(3, 2, 0, 1).contiguous()
        return value

    def convert_weight_to_official(self, value, param):
        if param is self.weight:
            # inverse: torch (out,in,kH,kW) -> official (kH,kW,in,out)
            return np.transpose(value, (2, 3, 1, 0)).copy()
        return value

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # official: explicit constant pad + VALID convolution
        p = self.padding
        if p:
            x = F.pad(x, (p, p, p, p), mode="constant", value=0.0)

        x = F.conv2d(x, weight, bias=self.bias if self.use_bias else None,
                     stride=self.strides, padding=0, dilation=self.dilations)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
        return r


nn.Conv2D = Conv2D
