"""Conv2DTranspose — torch implementation of the official leras contract
(Phase 3B).

Official behavior preserved:
- constructor signature (in_ch, out_ch, kernel_size, strides, padding,
  use_bias, use_wscale, kernel_initializer, bias_initializer,
  trainable, dtype, **kwargs);
- output sizing is the official ``deconv_length`` (SAME -> in*stride,
  VALID -> in*stride + (k-stride)_+, FULL -> in*stride - (stride+k-2));
  the torch equivalent is an F.conv_transpose2d with
  output_padding = target - base, where
  base = (in-1)*stride - 2*padding + kernel_size. If the required
  output_padding falls outside [0, stride) the configuration cannot be
  represented and an explicit ValueError is raised (no silent repair;
  all official DFL configs — k=3/4, stride 2, SAME/VALID — satisfy it);
- SAME padding uses padding=(k-1)//2 (the TF SAME transpose pad);
- use_wscale as in the official Conv2D (constant float, not saved;
  random_normal(0,1) forced when no kernel_initializer);
- weight is stored in torch layout (in_ch, out_ch, kH, kW); the official
  TF filter layout (kH, kW, out_ch, in_ch) is converted by
  ``convert_weight_layout`` (Phase 4), explicit axis map
  torch[in,out,h,w] = official[h,w,out,in] (permute(3,2,0,1)).

Device/dtype: nn.device / nn.floatx via the Phase 2 abstraction.
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
from .LayerBase import LayerBase


class Conv2DTranspose(LayerBase):
    """
    use_wscale      enables weight scale (equalized learning rate)
                    if kernel_initializer is None, it will be forced to random_normal
    """

    def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
        if not isinstance(strides, int):
            raise ValueError("strides must be an int type")
        kernel_size = int(kernel_size)

        if padding not in ('SAME', 'VALID', 'FULL'):
            raise ValueError("Wrong padding type. Should be SAME VALID or FULL")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
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
        # torch layout: (in_ch, out_ch, kH, kW)
        self.weight = torch.nn.Parameter(
            torch.empty(self.in_ch, self.out_ch, self.kernel_size, self.kernel_size,
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
    # official TF filter (kH, kW, out_ch, in_ch) -> torch (in_ch, out_ch, kH, kW)
    def convert_weight_layout(self, value, param):
        if param is self.weight:
            # explicit axis map, no element-count heuristics:
            # torch[in, out, h, w] = official[h, w, out, in]
            return torch.as_tensor(value).permute(3, 2, 0, 1).contiguous()
        return value

    def convert_weight_to_official(self, value, param):
        if param is self.weight:
            # inverse: torch (in,out,kH,kW) -> official (kH,kW,out,in)
            return np.transpose(value, (2, 3, 1, 0)).copy()
        return value

    def deconv_length(self, dim_size, stride_size, kernel_size, padding):
        assert padding in {'SAME', 'VALID', 'FULL'}
        if dim_size is None:
            return None
        if padding == 'VALID':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'FULL':
            dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
        elif padding == 'SAME':
            dim_size = dim_size * stride_size
        return dim_size

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # padding per the official TF semantics of tf.nn.conv2d_transpose:
        # SAME pads the input by floor((k-1)/2); VALID/FULL do not.
        p = (self.kernel_size - 1) // 2 if self.padding == 'SAME' else 0

        in_h, in_w = int(x.shape[2]), int(x.shape[3])
        target_h = self.deconv_length(in_h, self.strides, self.kernel_size, self.padding)
        target_w = self.deconv_length(in_w, self.strides, self.kernel_size, self.padding)

        base_h = (in_h - 1) * self.strides - 2 * p + self.kernel_size
        base_w = (in_w - 1) * self.strides - 2 * p + self.kernel_size
        out_pad_h = int(target_h - base_h)
        out_pad_w = int(target_w - base_w)

        # F.conv_transpose2d requires 0 <= output_padding < stride; the
        # official deconv_length sizing always satisfies this for the
        # configurations DeepFaceLab uses (k=3/4, stride 2, SAME/VALID).
        if not (0 <= out_pad_h < self.strides) or not (0 <= out_pad_w < self.strides):
            raise ValueError(
                f"Conv2DTranspose: required output_padding=({out_pad_h},{out_pad_w}) "
                f"is outside [0, {self.strides}) for input ({in_h},{in_w}) with "
                f"kernel {self.kernel_size}, stride {self.strides}, padding "
                f"{self.padding!r}; this configuration cannot be represented"
            )

        x = F.conv_transpose2d(
            x, weight,
            bias=self.bias if self.use_bias else None,
            stride=self.strides, padding=p,
            output_padding=(out_pad_h, out_pad_w),
        )

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
        return r


nn.Conv2DTranspose = Conv2DTranspose
