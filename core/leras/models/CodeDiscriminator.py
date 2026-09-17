"""CodeDiscriminator — torch implementation of the official leras contract
(Phase 3F).

Official behavior preserved (dead official TF reference:
``discriminators_tf.py``):
- constructor signature ``on_build(in_ch, code_res, ch=256,
  conv_kernel_initializer=None)`` (torch: ``__init__(in_ch, code_res,
  ch=256, conv_kernel_initializer=None, name=None)``);
- ``n_downscales = 1 + code_res // 8`` stride-2 convs, the first with
  kernel 4 and the rest with kernel 3 (official), SAME padding, channel
  progression ``ch * min(2**i, 8)``;
- final ``out_conv``: 1x1 VALID, 1 channel;
- forward: leaky_relu(0.1) after each downscale conv, then the raw
  ``out_conv`` output (no activation);
- ``conv_kernel_initializer`` (official name) is forwarded to the Conv2D
  layers as ``kernel_initializer`` (None -> the official glorot_uniform
  default);
- checkpoint naming: the official ``convs`` list is registered under the
  official ``convs_<i>`` module names (never a torch ``ModuleList`` — its
  ``convs/0/...`` keys are not official DFL checkpoint keys), so the torch
  dotted tree names map 1:1 to the official variable names via
  ``core.leras.checkpoint.official_name`` (e.g.
  ``dis/convs_0/conv1...`` -> ``convs_0/weight:0`` under the model's
  discriminator scope).
"""

import torch.nn.functional as F

from core.leras import nn
from core.leras.layers.LayerBase import LayerBase, build_leaf_weights, cascade_init_weights


class CodeDiscriminator(LayerBase):
    def __init__(self, in_ch, code_res, ch=256, conv_kernel_initializer=None, name=None):
        super().__init__(name=name)

        n_downscales = 1 + code_res // 8

        self.convs = []
        prev_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min( (2**i), 8 )
            conv = nn.Conv2D( prev_ch, cur_ch, kernel_size=4 if i == 0 else 3, strides=2, padding='SAME', kernel_initializer=conv_kernel_initializer)
            # official list scope name: 'convs_<i>' (ModelBase._build_sub)
            setattr(self, f"convs_{i}", conv)
            self.convs.append (conv)
            prev_ch = cur_ch

        self.out_conv =  nn.Conv2D( prev_ch, 1, kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)
        build_leaf_weights(self)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu( conv(x), 0.1 )
        return self.out_conv(x)

    def init_weights(self):
        cascade_init_weights(self)


nn.CodeDiscriminator = CodeDiscriminator
