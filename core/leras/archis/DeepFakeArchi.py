"""DeepFakeArchi — torch implementation of the official leras contract
(Phase 3F).

Official behavior preserved (dead official TF reference: ``archis_tf.py``):
- ``DeepFakeArchi(resolution, use_fp16=False, mod=None, opts=None)`` is a
  factory: when ``mod is None`` it defines the block classes ``Downscale``,
  ``DownscaleBlock``, ``Upscale``, ``ResidualBlock`` and the sub-model
  classes ``Encoder``/``Inter``/``Decoder``, closing over ``conv_dtype``,
  the ``act`` activation, ``use_fp16`` and ``opts`` exactly like the
  official code; the models instantiate ``archi.Encoder/Inter/Decoder``
  and register the instances as their own attributes (``name=...``) —
  the official calling pattern is unchanged;
- the official ``mod`` parameter only has a ``mod is None`` code branch
  (the docstring mentions 'quick', but the official code never implements
  it); the torch factory mirrors the official structure, so any other
  ``mod`` value is a dead end here exactly like it is officially;
- opts semantics (official):
    't'  encoder down1..down5 + res1/res5 chain (5 halvings -> res//32),
         inter without upscale1 (out res = lowest_dense_res), decoder
         upscale0..upscale3 + res0..res3;
    'd'  lowest_dense_res = resolution//32 (else //16), decoder x-head =
         sigmoid of depth_to_space(concat(out_conv, out_conv1..3, ch_axis))
         (2x resolution), extra mask stages upscalem3 ('d' without 't')
         or upscalem4 ('d' with 't');
    'u'  nn.pixel_norm(axes=-1) on the flattened encoder output
         (official epsilon 1e-6, Phase 3D op);
    'c'  CosRelu activation ``x*cos(x)`` (alpha ignored) in place of
         leaky_relu (Downscale/Upscale 0.1, ResidualBlock 0.2 otherwise);
- use_fp16: ``conv_dtype = torch.float16`` (else ``torch.float32``) on the
  Conv2D layers + boundary casts at the encoder/inter/decoder edges,
  exactly like the official ``tf.cast``; this is plain dtype handling —
  NO mixed-precision (AMP) policy is introduced (Phase 3F exclusion);
- checkpoint naming: official list-scope names are reproduced by
  registering list children under ``<attr>_<i>`` module names (the
  official ``ModelBase._build_sub`` names list elements ``f"{name}_{i}"``)
  while keeping the plain ``self.downs`` reference list for the forward
  loop; ``torch.nn.ModuleList`` is deliberately NOT used — its dotted
  index names (``downs/0/...``) would not be official DFL checkpoint keys.
  The torch dotted tree names therefore map 1:1 to the official variable
  names via ``core.leras.checkpoint.official_name``
  (e.g. ``encoder/down1/downs_0/conv1/weight:0``).
- layers are Phase 3B torch layers (official ``weight``/``bias``
  parameter names, official layouts); two-phase lifecycle preserved:
  children are registered into the module tree first, then the
  Conv2D/Dense leaves run ``build_weights()`` (the torch equivalent of
  the official ``ModelBase.build`` / ``build_weights`` phase);
  ``init_weights`` cascades to the whole sub-tree like the official
  archis.
"""

import torch

from core.leras import nn
from core.leras.layers.LayerBase import LayerBase, build_leaf_weights, cascade_init_weights


class DeepFakeArchi(nn.ArchiBase):
    """
    resolution

    mod     None - default
            'quick'

    opts    ''
            ''
            't'
    """
    def __init__(self, resolution, use_fp16=False, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''


        conv_dtype = torch.float16 if use_fp16 else torch.float32

        if 'c' in opts:
            def act(x, alpha=0.1):
                return x*torch.cos(x)
        else:
            def act(x, alpha=0.1):
                return torch.nn.functional.leaky_relu(x, alpha)

        if mod is None:
            class Downscale(LayerBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, name=None):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(name=name)
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME', dtype=conv_dtype)
                    build_leaf_weights(self)

                def forward(self, x):
                    x = self.conv1(x)
                    x = act(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

                def init_weights(self):
                    cascade_init_weights(self)

            class DownscaleBlock(LayerBase):
                def __init__(self, in_ch, ch, n_downscales, kernel_size, name=None):
                    super().__init__(name=name)

                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        d = Downscale(last_ch, cur_ch, kernel_size=kernel_size)
                        # official list scope name: 'downs_<i>' (ModelBase._build_sub)
                        setattr(self, f"downs_{i}", d)
                        self.downs.append (d)
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

                def init_weights(self):
                    cascade_init_weights(self)

            class Upscale(LayerBase):
                def __init__(self, in_ch, out_ch, kernel_size=3, name=None):
                    super().__init__(name=name)
                    self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    build_leaf_weights(self)

                def forward(self, x):
                    x = self.conv1(x)
                    x = act(x, 0.1)
                    x = nn.depth_to_space(x, 2)
                    return x

                def init_weights(self):
                    cascade_init_weights(self)

            class ResidualBlock(LayerBase):
                def __init__(self, ch, kernel_size=3, name=None):
                    super().__init__(name=name)
                    # Runtime FP16 training may opt the late encoder block
                    # into FP32 without changing its fp32 master weights or
                    # checkpoint names. Other blocks keep the official path.
                    self.fp16_fp32_island = False
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    build_leaf_weights(self)

                def forward(self, inp):
                    if self.fp16_fp32_island and torch.is_autocast_enabled('cuda'):
                        with torch.autocast('cuda', enabled=False):
                            return self._forward_block(inp.float())
                    return self._forward_block(inp)

                def _forward_block(self, inp):
                    x = self.conv1(inp)
                    x = act(x, 0.2)
                    x = self.conv2(x)
                    x = act(inp + x, 0.2)
                    return x

                def init_weights(self):
                    cascade_init_weights(self)

            class Encoder(LayerBase):
                def __init__(self, in_ch, e_ch, name=None):
                    self.in_ch = in_ch
                    self.e_ch = e_ch
                    super().__init__(name=name)

                    if 't' in opts:
                        self.down1 = Downscale(self.in_ch, self.e_ch, kernel_size=5)
                        self.res1 = ResidualBlock(self.e_ch)
                        self.down2 = Downscale(self.e_ch, self.e_ch*2, kernel_size=5)
                        self.down3 = Downscale(self.e_ch*2, self.e_ch*4, kernel_size=5)
                        self.down4 = Downscale(self.e_ch*4, self.e_ch*8, kernel_size=5)
                        self.down5 = Downscale(self.e_ch*8, self.e_ch*8, kernel_size=5)
                        self.res5 = ResidualBlock(self.e_ch*8)
                    else:
                        # official: n_downscales=4 if 't' not in opts else 5
                        # (this branch is only reached when 't' is not in opts)
                        self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4, kernel_size=5)

                    build_leaf_weights(self)

                def forward(self, x):
                    if use_fp16:
                        x = x.to(torch.float16)

                    if 't' in opts:
                        x = self.down1(x)
                        x = self.res1(x)
                        x = self.down2(x)
                        x = self.down3(x)
                        x = self.down4(x)
                        x = self.down5(x)
                        x = self.res5(x)
                    else:
                        x = self.down1(x)
                    x = nn.flatten(x)
                    if 'u' in opts:
                        x = nn.pixel_norm(x, axes=-1)

                    if use_fp16:
                        x = x.to(torch.float32)
                    return x

                def get_out_res(self, res):
                    return res // ( (2**4) if 't' not in opts else (2**5) )

                def get_out_ch(self):
                    return self.e_ch * 8

                def init_weights(self):
                    cascade_init_weights(self)

            lowest_dense_res = resolution // (32 if 'd' in opts else 16)

            class Inter(LayerBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, name=None):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(name=name)

                    self.dense1 = nn.Dense( in_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    if 't' not in opts:
                        self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                    build_leaf_weights(self)

                def forward(self, inp):
                    x = inp
                    x = self.dense1(x)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)

                    if use_fp16:
                        x = x.to(torch.float16)

                    if 't' not in opts:
                        x = self.upscale1(x)

                    return x

                def get_out_res(self):
                    return lowest_dense_res * 2 if 't' not in opts else lowest_dense_res

                def get_out_ch(self):
                    return self.ae_out_ch

                def init_weights(self):
                    cascade_init_weights(self)

            class Decoder(LayerBase):
                def __init__(self, in_ch, d_ch, d_mask_ch, name=None):
                    super().__init__(name=name)
                    if 't' not in opts:
                        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

                        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                        self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)

                        self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)

                        if 'd' in opts:
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        else:
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                    else:
                        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
                        self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
                        self.res3 = ResidualBlock(d_ch*2, kernel_size=3)

                        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
                        self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                        self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                        self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)

                        if 'd' in opts:
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        else:
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

                    build_leaf_weights(self)


                def forward(self, z):
                    # official: no input cast here — with use_fp16 the inter
                    # output already arrives in fp16 (Inter casts before its
                    # return); the conv weights are fp16 via conv_dtype
                    x = self.upscale0(z)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)

                    if 't' in opts:
                        x = self.upscale3(x)
                        x = self.res3(x)

                    if 'd' in opts:
                        x = torch.sigmoid( nn.depth_to_space(torch.cat( (self.out_conv(x),
                                                                         self.out_conv1(x),
                                                                         self.out_conv2(x),
                                                                         self.out_conv3(x)), nn.conv2d_ch_axis), 2) )
                    else:
                        x = torch.sigmoid(self.out_conv(x))

                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)

                    if 't' in opts:
                        m = self.upscalem3(m)
                        if 'd' in opts:
                            m = self.upscalem4(m)
                    else:
                        if 'd' in opts:
                            m = self.upscalem3(m)

                    m = torch.sigmoid(self.out_convm(m))

                    if use_fp16:
                        x = x.to(torch.float32)
                        m = m.to(torch.float32)

                    return x, m

                def init_weights(self):
                    cascade_init_weights(self)

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder


nn.DeepFakeArchi = DeepFakeArchi
