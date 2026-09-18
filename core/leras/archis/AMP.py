"""AMP — torch implementation of the official AMP archi (Phase 7).

Official behavior preserved (dead official TF reference:
``models/Model_AMP/Model_tf.py`` L138-255, the ``on_initialize`` inline
classes; AMP = the official morphable-pair model, NOT Automatic Mixed
Precision):

- ``AMPArchi(resolution, e_ch, ae_ch, inter_ch, inter_res, d_ch,
  d_mask_ch, use_fp16=False, input_ch=3)`` is a factory: it defines the
  block classes ``Downscale``/``Upscale``/``ResidualBlock`` and the
  sub-model classes ``Encoder``/``Inter``/``Decoder``; the AMP model
  instantiates ``archi.Encoder()`` / ``archi.Inter()`` /
  ``archi.Decoder()`` (NO per-submodel dimension arguments — exactly
  like the official ``on_build(self)`` no-arg constructors, which close
  over the ``on_initialize`` variables) and registers the instances as
  its own attributes (``name=...``) — the official calling pattern is
  unchanged. The factory closes over every dimension the official
  inline classes read (input_ch/e_ch/ae_ch/inter_ch/inter_res/d_ch/
  d_mask_ch/resolution/use_fp16/conv_dtype);
- ``Downscale``: Conv2D(k5, strides 2, SAME, conv_dtype) ->
  leaky_relu 0.1 (official L138-143); ``Upscale``: Conv2D(out*4, k3,
  SAME, conv_dtype) -> leaky_relu 0.1 -> depth_to_space(2) (L145-151);
  ``ResidualBlock``: conv1/conv2 (ch->ch, SAME, conv_dtype),
  leaky_relu 0.2 after conv1 and after the residual add (L153-163);
- ``Encoder``: down1/res1/down2/down3/down4/down5/res5 chain (k5,
  input_ch -> e_ch*8) + dense1 ( ((res//32)**2) * e_ch*8 -> ae_ch );
  forward = [conv_dtype input cast if use_fp16], chain, [float32 cast
  back], pixel_norm(flatten(x), axes=-1) (official 1e-6 epsilon via the
  Phase 3D op), dense1 -> flat (N, ae_ch) code (L165-190). The AMP
  encoder ALWAYS pixel-normalizes and owns dense1 (the SAEHD/DeepFake
  archi keeps dense1 in the inter and gates pixel_norm on 'u');
- ``Inter``: a SINGLE dense2 (ae_ch -> inter_res**2 * inter_ch) +
  ``nn.reshape_4D`` -> NCHW (N, inter_ch, inter_res, inter_res) under
  the AMP hard-coded NCHW data format — the AMP inter has no dense1
  and no upscale (L193-201);
- ``Decoder``: x-branch upscale0..3/res0..3 (inter_ch -> d_ch*8,
  d_ch*8, d_ch*4, d_ch*2) + out_conv (1x1) / out_conv1..3 (3x3) ->
  concat on the channel axis -> depth_to_space(2) -> sigmoid image;
  m-branch upscalem0..4 (inter_ch -> d_mask_ch*8, *8, *4, *2, *1) +
  out_convm (1x1, 1 channel) -> sigmoid mask; the use_fp16 boundary
  casts (input cast at the top, output cast to float32 at the end)
  mirror official L229-230/L252-254 — the official use_fp16 is an
  EXPORT-only dtype knob (training is always fp32); this is plain
  dtype handling — NO mixed-precision (AMP) policy is introduced
  (later phase) (L204-255);
- checkpoint naming: the torch dotted tree names ARE the official
  sub-names verbatim (encoder: down1/conv1, res1/conv1+conv2,
  down2..down5/conv1, res5/conv1+conv2, dense1; inter: dense2;
  decoder: upscale0..3/conv1, res0..3/conv1+conv2, upscalem0..4/conv1,
  out_convm, out_conv..out_conv3 — NO x_/m_ prefixes, NO downs_<i>
  list scopes), so the Phase 4 converter round-trips the official
  encoder/inter/decoder .npy files with only the generic Conv2D
  HWIO->OIHW / Dense-identity layout hooks (Phase 7 audit 3);
- ``exact_k_morph_mask``: the official AMP training morph mask
  (L360-367): per-sample vector of exactly
  k = int(inter_dims * morph_factor) ones + (inter_dims - k) zeros,
  ``tf.random.shuffle`` (uniform k-subset over the channels),
  ``tf.stop_gradient``, stacked over the batch as
  (N, inter_dims, 1, 1). Reproduced with ``torch.randperm`` per
  sample (the SAME uniform fixed-count k-subset joint distribution —
  deliberately NOT the i.i.d. Bernoulli of nn.random_binomial; the
  fixed-count property is load-bearing for the morph blend); the mask
  is a constant tensor (no autograd inputs), the torch analogue of
  the official stop_gradient. The official draws the mask on
  /CPU:0 (a TF device-placement detail of the multi-GPU graph); the
  Phase 2 single-device torch runtime generates it on nn.device —
  device placement of a constant does not change its values.
"""

import torch
import torch.nn.functional as F

from core.leras import nn
from core.leras.layers.LayerBase import LayerBase, build_leaf_weights, cascade_init_weights


class AMPArchi(nn.ArchiBase):
    """
    resolution    (a multiple of 32, 64-640 per the official prompt)

    use_fp16      export-only dtype knob (training is always fp32)

    input_ch      (official constant 3)
    """
    def __init__(self, resolution, e_ch, ae_ch, inter_ch, inter_res, d_ch, d_mask_ch, use_fp16=False, input_ch=3):
        super().__init__()

        self.resolution = resolution
        self.e_ch = e_ch
        self.ae_ch = ae_ch
        self.inter_ch = inter_ch
        self.inter_res = inter_res
        self.d_ch = d_ch
        self.d_mask_ch = d_mask_ch
        self.input_ch = input_ch

        conv_dtype = torch.float16 if use_fp16 else torch.float32

        class Downscale(LayerBase):
            def __init__(self, in_ch, out_ch, kernel_size=5, name=None):
                super().__init__(name=name)
                self.conv1 = nn.Conv2D( in_ch, out_ch, kernel_size=kernel_size, strides=2, padding='SAME', dtype=conv_dtype)
                build_leaf_weights(self)

            def forward(self, x):
                x = self.conv1(x)
                x = F.leaky_relu(x, 0.1)
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
                x = F.leaky_relu(x, 0.1)
                x = nn.depth_to_space(x, 2)
                return x

            def init_weights(self):
                cascade_init_weights(self)

        class ResidualBlock(LayerBase):
            def __init__(self, ch, kernel_size=3, name=None):
                super().__init__(name=name)
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                build_leaf_weights(self)

            def forward(self, inp):
                x = self.conv1(inp)
                x = F.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = F.leaky_relu(inp + x, 0.2)
                return x

            def init_weights(self):
                cascade_init_weights(self)

        class Encoder(LayerBase):
            def __init__(self, name=None):
                super().__init__(name=name)
                self.down1 = Downscale(input_ch, e_ch, kernel_size=5)
                self.res1 = ResidualBlock(e_ch)
                self.down2 = Downscale(e_ch, e_ch*2, kernel_size=5)
                self.down3 = Downscale(e_ch*2, e_ch*4, kernel_size=5)
                self.down4 = Downscale(e_ch*4, e_ch*8, kernel_size=5)
                self.down5 = Downscale(e_ch*8, e_ch*8, kernel_size=5)
                self.res5 = ResidualBlock(e_ch*8)
                self.dense1 = nn.Dense( (( resolution//(2**5) )**2) * e_ch*8, ae_ch )
                build_leaf_weights(self)

            def forward(self, x):
                if use_fp16:
                    x = x.to(torch.float16)
                x = self.down1(x)
                x = self.res1(x)
                x = self.down2(x)
                x = self.down3(x)
                x = self.down4(x)
                x = self.down5(x)
                x = self.res5(x)
                if use_fp16:
                    x = x.to(torch.float32)
                x = nn.pixel_norm(nn.flatten(x), axes=-1)
                x = self.dense1(x)
                return x

            def init_weights(self):
                cascade_init_weights(self)

        class Inter(LayerBase):
            def __init__(self, name=None):
                super().__init__(name=name)
                self.dense2 = nn.Dense(ae_ch, inter_res * inter_res * inter_ch)
                build_leaf_weights(self)

            def forward(self, inp):
                x = inp
                x = self.dense2(x)
                x = nn.reshape_4D (x, inter_res, inter_res, inter_ch)
                return x

            def init_weights(self):
                cascade_init_weights(self)

        class Decoder(LayerBase):
            def __init__(self, name=None):
                super().__init__(name=name)

                self.upscale0 = Upscale(inter_ch, d_ch*8, kernel_size=3)
                self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
                self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)

                self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                self.res1 = ResidualBlock(d_ch*8, kernel_size=3)
                self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
                self.res3 = ResidualBlock(d_ch*2, kernel_size=3)

                self.upscalem0 = Upscale(inter_ch, d_mask_ch*8, kernel_size=3)
                self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
                self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

                self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
                self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)

                build_leaf_weights(self)

            def forward(self, z):
                if use_fp16:
                    z = z.to(torch.float16)

                x = self.upscale0(z)
                x = self.res0(x)
                x = self.upscale1(x)
                x = self.res1(x)
                x = self.upscale2(x)
                x = self.res2(x)
                x = self.upscale3(x)
                x = self.res3(x)

                x = torch.sigmoid( nn.depth_to_space(torch.cat( (self.out_conv(x),
                                                                 self.out_conv1(x),
                                                                 self.out_conv2(x),
                                                                 self.out_conv3(x)), nn.conv2d_ch_axis), 2) )
                m = self.upscalem0(z)
                m = self.upscalem1(m)
                m = self.upscalem2(m)
                m = self.upscalem3(m)
                m = self.upscalem4(m)
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


def exact_k_morph_mask(n_samples, inter_dims, morph_factor, device=None, dtype=None):
    """Official AMP L360-367 training morph mask (see the module
    docstring). Returns a constant (no autograd) tensor of shape
    (n_samples, inter_dims, 1, 1): each sample row holds exactly
    k = int(inter_dims * morph_factor) ones at uniformly random
    channel positions. k > inter_dims (unreachable through the
    official morph_factor option space, which clips to 0.1..0.5 on
    first run) is clamped to inter_dims; the official TF code would
    raise on the negative tile repeat instead — a robustness-only
    difference outside the official option space."""
    if device is None:
        device = nn.device
    if dtype is None:
        dtype = nn.floatx if nn.floatx is not None else torch.float32

    k = min(int(inter_dims * morph_factor), inter_dims)

    mask = torch.zeros(n_samples, inter_dims, 1, 1, device=device, dtype=dtype)
    if k > 0:
        for i in range(n_samples):
            perm = torch.randperm(inter_dims, device=device)
            mask[i, perm[:k], 0, 0] = 1.0
    return mask


nn.AMPArchi = AMPArchi
nn.exact_k_morph_mask = exact_k_morph_mask
