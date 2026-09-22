"""XSeg — torch implementation of the official leras XSeg model (Phase 10B).

Replaces the official TensorFlow source in place (the official code is
preserved verbatim in ``XSeg_tf.py`` — same pattern as
``ModelBase_tf.py`` / ``discriminators_tf.py``). ``nn.XSeg`` is now
importable under the torch foundation (previously the file executed
``tf = nn.tf`` at import time and was gated behind
``hasattr(nn, 'tf')`` in ``core/leras/models/__init__.py``).

Official behavior preserved (identical architecture and semantics):

- ``on_build(in_ch, base_ch, out_ch)``: the exact official block tree —
  encoder ConvBlock (Conv2D 3x3 SAME -> FRNorm2D -> TLU) levels
  conv01/conv02 (base_ch) -> BlurPool(4), conv11/conv12 (2x) ->
  BlurPool(3), conv21/conv22 (4x) -> BlurPool(2), conv31..conv33 (8x)
  -> BlurPool(2), conv41..conv43 (8x) -> BlurPool(2), conv51..conv53
  (8x) -> BlurPool(2); bottleneck flatten -> Dense(4*4*base_ch*8 ->
  512) -> Dense(512 -> 4*4*base_ch*8) -> reshape_4D(4, 4, base_ch*8);
  decoder up5..up0 (Conv2DTranspose 3x3 SAME + FRNorm2D + TLU) with
  the official skip-concat blocks (uconv53/uconv43/uconv33 take
  base_ch*12, uconv22 takes base_ch*8 -> base_ch*4, uconv12
  base_ch*4 -> base_ch*2, uconv02 base_ch*2 -> base_ch) and
  ``out_conv`` (Conv2D base_ch -> out_ch 3x3 SAME).
- ``forward(inp, pretrain=False)``: identical control flow — the
  decoder concats each skip activation; with ``pretrain=True`` the
  skip is zeroed before its concat (the official ``tf.zeros_like``
  semantics, torch ``torch.zeros_like``); returns ``(logits,
  sigmoid(logits))`` exactly like the official model (callers — the
  inference wrapper and the pretraining path — select the tensor they
  need).
- weight inventory and order: the official 222-variable layout
  (per block ``conv/weight:0``, ``conv/bias:0``, ``frn/weight:0``,
  ``frn/bias:0``, ``frn/eps:0``, ``tlu/tau:0``; ``dense1`` /
  ``dense2`` weights+bias; BlurPool has no variables) in the official
  construction order, verified against the Phase 10A official
  TensorFlow reference for strict checkpoint loading.

Torch adaptations (documented, minimal):

- the three TF-only tensor ops in the decoder/skip path are the native
  torch equivalents with no semantic change: ``tf.concat`` ->
  ``torch.cat`` (same ``nn.conv2d_ch_axis`` as the official call —
  the torch foundation sets it per ``nn.data_format`` exactly like the
  TF runtime), ``tf.zeros_like`` -> ``torch.zeros_like``,
  ``tf.nn.sigmoid`` -> ``torch.sigmoid``. ``nn.flatten`` and
  ``nn.reshape_4D`` are the already-migrated torch ops (Phase 3C/3D,
  format-aware like their TF twins).
- the container lifecycle is the torch ``nn.ModelBase`` (Phase 5): the
  official attribute-discovery build, ``get_weights`` order,
  ``save_weights`` / ``load_weights`` and the Phase 4 checkpoint
  engine all work unchanged because the layer classes
  (Conv2D / Conv2DTranspose / Dense / FRNorm2D / TLU / BlurPool) are
  the same torch layers the SAEHD/AMP archis use since Phase 3.
- the official optimizer-state naming (Phase 10B): the official DFL
  optimizer names its per-parameter state after the trained
  variable's FULL name (``acc_<full_varname>`` with ``:`` -> ``_``,
  e.g. ``acc_XSeg/conv01/conv/weight_0:0`` in the official
  ``XSeg_256_opt.npy``). The torch optimizer emits that naming from a
  per-parameter binding (``param._dfl_name``), so the model binds
  every weight to its full official variable name (``self.name`` —
  the official model scope — plus the sub-name from its weight
  enumeration) and to its owning leaf layer (state LAYOUT delegation,
  the Phase 3E2 ``_dfl_owner_layer`` contract) at the end of
  ``build()``. (SAEHD/AMP perform the same binding in their
  ``Model.py`` because their components are separate named modules;
  here the full variable name is determined by the net's own scope
  name, so the binding lives in the model class and every consumer
  that optimizes the net — ``facelib/XSegNet`` in training mode,
  ``models/Model_XSeg/Model.py`` — gets the official state naming
  without duplicating logic.)

Official provenance: ``core/leras/models/XSeg.py`` at upstream
baseline ``e4b7543ffa1d73b26fce1e31852727f658ba490c`` (see
``XSeg_tf.py``).
"""

import torch

from core.leras import nn


class XSeg(nn.ModelBase):

    def build(self):
        # the base build discovers/registers the official block tree
        # (and, recursively, builds the nested ConvBlock / UpConvBlock
        # containers); the guard keeps an explicit second build() call
        # a no-op (the lazy __call__/get_weights builds are gated on
        # self.built already)
        if not self.built:
            super().build()
            self._bind_official_state_names()

    def _bind_official_state_names(self):
        # Phase 10B (official optimizer-state naming, module docstring):
        # bind every weight to its FULL official variable name (the
        # model scope — self.name, the official TF variable-scope name
        # the official TF runtime assigned, e.g.
        # 'XSeg/conv01/conv/weight:0') so the Phase 3E2 optimizer emits
        # the official state keys (e.g. 'acc_XSeg/conv01/conv/
        # weight_0:0'), and to its owning leaf layer (state LAYOUT
        # delegation for the optimizer's load/save/convert hooks)
        owners = {}
        for module in self.modules():
            for p in module.parameters(recurse=False):
                owners[id(p)] = module
        for sub_name, param in self._iter_official_weights():
            param._dfl_name = f"{self.name}/{sub_name}"
            owner = owners.get(id(param))
            if owner is not None:
                param._dfl_owner_layer = owner

    def on_build(self, in_ch, base_ch, out_ch):

        class ConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        class UpConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):
                self.conv = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        self.base_ch = base_ch

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = nn.BlurPool(filt_size=4)

        self.conv11 = ConvBlock(base_ch, base_ch * 2)
        self.conv12 = ConvBlock(base_ch * 2, base_ch * 2)
        self.bp1 = nn.BlurPool(filt_size=3)

        self.conv21 = ConvBlock(base_ch * 2, base_ch * 4)
        self.conv22 = ConvBlock(base_ch * 4, base_ch * 4)
        self.bp2 = nn.BlurPool(filt_size=2)

        self.conv31 = ConvBlock(base_ch * 4, base_ch * 8)
        self.conv32 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv33 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp3 = nn.BlurPool(filt_size=2)

        self.conv41 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv42 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv43 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp4 = nn.BlurPool(filt_size=2)

        self.conv51 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv52 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv53 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp5 = nn.BlurPool(filt_size=2)

        self.dense1 = nn.Dense(4 * 4 * base_ch * 8, 512)
        self.dense2 = nn.Dense(512, 4 * 4 * base_ch * 8)

        self.up5 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv53 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv52 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv51 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up4 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv43 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv42 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv41 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv33 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv32 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv31 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up2 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv22 = ConvBlock(base_ch * 8, base_ch * 4)
        self.uconv21 = ConvBlock(base_ch * 4, base_ch * 4)

        self.up1 = UpConvBlock(base_ch * 4, base_ch * 2)
        self.uconv12 = ConvBlock(base_ch * 4, base_ch * 2)
        self.uconv11 = ConvBlock(base_ch * 2, base_ch * 2)

        self.up0 = UpConvBlock(base_ch * 2, base_ch)
        self.uconv02 = ConvBlock(base_ch * 2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2D(base_ch, out_ch, kernel_size=3, padding='SAME')

    def forward(self, inp, pretrain=False):
        x = inp

        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)

        x = nn.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.reshape_4D(x, 4, 4, self.base_ch * 8)

        x = self.up5(x)
        if pretrain:
            x5 = torch.zeros_like(x5)
        x = self.uconv53(torch.cat((x, x5), nn.conv2d_ch_axis))
        x = self.uconv52(x)
        x = self.uconv51(x)

        x = self.up4(x)
        if pretrain:
            x4 = torch.zeros_like(x4)
        x = self.uconv43(torch.cat((x, x4), nn.conv2d_ch_axis))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        if pretrain:
            x3 = torch.zeros_like(x3)
        x = self.uconv33(torch.cat((x, x3), nn.conv2d_ch_axis))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        if pretrain:
            x2 = torch.zeros_like(x2)
        x = self.uconv22(torch.cat((x, x2), nn.conv2d_ch_axis))
        x = self.uconv21(x)

        x = self.up1(x)
        if pretrain:
            x1 = torch.zeros_like(x1)
        x = self.uconv12(torch.cat((x, x1), nn.conv2d_ch_axis))
        x = self.uconv11(x)

        x = self.up0(x)
        if pretrain:
            x0 = torch.zeros_like(x0)
        x = self.uconv02(torch.cat((x, x0), nn.conv2d_ch_axis))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)


nn.XSeg = XSeg
