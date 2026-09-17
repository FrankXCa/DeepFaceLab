"""Phase 3F acceptance: official DFL archis and discriminators in torch.

Covers the Phase 3F migration of the reusable architecture building
blocks (official TF references preserved in
core/leras/archis/archis_tf.py and core/leras/models/discriminators_tf.py):

- DeepFakeArchi factory: construction for the official option space
  ('' / 't' / 'd' / 'td' / 'u' / 'c' combinations, use_fp16), the
  official get_out_ch/get_out_res contracts, and the official encoder/
  inter/decoder tensor layouts (channel-major 2-D encoder codes, the
  official flatten/pixel_norm/reshape_4D/depth_to_space op boundaries,
  the official depth_to_space 2x 'd' decoder head, the mask heads)
- official option semantics: leaky_relu 0.1/0.2 vs the 'c' CosRelu
  x*cos(x) branch (alpha ignored), the 'u' pixel_norm on the flattened
  encoder output (official 1e-6 epsilon via the Phase 3D op), use_fp16
  conv dtypes and boundary casts (construction + dtype checks; no fp16
  numerics, no AMP policy)
- official checkpoint naming: list scopes convs_<i> / downs_<i> /
  upconvs_<i> (the official ModelBase._build_sub naming, NOT torch
  ModuleList dotted indices), 'weight'/'bias' singular parameter names,
  exact expected key sets, official-format save/load round trip
  (pickled dict protocol 4, strict load)
- CodeDiscriminator (official n_downscales = 1 + code_res//8, kernel 4
  then 3), PatchDiscriminator (the official patch_discriminator_kernels
  table - verified byte-for-byte against the baseline),
  UNetPatchDiscriminator (official find_archi/calc_receptive_field_size
  layer search, level_chs progression, the official U-Net skip pairing
  (insert(0) on both the encs list and the upconvs list aligns each
  upconv with the same-level encoder feature), 1x1 VALID
  out_conv/center convs, (center_out, x) outputs)
- the official DFL SAME->int padding conversion is inherited from the
  Phase 3B Conv2D (explicit symmetric padding + VALID) - e.g. even-kernel
  stride-2 convs map x -> x/2+1, which is the official behavior, not TF
  true SAME; shapes are pinned accordingly
- CPU execution, RTX 4090 execution through the Phase 2 device
  abstraction with CPU/GPU parity (skip on CPU-only environments),
  backward reaching every conv/dense parameter, NCHW/NHWC data-format
  equivalence, no TensorFlow import and no direct torch.cuda.* in the
  migrated sources (AST checks)

Parity labels: EXACT for same-dtype torch-vs-torch comparisons
(activation branches, depth_to_space flow vs the official manual RRC
rearrangement oracle, checkpoint key sets, save/load round trip);
WITHIN_TOLERANCE (1e-5) for f32 flow vs f64 NumPy references and
(1e-3) for the CPU/GPU execution parity band (measured RTX 4090 worst
case ~2.1e-4 f32 device noise); NOT_VERIFIED for TF runtime parity
(no TF environment in the tested venvs - no TF runtime parity is
claimed). Note: the official depth_to_space op has two source branches
in the baseline (the manual reshape/transpose code for CPU/NHWC and
the tf.depth_to_space built-in for NCHW-GPU, which groups channels
differently); the Phase 3C op reproduces the manual branch
data-format-independently, which is what these tests pin.
"""

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras import checkpoint as dfl_ckpt  # noqa: E402
import core.leras.models.PatchDiscriminator  # noqa: F401
# the package __init__ star-imports the class onto the package attribute
# name, shadowing the submodule; sys.modules is authoritative
_PD_mod = sys.modules["core.leras.models.PatchDiscriminator"]

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)


def init_cpu(data_format="NCHW"):
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", data_format)


def build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="", seed=0, use_fp16=False):
    """Build the official encoder/inter/decoder stack (the exact call
    pattern of the official SAEHD/Quick96 models)."""
    torch.manual_seed(seed)
    archi = dfl_nn.DeepFakeArchi(res, use_fp16=use_fp16, opts=opts)
    encoder = archi.Encoder(in_ch=3, e_ch=e_ch, name="encoder")
    encoder.init_weights()
    inter = archi.Inter(
        in_ch=encoder.get_out_ch() * encoder.get_out_res(res) ** 2,
        ae_ch=ae_ch, ae_out_ch=ae_ch, name="inter",
    )
    inter.init_weights()
    decoder = archi.Decoder(
        in_ch=inter.get_out_ch(), d_ch=d_ch, d_mask_ch=m_ch, name="decoder_src",
    )
    decoder.init_weights()
    return archi, encoder, inter, decoder


OFFICIAL_COMBOS = [
    # (res, opts, e_ch) - the canonical configurations of the official
    # models: Quick96 'ud' at 96, SAEHD df ''/'d' and liae 't'/'td'
    (64, "", 4),
    (64, "d", 4),
    (64, "ud", 4),
    (64, "t", 4),
    (64, "td", 4),
    (96, "ud", 4),
    (128, "td", 8),
    (256, "d", 8),
]


def _ref_official_depth_to_space(x, size):
    """Independent oracle mirroring the official manual ops
    implementation (core/leras/ops/__init__.py, the CPU/NHWC branch the
    Phase 3C op reproduces): channel index c = i*size*c_out + j*c_out + g
    maps to (group g, spatial h*size+i, w*size+j). Written with only
    reshape/permute so it is independent of the op's internal
    pre-permutation + F.pixel_shuffle implementation. NCHW in/out."""
    n, c_in, h, w = x.shape
    c_out = c_in // (size * size)
    x = x.reshape(n, size, size, c_out, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    return x.reshape(n, c_out, h * size, w * size)


# ---------------------------------------------------------------------------
# Factory / API contract
# ---------------------------------------------------------------------------

def test_archi_factory_exposes_official_classes():
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, opts="")
    for attr in ("Encoder", "Inter", "Decoder"):
        assert hasattr(archi, attr), attr
    # the factory is the official ArchiBase contract (plain class)
    assert isinstance(archi, dfl_nn.ArchiBase)
    with pytest.raises(Exception):
        archi.flow()
    assert archi.get_weights() is None


def test_official_aliases_on_nn():
    init_cpu()
    assert dfl_nn.DeepFakeArchi is not None
    assert dfl_nn.ArchiBase is not None
    assert dfl_nn.CodeDiscriminator is not None
    assert dfl_nn.PatchDiscriminator is not None
    assert dfl_nn.UNetPatchDiscriminator is not None


def test_mod_quick_official_dead_end():
    # official: only 'mod is None' is implemented; the docstring 'quick'
    # never reaches a code branch and the factory tail fails with
    # NameError - the torch port mirrors this dead end exactly
    init_cpu()
    with pytest.raises(NameError):
        dfl_nn.DeepFakeArchi(64, mod="quick")


# ---------------------------------------------------------------------------
# Official shape / option contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("res,opts,e_ch", OFFICIAL_COMBOS)
def test_official_out_ch_out_res_contracts(res, opts, e_ch):
    init_cpu()
    torch.manual_seed(0)
    archi = dfl_nn.DeepFakeArchi(res, opts=opts)
    encoder = archi.Encoder(in_ch=3, e_ch=e_ch, name="encoder")
    assert encoder.get_out_ch() == e_ch * 8
    expected_enc_res = res // (16 if "t" not in opts else 32)
    assert encoder.get_out_res(res) == expected_enc_res

    inter = archi.Inter(
        in_ch=encoder.get_out_ch() * expected_enc_res ** 2,
        ae_ch=4, ae_out_ch=4, name="inter",
    )
    ldr = res // (32 if "d" in opts else 16)
    assert inter.get_out_res() == (ldr * 2 if "t" not in opts else ldr)
    assert inter.get_out_ch() == 4


@pytest.mark.parametrize("res,opts,e_ch", OFFICIAL_COMBOS)
def test_encoder_output_layout(res, opts, e_ch):
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=res, e_ch=e_ch, ae_ch=4, d_ch=4, m_ch=2, opts=opts)
    x = torch.randn(2, 3, res, res)
    code = encoder(x)
    out_res = res // (16 if "t" not in opts else 32)
    assert tuple(code.shape) == (2, e_ch * 8 * out_res ** 2), tuple(code.shape)


@pytest.mark.parametrize("res,opts,e_ch", OFFICIAL_COMBOS)
def test_inter_output_layout(res, opts, e_ch):
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=res, e_ch=e_ch, ae_ch=4, d_ch=4, m_ch=2, opts=opts)
    x = torch.randn(1, 3, res, res)
    z = inter(encoder(x))
    ldr = res // (32 if "d" in opts else 16)
    out_res = ldr * 2 if "t" not in opts else ldr
    assert tuple(z.shape) == (1, 4, out_res, out_res), tuple(z.shape)


@pytest.mark.parametrize("res,opts,e_ch", OFFICIAL_COMBOS)
def test_decoder_output_layout(res, opts, e_ch):
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=res, e_ch=e_ch, ae_ch=4, d_ch=4, m_ch=2, opts=opts)
    x = torch.randn(1, 3, res, res)
    out, m = decoder(inter(encoder(x)))
    # official canonical combos (df '', d, td, ud) all reach full
    # resolution on both heads (the 'd' head doubles the inter map via
    # depth_to_space; the non-'d' 1x1 head sits on the full-res map)
    assert tuple(out.shape) == (1, 3, res, res), tuple(out.shape)
    assert tuple(m.shape) == (1, 1, res, res), tuple(m.shape)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid head
    assert m.min() >= 0.0 and m.max() <= 1.0      # sigmoid mask head


def test_decoder_d_head_doubles_inter_map():
    # 'd' head: 4 convs on the half-res map (res//2) + depth_to_space(2)
    # restore full resolution
    init_cpu()
    res = 64
    archi, encoder, inter, decoder = build_stack(res=res, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d")
    ldr = res // 32
    inter_out_res = ldr * 2
    assert inter.get_out_res() == inter_out_res
    x = torch.randn(1, 3, res, res)
    out, m = decoder(inter(encoder(x)))
    # the 'd' head doubles the res//2 feature map produced by the three
    # upscaler stages -> full resolution
    assert out.shape[-2] == 2 * (res // 2) == res
    # the 'd' branch owns out_conv1/2/3 and upscalem3 (mask)
    assert hasattr(decoder, "out_conv1") and hasattr(decoder, "out_conv3")
    assert hasattr(decoder, "upscalem3") and not hasattr(decoder, "upscalem4")


def test_decoder_t_branch_layers():
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="t")
    assert hasattr(decoder, "upscale3") and hasattr(decoder, "res3")
    assert not hasattr(decoder, "out_conv1")
    assert hasattr(decoder, "upscalem3") and not hasattr(decoder, "upscalem4")
    archi_td, encoder_td, inter_td, decoder_td = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="td")
    assert hasattr(decoder_td, "upscalem4")


def test_downscale_block_channel_progression():
    # official: cur_ch = ch * min(2**i, 8); accessed through the encoder's
    # down1 DownscaleBlock (the block classes are factory closures, not
    # archi attributes)
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, opts="")
    encoder = archi.Encoder(in_ch=3, e_ch=8, name="enc")
    block = encoder.down1  # DownscaleBlock(3, 8, n=4, k=5)
    expected_in_ch = [3, 8, 16, 32]
    for i, down in enumerate(block.downs):
        assert down.conv1.in_ch == expected_in_ch[i]
        assert down.conv1.strides == 2
        assert down.conv1.kernel_size == 5
    x = torch.randn(1, 3, 64, 64)
    assert tuple(block(x).shape) == (1, 64, 4, 4)


def test_upscale_depth_to_space_flow_parity():
    # EXACT: the Upscale block (via inter.upscale1) output must equal
    # conv -> act -> depth_to_space, where the oracle mirrors the
    # official manual ops implementation (the official CPU/NHWC branch
    # that the Phase 3C op reproduces data-format-independently; note
    # torch's F.pixel_shuffle groups channels differently and is
    # deliberately NOT the official semantics)
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, opts="")
    inter = archi.Inter(in_ch=16, ae_ch=4, ae_out_ch=4, name="inter")
    up = inter.upscale1  # Upscale(4, 4, k=3)
    x = torch.randn(1, 4, 8, 8)
    got = up(x)
    w = up.conv1.weight.detach()
    b = up.conv1.bias.detach()
    ref_conv = F.conv2d(x, w, b, stride=1, padding=1)  # SAME k=3 -> p=1
    ref = _ref_official_depth_to_space(F.leaky_relu(ref_conv, 0.1), 2)
    assert torch.allclose(got, ref, atol=0.0, rtol=0.0), "depth_to_space flow mismatch"


def test_leaky_relu_default_branches():
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, opts="")
    encoder = archi.Encoder(in_ch=3, e_ch=4, name="enc")
    ds = encoder.down1.downs[0]  # Downscale(3, 4, k=5)
    x = torch.randn(1, 3, 16, 16)
    conv_out = ds.conv1(x)
    assert torch.allclose(ds(x), F.leaky_relu(conv_out, 0.1), rtol=1e-6)
    inter = archi.Inter(in_ch=16, ae_ch=4, ae_out_ch=4, name="inter")
    up = inter.upscale1  # Upscale(4, 4)
    x2 = torch.randn(1, 4, 8, 8)
    ref = _ref_official_depth_to_space(F.leaky_relu(up.conv1(x2), 0.1), 2)
    assert torch.allclose(up(x2), ref, rtol=0.0, atol=0.0)


def test_cosrelu_option_c_branch():
    # 'c': act(x, alpha) = x * cos(x) for any alpha (alpha ignored)
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, opts="c")
    encoder = archi.Encoder(in_ch=3, e_ch=4, name="enc")
    ds = encoder.down1.downs[0]  # Downscale(3, 4, k=5)
    x = torch.randn(1, 3, 16, 16)
    conv_out = ds.conv1(x)
    assert torch.allclose(ds(x), conv_out * torch.cos(conv_out), rtol=1e-6, atol=1e-6)


def test_pixel_norm_encoder_option_u():
    # 'u' official: pixel_norm(flatten(x), axes=-1) with the official
    # 1e-6 epsilon (Phase 3D op) - cross-checked in-flow against the
    # unnormalized twin with identical seeded weights
    init_cpu()
    archi_u, encoder_u, _, _ = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="u", seed=123)
    archi_n, encoder_n, _, _ = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="", seed=123)
    for a, b in zip(encoder_u.parameters(), encoder_n.parameters()):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    x = torch.randn(1, 3, 64, 64)
    code_u = encoder_u(x).detach().cpu()
    code_n = encoder_n(x).detach().cpu()
    ref = code_n.double() / torch.sqrt(torch.mean(code_n.double() ** 2, dim=-1, keepdim=True) + 1e-6)
    assert torch.allclose(code_u.double(), ref, rtol=1e-4, atol=1e-5)


def test_use_fp16_construction():
    init_cpu()
    archi = dfl_nn.DeepFakeArchi(64, use_fp16=True, opts="td")
    encoder = archi.Encoder(in_ch=3, e_ch=4, name="encoder")
    decoder = archi.Decoder(in_ch=8, d_ch=4, d_mask_ch=2, name="dec")
    conv_dtypes = {p.dtype for m in (encoder, decoder) for n, p in m.named_parameters()
                   if n.endswith("weight") and "dense" not in n}
    assert conv_dtypes == {torch.float16}, conv_dtypes
    dense_dtypes = {p.dtype for n, p in decoder.named_parameters() if "dense" in n}
    assert dense_dtypes <= {torch.float32}  # official Dense: no dtype kwarg -> nn.floatx


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------

def test_code_discriminator_official_contract():
    init_cpu()
    for code_res in (4, 8, 16, 32):
        n_downscales = 1 + code_res // 8
        disc = dfl_nn.CodeDiscriminator(8, code_res=code_res, name="dis")
        assert len(disc.convs) == n_downscales
        kernels = [c.kernel_size for c in disc.convs]
        assert kernels[0] == 4 and all(k == 3 for k in kernels[1:])
        strides = [c.strides for c in disc.convs]
        assert all(s == 2 for s in strides)
        x = torch.randn(1, 8, code_res // 2, code_res // 2)
        out = disc(x)
        assert tuple(out.shape) == (1, 1, code_res // (2 ** n_downscales), code_res // (2 ** n_downscales))


def test_patch_discriminator_official_table():
    # the table is verified byte-for-byte against the baseline by the
    # migration itself; here: conv/stride/channel wiring follows it
    init_cpu()
    table = _PD_mod.patch_discriminator_kernels
    assert len(table) == 46
    for ps in (1, 3, 4, 8, 20, 23, 34, 46):
        suggested, layers = table[ps]
        disc = dfl_nn.PatchDiscriminator(patch_size=ps, in_ch=3, name="pd")
        assert [(c.kernel_size, c.strides) for c in disc.convs] == [tuple(l) for l in layers]
        prev = 3
        for i, c in enumerate(disc.convs):
            assert c.in_ch == prev
            assert c.out_ch == suggested * min(2 ** i, 8)
            prev = c.out_ch
        in_res = 4 * ps
        out = disc(torch.randn(1, 3, in_res, in_res))
        assert out.shape[1] == 1 and out.shape[2] >= 1 and out.shape[2] == out.shape[3]


def test_unet_find_archi_official_search():
    init_cpu()
    d = dfl_nn.UNetPatchDiscriminator(patch_size=20, in_ch=3, base_ch=32, name="D")
    assert d.find_archi(20) == [[3, 2], [3, 1], [3, 2], [3, 2], [3, 2]]
    for target in (1, 5, 9, 20, 33, 46):
        layers = d.find_archi(target)
        assert layers[0] == [3, 2]
        assert all(k == 3 for k, s in layers)
        assert all(s in (1, 2) for k, s in layers)


def test_unet_shapes_and_official_skip_order():
    # the official UNet skip concat requires every intermediate map to be
    # even (DFL SAME = explicit int padding + VALID: a k=3/s=2 conv maps
    # an odd map x -> (x+1)/2, and the SAME deconv doubles back to x+1)
    # - the official models feed power-of-2 crops (128/256), so the test
    # uses a 128 crop for every patch size, exactly like the official
    # SAEHD/AMP GAN discriminators
    init_cpu()
    crop = 128
    for ps, base in ((8, 32), (20, 32), (34, 16)):
        d = dfl_nn.UNetPatchDiscriminator(patch_size=ps, in_ch=3, base_ch=base, name="D_src")
        d.init_weights()
        x = torch.randn(1, 3, crop, crop)
        center, out = d(x)
        layers = d.find_archi(ps)
        # deepest map = crop halved once per stride-2 layer; the U-Net
        # restores the input resolution on the x output (official)
        deep = crop
        for k, s in layers:
            if s == 2:
                deep //= 2
        assert tuple(center.shape) == (1, 1, deep, deep), (ps, tuple(center.shape))
        assert tuple(out.shape) == (1, 1, crop, crop), (ps, tuple(out.shape))
        # official channel wiring: out_conv in = level_chs[-1]*2 (the
        # first level, i.e. the base channel, doubled by the final skip)
        level0 = min(base, 512)
        assert d.out_conv.in_ch == level0 * 2
        # upconvs_0 is the DEEPEST layer's upconv (insert(0) order); its
        # input is the deepest feature map (factor 1 at the last layer)
        assert d.upconvs[0].in_ch == level_chs_last(base, layers)


def level_chs_last(base, layers):
    seq = [min(base * (2 ** i), 512) for i in range(len(layers) + 1)]
    return seq[-1]


def test_unet_fp16_construction():
    init_cpu()
    d = dfl_nn.UNetPatchDiscriminator(patch_size=8, in_ch=3, base_ch=16, use_fp16=True, name="GAN")
    dtypes = {p.dtype for p in d.parameters()}
    assert dtypes == {torch.float16}, dtypes


# ---------------------------------------------------------------------------
# Official checkpoint naming / save-load
# ---------------------------------------------------------------------------

def expected_encoder_keys(res=64, e_ch=4, opts=""):
    # the expected key order is the module creation order of
    # Encoder.__init__ (torch named_parameters registration order)
    keys = []
    if "t" in opts:
        # official creation order: down1, res1, down2..down5, res5
        order = [("down1", 1), ("res1", 2), ("down2", 1), ("down3", 1),
                 ("down4", 1), ("down5", 1), ("res5", 2)]
        for name, n_conv in order:
            for c in range(1, n_conv + 1):
                keys += [f"{name}/conv{c}/weight:0", f"{name}/conv{c}/bias:0"]
    else:
        # down1 is a DownscaleBlock owning downs_0..downs_3
        for i in range(4):
            keys += [f"down1/downs_{i}/conv1/weight:0", f"down1/downs_{i}/conv1/bias:0"]
    return keys


def expected_inter_keys(opts="", ldr=4):
    keys = ["dense1/weight:0", "dense1/bias:0", "dense2/weight:0", "dense2/bias:0"]
    if "t" not in opts:
        keys += ["upscale1/conv1/weight:0", "upscale1/conv1/bias:0"]
    return keys


def test_official_checkpoint_keys_encoder():
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="")
    got = [dfl_ckpt.official_name(n) for n, p in encoder.named_parameters()]
    assert got == expected_encoder_keys(opts=""), got
    got_i = [dfl_ckpt.official_name(n) for n, p in inter.named_parameters()]
    assert got_i == expected_inter_keys(opts="", ldr=64 // 16), got_i

    archi_t, encoder_t, inter_t, _ = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="t")
    got_t = [dfl_ckpt.official_name(n) for n, p in encoder_t.named_parameters()]
    assert got_t == expected_encoder_keys(opts="t"), got_t
    assert [dfl_ckpt.official_name(n) for n, p in inter_t.named_parameters()] == expected_inter_keys(opts="t")


def test_official_checkpoint_keys_decoder_d():
    init_cpu()
    _, _, _, decoder = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d")
    got = [dfl_ckpt.official_name(n) for n, p in decoder.named_parameters()]
    expected = (
        ["upscale0/conv1/weight:0", "upscale0/conv1/bias:0",
         "upscale1/conv1/weight:0", "upscale1/conv1/bias:0",
         "upscale2/conv1/weight:0", "upscale2/conv1/bias:0",
         "res0/conv1/weight:0", "res0/conv1/bias:0", "res0/conv2/weight:0", "res0/conv2/bias:0",
         "res1/conv1/weight:0", "res1/conv1/bias:0", "res1/conv2/weight:0", "res1/conv2/bias:0",
         "res2/conv1/weight:0", "res2/conv1/bias:0", "res2/conv2/weight:0", "res2/conv2/bias:0",
         "upscalem0/conv1/weight:0", "upscalem0/conv1/bias:0",
         "upscalem1/conv1/weight:0", "upscalem1/conv1/bias:0",
         "upscalem2/conv1/weight:0", "upscalem2/conv1/bias:0",
         "out_conv/weight:0", "out_conv/bias:0",
         "out_conv1/weight:0", "out_conv1/bias:0",
         "out_conv2/weight:0", "out_conv2/bias:0",
         "out_conv3/weight:0", "out_conv3/bias:0",
         "upscalem3/conv1/weight:0", "upscalem3/conv1/bias:0",
         "out_convm/weight:0", "out_convm/bias:0"]
    )
    assert got == expected, got


def test_official_checkpoint_keys_discriminators():
    init_cpu()
    cd = dfl_nn.CodeDiscriminator(8, code_res=16, name="dis")
    got = [dfl_ckpt.official_name(n) for n, p in cd.named_parameters()]
    expected = []
    for i in range(1 + 16 // 8):
        expected += [f"convs_{i}/weight:0", f"convs_{i}/bias:0"]
    expected += ["out_conv/weight:0", "out_conv/bias:0"]
    assert got == expected, got

    pd = dfl_nn.PatchDiscriminator(patch_size=20, in_ch=3, name="pd")
    got_p = [dfl_ckpt.official_name(n) for n, p in pd.named_parameters()]
    n_layers = len(_PD_mod.patch_discriminator_kernels[20][1])
    # registration order is conv_i weight,bias interleaved (creation order)
    expected_p = []
    for i in range(n_layers):
        expected_p += [f"convs_{i}/weight:0", f"convs_{i}/bias:0"]
    expected_p += ["out_conv/weight:0", "out_conv/bias:0"]
    assert got_p == expected_p, got_p

    ud = dfl_nn.UNetPatchDiscriminator(patch_size=20, in_ch=3, base_ch=32, name="D_src")
    got_u = [dfl_ckpt.official_name(n) for n, p in ud.named_parameters()]
    assert got_u[0] == "in_conv/weight:0" and "in_conv/bias:0" in got_u
    n_layers = len(ud.find_archi(20))
    assert f"convs_{n_layers-1}/weight:0" in got_u
    assert f"upconvs_{n_layers-1}/weight:0" in got_u
    for key in ("out_conv/weight:0", "center_out/weight:0", "center_conv/weight:0"):
        assert key in got_u, key


def test_save_load_roundtrip_official_format():
    # NOTE: the sandbox tracks existing workspace paths; write into the
    # (existing) repo root and clean up afterwards instead of creating
    # a fresh temp directory
    init_cpu()
    _, encoder, _, _ = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d", seed=7)
    filename = str(REPO_ROOT / "_tmp_3f_save_load.npy")
    try:
        encoder.save_weights(filename)

        # official format: pickled dict (protocol 4) with official ':0' keys
        blob = Path(filename).read_bytes()
        d = pickle.loads(blob)
        assert isinstance(d, dict) and len(d) > 0
        assert all(k.endswith(":0") for k in d)
        assert "down1/downs_0/conv1/weight:0" in d
        assert d["down1/downs_0/conv1/weight:0"].ndim == 4  # official HWIO layout

        snapshot = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters()}

        archi2 = dfl_nn.DeepFakeArchi(64, opts="d")
        enc2 = archi2.Encoder(in_ch=3, e_ch=4, name="encoder")
        assert enc2.load_weights(filename) is True
        for n, p in enc2.named_parameters():
            assert torch.equal(p.detach().cpu(), snapshot[n]), n

        # strict load: an unexpected extra key must be rejected
        d2 = dict(d)
        d2["rogue/weight:0"] = np.zeros((1, 1), dtype=np.float32)
        bad = str(REPO_ROOT / "_tmp_3f_save_load_bad.npy")
        Path(bad).write_bytes(pickle.dumps(d2, 4))
        from core.leras.checkpoint import CheckpointLoadError
        enc3 = dfl_nn.DeepFakeArchi(64, opts="d").Encoder(in_ch=3, e_ch=4, name="encoder")
        with pytest.raises(CheckpointLoadError):
            enc3.load_weights(bad)
    finally:
        for p in (filename, str(REPO_ROOT / "_tmp_3f_save_load_bad.npy")):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def test_get_weights_flat_deterministic():
    init_cpu()
    _, encoder, inter, decoder = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="td", seed=3)
    w = encoder.get_weights()
    assert len(w) == sum(1 for _ in encoder.parameters())
    assert all(isinstance(p, torch.nn.Parameter) for p in w)
    # deterministic across repeated constructions (same architecture)
    archi2 = dfl_nn.DeepFakeArchi(64, opts="td")
    enc2 = archi2.Encoder(in_ch=3, e_ch=4, name="encoder")
    names1 = [dfl_ckpt.official_name(n) for n, p in encoder.named_parameters()]
    names2 = [dfl_ckpt.official_name(n) for n, p in enc2.named_parameters()]
    assert names1 == names2
    # the official model contract: flat per-submodel lists concatenate
    flat = encoder.get_weights() + inter.get_weights() + decoder.get_weights()
    assert len(flat) == sum(len(m.get_weights()) for m in (encoder, inter, decoder))


# ---------------------------------------------------------------------------
# Execution: backward, data formats, GPU parity
# ---------------------------------------------------------------------------

def test_backward_reaches_all_parameters():
    init_cpu()
    archi, encoder, inter, decoder = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d")
    x = torch.randn(1, 3, 64, 64)
    out, m = decoder(inter(encoder(x)))
    loss = out.sum() + m.sum()
    loss.backward()
    for m_ in (encoder, inter, decoder):
        for p in m_.parameters():
            assert p.grad is not None, p
    assert any(g is not None and torch.is_tensor(g) for g in [p.grad for p in decoder.parameters()])


def test_data_format_equivalence():
    # NCHW vs NHWC must produce the same tensors (boundary permutations
    # inside the layers) - no hidden NHWC assumptions in the archis;
    # inputs are fed in the active data_format layout, exactly like the
    # official placeholders (nn.get4Dshape)
    base = torch.randn(1, 3, 64, 64)  # canonical NCHW reference

    init_cpu("NCHW")
    archi1, enc1, inter1, dec1 = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d", seed=11)
    code1 = enc1(base)
    out1, m1 = dec1(inter1(code1))

    init_cpu("NHWC")
    archi2, enc2, inter2, dec2 = build_stack(res=64, e_ch=4, ae_ch=4, d_ch=4, m_ch=2, opts="d", seed=11)
    x_nhwc = dfl_nn.to_data_format(base, "NHWC", "NCHW").contiguous()
    code2 = enc2(x_nhwc)
    out2, m2 = dec2(inter2(code2))

    # the 2-D encoder code is layout-independent (channel-major flatten
    # under both formats)
    assert torch.allclose(code1, code2, atol=1e-5)
    assert dfl_nn.data_format == "NHWC"
    out2_nchw = dfl_nn.to_data_format(out2, "NCHW", "NHWC")
    m2_nchw = dfl_nn.to_data_format(m2, "NCHW", "NHWC")
    assert torch.allclose(out1, out2_nchw, atol=1e-4)
    assert torch.allclose(m1, m2_nchw, atol=1e-4)


def _build_gpu_twin(res=128, e_ch=8, ae_ch=8, d_ch=8, m_ch=4, ps=20, base_ch=32):
    archi = dfl_nn.DeepFakeArchi(res, opts="td")
    encoder = archi.Encoder(in_ch=3, e_ch=e_ch, name="encoder")
    encoder.init_weights()
    inter = archi.Inter(in_ch=encoder.get_out_ch() * encoder.get_out_res(res) ** 2,
                        ae_ch=ae_ch, ae_out_ch=ae_ch, name="inter")
    inter.init_weights()
    decoder = archi.Decoder(in_ch=inter.get_out_ch(), d_ch=d_ch, d_mask_ch=m_ch, name="decoder_src")
    decoder.init_weights()
    disc = dfl_nn.UNetPatchDiscriminator(patch_size=ps, in_ch=3, base_ch=base_ch, name="D_src")
    disc.init_weights()
    return (encoder, inter, decoder), disc


@requires_gpu
def test_gpu_execution_and_cpu_parity():
    # weights are generated on the CPU twin (seeded CPU RNG) and COPIED
    # to the GPU twin: torch's CUDA and CPU RNG streams produce different
    # sequences for the same seed (the official TF behavior), so seeding
    # both sides independently would compare different weights. This
    # isolates the device execution parity (the actual test target).
    init_cpu("NCHW")
    torch.manual_seed(42)
    mods_c, disc_c = _build_gpu_twin()
    x_c = torch.randn(1, 3, 128, 128)
    code_c = mods_c[0](x_c)
    z_c = mods_c[1](code_c)
    out_c, m_c = mods_c[2](z_c)
    c_c, o_c = disc_c(x_c)
    weights_c = [p.detach().clone() for p in
                 (list(mods_c[0].parameters()) + list(mods_c[1].parameters())
                  + list(mods_c[2].parameters()) + list(disc_c.parameters()))]

    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")
    dev = dfl_nn.device
    assert dev.type == "cuda"

    mods_g, disc_g = _build_gpu_twin()
    params_g = list(mods_g[0].parameters()) + list(mods_g[1].parameters()) + \
               list(mods_g[2].parameters()) + list(disc_g.parameters())
    with torch.no_grad():
        for pg, pc in zip(params_g, weights_c):
            assert tuple(pg.shape) == tuple(pc.shape)
            pg.data.copy_(pc.to(dev))

    x_g = x_c.to(dev)
    code_g = mods_g[0](x_g)
    z_g = mods_g[1](code_g)
    out_g, m_g = mods_g[2](z_g)
    c_g, o_g = disc_g(x_g)

    for name, a, b in (("encoder", code_g, code_c), ("inter", z_g, z_c),
                       ("decoder_x", out_g, out_c), ("decoder_m", m_g, m_c),
                       ("unet_center", c_g, c_c), ("unet_out", o_g, o_c)):
        # f32 GPU-vs-CPU accumulation noise across the ~25-layer stack
        # (different reduction orders; measured worst case on the RTX
        # 4090 is ~2.1e-4, dominated by the pixel_norm tail sensitivity
        # in the encoder); the band is still orders of magnitude tighter
        # than any semantic (layout/ordering) error
        assert torch.allclose(a.detach().cpu(), b, rtol=1e-3, atol=1e-3), name

    # and back to the GPU foundation for any later tests
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")


# ---------------------------------------------------------------------------
# Source hygiene (AST)
# ---------------------------------------------------------------------------

MIGRATED_SOURCES = [
    REPO_ROOT / "core" / "leras" / "archis" / "ArchiBase.py",
    REPO_ROOT / "core" / "leras" / "archis" / "DeepFakeArchi.py",
    REPO_ROOT / "core" / "leras" / "models" / "CodeDiscriminator.py",
    REPO_ROOT / "core" / "leras" / "models" / "PatchDiscriminator.py",
    REPO_ROOT / "core" / "leras" / "models" / "__init__.py",
    REPO_ROOT / "core" / "leras" / "nn.py",
]


def _walk_sources():
    for path in MIGRATED_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        yield path, tree


def test_no_tensorflow_import_in_migrated_sources():
    for path, tree in _walk_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("tensorflow"), (path, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tensorflow"), (path, node.module)


def test_no_direct_torch_cuda_in_migrated_sources():
    for path, tree in _walk_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "torch" and chain and chain[-1] == "cuda":
                    pytest.fail(f"direct torch.cuda.* in {path}: torch.{'.'.join(reversed(chain))}")


def test_dead_reference_files_are_not_imported_by_live_code():
    # the dead official TF references must stay out of the live import
    # graph (they would fail under the torch foundation)
    for name in ("archis/archis_tf.py", "models/discriminators_tf.py"):
        path = REPO_ROOT / "core" / "leras" / name
        assert path.exists(), name
    init_cpu()
    import core.leras.archis  # noqa: F401
    import core.leras.models  # noqa: F401
    assert "core.leras.archis.archis_tf" not in sys.modules
    assert "core.leras.models.discriminators_tf" not in sys.modules
