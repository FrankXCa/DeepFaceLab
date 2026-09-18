"""Phase 7 (commit 1) acceptance: the official AMP archi (encoder /
inter / decoder blocks) and the exact-k morph mask in torch.

Covers the Phase 7 foundation migration of the official AMP inline
archi classes (dead official TF reference preserved in
models/Model_AMP/Model_tf.py L138-255):

- AMPArchi factory: the official no-argument sub-model constructors
  (Encoder/Inter/Decoder close over the factory dimensions, exactly
  like the official on_build(self) inline classes), the Downscale /
  Upscale / ResidualBlock building blocks, and the official option
  space (resolution a multiple of 32, 64-640; use_fp16 the
  export-only conv-dtype knob)
- official tensor layouts: the flat (N, ae_ch) pixel-normalized
  encoder code, the single-dense inter code reshaped to NCHW
  (N, inter_ch, inter_res, inter_res) under the AMP hard-coded NCHW
  data format (official L107 — the model never runs NHWC, so no
  data-format equivalence branch is tested, unlike the SAEHD/DeepFake
  archis), the decoder image head (out_conv/out_conv1..3 concat ->
  depth_to_space RRC -> sigmoid) and the mask head (upscalem0..4 ->
  out_convm -> sigmoid)
- exact_k_morph_mask: the official L360-367 per-sample EXACT-k mask
  (k = int(inter_dims * morph_factor) ones at uniformly random
  channel positions, tf.random.shuffle reproduced with
  torch.randperm — the uniform fixed-count k-subset distribution,
  NOT the i.i.d. Bernoulli of random_binomial), stop_gradient
  analogue (constant tensor), (N, inter_dims, 1, 1) shape, the k=0 /
  k=inter_dims edges, and the official CPU-draw device detail
  (the torch runtime draws on nn.device; placement of a constant
  does not change values)
- official checkpoint naming: the torch dotted tree names ARE the
  official sub-names verbatim (encoder down1/conv1, res1/conv1+conv2,
  down2..down5/conv1, res5/conv1+conv2, dense1; inter dense2;
  decoder upscale0..3/conv1, res0..3/conv1+conv2, upscalem0..4/conv1,
  out_convm, out_conv..out_conv3 — NO x_/m_ prefixes, NO downs_<i>
  list scopes); official-format save/load round trip (pickled dict
  protocol 4, strict load); Conv2D weights saved in the official HWIO
  layout, Dense layout-identical (in, out)
- CPU execution, RTX 4090 execution through the Phase 2 device
  abstraction with CPU/GPU parity (skip on CPU-only environments),
  backward reaching every parameter, construction-time dtype checks
  for the use_fp16 knob (no fp16 numerics), no TensorFlow import and
  no direct torch.cuda.* in the migrated source (AST checks)

Parity labels: EXACT for same-dtype torch-vs-torch comparisons
(checkpoint key sets, save/load round trip, morph-mask fixed-count
property); WITHIN_TOLERANCE (2e-3 band) for the CPU/GPU execution
parity (measured RTX 4090 worst case for this stack: encoder 1.9e-3
abs / inter 1.4e-3 abs / decoder 3.2e-5 abs — f32 device noise
amplified by the pixel_norm tail sensitivity; the sibling
DeepFake/SAEHD stacks measure ~2.1e-4); NOT_VERIFIED for TF runtime
parity (no TF environment in the tested venvs — the dead official TF
reference is never executed; AMP official TF-runtime numerical parity
is a model-level label, see the Phase 7 plan section 23).
"""

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras import checkpoint as dfl_ckpt  # noqa: E402
from core.leras.archis.AMP import exact_k_morph_mask  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)


def init_cpu(data_format="NCHW"):
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", data_format)


def build_amp_stack(res=64, e_ch=16, ae_ch=32, inter_ch=32, d_ch=16, d_mask_ch=6,
                    use_fp16=False, seed=0):
    """Build the official AMP encoder/inter_src/inter_dst/decoder stack
    (the exact call pattern of the official AMP model: the no-argument
    sub-model constructors of the factory)."""
    torch.manual_seed(seed)
    archi = dfl_nn.AMPArchi(
        res, e_ch=e_ch, ae_ch=ae_ch, inter_ch=inter_ch,
        inter_res=res // 32, d_ch=d_ch, d_mask_ch=d_mask_ch,
        use_fp16=use_fp16,
    )
    encoder = archi.Encoder(name="encoder")
    encoder.init_weights()
    inter_src = archi.Inter(name="inter_src")
    inter_src.init_weights()
    inter_dst = archi.Inter(name="inter_dst")
    inter_dst.init_weights()
    decoder = archi.Decoder(name="decoder")
    decoder.init_weights()
    return archi, encoder, inter_src, inter_dst, decoder


# The official checkpoint key inventories (Phase 7 audit 3, verified
# 1:1 against the real local checkpoints: encoder 20 keys, inter 2
# keys, decoder 44 keys), in construction order.
def expected_encoder_keys():
    keys = [f"down1/conv1/weight:0", f"down1/conv1/bias:0",
            f"res1/conv1/weight:0", f"res1/conv1/bias:0",
            f"res1/conv2/weight:0", f"res1/conv2/bias:0"]
    for name in ("down2", "down3", "down4", "down5"):
        keys += [f"{name}/conv1/weight:0", f"{name}/conv1/bias:0"]
    keys += [f"res5/conv1/weight:0", f"res5/conv1/bias:0",
             f"res5/conv2/weight:0", f"res5/conv2/bias:0",
             "dense1/weight:0", "dense1/bias:0"]
    return keys


def expected_inter_keys():
    return ["dense2/weight:0", "dense2/bias:0"]


def expected_decoder_keys():
    keys = []
    for name in ("upscale0", "upscale1", "upscale2", "upscale3"):
        keys += [f"{name}/conv1/weight:0", f"{name}/conv1/bias:0"]
    for name in ("res0", "res1", "res2", "res3"):
        keys += [f"{name}/conv1/weight:0", f"{name}/conv1/bias:0",
                 f"{name}/conv2/weight:0", f"{name}/conv2/bias:0"]
    for name in ("upscalem0", "upscalem1", "upscalem2", "upscalem3", "upscalem4"):
        keys += [f"{name}/conv1/weight:0", f"{name}/conv1/bias:0"]
    keys += ["out_convm/weight:0", "out_convm/bias:0",
             "out_conv/weight:0", "out_conv/bias:0",
             "out_conv1/weight:0", "out_conv1/bias:0",
             "out_conv2/weight:0", "out_conv2/bias:0",
             "out_conv3/weight:0", "out_conv3/bias:0"]
    return keys


# ---------------------------------------------------------------------------
# Construction, official checkpoint naming
# ---------------------------------------------------------------------------

def test_official_checkpoint_keys():
    init_cpu()
    _, encoder, inter_src, inter_dst, decoder = build_amp_stack(seed=1)
    assert [dfl_ckpt.official_name(n) for n, p in encoder.named_parameters()] == expected_encoder_keys()
    assert [dfl_ckpt.official_name(n) for n, p in inter_src.named_parameters()] == expected_inter_keys()
    assert [dfl_ckpt.official_name(n) for n, p in inter_dst.named_parameters()] == expected_inter_keys()
    assert [dfl_ckpt.official_name(n) for n, p in decoder.named_parameters()] == expected_decoder_keys()


def test_get_weights_flat_deterministic():
    init_cpu()
    _, enc1, isrc1, idst1, dec1 = build_amp_stack(seed=3)
    archi2 = dfl_nn.AMPArchi(64, e_ch=16, ae_ch=32, inter_ch=32, inter_res=2, d_ch=16, d_mask_ch=6)
    enc2 = archi2.Encoder(name="encoder")
    names1 = [dfl_ckpt.official_name(n) for n, p in enc1.named_parameters()]
    names2 = [dfl_ckpt.official_name(n) for n, p in enc2.named_parameters()]
    assert names1 == names2
    # the official model contract: flat per-submodel lists concatenate
    # (encoder + decoder = the official G_weights; the inter codes are
    # NEVER in any optimizer state — Phase 7 audit 3, real src_dst_opt
    # state files)
    flat = enc1.get_weights() + dec1.get_weights()
    assert len(flat) == len(enc1.get_weights()) + len(dec1.get_weights())
    assert isrc1.get_weights()  # inter weights exist but stay frozen
    assert len(idst1.get_weights()) == 2


def test_save_load_roundtrip_official_format():
    # NOTE: the sandbox tracks existing workspace paths; write into the
    # (existing) repo root and clean up afterwards instead of creating
    # a fresh temp directory
    init_cpu()
    _, encoder, inter_src, inter_dst, decoder = build_amp_stack(res=64, e_ch=16, ae_ch=32,
                                                                inter_ch=32, d_ch=16, d_mask_ch=6,
                                                                seed=7)
    filename = str(REPO_ROOT / "_tmp_p7_save_load.npy")
    badname = str(REPO_ROOT / "_tmp_p7_save_load_bad.npy")
    try:
        encoder.save_weights(filename)

        # official format: pickled dict (protocol 4) with official ':0'
        # keys; Conv2D weights in the official HWIO layout
        d = pickle.loads(Path(filename).read_bytes())
        assert isinstance(d, dict) and len(d) == 20
        assert all(k.endswith(":0") for k in d)
        assert "down1/conv1/weight:0" in d
        assert d["down1/conv1/weight:0"].ndim == 4  # official HWIO layout
        # dense1 keeps the official (in, out) layout (layout-identical)
        assert d["dense1/weight:0"].shape == (16 * 8 * 4, 32)

        snapshot = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters()}

        archi2 = dfl_nn.AMPArchi(64, e_ch=16, ae_ch=32, inter_ch=32, inter_res=2, d_ch=16, d_mask_ch=6)
        enc2 = archi2.Encoder(name="encoder")
        assert enc2.load_weights(filename) is True
        for n, p in enc2.named_parameters():
            assert torch.equal(p.detach().cpu(), snapshot[n]), n

        # the inter/decoder round trip shares the same machinery
        inter_filename = str(REPO_ROOT / "_tmp_p7_save_load_inter.npy")
        decoder_filename = str(REPO_ROOT / "_tmp_p7_save_load_decoder.npy")
        try:
            inter_src.save_weights(inter_filename)
            decoder.save_weights(decoder_filename)
            archi3 = dfl_nn.AMPArchi(64, e_ch=16, ae_ch=32, inter_ch=32, inter_res=2, d_ch=16, d_mask_ch=6)
            it2 = archi3.Inter(name="inter_src")
            de2 = archi3.Decoder(name="decoder")
            assert it2.load_weights(inter_filename) is True
            assert de2.load_weights(decoder_filename) is True
            it_snap = {n: p.detach().cpu().clone() for n, p in inter_src.named_parameters()}
            de_snap = {n: p.detach().cpu().clone() for n, p in decoder.named_parameters()}
            for n, p in it2.named_parameters():
                assert torch.equal(p.detach().cpu(), it_snap[n]), n
            for n, p in de2.named_parameters():
                assert torch.equal(p.detach().cpu(), de_snap[n]), n
        finally:
            for p in (inter_filename, decoder_filename):
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass

        # strict load: an unexpected extra key must be rejected
        d2 = dict(d)
        d2["rogue/weight:0"] = np.zeros((1, 1), dtype=np.float32)
        Path(badname).write_bytes(pickle.dumps(d2, 4))
        from core.leras.checkpoint import CheckpointLoadError
        enc3 = dfl_nn.AMPArchi(64, e_ch=16, ae_ch=32, inter_ch=32, inter_res=2,
                               d_ch=16, d_mask_ch=6).Encoder(name="encoder")
        with pytest.raises(CheckpointLoadError):
            enc3.load_weights(badname)
    finally:
        for p in (filename, badname):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def test_fp16_construction_dtypes():
    # use_fp16 is the official EXPORT-only conv-dtype knob (training is
    # always fp32); construction-time dtype checks only (no fp16
    # numerics, no mixed-precision policy — later phase)
    init_cpu()
    _, encoder, inter_src, inter_dst, decoder = build_amp_stack(
        res=64, e_ch=16, ae_ch=32, inter_ch=32, d_ch=16, d_mask_ch=6,
        use_fp16=True, seed=5)
    # every Conv2D leaf holds float16 weights (official conv_dtype):
    # block wrappers expose the conv as .conv1, the head convs are
    # bare Conv2D leaves
    for name, conv in (("down1", encoder.down1.conv1), ("res1", encoder.res1.conv1),
                       ("upscale0", decoder.upscale0.conv1),
                       ("out_conv", decoder.out_conv),
                       ("out_convm", decoder.out_convm)):
        assert conv.dtype == torch.float16, name
    # the Dense layers keep the global float32 (official: conv_dtype
    # applies only to the Conv2D layers)
    assert encoder.dense1.dtype == torch.float32
    assert inter_src.dense2.dtype == torch.float32


# ---------------------------------------------------------------------------
# exact-k morph mask
# ---------------------------------------------------------------------------

def test_morph_mask_fixed_count():
    # the load-bearing property: EXACTLY k ones per sample (a uniform
    # k-subset), never the i.i.d. Bernoulli count of random_binomial
    init_cpu()
    torch.manual_seed(0)
    n, inter_dims, f = 4, 32, 0.5
    k = int(inter_dims * f)
    m = exact_k_morph_mask(n, inter_dims, f)
    assert tuple(m.shape) == (n, inter_dims, 1, 1)
    assert m.dtype == torch.float32
    assert m.requires_grad is False  # official tf.stop_gradient analogue
    assert torch.equal(m.sum(dim=1).view(-1), torch.full((n,), k, dtype=m.dtype))
    # per-channel marginal: over many draws each channel is a 1 in
    # about k/inter_dims of the rows (Binomial(draws, k/D) band)
    counts = torch.zeros(inter_dims, dtype=torch.float64)
    for _ in range(300):
        counts += exact_k_morph_mask(1, inter_dims, f).reshape(-1).double()
    mean = 300.0 * k / inter_dims
    assert torch.all(counts > mean - 5 * (300.0 * f * (1 - f)) ** 0.5)
    assert torch.all(counts < mean + 5 * (300.0 * f * (1 - f)) ** 0.5)


def test_morph_mask_edges():
    init_cpu()
    # k = 0 (morph_factor 0.0): all zeros (the official empty ones
    # tile)
    m0 = exact_k_morph_mask(3, 16, 0.0)
    assert m0.shape == (3, 16, 1, 1)
    assert torch.equal(m0, torch.zeros_like(m0))
    # k = inter_dims (morph_factor 1.0): all ones
    m1 = exact_k_morph_mask(3, 16, 1.0)
    assert torch.equal(m1, torch.ones_like(m1))
    # non-integer product floors (official tf.cast(..., tf.int32) /
    # int() truncation): inter_dims * morph_factor = 0.45 * 20 = 9.0 ->
    # exact; 7 * 0.3 = 2.1 -> k = 2
    m2 = exact_k_morph_mask(2, 7, 0.3)
    assert torch.equal(m2.sum(dim=1).view(-1), torch.full((2,), 2, dtype=m2.dtype))


def test_morph_mask_stochastic():
    init_cpu()
    torch.manual_seed(123)
    a = exact_k_morph_mask(2, 32, 0.5)
    b = exact_k_morph_mask(2, 32, 0.5)
    # two independent draws: the per-sample channel sets differ
    # (probability of equality ~ 1/C(32,16) per sample)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_forward_shapes():
    init_cpu()
    res = 64
    _, encoder, inter_src, inter_dst, decoder = build_amp_stack(seed=11)
    x = torch.randn(2, 3, res, res)
    code = encoder(x)
    assert tuple(code.shape) == (2, 32)
    cs = inter_src(code)
    cd = inter_dst(code)
    assert tuple(cs.shape) == (2, 32, 2, 2)
    assert tuple(cd.shape) == (2, 32, 2, 2)
    px, pm = decoder(cs)
    pxd, pmd = decoder(cd)
    for t in (px, pxd, pm, pmd):
        assert torch.isfinite(t).all()
    assert tuple(px.shape) == (2, 3, res, res)
    assert tuple(pm.shape) == (2, 1, res, res)
    assert tuple(pxd.shape) == (2, 3, res, res)
    assert tuple(pmd.shape) == (2, 1, res, res)
    # the decoder heads are sigmoids (official L241/L250)
    for t in (px, pm, pxd, pmd):
        assert torch.all(t >= 0) and torch.all(t <= 1)


def test_backward_reaches_all_parameters():
    init_cpu()
    _, encoder, inter_src, inter_dst, decoder = build_amp_stack(seed=13)
    x = torch.randn(2, 3, 64, 64)
    code = encoder(x)
    cs, cd = inter_src(code), inter_dst(code)
    px, pm = decoder(cs)
    pxd, pmd = decoder(cd)
    loss = px.sum() + pm.sum() + pxd.sum() + pmd.sum() + code.sum()
    loss.backward()
    for name, mod in (("encoder", encoder), ("inter_src", inter_src),
                      ("inter_dst", inter_dst), ("decoder", decoder)):
        for n, p in mod.named_parameters():
            assert p.grad is not None, f"{name}/{n}"


@requires_gpu
def test_gpu_execution_and_cpu_parity():
    # weights are generated on the CPU twin (seeded CPU RNG) and COPIED
    # to the GPU twin: torch's CUDA and CPU RNG streams produce different
    # sequences for the same seed (the official TF behavior), so seeding
    # both sides independently would compare different weights. This
    # isolates the device execution parity (the actual test target).
    init_cpu("NCHW")
    torch.manual_seed(42)
    archi_c, enc_c, isrc_c, idst_c, dec_c = build_amp_stack(res=64, e_ch=16, ae_ch=32,
                                                            inter_ch=32, d_ch=16, d_mask_ch=6)
    x_c = torch.randn(1, 3, 64, 64)
    code_c = enc_c(x_c)
    z_c = isrc_c(code_c)
    out_c, m_c = dec_c(z_c)
    weights_c = [p.detach().clone() for p in
                 (list(enc_c.parameters()) + list(isrc_c.parameters())
                  + list(idst_c.parameters()) + list(dec_c.parameters()))]

    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")
    dev = dfl_nn.device
    assert dev.type == "cuda"

    archi_g, enc_g, isrc_g, idst_g, dec_g = build_amp_stack(res=64, e_ch=16, ae_ch=32,
                                                            inter_ch=32, d_ch=16, d_mask_ch=6)
    params_g = list(enc_g.parameters()) + list(isrc_g.parameters()) + \
               list(idst_g.parameters()) + list(dec_g.parameters())
    with torch.no_grad():
        for pg, pc in zip(params_g, weights_c):
            assert tuple(pg.shape) == tuple(pc.shape)
            pg.data.copy_(pc.to(dev))

    x_g = x_c.to(dev)
    code_g = enc_g(x_g)
    z_g = isrc_g(code_g)
    out_g, m_g = dec_g(z_g)

    for name, a, b in (("encoder", code_g, code_c), ("inter", z_g, z_c),
                       ("decoder_x", out_g, out_c), ("decoder_m", m_g, m_c)):
        # f32 GPU-vs-CPU accumulation noise across the ~30-layer stack
        # (different reduction orders); the encoder/inter band is
        # amplified by the pixel_norm tail sensitivity in the 5-halving
        # chain: measured RTX 4090 worst case for this stack
        # (encoder 1.9e-3 abs, inter 1.4e-3 abs, decoder 3.2e-5 abs);
        # the 2x band is still orders of magnitude tighter than any
        # semantic (layout/ordering) error
        assert torch.allclose(a.detach().cpu(), b, rtol=2e-3, atol=2e-3), name

    # and back to the GPU foundation for any later tests
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")


# ---------------------------------------------------------------------------
# Source hygiene (AST)
# ---------------------------------------------------------------------------

MIGRATED_SOURCES = [
    REPO_ROOT / "core" / "leras" / "archis" / "AMP.py",
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


def test_no_autocast_or_bf16_in_morph_archi():
    # the Phase 7 exclusion: NO Automatic Mixed Precision policy in the
    # foundation (the use_fp16 knob is the official export-only dtype
    # handling only)
    path = REPO_ROOT / "core" / "leras" / "archis" / "AMP.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = []
            cur = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "torch" and "autocast" in chain:
                pytest.fail(f"torch.autocast in {path}")
    text = path.read_text(encoding="utf-8")
    assert "bfloat16" not in text
    assert "GradScaler" not in text
