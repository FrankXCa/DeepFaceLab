"""Phase 6B: SAEHD single-device training semantics coverage —
the CPU tier.

Testing pyramid (docs/PHASE6B_STATE.md session 7 mandate):

    formula-level CPU unit tests
        -> tiny-model CPU integration
        -> representative CUDA full-model integration
           (test_model_saehd_training_cuda.py)
        -> real-artifact CUDA / final lifecycle validation

This file is the CPU tier: it proves correctness and backend
neutrality and stays FAST. Formula-level units (no model
construction) pin the loss/op formulas (style moments, the
dssim+MSE vs moments distinction, the official DLoss per-sample
semantics). The tiny integration models (64px, batch 1, the
smallest practical dims) prove the official training contract on
CPU: the full loss stack (reconstruction DSSIM/MSE/mask head,
eyes/mouth priority, blur_out_mask target rewrite, the softened
loss masks, the true-face / GAN routing and update order), the
per-optimizer gradient ownership (src_dst_opt over
src_dst_trainable_weights — liae without random_warp excludes
inter_AB — / D_code_opt / D_src_dst_opt), gradient hygiene (no
stale .grad across iterations — the state-restored double step
produces bit-identical losses and weight deltas), the official
onTrainOneIter driver, the res<=256 onGetPreview layout, and
save -> strict resume -> continue.

The EXPENSIVE integrated coverage (the all-terms 128 df-udt
twin-difference pins, the GAN/true-face gradient pins, the
representative archi u/d/t/c training, the 256 real-shape
configuration, the res>256 preview branch, the real-checkpoint
one-step lifecycle) lives in
test_model_saehd_training_cuda.py (skipped without CUDA).

Numerical strategy (mandated fallback — the TF-runtime parity
probe in docs/PHASE6B_STATE.md found no official TF stack on
disk): formula-level tests on hand-computable synthetic tensors
plus the migrated-op parity tests of the earlier phases. Every
result here is labeled FORMULA_VERIFIED / TF_RUNTIME_NOT_VERIFIED;
no test claims runtime parity with the official TF graph.
"""

import contextlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import nn
from core.leras import ops as leras_ops

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_SAEHDTest.Model import (  # noqa: E402
    DEFAULT_SEED_OPTIONS,
    SAEHDHeadless,
    make_model as make_saehd,
    make_training_dirs,
)

# the all-terms seed — the CUDA tier's full model (and its
# per-option twins) is built from it
FULL_SEED = dict(
    archi="df-udt",
    true_face_power=0.2,
    face_style_power=25.0,
    bg_style_power=10.0,
    eyes_mouth_prio=True,
    blur_out_mask=True,
    masked_training=True,
    gan_power=0.05,
    random_warp=False,
)

# the tiny CPU integration configuration: the smallest practical
# SAEHD (64px, batch 1) — one training step per test unless the
# behavior requires two (stale-grad proof, optimizer continuity,
# save/resume/continue)
TINY = dict(resolution=64, batch_size=1, ae_dims=32, e_dims=16,
            d_dims=16, d_mask_dims=16)


# --- helpers -------------------------------------------------------------------

def seed(**overrides):
    s = dict(DEFAULT_SEED_OPTIONS)
    s.update(overrides)
    return s


def full_seed(**overrides):
    s = dict(FULL_SEED)
    s.update(overrides)
    return s


def construct(tmpdir, is_training=False, seed=None, cpu_only=True):
    """Training-context construction (debug=True in-process
    generator); ``is_training=False`` skips the facesets (the
    train closures are not built in that mode)."""
    if is_training:
        make_training_dirs(Path(tmpdir))
    return make_saehd(SAEHDHeadless, tmpdir, is_training=is_training,
                      seed=seed, debug=is_training, cpu_only=cpu_only)


def synth_samples(res, batch=1, data_format="NHWC", seed_no=0,
                  src_level=0.4, dst_level=0.6, mask_frac=0.55,
                  em_frac=0.3, em_mask_zero=False):
    """Deterministic synthetic training batch in ``data_format``
    layout: the 8 arrays in the official onTrainOneIter unpack
    order (warped_src, target_src, target_srcm, target_srcm_em,
    warped_dst, target_dst, target_dstm, target_dstm_em). Images
    are smooth BGR fields (different per side); the full-face mask
    is a centered disc, the eyes/mouth mask a smaller disc (zeroed
    when ``em_mask_zero``)."""
    rng = np.random.RandomState(seed_no)

    def img(level):
        base = np.clip(rng.rand(res, res, 3) * 0.3 + level, 0.0, 1.0)
        y, x = np.mgrid[0:res, 0:res].astype(np.float32)
        base = base + 0.1 * np.sin(x / 23.0)[..., None]
        base = np.clip(base, 0.0, 1.0)
        return np.repeat(base[None, ...], batch, 0).astype(np.float32)

    def disc(frac):
        y, x = np.mgrid[0:res, 0:res].astype(np.float32)
        r = np.sqrt((y - res / 2) ** 2 + (x - res / 2) ** 2) / (res / 2)
        m = np.where(r < frac, 1.0, 0.0).astype(np.float32)
        if em_mask_zero:
            m = np.zeros_like(m)
        return np.repeat(m[None, :, :, None], batch, 0)

    warped_src = img(src_level) + 0.05
    target_src = img(src_level)
    target_srcm = disc(mask_frac)
    target_srcm_em = disc(em_frac)
    warped_dst = img(dst_level) + 0.05
    target_dst = img(dst_level)
    target_dstm = disc(mask_frac)
    target_dstm_em = disc(em_frac)
    arrays = (warped_src, target_src, target_srcm, target_srcm_em,
              warped_dst, target_dst, target_dstm, target_dstm_em)
    if data_format == "NCHW":
        arrays = tuple(a.transpose(0, 3, 1, 2).copy() for a in arrays)
    return tuple(np.ascontiguousarray(a) for a in arrays)


def forward_ae(model, warped_src, warped_dst):
    """Test-side replica of the official per-tower forward (the same
    component chain the model executes), for hand-computing the
    expected loss terms. No grad: the terms are loss values, not
    updates."""
    with torch.no_grad():
        if 'df' in model.archi_type:
            src_code = model.inter(model.encoder(warped_src))
            dst_code = model.inter(model.encoder(warped_dst))
            pred_src_src, pred_src_srcm = model.decoder_src(src_code)
            pred_dst_dst, pred_dst_dstm = model.decoder_dst(dst_code)
            pred_src_dst, pred_src_dstm = model.decoder_src(dst_code)
            pred_src_dst_no_code_grad = model.decoder_src(dst_code.detach())[0]
        else:
            src_code = model.encoder(warped_src)
            src_ab = model.inter_AB(src_code)
            src_code_dec = torch.concat([src_ab, src_ab], dim=nn.conv2d_ch_axis)
            dst_code = model.encoder(warped_dst)
            dst_b = model.inter_B(dst_code)
            dst_ab = model.inter_AB(dst_code)
            dst_code_dec = torch.concat([dst_b, dst_ab], dim=nn.conv2d_ch_axis)
            src_dst_code = torch.concat([dst_ab, dst_ab], dim=nn.conv2d_ch_axis)
            pred_src_src, pred_src_srcm = model.decoder(src_code_dec)
            pred_dst_dst, pred_dst_dstm = model.decoder(dst_code_dec)
            pred_src_dst, pred_src_dstm = model.decoder(src_dst_code)
            pred_src_dst_no_code_grad = model.decoder(src_dst_code.detach())[0]
        return {
            'src_code': src_code, 'dst_code': dst_code,
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
            'pred_src_dst_no_code_grad': pred_src_dst_no_code_grad,
        }


def tensors8(samples):
    return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in samples)


def copy_components(src_model, dst_model):
    """Copy every component's weights from ``src_model`` to
    ``dst_model`` (identical archi/dims seeds -> identical shapes)
    so twin models start from identical weights."""
    for attr in ('encoder', 'inter', 'inter_AB', 'inter_B',
                 'decoder_src', 'decoder_dst', 'decoder'):
        a, b = getattr(src_model, attr, None), getattr(dst_model, attr, None)
        if a is not None and b is not None:
            for pa, pb in zip(a.parameters(), b.parameters()):
                pb.data.copy_(pa.data)
    c, d = getattr(src_model, 'code_discriminator', None), \
        getattr(dst_model, 'code_discriminator', None)
    if c is not None and d is not None:
        for pc, pd in zip(c.parameters(), d.parameters()):
            pd.data.copy_(pc.data)
    g, h = getattr(src_model, 'D_src', None), getattr(dst_model, 'D_src', None)
    if g is not None and h is not None:
        for pg, ph in zip(g.parameters(), h.parameters()):
            ph.data.copy_(pg.data)


def param_snapshots(model):
    """{param: pre-step weight clone} for the src_dst saveable set."""
    return {p: p.detach().clone() for p in model.src_dst_saveable_weights}


def opt_snapshots(opt):
    """{state tensor: clone} incl. the iterations counter (the
    optimizer's official get_weights() layout)."""
    return {w: w.clone() for w in opt.get_weights()}


def restore_opt(opt, snaps):
    for w in opt.get_weights():
        if w in snaps:
            w.copy_(snaps[w])


# --- shared tiny models (module scope: cheap to build) --------------------------

@pytest.fixture(scope="module")
def tiny_liae_model(tmp_path_factory):
    # liae-ud, the default 6B seed: masked_training on,
    # random_warp on, every power 0 -> pure reconstruction
    return construct(tmp_path_factory.mktemp("p6b_t_liae"), is_training=True,
                     seed=seed(**TINY), cpu_only=True)


@pytest.fixture(scope="module")
def tiny_df_model(tmp_path_factory):
    # the df component chain (encoder/inter/decoder_src/decoder_dst)
    return construct(tmp_path_factory.mktemp("p6b_t_df"), is_training=True,
                     seed=seed(archi="df-ud", **TINY), cpu_only=True)


@pytest.fixture(scope="module")
def tiny_nowarp_model(tmp_path_factory):
    # liae WITHOUT random_warp: the official rule excludes inter_AB
    # from src_dst_trainable_weights
    return construct(tmp_path_factory.mktemp("p6b_t_nowarp"), is_training=True,
                     seed=seed(random_warp=False, **TINY), cpu_only=True)


@pytest.fixture(scope="module")
def tiny_tfp_model(tmp_path_factory):
    # df + true-face + GAN: all three optimizers active — the
    # discriminator routing + update order on the CPU backend
    return construct(tmp_path_factory.mktemp("p6b_t_tfp"), is_training=True,
                     seed=seed(archi="df-ud", true_face_power=0.1,
                               gan_power=0.05, **TINY), cpu_only=True)


@pytest.fixture(scope="module")
def tiny_blur_model(tmp_path_factory):
    # blur_out_mask on: the outside-mask target rewrite in
    # _prepare_targets
    return construct(tmp_path_factory.mktemp("p6b_t_blur"), is_training=True,
                     seed=seed(blur_out_mask=True, **TINY), cpu_only=True)


@pytest.fixture(scope="module")
def tiny_em_model(tmp_path_factory):
    # eyes/mouth priority on (liae base) — the 300x MAE term
    return construct(tmp_path_factory.mktemp("p6b_t_em"), is_training=True,
                     seed=seed(eyes_mouth_prio=True, **TINY), cpu_only=True)


# --- formula-level units (no SAEHD model construction) --------------------------

# the class-level state attributes nn.initialize() mutates (the
# ``nn`` foundation is a stateful class, not a module)
_NN_STATE_ATTRS = ("current_DeviceConfig", "device", "floatx",
                   "data_format", "conv2d_ch_axis",
                   "conv2d_spatial_axes")


@contextlib.contextmanager
def nn_nchw_state():
    """Run the pure-op unit tests under NCHW without leaking the
    data-format change into the module-level model fixtures (the
    migrated nn state is global; the models in this file are all
    built NHWC by their debug flag). The state attributes
    nn.initialize() mutates are snapshotted and restored."""
    saved = {k: getattr(nn, k) for k in _NN_STATE_ATTRS}
    try:
        nn.initialize(nn.DeviceConfig.CPU(), data_format="NCHW")
        yield
    finally:
        for k, v in saved.items():
            setattr(nn, k, v)


def test_style_moments_formula_unit():
    """The migrated style_loss is the official per-channel MOMENTS
    formula (mean, then the std of the squared deviations,
    per-channel squared mean/std diffs summed, x loss_weight / C)
    — hand-computed on a tiny tensor. This pins the formula (no
    gram matrix) that the face-style term uses. No model is
    involved: the op is layout-aware and runs under a restored
    NCHW state (see nn_nchw_state)."""
    with nn_nchw_state():
        torch.manual_seed(0)
        t = torch.rand(1, 2, 4, 4)
        s = torch.rand(1, 2, 4, 4)
        out = leras_ops.style_loss(t, s, gaussian_blur_radius=0.0, loss_weight=1.0)
    per_ch = []
    for c in range(2):
        tc, sc = t[0, c], s[0, c]
        tm, sm = tc.mean(), sc.mean()
        tv = ((tc - tm).pow(2).mean()).sqrt()
        sv = ((sc - sm).pow(2).mean()).sqrt()
        per_ch.append((tm - sm) ** 2 + (tv - sv) ** 2)
    expected = sum(per_ch) / 2  # loss_weight / C
    assert torch.allclose(out.reshape(-1)[0], expected, rtol=1e-5, atol=1e-5)


def test_bg_style_is_not_the_moments_op_unit():
    """Guard: the bg 'style' term must not route through the
    moments style op — its official expression is dssim + MSE
    (the CUDA-tier twin pin checks it against dssim/MSE, not
    style_loss). Here: style_loss on the same tensors is a
    different value than the dssim+MSE expression (sanity that
    the two formulas are not interchangeable)."""
    with nn_nchw_state():
        torch.manual_seed(1)
        a = torch.rand(1, 3, 8, 8)
        b = torch.rand(1, 3, 8, 8)
        d = nn.dssim(a, b, max_val=1.0, filter_size=3)
        dssim_term = torch.mean(10 * 0.1 * d, dim=1)
        mse_term = torch.mean(10 * 0.1 * torch.square(a - b), dim=(1, 2, 3))
        moments = nn.style_loss(a, b, gaussian_blur_radius=0.0,
                                loss_weight=10000 * 0.1)
        bg = dssim_term + mse_term
    assert not torch.allclose(bg, moments, rtol=1e-3, atol=1e-3)


def test_dlloss_per_sample_semantics_unit():
    """The official DLoss (the Phase 6B foundation op
    nn.sigmoid_cross_entropy): elementwise sigmoid BCE with
    reduction='none' (the caller performs the official per-sample
    mean over axes (1,2,3) — torch's global 'mean' would also
    divide by the batch and is WRONG for the official
    per-sample-mean semantics) — the result is a per-sample
    vector (N,), not a scalar; the ones/zeros label semantics are
    distinct. Pinned against a hand-computed BCE on a tiny code
    discriminator (no full SAEHD model needed). The official
    two-phase layer lifecycle is completed here explicitly
    (build_leaf_weights in the constructor, then init_weights):
    without the init phase the conv weights are torch.empty
    allocator garbage and the logits — hence the label-distinct
    assert below — would depend on process memory contents."""
    with nn_nchw_state():
        torch.manual_seed(7)
        cd = nn.CodeDiscriminator(8, code_res=4, name="dis")
        cd.init_weights()  # official phase 2: glorot weights, zero biases
        x = torch.rand(2, 8, 4, 4)  # NCHW: 2 samples, 8 channels, 4x4
        y = cd(x)
        out = nn.sigmoid_cross_entropy(torch.ones_like(y), y)
        assert out.shape == (2,)  # per-sample vector
        ref = F.binary_cross_entropy_with_logits(
            y, torch.ones_like(y), reduction="none").mean(dim=(1, 2, 3))
        assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
        out0 = nn.sigmoid_cross_entropy(torch.zeros_like(y), y)
        assert out0.shape == (2,)
        assert not torch.allclose(out, out0, rtol=1e-3, atol=1e-3)


# --- reconstruction ------------------------------------------------------------

def test_recon_basic_iteration_cpu(tiny_liae_model):
    """One official iteration on the tiny liae model: the
    two-value loss return, the iter increment, exactly the
    src_loss/dst_loss history columns, the src_dst_opt iteration
    advance."""
    model = tiny_liae_model
    it_before = model.src_dst_opt.iterations.item()
    model.train_one_iter()
    assert model.iter == 1
    row = model.loss_history[-1]
    assert len(row) == 2
    assert all(math.isfinite(v) for v in row)
    assert model.src_dst_opt.iterations.item() == it_before + 1
    # the base seed has no true-face/GAN: those components do not
    # exist, so only src_dst_opt may step
    assert not hasattr(model, 'code_discriminator')
    assert not hasattr(model, 'D_src')


def test_recon_df_backend_neutrality_cpu(tiny_df_model):
    """The df component chain (encoder/inter/decoder_src/
    decoder_dst) completes one official iteration on the CPU
    backend: the same two-value loss contract as liae."""
    model = tiny_df_model
    it_before = model.src_dst_opt.iterations.item()
    model.train_one_iter()
    assert model.iter == 1
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)
    assert model.src_dst_opt.iterations.item() == it_before + 1


def test_recon_two_iterations_history_cpu(tiny_liae_model):
    """Two consecutive iterations: the history accumulates the
    official two columns per iter and the AdaBelief state advances
    (moment buffers become nonzero) — optimizer iteration
    continuity."""
    model = tiny_liae_model
    n0 = len(model.loss_history)
    model.train_one_iter()
    model.train_one_iter()
    assert len(model.loss_history) == n0 + 2
    for row in model.loss_history:
        assert len(row) == 2
    state_nonzero = False
    for w in model.src_dst_opt.get_weights()[1:]:
        if w.numel() and w.abs().max().item() > 0:
            state_nonzero = True
            break
    assert state_nonzero


def test_recon_src_dst_separation_cpu(tiny_liae_model):
    """The src and dst losses are independent: perturbing only the
    SRC targets changes only the src loss (the dst loss uses no
    src tensor in the base reconstruction-only seed). The weight
    state is restored between the two evaluations so both losses
    are computed at identical weights (each _src_dst_train call
    ends with the official optimizer step)."""
    model = tiny_liae_model
    samples = synth_samples(model.resolution, batch=1, seed_no=7)
    (ws, ts, tm, tm_em, wd, td, dm, dm_em) = tensors8(samples)

    snaps = param_snapshots(model)
    opt_snaps = opt_snapshots(model.src_dst_opt)
    s0, d0 = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)

    # restore the pre-step state (weights + optimizer state +
    # fresh grads) so the second evaluation sees identical weights
    for p, w in snaps.items():
        p.data.copy_(w)
    restore_opt(model.src_dst_opt, opt_snaps)
    for p in model.src_dst_saveable_weights:
        p.grad = None

    ts2 = (ts + 0.15).clamp(0, 1)
    s1, d1 = model._src_dst_train(ws, ts2, tm, tm_em, wd, td, dm, dm_em)
    assert torch.equal(d0, d1), "dst loss must not depend on src tensors"
    assert not torch.allclose(s0, s1)


def test_recon_masked_multiplicity_cpu(tiny_liae_model):
    """masked_training on: the loss masks REPLACE target/pred
    inside the dssim/mse terms (official L453-456, no additive
    split) — pinned by the exposed _prepare_targets: masked_opt ==
    raw * m_blur exactly; the mask softening is blur -> clip
    0..0.5 -> x2 with anti = 1 - m and the official dead-code
    style-mask override (stop-grad clip of the SRC mask)."""
    model = tiny_liae_model
    samples = synth_samples(model.resolution, batch=1, seed_no=11)
    t = model._prepare_targets(*samples)
    m = t['target_srcm_blur']
    assert torch.allclose(t['target_src_masked_opt'],
                          t['target_src'] * m, atol=0.0)
    assert torch.allclose(t['target_dst_masked_opt'],
                          t['target_dst'] * t['target_dstm_blur'], atol=0.0)
    expected_m = torch.clip(
        nn.gaussian_blur(t['target_srcm'], max(1, model.resolution // 32)),
        0, 0.5) * 2
    assert torch.allclose(m, expected_m, rtol=1e-5, atol=1e-5)
    assert torch.allclose(t['target_srcm_anti_blur'], 1.0 - m, atol=1e-6)
    # the style mask is the official override (L445): the stop-grad
    # clip of the SRC-side blurred mask; its complement is the bg
    # anti-mask applied to the DST tensors
    assert torch.allclose(t['style_mask_anti_blur'], 1.0 - m, atol=1e-6)


def test_recon_blur_out_mask_preprocessing_cpu(tiny_blur_model):
    """blur_out_mask on: outside-mask target pixels are replaced
    by the sigma-blur of target*mask_anti normalized by the
    sigma-blurred anti-mask (div-zero guarded); inside-mask pixels
    keep the raw target. Pinned against the official formula
    evaluated on the raw synthetic tensors."""
    model = tiny_blur_model
    raw = synth_samples(model.resolution, batch=1, seed_no=13)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(raw)
    t = model._prepare_targets(*raw)
    sigma = model.resolution / 128

    anti = 1 - tm
    x = nn.gaussian_blur(ts * anti, sigma)
    y = torch.where(1 - nn.gaussian_blur(tm, sigma) == 0,
                    torch.ones_like(tm), 1 - nn.gaussian_blur(tm, sigma))
    expected = ts * tm + (x / y) * anti
    assert torch.allclose(t['target_src'], expected, rtol=1e-5, atol=1e-5)

    anti_d = 1 - dm
    x_d = nn.gaussian_blur(td * anti_d, sigma)
    y_d = torch.where(1 - nn.gaussian_blur(dm, sigma) == 0,
                      torch.ones_like(dm), 1 - nn.gaussian_blur(dm, sigma))
    expected_d = td * dm + (x_d / y_d) * anti_d
    assert torch.allclose(t['target_dst'], expected_d, rtol=1e-5, atol=1e-5)

    # inside the mask (tm == 1) the replacement is exactly the raw
    # target (the anti factor is zero there)
    assert torch.allclose(t['target_src'] * tm, ts * tm, rtol=1e-5, atol=1e-5)


def test_recon_deterministic_double_step_no_stale_grad_cpu(tiny_liae_model):
    """The gradient-hygiene proof (mandated): a state-restored
    double step on identical inputs produces BIT-identical losses
    and weight deltas. If any .grad accumulated across the step
    boundary, the second backward would see g + g_stale and both
    the loss update and the weight delta would differ."""
    model = tiny_liae_model
    samples = synth_samples(model.resolution, batch=1, seed_no=17)
    xs = tensors8(samples)

    snaps = param_snapshots(model)
    opt_snaps = opt_snapshots(model.src_dst_opt)
    s0, d0 = model._src_dst_train(*xs)
    delta = {p: (p.detach().clone() - w) for p, w in snaps.items()}

    # restore the pre-step state exactly (weights + optimizer
    # iterations/moments + fresh grads) and repeat the identical
    # step
    for p, w in snaps.items():
        p.data.copy_(w)
    restore_opt(model.src_dst_opt, opt_snaps)
    for p in model.src_dst_saveable_weights:
        p.grad = None

    s1, d1 = model._src_dst_train(*xs)
    delta2 = {p: (p.detach().clone() - w) for p, w in snaps.items()}

    assert torch.equal(s0, s1)
    assert torch.equal(d0, d1)
    for p in snaps:
        assert torch.equal(delta[p], delta2[p]), "stale .grad accumulated"


def test_recon_gradient_ownership_cpu(tiny_liae_model):
    """After the src_dst step, every TRAINABLE generator parameter
    carries a fresh gradient (the official
    nn.gradients(G_loss, src_dst_trainable_weights) variable set).
    For the default liae + random_warp seed the trainable set is
    the full saveable set."""
    model = tiny_liae_model
    samples = synth_samples(model.resolution, batch=1, seed_no=19)
    model._src_dst_train(*tensors8(samples))
    assert model.src_dst_trainable_weights is model.src_dst_saveable_weights
    for p in model.src_dst_trainable_weights:
        assert p.grad is not None


def test_recon_liae_no_random_warp_excludes_inter_ab_cpu(tiny_nowarp_model):
    """The official liae rule (L335-338): without random_warp the
    trainable set EXCLUDES inter_AB — its weights never change
    after a src_dst step (its optimizer state, while present in
    src_dst_opt, stays untouched by the update op)."""
    model = tiny_nowarp_model
    assert model.src_dst_trainable_weights is not model.src_dst_saveable_weights
    inter_ab_ids = {id(p) for p in model.inter_AB.get_weights()}
    assert inter_ab_ids.isdisjoint(
        {id(p) for p in model.src_dst_trainable_weights})

    samples = synth_samples(model.resolution, batch=1, seed_no=23)
    before = [p.detach().clone() for p in model.inter_AB.get_weights()]
    model._src_dst_train(*tensors8(samples))
    after = model.inter_AB.get_weights()
    assert len(before) == len(after)
    for wb, wa in zip(before, after):
        assert torch.equal(wb, wa), "inter_AB weights must stay frozen"
    iters = model.src_dst_opt.get_weights()[0]
    assert iters.item() >= 1


# --- eyes / mouth priority (tiny twin pins) --------------------------------------

def test_eyes_mouth_zero_mask_identity_cpu(tiny_em_model, tiny_liae_model,
                                           tmp_path_factory):
    """With a ZERO eyes/mouth mask the 300x MAE term contributes
    exactly 0.0, so the src loss of the em-on model must equal
    the src loss of an em-off twin at identical weights (the term
    is the only difference)."""
    model = tiny_em_model
    twin = construct(tmp_path_factory.mktemp("p6b_t_em_twin"),
                     is_training=True, seed=seed(**TINY), cpu_only=True)
    copy_components(model, twin)
    samples = synth_samples(model.resolution, batch=1, seed_no=29,
                            em_mask_zero=True)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    s_model, _ = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    s_twin, _ = twin._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert torch.allclose(s_model, s_twin, rtol=1e-5, atol=1e-5)


def test_eyes_mouth_contribution_pin_cpu(tiny_em_model, tmp_path_factory):
    """The em term is pinned to its official expression: 300 *
    mean(|target*em - pred_src_src*em|, axes [1,2,3]) — the
    difference between the em-on model and an em-off twin at
    identical weights on a NONZERO em mask equals the hand-
    computed term."""
    model = tiny_em_model
    twin = construct(tmp_path_factory.mktemp("p6b_t_em_twin2"),
                     is_training=True, seed=seed(**TINY), cpu_only=True)
    copy_components(model, twin)
    samples = synth_samples(model.resolution, batch=1, seed_no=31)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    t = model._prepare_targets(*samples)
    with torch.no_grad():
        f = forward_ae(model, t['warped_src'], t['warped_dst'])
        expected = torch.mean(
            300 * torch.abs(t['target_src'] * t['target_srcm_em']
                            - f['pred_src_src'] * t['target_srcm_em']),
            dim=(1, 2, 3))
        assert expected.abs().max().item() > 0  # the mask is nonzero
    s_on, _ = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    s_off, _ = twin._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert torch.allclose(s_on - s_off, expected, rtol=1e-4, atol=1e-4)


# --- true face / GAN routing (tiny) ----------------------------------------------

def test_true_face_off_absent_cpu(tiny_liae_model):
    """true_face_power == 0: the code discriminator and D_code_opt
    are not constructed at all (official creation rule)."""
    model = tiny_liae_model
    assert model.options['true_face_power'] == 0
    assert not hasattr(model, 'code_discriminator')
    assert not hasattr(model, 'D_code_opt')


def test_true_face_liae_absent_cpu(tiny_nowarp_model):
    """The code discriminator is the official df-only path: a liae
    model carries no true-face component."""
    model = tiny_nowarp_model
    assert not hasattr(model, 'code_discriminator')


def test_gan_off_no_update_cpu(tiny_liae_model):
    """gan_power == 0: no D_src component, no GAN_opt, and a full
    official iteration runs only the src_dst step."""
    model = tiny_liae_model
    assert model.gan_power == 0
    assert not hasattr(model, 'D_src')
    assert not hasattr(model, 'D_src_dst_opt')
    it = model.src_dst_opt.iterations.item()
    model.train_one_iter()
    assert model.src_dst_opt.iterations.item() == it + 1


def test_discriminator_routing_update_order_cpu(tiny_tfp_model):
    """CPU backend neutrality for the discriminator paths: with
    true-face + GAN active, one official iteration advances all
    THREE optimizers by exactly one (the src_dst step first, then
    the code-D step, then the D_src step — the official update
    order). The per-group weight-change isolation (code-D step
    touches no D_src weight and vice versa) is pinned at full
    scale in the CUDA tier."""
    model = tiny_tfp_model
    it_gen = model.src_dst_opt.iterations.item()
    it_code = model.D_code_opt.iterations.item()
    it_gan = model.D_src_dst_opt.iterations.item()
    # the official driver (train_one_iter wraps onTrainOneIter and
    # appends the two-column loss history)
    model.train_one_iter()
    assert model.src_dst_opt.iterations.item() == it_gen + 1
    assert model.D_code_opt.iterations.item() == it_code + 1
    assert model.D_src_dst_opt.iterations.item() == it_gan + 1
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)


def test_clipgrad_step_cpu(tmp_path_factory):
    """clipgrad=True: the migrated global-norm clipnorm path runs
    inside the official step (the Phase 3E2 optimizer unit tests
    pin the formula; this proves the SAEHD wiring reaches it)."""
    model = construct(tmp_path_factory.mktemp("p6b_t_clip"), is_training=True,
                      seed=seed(clipgrad=True, **TINY), cpu_only=True)
    samples = synth_samples(model.resolution, batch=1, seed_no=61)
    model._src_dst_train(*tensors8(samples))
    assert model.src_dst_opt.iterations.item() == 1
    assert model.src_dst_opt.clipnorm == 1.0


def test_lr_dropout_cos_step_cpu(tmp_path_factory):
    """lr_dropout='y' + adabelief: the official construction
    coupling (lr_cos=500, lr_dropout=0.3) is live during training
    and the optimizer state (iterations) persists across steps."""
    model = construct(tmp_path_factory.mktemp("p6b_t_lr"), is_training=True,
                      seed=seed(lr_dropout='y', clipgrad=False, **TINY),
                      cpu_only=True)
    assert model.src_dst_opt.lr_cos == 500
    assert model.src_dst_opt.lr_dropout == 0.3
    samples = synth_samples(model.resolution, batch=1, seed_no=67)
    x = tensors8(samples)
    model._src_dst_train(*x)
    model._src_dst_train(*x)
    assert model.src_dst_opt.iterations.item() == 2


# --- save / resume ----------------------------------------------------------------

def test_save_resume_continue_cpu(tiny_liae_model):
    """Save after training -> strict resume in the same directory
    -> continue: the iter, the optimizer iterations, the loss
    history all continue; one more iteration after the resume
    keeps every counter consistent."""
    model = tiny_liae_model
    model.train_one_iter()
    it_after_train = model.iter
    iters_after_train = model.src_dst_opt.iterations.item()
    hist_after_train = [row[:] for row in model.loss_history]
    model.save()

    # resume with the SAME seed the original construction used:
    # the headless seed layer re-applies the seed over the restored
    # data.dat options (the production model has no such layer — it
    # restores purely from data.dat), so the resumed options must
    # equal the saved ones for the strict component loads
    model2 = make_saehd(SAEHDHeadless, str(model.saved_models_path),
                        is_training=True, seed=seed(**TINY), debug=True,
                        cpu_only=True)
    assert model2.iter == it_after_train
    assert model2.src_dst_opt.iterations.item() == iters_after_train
    assert [row[:] for row in model2.loss_history] == hist_after_train

    model2.train_one_iter()
    assert model2.iter == it_after_train + 1
    assert model2.src_dst_opt.iterations.item() == iters_after_train + 1
    assert len(model2.loss_history[-1]) == 2


# --- preview (res <= 256 branch; the res>256 branch is CUDA-tier) ----------------

def test_preview_res_le_256_layout_cpu(tiny_liae_model):
    """onGetPreview at resolution <= 256: exactly two previews
    ('SAEHD' and 'SAEHD masked'), the official row layouts, NHWC
    (n, H, W*cols, 3) strips, n_samples = min(4, batch,
    800//res) and the strip values are exactly the clipped
    target / AE_view outputs (AE fed with the UNWARPED targets).
    The res>256 six-preview branch is pinned in the CUDA tier
    (the layout code is resolution-independent)."""
    model = tiny_liae_model
    res = model.resolution
    assert res <= 256
    samples = synth_samples(res, batch=2, seed_no=71)
    (ws, ts, tm, tm_em, wd, td, dm, dm_em) = samples
    # the official onGetPreview contract: the generate_next_samples()
    # structure — two per-side lists of 4 arrays
    previews = model.onGetPreview(
        [(ws, ts, tm, tm_em), (wd, td, dm, dm_em)])
    assert [name for name, _ in previews] == ['SAEHD', 'SAEHD masked']
    n = min(4, model.get_batch_size(), 800 // res)
    # the official per-sample concatenate + np.concatenate(st,
    # axis=0) drops the leading axis when n_samples == 1
    want = (n, res, 5 * res, 3) if n > 1 else (res, 5 * res, 3)
    for _, img in previews:
        assert img.shape == want
    ae = model.AE_view(ts, td)
    SS = np.clip(np.asarray(ae[0]), 0, 1)
    DD = np.clip(np.asarray(ae[1]), 0, 1)
    SD = np.clip(np.asarray(ae[3]), 0, 1)
    S = np.clip(np.asarray(ts), 0, 1)
    D = np.clip(np.asarray(td), 0, 1)
    first = previews[0][1]
    first0 = first[0] if n > 1 else first
    assert np.allclose(first0[:, :res], S[0], atol=1e-6)
    assert np.allclose(first0[:, res:2 * res], SS[0], atol=1e-6)
    assert np.allclose(first0[:, 2 * res:3 * res], D[0], atol=1e-6)
    assert np.allclose(first0[:, 3 * res:4 * res], DD[0], atol=1e-6)
    assert np.allclose(first0[:, 4 * res:5 * res], SD[0], atol=1e-6)


# --- official nn.gradients batch-SUM pin (Phase 8 fix) -------------------------

def test_train_ones_seed_batch_sum_cpu(tmp_path_factory):
    """Official nn.gradients(loss_vec, vars) = the per-sample
    batch SUM (the model suggests batch 4-8) — the migrated
    torch.autograd.backward(loss_vec, torch.ones_like(loss_vec))
    must reproduce it: on a doubled batch [x1, x1] the gradient
    accumulated for EVERY trainable src_dst parameter is exactly
    2x the N=1 gradient at identical weights. The CPU kernels are
    deterministic for these shapes, so the strict Phase 7 AMP
    Q11-CPU-pin tolerance applies (rtol=1e-5, atol=1e-6). This
    test pins the SAEHD G-loss site (previously a bare
    .backward(), which torch only accepts for numel()==1 and
    which crashes for the official batch sizes) and the
    per-sample loss-vector layout itself."""
    model = construct(tmp_path_factory.mktemp("q11_g"), is_training=True,
                      seed=seed(**TINY), cpu_only=True)
    res = model.resolution

    s1t = tensors8(synth_samples(res, batch=1, seed_no=0))
    s2t = tensors8(synth_samples(res, batch=2, seed_no=0))

    snaps = param_snapshots(model)
    g1_src, g1_dst = model._src_dst_train(*s1t)   # N=1: grads + step
    grads_n1 = {id(p): p.grad.detach().clone()
                for p in model.src_dst_trainable_weights}
    assert torch.isfinite(g1_src).all()
    assert torch.isfinite(g1_dst).all()
    for p, w in snaps.items():
        p.data.copy_(w)                            # identical weights for N=2

    g2_src, g2_dst = model._src_dst_train(*s2t)    # N=2: grads + step
    assert torch.isfinite(g2_src).all()
    assert torch.isfinite(g2_dst).all()
    # the per-sample loss rows of the doubled batch equal the
    # duplicated N=1 rows (identical inputs, identical weights)
    assert torch.allclose(g2_src, torch.cat([g1_src, g1_src]),
                          rtol=1e-5, atol=1e-6)
    assert torch.allclose(g2_dst, torch.cat([g1_dst, g1_dst]),
                          rtol=1e-5, atol=1e-6)
    n_checked = 0
    for p in model.src_dst_trainable_weights:
        assert p.grad is not None, "src_dst weight without grad"
        g1 = grads_n1[id(p)]
        assert torch.allclose(p.grad, 2 * g1, rtol=1e-5, atol=1e-6), \
            f"batch-SUM doubling violated for a {tuple(p.shape)} weight"
        n_checked += 1
    assert n_checked > 0


def _all_component_weights(model):
    """Every component parameter (generator + discriminators) —
    the no_grad AE re-forwards inside the D closures make the D
    gradients depend on the generator weights, so the doubling
    comparison must restore all of them."""
    out = []
    for attr in ('encoder', 'inter', 'inter_AB', 'inter_B',
                 'decoder_src', 'decoder_dst', 'decoder',
                 'code_discriminator', 'D_src'):
        comp = getattr(model, attr, None)
        if comp is not None:
            out.extend(comp.parameters())
    return out


def test_disc_train_ones_seed_batch_sum_cpu(tmp_path_factory):
    """The same batch-SUM identity for the two discriminator
    nn.gradients sites (official L514 code-D, L537 D_src): on a
    doubled batch [x1, x1], every code-D weight gradient is 2x
    the N=1 value and every D_src weight gradient is 2x the N=1
    value, at identical weights (generator weights restored too,
    since the closures re-forward AE under no_grad).

    Tolerance (measured noise floor, Phase 7 style — never
    loosened to pass): the N=1 vs N=2 forward kernels differ by
    last-bit rounding (the tiny df-udt AE stack shows ~4e-6
    absolute in encoder rows; every archi layer is pure
    conv/linear — no batch-dependent layer exists), and the
    D_src gradients are small cancellation sums (the in_conv
    bias gradient ~1e-2..1e-4 sums O(1e4) mixed-sign terms), so
    the amplified deviation floor measured on the CPU tier is
    ~1.2e-6 absolute worst case (in_conv bias; other disc
    weights < 4e-7). The pin asserts atol=1e-5 (~8x the
    measured floor) + rtol=1e-4; any genuine semantic defect
    (crash at N>1, mean-reduction, missing 0.5, wrong label
    routing) would deviate by O(1e-2..1) — three or more
    orders of magnitude beyond the floor."""
    model = construct(tmp_path_factory.mktemp("q11_d"), is_training=True,
                      seed=full_seed(**TINY), cpu_only=True)
    res = model.resolution

    s1t = tensors8(synth_samples(res, batch=1, seed_no=0))
    s2t = tensors8(synth_samples(res, batch=2, seed_no=0))
    ws1 = s1t[0]
    wd1 = s1t[4]
    ws2 = s2t[0]
    wd2 = s2t[4]

    snaps = {p: p.detach().clone() for p in _all_component_weights(model)}
    opt_snaps_code = opt_snapshots(model.D_code_opt)
    opt_snaps_src = opt_snapshots(model.D_src_dst_opt)

    # N=1: both D closures (each steps its own optimizer)
    model._D_train(ws1, wd1)
    code_grads_n1 = {id(p): p.grad.detach().clone()
                     for p in model.code_discriminator.get_weights()
                     if p.grad is not None}
    model._D_src_dst_train(*s1t)
    dsrc_grads_n1 = {id(p): p.grad.detach().clone()
                     for p in model.D_src.get_weights()
                     if p.grad is not None}
    assert code_grads_n1 and dsrc_grads_n1

    # restore the pre-step state (weights + optimizer states) so the
    # N=2 evaluations run at identical weights
    for p, w in snaps.items():
        p.data.copy_(w)
    restore_opt(model.D_code_opt, opt_snaps_code)
    restore_opt(model.D_src_dst_opt, opt_snaps_src)

    # N=2: both D closures
    model._D_train(ws2, wd2)
    for p in model.code_discriminator.get_weights():
        assert p.grad is not None, "code-D weight without grad"
        g1 = code_grads_n1[id(p)]
        assert torch.allclose(p.grad, 2 * g1, rtol=1e-4, atol=1e-5), \
            f"code-D batch-SUM doubling violated for a {tuple(p.shape)} weight"
    model._D_src_dst_train(*s2t)
    for p in model.D_src.get_weights():
        assert p.grad is not None, "D_src weight without grad"
        g1 = dsrc_grads_n1[id(p)]
        assert torch.allclose(p.grad, 2 * g1, rtol=1e-4, atol=1e-5), \
            f"D_src batch-SUM doubling violated for a {tuple(p.shape)} weight"
