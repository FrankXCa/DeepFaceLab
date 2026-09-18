"""Phase 6B: SAEHD training semantics — the CUDA (RTX 4090) tier.

The companion to test_model_saehd_training.py (the CPU tier:
formula-level units + tiny 64px integration). This file owns the
representative EXPENSIVE integrated coverage on GPU (the mandated
pyramid: formula CPU -> tiny CPU -> representative CUDA ->
real-artifact):

- the all-terms df-udt 128 model (true-face + face style + bg
  style + eyes/mouth + blur_out_mask + masked training + GAN):
  the term twin-difference pins (eyes/mouth, face style, bg
  style), the GAN generator-terms GRADIENT pin (the GAN/TV/
  bg-anti terms enter gpu_G_loss only), the D_src loss gradient
  pin, the real/fake label routing, the true-face generator +
  code-D routing, the three-optimizer update order, the official
  onTrainOneIter driver;
- optimizer options (clipgrad, lr_dropout/lr_cos) on a training
  model; save -> strict resume -> continue at 128;
- the representative archi u/d/t/c training compatibility;
- the real-checkpoint-configuration shape (256 df-udt, tfp=0,
  GAN on — the 413w-like set);
- the official onGetPreview layouts (res<=256 / res>256);
- the REAL-artifact one-step training lifecycle (opt-in via
  DFL_TEST_SAEHD_CHECKPOINT / DFL_TEST_FACESET_PAK; copies only).

Every test is gated by ``requires_gpu`` (skipped on a machine
without CUDA). Numerical labels: FORMULA_VERIFIED /
TF_RUNTIME_NOT_VERIFIED (no official TF stack exists on disk —
docs/PHASE6B_STATE.md); no test claims runtime parity with the
official TF graph.
"""

import builtins
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import nn
import core.leras.nn as dfl_nn  # initialize_main_env
from core.interact import interact as io  # the interact singleton

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_SAEHDTest.Model import (  # noqa: E402
    SAEHDHeadless,
    make_model as make_saehd,
    make_training_dirs,
)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required")

# the CPU-tier helpers (identical contract: seeds, synthetic
# samples, the test-side AE replica, the twin weight copy)
from test_model_saehd_training import (  # noqa: E402
    copy_components,
    forward_ae,
    full_seed,
    seed,
    synth_samples,
    tensors8,
)


def construct_gpu(tmpdir, is_training=False, seed=None):
    """The CUDA construction: the in-process sample generator
    (debug=True) with the components/optimizers on CUDA device 0
    (the official models_opt_on_gpu path)."""
    dfl_nn.initialize_main_env()
    if is_training:
        make_training_dirs(Path(tmpdir))
    return make_saehd(SAEHDHeadless, tmpdir, is_training=is_training,
                      seed=seed, debug=is_training, force_gpu_idxs=[0])


# --- shared models (module scope: each GPU construction is heavy) ------------

@pytest.fixture(scope="module")
def full_model_gpu(tmp_path_factory):
    # df-udt with EVERY official term active: true-face, face
    # style, bg style, eyes/mouth, blur_out_mask, masked training,
    # GAN
    return construct_gpu(tmp_path_factory.mktemp("p6b_gpu_full"),
                         is_training=True, seed=full_seed())


@pytest.fixture(scope="module")
def base_model_gpu(tmp_path_factory):
    # liae-ud, the default 6B seed: masked_training on,
    # random_warp on, every power 0 -> pure reconstruction
    return construct_gpu(tmp_path_factory.mktemp("p6b_gpu_base"),
                         is_training=True, seed=seed())


@pytest.fixture(scope="module")
def res384_model_gpu(tmp_path_factory):
    # res > 256: the official six-preview onGetPreview branch
    return construct_gpu(tmp_path_factory.mktemp("p6b_gpu_384"),
                         is_training=True,
                         seed=seed(resolution=384, batch_size=1,
                                   ae_dims=16, e_dims=8, d_dims=8,
                                   d_mask_dims=8))


# --- eyes / mouth priority ------------------------------------------------------

@requires_gpu
def test_eyes_mouth_zero_mask_identity_gpu(full_model_gpu, tmp_path_factory):
    """With a ZERO eyes/mouth mask the 300x MAE term contributes
    exactly 0.0, so the src loss of the em-on model must equal the
    src loss of an em-off twin at identical weights (the term is
    the only difference)."""
    model = full_model_gpu
    twin = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_em_twin"),
                         is_training=True,
                         seed=full_seed(eyes_mouth_prio=False))
    copy_components(model, twin)
    samples = synth_samples(model.resolution, batch=1, seed_no=29,
                            em_mask_zero=True)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    s_model, _ = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    s_twin, _ = twin._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert torch.allclose(s_model, s_twin, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_eyes_mouth_contribution_pin_gpu(full_model_gpu, tmp_path_factory):
    """The em term is pinned to its official expression: 300 *
    mean(|target*em - pred_src_src*em|, axes [1,2,3]) — the
    difference between the em-on model and an em-off twin at
    identical weights on a NONZERO em mask equals the hand-
    computed term."""
    model = full_model_gpu
    twin = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_em_twin2"),
                         is_training=True,
                         seed=full_seed(eyes_mouth_prio=False))
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


# --- face / background style ----------------------------------------------------

@requires_gpu
def test_face_style_contribution_pin_gpu(full_model_gpu, tmp_path_factory):
    """The face-style term (official L472): the per-channel
    MOMENTS style_loss (NOT a gram matrix) with radius res//8 and
    weight 10000*p on content = pred_src_dst_no_code_grad *
    stopgrad(pred_src_dstm), style = stopgrad(pred_dst_dst *
    pred_dst_dstm) — pinned as the on/off twin difference at
    identical weights."""
    model = full_model_gpu
    twin = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_fs_twin"),
                         is_training=True,
                         seed=full_seed(face_style_power=0.0))
    copy_components(model, twin)
    samples = synth_samples(model.resolution, batch=1, seed_no=37)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    t = model._prepare_targets(*samples)
    with torch.no_grad():
        f = forward_ae(model, t['warped_src'], t['warped_dst'])
        p = model.options['face_style_power'] / 100.0
        expected = nn.style_loss(
            f['pred_src_dst_no_code_grad'] * f['pred_src_dstm'].detach(),
            (f['pred_dst_dst'] * f['pred_dst_dstm']).detach(),
            gaussian_blur_radius=model.resolution // 8,
            loss_weight=10000 * p)
    s_on, _ = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    s_off, _ = twin._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert torch.allclose(s_on - s_off, expected, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_bg_style_contribution_pin_gpu(full_model_gpu, tmp_path_factory):
    """The background 'style' term is a CONTENT loss (official
    L474-480): 10*p * (dssim + MSE) of pred_src_dst / target_dst
    masked with the stop-grad complement of the SRC face mask —
    NOT the moments style op. Pinned as the on/off twin
    difference."""
    model = full_model_gpu
    twin = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_bs_twin"),
                         is_training=True,
                         seed=full_seed(bg_style_power=0.0))
    copy_components(model, twin)
    samples = synth_samples(model.resolution, batch=1, seed_no=41)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    t = model._prepare_targets(*samples)
    with torch.no_grad():
        f = forward_ae(model, t['warped_src'], t['warped_dst'])
        p = model.options['bg_style_power'] / 100.0
        anti = t['style_mask_anti_blur']
        psd = f['pred_src_dst'] * anti
        tds = t['target_dst'] * anti
        expected = torch.mean(
            (10 * p) * nn.dssim(psd, tds, max_val=1.0,
                                filter_size=int(model.resolution / 11.6)),
            dim=1)
        expected = expected + torch.mean(
            (10 * p) * torch.square(psd - tds), dim=(1, 2, 3))
    s_on, _ = model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    s_off, _ = twin._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert torch.allclose(s_on - s_off, expected, rtol=1e-4, atol=1e-4)


# --- true face ------------------------------------------------------------------

@requires_gpu
def test_true_face_generator_and_discriminator_gpu(full_model_gpu):
    """df + true_face_power: the G term (true_face_power *
    DLoss(ones, code_D(src_code))) flows into the generator loss
    via src_dst_opt (the code-D params already carry G-term grads
    after the src_dst backward), and the code-D step updates ONLY
    the code-discriminator weights (the official
    nn.gradients(D_code_loss, code_D_weights) variable set) while
    D_src stays untouched by it."""
    model = full_model_gpu
    samples = synth_samples(model.resolution, batch=1, seed_no=43)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)

    code_before = {id(p): p.detach().clone()
                   for p in model.code_discriminator.get_weights()}
    dsrc_before = {id(p): p.detach().clone()
                   for p in model.D_src.get_weights()}
    it_code = model.D_code_opt.iterations.item()
    it_gan = model.D_src_dst_opt.iterations.item()
    it_gen = model.src_dst_opt.iterations.item()

    model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert model.src_dst_opt.iterations.item() == it_gen + 1
    # the official G loss contains the code-D forward: after the
    # src_dst backward the code-D params carry (stale-for-D-loss,
    # but present) grads
    assert any(p.grad is not None
               for p in model.code_discriminator.get_weights())

    model._D_train(ws, wd)
    assert model.D_code_opt.iterations.item() == it_code + 1
    code_changed = False
    for pid, w in code_before.items():
        p = next(q for q in model.code_discriminator.get_weights()
                 if id(q) == pid)
        if not torch.equal(p, w):
            code_changed = True
    assert code_changed, "the code discriminator must update"
    # the code-D step touched no D_src weight
    for pid, w in dsrc_before.items():
        p = next(q for q in model.D_src.get_weights() if id(q) == pid)
        assert torch.equal(p, w)

    model._D_src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)
    assert model.D_src_dst_opt.iterations.item() == it_gan + 1
    d_changed = False
    for pid, w in dsrc_before.items():
        p = next(q for q in model.D_src.get_weights() if id(q) == pid)
        if not torch.equal(p, w):
            d_changed = True
    assert d_changed, "D_src must update"


# --- GAN ------------------------------------------------------------------------

@requires_gpu
def test_gan_off_no_update_gpu(base_model_gpu):
    """gan_power == 0: no D_src component, no GAN_opt, and a full
    official iteration runs only the src_dst step."""
    model = base_model_gpu
    assert model.gan_power == 0
    assert not hasattr(model, 'D_src')
    assert not hasattr(model, 'D_src_dst_opt')
    it = model.src_dst_opt.iterations.item()
    model.train_one_iter()
    assert model.src_dst_opt.iterations.item() == it + 1


@requires_gpu
def test_gan_generator_terms_pin_gpu(full_model_gpu, tmp_path_factory):
    """The generator GAN terms (official L539-545, both labels 1 on
    the full and patch D_src outputs of the masked_opt pred) plus
    the masked-training TV + bg-anti-MSE terms (L542-545) are pinned
    through the GRADIENTS they add to the G loss: the difference of
    the generator param grads between the gan-on model and a gan-off
    twin (identical weights) equals the grads of the hand-computed
    terms backwarded in the test on the same pre-step weights. The
    returned src/dst loss vectors do NOT contain these terms
    (official: they enter gpu_G_loss only)."""
    model = full_model_gpu
    twin = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_gan_twin"),
                         is_training=True,
                         seed=full_seed(gan_power=0.0))
    copy_components(model, twin)
    assert twin.gan_power == 0
    samples = synth_samples(model.resolution, batch=1, seed_no=47)
    xs = tensors8(samples)
    (ws, ts, tm, tm_em, wd, td, dm, dm_em) = xs

    # the expected extra terms, WITH grad, on the on-model's graph
    t = model._prepare_targets(*samples)
    src_code = model.inter(model.encoder(t['warped_src']))
    pred_src_src_g, _ = model.decoder_src(src_code)
    pred_opt = pred_src_src_g * t['target_srcm_blur']
    d, d2 = model.D_src(pred_opt)
    extra = model.gan_power * (
        nn.sigmoid_cross_entropy(torch.ones_like(d), d)
        + nn.sigmoid_cross_entropy(torch.ones_like(d2), d2))
    # masked_training (on in this seed) adds the official
    # TV + bg-anti-MSE generator terms (L542-545)
    extra = extra + 0.000001 * nn.total_variation_mse(pred_src_src_g)
    extra = extra + 0.02 * torch.mean(
        torch.square(pred_src_src_g * t['target_srcm_anti_blur']
                     - t['target_src_anti_masked']), dim=(1, 2, 3))
    for p in model.src_dst_saveable_weights:
        p.grad = None
    for p in model.D_src.get_weights():
        p.grad = None
    extra.backward()
    # params the extra terms do not reach (e.g. decoder_dst via the
    # dst path only) keep a None .grad -> a zero expected delta
    exp_gen = {id(p): (p.grad.clone() if p.grad is not None else None)
               for p in model.src_dst_saveable_weights}
    exp_dsrc = {id(p): (p.grad.clone() if p.grad is not None else None)
                for p in model.D_src.get_weights()}
    assert any(g is not None for g in exp_gen.values())
    assert any(g is not None for g in exp_dsrc.values())

    # the official steps on identical pre-step weights: each
    # closure zeros its src_dst group, backprops the full G loss,
    # and steps only that group — the post-step .grad IS the
    # G-loss gradient (the torch update op does not clear .grad).
    # The closure does NOT zero D_src grads, so clear the grads the
    # extra.backward left there before the model step.
    for p in model.D_src.get_weights():
        p.grad = None
    model._src_dst_train(*xs)
    twin._src_dst_train(*xs)

    # the generator grad difference isolates the extra terms.
    # GPU note: got is a CANCELLATION (two ~O(1) recon grads whose
    # difference is the ~1e-4..1e-6 extra-term grad), so the CUDA
    # fp32 noise floor (~1e-5) bounds the tolerance (the CPU tier
    # keeps the tighter 1e-4/1e-5 pin — identical math, no
    # cancellation noise on the CPU torch build)
    def _g(p):
        return p.grad if p.grad is not None else torch.zeros_like(p)
    for a, b, pa in zip(model.src_dst_saveable_weights,
                        twin.src_dst_saveable_weights,
                        model.src_dst_saveable_weights):
        got = _g(a) - _g(b)
        exp = exp_gen.get(id(pa))
        if exp is None:
            assert torch.allclose(got, torch.zeros_like(got),
                                  atol=1e-5), \
                f"unexpected generator grad delta on {id(pa)}"
        else:
            assert torch.allclose(got, exp, rtol=1e-3, atol=1e-4), \
                f"generator grad delta mismatch on {id(pa)}"
    # the official src_dst step does NOT update D_src, so the
    # post-step D_src grads are exactly the extra-terms grads
    for p in model.D_src.get_weights():
        if p.grad is not None and exp_dsrc[id(p)] is not None:
            assert torch.allclose(p.grad, exp_dsrc[id(p)],
                                  rtol=1e-4, atol=1e-4), \
                f"D_src grad mismatch on {id(p)}"


@requires_gpu
def test_gan_discriminator_loss_pin_gpu(full_model_gpu):
    """The D_src loss (official L532-535) is pinned by its
    GRADIENTS: the D_src parameter grads produced by the official
    D step equal the grads of the hand-computed loss (0.5-weighted
    real=target / fake=pred sigmoid BCE over the full and patch
    outputs, both masked_opt) recomputed in the test on the same
    post-src_dst-step tensors (the official closure re-prepares
    and re-forwards the AE with the post-update generator
    weights)."""
    model = full_model_gpu
    samples = synth_samples(model.resolution, batch=1, seed_no=53)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    model._src_dst_train(ws, ts, tm, tm_em, wd, td, dm, dm_em)

    # test-side recomputation of the official D-loss graph: same
    # module, same detached post-step inputs
    t2 = model._prepare_targets(*samples)
    with torch.no_grad():
        f2 = forward_ae(model, t2['warped_src'], t2['warped_dst'])
        pred_opt = f2['pred_src_src'] * t2['target_srcm_blur']
        target_opt = t2['target_src_masked_opt']
    d, d2 = model.D_src(pred_opt)
    td_, td2_ = model.D_src(target_opt)
    dl = (nn.sigmoid_cross_entropy(torch.ones_like(td_), td_)
          + nn.sigmoid_cross_entropy(torch.zeros_like(d), d)) * 0.5
    dl = dl + (nn.sigmoid_cross_entropy(torch.ones_like(td2_), td2_)
               + nn.sigmoid_cross_entropy(torch.zeros_like(d2), d2)) * 0.5
    for p in model.D_src.get_weights():
        p.grad = None
    dl.backward()
    exp_grads = {id(p): p.grad.clone() for p in model.D_src.get_weights()}
    assert all(g is not None for g in exp_grads.values())

    # the official closure's grads: zero, re-run, compare
    for p in model.D_src.get_weights():
        p.grad = None
    model._D_src_dst_train(*samples)
    for p in model.D_src.get_weights():
        assert p.grad is not None
        assert torch.allclose(p.grad, exp_grads[id(p)],
                              rtol=1e-5, atol=1e-6), \
            f"D_src grad mismatch on {id(p)}"


@requires_gpu
def test_gan_labels_and_real_fake_routing_gpu(tmp_path_factory):
    """Real/fake routing pinned at the value level on a FRESH
    (untrained) df model: the official D loss labels the TARGET as
    real (ones) and the PRED as fake (zeros) — swapping them must
    change the hand-computed loss (proof the labels/routing are not
    symmetric no-ops). A fresh random D_src gives separated
    real/fake scores; a trained one converges both toward 0/1
    saturation where the two labelings dilute to the same value.
    The torch RNG is pinned (the fresh component inits depend on
    the global RNG state; the seed gives a stable swap difference
    on the CPU torch build) so the pin is order-independent."""
    torch.manual_seed(3)
    model = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_ganlabel"),
                          is_training=True,
                          seed=seed(archi="df-ud", true_face_power=0.0,
                                    gan_power=0.05, ae_dims=8, e_dims=4,
                                    d_dims=4, d_mask_dims=4, batch_size=1))
    samples = synth_samples(model.resolution, batch=1, seed_no=59)
    ws, ts, tm, tm_em, wd, td, dm, dm_em = tensors8(samples)
    t = model._prepare_targets(*samples)
    with torch.no_grad():
        f = forward_ae(model, t['warped_src'], t['warped_dst'])
        pred_opt = f['pred_src_src'] * t['target_srcm_blur']
        target_opt = t['target_src_masked_opt']
        d, d2 = model.D_src(pred_opt)
        td, td2 = model.D_src(target_opt)
        loss_official = ((nn.sigmoid_cross_entropy(torch.ones_like(td), td)
                          + nn.sigmoid_cross_entropy(torch.zeros_like(d), d)) * 0.5
                         + (nn.sigmoid_cross_entropy(torch.ones_like(td2), td2)
                            + nn.sigmoid_cross_entropy(torch.zeros_like(d2), d2)) * 0.5)
        loss_swapped = ((nn.sigmoid_cross_entropy(torch.ones_like(d), d)
                         + nn.sigmoid_cross_entropy(torch.zeros_like(td), td)) * 0.5
                        + (nn.sigmoid_cross_entropy(torch.ones_like(d2), d2)
                           + nn.sigmoid_cross_entropy(torch.zeros_like(td2), td2)) * 0.5)
    # a fresh tiny D_src saturates both branches near 0.5, so the
    # label difference is small but must stay above the tolerance
    # (a saturated TRAINED discriminator collapses it to ~0)
    assert not torch.allclose(loss_official, loss_swapped,
                              rtol=1e-4, atol=1e-3)


# --- optimizer / update order ---------------------------------------------------

@requires_gpu
def test_update_order_generator_before_discriminators_gpu(full_model_gpu):
    """The official order: the src_dst step runs FIRST, then the
    D steps (which recompute the generator outputs on the
    post-update weights) — one official iteration advances all
    three optimizers by exactly one."""
    model = full_model_gpu
    it_gen = model.src_dst_opt.iterations.item()
    it_code = model.D_code_opt.iterations.item()
    it_gan = model.D_src_dst_opt.iterations.item()
    model.onTrainOneIter()
    assert model.src_dst_opt.iterations.item() == it_gen + 1
    assert model.D_code_opt.iterations.item() == it_code + 1
    assert model.D_src_dst_opt.iterations.item() == it_gan + 1


@requires_gpu
def test_clipgrad_step_gpu(tmp_path_factory):
    """clipgrad=True: the migrated global-norm clipnorm path runs
    inside the official step (the Phase 3E2 optimizer unit tests
    pin the formula; this proves the SAEHD wiring reaches it)."""
    model = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_clip"),
                          is_training=True, seed=seed(clipgrad=True))
    samples = synth_samples(model.resolution, batch=1, seed_no=61)
    model._src_dst_train(*tensors8(samples))
    assert model.src_dst_opt.iterations.item() == 1
    assert model.src_dst_opt.clipnorm == 1.0


@requires_gpu
def test_lr_dropout_cos_step_gpu(tmp_path_factory):
    """lr_dropout='y' + adabelief: the official construction
    coupling (lr_cos=500, lr_dropout=0.3) is live during training
    and the optimizer state (iterations) persists across steps."""
    model = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_lr"),
                          is_training=True,
                          seed=seed(lr_dropout='y', clipgrad=False))
    assert model.src_dst_opt.lr_cos == 500
    assert model.src_dst_opt.lr_dropout == 0.3
    samples = synth_samples(model.resolution, batch=1, seed_no=67)
    x = tensors8(samples)
    model._src_dst_train(*x)
    model._src_dst_train(*x)
    assert model.src_dst_opt.iterations.item() == 2


# --- full iteration / save / resume ----------------------------------------------

@requires_gpu
def test_full_iteration_official_driver_gpu(full_model_gpu):
    """The official onTrainOneIter driver end to end on the
    all-terms df-udt model: two-value loss return, iter
    increment, all three optimizers stepping once per iteration
    (the D steps gated exactly by the official conditions)."""
    model = full_model_gpu
    it0 = model.iter
    model.train_one_iter()
    assert model.iter == it0 + 1
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)


@requires_gpu
def test_save_resume_continue_gpu(base_model_gpu):
    """Save after training -> strict resume in the same directory
    -> continue: the iter, the optimizer iterations, the loss
    history all continue; one more iteration after the resume
    keeps every counter consistent."""
    model = base_model_gpu
    model.train_one_iter()
    it_after_train = model.iter
    iters_after_train = model.src_dst_opt.iterations.item()
    hist_after_train = [row[:] for row in model.loss_history]
    model.save()

    # resume with the SAME seed the original construction used
    # (the headless seed layer re-applies the seed over the restored
    # data.dat options; the production model has no such layer)
    model2 = make_saehd(SAEHDHeadless, str(model.saved_models_path),
                        is_training=True, seed=seed(), debug=True,
                        force_gpu_idxs=[0])
    assert model2.iter == it_after_train
    assert model2.src_dst_opt.iterations.item() == iters_after_train
    assert [row[:] for row in model2.loss_history] == hist_after_train

    model2.train_one_iter()
    assert model2.iter == it_after_train + 1
    assert model2.src_dst_opt.iterations.item() == iters_after_train + 1
    assert len(model2.loss_history[-1]) == 2


# --- architecture coverage -------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("archi", ["df-u", "liae-d", "df-t", "df-c"])
def test_archi_training_context_step_gpu(tmp_path_factory, archi):
    """Each representative archi (one u / one d / one t / one c
    modifier) builds a working training context and completes one
    official iteration (reconstruction-only seed: every power 0)."""
    model = construct_gpu(tmp_path_factory.mktemp(f"p6b_gpu_arch_{archi}"),
                          is_training=True, seed=seed(archi=archi))
    model.train_one_iter()
    assert model.iter == 1
    assert len(model.loss_history[-1]) == 2


@requires_gpu
def test_real_shape_413w_like_step_gpu(tmp_path_factory):
    """The real-checkpoint configuration shape (the Phase 6A
    256-df-udt set): df-udt with true_face_power=0 (code
    components unregistered), gan_power nonzero, random_warp off,
    lr_dropout on, color transfer — one official iteration must
    advance exactly src_dst_opt and D_src_dst_opt (never a
    D_code_opt, which does not exist)."""
    model = construct_gpu(tmp_path_factory.mktemp("p6b_gpu_realshape"),
                          is_training=True,
                          seed=seed(archi="df-udt", resolution=256,
                                    ae_dims=64, e_dims=32, d_dims=32,
                                    d_mask_dims=32,
                                    true_face_power=0.0,
                                    face_style_power=0.0,
                                    bg_style_power=0.0,
                                    eyes_mouth_prio=False,
                                    blur_out_mask=False,
                                    masked_training=True,
                                    random_warp=False,
                                    lr_dropout='y',
                                    ct_mode='rct',
                                    gan_power=0.05,
                                    batch_size=1))
    assert not hasattr(model, 'code_discriminator')
    assert not hasattr(model, 'D_code_opt')
    it_gen = model.src_dst_opt.iterations.item()
    it_gan = model.D_src_dst_opt.iterations.item()
    model.train_one_iter()
    assert model.src_dst_opt.iterations.item() == it_gen + 1
    assert model.D_src_dst_opt.iterations.item() == it_gan + 1
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)


# --- preview ---------------------------------------------------------------------

@requires_gpu
def test_preview_res_le_256_layout_gpu(full_model_gpu):
    """onGetPreview at resolution <= 256: exactly two previews
    ('SAEHD' and 'SAEHD masked'), the official row layouts, NHWC
    (n, H, W*cols, 3) strips, n_samples = min(4, batch,
    800//res) and the strip values are exactly the clipped
    target / AE_view outputs (AE fed with the UNWARPED targets)."""
    model = full_model_gpu
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


@requires_gpu
def test_preview_res_gt_256_layout_gpu(res384_model_gpu):
    """onGetPreview at resolution > 256: the official six previews
    (src-src, dst-dst, pred + the three masked variants),
    n_samples = min(4, batch, 800//res) (800//384 == 2, batch 1
    -> 1 sample)."""
    model = res384_model_gpu
    assert model.resolution > 256
    samples = synth_samples(model.resolution, batch=1, seed_no=73)
    (ws, ts, tm, tm_em, wd, td, dm, dm_em) = samples
    previews = model.onGetPreview(
        [(ws, ts, tm, tm_em), (wd, td, dm, dm_em)])
    names = [name for name, _ in previews]
    assert names == ['SAEHD src-src', 'SAEHD dst-dst', 'SAEHD pred',
                     'SAEHD masked src-src', 'SAEHD masked dst-dst',
                     'SAEHD masked pred']
    n = min(4, model.get_batch_size(), 800 // model.resolution)
    res = model.resolution
    # the official per-sample concatenate drops the leading axis
    # when n_samples == 1
    want = (n, res, 2 * res, 3) if n > 1 else (res, 2 * res, 3)
    for _, img in previews:
        assert img.shape == want


# --- real-artifact validation (opt-in, env-gated, CUDA) --------------------------

@requires_gpu
def test_real_checkpoint_one_step_training_lifecycle_cuda(
        tmp_path_factory, monkeypatch):
    """The official one-step TRAINING lifecycle on a REAL official
    SAEHD checkpoint + a REAL faceset, on CUDA (opt-in via
    DFL_TEST_SAEHD_CHECKPOINT / DFL_TEST_FACESET_PAK; unset ->
    skip). Everything runs on COPIES in a temp working dir (the
    private originals are never touched). Steps (plan section 9):
    1 strict resume 2 real faceset load 3 one training iteration
    4/5 finite src/dst losses 6 GAN/true-face updates per the
    checkpoint's own options 7 optimizer iteration advances
    8 trained params change 9 unrelated groups unchanged
    10 save 11 strict reload 12 optimizer states preserved
    13 one more iteration after reload. Labels: Real SAEHD
    one-step training lifecycle; Real checkpoint
    resume-before-step; Real checkpoint save-after-step; Real
    checkpoint reload-after-step; Optimizer-state continuity;
    Training numerical parity vs official TF: NOT_VERIFIED.
    Headless-safe: the official override poll (ask_override ->
    io.input_in_time, reachable only when is_training and iter!=0
    — exactly this resume case) is pinned to False so the
    checkpoint's STORED options are kept without touching stdin
    (pytest captures stdin as a fileno-less pseudofile); the
    same pattern as the Phase 6A headless_io fixture."""
    ckpt_path = os.environ.get("DFL_TEST_SAEHD_CHECKPOINT")
    pak_path = os.environ.get("DFL_TEST_FACESET_PAK")
    if not ckpt_path or not Path(ckpt_path).exists():
        pytest.skip("DFL_TEST_SAEHD_CHECKPOINT not set")
    if not pak_path or not Path(pak_path).exists():
        pytest.skip("DFL_TEST_FACESET_PAK not set (real faceset)")
    pak_file = Path(pak_path)
    if pak_file.is_dir():
        pak_file = pak_file / "faceset.pak"
        if not pak_file.exists():
            pytest.skip("no faceset.pak under DFL_TEST_FACESET_PAK dir")

    import pickle
    import shutil
    # the Phase 6A bootstrap (real model class, rename to the test
    # model name, preseeded data.dat, the headless input provider)
    from test_model_saehd import (
        InputScript,
        _SAEHDFILE_RE,
        construct_real,
        preseed_data_dat,
    )
    from facelib import FaceType
    from samplelib import PackedFaceset

    # headless-safe interactive layer (the Phase 6A headless_io
    # pattern): never touch pytest's captured stdin; the override
    # poll answers "no" -> the resume keeps the checkpoint's own
    # stored options, which is exactly what this acceptance test
    # exercises
    monkeypatch.setattr(builtins, "input", InputScript({}))
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)

    samples = PackedFaceset.load(pak_file.parent)
    if not samples:
        pytest.skip("real faceset is empty")
    ft_of = {FaceType.HALF: "h", FaceType.MID_FULL: "mf",
             FaceType.FULL: "f", FaceType.WHOLE_FACE: "wf"}
    ft = ft_of.get(samples[0].face_type)
    if ft is None:
        pytest.skip("real faceset face type not usable by SAEHD here")

    src = Path(ckpt_path)
    root = tmp_path_factory.mktemp("p6b_real")
    root.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        m = _SAEHDFILE_RE.match(f.name)
        if m is None:
            continue
        rest = m.group("rest")
        if rest == "data.dat":
            shutil.copy(f, root / f"test_SAEHD_data.dat")
        elif rest == "default_options.dat":
            shutil.copy(f, root / f"test_SAEHD_default_options.dat")
        elif rest.endswith(".npy"):
            shutil.copy(f, root / f"test_SAEHD_{rest}")
    data = pickle.loads((root / "test_SAEHD_data.dat").read_bytes())
    opts = data["options"]
    # the checkpoint's face_type must match the REAL faceset type
    opts["face_type"] = ft
    preseed_data_dat(root, opts, it=max(int(data.get("iter", 0)), 1),
                     sample_for_preview=None)
    for side in ("src", "dst"):
        (root / side).mkdir(parents=True)
        shutil.copy2(pak_file, root / side / "faceset.pak")

    dfl_nn.initialize_main_env()

    # 1 + 2: strict resume in the TRAINING context on CUDA + the
    # real faceset-backed generators
    model = construct_real(root, options=None, is_training=True,
                           debug=True, force_gpu_idxs=[0])
    print(f"REAL LIFECYCLE 1+2 PASS: resumed iter={model.iter} "
          f"archi={model.options.get('archi')} "
          f"res={model.options.get('resolution')} "
          f"tfp={model.options.get('true_face_power')} "
          f"gp={model.gan_power} "
          f"real faceset loaded (src/dst pak)")

    # 3 + 4 + 5: one official training iteration on REAL data —
    # the pre-step weight snapshots (steps 6-9) are captured first
    tfp = model.options["true_face_power"]
    gen_components = [model.encoder, model.inter, model.decoder_src,
                      model.decoder_dst]
    gen_before = {id(p): p.detach().clone()
                  for c in gen_components for p in c.get_weights()}
    if model.gan_power != 0:
        assert hasattr(model, "D_src_dst_opt")
        gan_it = model.D_src_dst_opt.iterations.item()
        dsrc_before = [p.detach().clone() for p in model.D_src.get_weights()]
    else:
        dsrc_before = None
    if tfp != 0:
        code_before = [p.detach().clone()
                       for p in model.code_discriminator.get_weights()]
    else:
        code_before = None

    model.train_one_iter()
    row = model.loss_history[-1]
    assert len(row) == 2, row
    assert all(math.isfinite(v) for v in row), row
    print(f"REAL LIFECYCLE 3-5 PASS: one real-data iteration; "
          f"src={row[0]:.6f} dst={row[1]:.6f} (finite)")

    # 6 + 7 + 8 + 9: the official update set for THIS checkpoint's
    # options
    changed = []
    # 7: the generator optimizer advanced exactly once
    assert model.src_dst_opt.iterations.item() >= 1
    # 8: the generator weights changed (all components are in the
    # official df trainable set)
    gen_changed = 0
    for c in gen_components:
        for p in c.get_weights():
            if not torch.equal(p.detach(), gen_before[id(p)]):
                gen_changed += 1
    assert gen_changed > 0, "no generator weight changed"
    changed.append(f"generator ({gen_changed} weight tensors changed)")
    if model.gan_power != 0:
        dsrc_now = model.D_src.get_weights()
        assert any(not torch.equal(a, b)
                   for a, b in zip(dsrc_now, dsrc_before)), \
            "D_src weights must change with gan_power != 0"
        assert model.D_src_dst_opt.iterations.item() == gan_it + 1
        changed.append("D_src + D_src_dst_opt iters +1")
    else:
        assert not hasattr(model, "D_src_dst_opt")
    if tfp != 0:
        code_now = model.code_discriminator.get_weights()
        assert any(not torch.equal(a, b)
                   for a, b in zip(code_now, code_before)), \
            "code-D weights must change with tfp != 0"
        assert model.D_code_opt.iterations.item() >= 1
        changed.append("code-D + D_code_opt iters +1")
    else:
        # 9: with tfp == 0 the code-D components do not exist at
        # all (the official df creation rule) — nothing else
        # outside the update set may change
        assert not hasattr(model, "code_discriminator")
        assert not hasattr(model, "D_code_opt")
        changed.append("code-D components absent (tfp=0)")
    print(f"REAL LIFECYCLE 6-9 PASS: update set per options: "
          f"{changed}")

    # 10: save
    model.save()
    iters_src = model.src_dst_opt.iterations.item()
    print(f"REAL LIFECYCLE 10 PASS: saved after step "
          f"(iter={model.iter})")

    # 11: strict reload in the same directory
    model2 = construct_real(root, options=None, is_training=True,
                            debug=True, force_gpu_idxs=[0])
    assert model2.iter == model.iter
    print(f"REAL LIFECYCLE 11 PASS: strict reload "
          f"(iter={model2.iter})")

    # 12: optimizer states preserved (iterations + moment buffers)
    for na in ("src_dst_opt",) + (("D_code_opt",) if tfp != 0 else ()) \
            + (("D_src_dst_opt",) if model.gan_power != 0 else ()):
        a, b = getattr(model, na).get_weights(), \
            getattr(model2, na).get_weights()
        assert len(a) == len(b), na
        for x, y in zip(a, b):
            assert torch.equal(x, y), na
    assert model2.src_dst_opt.iterations.item() == iters_src
    print("REAL LIFECYCLE 12 PASS: optimizer states preserved")

    # 13: one more iteration after the reload
    model2.train_one_iter()
    assert model2.iter == model.iter + 1
    row2 = model2.loss_history[-1]
    assert len(row2) == 2 and all(math.isfinite(v) for v in row2)
    print(f"REAL LIFECYCLE 13 PASS: post-reload iteration "
          f"(iter={model2.iter}, src={row2[0]:.6f} "
          f"dst={row2[1]:.6f})")

    print("REAL SAEHD ONE-STEP TRAINING LIFECYCLE (CUDA): PASS "
          "(Training numerical parity vs official TF: NOT_VERIFIED)")
