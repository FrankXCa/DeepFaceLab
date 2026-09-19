"""Phase 7: AMP single-device training semantics coverage —
the CPU tier.

Testing pyramid (docs/PHASE7_PLAN.md sections 16/17):

    formula-level CPU unit tests (the archi tier:
        test_archi_amp.py — official key inventories, the exact-k
        morph-mask property, save/load round trip, fp16
        construction, CPU/GPU parity)
    -> tiny-model CPU integration (THIS file: the official loss
        stack, train closures, update routing, morph semantics,
        preview, options, checkpoint lifecycle)
    -> tiny CUDA incremental-peak + real-artifact acceptance
        (test_model_amp_training_cuda.py)

This file is the CPU tier: it proves correctness and backend
neutrality and stays FAST. The tiny integration models (64px,
batch 1, the smallest practical dims) drive the REAL production
``models.Model_AMP.AMPModel`` (through the test-only
``AMPHeadless`` option layer) on CPU: the complete official loss
stack (the five-term src/dst stacks, the background weak loss,
the official GAN terms, the official 8-term D loss x 1/8), the
official two-optimizer construction and per-optimizer gradient
ownership (src_dst_opt over G_weights = encoder+decoder only;
GAN_opt over the FULL GAN state — never prefix-filtered), the
Q11 batch-SUM backward seed (the N=2-identical-samples gradient
doubling pin), the official two-phase GAN train (post-step D),
the official exact-k training morph mask + deterministic
floor-slice inference morph, the official preview layout
(3 named strips, one-sample rendering), the official options
layer (first-run clipping + the first-run default_options.dat
snapshot STAYS enabled — AMP never calls
disable_default_options_autosave, the F-A4 note), the official
model_filename_list, save -> strict resume -> continue, the
official gan_model_changed GAN+GAN_opt re-init rule and the
missing-file hard error, the export_dfm deferral stub, the
predictor/merger inference paths.

Numerical labels (mandated): every formula pin here is
FORMULA_VERIFIED; TF runtime parity is TF_RUNTIME_NOT_VERIFIED
(no official TF stack exists on disk — the official Model_tf.py
is a dead reference); no test claims runtime parity with the
official TF graph.
"""

import builtins
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.interact import interact as io
from core.leras import nn

from models.Model_AMP import Model as AMPModel

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    AMPModelGanChanged,
    make_model as make_amp,
    make_training_dirs,
)

# the tiny CPU integration configuration: the smallest practical
# AMP (64px, batch 1 — the official debug=True sample generator
# forces N=1; the closures themselves are driven with synthetic
# N > 1 batches directly)
TINY = dict(resolution=64, batch_size=1, ae_dims=32, inter_dims=32,
            e_dims=16, d_dims=16, d_mask_dims=6)


# --- helpers -------------------------------------------------------------------

def seed(**overrides):
    from Model_AMPTest.Model import DEFAULT_SEED_OPTIONS
    s = dict(DEFAULT_SEED_OPTIONS)
    s.update(overrides)
    return s


def construct(tmpdir, is_training=False, seed_options=None, cpu_only=True):
    """Training-context construction (debug=True in-process
    generator); ``is_training=False`` skips the facesets (the
    train closures are not built in that mode)."""
    if is_training:
        make_training_dirs(Path(tmpdir))
    return make_amp(AMPHeadless, tmpdir, is_training=is_training,
                    seed=seed_options, debug=is_training, cpu_only=cpu_only)


def synth_samples_nchw(res, batch=1, seed_no=0,
                       src_level=0.4, dst_level=0.6, mask_frac=0.55,
                       em_frac=0.3):
    """Deterministic synthetic training batch in the AMP model
    data format (the official hard-wired NCHW): the 8 arrays in
    the official onTrainOneIter unpack order (warped_src,
    target_src, target_srcm, target_srcm_em, warped_dst,
    target_dst, target_dstm, target_dstm_em). Images are smooth
    BGR fields (different per side); the full-face mask is a
    centered disc, the eyes/mouth mask a smaller disc."""
    rng = np.random.RandomState(seed_no)

    def img(level):
        base = np.clip(rng.rand(3, res, res) * 0.3 + level, 0.0, 1.0)
        y, x = np.mgrid[0:res, 0:res].astype(np.float32)
        base = base + 0.1 * np.sin(x / 23.0)[None, :, :]
        base = np.clip(base, 0.0, 1.0)
        return np.repeat(base[None, ...], batch, 0).astype(np.float32)

    def disc(frac):
        y, x = np.mgrid[0:res, 0:res].astype(np.float32)
        r = np.sqrt((y - res / 2) ** 2 + (x - res / 2) ** 2) / (res / 2)
        m = np.where(r < frac, 1.0, 0.0).astype(np.float32)
        return np.repeat(m[None, None], batch, 0).astype(np.float32)

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
    return arrays


def grad_map(model, params):
    """param -> detached grad tensor (a copy) for a weight list."""
    out = {}
    for p in params:
        out[id(p)] = None if p.grad is None else p.grad.detach().clone()
    return out


# --- official options layer (the REAL on_initialize_options) -------------------

def test_options_first_run_clipping_and_snapshot(tmp_path, monkeypatch):
    """The real (non-headless) first-run option layer: the
    official clipping rules (resolution a multiple of 32 in
    64-640, ae 32-1024, inter 32-2048, e/d/d_mask 16-256 with
    the even rounding, morph 0.1-0.5, the d_mask_dims derived
    default), the official prompt flow, the
    ``gan_model_changed`` detection (False when stored ==
    prompted) and the Q4 pin: the official first-run
    ``default_options.dat`` snapshot STAYS ENABLED (AMP never
    calls disable_default_options_autosave — F-A4)."""
    overrides = {
        'Resolution': 200,                    # -> 192 (32 * 6)
        'Face type': 'HEAD',                  # -> 'head' (.lower())
        'AutoEncoder dimensions': 5000,       # -> 1024
        'Inter dimensions': 5,                # -> 32
        'Encoder dimensions': 100,            # -> 100 (even)
        'Decoder dimensions': 100,            # -> 100 (even)
        'Decoder mask dimensions': 33,        # -> 34 (even rounding)
        'Morph factor.': 2.0,                 # -> 0.5
        'Use learning rate dropout': 'cpu',
        'Uniform yaw distribution of samples': True,
        'Blur out mask': True,
        'GAN power': 0.1,                     # -> the conditional gan prompts
                                             # (no size overrides: the
                                             # prompts answer with the
                                             # code defaults 192//8 = 24
                                             # and 16, so the official
                                             # L101 stored==prompted
                                             # comparison stays False)
        'Place models and optimizer on GPU': False,
        'Enable random warp of samples': False,
        'Color transfer for src faceset': 'none',
        'Enable gradient clipping': True,
    }

    def fake(label, default, *args, **kwargs):
        return overrides.get(label, default)

    for method in ('input_int', 'input_str', 'input_number', 'input_bool'):
        monkeypatch.setattr(io, method, fake)

    root = Path(tmp_path) / 'model'
    root.mkdir(parents=True)
    # cpu_only=True: the real-model constructor must initialize nn
    # without the interactive device prompt (headless test context)
    model = AMPModel(
        is_training=False,
        saved_models_path=root,
        training_data_src_path=root / 'src',
        training_data_dst_path=root / 'dst',
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name='test_AMP',
        cpu_only=True,
    )

    assert model.options['resolution'] == 192
    assert model.options['face_type'] == 'head'
    assert model.options['ae_dims'] == 1024
    assert model.options['inter_dims'] == 32
    assert model.options['e_dims'] == 100
    assert model.options['d_dims'] == 100
    assert model.options['d_mask_dims'] == 34
    assert model.options['morph_factor'] == 0.5
    assert model.options['lr_dropout'] == 'cpu'
    assert model.options['gan_power'] == 0.1
    assert model.options['gan_patch_size'] == 24  # the 192//8 code default
    assert model.options['gan_dims'] == 16        # the code default
    assert model.options['models_opt_on_gpu'] is False
    assert model.options['random_warp'] is False
    assert model.options['clipgrad'] is True
    # truthiness (the official expression yields a numpy bool via
    # the `or` chain over np-clipped scalars — not Python's False)
    assert not model.gan_model_changed

    # Q4: the official first-run snapshot stays enabled (AMP never
    # calls disable_default_options_autosave). The official naming:
    # the default-options snapshot uses the module-derived
    # model_class_name ('AMP'), not the force-able model_name
    assert (root / 'AMP_default_options.dat').exists()


# --- construction / official naming --------------------------------------------

def test_tiny_training_construction_official_naming(tmp_path):
    """Tiny training construction (gan off): the official
    model_filename_list (the official .npy filenames, [model,
    file] pairs / (opt, file) tuples), the official component
    names, the official G_weights set (encoder+decoder — the
    inter heads are bound but NEVER optimized) and the official
    optimizer-state naming (iters + ms_/vs_ per official
    variable name)."""
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))

    names = [filename for _, filename in model.model_filename_list]
    assert names == ['encoder.npy', 'inter_src.npy', 'inter_dst.npy',
                     'decoder.npy', 'src_dst_opt.npy']
    assert not hasattr(model, 'GAN')
    assert not hasattr(model, 'GAN_opt')

    # G_weights = encoder + decoder only (official L301)
    g = model.G_weights
    enc = set(id(p) for p in model.encoder.get_weights())
    dec = set(id(p) for p in model.decoder.get_weights())
    assert set(id(p) for p in g) == enc | dec
    inter = set(id(p) for p in (model.inter_src.get_weights()
                                + model.inter_dst.get_weights()))
    assert not (set(id(p) for p in g) & inter)

    # every optimized parameter is bound to its official DFL name,
    # and the state key set is exactly iters + ms_/vs_ per name
    g_names = [p._dfl_name for p in g]
    assert all(n.startswith(('encoder/', 'decoder/')) for n in g_names)
    state_names = set(model.src_dst_opt._state_official_names)
    expected = set()
    for n in g_names:
        # the official state sub-name rule (optimizers_tf.py L180,
        # ported at OptimizerBase._state_sub_name):
        # f'{prefix}_{name}'.replace(':', '_') + ':0'
        expected.add(("ms_" + n).replace(":", "_") + ":0")
        expected.add(("vs_" + n).replace(":", "_") + ":0")
    assert state_names == expected
    # the inter heads are bound (official sub-names) ...
    for p in model.inter_src.get_weights():
        assert p._dfl_name.startswith('inter_src/')
    for p in model.inter_dst.get_weights():
        assert p._dfl_name.startswith('inter_dst/')
    # ... but never appear in the optimizer state
    assert not any('inter_' in n for n in state_names)

    # the model data format is the official hard-wired NCHW
    assert model.model_data_format == 'NCHW'
    assert nn.data_format == 'NCHW'


def test_gan_construction_full_opt_state_inventory(tmp_path):
    """gan_power != 0: the official GAN (name 'GAN', file
    GAN.npy) + GAN_opt (file GAN_opt.npy) construction, and the
    D-5 pin: the GAN_opt state is the FULL per-parameter
    inventory (iters + ms_/vs_ for EVERY GAN weight — never
    filtered by a parameter-name prefix)."""
    model = construct(tmp_path, is_training=True,
                      seed_options=seed(gan_power=0.1, gan_patch_size=8,
                                        gan_dims=4, **TINY))

    names = [filename for _, filename in model.model_filename_list]
    assert names[-2:] == ['GAN.npy', 'GAN_opt.npy']
    assert model.GAN.name == 'GAN'

    gweights = model.GAN.get_weights()
    g_names = [p._dfl_name for p in gweights]
    assert all(n.startswith('GAN/') for n in g_names)
    state_names = set(model.GAN_opt._state_official_names)
    expected = set()
    for n in g_names:
        # the official state sub-name rule (optimizers_tf.py L180,
        # ported at OptimizerBase._state_sub_name):
        # f'{prefix}_{name}'.replace(':', '_') + ':0'
        expected.add(("ms_" + n).replace(":", "_") + ":0")
        expected.add(("vs_" + n).replace(":", "_") + ":0")
    assert state_names == expected
    assert len(state_names) == 2 * len(gweights)

    # the G step trains the GAN weights (they feed the G loss via
    # the generator GAN terms) but never steps them: G_weights
    # excludes them
    assert not any(p._dfl_name.startswith('GAN/') for p in model.G_weights)


# --- Q11: the official batch-SUM backward seed ---------------------------------

def test_train_ones_seed_batch_sum(tmp_path):
    """Q11 (probe-verified): the official TF
    nn.gradients(loss_vec, vars) on the per-sample (N,) loss
    vector seeds the backward with ones = the batch SUM over
    samples. Pinned by the deterministic doubling property at
    IDENTICAL weights: with morph_factor=0.0 (k=0, a mask-free
    deterministic forward) and two identical samples, the N=2
    backward produces exactly 2x the N=1 per-sample gradient on
    every G_weights parameter (a batch-MEAN implementation —
    the rejected externals' deviation — would produce the SAME
    gradient instead of the doubled one, a factor-of-2 gap no
    tolerance can hide). The update op is shadowed to a no-op
    (test-side) so both backward passes run at the identical
    weights; real steps before/after prove the full closure
    (update + counter + weight movement)."""
    model = construct(tmp_path, is_training=True,
                      seed_options=seed(morph_factor=0.0, **TINY))
    res = model.resolution

    x1 = synth_samples_nchw(res, batch=1, seed_no=1)
    x2 = synth_samples_nchw(res, batch=2, seed_no=1)

    # a real first step: the full closure (backward + update)
    w_pre = {id(p): p.detach().clone() for p in model.G_weights}
    model.train(*x1)
    assert model.src_dst_opt.iterations.item() == 1
    # the weights actually moved (the src_dst_opt step ran) —
    # against the PRE-step snapshot
    for p in model.G_weights:
        assert not torch.equal(w_pre[id(p)], p.detach()), p._dfl_name

    # the exact batch-sum pin at identical weights: shadow the
    # update op (test-side no-op) so both backward passes see
    # the same weights
    real_update_op = model.src_dst_opt.get_update_op
    model.src_dst_opt.get_update_op = lambda pairs: (lambda: None)
    try:
        model.train(*x1)
        g1 = grad_map(model, model.G_weights)
        # a non-degenerate gradient exists on the group
        assert any(v is not None and v.abs().sum() > 1e-3 for v in g1.values())
        # the frozen inter heads never receive grads (the official
        # gradient requests cover only G_weights / the GAN weights —
        # the inter weights are autograd constants, finding F4)
        assert all(p.grad is None
                   for p in model.inter_src.get_weights()
                   + model.inter_dst.get_weights())

        model.train(*x2)
        g2 = grad_map(model, model.G_weights)
        for p in model.G_weights:
            v1, v2 = g1[id(p)], g2[id(p)]
            assert v1 is not None and v2 is not None
            # the N=2 identical-samples gradient is the doubled
            # per-sample gradient (float32 matmul rounding: a tight
            # relative tolerance, still ~100x below the factor-of-2
            # gap a batch-MEAN implementation would leave)
            assert torch.allclose(v2, 2 * v1, rtol=1e-5, atol=1e-6), p._dfl_name
    finally:
        model.src_dst_opt.get_update_op = real_update_op

    # the returned loss vectors are the official per-sample (N,)
    # snapshots (the 5-term stacks, pre-GAN terms)
    s1, d1 = model.train(*x1)
    assert s1.shape == (1,) and d1.shape == (1,)
    s2, d2 = model.train(*x2)
    assert s2.shape == (2,) and d2.shape == (2,)
    assert torch.isfinite(s2).all() and torch.isfinite(d2).all()
    # identical samples -> identical per-sample losses
    assert torch.allclose(s2[0], s2[1], rtol=1e-5, atol=1e-6)
    assert model.src_dst_opt.iterations.item() == 3


def test_gan_two_phase_post_step(tmp_path):
    """The official two-phase GAN train: the G step updates
    ONLY the G_weights (the GAN weights take grads but are
    never stepped); the D step (GAN_train) recomputes the
    generator with the post-update weights (a fresh exact-k
    mask draw, gradient-free), updates ONLY the GAN weights
    through GAN_opt — the G weights are untouched by the D
    step, and the G step's stale GAN grads must not leak into
    the D step (the closure re-zeros the GAN group)."""
    model = construct(tmp_path, is_training=True,
                      seed_options=seed(gan_power=0.1, gan_patch_size=8,
                                        gan_dims=4, **TINY))
    res = model.resolution
    x = synth_samples_nchw(res, batch=1, seed_no=7)

    g_before = {id(p): p.detach().clone() for p in model.G_weights}
    d_before = {id(p): p.detach().clone() for p in model.GAN.get_weights()}

    model.train(*x)
    # G step: G updated, D untouched (only src_dst_opt stepped)
    assert model.src_dst_opt.iterations.item() == 1
    assert model.GAN_opt.iterations.item() == 0
    assert any(not torch.equal(g_before[id(p)], p.detach())
               for p in model.G_weights)
    assert all(torch.equal(d_before[id(p)], p.detach())
               for p in model.GAN.get_weights())
    # the POST-G-step G weights: the baseline for the D-step
    # "G untouched" check (the G step itself moved them)
    g_post_g = {id(p): p.detach().clone() for p in model.G_weights}

    model.GAN_train(*x)
    # D step: D updated (GAN_opt stepped), G untouched
    assert model.GAN_opt.iterations.item() == 1
    assert model.src_dst_opt.iterations.item() == 1
    assert any(not torch.equal(d_before[id(p)], p.detach())
               for p in model.GAN.get_weights())
    for p in model.G_weights:
        assert torch.equal(g_post_g[id(p)], p.detach()), p._dfl_name


# --- the official onTrainOneIter driver ----------------------------------------

def test_ontrainoneiter_driver(tmp_path):
    """The official L646-657 driver through the real lifecycle
    (the sample generators fed from the test facesets,
    debug=True -> N=1): onTrainOneIter runs the G step (and the
    D step iff gan_power != 0), advances both optimizer
    counters and returns the two per-sample-mean losses as
    (name, float) pairs (the official L657 return); the
    train_one_iter wrapper adds the lifecycle bookkeeping
    (iter increment, the loss_history row, the
    (iter, iter_time) return)."""
    model = construct(tmp_path, is_training=True,
                      seed_options=seed(gan_power=0.1, gan_patch_size=8,
                                        gan_dims=4, **TINY))
    # the direct onTrainOneIter contract: the official L657
    # return — the per-sample means as (name, float) pairs. A
    # direct call runs the driver (both optimizer counters
    # advance) but leaves the lifecycle bookkeeping (iter /
    # loss_history — the train_one_iter wrapper's job) untouched
    results = model.onTrainOneIter()
    assert [name for name, _ in results] == ['src_loss', 'dst_loss']
    src_loss, dst_loss = [v for _, v in results]
    assert isinstance(src_loss, float) and isinstance(dst_loss, float)
    assert np.isfinite(src_loss) and np.isfinite(dst_loss)
    assert model.iter == 0
    assert len(model.loss_history) == 0
    assert model.src_dst_opt.iterations.item() == 1
    assert model.GAN_opt.iterations.item() == 1

    # the wrapper: the same driver plus the lifecycle
    # bookkeeping (iter increment, the two recorded losses, the
    # (iter, iter_time) return)
    results = model.train_one_iter()
    assert results[0] == model.iter == 1
    assert len(model.loss_history) == 1
    assert len(model.loss_history[-1]) == 2
    assert all(np.isfinite(v) for v in model.loss_history[-1])
    assert model.src_dst_opt.iterations.item() == 2
    assert model.GAN_opt.iterations.item() == 2


# --- the official morph semantics ------------------------------------------------

def test_ae_view_floor_slice_morph_zero_identity(tmp_path):
    """The official DETERMINISTIC inference morph
    (L370-372): k = int(inter_dims * morph_value) LEADING
    inter channels from the inter_src head, the remainder from
    the inter_dst head. Pinned at the k=0 edge: morph_value
    0.0 -> the slice code is the pure inter_dst code, so the
    SD tile equals the DD tile bit-for-bit (the decoder is
    deterministic under no_grad); morph_value 1.0 -> the pure
    inter_src code, cross-checked against the model's own
    components."""
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))
    res = model.resolution
    ts, td = (synth_samples_nchw(res, batch=1, seed_no=3)[1],
              synth_samples_nchw(res, batch=1, seed_no=3)[5])

    # morph 0.0: SD_000 == DD bit-for-bit (k = 0)
    out = model.AE_view(ts, td, 0.0)
    SS, DD, DDM, SD, SDM = [np.asarray(x) for x in out]
    assert SS.shape == (1, 3, res, res)
    assert DD.shape == (1, 3, res, res)
    assert DDM.shape == (1, 1, res, res)
    assert SD.shape == (1, 3, res, res)
    assert SDM.shape == (1, 1, res, res)
    np.testing.assert_allclose(SD, DD, rtol=0, atol=0)
    # the DDM mask is a sigmoid output in [0, 1]
    assert np.isfinite(DDM).all() and (DDM >= 0.0).all() and (DDM <= 1.0).all()

    # morph 1.0: SD_100 = decoder(dst_inter_src code) — the
    # inter_src head on the dst code (cross-checked with the
    # model's own components)
    out = model.AE_view(ts, td, 1.0)
    SD_100, SDM_100 = [np.asarray(x) for x in out[3:5]]
    with torch.no_grad():
        nn.initialize(nn.DeviceConfig([]), 'float32', 'NCHW')
        d = torch.from_numpy(np.ascontiguousarray(td)).to(device=nn.device, dtype=nn.floatx)
        code = model.encoder(d)
        z = model.inter_src(code)
        img, m = model.decoder(z)
    np.testing.assert_allclose(SD_100, img.cpu().numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(SDM_100, m.cpu().numpy(), rtol=0, atol=0)


def test_ae_merge_nontraining_morph_edges(tmp_path):
    """The official non-training AE_merge (L519-534): the dst
    code chain + the morph-value floor slice. The official
    caller contract is batched NCHW (the official 4D
    placeholder shape); the official RETURN order is
    (bgr, DST mask, SRC mask) = (pred_src_dst, pred_dst_dstm,
    pred_src_dstm) (L532). morph 0.0 -> src_dst_code = the
    pure inter_dst code, so the merged image equals the
    decoder image head of the dst code and BOTH masks come
    from the same deterministic decoder computation
    (identical); morph 1.0 -> the pure inter_src code,
    cross-checked with the model's own components. The
    official predictor_func caller contract (L707-712): NHWC
    in, per-sample outputs — and the official return-order
    swap: 2nd return = SRC-side mask, 3rd = DST-side mask."""
    model = construct(tmp_path, is_training=False, seed_options=seed(**TINY))
    res = model.resolution
    # the official AE_merge caller contract: batched NCHW
    face = synth_samples_nchw(res, batch=1, seed_no=5)[1]

    with torch.no_grad():
        nn.initialize(nn.DeviceConfig([]), 'float32', 'NCHW')
        f = torch.from_numpy(np.ascontiguousarray(face)).to(device=nn.device, dtype=nn.floatx)
        code = model.encoder(f)
        z_dst, z_src = model.inter_dst(code), model.inter_src(code)
        img_dst, mask_dst = model.decoder(z_dst)
        img_src, _ = model.decoder(z_src)

    # morph 0.0: src_dst_code = the pure inter_dst code -> the
    # merged image is the decoder image head of the dst code and
    # BOTH masks come from the same deterministic decoder
    # computation (bit-identical)
    bgr, m_dst, m_src = model.AE_merge(face, 0.0)
    np.testing.assert_allclose(bgr, img_dst.cpu().numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(m_dst, m_src, rtol=0, atol=0)
    np.testing.assert_allclose(m_dst, mask_dst.cpu().numpy(), rtol=0, atol=0)
    assert bgr.shape == (1, 3, res, res)
    assert m_dst.shape == (1, 1, res, res)
    assert m_src.shape == (1, 1, res, res)

    # morph 1.0
    bgr1, _, _ = model.AE_merge(face, 1.0)
    np.testing.assert_allclose(bgr1, img_src.cpu().numpy(), rtol=0, atol=0)

    # the official predictor_func caller contract: NHWC face in
    # (converted internally, L708), per-sample outputs out
    face_nhwc = face[0].transpose(1, 2, 0).copy()
    p_bgr, p_msrc, p_md = model.predictor_func(face_nhwc, 0.5)
    assert p_bgr.shape == (res, res, 3)
    assert p_msrc.shape == (res, res)
    assert p_md.shape == (res, res)
    assert p_bgr.dtype == np.float32


def test_export_dfm_deferral_stub(tmp_path):
    """The official export_dfm is a TF/tf2onnx graph export —
    out of scope for Phase 7 (the ONNX/DFM exclusion): the
    torch model raises NotImplementedError, like the Phase 6B
    SAEHD port."""
    model = construct(tmp_path, is_training=False, seed_options=seed(**TINY))
    with pytest.raises(NotImplementedError, match='ONNX'):
        model.export_dfm()


# --- preview (the official L660-705 layout) --------------------------------------

def test_preview_layout_and_tiles(tmp_path):
    """The official onGetPreview: AE_view fed with the
    UNWARPED targets, the 3 named strips x 2 rows x 3 tiles
    (the DDM masks 3-channel repeated), the official
    one-sample rendering (n_samples bounds only the RANDOM
    index; for_history=True fixes it to 0), NHWC strips
    (2R, 3R, 3), values in [0, 1]."""
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))
    res = model.resolution
    samples = synth_samples_nchw(res, batch=2, seed_no=71)
    (ws, ts, tm, tm_em, wd, td, dm, dm_em) = samples
    previews = model.onGetPreview([(ws, ts, tm, tm_em), (wd, td, dm, dm_em)],
                                  for_history=True)

    assert [name for name, _ in previews] == [
        'AMP morph 1.0', 'AMP morph list', 'AMP morph list masked']
    for _, img in previews:
        assert img.shape == (2 * res, 3 * res, 3)
        assert np.isfinite(img).all()
        assert (img >= 0.0).all() and (img <= 1.0).all()

    # for_history=True -> i = 0: recompute the official strip 1
    # tiles from the AE_view outputs (the SS tile is stochastic
    # per call — pin the deterministic tiles instead)
    # the official onGetPreview conversion (L664): nn.to_data_format
    # to NHWC on the 4D (1,3,R,R) NCHW AE_view outputs — the
    # batch-0 tile
    def _tile(x):
        return nn.to_data_format(np.clip(np.asarray(x), 0.0, 1.0),
                                 'NHWC', 'NCHW')[0]

    ae0 = [_tile(x) for x in model.AE_view(ts, td, 0.0)]
    SS0, DD0, DDM0 = ae0[0], ae0[1], ae0[2]
    S0 = np.clip(ts[0].transpose(1, 2, 0).astype(np.float32), 0.0, 1.0)
    D0 = np.clip(td[0].transpose(1, 2, 0).astype(np.float32), 0.0, 1.0)
    first = previews[0][1]
    top, bottom = first[:res], first[res:]
    # row 1: [S, D, DD * DDM_000] (the DDM 3-channel repeat)
    ddm03 = np.repeat(DDM0, 3, axis=-1)
    np.testing.assert_allclose(top[:, :res], S0, atol=1e-6)
    np.testing.assert_allclose(top[:, res:2 * res], D0, atol=1e-6)
    np.testing.assert_allclose(top[:, 2 * res:3 * res], DD0 * ddm03, atol=1e-6)
    # row 2: [SS, DD, SD_100] — the SS tile is stochastic (the
    # exact-k mask is re-drawn on EVERY AE_view call — the
    # preview's own call drew a different mask than any
    # recompute here), so it is contract-checked only (the plan
    # test policy: SS contract-only); DD / SD_100 are
    # deterministic (the decoder path has no randomness)
    ae1 = [_tile(x) for x in model.AE_view(ts, td, 1.0)]
    SD100 = ae1[3]
    ss_tile = bottom[:, :res]
    assert ss_tile.shape == (res, res, 3)
    assert np.isfinite(ss_tile).all()
    assert (ss_tile >= 0.0).all() and (ss_tile <= 1.0).all()
    np.testing.assert_allclose(bottom[:, res:2 * res], DD0, atol=1e-6)
    np.testing.assert_allclose(bottom[:, 2 * res:3 * res], SD100, atol=1e-6)
    # the masked strip's first tile is the plain DD PREDICTION
    # tile (official L701: row 1 = [DD, SD*DDM*SDM, SD*DDM*SDM])
    # — pinned against the deterministic DD, not the raw target
    third = previews[2][1]
    np.testing.assert_allclose(third[:res, :res], DD0, atol=1e-6)


# --- optimizer option couplings ---------------------------------------------------

def test_lr_dropout_y_cpu_coupling(tmp_path):
    """The official lr_dropout coupling: 'y' AND 'cpu' (Q3:
    equivalent on the single-device torch foundation — the
    official AMP never passes the SAEHD lr_dropout_on_cpu
    knob) select lr_cos=500 / lr_dropout=0.3; 'n' selects
    lr_cos=0 / lr_dropout=1.0; clipgrad -> clipnorm=1.0."""
    for value in ('y', 'cpu'):
        model = construct(tmp_path, is_training=True,
                          seed_options=seed(lr_dropout=value, **TINY))
        assert model.src_dst_opt.lr_cos == 500
        assert model.src_dst_opt.lr_dropout == 0.3
        assert model.src_dst_opt.clipnorm == 0.0

    model = construct(tmp_path, is_training=True,
                      seed_options=seed(lr_dropout='n', clipgrad=True, **TINY))
    assert model.src_dst_opt.lr_cos == 0
    assert model.src_dst_opt.lr_dropout == 1.0
    assert model.src_dst_opt.clipnorm == 1.0


def test_should_save_preview_history_cadence(tmp_path, monkeypatch):
    """The official history cadence: every 10 * max(1,
    resolution // 64) iters off-colab (10 for the 64px tiny
    model), every 100 on colab. The official colab heuristic
    depends on the venv composition (IPython importable or not
    — the CPU venv may have it), so the non-colab branch is
    pinned deterministically; the colab branch is the official
    iter % 100 rule (port-identical, not exercised here)."""
    monkeypatch.setattr(io, "is_colab", lambda: False)
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))
    model.set_iter(5)
    assert model.should_save_preview_history() is False
    model.set_iter(10)
    assert model.should_save_preview_history() is True
    model.set_iter(20)
    assert model.should_save_preview_history() is True


# --- save / resume -----------------------------------------------------------------

def test_save_resume_continue_cpu(tmp_path):
    """Save after training -> strict resume in the same
    directory -> continue: the iter, the optimizer
    iterations, the loss history all continue; one more
    iteration after the resume keeps every counter
    consistent. The headless seed layer re-applies the seed
    over the restored data.dat options, so the resumed
    options equal the saved ones for the strict component
    loads."""
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))
    model.train_one_iter()
    it_after_train = model.iter
    iters_after_train = model.src_dst_opt.iterations.item()
    hist_after_train = [row[:] for row in model.loss_history]
    model.save()

    model2 = make_amp(AMPHeadless, str(model.saved_models_path),
                      is_training=True, seed=seed(**TINY), debug=True,
                      cpu_only=True)
    assert model2.iter == it_after_train
    assert model2.src_dst_opt.iterations.item() == iters_after_train
    assert [row[:] for row in model2.loss_history] == hist_after_train

    model2.train_one_iter()
    assert model2.iter == it_after_train + 1
    assert model2.src_dst_opt.iterations.item() == iters_after_train + 1
    assert len(model2.loss_history[-1]) == 2


def test_missing_resume_file_hard_error(tmp_path):
    """A missing required component file on resume fails
    explicitly (FileNotFoundError) instead of the official
    silent re-initialization (the Phase 4/5 strict policy)."""
    model = construct(tmp_path, is_training=True, seed_options=seed(**TINY))
    model.train_one_iter()
    model.save()

    encoder_file = model.get_strpath_storage_for_file('encoder.npy')
    os.remove(encoder_file)

    with pytest.raises(FileNotFoundError, match='required component file'):
        make_amp(AMPHeadless, str(model.saved_models_path),
                 is_training=True, seed=seed(**TINY), debug=True,
                 cpu_only=True)


def test_gan_model_changed_reinit(tmp_path, monkeypatch):
    """The official gan_model_changed re-init rule (L539-541,
    extended to GAN_opt per the Phase 6B precedent — the
    official silent GAN_opt fallback made explicit under the
    strict load policy): on a resume where the official L101
    detection fired (the stored default differs from the
    re-prompted option — the user changed the GAN size), GAN
    and GAN_opt are re-initialized (a fresh optimizer state)
    while every other component loads strictly from the saved
    files. A headless resume has no override prompt, so the
    test simulates the official detection with the harness
    subclass AMPModelGanChanged (defined in the Model_AMPTest
    package so the official module-derived model_class_name
    works; it runs the REAL on_initialize_options and then
    forces the flag — the re-init rule under test is the
    official load-loop behavior). The official input_in_time
    spawns a stdin-poll subprocess (a real stdin fileno is
    required); under pytest capture stdin is a pseudofile, so
    the test neutralizes the interactive layer (the same
    headless pin the first-run options test applies — the
    override never fires)."""
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    monkeypatch.setattr(io, "input_in_time", lambda *a, **k: False)

    # phase 1: train + save with gan_patch_size=8
    model = construct(tmp_path, is_training=True,
                      seed_options=seed(gan_power=0.1, gan_patch_size=8,
                                        gan_dims=4, **TINY))
    model.train_one_iter()
    enc_before = [p.detach().clone().cpu() for p in model.encoder.get_weights()]
    gan_opt_iters_before = model.GAN_opt.iterations.item()
    assert gan_opt_iters_before == 1
    model.save()

    # resume through the REAL on_initialize_options (headless:
    # stored options restored, no first-run prompts, ask_override
    # is a bounded non-blocking poll)
    root = Path(model.saved_models_path)
    model2 = AMPModelGanChanged(
        is_training=True,
        saved_models_path=root,
        training_data_src_path=root / 'src',
        training_data_dst_path=root / 'dst',
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name='test_AMP',
        debug=True,
        cpu_only=True,
    )

    assert model2.gan_model_changed is True
    # the changed discriminator got a FRESH optimizer state
    assert model2.GAN_opt.iterations.item() == 0
    # the unchanged components loaded strictly (the encoder
    # weights are bit-identical to the saved ones)
    enc_after = [p.detach().cpu() for p in model2.encoder.get_weights()]
    for a, b in zip(enc_before, enc_after):
        torch.testing.assert_close(a, b)
    # the rest of the optimizer state survived the resume
    assert model2.src_dst_opt.iterations.item() == 1
    assert model2.iter == 1
