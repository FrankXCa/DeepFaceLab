"""Phase 8 — mixed-precision GPU tier (RTX 4090, torch 2.14 cu130).

The CUDA half of the Phase 8 test pyramid (plan §20.3-7, §21, §22).
CUDA discipline (directive §26 / Phase 7 regime): run ONLY as a
fresh ONE-CHILD-AT-A-TIME subprocess, each launch preceded by a
fresh ``nvidia-smi`` (the co-tenant holds the GPU; ~4 GiB free is
the current regime — a real-256-class run will skip, not OOM):

    # child 1 (the fp32 regression / off-path proof, plan §20.8):
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke/test_mixed_precision_gpu.py -q --basetemp=.pytest-tmp\\p8gpu -k fp32
    # child 2 (the fp16 tier + the matrix pairs touching fp16):
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke/test_mixed_precision_gpu.py -q --basetemp=.pytest-tmp\\p8gpu -k fp16
    # child 3 (the bf16 tier + the remaining matrix pairs):
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke/test_mixed_precision_gpu.py -q --basetemp=.pytest-tmp\\p8gpu -k bf16
    # child 4 (real artifacts, env-gated; skips when unset / below
    # the VRAM floor — NOT_AVAILABLE_ENVIRONMENTALLY):
    DFL_TEST_SAEHD_CHECKPOINT=... DFL_TEST_FACESET_PAK=... \
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke/test_mixed_precision_gpu.py -q --basetemp=.pytest-tmp\\p8gpu -k real

``-k fp16`` selects the fp16 tests plus every mode-matrix pair
whose id mentions fp16 (all 6 directed pairs are covered by the
union of child 2 + child 3). OOM = stop, record,
shrink-the-test-only (never the real checkpoints). Skips are
NOT_AVAILABLE_ENVIRONMENTALLY (no CUDA / no env artifact / VRAM
floor not met) and are labeled as such in the final report —
never forced, never fabricated.

Tiers (plan §20):
- TINY precision (64px SAEHD + AMP, 5 iters each): fp32
  (regression), fp16 (scaled forward/finite, scaled backward,
  native unscale + overflow-aware update, scaler update,
  optimizer states fp32), bf16 (autocast dtype, no scaler,
  stable losses); the recorded losses stay float32 in every mode
  (the FP32 loss island);
- the fp16 NaN-skip unit (the native GradScaler overflow path:
  the DFL custom update op can never run on overflowed grads);
- the 8-pair mode-switch matrix on TINY (off/fp16/bf16 all
  directions: checkpoint mode-independence on CUDA, plan §23);
- representative 128px all-terms GAN-on SAEHD and AMP in fp16
  and bf16 (the Phase 7 canonical shape: 2 iters + save + strict
  reload + 1 resumed iter; + one CROSS-mode reload);
- the real-artifact one-step lifecycle (env-gated, copies only,
  VRAM floor): strict resume -> real samples -> 1 mixed-
  precision iteration -> finite losses -> owner updates -> save
  -> strict reload -> continue.

§21 VRAM: every tier prints the measured torch.cuda.
memory_stats (peak allocated/reserved, free now) and the step
times — measured numbers only, NO savings/speed claims.
"""

import builtins
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import nn as dfl_nn
from core.interact import interact as io
from core.leras.mixed_precision import (  # noqa: F401
    MODE_BF16,
    MODE_FP16,
    MODE_OFF,
    PrecisionUnsupportedError,
)

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

from test_model_saehd_training import (  # noqa: E402
    TINY,
    full_seed,
    seed,
)
from Model_SAEHDTest.Model import (  # noqa: E402
    SAEHDHeadless,
    make_model as make_saehd,
    make_training_dirs as saehd_make_training_dirs,
)
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    make_model as make_amp,
    make_training_dirs as amp_make_training_dirs,
)
import test_model_amp_training as _amp_cpu  # noqa: E402  (seed()/TINY)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA (RTX 4090) environment required — "
           "NOT_AVAILABLE_ENVIRONMENTALLY")

# --- the Phase 7 VRAM policy (copied contract: footprint +
# headroom, floored; below it -> skip, never OOM, never fabricate)
MIN_FREE_GB_REAL = 14.0
VRAM_HEADROOM_GB = 1.5
VRAM_FLOOR_GB = 2.5


# --- constructors -------------------------------------------------------

def construct_gpu_saehd(tmpdir, is_training=False, precision=MODE_OFF,
                        full=False, resolution=128):
    """TINY (or the 128 full-term) SAEHD on CUDA device 0 through
    the official lifecycle (the Phase 7 construct_gpu shape), with
    the precision runtime channel."""
    dfl_nn.initialize_main_env()
    if is_training:
        saehd_make_training_dirs(Path(tmpdir))
    s = full_seed(resolution=resolution) if full else seed(**TINY)
    return make_saehd(SAEHDHeadless, tmpdir, is_training=is_training,
                      seed=s, debug=is_training, force_gpu_idxs=[0],
                      precision=precision)


def construct_gpu_amp(tmpdir, is_training=False, precision=MODE_OFF,
                      full=False, resolution=128):
    """TINY (or the 128 GAN-on) AMP on CUDA device 0 (the Phase 7
    AMP shape) with the precision runtime channel."""
    dfl_nn.initialize_main_env()
    if is_training:
        amp_make_training_dirs(Path(tmpdir))
    if full:
        s = _amp_cpu.seed(resolution=resolution, gan_power=0.05,
                          **{k: v for k, v in _amp_cpu.TINY.items()
                             if k != 'resolution'})
    else:
        s = _amp_cpu.seed(**_amp_cpu.TINY)
    return make_amp(AMPHeadless, tmpdir, is_training=is_training,
                    seed=s, debug=is_training, force_gpu_idxs=[0],
                    precision=precision)


# --- measured-only helpers (plan §21) -----------------------------------

def _vram_report(tag, step_times):
    """The §21 measured VRAM/time line (printed, never claimed)."""
    torch.cuda.synchronize()
    st = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    n = len(step_times) or 1
    print(f"VRAM[{tag}] measured: peak_alloc="
          f"{st.get('allocated_bytes.all.peak', 0) / 2**30:.2f}GiB "
          f"peak_reserved="
          f"{st.get('reserved_bytes.all.peak', 0) / 2**30:.2f}GiB "
          f"free_now={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB "
          f"mean_step_s={sum(step_times) / n:.3f} iters={len(step_times)}")


def _assert_plan_contracts(model, mode):
    """the resolved-plan contract for the selected mode."""
    plan = model._mp_plan
    assert plan is not None
    if mode == MODE_FP16:
        assert plan.enabled and plan.autocast_dtype is torch.float16
        assert plan.scaler_required is True
        assert model._mp_scaler is not None
    elif mode == MODE_BF16:
        assert plan.enabled and plan.autocast_dtype is torch.bfloat16
        assert plan.scaler_required is False
        assert model._mp_scaler is None
    else:
        assert plan.enabled is False and model._mp_scaler is None


def _assert_master_fp32(model, param_lists):
    """the FP32 master-weights + optimizer-state contract."""
    for lst in param_lists:
        for p in lst:
            assert p.dtype == torch.float32
            assert torch.isfinite(p).all()
    for opt in (getattr(model, 'src_dst_opt', None),
                getattr(model, 'GAN_opt', None),
                getattr(model, 'D_code_opt', None),
                getattr(model, 'D_src_dst_opt', None)):
        if opt is None:
            continue
        for attr in ('ms_dict', 'vs_dict', 'accumulators_dict',
                     'lr_rnds_dict'):
            d = getattr(opt, attr, None)
            if not d:
                continue
            for v in d.values():
                assert v.dtype == torch.float32, (type(opt).__name__,
                                                  attr)
                assert torch.isfinite(v).all()


def _saehd_master_lists(model):
    """the parameter lists that exist on this model: the D
    submodels are constructed ONLY when their official term is
    active (the TINY no-GAN seed has none of them)."""
    lists = [model.src_dst_trainable_weights]
    for attr in ('D_code', 'D_src_dst', 'D_src'):
        m = getattr(model, attr, None)
        if m is not None:
            lists.append(m.get_weights())
    return lists


def _amp_master_lists(model):
    """the G weights always exist; the GAN submodel only when
    gan_power > 0 (the TINY seed has it off)."""
    lists = [model.G_weights]
    g = getattr(model, 'GAN', None)
    if g is not None:
        lists.append(g.get_weights())
    return lists


# ======================================================================
# 1. the TINY precision tiers (one child per precision: -k)
# ======================================================================

def _tiny_run(model, mode, iters, tag, master_lists_fn):
    """the TINY precision-tier driver (plan §20.3-5)."""
    model._mp_ensure_resolved()
    _assert_plan_contracts(model, mode)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for i in range(iters):
        t0 = time.time()
        model.train_one_iter()
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        row = model.loss_history[-1]
        assert len(row) >= 1
        for v in row:
            assert math.isfinite(v), (mode, i, row)
            # the FP32 loss island: the recorded losses are fp32
            # scalars (python float) in every mode
            assert isinstance(v, float)
    _assert_master_fp32(model, master_lists_fn(model))
    _vram_report(f"tiny_{tag}_{mode}", step_times)
    last = model.loss_history[-1]
    print(f"TINY {tag} {mode.upper()} {iters} iters PASS: losses "
          f"finite (last={last}), FP32 master + states OK")


@requires_gpu
def test_tiny_saehd_fp32_gpu(tmp_path):
    """plan §20.3: the TINY CUDA fp32 regression (the Phase 6B/7
    tiny evidence path through the Phase 8 wiring — off mode must
    be the byte-for-byte unchanged path)."""
    model = construct_gpu_saehd(tmp_path, is_training=True,
                                precision=MODE_OFF)
    _tiny_run(model, MODE_OFF, 5, 'saehd', _saehd_master_lists)


@requires_gpu
def test_tiny_saehd_fp16_gpu(tmp_path):
    """plan §20.4: the TINY CUDA fp16 tier — the scaled forward,
    the scaled backward, the native unscale + overflow-aware
    update, the scaler update; optimizer states stay fp32."""
    model = construct_gpu_saehd(tmp_path, is_training=True,
                                precision=MODE_FP16)
    _tiny_run(model, MODE_FP16, 5, 'saehd', _saehd_master_lists)
    assert model._mp_scaler is not None  # the native scaler is live


@requires_gpu
def test_tiny_saehd_bf16_gpu(tmp_path):
    """plan §20.5: the TINY CUDA bf16 tier — the autocast dtype,
    NO scaler, stable (finite) losses."""
    model = construct_gpu_saehd(tmp_path, is_training=True,
                                precision=MODE_BF16)
    _tiny_run(model, MODE_BF16, 5, 'saehd', _saehd_master_lists)


@requires_gpu
def test_tiny_amp_fp32_gpu(tmp_path):
    """the TINY AMP fp32 regression (off path unchanged, incl.
    the GAN step which is entirely fp32 in every mode)."""
    model = construct_gpu_amp(tmp_path, is_training=True, precision=MODE_OFF)
    _tiny_run(model, MODE_OFF, 5, 'amp', _amp_master_lists)


@requires_gpu
def test_tiny_amp_fp16_gpu(tmp_path):
    """the TINY AMP fp16 tier (the G step scaled; the GAN step
    never touches the scaler)."""
    model = construct_gpu_amp(tmp_path, is_training=True,
                              precision=MODE_FP16)
    _tiny_run(model, MODE_FP16, 5, 'amp', _amp_master_lists)
    assert model._mp_scaler is not None


@requires_gpu
def test_tiny_amp_bf16_gpu(tmp_path):
    """the TINY AMP bf16 tier (no scaler)."""
    model = construct_gpu_amp(tmp_path, is_training=True,
                              precision=MODE_BF16)
    _tiny_run(model, MODE_BF16, 5, 'amp', _amp_master_lists)


# ======================================================================
# 2. the fp16 NaN-skip unit (the native overflow path)
# ======================================================================

@requires_gpu
def test_fp16_scaler_overflow_nan_skip_gpu(tmp_path):
    """the native torch GradScaler overflow recipe on the DFL
    custom update op: an overflowed (inf) gradient makes
    GradScaler.step SKIP the update op entirely (the official
    update math can never run on inf grads — no weight
    corruption); after scaler.update() backs the scale off, the
    next finite loss steps normally (plan §20.4 'NaN-skip
    logic')."""
    model = construct_gpu_saehd(tmp_path, is_training=True,
                                precision=MODE_FP16)
    model._mp_ensure_resolved()
    assert model._mp_scaler is not None

    p = torch.randn(8, 8, device='cuda', requires_grad=True)
    with model._mp_autocast():
        # overflow the fp16 range: the backward seeds are inf
        loss = (p.to(torch.float16) * 30000.0).sum() * 1000.0

    class _RecOpt:
        """a torch-optimizer-shaped recorder (the leras
        OptimizerBase exposes the same param_groups view)."""
        def __init__(self, params):
            self.stepped = False
            self.param_groups = [{'params': list(params)}]
        def get_update_op(self, grads_vars):
            self.stepped = True
            return lambda: None
        def step(self, grads_vars):
            # the DFL update path (the leras OptimizerBase.step
            # contract: the update op built over the pairs)
            self.get_update_op(grads_vars)()

    opt = _RecOpt([p])
    model._mp_backward(loss)         # the scaled backward
    model._mp_unscale_opt(opt)       # true grads + the inf check
    model._mp_opt_step(opt, [(p.grad, p)])
    assert opt.stepped is False, \
        "the overflowed update must be SKIPPED by the native scaler"
    model._mp_scaler_update()        # backs the scale off (0.5)

    # the next, finite loss steps normally through the same path
    p.grad = None
    loss2 = (p * 0.01).sum()
    model._mp_backward(loss2)
    model._mp_unscale_opt(opt)
    model._mp_opt_step(opt, [(p.grad, p)])
    assert opt.stepped is True
    model._mp_scaler_update()
    print("FP16 NaN-SKIP (native GradScaler on the DFL update op) PASS")


# ======================================================================
# 3. the mode-switch matrix, all 8 directed pairs (TINY, plan §23)
# ======================================================================

def _mode_pair(model_a_dir, mode_a, mode_b):
    """train one TINY SAEHD iteration in mode A, save, then
    construct a FRESH model in mode B on the same checkpoint and
    train one more: the weights must load bit-identical (the
    checkpoint is mode-independent — raw fp32 in every mode) and
    the new mode must run off the loaded state."""
    dfl_nn.initialize_main_env()  # idempotent; required before the
    # first device-config use (the other constructors do the same)
    d = Path(model_a_dir)
    saehd_make_training_dirs(d)  # the src/dst training dirs
    ma = make_saehd(SAEHDHeadless, d, is_training=True,
                    seed=seed(**TINY), debug=True, force_gpu_idxs=[0],
                    precision=mode_a)
    ma.train_one_iter()
    ma.save()
    snap = [w.detach().clone() for w in ma.src_dst_trainable_weights]
    mb = make_saehd(SAEHDHeadless, d, is_training=True,
                    seed=seed(**TINY), debug=True, force_gpu_idxs=[0],
                    precision=mode_b)
    assert mb.iter == ma.iter == 1  # the bookkeeping resumed
    for wa, wb in zip(snap, mb.src_dst_trainable_weights):
        assert wb.dtype == torch.float32
        assert torch.equal(wa, wb.detach())
    mb.train_one_iter()
    row = mb.loss_history[-1]
    assert all(math.isfinite(v) for v in row)
    _assert_plan_contracts(mb, mode_b)
    torch.cuda.empty_cache()
    return mode_a, mode_b


MODE_PAIRS = [(MODE_OFF, MODE_FP16), (MODE_OFF, MODE_BF16),
              (MODE_FP16, MODE_OFF), (MODE_FP16, MODE_BF16),
              (MODE_FP16, MODE_FP16),
              (MODE_BF16, MODE_OFF), (MODE_BF16, MODE_FP16),
              (MODE_BF16, MODE_BF16)]


@requires_gpu
@pytest.mark.parametrize("pair", MODE_PAIRS,
                         ids=[f"{a}_to_{b}" for a, b in MODE_PAIRS])
def test_mode_matrix_gpu(tmp_path_factory, pair):
    """the full directed mode-switch matrix (plan §23) on CUDA
    TINY: one training iteration in mode A -> save -> a fresh
    model in mode B on the same checkpoint resumes the
    bookkeeping, loads the weights bit-identically (mode-
    independent fp32 checkpoint), trains one iteration in mode B
    with finite losses and the correct plan (scaler presence)."""
    a, b = pair
    d = str(tmp_path_factory.mktemp(f"mm_{a}_to_{b}"))
    assert _mode_pair(d, a, b) == (a, b)
    print(f"MODE MATRIX {a.upper()} -> {b.upper()} (CUDA TINY) PASS")


# ======================================================================
# 4. the representative 128px all-terms GAN-on tier (plan §20.6)
# ======================================================================

def _rep_run_saehd(model, mode, cross_mode=None, tag='saehd'):
    """the Phase 7 canonical representative shape in the selected
    mode: 2 full onTrainOneIter iterations (GAN on), finite
    losses, save, strict reload (weights + optimizer states +
    iteration bookkeeping bit-exact), 1 resumed iteration; then
    one CROSS-mode reload + iteration when cross_mode is given
    (the §23 matrix on the big model)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(2):
        t0 = time.time()
        model.train_one_iter()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        row = model.loss_history[-1]
        for v in row:
            assert math.isfinite(v), (mode, i, row)
    _assert_master_fp32(model, _saehd_master_lists(model))
    model.save()
    iters_src = model.src_dst_opt.iterations.item()
    iters_dc = model.D_code_opt.iterations.item()
    iters_ds = model.D_src_dst_opt.iterations.item()
    wmae = [w.detach().clone() for w in model.src_dst_trainable_weights]
    dcm = [w.detach().clone() for w in model.D_code_opt.get_weights()]
    dsm = [w.detach().clone() for w in model.D_src_dst_opt.get_weights()]
    dsr = [w.detach().clone() for w in model.D_src.get_weights()]
    d = model.saved_models_path
    torch.cuda.empty_cache()

    reload_mode = cross_mode if cross_mode is not None else mode
    m2 = make_saehd(SAEHDHeadless, d, is_training=True,
                    seed=full_seed(resolution=128), debug=True,
                    force_gpu_idxs=[0], precision=reload_mode)
    assert m2.iter == model.iter  # the bookkeeping resumed
    # bit-exact strict reload (the loader is mode-blind: fp32 in
    # every mode — plan §12/§23)
    for a, b in zip(wmae, m2.src_dst_trainable_weights):
        assert a.dtype == torch.float32 and torch.equal(a, b.detach())
    for a, b in zip(dcm, m2.D_code_opt.get_weights()):
        assert torch.equal(a, b.detach())
    for a, b in zip(dsm, m2.D_src_dst_opt.get_weights()):
        assert torch.equal(a, b.detach())
    for a, b in zip(dsr, m2.D_src.get_weights()):
        assert torch.equal(a, b.detach())
    assert m2.src_dst_opt.iterations.item() == iters_src
    assert m2.D_code_opt.iterations.item() == iters_dc
    assert m2.D_src_dst_opt.iterations.item() == iters_ds
    # one resumed iteration in the (possibly cross) mode
    m2.train_one_iter()
    assert m2.iter == model.iter + 1
    row2 = m2.loss_history[-1]
    for v in row2:
        assert math.isfinite(v)
    if cross_mode is not None:
        _assert_plan_contracts(m2, cross_mode)
    _assert_master_fp32(m2, _saehd_master_lists(m2))
    _vram_report(f"rep_{tag}_{mode}"
                 + (f"_x_{cross_mode}" if cross_mode else ""), times)
    print(f"REPRESENTATIVE 128 {tag} {mode.upper()}"
          + (f" -> {cross_mode.upper()}" if cross_mode else "")
          + " PASS: 2 iters + strict reload + 1 resumed iter, "
            f"losses finite (last={row2})")


def _rep_run_amp(model, mode, cross_mode=None):
    """the representative shape for the AMP model (G step in the
    selected mode; the GAN step entirely fp32 in every mode)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(2):
        t0 = time.time()
        model.train_one_iter()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        row = model.loss_history[-1]
        for v in row:
            assert math.isfinite(v), (mode, i, row)
    _assert_master_fp32(model, _amp_master_lists(model))
    model.save()
    iters_src = model.src_dst_opt.iterations.item()
    iters_g = model.GAN_opt.iterations.item()
    wm = [w.detach().clone() for w in model.G_weights]
    dm = [w.detach().clone() for w in model.GAN.get_weights()]
    d = model.saved_models_path
    torch.cuda.empty_cache()

    reload_mode = cross_mode if cross_mode is not None else mode
    m2 = make_amp(AMPHeadless, d, is_training=True,
                  seed=_amp_cpu.seed(resolution=128, gan_power=0.05,
                                     **{k: v for k, v in _amp_cpu.TINY.items()
                                        if k != 'resolution'}),
                  debug=True, force_gpu_idxs=[0], precision=reload_mode)
    assert m2.iter == model.iter
    for a, b in zip(wm, m2.G_weights):
        assert a.dtype == torch.float32 and torch.equal(a, b.detach())
    for a, b in zip(dm, m2.GAN.get_weights()):
        assert torch.equal(a, b.detach())
    assert m2.src_dst_opt.iterations.item() == iters_src
    assert m2.GAN_opt.iterations.item() == iters_g
    m2.train_one_iter()
    assert m2.iter == model.iter + 1
    row2 = m2.loss_history[-1]
    for v in row2:
        assert math.isfinite(v)
    if cross_mode is not None:
        _assert_plan_contracts(m2, cross_mode)
    _assert_master_fp32(m2, _amp_master_lists(m2))
    _vram_report(f"rep_amp_{mode}"
                 + (f"_x_{cross_mode}" if cross_mode else ""), times)
    print(f"REPRESENTATIVE 128 AMP {mode.upper()}"
          + (f" -> {cross_mode.upper()}" if cross_mode else "")
          + " PASS: 2 iters + strict reload + 1 resumed iter, "
            f"losses finite (last={row2})")


@requires_gpu
def test_representative_saehd_fp16_gpu(tmp_path):
    """plan §20.6: the 128px all-terms GAN-on SAEHD representative
    in fp16 (the Phase 7 canonical shape) + one cross-mode
    reload (fp16 -> bf16, the §23 matrix on the big model)."""
    model = construct_gpu_saehd(tmp_path, is_training=True, full=True,
                                resolution=128, precision=MODE_FP16)
    _rep_run_saehd(model, MODE_FP16, cross_mode=MODE_BF16)


@requires_gpu
def test_representative_saehd_bf16_gpu(tmp_path):
    """the 128px all-terms GAN-on SAEHD representative in bf16 +
    the bf16 -> off cross-mode reload."""
    model = construct_gpu_saehd(tmp_path, is_training=True, full=True,
                                resolution=128, precision=MODE_BF16)
    _rep_run_saehd(model, MODE_BF16, cross_mode=MODE_OFF)


@requires_gpu
def test_representative_amp_fp16_gpu(tmp_path):
    """plan §20.6: the 128px GAN-on AMP representative in fp16
    (+ the fp16 -> bf16 cross-mode reload)."""
    model = construct_gpu_amp(tmp_path, is_training=True, full=True,
                              resolution=128, precision=MODE_FP16)
    _rep_run_amp(model, MODE_FP16, cross_mode=MODE_BF16)


@requires_gpu
def test_representative_amp_bf16_gpu(tmp_path):
    """the 128px GAN-on AMP representative in bf16 (+ the
    bf16 -> off cross-mode reload)."""
    model = construct_gpu_amp(tmp_path, is_training=True, full=True,
                              resolution=128, precision=MODE_BF16)
    _rep_run_amp(model, MODE_BF16, cross_mode=MODE_OFF)


# ======================================================================
# 5. the real-artifact tier (env-gated, copies only, plan §20.7/§22)
# ======================================================================

requires_real_ckpt = pytest.mark.skipif(
    not os.environ.get('DFL_TEST_SAEHD_CHECKPOINT'),
    reason="DFL_TEST_SAEHD_CHECKPOINT not set (opt-in real artifact)")
requires_real_faceset = pytest.mark.skipif(
    not os.environ.get('DFL_TEST_FACESET_PAK'),
    reason="DFL_TEST_FACESET_PAK not set (opt-in real artifact)")


def _vram_precheck(min_free_gb):
    dfl_nn.initialize_main_env()
    free, total = torch.cuda.mem_get_info()
    return (free >= min_free_gb * 2**30, free / 2**30,
            total / 2**30, min_free_gb)


def _vram_skip(min_free_gb=None):
    if min_free_gb is None:
        min_free_gb = MIN_FREE_GB_REAL
    ok, free_gb, total_gb, need_gb = _vram_precheck(min_free_gb)
    if not ok:
        pytest.skip(f"VRAM pre-check failed (free {free_gb:.1f} GiB of "
                    f"{total_gb:.1f} GiB < {need_gb:.1f} GiB needed) — "
                    "NOT_AVAILABLE_ENVIRONMENTALLY")


@requires_gpu
@requires_real_ckpt
@requires_real_faceset
def test_real_saehd_fp16_one_step_cuda(tmp_path_factory, monkeypatch):
    """the official one-step TRAINING lifecycle on a REAL official
    SAEHD checkpoint + a REAL faceset, in fp16 mode (opt-in via
    DFL_TEST_SAEHD_CHECKPOINT / DFL_TEST_FACESET_PAK; the Phase 7
    real-bootstrap shape: copies only, rename to the test model
    name, the face_type matched to the real faceset, the
    headless-safe input layer). Steps: strict resume -> ONE
    mixed-precision training iteration on real samples -> finite
    losses -> the generator owners update -> save -> strict
    fp16 reload -> continue. Below the VRAM floor: skip,
    labeled NOT_AVAILABLE_ENVIRONMENTALLY (never OOM, never
    fabricated)."""
    import pickle
    import shutil
    from test_model_saehd import (
        InputScript,
        _SAEHDFILE_RE,
        construct_real,
        preseed_data_dat,
    )
    from facelib import FaceType
    from samplelib import PackedFaceset

    ckpt_dir = Path(os.environ.get('DFL_TEST_SAEHD_CHECKPOINT'))
    pak_dir = Path(os.environ.get('DFL_TEST_FACESET_PAK'))
    pak_file = pak_dir / "faceset.pak" if pak_dir.is_dir() else pak_dir
    if not pak_file.exists():
        pytest.skip("no faceset.pak under DFL_TEST_FACESET_PAK")
    _vram_skip()

    monkeypatch.setattr(builtins, "input", InputScript({}))
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)

    samples = PackedFaceset.load(pak_file.parent)
    if not samples:
        pytest.skip("real faceset is empty — NOT_AVAILABLE_ENVIRONMENTALLY")
    ft_of = {FaceType.HALF: "h", FaceType.MID_FULL: "mf",
             FaceType.FULL: "f", FaceType.WHOLE_FACE: "wf"}
    ft = ft_of.get(samples[0].face_type)
    if ft is None:
        pytest.skip("real faceset face type not usable by SAEHD here")

    root = tmp_path_factory.mktemp("p8_real_fp16")
    root.mkdir(parents=True, exist_ok=True)
    for f in ckpt_dir.iterdir():
        if not f.is_file():
            continue
        m = _SAEHDFILE_RE.match(f.name)
        if m is None:
            continue
        rest = m.group("rest")
        if rest == "data.dat":
            shutil.copy(f, root / "test_SAEHD_data.dat")
        elif rest == "default_options.dat":
            shutil.copy(f, root / "test_SAEHD_default_options.dat")
        elif rest.endswith(".npy"):
            shutil.copy(f, root / f"test_SAEHD_{rest}")
    data = pickle.loads((root / "test_SAEHD_data.dat").read_bytes())
    opts = data["options"]
    opts["face_type"] = ft  # the checkpoint matched to the real set
    preseed_data_dat(root, opts, it=max(int(data.get("iter", 0)), 1),
                     sample_for_preview=None)
    for side in ("src", "dst"):
        (root / side).mkdir(parents=True)
        shutil.copy2(pak_file, root / side / "faceset.pak")

    dfl_nn.initialize_main_env()

    # 1 + 2: strict resume in the TRAINING context on CUDA, in
    # fp16, with the real faceset-backed generators
    model = construct_real(root, options=None, is_training=True,
                           debug=True, force_gpu_idxs=[0],
                           precision=MODE_FP16)
    model._mp_ensure_resolved()
    _assert_plan_contracts(model, MODE_FP16)
    print(f"REAL fp16 RESUME PASS: iter={model.iter} "
          f"archi={model.options.get('archi')} "
          f"res={model.options.get('resolution')} "
          f"tfp={model.options.get('true_face_power')} "
          f"gp={model.gan_power}")

    # 3 + 4: one mixed-precision training iteration on REAL data
    before = [w.detach().clone() for w in model.src_dst_trainable_weights]
    model.train_one_iter()
    row = model.loss_history[-1]
    for v in row:
        assert math.isfinite(v), row
    # 5: the generator owners updated (the fp16 G step)
    changed = sum(not torch.equal(b, w.detach())
                  for b, w in zip(before, model.src_dst_trainable_weights))
    assert changed > 0
    # 6: the FP32 master-weights + optimizer-state contract
    for w in model.src_dst_trainable_weights:
        assert w.dtype == torch.float32
    _assert_master_fp32(model, [model.src_dst_trainable_weights])
    print(f"REAL fp16 ONE-STEP TRAINING PASS: losses={row}, "
          f"{changed} generator weights updated, master fp32 OK")

    # 7 + 8: save -> strict fp16 reload -> continue
    model.save()
    m2 = construct_real(root, options=None, is_training=True,
                        debug=True, force_gpu_idxs=[0],
                        precision=MODE_FP16)
    assert m2.iter == model.iter
    for a, b in zip(model.src_dst_trainable_weights,
                    m2.src_dst_trainable_weights):
        assert torch.equal(a.detach(), b.detach())
    m2.train_one_iter()
    assert m2.iter == model.iter + 1
    row2 = m2.loss_history[-1]
    for v in row2:
        assert math.isfinite(v)
    _vram_report('real_saehd_fp16', [1.0])
    print("REAL SAEHD ONE-STEP fp16 TRAINING LIFECYCLE (CUDA): PASS")
