"""Phase 8 — mixed-precision POLICY tests: the CPU tier.

The fast, deterministic CPU half of the Phase 8 test pyramid
(plan §20.1/20.2 — the CUDA tier lives in
test_mixed_precision_gpu.py, one fresh child per precision):

1. Static policy/contract tests: mode resolution for
   off/fp16/bf16 x CPU (the CUDA capability ordering is pinned
   by the capability table + the real-device checks in the GPU
   file), the explicit ``PrecisionUnsupportedError`` (a
   ``ValueError``) on every unsupported pair — Milestone E:
   NO silent fallback; the fp32-island helper degeneration in
   'off' mode (nullcontext + no scaler + the exact Phase 7
   code path);
2. CPU option/checkpoint tests: TINY SAEHD + TINY AMP in 'off'
   and bf16-on-CPU (finite losses, update success, save/
   reload/continue, mode switch off<->bf16 on resume);
   fp16-on-CPU => ValueError; the precision channel never
   enters the io option keys, never writes an extra file, and
   leaves the saved data.dat key set mode-invariant (D-6);
   every loaded parameter stays float32 in every mode (the
   FP32 master-weights contract, plan §12).

Parity labels: NEW_PHASE8_DESIGN feature (no official TF
counterpart); 'off'-mode behavior is the unchanged Phase 6B/7
path (its regression pins are the Phase 7 suites, re-run green
in this phase — see docs/PHASE8_STATE.md M8).

Run:  .\\.venv-cpu\\Scripts\\python.exe -m pytest
      tests/smoke/test_mixed_precision_policy.py -q
      --basetemp=.pytest-tmp\\p8cpu
"""

import math
import pickle
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (npy contract in the checkpoint test)
import pytest
import torch

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import nn  # noqa: F401  (the dfl nn module)
from core.interact import interact as io  # noqa: F401  (log sink)
from core.leras import mixed_precision as mp
from core.leras.mixed_precision import (  # noqa: F401
    MODE_BF16,
    MODE_FP16,
    MODE_OFF,
    PrecisionUnsupportedError,
    resolve_precision,
)

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

# the CPU-tier TINY SAEHD configuration + synthetic sample
# helpers (the Phase 6B/7 tiny definitions, plan §20.2)
from test_model_saehd_training import (  # noqa: E402
    TINY,
    construct,
    seed,
    synth_samples,
    tensors8,
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


def construct_prec(tmpdir, precision, is_training=True, seed_opts=None):
    """TINY SAEHD on CPU through the official lifecycle with the
    precision runtime channel passed through the constructor
    (``make_model`` forwards **device_kwargs to the model)."""
    if is_training:
        saehd_make_training_dirs(Path(tmpdir))
    return make_saehd(SAEHDHeadless, tmpdir, is_training=is_training,
                      seed=seed(**(seed_opts or TINY)),
                      debug=is_training, cpu_only=True,
                      precision=precision)


# ======================================================================
# 1. the policy layer (pure — no model construction)
# ======================================================================

def test_policy_off_any_device():
    """'off' resolves on EVERY device: the no-op plan (nullcontext,
    no scaler) — the exact Phase 7 fp32 code path."""
    for dev in (torch.device('cpu'), torch.device('cuda')):
        p = resolve_precision(MODE_OFF, dev)
        assert p.mode == MODE_OFF
        assert p.device_type == dev.type
        assert not p.enabled
        assert p.autocast_dtype is None
        assert p.scaler_required is False
        assert p.make_scaler() is None
        with p.autocast_context():
            pass  # nullcontext — inert
        assert MODE_OFF in p.describe()


def test_policy_bf16_cpu():
    """bf16 on CPU: autocast dtype bfloat16, NO scaler (bf16 has
    the fp32 exponent range), the forward math downcasts."""
    p = resolve_precision(MODE_BF16, torch.device('cpu'))
    assert p.enabled
    assert p.scaler_required is False
    assert p.autocast_dtype is torch.bfloat16
    assert p.make_scaler() is None
    x = torch.randn(2, 3)
    w = torch.randn(4, 3)
    with p.autocast_context():
        out = torch.nn.functional.linear(x, w)
    assert out.dtype == torch.bfloat16


def test_policy_fp16_cpu_explicit_error():
    """fp16 on CPU: EXPLICIT PrecisionUnsupportedError (a
    ValueError) naming the mode and device — never a silent
    fallback (Milestone E)."""
    cpu = torch.device('cpu')
    with pytest.raises(PrecisionUnsupportedError):
        resolve_precision(MODE_FP16, cpu)
    with pytest.raises(ValueError) as ei:
        resolve_precision(MODE_FP16, cpu)
    msg = str(ei.value).lower()
    assert 'fp16' in msg and 'cpu' in msg


def test_policy_cuda_unavailable_explicit_error():
    """fp16/bf16 on a CUDA device type on a machine WITHOUT CUDA:
    the policy layer fails explicitly (a clean
    PrecisionUnsupportedError, never a torch internal crash)."""
    if torch.cuda.is_available():
        pytest.skip('CUDA is available here — the no-CUDA branch '
                    'cannot be exercised (covered by the CPU venv)')
    cuda = torch.device('cuda')
    for mode in (MODE_FP16, MODE_BF16):
        with pytest.raises(PrecisionUnsupportedError):
            resolve_precision(mode, cuda)


def test_policy_unknown_mode_explicit_error():
    """'auto' is NOT a mode (plan §7); unknown mode strings and
    device types fail explicitly."""
    cpu = torch.device('cpu')
    for bad in ('auto', 'fp32', 'mixed', 'half', None):
        with pytest.raises(PrecisionUnsupportedError):
            resolve_precision(bad, cpu)
    with pytest.raises(PrecisionUnsupportedError):
        resolve_precision(MODE_BF16, torch.device('opencl'))


def test_capability_table_pins():
    """the sm_ capability ordering (fp16: sm>=5.3; bf16:
    sm>=8.0) is pinned — the CUDA tier checks the REAL device
    against these (test_mixed_precision_gpu.py)."""
    assert mp._FP16_MIN_CAP == (5, 3)
    assert mp._BF16_MIN_CAP == (8, 0)
    assert mp._cap_ge((8, 9), (8, 0)) is True    # RTX 4090 = sm_89
    assert mp._cap_ge((7, 5), (8, 0)) is False   # Turing: no bf16
    assert mp._cap_ge((5, 3), (5, 3)) is True
    assert mp._cap_ge((5, 2), (5, 3)) is False
    assert mp._cap_ge((3, 5), (5, 3)) is False


# ======================================================================
# 2. the ModelBase runtime channel (TINY models, CPU)
# ======================================================================

def test_channel_off_degenerates_to_phase7_path(tmp_path):
    """precision='off' (the default): nullcontext autocast, no
    scaler, _mp_backward = the plain explicit-grad backward,
    _mp_opt_step = the exact official update op, scaler update a
    no-op — the literal Phase 6B/7 code path."""
    model = construct_prec(tmp_path, MODE_OFF)
    model._mp_ensure_resolved()
    assert model.precision == MODE_OFF
    assert model._mp_plan is not None
    assert model._mp_plan.enabled is False
    assert model._mp_scaler is None

    region = model._mp_autocast()
    assert type(region).__name__ == 'nullcontext'

    t = torch.tensor([1.0, 2.0], requires_grad=True)
    model._mp_backward(t)  # off mode: the plain explicit-grad backward
    assert torch.equal(t.grad, torch.ones(2))

    class _DummyOpt:
        # duck-typed official-shape optimizer: the off-mode
        # _mp_opt_step must call get_update_op(pairs)() and nothing
        # else
        def __init__(self):
            self.calls = []
        def get_update_op(self, grads_vars):
            self.calls.append(grads_vars)
            return lambda: None

    opt = _DummyOpt()
    model._mp_opt_step(opt, [ (t.grad, t) ])
    assert len(opt.calls) == 1
    # the update op received the (grad, param) pair VERBATIM
    assert opt.calls[0][0][0] is t.grad
    assert opt.calls[0][0][1] is t
    model._mp_scaler_update()  # no-op in off mode

    # resolution is idempotent (one-shot)
    plan = model._mp_plan
    model._mp_ensure_resolved()
    assert model._mp_plan is plan


def test_channel_bf16_cpu_train_step(tmp_path):
    """bf16 on CPU end-to-end on the TINY SAEHD: the G step runs
    its AE forward under the bf16 autocast region, the FP32 loss
    island follows, the loss is finite, the generator weights are
    updated and STAY float32 (the FP32 master contract)."""
    model = construct_prec(tmp_path, MODE_BF16)
    model._mp_ensure_resolved()
    assert model._mp_plan.enabled is True
    assert model._mp_plan.autocast_dtype is torch.bfloat16
    assert model._mp_scaler is None  # bf16: NO scaler

    samples = synth_samples(64, batch=1, seed_no=0)
    t8 = tensors8(samples)
    before = [p.detach().clone() for p in model.src_dst_trainable_weights]
    src_loss, dst_loss = model._src_dst_train(*t8)
    for v in list(src_loss) + list(dst_loss):
        assert math.isfinite(float(v))
    changed = sum(not torch.equal(p.detach(), b)
                  for p, b in zip(model.src_dst_trainable_weights,
                                  before))
    assert changed > 0  # the update really happened
    for p in model.src_dst_trainable_weights:
        assert p.dtype == torch.float32


def test_channel_amp_bf16_cpu_train_step(tmp_path):
    """the TINY AMP model on CPU in bf16: one full onTrainOneIter
    (G step under bf16 autocast + boundary cast; the GAN step is
    off by default in the tiny seed) — finite losses, fp32
    master weights, no scaler."""
    s = _amp_cpu.seed(**_amp_cpu.TINY)
    amp_make_training_dirs(Path(tmp_path))
    model = make_amp(AMPHeadless, tmp_path, is_training=True,
                     seed=s, debug=True, cpu_only=True,
                     precision=MODE_BF16)
    model.train_one_iter()
    assert model.iter == 1
    row = model.loss_history[-1]
    assert all(math.isfinite(v) for v in row)
    assert model._mp_scaler is None
    assert model._mp_plan.enabled is True
    for p in model.G_weights:
        assert p.dtype == torch.float32


def test_channel_fp16_cpu_explicit_error_before_any_work(tmp_path):
    """fp16 requested on a CPU-only model: the explicit error
    fires at training start (the first step), BEFORE any training
    work — no weight is touched, no silent fp32 downgrade."""
    model = construct_prec(tmp_path, MODE_FP16)
    before = [p.detach().clone() for p in model.src_dst_trainable_weights]
    with pytest.raises(PrecisionUnsupportedError):
        model.train_one_iter()
    for b, p in zip(before, model.src_dst_trainable_weights):
        assert torch.equal(b, p.detach())  # nothing trained
    assert model.iter == 0


# ======================================================================
# 3. options / checkpoint contracts (D-6 + plan §4/§12/§23)
# ======================================================================

def test_precision_is_a_runtime_channel_not_an_option(tmp_path):
    """the precision parameter is a RUNTIME constructor channel
    (like cpu_only / force_gpu_idxs): it never becomes an io
    option key, so no option key exists for it in model.options
    (D-6: no new option keys, no torch_amp_* keys)."""
    model = construct_prec(tmp_path, MODE_BF16)
    assert model.precision == MODE_BF16
    for banned in ('precision', 'use_bf16', 'torch_amp', 'torch_amp_state'):
        assert banned not in model.options
    # and the official, SEPARATE use_fp16 export knob keeps its
    # official semantics (a plain option if the model owns it)
    if 'use_fp16' in model.options:
        assert model.options['use_fp16'] in (False, True)


def test_no_extra_files_after_bf16_training(tmp_path):
    """BF16 creates no precision checkpoint state; previews may be async."""
    model = construct_prec(tmp_path, MODE_BF16)
    checkpoint_suffixes = {'.npy', '.dat', '.pth'}
    files_before = {str(p.relative_to(tmp_path))
                    for p in Path(tmp_path).rglob('*')
                    if p.is_file() and p.suffix in checkpoint_suffixes}
    model.train_one_iter()
    model.train_one_iter()
    files_after = {str(p.relative_to(tmp_path))
                   for p in Path(tmp_path).rglob('*')
                   if p.is_file() and p.suffix in checkpoint_suffixes}
    assert files_after == files_before
    assert all('precision' not in p.lower() and 'torch_amp' not in p.lower()
               for p in files_after)


def test_checkpoint_mode_independent_and_keys_invariant(tmp_path_factory):
    """mode switch on resume (plan §20.2/§23): a model trained in
    'off' saves the checkpoint; a FRESH model constructed on the
    SAME directory in bf16 mode loads it (weights bit-identical,
    float32), trains one step and continues from the saved
    iteration; the saved data.dat key sets of both modes are
    IDENTICAL (no precision key — D-6)."""
    d = tmp_path_factory.mktemp('mp_mode_switch')
    m_off = construct_prec(d, MODE_OFF)
    for _ in range(2):
        m_off.train_one_iter()
    assert m_off.iter == 2
    m_off.save()
    data_dat = d / (m_off.get_model_name() + '_data.dat')
    assert data_dat.exists()
    with open(data_dat, 'rb') as f:
        keys_off = set(pickle.load(f))
    assert 'precision' not in keys_off

    # the fresh bf16 model on the SAME checkpoint (strict resume)
    saehd_make_training_dirs(Path(d))
    m_bf = make_saehd(SAEHDHeadless, d, is_training=True,
                      seed=seed(**TINY), debug=True, cpu_only=True,
                      precision=MODE_BF16)
    assert m_bf.iter == 2  # the bookkeeping resumed
    # bit-identical float32 weights across the mode switch
    assert len(m_bf.src_dst_trainable_weights) == \
        len(m_off.src_dst_trainable_weights)
    for a, b in zip(m_off.src_dst_trainable_weights,
                    m_bf.src_dst_trainable_weights):
        assert a.dtype == torch.float32
        assert torch.equal(a.detach(), b.detach())
    # one step in the NEW mode off the loaded state
    m_bf.train_one_iter()
    assert m_bf.iter == m_off.iter + 1
    row = m_bf.loss_history[-1]
    assert all(math.isfinite(v) for v in row)
    # the saved bookkeeping keys are mode-invariant
    m_bf.save()
    with open(data_dat, 'rb') as f:
        keys_bf = set(pickle.load(f))
    assert keys_off == keys_bf
    assert 'precision' not in keys_bf
    # no saved file is float64: the FP32 master contract means
    # every persisted tensor is float32 in every mode (the
    # parameter dtype equality above pins the in-memory side)
    for p in d.rglob('*.npy'):
        arr = np.load(p, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            assert arr.dtype != np.float64, p.name


def test_preview_identical_across_modes(tmp_path_factory):
    """completion criterion 6: preview/inference is MODE-INDEPENDENT.
    A model trained in 'off' saves the checkpoint; two FRESH
    inference models on the SAME bit-identical fp32 checkpoint —
    one in off mode, one in bf16 mode — return numerically
    IDENTICAL ``predictor_func`` outputs (bgr + both mask
    channels) for the same face: the preview path never enters
    the precision autocast boundary in any mode (the fp32
    inference island), so no precision artifact can leak into
    preview output."""
    d = tmp_path_factory.mktemp('mp_preview')
    m_tr = construct_prec(d, MODE_OFF)
    m_tr.train_one_iter()
    m_tr.save()
    # both comparison sides: inference models on the same weights
    m_off = construct_prec(d, MODE_OFF, is_training=False)
    m_bf = construct_prec(d, MODE_BF16, is_training=False)
    face = (np.random.rand(64, 64, 3) * 2.0 - 1.0).astype(np.float32)
    out_off = m_off.predictor_func(face)
    out_bf = m_bf.predictor_func(face)
    assert len(out_off) == 3 and len(out_bf) == 3
    for a, b in zip(out_off, out_bf):
        assert isinstance(b, np.ndarray)
        assert b.shape == a.shape and b.dtype == a.dtype
        np.testing.assert_array_equal(a, b)
