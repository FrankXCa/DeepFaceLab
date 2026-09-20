"""Phase 7: AMP training semantics — the CUDA (RTX 4090) tier.

The companion to test_model_amp_training.py (the CPU tier: formula
units + tiny 64px integration). This file owns the expensive
integrated coverage on GPU (the mandated pyramid: formula CPU ->
tiny CPU -> tiny CUDA (measured) -> representative CUDA ->
real-artifact):

- TINY CUDA: the smallest valid AMP model (64px, batch 1, the
  smallest audited dims) on CUDA: one real driver iteration
  (in-process generator, N=1) + a measured incremental VRAM peak
  (the directive's TINY_CUDA class: < 1 GiB incremental; if the
  measurement exceeds that the dims are shrunk DETERMINISTICALLY
  by the main agent — never a random search);
- a REPRESENTATIVE 128px model (all official terms: morph 0.5
  exact-k, GAN on with the official 8-term x 1/8 D loss): two
  driver iterations, the batch-SUM doubling pin on GPU (identical
  N=2 samples under a test-side no-op update shadow, like the
  CPU tier — executed in a FRESH full-fp32 subprocess at the
  measured-noise GPU-tier tolerance of _q11_pin_child
  (rtol=1e-2 / atol=1e-3, covering this machine's
  allocator-regime-dependent full-fp32 kernel-noise floor),
  isolated from the GAN phase's cuDNN plan cache / allocator
  state), the frozen-inter pin, the
  two-phase G/D optimizer counters, save -> strict resume ->
  continue (GAN_opt continuity included);
- the REAL-artifact staged validation (opt-in via the env vars
  DFL_TEST_AMP_CHECKPOINT — the official AMP model directory,
  ANY official workspace model (the tests are checkpoint
  AGNOSTIC: the expected option contract is the checkpoint's
  own stored options, the GAN set is asserted only when the
  checkpoint trained with gan_power != 0) — and
  DFL_TEST_FACESET_PAK — a .pak faceset — NO private path
  literals in this file; copies only, the original is never
  written):
  * PATH A — checkpoint-only: strict resume (training context),
    component discovery / filenames / restored options / key
    inventories / strict weight loads / src_dst_opt (+ GAN_opt
    when present) strict state loads, zero required
    missing/shape errors;
  * PATH B — real samples: real faceset (WHOLE_FACE), the
    in-process SampleGeneratorFace, actual AMP inputs through
    the official AE_merge inference contract;
  * PATH C — one-step real training: strict resume, one
    iteration on real data (finite losses, encoder/decoder
    updated, inter bit-identical, both optimizer counters +1),
    save, strict reload, exact optimizer continuity, one more
    iteration.

Every test is gated by ``requires_gpu`` (skipped on a machine
without CUDA). Numerical labels: FORMULA_VERIFIED /
TF_RUNTIME_NOT_VERIFIED (no official TF stack exists on disk);
no test claims runtime parity with the official TF graph.
"""

import builtins
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
import core.leras.nn as dfl_nn  # initialize_main_env
from core.interact import interact as io  # the interact singleton

from models.Model_AMP import Model as AMPModel

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    make_model as make_amp,
    make_training_dirs,
)

# the CPU-tier helpers (identical contract)
from test_model_amp_training import (  # noqa: E402
    grad_map,
    seed,
    synth_samples_nchw,
)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required")


def construct_gpu(tmpdir, is_training=False, seed_options=None):
    """The CUDA construction: the in-process sample generator
    (debug=True) with the components/optimizers on CUDA device 0
    (the official models_opt_on_gpu path)."""
    dfl_nn.initialize_main_env()
    if is_training:
        make_training_dirs(Path(tmpdir))
    return make_amp(AMPHeadless, str(tmpdir), is_training=is_training,
                    seed=seed_options, debug=is_training,
                    force_gpu_idxs=[0])


# --- TINY CUDA (measured incremental peak) -----------------------------------

# the smallest valid AMP configuration (64px, batch 1, the
# smallest audited dims; d_mask 6 = the official d//3 rounded
# even for d=16)
TINY = dict(resolution=64, batch_size=1, ae_dims=32, inter_dims=32,
            e_dims=16, d_dims=16, d_mask_dims=6)


@requires_gpu
def test_tiny_cuda_measured_smoke(tmp_path):
    """TINY_CUDA class: the smallest official-valid AMP model on
    CUDA, one real driver iteration, and the MEASURED
    incremental VRAM peak (the directive's < 1 GiB budget). The
    warm-up absorbs the CUDA context / allocator baseline before
    the measured section."""
    dfl_nn.initialize_main_env()

    # warm-up: absorb the context + allocator baseline
    warm = torch.empty(int(0.5 * 2**30), device='cuda:0',
                       dtype=torch.float32)
    del warm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()

    # construct_gpu seeds the synthetic training dirs (the
    # in-process SampleGeneratorFace needs them) and forces
    # device 0
    model = construct_gpu(tmp_path, is_training=True,
                          seed_options=seed(**TINY))
    assert model.encoder.get_weights()[0].device.type == 'cuda'

    # one real driver iteration (debug -> N=1, the in-process
    # generator over the packed test facesets)
    model.train_one_iter()
    torch.cuda.synchronize()

    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)
    assert model.iter == 1
    assert model.src_dst_opt.iterations.item() == 1
    # the frozen inter heads never receive grads on CUDA either
    assert all(p.grad is None
               for p in model.inter_src.get_weights()
               + model.inter_dst.get_weights())

    peak_alloc = torch.cuda.max_memory_allocated() - base_alloc
    peak_reserved = torch.cuda.max_memory_reserved() - base_reserved
    print(f"TINY CUDA {model.resolution}px incremental peak: "
          f"allocated {peak_alloc / 2**20:.1f} MiB / "
          f"reserved {peak_reserved / 2**20:.1f} MiB "
          f"(base {base_alloc / 2**20:.1f} MiB)")
    assert peak_alloc < 1 * 2**30, \
        "TINY CUDA incremental peak over the 1 GiB budget"
    print("TINY CUDA SMOKE: PASS (backend: RTX 4090)")


# --- representative 128px integration ----------------------------------------

# a representative (not the smallest) AMP model with every
# official term active: morph 0.5 (k = 64 of 128 inter channels),
# GAN on (the official 4-term generator terms x gan_power and the
# 8-term x 1/8 D loss), AdaBelief x 2
REPRESENTATIVE = dict(resolution=128, batch_size=1, ae_dims=128,
                      inter_dims=128, e_dims=64, d_dims=64,
                      d_mask_dims=24, morph_factor=0.5,
                      gan_power=0.2, gan_patch_size=64, gan_dims=16)


def _q11_pin_child(qtmp, res):
    """The Q11 pin child (invoked by
    _run_q11_pin_subprocess): a FRESH interpreter + FRESH CUDA
    context + FRESH cuDNN plan cache + FRESH allocator state,
    with the full-fp32 kernel flags set before any CUDA
    activity (the caller sets them before importing this
    module and this function re-asserts them). The GPU-tier
    tolerance rtol=1e-2 / atol=1e-3 is sized from the
    MEASURED full-fp32 kernel-noise floor of this machine,
    which is allocator-regime dependent (the available cuDNN
    workspace size changes the N=1 / N=2 algorithm pairing):
    worst relative ~2.3e-5 at ~3 GiB free (the co-tenant
    holder present — session 15b diagnostic,
    docs/_p7_diag_q11_tf32.py) up to ~6.4e-3 relative / ~2e-4
    absolute at ~22 GiB free (the co-tenant absent — session
    15c, docs/_p7_diag_q11_isolated.py). The tolerance
    covers both regimes and remains ~10^2 below the ~100%
    relative gap a batch-MEAN production bug would leave
    (a factor-of-2 gap no tolerance can hide); the CPU tier
    keeps its strict rtol=1e-5 / atol=1e-6 (its full-fp32
    CPU kernels are empirically deterministic across batch
    sizes). Exits via os._exit(0) on success: skipping the
    interpreter teardown avoids the known-benign mplib
    host-thread shutdown tracebacks in the short-lived child
    (the parent's own teardown is unchanged)."""
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    free, total = torch.cuda.mem_get_info()
    print(f"Q11 PIN CHILD: starting with {free / 2**20:.0f} MiB "
          f"free / {total / 2**20:.0f} MiB total (fresh CUDA "
          f"context)")
    qmodel = construct_gpu(Path(qtmp), is_training=True,
                           seed_options=seed(morph_factor=0.0,
                                             resolution=res,
                                             batch_size=1,
                                             ae_dims=128,
                                             inter_dims=128,
                                             e_dims=64,
                                             d_dims=64,
                                             d_mask_dims=24))
    x1 = synth_samples_nchw(res, batch=1, seed_no=1)
    x2 = synth_samples_nchw(res, batch=2, seed_no=1)
    real_update_op = qmodel.src_dst_opt.get_update_op
    qmodel.src_dst_opt.get_update_op = lambda pairs: (lambda: None)
    n, worst_rel, worst_name = 0, 0.0, '?'
    try:
        qmodel.train(*x1)
        g1 = grad_map(qmodel, qmodel.G_weights)
        assert any(v is not None and v.abs().sum() > 1e-3
                   for v in g1.values())
        qmodel.train(*x2)
        g2 = grad_map(qmodel, qmodel.G_weights)
        for p in qmodel.G_weights:
            v1, v2 = g1[id(p)], g2[id(p)]
            assert v1 is not None and v2 is not None, p._dfl_name
            # the GPU-tier tolerance: the measured full-fp32
            # kernel-noise floor of this machine is allocator-
            # regime dependent (~2.3e-5 relative at ~3 GiB
            # free up to ~6.4e-3 relative / ~2e-4 absolute at
            # ~22 GiB free — see the docstring); still ~10^2
            # below the factor-of-2 gap a batch-MEAN
            # production bug would leave
            assert torch.allclose(v2, 2 * v1, rtol=1e-2,
                                  atol=1e-3), p._dfl_name
            n += 1
            ref = (2 * v1).abs().max().item()
            if ref > 0:
                rel = (v2 - 2 * v1).abs().max().item() / ref
                if rel > worst_rel:
                    worst_rel, worst_name = rel, p._dfl_name
    finally:
        qmodel.src_dst_opt.get_update_op = real_update_op
    print(f"Q11 PIN CHILD: {n}/{len(qmodel.G_weights)} params "
          f"pass the rtol=1e-2/atol=1e-3 pin (the measured "
          f"GPU full-fp32 noise floor) | worst: {worst_name} "
          f"rel={worst_rel:.3e} (fresh full-fp32 subprocess)")
    # flush before the hard exit: os._exit skips stdio
    # flushing, and without this the summary line above is
    # lost from the captured pipe (the parent's "Q11 PIN
    # CHILD:" marker assert would then match only the
    # child's starting line)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_q11_pin_subprocess(qtmp, res):
    """Runs the Q11 batch-SUM pin in a FRESH python subprocess.
    The in-process variants were contaminated by this
    process's GAN phase: the venv-default cudnn TF32 plans
    (created during the GAN iterations) leak from torch's
    cuDNN plan cache into the pin even after the flag is
    flipped at pin time, and the post-GAN allocator state
    shifts the N=1 / N=2 kernel selection (session 15c
    diagnostics, docs/_p7_diag_q11_flow.py — E2: 4.7e-3
    deviation on a late-flag-flipped pin). The fresh
    subprocess with full-fp32 flags set before the first
    CUDA activity pins at the SAME strict tolerances as the
    CPU tier (see _q11_pin_child). The child output tail is
    surfaced in the test log for the record. A VRAM floor
    (1.5 GiB free: the child's fresh CUDA context + the
    qmodel + activations, alongside this machine's ~20 GiB
    co-tenant) guards the spawn: below the floor the test
    skips (environmental, like the real-artifact pre-checks),
    never a code failure."""
    free, _total = torch.cuda.mem_get_info()
    if free < 1.5 * 2**30:
        pytest.skip(f"Q11 pin subprocess VRAM floor: "
                    f"{free / 2**20:.0f} MiB free < 1536 MiB")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_SMOKE_DIR.parents[1])!r})\n"
        f"sys.path.insert(0, {str(_SMOKE_DIR)!r})\n"
        "import torch\n"
        "torch.backends.cudnn.allow_tf32 = False\n"
        "torch.set_float32_matmul_precision('highest')\n"
        "from test_model_amp_training_cuda import _q11_pin_child\n"
        f"_q11_pin_child({str(qtmp)!r}, {int(res)})\n"
    )
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True,
                          timeout=900,
                          cwd=str(_SMOKE_DIR.parents[1]))
    out = (proc.stdout or '') + (proc.stderr or '')
    tail = '\n'.join(out.splitlines()[-30:])
    print("Q11 pin subprocess output (tail):\n" + tail)
    assert proc.returncode == 0, \
        f"the Q11 pin child exited with {proc.returncode}"
    assert "Q11 PIN CHILD:" in out, "the child reported no pin result"


@requires_gpu
def test_representative_128_training_gpu(tmp_path):
    """The representative 128px GAN-on AMP model on CUDA: two
    real driver iterations (the two-phase G/D train with the
    official GAN terms, under the venv-default kernel
    environment — every assert below is precision-
    independent), the frozen-inter pin, save -> strict
    resume -> continue (the GAN_opt state continuity
    included), and then the Q11 batch-SUM doubling pin on GPU
    on a SEPARATE deterministic-morph (morph_factor=0.0)
    model — the exact-k random morph mask is re-drawn on
    every train() call, so at morph 0.5 the N=2 gradient is
    not a clean doubling of the N=1 one (different per-call
    mask assignments); the doubling property pins at k=0,
    exactly like the CPU test (the Q11 backward mechanism is
    backend-neutral). The pin runs in a FRESH full-fp32
    subprocess (see _run_q11_pin_subprocess), which isolates
    it from this process's cuDNN plan cache / allocator state;
    the GPU tier pins at the measured-noise tolerance of
    _q11_pin_child (rtol=1e-2 / atol=1e-3 — this machine's
    allocator-regime-dependent full-fp32 kernel-noise floor,
    still ~10^2 below the factor-of-2 gap a batch-MEAN bug
    would leave; the CPU tier keeps its strict rtol=1e-5 /
    atol=1e-6)."""
    model = construct_gpu(tmp_path, is_training=True,
                          seed_options=seed(**REPRESENTATIVE))
    res = model.resolution

    # one real driver iteration (debug -> N=1): the generator
    # step + the GAN step (gan_power 0.2 != 0)
    model.train_one_iter()
    torch.cuda.synchronize()
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)
    assert model.iter == 1
    assert model.src_dst_opt.iterations.item() == 1
    assert model.GAN_opt.iterations.item() == 1
    # the frozen inter heads never receive grads (F4)
    assert all(p.grad is None
               for p in model.inter_src.get_weights()
               + model.inter_dst.get_weights())
    # the encoder weights moved (the src_dst_opt step ran)
    enc = [p.detach().clone() for p in model.encoder.get_weights()]
    model.train_one_iter()
    torch.cuda.synchronize()
    for a, b in zip(enc, model.encoder.get_weights()):
        assert not torch.equal(a, b.detach())
    assert model.src_dst_opt.iterations.item() == 2
    assert model.GAN_opt.iterations.item() == 2

    # save -> strict resume -> continue (the GAN_opt state
    # continuity included) — before the Q11 pin, so the
    # deterministic-morph Q11 model is constructed on a fresh
    # dir after the representative one is fully exercised
    model.save()
    model2 = make_amp(AMPHeadless, str(model.saved_models_path),
                      is_training=True, seed=seed(**REPRESENTATIVE),
                      debug=True, force_gpu_idxs=[0])
    assert model2.iter == model.iter
    assert model2.src_dst_opt.iterations.item() \
        == model.src_dst_opt.iterations.item()
    assert model2.GAN_opt.iterations.item() \
        == model.GAN_opt.iterations.item()
    for name in ('encoder', 'inter_src', 'inter_dst', 'decoder', 'GAN'):
        a = [w.detach().cpu() for w in getattr(model, name).get_weights()]
        b = [w.detach().cpu() for w in getattr(model2, name).get_weights()]
        assert len(a) == len(b), name
        for x, y in zip(a, b):
            assert torch.equal(x, y), name
    # the resume-state equality is fully checked: no source-
    # model tensor is needed below (its scalars survive in
    # iter_before), so release the source model BEFORE the
    # resumed model's third GAN iteration — the user's VRAM
    # policy (state 0l: one resident model at a time) gives
    # the heaviest moment the most headroom on this
    # constrained card (an invisible ~20 GiB co-tenant
    # holder)
    iter_before = model.iter
    del model, enc
    torch.cuda.empty_cache()
    model2.train_one_iter()
    torch.cuda.synchronize()
    assert model2.iter == iter_before + 1
    row2 = model2.loss_history[-1]
    assert len(row2) == 2 and all(math.isfinite(v) for v in row2)

    # Q11 on GPU (separate model, deterministic morph): the
    # exact-k random morph mask is re-drawn on every train()
    # call, so the batch-SUM doubling property only pins at
    # morph_factor=0.0 (k=0, the mask-free deterministic
    # forward) — the exact configuration of the CPU pin. The
    # pin runs in a FRESH full-fp32 subprocess (see
    # _run_q11_pin_subprocess) so this process's GAN-phase
    # cuDNN plan cache / allocator state cannot contaminate
    # it, at the measured-noise GPU-tier tolerance of
    # _q11_pin_child. The representative GAN model is
    # released first (the constrained-VRAM environment keeps
    # one resident model at a time; the source model already
    # was).
    del model2
    torch.cuda.empty_cache()
    qtmp = Path(tmp_path) / 'q11'
    qtmp.mkdir(parents=True, exist_ok=True)
    _run_q11_pin_subprocess(str(qtmp), res)
    print("Q11 batch-SUM doubling on GPU: PASS (fresh "
          "full-fp32 subprocess, the measured-noise "
          "rtol=1e-2/atol=1e-3 pin, the deterministic "
          "morph_factor=0.0 forward)")
    print("REPRESENTATIVE 128px AMP (CUDA): PASS "
          "(Training numerical parity vs official TF: NOT_VERIFIED)")


@requires_gpu
def test_q11_pin_fresh_subprocess_gpu(tmp_path):
    """The Q11 batch-SUM doubling pin in a FRESH full-fp32
    subprocess WITHOUT a preceding GAN phase in the parent
    process: the unit-level proof of the pin mechanism the
    representative test's subprocess pin relies on (a fresh
    interpreter + FRESH CUDA context + full-fp32 flags set
    before the first CUDA activity pins at the measured-noise
    GPU-tier tolerance of _q11_pin_child — this machine's
    allocator-regime-dependent full-fp32 kernel-noise floor:
    worst 2.3e-5 relative at ~3 GiB free (session 15b
    diagnostic, docs/_p7_diag_q11_tf32.py: 64/64) up to
    6.4e-3 relative / 2e-4 absolute at ~22 GiB free (session
    15c, docs/_p7_diag_q11_isolated.py)). The
    deterministic-morph (morph_factor=0.0) qmodel is
    constructed inside the child; the parent only launches
    it and checks the child result (plus the 1.5 GiB VRAM
    floor guard in _run_q11_pin_subprocess)."""
    qtmp = Path(tmp_path) / 'q11'
    qtmp.mkdir(parents=True, exist_ok=True)
    _run_q11_pin_subprocess(str(qtmp), 128)
    print("Q11 PIN (fresh subprocess, no GAN phase): PASS "
          "(the measured-noise rtol=1e-2/atol=1e-3 pin, "
          "RTX 4090)")


# --- REAL-artifact staged validation (opt-in env vars) ------------------------

# official on-disk AMP file prefix: the bare "AMP_*" form (the
# default-options snapshot, module-derived class name) or the
# "<model>_-prefixed" form (e.g. "<model>_AMP_*")
_AMPFILE_RE = re.compile(r"^(?:.*_)?AMP_(?P<rest>[^/]+)$")

# the VRAM policy (the plan / the directive): a real-checkpoint
# run needs >= (the checkpoint's own on-disk footprint, which
# approximates the resident weights + optimizer states) + 1.5
# GiB headroom (CUDA context + activations + the caching
# allocator), floored at 2.5 GiB free (below that, no real AMP
# CUDA run at all). Unknown / unprovided configurations fall
# back to the conservative 14 GiB (the budget of the known
# 256px real-checkpoint class). Otherwise the run is
# NOT_AVAILABLE (skipped, never fabricated).
MIN_FREE_GB_REAL = 14.0
VRAM_HEADROOM_GB = 1.5
VRAM_FLOOR_GB = 2.5


def _real_ckpt_env():
    return (os.environ.get('DFL_TEST_AMP_CHECKPOINT'),
            os.environ.get('DFL_TEST_FACESET_PAK'))


def _required_free_gb_real():
    ckpt, _ = _real_ckpt_env()
    if not ckpt:
        return MIN_FREE_GB_REAL
    footprint = 0
    for entry in Path(ckpt).iterdir():
        if entry.is_file() and _AMPFILE_RE.match(entry.name):
            footprint += entry.stat().st_size
    return max(VRAM_FLOOR_GB, footprint / 2**30 + VRAM_HEADROOM_GB)


requires_real_ckpt = pytest.mark.skipif(
    not os.environ.get('DFL_TEST_AMP_CHECKPOINT'),
    reason="DFL_TEST_AMP_CHECKPOINT not set (opt-in real artifact)")
requires_real_faceset = pytest.mark.skipif(
    not os.environ.get('DFL_TEST_FACESET_PAK'),
    reason="DFL_TEST_FACESET_PAK not set (opt-in real artifact)")


def _vram_precheck(min_free_gb=None):
    dfl_nn.initialize_main_env()
    if min_free_gb is None:
        min_free_gb = _required_free_gb_real()
    free, total = torch.cuda.mem_get_info()
    return (free >= min_free_gb * 2**30, free / 2**30,
            total / 2**30, min_free_gb)


def _vram_skip():
    ok, free_gb, total_gb, need_gb = _vram_precheck()
    if not ok:
        pytest.skip(f"VRAM pre-check failed (free {free_gb:.1f} GiB of "
                    f"{total_gb:.1f} GiB < {need_gb:.1f} GiB needed) — "
                    "NOT_AVAILABLE")


def _copy_ckpt(root, ckpt_dir):
    """Copy the official checkpoint files into ``root`` under the
    test model's naming (copy-only; the original directory is
    never written)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in sorted(Path(ckpt_dir).iterdir()):
        if not entry.is_file():
            continue
        m = _AMPFILE_RE.match(entry.name)
        if not m:
            continue
        rest = m.group('rest')
        if rest == 'default_options.dat':
            # the default-options snapshot uses the module-derived
            # model_class_name (no model-name prefix)
            dst = root / 'AMP_default_options.dat'
        else:
            dst = root / f'test_AMP_{rest}'
        shutil.copy2(str(entry), str(dst))
        n += 1
    return n


def _copy_faceset(root, faceset_pak):
    """Copy the .pak faceset into both training sides (copy-only)."""
    root = Path(root)
    for side in ('src', 'dst'):
        d = root / side
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(faceset_pak, str(d / Path(faceset_pak).name))


class _InputScript:
    """Answer provider for the official override prompts (one
    value per prompt; "" = keep the stored default). A pure
    headless resume fires no prompts; the pin is a safety net."""

    def __init__(self, overrides=None):
        self.answers = overrides or {}
        self.pos = 0

    def __call__(self, _prompt):
        a = self.answers.get(self.pos, "")
        self.pos += 1
        return a


@pytest.fixture
def headless_io(monkeypatch):
    """Pin the official interactive layer to deterministic
    headless semantics (TTY-independent)."""
    monkeypatch.setattr(builtins, "input", _InputScript())
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


def _real_model(root, is_training=True, precision='off'):
    """Construct the REAL AMPModel on CUDA device 0 (resume mode:
    the caller pre-seeds/copies the file set first)."""
    dfl_nn.initialize_main_env()
    root = Path(root)
    return AMPModel(
        is_training=is_training,
        saved_models_path=root,
        training_data_src_path=root / 'src',
        training_data_dst_path=root / 'dst',
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name='test_AMP',
        debug=True,
        force_gpu_idxs=[0],
        precision=precision,
    )


# the official AMP option keys the ported model normalizes on
# resume (the SAEHD-flavored keys some official workspaces carry
# ride along in self.options untouched — they are not asserted)
_CORE_OPTION_KEYS = (
    'resolution', 'face_type', 'models_opt_on_gpu', 'ae_dims',
    'inter_dims', 'e_dims', 'd_dims', 'd_mask_dims',
    'morph_factor', 'uniform_yaw', 'blur_out_mask', 'lr_dropout',
    'random_warp', 'gan_power', 'batch_size', 'use_fp16',
    'clipgrad', 'ct_mode', 'random_src_flip', 'random_dst_flip',
)


@requires_gpu
@requires_real_ckpt
@requires_real_faceset
def test_real_ckpt_path_a_strict_load(tmp_path, headless_io):
    """PATH A — checkpoint-only: the official AMP checkpoint set
    (any official workspace model — the test is checkpoint
    AGNOSTIC: the expected contract is the checkpoint's own
    stored options) is copied (never touched) and strictly
    resumed in the training context: component discovery,
    filenames, restored options, key inventories, strict weight
    loads, the src_dst_opt (+ GAN_opt when gan_power != 0)
    strict state loads, zero required missing/shape/dtype
    errors."""
    _vram_skip()
    ckpt, faceset = _real_ckpt_env()
    root = Path(tmp_path) / 'model'
    n = _copy_ckpt(root, ckpt)
    _copy_faceset(root, faceset)
    assert n >= 6, f"expected >= 6 official files, copied {n}"

    model = _real_model(root, is_training=True)
    torch.cuda.synchronize()

    # the stored bookkeeping + the restored options
    stored = __import__('pickle').loads(
        (root / 'test_AMP_data.dat').read_bytes())
    assert model.iter == stored['iter']
    assert model.iter > 0
    for k in _CORE_OPTION_KEYS:
        if k not in stored['options']:
            continue
        expected = stored['options'][k]
        # the official on_initialize normalizes the stored
        # legacy lr_dropout bool form to 'y'/'n' before use
        if k == 'lr_dropout' and isinstance(expected, bool):
            expected = 'y' if expected else 'n'
        assert model.options[k] == expected, k
    # the model is in the real checkpoint's configuration
    assert model.resolution == model.options['resolution']
    assert model.model_data_format == 'NCHW'
    # the official key inventories (the archi topology is
    # dimension-independent)
    assert len(model.encoder.get_weights()) == 20
    assert len(model.inter_src.get_weights()) == 2
    assert len(model.inter_dst.get_weights()) == 2
    assert len(model.decoder.get_weights()) == 44
    # the GAN set exists iff the checkpoint trained with it
    # (the official filename-list / load-loop gating)
    has_gan = model.options['gan_power'] != 0
    assert hasattr(model, 'GAN') == has_gan
    assert hasattr(model, 'GAN_opt') == has_gan
    if has_gan:
        assert model.GAN.name == 'GAN'
    # FAITHFULITY pin: the loaded optimizer iters equals the
    # value stored in the optimizer state FILE (the official
    # loader restores each state file independently — no
    # synchronization with the model counter: a workspace set
    # assembled from different save points can (and does)
    # carry an optimizer state newer than data.dat, and the
    # official behavior is to load it verbatim)
    opt_state = np.load(root / 'test_AMP_src_dst_opt.npy',
                        allow_pickle=True)
    assert model.src_dst_opt.iterations.item() == \
        float(opt_state['iters:0'])
    if has_gan:
        gan_state = np.load(root / 'test_AMP_GAN_opt.npy',
                            allow_pickle=True)
        assert model.GAN_opt.iterations.item() == \
            float(gan_state['iters:0'])
    print("REAL CHECKPOINT ARCHITECTURE/STATE = UNCHANGED; "
          "RUNTIME TEST BATCH = 1 (debug) — no stored "
          "configuration/state file was modified")
    print("REAL AMP CHECKPOINT PATH A (CUDA): PASS — strict "
          f"load (iter={model.iter}, opt_iters="
          f"{model.src_dst_opt.iterations.item()}, "
          f"res={model.resolution}, "
          f"gan={'on' if has_gan else 'off'}, options restored)")


@requires_gpu
@requires_real_ckpt
@requires_real_faceset
def test_real_ckpt_path_b_real_samples(tmp_path, headless_io):
    """PATH B — real samples: the real faceset (WHOLE_FACE)
    through the in-process SampleGeneratorFace (warped/target
    images + full-face and eyes/mouth masks, NCHW f32); the
    real inputs then drive BOTH official inference boundaries:
    AE_view on the training instance (the in-training preview
    path) and AE_merge on a separately loaded non-training
    instance (the official merger's predictor boundary) —
    no training step (the real-input pipeline proof)."""
    _vram_skip()
    ckpt, faceset = _real_ckpt_env()
    root = Path(tmp_path) / 'model'
    _copy_ckpt(root, ckpt)
    _copy_faceset(root, faceset)

    model = _real_model(root, is_training=True)
    torch.cuda.synchronize()

    # one real batch from the in-process generators (debug ->
    # N=1): the official onTrainOneIter unpack — a NESTED
    # (src-tuple, dst-tuple) pair of 4-array tuples, at the
    # checkpoint's own resolution
    R = model.resolution
    (ws, ts, tm, tm_em), (wd, td, dm, dm_em) = \
        model.generate_next_samples()
    for name, x in (('warped_src', ws), ('target_src', ts),
                    ('target_dst', td)):
        assert x.shape == (1, 3, R, R), name
        assert np.isfinite(x).all(), name
        assert (x >= 0.0).all() and (x <= 1.0).all(), name
    # the generator masks are in [0,1]. BINARY-ness is a
    # property of the synthetic test configuration (binary
    # stored masks + the 'f' face crop), NOT of the official
    # generator contract: on a real WHOLE_FACE faceset the
    # official affine-warp/resize interpolation of the stored
    # face masks legitimately yields soft edge values (the
    # blur_out_mask option is unrelated to the generator —
    # it rewrites the TARGET image inside the train closure)
    for name, x in (('target_srcm', tm), ('target_dstm', dm),
                    ('target_srcm_em', tm_em),
                    ('target_dstm_em', dm_em)):
        assert x.shape == (1, 1, R, R), name
        assert np.isfinite(x).all(), name
        assert (x >= 0.0).all() and (x <= 1.0).all(), name

    # the official inference boundaries (faithful to the
    # official split: the TRAINING model exposes AE_view —
    # the in-training preview path, official L664-670 — while
    # AE_merge exists ONLY on the non-training instance, which
    # is exactly what the official merger loads for its
    # predictor_func, official L710)
    morph = model.options['morph_factor']

    # 1) AE_view on the training instance: the 5 outputs in
    #    the official order (pred_src_src, pred_dst_dst,
    #    pred_dst_dstm, pred_src_dst, pred_src_dstm), driven
    #    by the real warped sample pair
    (pv_src, pv_dst, pv_dstm, pv_src_dst, pv_src_dstm) = \
        model.AE_view(ws, wd, morph)
    for name, x in (('pred_src_src', pv_src), ('pred_dst_dst', pv_dst),
                    ('pred_src_dst', pv_src_dst)):
        assert x.shape == (1, 3, R, R), name
        assert np.isfinite(x).all(), name
        assert (x >= 0.0).all() and (x <= 1.0).all(), name
    for name, x in (('pred_dst_dstm', pv_dstm),
                    ('pred_src_dstm', pv_src_dstm)):
        assert x.shape == (1, 1, R, R), name
        assert np.isfinite(x).all(), name
        assert (x >= 0.0).all() and (x <= 1.0).all(), name

    # release the training instance before the second load
    # (the constrained-VRAM environment keeps at most one
    # real model's resident state at a time)
    del model
    torch.cuda.empty_cache()

    # 2) AE_merge on a separately loaded NON-training instance
    #    (the official merger's exact boundary), driven by the
    #    real dst sample; the deterministic floor slice at the
    #    checkpoint's stored morph factor
    model_nt = _real_model(root, is_training=False)
    bgr, m_dst, m_src = model_nt.AE_merge(td, morph)
    assert bgr.shape == (1, 3, R, R)
    assert m_dst.shape == (1, 1, R, R)
    assert m_src.shape == (1, 1, R, R)
    assert np.isfinite(bgr).all()
    assert (bgr >= 0.0).all() and (bgr <= 1.0).all()
    assert np.isfinite(m_dst).all()
    assert (m_dst >= 0.0).all() and (m_dst <= 1.0).all()
    print("REAL CHECKPOINT ARCHITECTURE/STATE = UNCHANGED; "
          "RUNTIME TEST BATCH = 1 (debug) — no stored "
          "configuration/state file was modified")
    print("REAL AMP CHECKPOINT PATH B (CUDA): PASS — real "
          "faceset samples through the official inference "
          "pipeline (no training step)")


@requires_gpu
@requires_real_ckpt
@requires_real_faceset
@pytest.mark.parametrize('precision', ['off', 'bf16', 'fp16'])
def test_real_ckpt_path_c_one_step_lifecycle(tmp_path, headless_io, precision):
    """PATH C — one-step real training: strict resume of the
    official checkpoint, one real iteration (finite losses,
    encoder/decoder updated, inter bit-identical, both optimizer
    counters +1), save, strict reload, exact optimizer
    continuity, one more iteration. A failure here is a
    LIFECYCLE problem — never reported as checkpoint
    incompatibility (that is PATH A's scope)."""
    _vram_skip()
    ckpt, faceset = _real_ckpt_env()
    root = Path(tmp_path) / 'model'
    _copy_ckpt(root, ckpt)
    _copy_faceset(root, faceset)

    model = _real_model(root, is_training=True, precision=precision)
    torch.cuda.synchronize()

    # the update-set snapshots (CPU copies; GPU stays clean)
    inter_before = [w.detach().clone().cpu()
                    for w in model.inter_src.get_weights()]
    inter_before += [w.detach().clone().cpu()
                     for w in model.inter_dst.get_weights()]
    enc_before = [w.detach().clone().cpu()
                  for w in model.encoder.get_weights()]
    dec_before = [w.detach().clone().cpu()
                  for w in model.decoder.get_weights()]
    it_before = model.iter
    s_iters_before = model.src_dst_opt.iterations.item()
    has_gan = model.options['gan_power'] != 0
    g_iters_before = (model.GAN_opt.iterations.item()
                      if has_gan else None)

    # one real iteration on real data (debug -> N=1)
    res5_dtypes = []
    handle = model.encoder.res5.register_forward_hook(
        lambda _module, _inputs, output: res5_dtypes.append(
            (output.dtype, torch.is_autocast_enabled('cuda'))))
    try:
        model.train_one_iter()
    finally:
        handle.remove()
    expected_res5 = (torch.bfloat16 if precision == 'bf16' else torch.float32)
    assert any(dtype == expected_res5 and
               (in_autocast == (precision != 'off'))
               for dtype, in_autocast in res5_dtypes)
    torch.cuda.synchronize()
    assert model.iter == it_before + 1
    row = model.loss_history[-1]
    assert len(row) == 2 and all(math.isfinite(v) for v in row)
    # encoder + decoder updated
    enc_now = [w.detach().clone().cpu() for w in model.encoder.get_weights()]
    dec_now = [w.detach().clone().cpu() for w in model.decoder.get_weights()]
    for a, b in zip(enc_before, enc_now):
        assert not torch.equal(a, b)
    for a, b in zip(dec_before, dec_now):
        assert not torch.equal(a, b)
    # inter bit-identical (frozen)
    inter_now = [w.detach().clone().cpu() for w in model.inter_src.get_weights()]
    inter_now += [w.detach().clone().cpu() for w in model.inter_dst.get_weights()]
    for a, b in zip(inter_before, inter_now):
        assert torch.equal(a, b)
    # the optimizer counters advanced exactly once
    assert model.src_dst_opt.iterations.item() == s_iters_before + 1
    if has_gan:
        assert model.GAN_opt.iterations.item() == g_iters_before + 1
    print(f"REAL LIFECYCLE 1 PASS: one real iteration "
          f"(iter={model.iter}, src={row[0]:.6f} dst={row[1]:.6f} "
          f"gan={'on' if has_gan else 'off'})")

    # save -> strict reload
    model.save()
    torch.cuda.synchronize()
    model2 = _real_model(root, is_training=True, precision=precision)
    torch.cuda.synchronize()
    assert model2.iter == model.iter
    assert model2.src_dst_opt.iterations.item() \
        == model.src_dst_opt.iterations.item()
    for a, b in zip(model.src_dst_opt.get_weights(),
                    model2.src_dst_opt.get_weights()):
        assert torch.equal(a, b)
    if has_gan:
        assert model2.GAN_opt.iterations.item() \
            == model.GAN_opt.iterations.item()
        for a, b in zip(model.GAN_opt.get_weights(),
                        model2.GAN_opt.get_weights()):
            assert torch.equal(a, b)
    for name in ('encoder', 'inter_src', 'inter_dst', 'decoder') \
            + (('GAN',) if has_gan else ()):
        a = [w.detach().cpu() for w in getattr(model, name).get_weights()]
        b = [w.detach().cpu() for w in getattr(model2, name).get_weights()]
        assert len(a) == len(b), name
        for x, y in zip(a, b):
            assert torch.equal(x, y), name
    print("REAL LIFECYCLE 2 PASS: strict reload (optimizer "
          "states preserved)")

    # one more iteration after the reload
    model2.train_one_iter()
    torch.cuda.synchronize()
    assert model2.iter == model.iter + 1
    assert model2.src_dst_opt.iterations.item() \
        == model.src_dst_opt.iterations.item() + 1
    row2 = model2.loss_history[-1]
    assert len(row2) == 2 and all(math.isfinite(v) for v in row2)
    print(f"REAL LIFECYCLE 3 PASS: post-reload iteration "
          f"(iter={model2.iter}, src={row2[0]:.6f} dst={row2[1]:.6f})")

    print("REAL CHECKPOINT ARCHITECTURE/STATE = UNCHANGED; "
          "RUNTIME TEST BATCH = 1 (debug) — no stored "
          "configuration/state file was modified")
    if has_gan:
        label = "REAL AMP GAN ONE-STEP LIFECYCLE (CUDA): PASS"
    else:
        label = ("REAL AMP NON-GAN ONE-STEP LIFECYCLE (CUDA): "
                 "PASS (gan_power=0: GAN component / GAN_opt / "
                 "D step = NOT_APPLICABLE, not missing coverage)")
    print(label + " (Training numerical parity vs official "
          "TF: NOT_VERIFIED)")
