"""Phase 12 Commit 4 acceptance: SAEHD multi-device (multi-replica)
training semantics on the REAL production model.

This suite drives the actual ``Model_SAEHD`` training closures and
lifecycle (through the test-only ``SAEHDHeadless`` option seam, whose
REAL ``on_initialize`` runs unchanged) — no test-only model fakes in
the production flow. It covers the approved Phase 12 plan's B15 CPU
matrix plus the B15 GPU tiers A-D:

CPU tier (N == 1 and SIMULATED_MULTI_REPLICA on CPU):

- N == 1 invariance (df AND liae): the single-replica plan carries no
  components/mirrors; ``train_one_iter`` and direct closure calls run
  the byte-identical single-device path (one canonical src_dst_opt
  step, per-sample loss vectors, unchanged checkpoint/save
  ownership); teardown leaves the model intact
- SIMULATED_MULTI_REPLICA (two LOGICAL replicas on one CPU device via
  the narrow ``ReplicaPlan.from_torch_devices`` abstraction routed
  through the production ``from_device_config`` seam — labeled
  SIMULATED_MULTI_REPLICA; physical multi-GPU stays
  PENDING_ENVIRONMENTALLY / NOT_VERIFIED): full ``train_one_iter``
  for df and liae — the official per-replica loop with equal
  contiguous shards of EVERY tensor, per-replica forward/loss on the
  canonical (r == 0) / mirror (r > 0) modules, the replica-mean
  aggregation, exactly ONE canonical src_dst_opt step, mirror sync
  after the successful step only, finite loss history
- the official batch formula ``b = max(1, floor(B/N))`` with B = the
  batch the generators actually produced: the model-level rows
  (B=7,N=2 -> b=3, effective 6; B=2,N=3 -> b=1, effective 3;
  B=1,N=2 -> b=1, effective 2; B=0,N=2 -> b=1, effective 2) plus the
  closure-level integration (a fetched official effective batch
  splits into exact per-replica shards of size b)
- contiguous shard order (forward hooks on the canonical AND the
  mirror encoder observe replica 0 receiving samples [0..b) and
  replica 1 samples [b..2b)) and the SAME shard indices applied to
  every SAEHD tensor (the per-sample loss values from the multi run
  match, element by element, the model's own single-replica
  per-sample reference on the identical weights — a shard/tensor
  misalignment would break the match on either the src or the dst
  channel)
- loss contract: per-replica per-sample (src, dst) vectors
  CONCATENATED in replica/shard order; the vector length equals the
  (adjusted) global fetched batch; the ``loss_history`` mean is the
  mean of the concatenated vector
- exactly one canonical G step per global iteration; optimizer
  ownership is canonical-only (mirror parameters never enter any of
  src_dst_opt / D_code_opt / D_src_dst_opt state); mirror sync
  happens only after a successful canonical G step (post-step
  bit-exact mirrors)
- option combos: true_face OFF/ON and GAN OFF/ON (and both) select
  the code-D / D_src mirror components and exactly one canonical
  D step each when active; D steps run AFTER the G step, in the
  G -> D-code -> D_src order, on the POST-G weights (the closure
  entry snapshots of the canonical encoder are identical at both D
  entries and differ from the pre-iteration snapshot)
- mirror component coverage for df (encoder, inter, decoder_src,
  decoder_dst, + code_discriminator / D_src when active) and liae
  (encoder, inter_AB, inter_B, decoder, + code_discriminator when
  active)
- checkpoint/Saveable structural isolation: the SIMULATED model's
  ``model_filename_list`` and the saved files are the OFFICIAL
  single-count set (mirrors add no filename, no file, no key); a
  FRESH single-replica training model strict-loads the
  SIMULATED-produced checkpoint with bit-exact canonical components
  (cross-count resume acceptance is Commit 6 — this is the
  structural boundary only)
- preview (AE_view) and merge (AE_merge / get_MergerConfig /
  predictor_func) use the canonical modules only — a deliberately
  desynced mirror is invisible to them
- export surfaces are unchanged by Commit 4 (the ONNX/DFM export
  regression suite covers them in the B17 matrix; structurally the
  diff touches no export code)
- teardown: ``finalize()`` disposes the plan, releases the mirrors,
  and leaves the canonical model fully usable

GPU tier (1x physical GPU, SIMULATED_MULTI_REPLICA: logical replica
0 and 1 BOTH on cuda:0 — no physical multi-GPU claim; physical
2-GPU FP16 acceptance stays EXPERIMENTAL / NOT_VERIFIED):

A. bf16 SIMULATED N==2 full iteration (no scaler; the D steps run
   entirely FP32)
B. fp16 SIMULATED N==2 success: ONE global native torch GradScaler
   (one model-level scaler for the whole run), exactly ONE canonical
   unscale/step/update per attempt, the scale unchanged after a
   successful non-overflow attempt
C. fp16 Class B — a later replica's SCALED gradients overflow:
   the attempt is SKIPPED (no canonical G update, no iteration
   increment, no mirror sync), ONE scaler update with backoff,
   ALL canonical + mirror grads cleaned, the SAME fetched samples
   are retried, the retry succeeds, and each D step runs exactly
   ONCE — after the successful G step, not once per failed retry
D. fp16 Class A — a nonfinite forward/loss in a later replica: a
   HARD ``FloatingPointError`` (never a ``SkippedGeneratorStep``,
   never retried, never scaler-recorded), no canonical G update, no
   D steps, no scaler update, grads cleared, no mirror sync

The Class A/B failure INJECTION wraps the model's ``_mp_backward``
hook (the production hook the closures already call) to inject ONLY
the failure condition; no algorithm is re-implemented by the test.

Provenance: INDEPENDENT_REIMPLEMENTATION — the official DFL multi-GPU
semantics (per-GPU loss, per-GPU gradients, official replica mean,
one canonical update op, D steps after the G step) are the semantic
authority; no external code was copied.
"""

import builtins
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

import core.leras.models  # noqa: F401,E402  (binds nn.ModelBase)
from core.leras import nn as dfl_nn  # noqa: E402
from core.leras.multidevice import ReplicaPlan  # noqa: E402

from Model_SAEHDTest.Model import (  # noqa: E402
    DEFAULT_SEED_OPTIONS,
    SAEHDHeadless,
    make_model as make_saehd,
    make_training_dirs,
)
from test_model_saehd_training import (  # noqa: E402
    synth_samples,
    tensors8,
    forward_ae,
    copy_components,
)

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Phase 12 GPU test: CUDA device required",
)

CPU = torch.device("cpu")
CUDA = torch.device("cuda:0")

# the native GradScaler default start scale (2**16) and its one-
# backoff (x0.5) result
FP16_INIT_SCALE = 2.0 ** 16
FP16_HALF_SCALE = 2.0 ** 15

MODEL_NAME = "test_SAEHD"

_GPU_MAIN_ENV_DONE = False

# the ORIGINAL production plan factory captured at import time
# (before any test patch): the fresh-loader test restores it so the
# resume runs the production device-config routing instead of the
# still-active SIMULATED_MULTI_REPLICA patch (the monkeypatch
# fixture undoes its patches only at test END, after the body)
_REAL_FROM_DEVICE_CONFIG = dfl_nn.ReplicaPlan.__dict__.get(
    "from_device_config")


def _ensure_gpu_main_env():
    """Publish the Phase 2 device main environment once per process
    (required by DeviceConfig.GPUIndexes); CPU-only tests never
    touch the device list."""
    global _GPU_MAIN_ENV_DONE
    if not _GPU_MAIN_ENV_DONE:
        dfl_nn.initialize_main_env()
        _GPU_MAIN_ENV_DONE = True


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    """Every test runs with a CPU + NCHW foundation unless it
    initializes something else explicitly (the GPU tests switch to
    the primary-GPU config through the model constructor)."""
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")


@pytest.fixture
def headless_io(monkeypatch):
    """Pin the official interactive layer to deterministic headless
    semantics (the real SAEHD resume constructor never prompts). The
    model imports the interact singleton exactly like production
    (``from core.interact import interact as io``).

    Documented exception to the approved patch-seam set (Subagent B
    finding, Phase 12 Commit 4): ``input_in_time`` is the
    production interact singleton's prompt-timing predicate, and
    patching it (with ``builtins.input``) affects ONLY the
    interactive prompt layer — no model, closure, or optimizer code.
    This is the same sanctioned headless pattern the reference
    suites use (test_model_saehd.py / test_model_saehd_training.py
    import the interact singleton identically); the production
    on_initialize_options body still runs unmodified."""
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    from core.interact import interact as io
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


# ---------------------------------------------------------------------------
# seeds / construction
# ---------------------------------------------------------------------------

# the tiny CPU integration configuration: the smallest practical
# SAEHD (64px) — all option keys explicit so no seed key can drift
# across tests
CPU_SEED = dict(
    resolution=64,
    batch_size=2,
    face_type="f",
    models_opt_on_gpu=True,
    ae_dims=16,
    e_dims=8,
    d_dims=8,
    d_mask_dims=8,
    masked_training=True,
    eyes_mouth_prio=False,
    uniform_yaw=False,
    blur_out_mask=False,
    adabelief=True,
    lr_dropout="n",
    random_warp=False,
    random_hsv_power=0.0,
    true_face_power=0.0,
    face_style_power=0.0,
    bg_style_power=0.0,
    ct_mode="none",
    clipgrad=False,
    pretrain=False,
    gan_power=0.0,
    gan_patch_size=16,
    gan_dims=8,
    random_src_flip=False,
    random_dst_flip=False,
)


def cpu_seed(archi="df", **overrides):
    s = dict(DEFAULT_SEED_OPTIONS)
    s.update(CPU_SEED)
    s["archi"] = archi
    s.update(overrides)
    return s


def gpu_seed(archi="df", **overrides):
    """The tiny GPU configuration (bf16/fp16 tiers) — df only: the
    D-step GPU coverage pairs with the df archi's code-D/D_src
    chain."""
    s = cpu_seed(archi, **overrides)
    return s


def patch_plan(monkeypatch, dev, n):
    """Route the production ``ReplicaPlan.from_device_config`` seam
    to the narrow Level-B ``from_torch_devices`` abstraction: n
    LOGICAL replicas on ONE physical device (SIMULATED_MULTI_REPLICA).
    The production ``on_initialize`` then wires the real components
    and mirrors through the unchanged factory path."""
    devices = [dev] * n

    def fake_from_device_config(cls, cfg):
        return ReplicaPlan.from_torch_devices(dev, devices)

    monkeypatch.setattr(
        dfl_nn.ReplicaPlan, "from_device_config",
        classmethod(fake_from_device_config))


def build_cpu(tmp, monkeypatch, n=1, **seed_kwargs):
    """A real training SAEHD on CPU (the in-process debug generator);
    n > 1 routes the production plan seam to the n-logical-replica
    SIMULATED_MULTI_REPLICA plan before on_initialize runs."""
    root = Path(tmp) / f"m{n}"
    if n > 1:
        patch_plan(monkeypatch, CPU, n)
    make_training_dirs(root)
    seed = cpu_seed(**seed_kwargs)
    model = make_saehd(SAEHDHeadless, root, is_training=True, seed=seed,
                       debug=True, cpu_only=True)
    return model


def build_gpu(tmp, monkeypatch, cls, n=2, precision="off", **seed_kwargs):
    """A real training SAEHD on the single physical GPU (logical
    replica 0 and 1 both on cuda:0 when n > 1 —
    SIMULATED_MULTI_REPLICA)."""
    _ensure_gpu_main_env()
    root = Path(tmp) / f"m{n}_{precision}"
    if n > 1:
        patch_plan(monkeypatch, CUDA, n)
    make_training_dirs(root)
    seed = gpu_seed(**seed_kwargs)
    model = make_saehd(cls, root, is_training=True, seed=seed,
                       debug=True, force_gpu_idxs=[0], precision=precision)
    return model


def resume_real(root, headless=False):
    """Construct the REAL (non-headless) SAEHDModel in resume mode
    from a saved directory — the fresh single-replica loader."""
    from models.Model_SAEHD import Model as SAEHDModel
    return SAEHDModel(
        is_training=True,
        saved_models_path=Path(root),
        training_data_src_path=Path(root) / "src",
        training_data_dst_path=Path(root) / "dst",
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name=MODEL_NAME,
        debug=True,
        cpu_only=True,
    )


def stub_fetch(model, a8):
    """Replace the sample fetch with a fixed synthetic batch (the
    Level-B controlled fetch: production's on_initialize adjustment
    guarantees fetched == N*b in real runs; the stub gives the tests
    an exact controlled shard size). ``onTrainOneIter`` fetches
    through ``generate_next_samples`` on EVERY iteration, so the
    stubbed instance attribute drives ``train_one_iter`` directly
    (the construction-time ``last_sample`` is never consumed by the
    SAEHD iteration flow)."""
    def fetch():
        return ((a8[0], a8[1], a8[2], a8[3]),
                (a8[4], a8[5], a8[6], a8[7]))
    model.generate_next_samples = fetch
    return model


# ---------------------------------------------------------------------------
# marked samples (contiguous-shard / shard-consistency proofs)
# ---------------------------------------------------------------------------

def _wave(res, freq, phase):
    y, x = np.mgrid[0:res, 0:res].astype(np.float32)
    return (0.5 + 0.5 * np.sin(
        2.0 * np.pi * freq * (x + 0.6 * y) / res + phase
    )).astype(np.float32)


def marked8(res, n, em_flags):
    """n MARKED samples in the official 8-array order (NHWC):

    - sample i carries its own wave frequency (1 + 2i) on both sides
      — spatial structure keeps every BatchNorm alive and makes each
      sample's per-sample loss value distinct;
    - the src side has channel gains (1.0, 0.6, 0.3), the dst side
      (0.3, 0.6, 1.0) — a per-side channel signature that a forward
      hook decodes unambiguously (side = argmax channel mean);
    - within a side, sample i's base level is 0.05 + 0.2*i, so the
      input mean (level + ~0.15) identifies the sample index;
    - the full-face mask is on for every sample; the eyes/mouth mask
      is present iff ``em_flags[i]`` (so with eyes_mouth_prio the
      300-term fires on exactly the flagged samples).
    """
    src_levels = [0.05 + 0.2 * i for i in range(n)]
    dst_levels = [0.05 + 0.2 * i for i in range(n)]
    # the eyes/mouth mask is a per-sample 3D array — np.stack over
    # it yields the official 4D batch layout (a 4D element would
    # stack to 5D and leave an extra reduction dim in the
    # eyes_mouth_prio term); the full-face mask is the 4D
    # single-sample form because np.repeat along axis 0 tiles it
    # (np.repeat on a 3D array REPEATS ROWS instead of tiling —
    # both stacks yield (n, res, res, 1) with these forms)
    em = np.ones((res, res, 1), np.float32)
    full = np.ones((1, res, res, 1), np.float32)

    def side_arrays(levels, gains, phases):
        warped = np.empty((n, res, res, 3), np.float32)
        target = np.empty((n, res, res, 3), np.float32)
        for i in range(n):
            f = 1.0 + 2.0 * i
            for c in range(3):
                warped[i, :, :, c] = np.clip(
                    levels[i] + 0.30 * gains[c] * _wave(res, f, phases[c]),
                    0.0, 1.0)
                target[i, :, :, c] = np.clip(
                    levels[i] + 0.05 + 0.30 * gains[c] *
                    _wave(res, f, phases[c] + 0.55),
                    0.0, 1.0)
        return warped, target

    warped_src, target_src = side_arrays(
        src_levels, (1.0, 0.6, 0.3), (0.0, 0.9, 1.7))
    warped_dst, target_dst = side_arrays(
        dst_levels, (0.3, 0.6, 1.0), (0.4, 1.3, 2.1))

    em_stack = np.stack(
        [em if flag else np.zeros_like(em) for flag in em_flags], axis=0)
    full_stack = np.repeat(full, n, 0)
    return (warped_src, target_src, full_stack, em_stack,
            warped_dst, target_dst, full_stack, em_stack)


def decode_side_sample(x, n, data_format="NHWC"):
    """Decode (side, sample_index) of a marked sample's input from
    its channel means (side = argmax channel; sample = nearest
    per-side level)."""
    if data_format == "NCHW":
        means = [float(x[c].mean()) for c in range(3)]
    else:
        means = [float(x[..., c].mean()) for c in range(3)]
    side = "src" if means[0] >= means[2] else "dst"
    base = means[0] if side == "src" else means[2]
    # the wave adds ~+0.15 * gain to the mean; the per-side level
    # spacing is 0.2, far above the decode jitter
    idx = int(round((base - 0.20) / 0.2))
    return side, max(0, min(n - 1, idx))


# ---------------------------------------------------------------------------
# state snapshot / bit-exact comparison helpers
# ---------------------------------------------------------------------------

def _module_tensors(mod):
    return list(mod.parameters()) + list(mod.buffers())


def snapshot_state(model):
    """Bit-exact snapshot of every plan component's canonical
    parameters/buffers AND every mirror's — the restorable pre-step
    state of the whole replica set."""
    plan = model.replica_plan
    mods = []
    if plan is not None and plan.is_multi:
        for c in plan.components:
            mods.append(c.canonical)
            mods.extend(c.mirrors)
    else:
        for c in (model.encoder, model.inter, model.decoder_src,
                  model.decoder_dst) if 'df' in model.archi_type else \
            (model.encoder, model.inter_AB, model.inter_B,
             model.decoder):
            mods.append(c)
        if model.options["true_face_power"] != 0:
            mods.append(model.code_discriminator)
        if model.options["gan_power"] != 0:
            mods.append(model.D_src)
    return [(m, {id(t): t.detach().clone() for t in _module_tensors(m)})
            for m in mods]


def restore_state(model, snap):
    # under no_grad: the restorations write through the
    # requires-grad LEAF parameters (in-place on a tracked leaf
    # would otherwise raise) — the snapshot values are detached
    # clones, so no autograd bookkeeping is needed for a restore
    with torch.no_grad():
        for mod, snaps in snap:
            for t in _module_tensors(mod):
                if id(t) in snaps:
                    t.copy_(snaps[id(t)])


def modules_bit_equal(a, b):
    """Parameters AND buffers bit-identical between two modules."""
    ta = [t.detach().cpu() for t in _module_tensors(a)]
    tb = [t.detach().cpu() for t in _module_tensors(b)]
    if len(ta) != len(tb):
        return False
    return all(torch.equal(x, y) for x, y in zip(ta, tb))


def _configs_equal(a, b):
    """Element-wise MergerConfig equality. The config's first
    element is the BOUND ``predictor_func`` method — bound methods
    never compare equal across instances, so that slot is compared
    by the identity of the underlying function (both twins expose
    the class-level ``predictor_func``); the size tuple and the
    MergerConfigMasked compare with ``==`` (the class implements
    the official ``__eq__``)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if callable(x) and callable(y):
            if not (type(x) is type(y)
                    and x.__func__ is y.__func__):
                return False
        elif x != y:
            return False
    return True


def assert_mirrors_synced(model):
    """Every mirror of every component is bit-exact with its
    canonical (parameters AND class-A buffers)."""
    plan = model.replica_plan
    assert plan is not None and plan.is_multi
    for c in plan.components:
        assert c.canonical_params, c.canonical.name
        for r in range(1, plan.num_replicas):
            mirror = c.mirrors[r - 1]
            assert modules_bit_equal(c.canonical, mirror), \
                f"{c.canonical.name} mirror (replica {r}) not synced"


def canonical_name_map(model, plan):
    """{canonical.name: ComponentMirrorSet} for the plan."""
    return {c.canonical.name: c for c in plan.components}


def assert_optimizer_ownership(model, plan, opt_name, comp_name):
    """The optimizer tracks the CANONICAL parameters of the component
    and NONE of its mirrors (mirror params are never optimizer state)."""
    opt = getattr(model, opt_name)
    keys = opt._weight_keys
    c = next(c for c in plan.components
             if c.canonical is getattr(model, comp_name))
    for p in c.canonical_params:
        assert id(p) in keys, \
            f"{opt_name}: canonical param of {comp_name} untracked"
    for r in range(1, plan.num_replicas):
        for mp in c.mirror_params[r]:
            assert id(mp) not in keys, \
                f"{opt_name}: mirror (replica {r}) of {comp_name} " \
                f"must not be optimizer state"


def assert_no_mirror_in_any_optimizer(model, plan):
    """No mirror parameter of ANY component is tracked by ANY of the
    model's optimizers."""
    mirror_ids = set()
    for c in plan.components:
        for r in range(1, plan.num_replicas):
            for mp in c.mirror_params[r]:
                mirror_ids.add(id(mp))
    for opt_name in ("src_dst_opt", "D_code_opt", "D_src_dst_opt"):
        opt = getattr(model, opt_name, None)
        if opt is None:
            continue
        assert not mirror_ids.intersection(opt._weight_keys), \
            f"{opt_name} owns mirror parameters"


def component_names(model):
    plan = model.replica_plan
    if plan is None or not plan.is_multi:
        return set()
    return {c.canonical.name for c in plan.components}


def plan_component_names(model):
    plan = model.replica_plan
    assert plan is not None
    if not plan.is_multi:
        return set()
    return {c.canonical.name for c in plan.components}


# ---------------------------------------------------------------------------
# the GPU-tier counting/injection subclass (wraps ONLY the production
# _mp_* hooks; the closures and the algorithm are untouched)
# ---------------------------------------------------------------------------

class _HookedSAEHD(SAEHDHeadless):
    """SAEHDHeadless + counters around the REAL ModelBase ``_mp_*``
    hooks (delegation via super() — the production hook bodies run
    unchanged) and an optional Class B injection: the ``inject`` knob
    multiplies the LATER replica's (the 2nd ``_mp_backward`` call)
    loss of the FIRST attempt by 1e30 — the scaled gradients
    overflow, the native GradScaler's found_inf check fires, the
    attempt is skipped and retried on the same samples. The test
    injects ONLY the failure condition through the hook the
    production closures already call."""

    def _init_hook_state(self):
        self._hook_counts = {"backward": 0, "unscale": 0, "step": 0,
                             "update": 0, "stepped": []}
        self._failed_attempt_cleanup = None
        self.inject_class_b = False

    def _mp_backward(self, loss_vec):
        self._hook_counts["backward"] += 1
        if (self.inject_class_b and self._hook_counts["backward"] == 2
                and self._hook_counts["step"] == 0):
            # the later replica of the FIRST attempt: force a
            # scaled-gradient overflow
            loss_vec = loss_vec * 1e30
        if self.inject_class_b and self._hook_counts["backward"] == 3:
            # the RETRY's first backward (BEFORE its own backwards
            # run): the failed attempt's cleanup — the production
            # clear_replica_grads, which runs AFTER the failed
            # attempt's scaler update and BEFORE the retry — must
            # have left ALL canonical + mirror grads cleared
            plan = self.replica_plan
            grads = []
            if plan is not None and plan.is_multi:
                for c in plan.components:
                    for t in _module_tensors(c.canonical):
                        grads.append(t.grad is None)
                    for r in range(1, plan.num_replicas):
                        for t in _module_tensors(c.mirrors[r - 1]):
                            grads.append(t.grad is None)
            self._failed_attempt_cleanup = (
                all(grads) if grads else False)
        return super()._mp_backward(loss_vec)

    def _mp_unscale_opt(self, opt, active_weights=None):
        self._hook_counts["unscale"] += 1
        return super()._mp_unscale_opt(opt, active_weights)

    def _mp_opt_step(self, opt, grads_vars):
        r = super()._mp_opt_step(opt, grads_vars)
        self._hook_counts["step"] += 1
        self._hook_counts["stepped"].append(r)
        return r

    def _mp_scaler_update(self):
        # NOTE: the Class B grad cleanup is NOT captured here — the
        # production pipeline is unscale -> step -> update ->
        # clear_replica_grads, so at the moment this hook runs the
        # failed attempt's grads are still set (they are cleared
        # immediately after the update, before the retry). The
        # cleanup is observed at the retry's first backward instead
        # (see _mp_backward).
        self._hook_counts["update"] += 1
        return super()._mp_scaler_update()


# ModelBase derives ``model_class_name`` from the CLASS MODULE's
# file parent-directory name (``rsplit("_", 1)[1]``): this test
# module lives in the 'smoke' directory (no '_' in the name ->
# IndexError at construction), so the counting subclass is bound to
# a synthetic module whose __file__ points into the VEHICLE
# package directory (Model_SAEHDTest) — the exact derivation the
# vehicle's own SAEHDHeadless undergoes. inspect.getmodule
# resolves the class through this __module__, so no production
# code path is altered and no new file is added.
_hooked_gpu_module = types.ModuleType("_saehd_hooked_gpu_module")
_hooked_gpu_module.__file__ = str(
    Path(__file__).resolve().parent / "Model_SAEHDTest"
    / "hooked_gpu.py")
sys.modules["_saehd_hooked_gpu_module"] = _hooked_gpu_module
_HookedSAEHD.__module__ = "_saehd_hooked_gpu_module"


class _OrderHookSAEHD(SAEHDHeadless):
    """SAEHDHeadless + INITIALIZATION-ORDER recording for the
    §10.1.1 binding-order test. The overrides wrap ONLY the
    production entry points the lifecycle itself calls (each
    delegates to super(), so the production bodies run unchanged):
    the sample-generator completion point
    (``set_training_data_generators``) and the ONE mixed-precision
    resolution (``_mp_ensure_resolved`` — the same one-shot method
    production on_initialize triggers early for N > 1 and the lazy
    training lifecycle calls later). The remaining events
    (plan_created, mirror_built, initial_sync, first_forward) are
    recorded by the test's monkeypatched delegating wrappers around
    the production plan factory, plan methods, and train closure.
    ``shared_events`` (set by the test before construction) is the
    one event list the whole construction + first iteration is
    recorded into."""

    shared_events = None

    def __init__(self, *args, **kwargs):
        self._order_events = (
            type(self).shared_events
            if type(self).shared_events is not None else [])
        super().__init__(*args, **kwargs)

    def set_training_data_generators(self, generator_list):
        r = super().set_training_data_generators(generator_list)
        self._order_events.append("generators_done")
        return r

    def _mp_ensure_resolved(self):
        r = super()._mp_ensure_resolved()
        if (self._mp_plan is not None
                and "precision_resolved" not in self._order_events):
            self._order_events.append("precision_resolved")
        return r


# the same ModelBase model_class_name binding as _HookedSAEHD (the
# class module's parent directory must parse as
# '<prefix>_<suffix>'): a second synthetic module inside the
# vehicle package
_order_hook_module = types.ModuleType("_saehd_order_hook_module")
_order_hook_module.__file__ = str(
    Path(__file__).resolve().parent / "Model_SAEHDTest"
    / "order_hook.py")
sys.modules["_saehd_order_hook_module"] = _order_hook_module
_OrderHookSAEHD.__module__ = "_saehd_order_hook_module"


# ---------------------------------------------------------------------------
# CPU tier — N == 1 invariance
# ---------------------------------------------------------------------------

def test_n1_single_replica_invariance_df_cpu(tmp_path, monkeypatch):
    """B15-1: the N == 1 (df) production run is the unchanged
    single-device path: no components/mirrors, one canonical step,
    per-sample loss vectors, unchanged checkpoint ownership, clean
    teardown."""
    m = build_cpu(tmp_path, monkeypatch, archi="df")
    plan = m.replica_plan
    assert plan is not None and not plan.is_multi
    assert plan.components == []          # no mirrors, ever, at N == 1

    # one real training iteration (the lifecycle's own generator)
    it, dt = m.train_one_iter()
    assert it == 1
    assert int(m.src_dst_opt.iterations.item()) == 1
    losses = m.loss_history[-1]
    assert len(losses) == 2 and all(math.isfinite(v) for v in losses)

    # a direct closure call on a 2-sample fetch: the per-sample
    # vectors (the single-replica path returns them as-is)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=7)
    src_vec, dst_vec = m._src_dst_train(*a8)
    assert src_vec.shape == (2,) and dst_vec.shape == (2,)
    assert bool(torch.isfinite(src_vec).all())
    assert bool(torch.isfinite(dst_vec).all())
    # the G step of the closure call: exactly one more iteration
    assert int(m.src_dst_opt.iterations.item()) == 2

    # checkpoint ownership: the official single-count filename set
    names = [f for _, f in m.model_filename_list]
    assert set(names) == {"encoder.npy", "inter.npy", "decoder_src.npy",
                          "decoder_dst.npy", "src_dst_opt.npy"}

    # teardown: the plan is disposed and released; the canonical
    # model stays fully usable (inference boundary on the canonical
    # modules — CPU model, CPU tensors)
    m.finalize()
    assert m.replica_plan is None
    a8 = tensors8(synth_samples(
        64, batch=1, data_format=m.model_data_format, seed_no=8))
    ws, wd = a8[0], a8[4]
    with torch.no_grad():
        preds = forward_ae(m, ws, wd)
    assert preds is not None


def test_n1_single_replica_invariance_liae_cpu(tmp_path, monkeypatch):
    """B15-2: the N == 1 (liae) production run is the unchanged
    single-device path (same contract as the df tier)."""
    m = build_cpu(tmp_path, monkeypatch, archi="liae-ud")
    plan = m.replica_plan
    assert plan is not None and not plan.is_multi
    assert plan.components == []

    it, _ = m.train_one_iter()
    assert it == 1
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert all(math.isfinite(v) for v in m.loss_history[-1])

    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=9)
    src_vec, dst_vec = m._src_dst_train(*a8)
    assert src_vec.shape == (2,) and dst_vec.shape == (2,)

    names = [f for _, f in m.model_filename_list]
    assert set(names) == {"encoder.npy", "inter_AB.npy", "inter_B.npy",
                          "decoder.npy", "src_dst_opt.npy"}

    m.finalize()
    assert m.replica_plan is None


# ---------------------------------------------------------------------------
# CPU tier — SIMULATED_MULTI_REPLICA full iterations
# ---------------------------------------------------------------------------

def test_sim_n2_df_full_iteration_cpu(tmp_path, monkeypatch):
    """B15-3: SIMULATED_MULTI_REPLICA N == 2 (df) full
    ``train_one_iter`` — the production on_initialize wires the real
    components/mirrors through the (Level-B-routed) plan seam; the
    official per-replica loop runs with the stubbed 2-sample fetch
    (b == 1 per replica); one canonical G step; bit-exact post-step
    mirrors; canonical-only optimizer ownership."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    plan = m.replica_plan
    assert plan.is_multi and plan.num_replicas == 2
    assert component_names(m) == {"encoder", "inter", "decoder_src",
                                  "decoder_dst"}
    for c in plan.components:
        assert len(c.mirrors) == 1

    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=1)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1
    # exactly ONE canonical src_dst_opt step (not one per replica)
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert all(math.isfinite(v) for v in m.loss_history[-1])
    # the mirrors are bit-exact with the canonical after the
    # successful step
    assert_mirrors_synced(m)
    # optimizer ownership: canonical params tracked, mirrors never
    assert_optimizer_ownership(m, plan, "src_dst_opt", "encoder")
    assert_optimizer_ownership(m, plan, "src_dst_opt", "inter")
    assert_optimizer_ownership(m, plan, "src_dst_opt", "decoder_src")
    assert_optimizer_ownership(m, plan, "src_dst_opt", "decoder_dst")
    assert_no_mirror_in_any_optimizer(m, plan)
    m.finalize()
    assert m.replica_plan is None


def test_sim_n2_liae_full_iteration_cpu(tmp_path, monkeypatch):
    """B15-4: SIMULATED_MULTI_REPLICA N == 2 (liae) full
    ``train_one_iter`` (the liae component set: encoder, inter_AB,
    inter_B, decoder)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="liae-ud")
    plan = m.replica_plan
    assert plan.is_multi and plan.num_replicas == 2
    assert component_names(m) == {"encoder", "inter_AB", "inter_B",
                                  "decoder"}

    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=2)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert all(math.isfinite(v) for v in m.loss_history[-1])
    assert_mirrors_synced(m)
    assert_no_mirror_in_any_optimizer(m, plan)
    m.finalize()
    assert m.replica_plan is None


def test_official_batch_formula_cpu(tmp_path, monkeypatch):
    """B15-5: the official per-replica size formula b = max(1,
    floor(B/N)) with B = the fetched batch.

    MODEL level: the UNCHANGED official on_initialize block
    (gpu_count = max(1, len(devices)); bs_per_gpu = max(1,
    batch_size // gpu_count); set_batch_size(gpu_count * bs_per_gpu))
    is EXECUTED in the production builds below. The ModelBase
    lifecycle assigns self.batch_size BEFORE on_initialize
    (from the saved data.dat — default 1 on a first run), and the
    vehicle's headless seed updates self.options, not self.batch_size,
    so the block executes on the LIFECYCLE value and writes its
    result back to self.options. On this single-GPU host the
    selected device set is always one device (the SIMULATED seam
    patches the plan routing, never the DeviceConfig), so
    gpu_count == 1 and the executed arithmetic is the identity —
    asserted for both the plain build and the SIMULATED N == 2 seam
    build (which additionally proves the seam does not leak into
    device selection). The B == 0 guard and the multi-device
    (gpu_count > 1) division are pinned by the arithmetic formula
    table and are PENDING_ENVIRONMENTALLY in production execution
    on this host (one physical GPU; a saved batch_size of 0 is
    un-savable because the block normalizes it).

    CLOSURE level: feeds the official effective batch (N*b) through
    the real multi closure and asserts exact per-replica shards of
    size b."""
    # the official formula table: pins the official arithmetic
    # (production execution of the gpu_count > 1 branch is
    # PENDING_ENVIRONMENTALLY on this single-GPU host)
    for B, N, (b, effective) in ((7, 2, (3, 6)), (2, 3, (1, 3)),
                                 (1, 2, (1, 2)), (0, 2, (1, 2))):
        assert max(1, B // N) == b
        assert N * b == effective

    # model level: PRODUCTION execution of the official on_initialize
    # block — the identity at gpu_count == 1, executed on the
    # lifecycle batch value (default 1 on a first run; the headless
    # seed updates self.options only) and written back to
    # self.options by the lifecycle
    m1 = build_cpu(tmp_path / "b1", monkeypatch, n=1)
    assert m1.get_batch_size() == m1.options["batch_size"] == 1
    # the SIMULATED N == 2 seam build: the block still executes with
    # gpu_count == 1 (the seam routes the plan only — device
    # selection is untouched) and the write-back holds
    m2 = build_cpu(tmp_path / "b2", monkeypatch, n=2, batch_size=5)
    assert m2.get_batch_size() == m2.options["batch_size"] == 1
    m1.finalize()
    m2.finalize()
    assert m1.replica_plan is None
    assert m2.replica_plan is None

    # closure level: fetch the official effective batch (B=7, N=2 ->
    # 6) through the SIMULATED N == 2 production closures
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    a8 = synth_samples(64, batch=6, data_format=m.model_data_format,
                       seed_no=3)
    src_vec, dst_vec = m._src_dst_train(*a8)
    # b = max(1, 6 // 2) = 3 per replica; the concatenated vectors
    # span the whole fetched effective batch
    assert src_vec.shape == (6,) and dst_vec.shape == (6,)
    assert bool(torch.isfinite(src_vec).all())
    assert bool(torch.isfinite(dst_vec).all())
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)

    # and b == 1 rows: B=1 -> effective 2 (N=2): the closure splits a
    # 2-sample fetch into two 1-sample shards
    m2 = build_cpu(tmp_path / "m2x", monkeypatch, n=2, archi="df")
    a2 = synth_samples(64, batch=2, data_format=m2.model_data_format,
                       seed_no=4)
    src2, dst2 = m2._src_dst_train(*a2)
    assert src2.shape == (2,) and dst2.shape == (2,)


def test_contiguous_shard_order_cpu(tmp_path, monkeypatch):
    """B15-6: EQUAL CONTIGUOUS shards in sample order — replica 0
    receives samples [0..b), replica 1 receives [b..2b): forward
    hooks on the canonical encoder (replica 0) and the mirror
    encoder (replica 1) decode the marked sample identity of EVERY
    SAMPLE ROW of each forward input (the encoder runs once per
    side per replica over the whole b-sample shard); the decoded
    GLOBAL index (from the per-sample level marking) proves which
    samples landed in which replica, in call order (src side
    forward first, then dst side)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    a8 = marked8(64, 4, em_flags=[False, True, False, True])
    b = 4 // 2  # = 2 per replica

    seen = {"canon": [], "mirror": []}
    enc_comp = next(c for c in m.replica_plan.components
                    if c.canonical is m.encoder)

    # torch 2.14 invokes forward hooks with (module, args, kwargs);
    # the encoder runs ONCE PER SIDE per replica over the whole
    # b-sample shard, so the marked identity is decoded PER SAMPLE
    # ROW (a per-tensor decode would average across the shard's
    # samples and could not separate them)
    def _decode_shard(x, sink):
        for i in range(x.shape[0]):
            sink.append(decode_side_sample(
                x[i], 4, m.model_data_format))

    def canon_hook(mod, inp, *rest):
        _decode_shard(inp[0], seen["canon"])
    def mirror_hook(mod, inp, *rest):
        _decode_shard(inp[0], seen["mirror"])

    m.encoder.register_forward_hook(canon_hook)
    enc_comp.mirrors[0].register_forward_hook(mirror_hook)

    src_vec, dst_vec = m._src_dst_train(*a8)
    assert src_vec.shape == (4,)

    # replica 0's shards carry global samples 0,1 (both sides, src
    # forward first); replica 1's shards carry samples 2,3 — the
    # contiguous, sample-order sharding, proven by the DECODED
    # sample identity, not by the in-shard position
    assert seen["canon"] == [("src", 0), ("src", 1),
                             ("dst", 0), ("dst", 1)]
    assert seen["mirror"] == [("src", 2), ("src", 3),
                              ("dst", 2), ("dst", 3)]


def test_same_shard_all_tensors_and_concat_order_cpu(tmp_path, monkeypatch):
    """B15-7/8/9: the SAME shard indices apply to every SAEHD tensor
    and the per-replica per-sample loss vectors are concatenated in
    replica/shard order. Proof: the per-sample loss values of the
    SIMULATED N == 2 run (b == 1 per replica) match, element by
    element, the model's own single-replica per-sample reference on
    the IDENTICAL weights (the runtime plan dispatch is temporarily
    disabled for the reference — the closures, weights and buffers
    are the same objects). A misshard of ANY input tensor (src or
    dst image, mask or em-mask) would shift one channel's per-sample
    value away from its reference."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  eyes_mouth_prio=True)
    a8 = marked8(64, 2, em_flags=[False, True])

    snap = snapshot_state(m)
    src_vec, dst_vec = m._src_dst_train(*a8)
    assert src_vec.shape == (2,) and dst_vec.shape == (2,)

    # the per-sample references through the model's own
    # single-replica path (identical weights, restored per sample)
    plan = m.replica_plan
    m.replica_plan = None
    try:
        for i in range(2):
            restore_state(m, snap)
            sl = tuple(a[i:i + 1] for a in a8)
            r_src, r_dst = m._src_dst_train(*sl)
            assert r_src.shape == (1,) and r_dst.shape == (1,)
            # replica r's sample r comes FIRST in the concat
            assert torch.equal(src_vec[i], r_src[0]), \
                f"src channel sample {i} mismatch"
            assert torch.equal(dst_vec[i], r_dst[0]), \
                f"dst channel sample {i} mismatch"
    finally:
        m.replica_plan = plan
    restore_state(m, snap)


def test_loss_vector_length_and_history_mean_cpu(tmp_path, monkeypatch):
    """B15-10/11: the concatenated loss vector length equals the
    (adjusted) global fetched batch (here N*b with the stubbed
    fetch), and the ``loss_history`` entry is the mean of the
    concatenated per-sample vector (the unchanged onTrainOneIter
    contract: ``float(src_loss.mean().detach())``)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    a8 = synth_samples(64, batch=4, data_format=m.model_data_format,
                       seed_no=5)
    stub_fetch(m, a8)
    # vector length on a 4-sample fetch (b == 2): the full global
    # batch, not the per-replica shard
    snap = snapshot_state(m)
    src_vec, dst_vec = m._src_dst_train(*a8)
    assert src_vec.shape == (4,) and dst_vec.shape == (4,)

    # history mean on the SAME model at the SAME pre-step weights:
    # restore, then drive one full train_one_iter on the identical
    # stubbed fetch — the closure recomputes the identical
    # concatenated vectors, so the stored history values are exactly
    # the means of the closure-returned vectors
    restore_state(m, snap)
    it, _ = m.train_one_iter()
    assert it == 1
    hist = m.loss_history[-1]
    assert len(hist) == 2
    assert hist[0] == float(src_vec.mean())
    assert hist[1] == float(dst_vec.mean())


def test_one_canonical_g_step_per_iteration_cpu(tmp_path, monkeypatch):
    """B15-12: exactly ONE canonical G step per global iteration —
    two iterations advance src_dst_opt by exactly two (never by N
    per iteration)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=6)
    stub_fetch(m, a8)
    m.train_one_iter()
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 2
    assert m.get_iter() == 2


def test_mirror_sync_after_successful_step_cpu(tmp_path, monkeypatch):
    """B15-14: the mirrors are synced AFTER a successful canonical
    G step — post-iteration, every mirror (incl. the discriminators')
    is bit-exact with its canonical, and the canonical weights were
    actually updated by the step (the mirrors followed)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=10)
    snap = snapshot_state(m)
    stub_fetch(m, a8)
    m.train_one_iter()
    # the G step changed the canonical AE (nonzero loss -> nonzero
    # grads -> the update op moved the weights)
    enc_before = snap[0][1][id(next(m.encoder.parameters()))]
    assert not torch.equal(enc_before,
                           next(m.encoder.parameters()).detach().cpu())
    assert_mirrors_synced(m)


def test_initialization_order_canon_gen_precision_mirror_sync_forward_cpu(
        tmp_path, monkeypatch):
    """§10.1.1 BINDING initialization ORDER (independent-review
    round 1) on a SIMULATED N == 2 production build — the observed
    event sequence of ONE real construction + first iteration:

      plan_created < generators_done and precision_resolved
                  < mirror_built < initial_sync < first_forward

    ``plan_created`` marks the completion of the canonical
    initialization that must precede every mirror artifact (the
    plan is the first multi-replica artifact on_initialize
    produces, immediately after the checkpoint load/init loop; the
    plan object is the device TOPOLOGY only — no mirror exists
    until ``add_component``). ``generators_done`` = all sample
    generators constructed (on the adjusted global batch).
    ``precision_resolved`` = the Commit-3 ONE global all-device
    PrecisionPlan/scaler resolution — for N > 1 production
    on_initialize triggers it at plan creation, i.e. BEFORE any
    mirror is constructed. BOTH ``generators_done`` and
    ``precision_resolved`` must precede ``mirror_built`` (their
    relative order to each other is not pinned). ``mirror_built`` =
    the first ``add_component`` (mirrors built from the FINAL
    canonical weights); ``initial_sync`` = the canonical -> mirror
    parameters + class-A buffers sync; ``first_forward`` = the
    first production ``_src_dst_train`` call (the multi closure —
    and the first replica forward — runs inside it).

    The first iteration is driven through the suite's standard
    controlled 2-sample fetch (the suite-wide stub_fetch pattern):
    this host's in-process debug generator yields fewer than N
    samples per fetch, while production runs fetch exactly N*b
    (the on_initialize batch adjustment); the initialization
    ORDER under test completes entirely during on_initialize,
    before any fetch is consumed.

    Instrumentation wraps/delegates to the REAL production methods
    (plan factory seam, ``set_training_data_generators``,
    ``_mp_ensure_resolved``, plan ``add_component`` /
    ``initial_sync``, the train closure) — no fake algorithm
    replaces production behavior; production initialization runs
    end to end."""
    events = []
    _OrderHookSAEHD.shared_events = events

    # the SIMULATED seam, recording plan creation (topology only —
    # the mirrors arrive at the deferred mirror phase)
    def fake_from_device_config(cls, cfg):
        p = ReplicaPlan.from_torch_devices(CPU, [CPU, CPU])
        events.append("plan_created")
        return p

    monkeypatch.setattr(
        dfl_nn.ReplicaPlan, "from_device_config",
        classmethod(fake_from_device_config))

    # plan-phase events: delegating wrappers around the REAL plan
    # methods (this test's plan is the only one the process
    # constructs)
    real_add = dfl_nn.ReplicaPlan.add_component

    def add_rec(self, *a, **k):
        r = real_add(self, *a, **k)
        events.append("mirror_built")
        return r

    monkeypatch.setattr(dfl_nn.ReplicaPlan, "add_component", add_rec)
    real_sync = dfl_nn.ReplicaPlan.initial_sync

    def sync_rec(self, *a, **k):
        r = real_sync(self, *a, **k)
        events.append("initial_sync")
        return r

    monkeypatch.setattr(dfl_nn.ReplicaPlan, "initial_sync", sync_rec)

    root = Path(tmp_path) / "m2"
    make_training_dirs(root)
    m = make_saehd(_OrderHookSAEHD, root, is_training=True,
                   seed=cpu_seed(archi="df"), debug=True, cpu_only=True)

    # the production train closure, wrapped for the first-forward
    # marker (onTrainOneIter calls self._src_dst_train at call
    # time — the multi closure, and the first replica forward, run
    # inside that call)
    orig_train = m._src_dst_train

    def first_forward(*a, **k):
        events.append("first_forward")
        return orig_train(*a, **k)

    m._src_dst_train = first_forward

    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=42)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1

    for ev in ("plan_created", "generators_done", "precision_resolved",
              "mirror_built", "initial_sync", "first_forward"):
        assert ev in events, f"event {ev} missing from sequence {events}"
    pos = {ev: events.index(ev)
           for ev in ("plan_created", "generators_done",
                      "precision_resolved", "mirror_built",
                      "initial_sync", "first_forward")}
    # §10.1.1: canonical init complete before BOTH the sample
    # generators and the all-device precision resolution ...
    assert pos["plan_created"] < pos["generators_done"]
    assert pos["plan_created"] < pos["precision_resolved"]
    # ... and BOTH complete before ANY mirror is constructed ...
    assert pos["generators_done"] < pos["mirror_built"]
    assert pos["precision_resolved"] < pos["mirror_built"]
    # ... and the mirrors precede the initial sync, which precedes
    # the first replica forward
    assert pos["mirror_built"] < pos["initial_sync"] < pos["first_forward"]
    _OrderHookSAEHD.shared_events = None


def test_true_face_off_no_code_d_cpu(tmp_path, monkeypatch):
    """B15-15: true_face_power == 0 -> the code discriminator is NOT
    mirrored, owns no optimizer, and the iteration runs the G step
    only."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    # canonical component name of the code-D module is 'dis'
    assert "dis" not in component_names(m)
    assert not hasattr(m, "D_code_opt")
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=11)
    stub_fetch(m, a8)
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)


def test_true_face_on_code_d_step_cpu(tmp_path, monkeypatch):
    """B15-16: true_face_power != 0 -> the code discriminator joins
    the mirrored set; the iteration performs exactly ONE canonical
    D_code_opt step (after the G step) and syncs the code-D mirror."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1)
    assert "dis" in component_names(m)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=12)
    stub_fetch(m, a8)
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.D_code_opt.iterations.item()) == 1
    assert not hasattr(m, "D_src_dst_opt")
    assert_mirrors_synced(m)
    assert_optimizer_ownership(m, m.replica_plan, "D_code_opt",
                               "code_discriminator")


def test_gan_off_no_dsrc_cpu(tmp_path, monkeypatch):
    """B15-17: gan_power == 0 -> D_src is NOT mirrored, owns no
    optimizer, and the iteration performs no D_src step."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    assert "D_src" not in component_names(m)
    assert not hasattr(m, "D_src_dst_opt")
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=13)
    stub_fetch(m, a8)
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)


def test_gan_on_dsrc_step_cpu(tmp_path, monkeypatch):
    """B15-18: gan_power != 0 -> D_src joins the mirrored set; the
    iteration performs exactly ONE canonical D_src_dst_opt step and
    syncs the D_src mirror."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  gan_power=0.05)
    assert "D_src" in component_names(m)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=14)
    stub_fetch(m, a8)
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.D_src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)
    assert_optimizer_ownership(m, m.replica_plan, "D_src_dst_opt",
                               "D_src")


def test_d_steps_post_g_weights_and_order_cpu(tmp_path, monkeypatch):
    """B15-19/20/21: the D steps run in the G -> D-code -> D_src
    order and on the POST-G weights: the canonical encoder snapshot
    at the D-code closure entry equals the one at the D_src closure
    entry (the AE is untouched between them) and differs from the
    pre-iteration snapshot (the G step updated it first)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=15)
    pre_enc = next(m.encoder.parameters()).detach().clone()

    events = []
    orig_g, orig_d1, orig_d2 = (m._src_dst_train, m._D_train,
                                m._D_src_dst_train)

    def enc_snap():
        return next(m.encoder.parameters()).detach().clone()

    m._src_dst_train = lambda *a: (
        events.append(("G", enc_snap())), orig_g(*a))[1]
    m._D_train = lambda *a: (
        events.append(("D_code", enc_snap())), orig_d1(*a))[1]
    m._D_src_dst_train = lambda *a: (
        events.append(("D_src", enc_snap())), orig_d2(*a))[1]

    stub_fetch(m, a8)
    m.train_one_iter()

    # the official order: G, then code-D, then D_src
    assert [tag for tag, _ in events] == ["G", "D_code", "D_src"]
    _, g_snap = events[0]
    _, dc_snap = events[1]
    _, ds_snap = events[2]
    # the G closure entry saw the pre-iteration AE weights...
    assert torch.equal(g_snap, pre_enc)
    # ...and the G step moved the AE before any D step ran
    assert not torch.equal(dc_snap, pre_enc)
    # both D steps saw the SAME post-G AE weights (the AE is
    # untouched between the two D closures — the D steps update
    # only their own components)
    assert torch.equal(dc_snap, ds_snap)


def test_df_mirror_component_coverage_cpu(tmp_path, monkeypatch):
    """B15-22: the df mirror component set is EXACTLY the active
    trainable AE components (+ code-D / D_src when their powers are
    on), each with n - 1 mirrors, identity-mapped to the model's
    canonical modules."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    plan = m.replica_plan
    names = {c.canonical.name: c for c in plan.components}
    # the code-D's canonical NAME is 'dis' (its model attribute is
    # code_discriminator)
    assert set(names) == {"encoder", "inter", "decoder_src",
                          "decoder_dst", "dis", "D_src"}
    assert names["encoder"].canonical is m.encoder
    assert names["inter"].canonical is m.inter
    assert names["decoder_src"].canonical is m.decoder_src
    assert names["decoder_dst"].canonical is m.decoder_dst
    assert names["dis"].canonical is m.code_discriminator
    assert names["D_src"].canonical is m.D_src
    for c in plan.components:
        assert len(c.mirrors) == plan.num_replicas - 1


def test_liae_mirror_component_coverage_cpu(tmp_path, monkeypatch):
    """B15-23: the liae mirror component set is EXACTLY the active
    AE components (encoder, inter_AB, inter_B, decoder) — the code
    discriminator is created in the df branch ONLY (true_face is a
    df-only knob the official option input zeroes for non-df
    archis, Model.py L341-343; the existing
    test_true_face_zeroed_for_non_df_cpu enforces that rule), so a
    liae model can never carry a mirrored code-D."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="liae-ud")
    plan = m.replica_plan
    names = {c.canonical.name: c for c in plan.components}
    assert set(names) == {"encoder", "inter_AB", "inter_B", "decoder"}
    assert names["encoder"].canonical is m.encoder
    assert names["inter_AB"].canonical is m.inter_AB
    assert names["inter_B"].canonical is m.inter_B
    assert names["decoder"].canonical is m.decoder
    for c in plan.components:
        assert len(c.mirrors) == 1


# true_face is a df-only knob (the official option input zeroes it
# for non-df archis — Model.py L341-343), so the matrix pairs
# tf != 0.0 with df only; liae rows run at the documented tf == 0.0
@pytest.mark.parametrize("archi, tf, gan", [
    ("df", 0.0, 0.0), ("df", 0.1, 0.0),
    ("df", 0.0, 0.05), ("df", 0.1, 0.05),
    ("liae-ud", 0.0, 0.0), ("liae-ud", 0.0, 0.05),
])
def test_option_combination_matrix_cpu(tmp_path, monkeypatch,
                                        archi, tf, gan):
    """B15-24: the option matrix — true_face OFF/ON x GAN OFF/ON
    selects the discriminator mirror components and exactly one
    canonical D step per active discriminator (the G step always
    runs exactly once; true_face is exercised on df only, the archi
    the official option input accepts it for)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi=archi,
                  true_face_power=tf, gan_power=gan)
    names = component_names(m)
    assert ("dis" in names) == (tf != 0.0)
    assert ("D_src" in names) == (gan != 0.0)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=16)
    stub_fetch(m, a8)
    m.train_one_iter()
    assert int(m.src_dst_opt.iterations.item()) == 1
    if tf != 0.0:
        assert int(m.D_code_opt.iterations.item()) == 1
    else:
        assert not hasattr(m, "D_code_opt")
    if gan != 0.0:
        assert int(m.D_src_dst_opt.iterations.item()) == 1
    else:
        assert not hasattr(m, "D_src_dst_opt")
    assert_mirrors_synced(m)


# ---------------------------------------------------------------------------
# CPU tier — checkpoint / preview / merge / export / teardown
# ---------------------------------------------------------------------------

def test_n1_checkpoint_ownership_unchanged_cpu(tmp_path, monkeypatch):
    """B15-26: the N == 1 checkpoint save is the official
    single-count set (no mirror file, no mirror key)."""
    m = build_cpu(tmp_path, monkeypatch, n=1, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    # the official DFL filenames: the D_src component saves to
    # 'GAN.npy' and its optimizer to 'GAN_opt.npy'
    names = sorted(f for _, f in m.model_filename_list)
    assert names == sorted(["encoder.npy", "inter.npy", "decoder_src.npy",
                            "decoder_dst.npy", "code_discriminator.npy",
                            "GAN.npy", "src_dst_opt.npy",
                            "D_code_opt.npy", "GAN_opt.npy"])
    root = m.saved_models_path
    m.set_iter(1)
    m.save()
    # save() writes <MODEL_NAME>_<filename> to disk; compare the
    # on-disk names with the prefix stripped against the
    # model_filename_list basenames
    files = {p.name[len(MODEL_NAME) + 1:]
             for p in Path(root).iterdir()
             if p.name.startswith(MODEL_NAME + "_")
             and p.name.endswith(".npy")}
    assert files == set(names)


def test_sim_n2_checkpoint_excludes_mirrors_cpu(tmp_path, monkeypatch):
    """B15-27: the SIMULATED_MULTI_REPLICA model's checkpoint is the
    official single-count set — the mirrors own no filename, no
    file and no save key."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=17)
    stub_fetch(m, a8)
    m.train_one_iter()
    names = sorted(f for _, f in m.model_filename_list)
    assert names == sorted(["encoder.npy", "inter.npy", "decoder_src.npy",
                            "decoder_dst.npy", "code_discriminator.npy",
                            "GAN.npy", "src_dst_opt.npy",
                            "D_code_opt.npy", "GAN_opt.npy"])
    root = m.saved_models_path
    m.set_iter(1)
    m.save()
    # save() writes <MODEL_NAME>_<filename> to disk; compare the
    # on-disk names with the prefix stripped against the
    # model_filename_list basenames
    files = {p.name[len(MODEL_NAME) + 1:]
             for p in Path(root).iterdir()
             if p.name.startswith(MODEL_NAME + "_")
             and p.name.endswith(".npy")}
    assert files == set(names)


def test_checkpoint_structural_invariance_cpu(tmp_path, monkeypatch):
    """B15-28: structural checkpoint invariance — the SIMULATED
    model's saved payload (per-file tensor count and shapes) is
    identical to a N == 1 model with the same options (the mirrors
    add nothing to the checkpoint schema)."""
    common = dict(archi="df", true_face_power=0.1, gan_power=0.05)
    m1 = build_cpu(tmp_path / "single", monkeypatch, n=1, **common)
    m2 = build_cpu(tmp_path / "multi", monkeypatch, n=2, **common)
    a8 = synth_samples(64, batch=2, data_format=m2.model_data_format,
                       seed_no=18)
    stub_fetch(m2, a8)
    m2.train_one_iter()
    for m in (m1, m2):
        m.set_iter(1)
        m.save()

    def _shapes(obj):
        # component files hold LISTS of tensors, optimizer files
        # hold DICTs of state (possibly nested) — the invariance
        # compares the full structure, sorted for determinism
        if isinstance(obj, dict):
            return tuple((k, _shapes(v)) for k, v in sorted(obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(_shapes(v) for v in obj)
        return tuple(np.shape(obj))

    def payload(root):
        out = {}
        for p in sorted(Path(root).iterdir()):
            if p.name.startswith(MODEL_NAME + "_") \
                    and p.name.endswith(".npy"):
                arr = np.load(p, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    arr = arr.item()
                out[p.name] = _shapes(arr)
        return out

    p1 = payload(m1.saved_models_path)
    p2 = payload(m2.saved_models_path)
    assert set(p1) == set(p2)
    for k in p1:
        assert p1[k] == p2[k], k


def test_fresh_n1_strict_load_of_sim_n2_checkpoint_cpu(tmp_path,
                                                       monkeypatch,
                                                       headless_io):
    """B15-29: a FRESH single-replica training model strict-loads
    the SIMULATED_MULTI_REPLICA-produced checkpoint — every
    canonical component and optimizer file loads bit-exactly
    (structural isolation; cross-count resume acceptance is Commit
    6 and is NOT claimed here)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df",
                  true_face_power=0.1, gan_power=0.05)
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=19)
    stub_fetch(m, a8)
    m.train_one_iter()
    root = m.saved_models_path
    m.set_iter(1)
    m.save()

    # the fresh loader: the REAL (non-headless) SAEHDModel in resume
    # mode on CPU (the production single-replica plan). The
    # SIMULATED plan-seam patch from build_cpu is still active
    # (fixture undo runs at test end), so restore the PRODUCTION
    # factory first: the fresh loader must not inherit the N == 2
    # simulated routing.
    if _REAL_FROM_DEVICE_CONFIG is not None:
        monkeypatch.setattr(dfl_nn.ReplicaPlan, "from_device_config",
                            _REAL_FROM_DEVICE_CONFIG)
    m2 = resume_real(root)
    assert m2.is_first_run() is False
    assert m2.get_iter() == 1
    plan = m2.replica_plan
    assert plan is not None and not plan.is_multi
    assert plan.components == []

    # every component/optimizer loads bit-exactly from the
    # SIMULATED-produced files (matched by name — the optimizer
    # display names, e.g. 'GAN_opt', are not Python attribute names)
    src_by_name = {md.name: md for md, _ in m.model_filename_list}
    for mod, filename in m2.model_filename_list:
        other = src_by_name.get(mod.name)
        assert other is not None, filename
        wa = [w.detach().cpu() for w in mod.get_weights()]
        wb = [w.detach().cpu() for w in other.get_weights()]
        assert len(wa) == len(wb), filename
        for x, y in zip(wa, wb):
            assert x.shape == y.shape, filename
            assert torch.equal(x, y), filename
    m2.finalize()


def test_preview_ae_view_canonical_only_cpu(tmp_path, monkeypatch):
    """B15-30: AE_view (the preview path) uses the CANONICAL modules
    only — a deliberately desynced mirror (and buffer) is invisible
    to the preview output."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=20)
    # the official feed contract: the batched 4D (NHWC) arrays pass
    # through as-is (a 5D array would crash the Conv2D NHWC permute)
    ws = a8[0]
    wd = a8[4]
    ref = m.AE_view(ws, wd)

    # desync the mirror encoder: parameters AND a class-A buffer
    # (no_grad: the mirror params are requires-grad leaves)
    enc_comp = next(c for c in m.replica_plan.components
                    if c.canonical is m.encoder)
    mirror_enc = enc_comp.mirrors[0]
    with torch.no_grad():
        next(mirror_enc.parameters()).add_(0.5)
        for buf in mirror_enc.buffers():
            buf.add_(1.0)

    out = m.AE_view(ws, wd)
    for a, b in zip(ref, out):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_merge_canonical_only_cpu(tmp_path, monkeypatch):
    """B15-31: the merge SURFACE (get_MergerConfig / predictor_func)
    is canonical-only and mirror-invariant — the SIMULATED model's
    merge config equals the N == 1 twin's (identical options), and a
    desynced mirror cannot perturb the config (the predictor slot is
    compared by bound-function identity, so a swapped predictor fails
    rather than trivially passing). This is a STRUCTURAL
    verification: predictor_func itself is never EXECUTED here — on
    a training model it cannot be (AE_merge is bound only in the
    non-training branch), so the behavioral desync-proof of the
    executed inference boundary is the preview test's (AE_view) and
    the B17 export regression's, not this test's."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    m1 = build_cpu(tmp_path / "single", monkeypatch, n=1, archi="df")
    cfg_multi = m.get_MergerConfig()
    cfg_single = m1.get_MergerConfig()
    assert _configs_equal(cfg_multi, cfg_single)

    # desync a mirror; the merge config is still invariant (the
    # config is built from the canonical archi/dims only; no_grad:
    # the mirror param is a requires-grad leaf)
    enc_comp = next(c for c in m.replica_plan.components
                    if c.canonical is m.encoder)
    with torch.no_grad():
        next(enc_comp.mirrors[0].parameters()).add_(0.5)
    assert _configs_equal(m.get_MergerConfig(), cfg_multi)


def test_export_surface_unchanged_cpu(tmp_path, monkeypatch):
    """B15-32: Commit 4 changes no export/inference behavior — with
    identical canonical weights the SIMULATED model's inference
    boundary (AE_view over the canonical chain) is bit-identical to
    the N == 1 twin's, and a deliberately desynced mirror stays
    invisible to it (the ONNX/DFM exporters themselves are covered
    by the B17 export regression suite; structurally the Commit-4
    diff touches no export code)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    m1 = build_cpu(tmp_path / "single", monkeypatch, n=1, archi="df")
    # identical weights on both models (fresh twins: the canonical
    # AE components are copied over; the mirrors follow the sync)
    copy_components(m1, m)
    m.replica_plan.sync_from_canonical()

    a8 = synth_samples(64, batch=1, data_format=m.model_data_format,
                       seed_no=21)
    out_multi = m.AE_view(a8[0], a8[4])
    out_single = m1.AE_view(a8[0], a8[4])
    assert len(out_multi) == len(out_single)
    for a, b in zip(out_multi, out_single):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    # desync a mirror: the inference boundary is still canonical-only
    # (no_grad: the mirror param is a requires-grad leaf)
    enc_comp = next(c for c in m.replica_plan.components
                    if c.canonical is m.encoder)
    with torch.no_grad():
        next(enc_comp.mirrors[0].parameters()).add_(0.5)
    out_multi2 = m.AE_view(a8[0], a8[4])
    for a, b in zip(out_multi2, out_single):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_teardown_releases_mirrors_cpu(tmp_path, monkeypatch):
    """B15-33: ``finalize()`` disposes the plan and releases the
    mirrors (no further use, no mirror references), leaving the
    canonical model fully usable (the checkpoint-owned state is
    untouched by the disposable mirror teardown)."""
    m = build_cpu(tmp_path, monkeypatch, n=2, archi="df")
    plan = m.replica_plan
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=22)
    stub_fetch(m, a8)
    m.train_one_iter()

    m.finalize()
    assert m.replica_plan is None
    assert plan.is_disposed
    # the disposed plan refuses all further use
    with pytest.raises(Exception):
        plan.sync_from_canonical()
    # the canonical model still runs its (single-replica) inference
    # boundary on the untouched canonical weights
    with torch.no_grad():
        preds = forward_ae(m,
                           torch.from_numpy(np.ascontiguousarray(a8[0])),
                           torch.from_numpy(np.ascontiguousarray(a8[4])))
    assert preds is not None


# ---------------------------------------------------------------------------
# GPU tier A-D (SIMULATED_MULTI_REPLICA on one physical GPU:
# logical replica 0 and 1 both on cuda:0)
# ---------------------------------------------------------------------------

@requires_gpu
def test_gpu_bf16_sim_n2(tmp_path, monkeypatch):
    """GPU-A: bf16 SIMULATED_MULTI_REPLICA N == 2 full iteration on
    cuda:0 (both logical replicas) — no scaler in bf16, finite
    losses, one canonical step per optimizer, the D steps run
    entirely FP32, mirrors synced after the steps."""
    m = build_gpu(tmp_path, monkeypatch, _HookedSAEHD, n=2,
                  precision="bf16",
                  true_face_power=0.1, gan_power=0.05)
    m._init_hook_state()
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=30)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1
    assert m._mp_scaler is None
    # the precision sequence hooks run once per attempt in EVERY
    # mode (native no-ops in bf16): one per-replica backward, one
    # canonical unscale/step/update, the step succeeded
    assert m._hook_counts == {"backward": 2, "unscale": 1, "step": 1,
                              "update": 1, "stepped": [True]}
    assert all(math.isfinite(v) for v in m.loss_history[-1])
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.D_code_opt.iterations.item()) == 1
    assert int(m.D_src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)
    m.finalize()


@requires_gpu
def test_gpu_fp16_sim_n2_success(tmp_path, monkeypatch):
    """GPU-B: fp16 SIMULATED_MULTI_REPLICA N == 2 success on cuda:0
    — ONE global native torch GradScaler for the run; exactly one
    canonical unscale/step/update for the two per-replica backwards;
    the scale stays at its initial value after a successful
    non-overflow attempt; the D steps run once, after the G step."""
    m = build_gpu(tmp_path, monkeypatch, _HookedSAEHD, n=2,
                  precision="fp16",
                  true_face_power=0.1, gan_power=0.05)
    m._init_hook_state()
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=31)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1
    # the ONE global native scaler
    assert isinstance(m._mp_scaler, torch.amp.GradScaler)
    # at the production initial scale (2**16) a NATURAL Class B
    # overflow on the first attempt is legitimate GradScaler
    # behavior (the synthetic gradients x 65536 can exceed the
    # fp16 max): the invariants hold for ANY number of bounded
    # attempts — each attempt has its own per-replica backwards
    # and exactly ONE canonical unscale/step/update, the LAST
    # attempt being the successful canonical step
    n = m._hook_counts["update"]
    assert n >= 1
    assert m._hook_counts["backward"] == 2 * n
    assert m._hook_counts["unscale"] == n
    assert m._hook_counts["step"] == n
    assert m._hook_counts["stepped"][-1] is True
    assert all(v is False for v in m._hook_counts["stepped"][:-1])
    # every overflowing attempt backed the one global scale off
    # by 0.5 (no growth is possible within the bounded retry
    # window)
    overflows = sum(1 for v in m._hook_counts["stepped"]
                    if v is False)
    assert (m._mp_scaler.get_scale()
            == FP16_INIT_SCALE * (0.5 ** overflows))
    assert all(math.isfinite(v) for v in m.loss_history[-1])
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.D_code_opt.iterations.item()) == 1
    assert int(m.D_src_dst_opt.iterations.item()) == 1
    assert_mirrors_synced(m)
    m.finalize()


@requires_gpu
def test_gpu_fp16_class_b_retry(tmp_path, monkeypatch):
    """GPU-C: fp16 Class B on the REAL SAEHD — the LATER replica's
    scaled gradients overflow on the first attempt: the attempt is
    SKIPPED (no canonical G update, no iteration increment, no
    mirror sync), exactly ONE scaler update with backoff, all
    canonical + mirror grads cleaned, the SAME fetched samples are
    retried, the retry succeeds with one canonical step, and each D
    step runs exactly ONCE (after the successful G step)."""
    m = build_gpu(tmp_path, monkeypatch, _HookedSAEHD, n=2,
                  precision="fp16",
                  true_face_power=0.1, gan_power=0.05)
    m._init_hook_state()
    m.inject_class_b = True
    a8 = synth_samples(64, batch=2, data_format=m.model_data_format,
                       seed_no=32)
    stub_fetch(m, a8)

    # the canonical weights before the run (the failed attempt must
    # leave them untouched)
    pre_enc = next(m.encoder.parameters()).detach().clone()

    it, _ = m.train_one_iter()
    assert it == 1
    # attempt 1: 2 backwards then a SKIPPED canonical step + one
    # scaler update (backoff); attempt 2 (same samples): 2 backwards
    # then the successful canonical step + one scaler update
    assert m._hook_counts["backward"] == 4
    assert m._hook_counts["unscale"] == 2
    assert m._hook_counts["step"] == 2
    assert m._hook_counts["update"] == 2
    assert m._hook_counts["stepped"] == [False, True]
    # the ONE global scaler backed off exactly once
    assert m._mp_scaler.get_scale() == FP16_HALF_SCALE
    # the failed attempt cleaned ALL canonical + mirror grads
    # (observed at the retry's first backward — the production
    # cleanup runs after the failed attempt's scaler update)
    assert m._failed_attempt_cleanup, "grads not cleaned after skip"
    # one canonical G step total (the successful retry)
    assert int(m.src_dst_opt.iterations.item()) == 1
    # each D step ran exactly ONCE — after the successful G step,
    # not once per failed retry
    assert int(m.D_code_opt.iterations.item()) == 1
    assert int(m.D_src_dst_opt.iterations.item()) == 1
    # the successful retry's canonical step moved the AE...
    assert not torch.equal(
        pre_enc.detach().cpu(),
        next(m.encoder.parameters()).detach().cpu())
    # ...and the successful attempt synced the mirrors
    assert_mirrors_synced(m)
    assert all(math.isfinite(v) for v in m.loss_history[-1])
    m.finalize()


@requires_gpu
def test_gpu_fp16_class_a_hard_error(tmp_path, monkeypatch):
    """GPU-D: fp16 Class A on the REAL SAEHD — a nonfinite forward
    on the LATER replica (an fp16-overflowing input shard) raises a
    HARD ``FloatingPointError`` from the production closure: no
    ``SkippedGeneratorStep``, no retry, no canonical G update, no D
    steps, no scaler update (the scale stays at its initial value),
    the grads are cleared, and no mirror sync occurs."""
    m = build_gpu(tmp_path, monkeypatch, _HookedSAEHD, n=2,
                  precision="fp16",
                  true_face_power=0.1, gan_power=0.05)
    m._init_hook_state()
    # the marked fetch: replica 0 gets the finite sample 0, replica
    # 1 gets sample 1 with its dst-side warped input blown up
    # (1e6 >> the fp16 max of 65504 — the autocast forward on that
    # shard overflows to nonfinite)
    a8 = list(marked8(64, 2, em_flags=[False, True]))
    a8[4] = a8[4].copy()
    a8[4][1, ...] = a8[4][1, ...] + 1.0e6
    a8 = tuple(np.ascontiguousarray(a) for a in a8)
    stub_fetch(m, a8)

    snap = snapshot_state(m)
    pre_enc = next(m.encoder.parameters()).detach().clone()

    with pytest.raises(FloatingPointError):
        m.train_one_iter()

    # replica 0 ran its backward; replica 1 died at the nonfinite
    # forward check BEFORE its backward; no canonical precision
    # sequence ran at all
    assert m._hook_counts["backward"] == 1
    assert m._hook_counts["unscale"] == 0
    assert m._hook_counts["step"] == 0
    assert m._hook_counts["update"] == 0
    assert m._hook_counts["stepped"] == []
    # no canonical G update, no D steps
    assert int(m.src_dst_opt.iterations.item()) == 0
    assert int(m.D_code_opt.iterations.item()) == 0
    assert int(m.D_src_dst_opt.iterations.item()) == 0
    # no scaler update: the scale is still the initial value
    assert m._mp_scaler.get_scale() == FP16_INIT_SCALE
    # the weights are untouched (canonical AND mirrors — no sync)
    assert torch.equal(pre_enc.detach().cpu(),
                       next(m.encoder.parameters()).detach().cpu())
    for mod, snaps in snap:
        for t in _module_tensors(mod):
            if id(t) in snaps:
                assert torch.equal(snaps[id(t)].cpu(), t.detach().cpu())
    # the Class A cleanup cleared the canonical + mirror grads
    plan = m.replica_plan
    for c in plan.components:
        for t in _module_tensors(c.canonical):
            assert t.grad is None
        for r in range(1, plan.num_replicas):
            for t in _module_tensors(c.mirrors[r - 1]):
                assert t.grad is None
    m.finalize()
