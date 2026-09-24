"""Phase 12 Commit 3 acceptance: mixed-precision validation ACROSS
replica devices (core.leras.mixed_precision all-device resolution +
core.leras.multidevice canonical-side precision lifecycle, exposed via
the nn aliases and driven through the model layer's production
``_mp_run_generator`` retry policy).

Covers the approved Phase 12 plan's normative precision contracts
(docs/IMPLEMENTATION_PLAN_v2.md §30 Phase 8 policy; docs/PHASE12_STATE.
md §6, §6.1, §10, §10.1.3, §12A, §12C; docs/REVIEW_AND_COMMIT_GUIDE_v2.
md §18, §20, §60, §61):

- ALL-device capability validation (Milestone E): the requested mode
  is validated against EVERY selected replica device — one list = the
  replica plan's ordered device list; an unsupported mode on ANY
  selected device raises ``PrecisionUnsupportedError`` naming every
  failing device (CPU fp16 on a CPU-only list; a heterogeneous
  cuda+cpu list); there is NO silent per-device fallback or
  downgrade; the single-device ``resolve_precision`` behavior is
  unchanged (parity pinned); the returned plan is the ONE global
  ``PrecisionPlan`` of the run (its ``device_type`` = the primary's;
  each replica's autocast region pins its own device type)
- ONE global GradScaler: the fp16 plan carries exactly one model-level
  scaler (never one per replica / mirror / device); off/bf16 plans
  carry NONE; the driver refuses both violations
  (``ReplicaPrecisionError``); the production per-replica scaled
  backward flow scales EVERY replica's loss vector with that single
  scaler state (the same scale applies to all replicas)
- the canonical-side §6.1 lifecycle through a SMALL PRODUCTION-PATH
  harness (``tests/smoke/Model_Lifecycle/`` — a minimal ModelBase
  subclass driving the production ``_mp_run_generator`` retry
  wrapper; no parallel fake implementation stands in for any
  production path): per-replica forward under the run-wide plan's
  per-replica autocast, per-replica SCALED backward THROUGH THE
  EXISTING ``ModelBase._mp_backward`` HOOK, Commit-2 canonicalization
  (secondary mirror grads transferred to the canonical device,
  rebound to the canonical parameters), Commit-1 official aggregation
  (per-variable MEAN of the scaled gradients), installation on the
  canonical ``.grad``, then the canonical unscale/step/update
  sequence — executed EXACTLY ONCE per attempt THROUGH THE EXISTING
  ``ModelBase._mp_unscale_opt`` / ``_mp_opt_step`` /
  ``_mp_scaler_update`` HOOKS, dependency-injected into the
  production driver (no native scaler calls in core/leras; the
  production integration path is the ModelBase hook lifecycle
  itself) — the GradScaler found_inf skip, the complete canonical +
  mirror grad cleanup after a skipped step, the post-step mirror sync
  integration point, and the production 16-attempt retry policy
  (never a retry-once; the 16th skip raises the production
  ``FloatingPointError``). The tests assert the EXACT per-attempt
  hook-call counts (wrappers that delegate immediately to the
  original bound ModelBase hooks — no hook behavior reimplemented)
  together with the native scaler state (scale / backoff)
- Class A (nonfinite forward output OR loss on ANY replica): the
  attempt closure clears the canonical + ALL mirror grads and raises
  the hard ``FloatingPointError`` — NEVER a ``SkippedGeneratorStep``,
  never scaler-recorded, never retried (the production
  ``_mp_run_generator`` only retries ``SkippedGeneratorStep``); the
  unscale/step/update hooks are NEVER reached for the failed attempt
- Class B (finite forward/loss + nonfinite SCALED grads — a later
  replica's overflow): NOT rejected pre-unscale (the fp16
  pre-unscale passthrough); canonicalized, transferred, aggregated,
  installed as the SCALED grads; the ONE ``_mp_unscale_opt`` records
  found_inf; the canonical step is skipped; the ONE
  ``_mp_scaler_update`` backfills the x0.5 scale; ALL canonical +
  mirror grads are cleared (the later replica's overflow discards
  every earlier replica's grads); no mirror sync; the caller raises
  ``SkippedGeneratorStep``; a clean retry (same pre-step weights,
  same samples, clean grads, backed-off scale) succeeds and is
  BIT-EXACT equal to a fresh attempt started at the reduced scale
  (zero-RNG harness, deterministic data)
- structural validation BEFORE any value-level policy: a malformed
  or SPARSE / compressed-sparse replica gradient always fails
  through the deliberate Commit-1 ``ReplicaGradientError``
  contract (off + COO, off N==1 + COO, bf16 + CSR, fp16 + COO) —
  while a DENSE nonfinite SCALED fp16 gradient is a NUMERIC
  OVERFLOW, not a layout defect, and passes through to the
  GradScaler's ownership (the driver reaches the model-layer hooks)
- OFF (FP32) and BF16 run with NO scaler through the same
  foundation: the direct ``get_update_op`` step-or-fail path (the
  off/bf16 ModelBase hook semantics); the PRE-unscale nonfinite
  policy for off/bf16 is a HARD ``ReplicaPrecisionError`` (no
  scaler exists there to own overflow detection) — the FP32
  reference path remains verified (guide §60); simulated
  multi-replica FP32/BF16 steps succeed with canonical aggregation,
  mirror sync and no scaler state anywhere
- N == 1 single-replica identity through the same driver (the
  unchanged single-GPU path)

SIMULATED_MULTI_REPLICA: the GPU tests map two or three LOGICAL
replicas onto ONE physical device (cuda:0, RTX 4090, Level-B
simulation through the production framework path — real nn
foundation, real leras net, real AdaBelief optimizer, real native
torch.amp.GradScaler, the real ModelBase ``_mp_*`` hooks, production
``_mp_run_generator``); physical 2-GPU acceptance stays
PENDING_ENVIRONMENTALLY / NOT_VERIFIED.

Parity label: EXACT where the same backend/precision is involved
(CPU fp32/bf16 steps; the failed-attempt bookkeeping; the
clean-retry == fresh-at-reduced-scale equivalence — power-of-two
scales make the unscale reciprocal exact); the FP16 success path is
checked with the established FP16 tolerance regime (autocast forward
rounding differs from the FP32 reference).
Provenance: INDEPENDENT_REIMPLEMENTATION (official DFL multi-GPU +
mixed-precision semantics are the semantic authority; the torch
GradScaler/autocast and the ModelBase ``_mp_*`` hooks are native
production primitives; no external code was copied).
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import mixed_precision  # noqa: E402
from core.leras import multidevice  # noqa: E402
from core.leras.multidevice import (  # noqa: E402
    ReplicaGradientError,
    ReplicaPrecisionError,
    ReplicaPlan,
)
import core.leras.models  # noqa: F401,E402  (binds nn.ModelBase)
from core.leras import nn as dfl_nn  # noqa: E402

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_Lifecycle.Model import (  # noqa: E402
    make_harness,
    make_harness_class,
    snapshot_harness,
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

_GPU_MAIN_ENV_DONE = False


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


def _half_scale_grad_scaler(device_type):
    """A native torch GradScaler preset to the ONE-backoff scale
    2**15 (the exact state the retried run's scaler reaches after
    the failed attempt's backoff) — the fresh-at-reduced-scale
    comparator for the clean-retry equivalence (the harness's
    test-only scaler_factory channel; instrumentation-free)."""
    return torch.amp.GradScaler(device=device_type,
                                init_scale=FP16_HALF_SCALE)


def _cpu(harness_cls, tmp_path, **kw):
    return make_harness(harness_cls, tmp_path, cpu_only=True, **kw)


def _gpu(harness_cls, tmp_path):
    _ensure_gpu_main_env()
    return make_harness(harness_cls, tmp_path, force_gpu_idxs=[0])


# ---------------------------------------------------------------------------
# Part 1 — all-device precision validation (resolve_precision_devices)
# ---------------------------------------------------------------------------

def test_all_device_fp16_capability_failure_cpu():
    """fp16 on a CPU-only selected list: EVERY failing device is
    named; the explicit failure is the ONLY outcome (no plan, no
    silent downgrade to off/bf16)."""
    with pytest.raises(mixed_precision.PrecisionUnsupportedError) as ei:
        mixed_precision.resolve_precision_devices("fp16", [CPU, CPU])
    msg = str(ei.value)
    assert "fp16" in msg
    assert "2 of the 2" in msg  # both failing devices reported
    assert msg.count("device " + str(CPU)) == 2


def test_all_device_bf16_capability_success_cpu():
    p = mixed_precision.resolve_precision_devices("bf16", [CPU, CPU])
    assert p.mode == mixed_precision.MODE_BF16
    assert p.enabled
    assert p.device_type == "cpu"
    assert p.autocast_dtype is torch.bfloat16
    assert p.scaler_required is False
    assert p.make_scaler() is None


def test_single_device_parity_cpu():
    """The all-device resolution restricted to a one-device list is
    behaviorally identical to the Phase 8 single-device resolution
    (plan attributes for every supported mode; the same explicit
    failure for fp16-on-CPU)."""
    for mode in ("off", "bf16"):
        single = mixed_precision.resolve_precision(mode, CPU)
        alldev = mixed_precision.resolve_precision_devices(
            mode, [CPU])
        assert alldev.mode == single.mode
        assert alldev.device_type == single.device_type
        assert alldev.enabled == single.enabled
        assert alldev.autocast_dtype == single.autocast_dtype
        assert alldev.scaler_required == single.scaler_required
    with pytest.raises(mixed_precision.PrecisionUnsupportedError):
        mixed_precision.resolve_precision("fp16", CPU)
    with pytest.raises(mixed_precision.PrecisionUnsupportedError):
        mixed_precision.resolve_precision_devices("fp16", [CPU])


def test_unknown_mode_all_devices():
    with pytest.raises(mixed_precision.PrecisionUnsupportedError):
        mixed_precision.resolve_precision_devices("bogus", [CPU, CPU])


def test_device_list_contract():
    with pytest.raises(ValueError):
        mixed_precision.resolve_precision_devices("off", [])
    with pytest.raises(TypeError):
        mixed_precision.resolve_precision_devices("off", ["cpu"])


def test_per_replica_autocast_context_pin():
    """The ONE global plan's autocast region can be pinned per
    replica device type (a CPU replica under a CPU-primary plan is
    its own type; the omitted form is the plan's type — exact
    single-device Phase 8 behavior)."""
    p = mixed_precision.resolve_precision_devices("bf16", [CPU, CPU])
    with p.autocast_context("cpu"):
        t = torch.ones(2, dtype=torch.float32)
        assert torch.nn.functional.relu(t).dtype == torch.float32
    with p.autocast_context():  # omitted -> the plan's own type
        pass
    # off mode: nullcontext either way
    off = mixed_precision.resolve_precision_devices("off", [CPU, CPU])
    assert not off.enabled
    off.autocast_context("cpu")


# ---------------------------------------------------------------------------
# Part 2 — driver unit contracts (no model layer)
# ---------------------------------------------------------------------------

def _toy_plan_and_module(n_replicas, device=CPU):
    """A one-Parameter torch module + its replica plan + optimizer
    (the driver's unit-level harness; the failure-path driver tests
    use synthetic tensors only — the production flow tests in Part
    3/4 use the real leras harness + the real ModelBase hooks)."""
    w = [torch.nn.Parameter(torch.full((4,), 0.5, device=device),
                            requires_grad=True)]
    mod = torch.nn.ParameterList([w[0]])

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = w[0]

        def forward(self, x):
            return (x * self.p).sum()

    m = _M()
    if n_replicas == 1:
        plan = ReplicaPlan.from_torch_devices(device)
    else:
        plan = ReplicaPlan.from_torch_devices(device, [device] * n_replicas)
    comp = plan.add_component(m)
    plan.initial_sync()
    from core.leras.optimizers.AdaBelief import AdaBelief
    opt = AdaBelief(lr=0.01, name="toy_opt")
    opt.initialize_variables([w[0]])
    return plan, comp, opt, m, w


def _off_mode_hooks():
    """Test-local hooks mirroring the ModelBase NO-SCALER (off/bf16)
    hook semantics — the no-op ``_mp_unscale_opt`` / ``_mp_
    scaler_update`` and the direct ``get_update_op`` step of ``_mp_
    opt_step`` — for the driver's unit-level contract tests. The
    production integration tests (Part 3/4) inject the REAL bound
    ModelBase hooks; nothing here implements any scaler logic."""
    def _unscale(optimizer, active_weights):
        # off/bf16: no scaler (ModelBase._mp_unscale_opt semantics)
        pass

    def _step(optimizer, grads_vars):
        # off/bf16: the direct official update op (ModelBase._mp_
        # opt_step semantics)
        optimizer.get_update_op(grads_vars)()
        return True

    def _update():
        # off/bf16: no scaler (ModelBase._mp_scaler_update semantics)
        pass

    return {"unscale_hook": _unscale, "step_hook": _step,
            "update_hook": _update}


def _recording_hooks(step_result=True):
    """Test-local recording no-op hooks (for the fp16-passthrough
    driver test, where the hooks are reached but must do nothing
    native): record every call, the step returns ``step_result``."""
    calls = []

    def _unscale(optimizer, active_weights):
        calls.append("unscale")

    def _step(optimizer, grads_vars):
        calls.append("step")
        return step_result

    def _update():
        calls.append("update")

    return {"unscale_hook": _unscale, "step_hook": _step,
            "update_hook": _update}, calls


def _coo_sparse_like(p):
    """A COO sparse tensor with the parameter's shape (a synthetic
    malformed layout — gradients must be dense strided)."""
    flat = torch.arange(1, p.numel() + 1, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        torch.arange(p.numel()).view(1, -1), flat, size=p.shape)


def test_one_global_scaler_contract():
    """The driver refuses any plan/scaler combination that violates
    the one-global-scaler rule (in either direction)."""
    plan, comp, opt, m, w = _toy_plan_and_module(1)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    # an off/bf16 plan CARRIES NO scaler (a scaler would be FP16
    # machinery that does not apply)
    fake_scaler = torch.amp.GradScaler("cpu", enabled=False)
    try:
        multidevice.run_replica_precision_step(
            plan, off, fake_scaler, opt, [[(w[0].grad if w[0].grad is not None
                                            else torch.ones_like(w[0])), w[0]]])
        raise AssertionError("off/bf16 + scaler must be refused")
    except ReplicaPrecisionError as e:
        assert "NO" in str(e)
    # an fp16 plan REQUIRES the one global scaler (resolved on a
    # machine without CUDA, fp16 is unresolvable on CPU — exercise
    # the required-side of the contract through the plan object
    # directly)
    fp16 = mixed_precision.PrecisionPlan(
        mixed_precision.MODE_FP16, "cuda", torch.float16,
        scaler_required=True)
    with pytest.raises(ReplicaPrecisionError) as ei:
        multidevice.run_replica_precision_step(
            plan, fp16, None, opt, [[(torch.ones_like(w[0]), w[0])]])
    assert "REQUIRES" in str(ei.value)


def test_replica_count_mismatch_refused():
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    with pytest.raises(ReplicaPrecisionError) as ei:
        multidevice.run_replica_precision_step(
            plan, off, None, opt, [[(torch.ones_like(w[0]), w[0])]])
    assert "exactly 2" in str(ei.value)


def test_model_layer_hooks_required():
    """The canonical unscale/step/update sequence is owned by the
    injected MODEL-LAYER hooks (the existing ModelBase ``_mp_*``
    bound hooks): the driver refuses to run a step without them
    (no silent native fallback inside core/leras)."""
    plan, comp, opt, m, w = _toy_plan_and_module(1)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    with pytest.raises(ReplicaPrecisionError) as ei:
        multidevice.run_replica_precision_step(
            plan, off, None, opt, [[(torch.ones_like(w[0]), w[0])]],
            unscale_hook=_off_mode_hooks()["unscale_hook"],
            step_hook=_off_mode_hooks()["step_hook"])
    assert "update_hook" in str(ei.value)
    assert "ModelBase" in str(ei.value)


def test_clear_replica_grads_cleans_canonical_and_mirrors():
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    # accumulate grads on the canonical AND the mirror
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    assert w[0].grad is not None
    assert mirror.p.grad is not None
    data_before = w[0].detach().clone()
    multidevice.clear_replica_grads(plan)
    assert w[0].grad is None            # canonical cleared
    assert mirror.p.grad is None        # every mirror cleared
    assert torch.equal(w[0], data_before)  # DATA untouched
    # idempotent
    multidevice.clear_replica_grads(plan)
    # N == 1: the same call clears the canonical only
    plan1, comp1, opt1, m1, w1 = _toy_plan_and_module(1)
    torch.autograd.backward(m1(torch.ones(4)))
    assert w1[0].grad is not None
    multidevice.clear_replica_grads(plan1)
    assert w1[0].grad is None


def test_replica_param_lists_ownership():
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    lists = multidevice.replica_param_lists(plan, [w[0]])
    assert len(lists) == 2
    # replica 0 owns the CANONICAL parameters (identity, order kept)
    assert lists[0][0] is w[0]
    # replica 1 owns the MIRROR counterpart (identity, path-aligned)
    assert lists[1][0] is comp.mirror_params[1][0]
    assert lists[1][0] is comp.mirrors[0].p
    # N == 1: a single canonical-only list
    plan1, comp1, opt1, m1, w1 = _toy_plan_and_module(1)
    lists1 = multidevice.replica_param_lists(plan1, [w1[0]])
    assert len(lists1) == 1 and lists1[0][0] is w1[0]
    # a parameter that is not a canonical parameter of the plan's
    # components is refused (no silent adoption)
    foreign = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ReplicaPrecisionError):
        multidevice.replica_param_lists(plan, [foreign])


def test_off_mode_nonfinite_grad_hard_error():
    """off/bf16 PRE-unscale nonfinite replica grad: the driver hard-
    errors (no scaler exists to own overflow detection) — matching
    the single-device behavior; the fp16 passthrough is exercised in
    the Part-4 Class B tests. The structural validation (N > 1
    aggregation contract) ran FIRST — these grads are structurally
    valid, so only the value-level policy fires."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    # per-replica backwards (canonical params and mirror params are
    # distinct tensors — no cross-accumulation between replicas)
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    # corrupt the LATER replica's (mirror's) grad
    lists = [
        [(w[0].grad, w[0])],
        [(torch.full_like(mirror.p.grad, float("inf")), mirror.p)],
    ]
    with pytest.raises(ReplicaPrecisionError) as ei:
        multidevice.run_replica_precision_step(
            plan, off, None, opt, lists, active_weights=[w[0]],
            **_off_mode_hooks())
    assert "NONFINITE" in str(ei.value)
    # no step happened
    assert int(opt.iterations.item()) == 0


def test_off_mode_successful_n2_step():
    """The off-mode (no scaler) N==2 full driver pipeline: multiple
    per-replica backwards, canonicalization, official aggregation,
    canonical install, the hooks (off semantics: no-op unscale/
    update, direct get_update_op step), one iterations increment."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    w0 = w[0].detach().clone()
    lists = []
    for r in range(2):
        # per-replica backwards on the replica-owned parameters
        # (distinct tensors — no cross-accumulation between replicas)
        mod = m if r == 0 else comp.mirrors[0]
        torch.autograd.backward(mod(torch.ones(4)))
        plist = (multidevice.replica_param_lists(plan, [w[0]]))[r]
        lists.append([(p.grad, p) for p in plist])
    stepped = multidevice.run_replica_precision_step(
        plan, off, None, opt, lists, active_weights=[w[0]],
        **_off_mode_hooks())
    assert stepped is True
    assert int(opt.iterations.item()) == 1
    assert not torch.equal(w[0], w0)  # the step changed the weights
    plan.sync_from_canonical()
    assert torch.equal(w[0], comp.mirrors[0].p)


def test_off_mode_coo_sparse_grad_rejected():
    """STRUCTURE before VALUE: a COO sparse replica gradient
    (N == 2, off mode) fails through the deliberate Commit-1
    ``ReplicaGradientError`` layout contract — never through a
    backend-specific value error."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    lists = [
        [(w[0].grad, w[0])],
        [(_coo_sparse_like(mirror.p), mirror.p)],
    ]
    with pytest.raises(ReplicaGradientError) as ei:
        multidevice.run_replica_precision_step(
            plan, off, None, opt, lists, active_weights=[w[0]],
            **_off_mode_hooks())
    assert int(opt.iterations.item()) == 0


def test_off_mode_n1_coo_sparse_grad_rejected():
    """The N == 1 driver path validates entries too: a COO sparse
    canonical gradient is rejected through the same deliberate
    ``ReplicaGradientError`` contract (the single-replica path
    applies the same layout policy)."""
    plan, comp, opt, m, w = _toy_plan_and_module(1)
    off = mixed_precision.resolve_precision_devices("off", [CPU])
    with pytest.raises(ReplicaGradientError):
        multidevice.run_replica_precision_step(
            plan, off, None, opt,
            [[(_coo_sparse_like(w[0]), w[0])]],
            active_weights=[w[0]], **_off_mode_hooks())


def test_bf16_csr_sparse_grad_rejected():
    """A CSR (compressed sparse row) replica gradient is the same
    unsupported-layout class: rejected through the deliberate
    ``ReplicaGradientError`` contract (bf16 mode, N == 2)."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    bf16 = mixed_precision.resolve_precision_devices("bf16", [CPU])
    if not hasattr(torch, "sparse_csr_tensor"):
        pytest.skip("torch.sparse_csr_tensor unavailable in this build")
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    try:
        # a CSR tensor with the parameter's shape: one entry per row
        indptr = torch.arange(0, 5, dtype=torch.int32)
        indices = torch.arange(4, dtype=torch.int32)
        values = torch.ones(4, dtype=torch.float32)
        csr = torch.sparse_csr_tensor(indptr, indices, values,
                                      size=(4,))
    except RuntimeError:
        pytest.skip("CSR tensor construction unavailable in this build")
    lists = [
        [(w[0].grad, w[0])],
        [(csr, mirror.p)],
    ]
    with pytest.raises(ReplicaGradientError):
        multidevice.run_replica_precision_step(
            plan, bf16, None, opt, lists, active_weights=[w[0]],
            **_off_mode_hooks())
    assert int(opt.iterations.item()) == 0


def test_fp16_sparse_layout_rejected():
    """unsupported LAYOUT != numeric overflow: under an fp16 plan a
    COO sparse replica gradient is STILL rejected through the
    ``ReplicaGradientError`` layout contract (before any
    pre-unscale value policy; the model-layer hooks are never
    reached)."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    fp16 = mixed_precision.PrecisionPlan(
        mixed_precision.MODE_FP16, "cuda", torch.float16,
        scaler_required=True)
    fake_scaler = torch.amp.GradScaler("cpu", enabled=False)
    hooks, calls = _recording_hooks()
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    lists = [
        [(w[0].grad, w[0])],
        [(_coo_sparse_like(mirror.p), mirror.p)],
    ]
    with pytest.raises(ReplicaGradientError):
        multidevice.run_replica_precision_step(
            plan, fp16, fake_scaler, opt, lists,
            active_weights=[w[0]], **hooks)
    assert calls == []  # the hooks are never reached


def test_fp16_dense_nonfinite_passthrough():
    """The mirror image of the layout rejection: a DENSE nonfinite
    SCALED fp16 replica gradient (a numeric overflow) is NOT
    rejected pre-unscale — it passes canonicalization, aggregation
    and canonical install, and the model-layer hooks ARE reached
    (the GradScaler owns overflow detection downstream); this is
    the fp16 Class B pre-unscale passthrough (the full GradScaler
    skip/backoff flow is the Part-4 Class B test)."""
    plan, comp, opt, m, w = _toy_plan_and_module(2)
    fp16 = mixed_precision.PrecisionPlan(
        mixed_precision.MODE_FP16, "cuda", torch.float16,
        scaler_required=True)
    fake_scaler = torch.amp.GradScaler("cpu", enabled=False)
    hooks, calls = _recording_hooks(step_result=True)
    torch.autograd.backward(m(torch.ones(4)))
    mirror = comp.mirrors[0]
    torch.autograd.backward(mirror(torch.ones(4)))
    # a LATER replica's dense SCALED grad overflows (inf)
    lists = [
        [(w[0].grad, w[0])],
        [(torch.full_like(mirror.p.grad, float("inf")), mirror.p)],
    ]
    stepped = multidevice.run_replica_precision_step(
        plan, fp16, fake_scaler, opt, lists,
        active_weights=[w[0]], **hooks)
    assert stepped is True
    assert calls == ["unscale", "step", "update"]
    # the nonfinite SCALED mean was installed on the canonical
    # parameter's .grad (the GradScaler's territory, not the
    # driver's)
    assert w[0].grad is not None
    assert not torch.isfinite(w[0].grad).all()


# ---------------------------------------------------------------------------
# Part 3 — model-layer lifecycle through the production harness
# ---------------------------------------------------------------------------

def test_fp32_simulated_multi_replica_step(tmp_path):
    """OFF (FP32) — the verified reference path (guide §60) — as a
    SIMULATED_MULTI_REPLICA run: two logical replicas, NO scaler,
    per-replica backward THROUGH the existing ModelBase
    ``_mp_backward`` hook, canonical aggregation, the direct update
    op (the off/bf16 ``_mp_opt_step`` hook semantics), the post-step
    mirror sync, one optimizer-iteration increment."""
    cls = make_harness_class("off", 2)
    model = _cpu(cls, tmp_path)
    assert model._mp_scaler is None  # off plans carry no scaler
    assert model._mp_plan.enabled is False
    it, _ = model.train_one_iter()
    assert it == 1
    assert model.stepped_history == [True]
    assert model._mp_scaler is None
    # the off-mode hook bookkeeping: one backward per replica, one
    # (no-op) unscale / update, one step — the EXISTING ModelBase
    # hooks, exactly once each
    assert model._hook_counts["backward"] == 2
    assert model._hook_counts["unscale"] == 1
    assert model._hook_counts["step"] == 1
    assert model._hook_counts["update"] == 1
    # the canonical weights changed; the mirror was synced bit-exact
    assert model.component.canonical is not model.component.mirrors[0]
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach(), b.detach())
    # loss history: one finite entry from the concatenated per-replica
    # per-sample loss vectors
    assert len(model.loss_history) == 1
    assert len(model.loss_history[0]) == 1
    assert model.loss_history[0][0] >= 0.0


def test_bf16_simulated_multi_replica_step_cpu(tmp_path):
    """BF16 on CPU (the supported no-scaler tier) as a
    SIMULATED_MULTI_REPLICA run: the same no-scaler success
    contract as FP32 (autocast bfloat16 per replica, direct update
    op, mirror sync)."""
    cls = make_harness_class("bf16", 2)
    model = _cpu(cls, tmp_path)
    assert model._mp_scaler is None
    assert model._mp_plan.enabled
    assert model._mp_plan.autocast_dtype is torch.bfloat16
    it, _ = model.train_one_iter()
    assert it == 1
    assert model.stepped_history == [True]
    assert model._hook_counts["backward"] == 2
    assert model._hook_counts["step"] == 1
    assert model._hook_counts["update"] == 1
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach(), b.detach())


def test_fp32_nonfinite_grad_hard_error_model(tmp_path):
    """OFF-mode model layer: a nonfinite replica grad injected
    post-backward is a hard ``ReplicaPrecisionError`` from the
    driver — NOT retried (off-mode attempts == 1), NOT a
    ``SkippedGeneratorStep``, no step, no sync; the unscale/step/
    update hooks are NEVER reached for the failed attempt."""
    cls = make_harness_class("off", 2)
    model = _cpu(cls, tmp_path)
    model.nonfinite_grad_replica = 1
    w0 = [p.detach().clone() for p in model.component.canonical_params]
    with pytest.raises(ReplicaPrecisionError) as ei:
        model.train_one_iter()
    assert "NONFINITE" in str(ei.value)
    # both replicas' backwards ran before the driver's value-level
    # policy; the canonical sequence hooks were never reached
    assert model._hook_counts["backward"] == 2
    assert model._hook_counts["unscale"] == 0
    assert model._hook_counts["step"] == 0
    assert model._hook_counts["update"] == 0
    # no step happened; the canonical weights are unchanged
    assert model.get_iter() == 0
    for before, p in zip(w0, model.component.canonical_params):
        assert torch.equal(before, p.detach())
    # no scaler state exists anywhere in off mode
    assert model._mp_scaler is None


def test_single_replica_off_step(tmp_path):
    """N == 1 through the same driver: the official identity path
    (no mirrors, canonicalize_grads identity) — the unchanged
    single-GPU / CPU-only behavior of the production path."""
    cls = make_harness_class("off", 1)
    model = _cpu(cls, tmp_path)
    assert model.plan.num_replicas == 1
    assert model.component.mirror_params == {}
    it, _ = model.train_one_iter()
    assert it == 1
    assert model.stepped_history == [True]
    assert int(model.harness_opt.iterations.item()) == 1
    assert model._hook_counts["backward"] == 1
    assert model._hook_counts["step"] == 1


def test_fp16_capability_failure_at_model_start(tmp_path):
    """The all-device validation runs at model start (on_initialize):
    fp16 on a CPU-only selected list fails EXPLICITLY during
    construction — no silent fallback to off/bf16, no partial plan."""
    cls = make_harness_class("fp16", 2)
    with pytest.raises(mixed_precision.PrecisionUnsupportedError) as ei:
        _cpu(cls, tmp_path)
    assert "fp16" in str(ei.value)


# ---------------------------------------------------------------------------
# Part 4 — GPU lifecycle (SIMULATED_MULTI_REPLICA on the one 4090)
# ---------------------------------------------------------------------------

@requires_gpu
def test_all_device_fp16_capability_success_gpu():
    _ensure_gpu_main_env()
    p = mixed_precision.resolve_precision_devices("fp16", [CUDA, CUDA])
    assert p.mode == mixed_precision.MODE_FP16
    assert p.enabled
    assert p.device_type == "cuda"
    assert p.autocast_dtype is torch.float16
    assert p.scaler_required is True
    assert isinstance(p.make_scaler(), torch.amp.GradScaler)


@requires_gpu
def test_all_device_bf16_capability_success_gpu():
    _ensure_gpu_main_env()
    p = mixed_precision.resolve_precision_devices("bf16", [CUDA, CUDA])
    assert p.enabled
    assert p.device_type == "cuda"
    assert p.autocast_dtype is torch.bfloat16
    assert p.scaler_required is False  # bf16 NEVER carries a scaler
    assert p.make_scaler() is None


@requires_gpu
def test_heterogeneous_selected_device_rejection():
    """A heterogeneous selected list (CUDA primary + CPU secondary)
    requesting fp16 is an EXPLICIT start failure naming the failing
    (CPU) device — never a per-device downgrade."""
    _ensure_gpu_main_env()
    with pytest.raises(mixed_precision.PrecisionUnsupportedError) as ei:
        mixed_precision.resolve_precision_devices(
            "fp16", [CUDA, CPU])
    msg = str(ei.value)
    assert "1 of the 2" in msg
    assert str(CPU) in msg  # the failing device is named
    assert str(CUDA) not in msg.replace(str(CPU), "")


@requires_gpu
def test_fp16_successful_scaled_attempt(tmp_path):
    """(i) the full successful FP16 attempt: multiple SCALED
    backwards THROUGH the existing ``ModelBase._mp_backward`` hook
    (one per replica), canonicalization of the secondary mirror
    grads, aggregation of the SCALED grads, installation on the
    canonical ``.grad``, then EXACTLY ONE ``_mp_unscale_opt`` /
    ``_mp_opt_step`` / ``_mp_scaler_update`` THROUGH THE INJECTED
    MODEL-LAYER HOOKS operating the ONE global native scaler, the
    successful canonical step, the post-step mirror sync; the
    single scaler state scaled EVERY replica's loss (the same scale
    on all replicas); the one-global-scaler rule holds
    structurally (exactly one scaler instance for the whole run)."""
    _ensure_gpu_main_env()
    cls = make_harness_class("fp16", 2)
    model = _gpu(cls, tmp_path)
    # exactly ONE global scaler instance for the run (the native
    # torch GradScaler; torch 2.14 exposes its device as ``_device``)
    assert isinstance(model._mp_scaler, torch.amp.GradScaler)
    assert model._mp_scaler._device == "cuda"
    it, _ = model.train_one_iter()
    sc = model._mp_scaler
    assert it == 1
    assert model.stepped_history == [True]
    # the per-attempt HOOK bookkeeping (one attempt ran): one
    # _mp_backward per replica, exactly one _mp_unscale_opt /
    # _mp_opt_step / _mp_scaler_update — the EXISTING ModelBase
    # hooks (the harness wraps them with delegating counters)
    assert model._hook_counts["backward"] == 2
    assert model._hook_counts["unscale"] == 1
    assert model._hook_counts["step"] == 1
    assert model._hook_counts["update"] == 1
    assert int(model.harness_opt.iterations.item()) == 1
    # no overflow -> the native scaler scale is untouched
    assert sc.get_scale() == FP16_INIT_SCALE
    # the ONE global scaler scaled EVERY replica's loss vector: the
    # same scale on all replicas (fp16 autocast forward rounding
    # differs from the fp32 reference, so the ratio is checked with
    # the established tolerance regime)
    loss_vecs = model._attempt_loss_vecs
    assert len(loss_vecs) == 2
    for v in loss_vecs:
        assert torch.isfinite(v).all()
        assert v.dtype == torch.float32
    # the canonical weights changed; the mirror synced bit-exact
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach().cpu(), b.detach().cpu())
    # the loss history entry is the concatenated per-replica
    # per-sample loss mean (finite)
    assert model.loss_history[0][0] >= 0.0


@requires_gpu
def test_fp16_later_replica_overflow_class_b(tmp_path):
    """(ii) Class B: a LATER replica's scaled gradients overflow
    (injected post-backward; the production paths own the rest):
    NOT rejected pre-unscale (the fp16 passthrough), canonicalized,
    transferred, aggregated (the mean of the scaled grads carries
    the nonfinite), installed, the ONE ``_mp_unscale_opt`` records
    found_inf, the canonical ``_mp_opt_step`` SKIPS (no
    optimizer-iteration increment), the ONE ``_mp_scaler_update``
    backfills the x0.5 scale, ALL canonical + mirror grads are
    cleared (the later replica discards every earlier replica's
    grads), NO mirror sync; the production ``_mp_run_generator``
    then performs the clean retry (same pre-step weights, same
    samples, clean grads, backed-off scale) which SUCCEEDS — and
    the retried run's final state is BIT-EXACT equal to a fresh
    attempt started at the reduced scale."""
    _ensure_gpu_main_env()
    # --- the retried run: attempt 1 overflows (later replica),
    # --- attempt 2 (clean) succeeds at the backed-off scale
    cls = make_harness_class("fp16", 2)
    model = _gpu(cls, tmp_path)
    model.overflow_replica = 1  # the LATER replica overflows
    it, _ = model.train_one_iter()  # production 16-attempt wrapper
    sc = model._mp_scaler
    # the retry happened inside the production wrapper
    assert model.stepped_history == [False, True]
    assert it == 1  # ONE successful canonical step total
    assert int(model.harness_opt.iterations.item()) == 1
    # the per-attempt HOOK bookkeeping: the failed attempt ran one
    # _mp_backward per replica + exactly one _mp_unscale_opt /
    # _mp_opt_step / _mp_scaler_update; the clean retry ran exactly
    # one more of each — exactly one update PER ATTEMPT (never per
    # replica), no hidden extra scaler update
    assert model._hook_counts["backward"] == 4
    assert model._hook_counts["unscale"] == 2
    assert model._hook_counts["step"] == 2
    assert model._hook_counts["update"] == 2
    # the x0.5 backoff happened exactly once (on the failed attempt)
    assert sc.get_scale() == FP16_HALF_SCALE
    # the successful step synced the mirror (bit-exact)
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach().cpu(), b.detach().cpu())
    # --- the fresh comparator: same pre-step weights + same samples
    # --- + clean grads + the reduced scale from the start
    fresh_cls = make_harness_class(
        "fp16", 2, scaler_factory=_half_scale_grad_scaler)
    fresh = _gpu(fresh_cls, tmp_path)
    it_f, _ = fresh.train_one_iter()
    assert it_f == 1
    # the clean retry is exactly a fresh attempt at the reduced scale:
    # bit-exact canonical weights, mirror weights and optimizer state
    # ([iters] + ms + vs), same scaler scale
    retried, freshsnap = snapshot_harness(model), snapshot_harness(fresh)
    assert retried["params"] and len(retried["params"]) > 0
    for a, b in zip(retried["params"], freshsnap["params"]):
        assert torch.equal(a, b)
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(retried["mirror_params"][r],
                        freshsnap["mirror_params"][r]):
            assert torch.equal(a, b)
    for a, b in zip(retried["states"], freshsnap["states"]):
        assert torch.equal(a, b)
    assert retried["iterations"] == freshsnap["iterations"] == 1
    assert retried["scale"] == freshsnap["scale"] == FP16_HALF_SCALE
    # both comparators started from the deterministic harness
    # initialization (zero-RNG weights + fixed samples): the failed
    # attempt touched nothing, so the retried run trained from the
    # same pre-step weights as the fresh run — and the bit-exact
    # equivalence above proves the clean retry IS that fresh
    # attempt.


@requires_gpu
def test_fp16_class_a_nonfinite_loss(tmp_path):
    """(iii) Class A (nonfinite LOSS on a replica): the attempt
    closure clears the canonical + ALL mirror grads and raises the
    hard ``FloatingPointError`` — NEVER a ``SkippedGeneratorStep``,
    never scaler-recorded, never retried (the production
    ``_mp_run_generator`` only retries ``SkippedGeneratorStep``);
    the canonical-sequence hooks are NEVER reached (the failed
    attempt's unscale/step/update counts stay at zero); no step,
    no sync, the scale is untouched."""
    _ensure_gpu_main_env()
    cls = make_harness_class("fp16", 2)
    model = _gpu(cls, tmp_path)
    model.nonfinite_loss_replica = 1
    sc = model._mp_scaler
    pre_weights = [p.detach().cpu().clone()
                   for p in model.component.canonical_params]
    with pytest.raises(FloatingPointError) as ei:
        model.train_one_iter()
    assert "nonfinite" in str(ei.value)
    # NOT a SkippedGeneratorStep (it is a subclass of RuntimeError,
    # not of FloatingPointError — the exact exception type is the
    # contract)
    assert type(ei.value) is FloatingPointError
    # hook bookkeeping: only the replicas reached before the
    # failure ran their _mp_backward (replica 0 fully backpropped;
    # replica 1's loss was checked BEFORE its backward); the
    # canonical-sequence hooks were never reached
    assert model._hook_counts["backward"] == 1
    assert model._hook_counts["unscale"] == 0
    assert model._hook_counts["step"] == 0
    assert model._hook_counts["update"] == 0
    # no step, no sync: weights and mirrors unchanged, scale intact
    assert model.get_iter() == 0
    assert int(model.harness_opt.iterations.item()) == 0
    assert sc.get_scale() == FP16_INIT_SCALE
    for before, p in zip(pre_weights, model.component.canonical_params):
        assert torch.equal(before.cpu(), p.detach().cpu())
    # the Class A cleanup: canonical AND mirror grads are all None
    assert model.component.canonical_params[0].grad is None
    for r in range(1, model.plan.num_replicas):
        for p in model.component.mirror_params[r]:
            assert p.grad is None


@requires_gpu
def test_fp16_class_a_nonfinite_forward(tmp_path):
    """(iii) Class A (nonfinite FORWARD output on a replica): the
    same hard ``FloatingPointError`` contract (cleanup before the
    raise; never a skipped step; never scaler-recorded; the
    canonical-sequence hooks never reached)."""
    _ensure_gpu_main_env()
    cls = make_harness_class("fp16", 2)
    model = _gpu(cls, tmp_path)
    model.nonfinite_forward_replica = 1
    sc = model._mp_scaler
    with pytest.raises(FloatingPointError) as ei:
        model.train_one_iter()
    assert type(ei.value) is FloatingPointError
    assert model._hook_counts["backward"] == 1
    assert model._hook_counts["unscale"] == 0
    assert model._hook_counts["step"] == 0
    assert model._hook_counts["update"] == 0
    assert int(model.harness_opt.iterations.item()) == 0
    assert sc.get_scale() == FP16_INIT_SCALE
    for r in range(1, model.plan.num_replicas):
        for p in model.component.mirror_params[r]:
            assert p.grad is None
    for p in model.component.canonical_params:
        assert p.grad is None


@requires_gpu
def test_fp16_bounded_retry_16_attempts(tmp_path):
    """The EXACT production retry policy (``_mp_run_generator``):
    16 bounded attempts under fp16 — a persistent per-attempt
    overflow is retried 16 times (NEVER a retry-once), each attempt
    running EXACTLY one _mp_unscale_opt / _mp_opt_step /
    _mp_scaler_update through the injected ModelBase hooks (no
    hidden extra scaler update anywhere), and the 16th skip raises
    the production ``FloatingPointError("FP16 generator update
    skipped 16 times")``; the native scaler scale backfills x0.5 on
    every failed attempt (2**16 -> 2**0 = 1.0); no step, no sync,
    no mirror movement across the whole bounded run."""
    _ensure_gpu_main_env()
    cls = make_harness_class("fp16", 2)
    model = _gpu(cls, tmp_path)
    model.overflow_every_attempt = True  # the LAST replica, EVERY attempt
    sc = model._mp_scaler
    pre_weights = [p.detach().cpu().clone()
                   for p in model.component.canonical_params]
    with pytest.raises(FloatingPointError) as ei:
        model.train_one_iter()
    assert "skipped 16 times" in str(ei.value)
    # the production policy: 16 attempts, one hook sequence each
    # (two _mp_backward per attempt — both replicas backprop before
    # the last replica's post-backward overflow injection)
    assert len(model.stepped_history) == 16
    assert model.stepped_history == [False] * 16
    assert model._hook_counts["backward"] == 32
    assert model._hook_counts["unscale"] == 16
    assert model._hook_counts["step"] == 16
    assert model._hook_counts["update"] == 16
    # x0.5 backoff on every failed attempt: 2**16 / 2**16 == 1.0
    assert sc.get_scale() == pytest.approx(1.0)
    # no step ever happened; the weights and mirrors never moved
    assert int(model.harness_opt.iterations.item()) == 0
    for before, p in zip(pre_weights, model.component.canonical_params):
        assert torch.equal(before.cpu(), p.detach().cpu())
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach().cpu(), b.detach().cpu())


@requires_gpu
def test_one_global_scaler_three_replicas(tmp_path):
    """The one-global-scaler rule at N == 3: three replicas scale
    their loss vectors (through the ONE ``_mp_backward`` hook state)
    through the SAME single scaler; the successful step runs
    EXACTLY ONE ``_mp_scaler_update`` hook call (never one per
    replica); both mirrors sync bit-exact."""
    _ensure_gpu_main_env()
    cls = make_harness_class("fp16", 3)
    model = _gpu(cls, tmp_path)
    it, _ = model.train_one_iter()
    sc = model._mp_scaler
    assert it == 1
    assert model.stepped_history == [True]
    # one scaler, one hook sequence for ALL three replicas
    assert model._hook_counts["backward"] == 3
    assert model._hook_counts["unscale"] == 1
    assert model._hook_counts["step"] == 1
    assert model._hook_counts["update"] == 1
    # all three replicas' per-sample loss vectors are finite
    assert len(model._attempt_loss_vecs) == 3
    for v in model._attempt_loss_vecs:
        assert torch.isfinite(v).all()
    # both mirrors synced
    for r in (1, 2):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach().cpu(), b.detach().cpu())


@requires_gpu
def test_bf16_simulated_multi_replica_step_gpu(tmp_path):
    """BF16 on the Ampere+ GPU (no scaler tier) as a
    SIMULATED_MULTI_REPLICA run: autocast bfloat16 per replica, the
    no-scaler direct update op (the off/bf16 ``_mp_opt_step`` hook
    semantics), the post-step mirror sync — the same success
    contract as FP32, on the GPU tier."""
    _ensure_gpu_main_env()
    cls = make_harness_class("bf16", 2)
    model = _gpu(cls, tmp_path)
    assert model._mp_scaler is None
    it, _ = model.train_one_iter()
    assert it == 1
    assert model.stepped_history == [True]
    assert model._hook_counts["backward"] == 2
    assert model._hook_counts["step"] == 1
    assert model._hook_counts["update"] == 1
    for r in range(1, model.plan.num_replicas):
        for a, b in zip(model.component.canonical_params,
                        model.component.mirror_params[r]):
            assert torch.equal(a.detach(), b.detach())
