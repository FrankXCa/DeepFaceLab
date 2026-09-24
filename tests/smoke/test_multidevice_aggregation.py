"""Phase 12 Commit 1 acceptance: official-compatible replica gradient
aggregation (core.leras.multidevice.average_gv_list, exposed as
nn.average_gv_list).

Covers the torch port of the official DFL ``average_gv_list`` op
(provenance: INDEPENDENT_REIMPLEMENTATION — the official TF source is
the semantic authority, preserved in core/leras/ops/ops_tf.py; the code
is re-implemented, not copied):

- N==1 identity: the single replica list is returned UNCHANGED (same
  list object, same gradient tensor objects, same parameter objects;
  no validation, no cloning) — the official identity behavior
- N>1 exact mean over the replica axis: G_final = (1/N) * sum_r G_r
  (torch analogue of the official
  reduce_mean(concat([expand_dims(g, 0) for g in gs], 0), 0)); exact
  values for N=2 and N=3 (inputs chosen so the quotients are exact in
  fp32); dtype/shape/device of the replica grads preserved; the result
  pairs each mean grad with the CANONICAL (replica 0) parameter object
- multiple variables aggregated independently in replica-0 order
- fail-fast validation (ReplicaGradientError, no silent zip / no
  implicit cast / no silent densification): empty input, None
  gradients, non-tensor gradients, non-strided-layout gradients
  (ONLY torch.strided is accepted — sparse COO, compressed-sparse
  CSR, and CSC layouts all fail fast; the layout check is
  deliberate because torch reports is_sparse == False for CSR),
  replica-list length mismatch, shape mismatch, dtype mismatch,
  parameter association mismatch (a later replica references a
  different parameter object at a position), duplicate parameter
  within one replica list, cross-device gradient mismatch (Commit-1
  device contract: the caller normalizes replica grads onto the
  aggregation/canonical device; the helper NEVER transfers)
- finiteness is caller-owned by design: the helper performs NO blanket
  torch.isfinite rejection (FP16 Class B deliberately passes scaled
  inf/nan grads to the caller's torch.amp.GradScaler; fp32/bf16
  nonfinite rejection is the model closure's job) — pinned by an
  inf/nan passthrough test
- optimizer interaction: replica grads -> average_gv_list ->
  get_update_op; the one-step AdaBelief update (official global-norm
  clip applied to the AGGREGATED grads) matches an independent NumPy
  reference of the official formula applied to the MEAN grads;
  iterations increments once; no mirrors involved
- the aggregation consumes NO torch RNG state (the optimizer-owned
  lr_dropout mask draws stay exactly optimizer-side; p=1.0 ==
  disabled)
- official parity: for N=2, mean(G0, G1) == (G0 + G1) / 2 ==
  full_batch_sum / 2 asserted EXACTLY against an independent
  full-batch reference
- nn.average_gv_list is the same function object as the module's

Parity label: EXACT (deterministic single-thread CPU values; the
selected inputs make every fp32 quotient exact). GPU cross-device
tests run on the RTX 4090 when CUDA is available (skipped on
CPU-only environments). No TensorFlow import, no direct CUDA call in
the source.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras.multidevice import (average_gv_list,  # noqa: E402
                                    ReplicaGradientError)

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Phase 12 GPU test: CUDA device required",
)

# the OFFICIAL optimizer denominator epsilon (finfo RESOLUTION,
# decimal): 1e-06 for f32.
OFFICIAL_EPS_F32 = 1e-6


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    """Every test runs with a CPU + NCHW foundation unless it
    initializes something else explicitly."""
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")


def _param(shape, values=None, name=None, device=None):
    if values is None:
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    p = torch.nn.Parameter(torch.from_numpy(values.astype(np.float32)))
    if device is not None:
        p = p.to(device)
    if name is not None:
        p._dfl_name = name
    return p


def _grad(values, device=None):
    g = torch.tensor(np.array(values, dtype=np.float32))
    if device is not None:
        g = g.to(device)
    return g


# ---------------------------------------------------------------------------
# independent NumPy references of the official optimizer formulas
# (the torch implementation is never its own oracle)
# ---------------------------------------------------------------------------

def _ref_adabelief_one_step(g, lr=0.1, b1=0.9, b2=0.999, eps=OFFICIAL_EPS_F32):
    # official AdaBelief, one step from zero state, mask == 1.0:
    #   m = (1-b1)*g
    #   v = (1-b2)*(g - m)^2
    #   x += -lr*m / (sqrt(v) + eps)
    g = np.asarray(g, dtype=np.float64)
    m = (1.0 - b1) * g
    v = (1.0 - b2) * (g - m) ** 2
    return -lr * m / (np.sqrt(v) + eps)


def _ref_global_clip(grads, clipnorm):
    # official: float32 global norm over ALL gradients; each grad
    # scaled by clipnorm/norm when norm >= clipnorm
    n = float(np.sqrt(sum(np.float32(np.sum(np.float32(g) ** 2)) for g in grads)))
    return [g * (clipnorm / n) if n >= clipnorm else g for g in grads]


# ---------------------------------------------------------------------------
# public API / alias
# ---------------------------------------------------------------------------

def test_nn_alias_is_the_multidevice_function():
    # the call-site alias is the exact same function object
    assert dfl_nn.average_gv_list is average_gv_list


# ---------------------------------------------------------------------------
# N == 1 identity (official behavior)
# ---------------------------------------------------------------------------

def test_n1_identity_single_variable():
    p = _param((2,), name="w:0")
    g = _grad([1.0, 2.0])
    gv_list = [(g, p)]
    out = average_gv_list([gv_list])
    # official: return grad_var_list[0] unchanged — same list object,
    # same gradient tensor object, same parameter object
    assert out is gv_list
    assert out[0][0] is g
    assert out[0][1] is p
    # and through the public alias
    assert dfl_nn.average_gv_list([gv_list]) is gv_list


def test_n1_identity_multiple_variables():
    a = _param((2,), name="a:0")
    b = _param((3,), name="b:0")
    ga, gb = _grad([1.0, 2.0]), _grad([3.0, 4.0, 5.0])
    gv_list = [(ga, a), (gb, b)]
    out = average_gv_list([gv_list])
    assert out is gv_list
    assert out[0][0] is ga and out[0][1] is a
    assert out[1][0] is gb and out[1][1] is b


def test_empty_replica_list_raises():
    with pytest.raises(ReplicaGradientError, match="at least one replica"):
        average_gv_list([])


# ---------------------------------------------------------------------------
# N > 1 exact mean (official reduce_mean over the replica axis)
# ---------------------------------------------------------------------------

def test_n2_exact_mean():
    p = _param((2,), name="w:0")
    g0, g1 = _grad([1.0, 3.0]), _grad([5.0, 7.0])
    out = average_gv_list([[ (g0, p) ], [ (g1, p) ]])
    assert len(out) == 1
    mean_g, canon = out[0]
    assert canon is p  # canonical (replica 0) parameter object
    # (1+5)/2 = 3.0, (3+7)/2 = 5.0 — exact in fp32
    assert torch.equal(mean_g, torch.tensor([3.0, 5.0], dtype=torch.float32))
    # the result is a fresh tensor: NOT aliased to either replica grad
    assert mean_g is not g0 and mean_g is not g1
    assert mean_g.dtype == torch.float32
    assert mean_g.device.type == "cpu"


def test_n3_exact_mean():
    p = _param((2,), name="w:0")
    # sums (1+4+7)=12, (2+5+8)=15 -> means 4.0, 5.0, exact in fp32
    g0, g1, g2 = _grad([1.0, 2.0]), _grad([4.0, 5.0]), _grad([7.0, 8.0])
    out = average_gv_list([[ (g0, p) ], [ (g1, p) ], [ (g2, p) ]])
    mean_g, canon = out[0]
    assert canon is p
    assert torch.equal(mean_g, torch.tensor([4.0, 5.0], dtype=torch.float32))


def test_mean_preserves_dtype_shape_device():
    p = _param((2, 2), name="w:0")
    g0 = _grad([[1.0, 3.0], [5.0, 7.0]])
    g1 = _grad([[3.0, 5.0], [7.0, 9.0]])
    out = average_gv_list([[ (g0, p) ], [ (g1, p) ]])
    mean_g = out[0][0]
    assert mean_g.dtype == torch.float32
    assert mean_g.shape == (2, 2)
    assert mean_g.device.type == "cpu"
    # no second division: exactly (g0+g1)/2, elementwise
    assert torch.equal(mean_g, torch.tensor([[2.0, 4.0], [6.0, 8.0]],
                                            dtype=torch.float32))


def test_multi_variable_independent():
    a = _param((2,), name="a:0")
    b = _param((2,), name="b:0")
    ga0, ga1 = _grad([1.0, 3.0]), _grad([5.0, 7.0])
    gb0, gb1 = _grad([-2.0, -4.0]), _grad([10.0, -2.0])
    out = average_gv_list([[ (ga0, a), (gb0, b) ],
                           [ (ga1, a), (gb1, b) ]])
    # canonical order and canonical parameter objects preserved
    assert [pair[1] for pair in out] == [a, b]
    assert torch.equal(out[0][0], torch.tensor([3.0, 5.0], dtype=torch.float32))
    # (-2+10)/2 = 4.0, (-4-2)/2 = -3.0
    assert torch.equal(out[1][0], torch.tensor([4.0, -3.0], dtype=torch.float32))


# ---------------------------------------------------------------------------
# fail-fast validation (no silent zip, no implicit cast, no densification)
# ---------------------------------------------------------------------------

def test_none_grad_raises():
    p = _param((2,), name="w:0")
    with pytest.raises(ReplicaGradientError, match="None"):
        average_gv_list([[ (None, p) ], [ (_grad([1.0, 2.0]), p) ]])
    with pytest.raises(ReplicaGradientError, match="None"):
        average_gv_list([[ (_grad([1.0, 2.0]), p) ], [ (None, p) ]])


def test_non_tensor_grad_raises():
    p = _param((2,), name="w:0")
    with pytest.raises(ReplicaGradientError, match="tensor"):
        average_gv_list([[ (np.array([1.0, 2.0], np.float32), p) ],
                         [ (_grad([1.0, 2.0]), p) ]])


# Layout gate: ONLY torch.strided (dense) gradients are accepted. Every
# sparse / compressed-sparse layout must fail fast with
# ReplicaGradientError (never a generic RuntimeError) before anything
# can reach torch.stack, and nothing is ever densified silently.

def test_sparse_coo_layout_raises():
    p = _param((2,), name="w:0")
    sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1]]),
        values=torch.tensor([1.0, 2.0]),
        size=(2,))
    assert sparse.layout != torch.strided
    with pytest.raises(ReplicaGradientError, match="torch.strided"):
        average_gv_list([[ (sparse, p) ], [ (_grad([1.0, 2.0]), p) ]])


def test_sparse_csr_layout_raises():
    # CSR is the layout the old ``is_sparse`` check missed: torch reports
    # ``is_sparse == False`` for CSR tensors, but the layout gate must
    # still reject it.
    p = _param((2,), name="w:0")
    csr = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    assert csr.layout == torch.sparse_csr
    with pytest.raises(ReplicaGradientError, match="torch.strided"):
        average_gv_list([[ (csr, p) ], [ (_grad([1.0, 2.0]), p) ]])


def test_sparse_csc_layout_raises():
    # Additional compressed-sparse layout (column-major): guarded so the
    # suite degrades to a skip on torch builds without the CSC API.
    if not hasattr(torch.Tensor, "to_sparse_csc"):
        pytest.skip("Tensor.to_sparse_csc not available in this torch build")
    p = _param((2,), name="w:0")
    csc = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csc()
    assert csc.layout == torch.sparse_csc
    with pytest.raises(ReplicaGradientError, match="torch.strided"):
        average_gv_list([[ (csc, p) ], [ (_grad([1.0, 2.0]), p) ]])


def test_replica_list_length_mismatch_raises():
    a = _param((2,), name="a:0")
    b = _param((2,), name="b:0")
    with pytest.raises(ReplicaGradientError, match="same number of pairs"):
        average_gv_list([[ (_grad([1.0, 3.0]), a) ],
                        [ (_grad([5.0, 7.0]), a), (_grad([1.0, 1.0]), b) ]])


def test_shape_mismatch_raises():
    p = _param((2,), name="w:0")
    with pytest.raises(ReplicaGradientError, match="shape"):
        average_gv_list([[ (_grad([1.0, 3.0]), p) ],
                         [ (_grad([5.0, 7.0, 9.0]), p) ]])


def test_dtype_mismatch_raises():
    p = _param((2,), name="w:0")
    g_f16 = torch.tensor([5.0, 7.0], dtype=torch.float16)
    with pytest.raises(ReplicaGradientError, match="dtype"):
        average_gv_list([[ (_grad([1.0, 3.0]), p) ], [ (g_f16, p) ]])


def test_association_mismatch_raises():
    a = _param((2,), name="a:0")
    a_other = _param((2,), name="a2:0")  # different object, same slot
    with pytest.raises(ReplicaGradientError, match="canonical parameter"):
        average_gv_list([[ (_grad([1.0, 3.0]), a) ],
                         [ (_grad([5.0, 7.0]), a_other) ]])


def test_duplicate_param_in_one_replica_raises():
    a = _param((2,), name="a:0")
    with pytest.raises(ReplicaGradientError, match="ambiguous"):
        average_gv_list([[ (_grad([1.0, 3.0]), a), (_grad([5.0, 7.0]), a) ],
                         [ (_grad([3.0, 5.0]), a), (_grad([7.0, 9.0]), a) ]])


@requires_gpu
def test_cross_device_mismatch_raises_and_never_transfers():
    # Commit-1 device contract: the caller normalizes ALL replica grads
    # onto the aggregation/canonical device. Replica 0's grad on
    # cuda:0 (the aggregation device), replica 1's grad still on cpu,
    # SAME canonical parameter object -> the device check must fire
    # and the helper must NOT move any tensor.
    p = _param((2,), name="w:0")
    g_cpu = _grad([1.0, 3.0])
    g_gpu = _grad([5.0, 7.0], device="cuda:0")
    with pytest.raises(ReplicaGradientError, match="device"):
        average_gv_list([[ (g_gpu, p) ], [ (g_cpu, p) ]])
    # neither grad was transferred by the helper
    assert g_cpu.device.type == "cpu"
    assert g_gpu.device.type == "cuda"
    # the same-device call succeeds on GPU (deterministic aggregation):
    # mean of replica grads [1,3] and [5,7] is exactly [3,5]
    g_gpu_a = _grad([1.0, 3.0], device="cuda:0")
    g_gpu_b = _grad([5.0, 7.0], device="cuda:0")
    out = average_gv_list([[ (g_gpu_a, p) ], [ (g_gpu_b, p) ]])
    mean_g = out[0][0]
    assert mean_g.device.type == "cuda"
    assert torch.equal(mean_g.cpu(), torch.tensor([3.0, 5.0], dtype=torch.float32))


# ---------------------------------------------------------------------------
# finiteness is caller-owned (no blanket rejection by design)
# ---------------------------------------------------------------------------

def test_nonfinite_grads_pass_through_by_design():
    # The helper performs NO blanket torch.isfinite rejection: FP16
    # Class B deliberately passes scaled inf/nan grads to the
    # caller's GradScaler; fp32/bf16 nonfinite rejection is the
    # model closure's job. Nonfinite values flow through the mean
    # unchanged.
    p = _param((2,), name="w:0")
    g0 = torch.tensor([1.0, float("inf")], dtype=torch.float32)
    g1 = torch.tensor([3.0, float("nan")], dtype=torch.float32)
    out = dfl_nn.average_gv_list([[ (g0, p) ], [ (g1, p) ]])
    mean_g, canon = out[0]
    assert canon is p
    # finite element averages normally: (1+3)/2 = 2.0 exactly
    assert torch.equal(mean_g[0:1], torch.tensor([2.0], dtype=torch.float32))
    # nonfinite element stays nonfinite: inf+nan -> nan, nan/2 -> nan
    assert torch.isnan(mean_g[1])


# ---------------------------------------------------------------------------
# optimizer interaction (no mirrors involved; Commit-1 boundary)
# ---------------------------------------------------------------------------

def test_optimizer_interaction_with_clipping_matches_reference():
    # replica batch-SUM grads -> average_gv_list -> one AdaBelief step;
    # the official global-norm clip acts on the AGGREGATED grads; the
    # resulting update must match the independent NumPy reference of
    # the official formula applied to the MEAN grads.
    p1 = _param((2,), values=np.array([10.0, 10.0], np.float32), name="a:0")
    p2 = _param((2,), values=np.array([3.0, 4.0], np.float32), name="b:0")
    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=1.0, lr_dropout=1.0, name="t")
    opt.initialize_variables([p1, p2])

    # per-replica batch-SUM grads (official replica semantics)
    reps = [[ (_grad([20.0, 20.0]), p1), (_grad([6.0, 8.0]), p2) ],
            [ (_grad([10.0, 10.0]), p1), (_grad([3.0, 4.0]), p2) ]]
    agged = dfl_nn.average_gv_list(reps)
    # canonical association: replica-0 parameter objects, replica-0 order
    assert [pv for _, pv in agged] == [p1, p2]
    mean1 = agged[0][0].numpy().astype(np.float64)
    mean2 = agged[1][0].numpy().astype(np.float64)
    # the mean grads are exactly (g0+g1)/2 (exact in fp32)
    assert np.abs(mean1 - np.array([15.0, 15.0])).max() == 0.0
    assert np.abs(mean2 - np.array([4.5, 6.0])).max() == 0.0

    p10 = np.array([10.0, 10.0])
    p20 = np.array([3.0, 4.0])
    opt.get_update_op(agged)()
    assert int(opt.iterations.item()) == 1  # ONE canonical step

    # global norm of the AGGREGATED grads: sqrt(15^2+15^2+4.5^2+6^2)
    # = sqrt(506.25) ~ 22.5 > 1 -> clip active, each grad scaled by
    # 1/22.5 (official per-gradient c/n scaling)
    c1, c2 = _ref_global_clip([mean1, mean2], 1.0)
    d1 = _ref_adabelief_one_step(c1)
    d2 = _ref_adabelief_one_step(c2)
    assert np.abs(p1.detach().numpy() - (p10 + d1)).max() < 1e-5
    assert np.abs(p2.detach().numpy() - (p20 + d2)).max() < 1e-5


def test_aggregation_consumes_no_rng_state():
    # aggregation is a pure deterministic function: it must not
    # advance torch's RNG state, so the optimizer-owned lr_dropout
    # mask draws stay exactly optimizer-side (no extra mask draws via
    # the aggregation path).
    p = _param((4,), name="w:0")
    reps = [[ (_grad([1.0, 2.0, 3.0, 4.0]), p) ],
            [ (_grad([5.0, 6.0, 7.0, 8.0]), p) ]]
    state_before = torch.get_rng_state()
    agged = dfl_nn.average_gv_list(reps)
    assert torch.equal(torch.get_rng_state(), state_before)

    # with p=1.0 (mask disabled) the step on the aggregated grads is
    # deterministic and matches the independent reference
    opt = dfl_nn.AdaBelief(lr=0.1, lr_dropout=1.0, name="t")
    opt.initialize_variables([p])
    p0 = p.detach().numpy().astype(np.float64)
    opt.get_update_op(agged)()
    d = _ref_adabelief_one_step(np.array([3.0, 4.0, 5.0, 6.0]))
    assert np.abs(p.detach().numpy() - (p0 + d)).max() < 1e-5
    assert int(opt.iterations.item()) == 1


# ---------------------------------------------------------------------------
# official parity
# ---------------------------------------------------------------------------

def test_official_parity_mean_is_full_batch_sum_over_n():
    # official replica semantics: each replica's grad is the batch SUM
    # over its own samples; the final grad is the MEAN over replicas.
    # For a full batch split evenly over the two replicas,
    # (G0 + G1) / 2 == full_batch_sum / 2 — asserted EXACTLY.
    p = _param((3,), name="w:0")
    full = np.array([2.0, 5.0, 7.0], np.float64)   # full-batch sum
    half = full / 2.0                                # per-replica batch sums
    g0 = torch.tensor(half, dtype=torch.float32)
    g1 = torch.tensor(half, dtype=torch.float32)
    out = dfl_nn.average_gv_list([[ (g0, p) ], [ (g1, p) ]])
    mean_g, canon = out[0]
    assert canon is p
    # EXACT: the aggregated grad equals the full-batch sum / 2
    assert torch.equal(mean_g, torch.tensor(half, dtype=torch.float32))
    # and the general official form: (G0 + G1) / 2
    assert torch.equal(mean_g, ((g0 + g1) / 2).to(torch.float32))
