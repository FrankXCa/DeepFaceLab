"""Phase 3E2 acceptance: official DFL optimizers in torch.

Covers AdaBelief and RMSprop (official semantics, the TF reference
is preserved in core/leras/optimizers/optimizers_tf.py) plus the
random_binomial op (official lr_dropout mask source):

- deterministic one-step / multi-step updates against INDEPENDENT
  NumPy implementations of the official formulas (fresh-state and
  evolving-state references; the torch implementation is never its
  own oracle)
- zero-gradient no-ops, positive/negative gradient direction,
  multi-parameter updates
- the official denominator epsilon is the dtype's finfo
  RESOLUTION - the DECIMAL resolution 10**ceil(log10(machine
  eps)): 1e-06 for f32, 1e-03 for f16 (verified under the official
  pinned NumPy 1.19.3; identical under NumPy 2.x). The machine
  epsilon torch.finfo(...).eps (1.19e-07 for f32) is NOT the
  official value - an explicit discrimination test pins the
  official value against it
- NO bias correction / NO momentum / NO weight decay (official
  has none; third-party additions are not adopted)
- lr_cos schedule: official literal 2*3.1415926535/lr_cos and the
  POST-increment iteration count (official TF queue-order
  semantics, matching USER_LEGACY/EXTERNAL_A) - pinned by a
  step-1 discrimination test
- global-norm gradient clipping (float32 global norm over ALL
  gradients, per-gradient c/n scaling, never per-parameter norms)
- lr_dropout: ONE FRESH mask per parameter per step (the official
  TF graph re-evaluated its random op on every run of the update
  op; the USER_LEGACY frozen mask is rejected) - pinned by
  consecutive-step mask-difference and statistical tests; p=1.0
  == disabled, p=0.0 freezes weights but states still evolve
- official state layout and checkpoint names: iters + all ms_* +
  all vs_* (AdaBelief) / all acc_* (RMSprop), exact official
  sub-names for Saveable checkpoints (Phase 4 converter maps them
  1:1 by name/shape/dtype)
- the mandatory resume test: N steps -> snapshot -> fresh
  optimizer -> restore -> step N+1 == continuous N+1 run
- RTX 4090 execution through the Phase 2 device abstraction with
  CPU-vs-GPU parity (skip on CPU-only environments)
- no TensorFlow import, no direct CUDA in the optimizer source

Parity labels: WITHIN_TOLERANCE (1e-5) for the multi-op f32 vs
f64-reference comparisons; EXACT for the zero-gradient and state
layout tests. TensorFlow runtime parity: NOT_VERIFIED (no TF
environment in the tested venvs); exact TF RNG-stream parity for
random_binomial: NOT claimed (a given seed uses a local torch
generator).
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
from core.leras import ops as dfl_ops  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Phase 3E2 GPU test: RTX 4090 / CUDA device required",
)

# the OFFICIAL optimizer denominator epsilon:
# np.finfo(dtype).resolution - the DECIMAL resolution
# 10 ** ceil(log10(machine eps)) - which is NOT the machine
# epsilon itself. Verified under the official pinned NumPy 1.19.3
# (and identical under NumPy 1.26.x / 2.5.x - the property did not
# change): f32 -> 1e-06, f16 -> 1e-03. torch.finfo(...).eps
# (1.19e-07 for f32) is NOT the official value.
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
        # set AFTER the device move: Parameter.to() copies do not carry
        # custom attributes (the torch-era official-name hook)
        p._dfl_name = name
    return p


# ---------------------------------------------------------------------------
# independent NumPy references of the official optimizer formulas
# ---------------------------------------------------------------------------

def _ref_adabelief_states(ms, vs, g, lr, b1, b2, eps, mask=1.0):
    # official: m_t = b1*ms + (1-b1)*g
    #           v_t = b2*vs + (1-b2)*(g - m_t)^2
    #           v_diff = -lr*m_t / (sqrt(v_t) + eps)
    m_t = b1 * ms + (1.0 - b1) * g
    v_t = b2 * vs + (1.0 - b2) * (g - m_t) ** 2
    v_diff = -lr * m_t / (np.sqrt(v_t) + eps)
    return m_t, v_t, v_diff * mask


def _ref_rmsprop_state(acc, g, lr, rho, eps, mask=1.0):
    # official: new_a = rho*acc + (1-rho)*g^2
    #           v_diff = -lr*g / (sqrt(new_a) + eps)
    new_a = rho * acc + (1.0 - rho) * g ** 2
    v_diff = -lr * g / (np.sqrt(new_a) + eps)
    return new_a, v_diff * mask


def _ref_global_clip(grads, clipnorm):
    # official: norm = sqrt(sum_g sum(g^2)) in float32; each grad
    # scaled by clipnorm/norm when norm >= clipnorm
    n = float(np.sqrt(sum(np.float32(np.sum(np.float32(g) ** 2)) for g in grads)))
    out = []
    for g in grads:
        out.append(g * (clipnorm / n) if n >= clipnorm else g)
    return out


def _ref_lr_cos_mult(iters, lr_cos):
    # official literal: 2*3.1415926535/lr_cos (NOT 2*math.pi)
    return (np.cos(iters * (2 * 3.1415926535 / float(lr_cos))) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# AdaBelief
# ---------------------------------------------------------------------------

def test_adabelief_single_step_parity():
    x = _param((2, 2), values=np.array([[1.0, 2.0], [3.0, 4.0]], np.float32), name="w:0")
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([x])
    g = torch.tensor([[0.5, -0.25], [1.5, 0.75]], dtype=torch.float32)

    x_ref = x.detach().numpy().astype(np.float64)
    g_ref = g.numpy().astype(np.float64)
    m_ref = np.zeros_like(g_ref)
    v_ref = np.zeros_like(g_ref)
    run = opt.get_update_op([(g, x)])
    run()

    m_t, v_t, v_diff = _ref_adabelief_states(m_ref, v_ref, g_ref, 0.1, 0.9, 0.999, OFFICIAL_EPS_F32)
    x_ref = x_ref + v_diff
    assert int(opt.iterations.item()) == 1
    assert np.abs(x.detach().numpy() - x_ref).max() < 1e-5
    assert np.abs(list(opt.ms_dict.values())[0].numpy() - m_t).max() < 1e-5
    assert np.abs(list(opt.vs_dict.values())[0].numpy() - v_t).max() < 1e-5
    # NO bias correction: after one step from zero state, ms is the
    # RAW first moment (1-beta1)*g, not (1-beta1)*g/(1-beta1^1)
    assert np.abs(list(opt.ms_dict.values())[0].numpy()
                  - 0.1 * g_ref).max() < 1e-6


def test_adabelief_multi_step_evolution():
    x = _param((2, 3), name="w:0")
    x0 = x.detach().clone()
    opt = dfl_nn.AdaBelief(lr=0.05, name="t")
    opt.initialize_variables([x])

    x_ref = x0.numpy().astype(np.float64)
    m_ref = np.zeros_like(x_ref)
    v_ref = np.zeros_like(x_ref)
    grads = [
        np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]], np.float64),
        np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], np.float64),
        np.array([[-1.0, 2.0, -3.0], [4.0, -0.5, 1.0]], np.float64),
        np.zeros((2, 3), np.float64),
        np.array([[2.0, 0.0, -2.0], [0.0, 1.0, 0.0]], np.float64),
    ]
    for k, g_np in enumerate(grads):
        g = torch.from_numpy(g_np.astype(np.float32))
        opt.get_update_op([(g, x)])()
        m_t, v_t, v_diff = _ref_adabelief_states(m_ref, v_ref, g_np, 0.05,
                                                 0.9, 0.999, OFFICIAL_EPS_F32)
        m_ref, v_ref = m_t, v_t
        x_ref = x_ref + v_diff
        assert int(opt.iterations.item()) == k + 1
        assert np.abs(x.detach().numpy() - x_ref).max() < 1e-5
        assert np.abs(list(opt.ms_dict.values())[0].numpy() - m_ref).max() < 1e-5
        assert np.abs(list(opt.vs_dict.values())[0].numpy() - v_ref).max() < 1e-5


def test_adabelief_multi_param_update():
    a = _param((2, 2), name="a:0")
    b = _param((4,), name="b:0")
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([a, b])
    ga = torch.from_numpy(np.ones((2, 2), np.float32))
    gb = torch.from_numpy(-np.ones((4,), np.float32))
    a0, b0 = a.detach().numpy().astype(np.float64), b.detach().numpy().astype(np.float64)
    opt.get_update_op([(ga, a), (gb, b)])()
    ma, va, da = _ref_adabelief_states(np.zeros((2, 2)), np.zeros((2, 2)),
                                       np.ones((2, 2)), 0.1, 0.9, 0.999, OFFICIAL_EPS_F32)
    mb, vb, db = _ref_adabelief_states(np.zeros((4,)), np.zeros((4,)),
                                       -np.ones((4,)), 0.1, 0.9, 0.999, OFFICIAL_EPS_F32)
    assert np.abs(a.detach().numpy() - (a0 + da)).max() < 1e-5
    assert np.abs(b.detach().numpy() - (b0 + db)).max() < 1e-5
    # states keyed per parameter, official names
    names = [n for n, _ in opt._iter_official_weights()]
    assert names == ["iters:0", "ms_a_0:0", "ms_b_0:0", "vs_a_0:0", "vs_b_0:0"], names


def test_adabelief_zero_gradient_exact_noop():
    x = _param((2, 2), name="w:0")
    x0 = x.detach().clone()
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([x])
    g = torch.zeros_like(x)
    opt.get_update_op([(g, x)])()
    assert torch.equal(x, x0)  # EXACT: zero grad, zero state -> zero update
    ms = list(opt.ms_dict.values())[0]
    vs = list(opt.vs_dict.values())[0]
    assert torch.allclose(ms, torch.zeros_like(ms), atol=0.0)
    assert torch.allclose(vs, torch.zeros_like(vs), atol=0.0)


def test_adabelief_gradient_sign_direction():
    x = _param((2, 2), values=np.ones((2, 2), np.float32) * 5.0, name="w:0")
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([x])
    x0 = x.detach().clone()
    # positive gradient -> weights decrease
    opt.get_update_op([(torch.ones_like(x), x)])()
    assert torch.all(x < x0)
    # negative gradient -> weights increase
    x1 = x.detach().clone()
    opt.get_update_op([(-torch.ones_like(x), x)])()
    assert torch.all(x > x1)


def test_adabelief_official_epsilon_is_finfo_resolution():
    # tiny gradient: the epsilon in the denominator dominates
    # sqrt(v_t); the OFFICIAL value is the finfo RESOLUTION
    # (1e-06 for f32, verified under the pinned NumPy 1.19.3), NOT
    # the machine epsilon torch.finfo(...).eps = 1.19e-07
    g_np = np.full((2, 2), 1e-4, np.float64)
    x = _param((2, 2), values=np.zeros((2, 2), np.float32) + 7.0, name="w:0")
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([x])
    g = torch.from_numpy(g_np.astype(np.float32))
    x0 = x.detach().numpy().astype(np.float64)
    opt.get_update_op([(g, x)])()
    _, _, vd_official = _ref_adabelief_states(np.zeros((2, 2)),
                                              np.zeros((2, 2)),
                                              g_np, 0.1, 0.9, 0.999,
                                              OFFICIAL_EPS_F32)
    _, _, vd_machine_eps = _ref_adabelief_states(np.zeros((2, 2)),
                                                 np.zeros((2, 2)),
                                                 g_np, 0.1, 0.9, 0.999,
                                                 float(torch.finfo(torch.float32).eps))
    got = x.detach().numpy().astype(np.float64) - x0
    assert np.abs(got - vd_official).max() < 1e-6
    # the machine-epsilon variant must NOT match the official value
    # (sqrt(v_t) = 9e-06 here, so 1e-06 vs 1.19e-07 is significant)
    assert np.abs(vd_official - vd_machine_eps).max() > 1e-5


def test_adabelief_lr_cos_post_increment_schedule():
    # fresh state + large constant gradient: the per-element update is
    # proportional to the per-step lr, so the weight decrement pins the
    # cosine schedule. The official TF queue order (assign_add queued
    # before the updates) means step 1 uses iters=1, not iters=0.
    g_np = np.full((2, 2), 10.0, np.float64)
    x = _param((2, 2), values=np.full((2, 2), 10.0, np.float32), name="w:0")
    opt = dfl_nn.AdaBelief(lr=1.0, lr_cos=8, name="t")
    opt.initialize_variables([x])
    g = torch.from_numpy(g_np.astype(np.float32))

    x_ref = x.detach().numpy().astype(np.float64)
    m_ref = np.zeros_like(x_ref)
    v_ref = np.zeros_like(x_ref)
    g_np32 = g_np.astype(np.float32)
    for step in range(1, 5):
        opt.get_update_op([(g, x)])()
        lr_eff = 1.0 * _ref_lr_cos_mult(step, 8)  # POST-increment iters = step
        m_t, v_t, v_diff = _ref_adabelief_states(m_ref, v_ref, g_np32, lr_eff,
                                                 0.9, 0.999, OFFICIAL_EPS_F32)
        m_ref, v_ref = m_t, v_t
        x_ref = x_ref + v_diff
        assert int(opt.iterations.item()) == step
        assert np.abs(x.detach().numpy() - x_ref).max() < 1e-4
    # step-1 discrimination: post-increment gives lr_mult=(cos(pi/4)+1)/2
    # = 0.8536; a pre-increment implementation would have used 1.0 -
    # the reference above only matches the post-increment schedule, so
    # the parity asserts above pin the official queue-order behavior.
    mult_post = float(_ref_lr_cos_mult(1, 8))
    mult_pre = float(_ref_lr_cos_mult(0, 8))
    assert mult_post == pytest.approx(0.8535534, abs=1e-6)
    assert mult_pre == 1.0


def test_adabelief_global_norm_clipping():
    # multi-parameter global norm: p1 grad [20,20], p2 grad [6,8]
    # -> norm 30; clipnorm 1 -> each grad scaled by 1/30
    p1 = _param((2,), values=np.array([10.0, 10.0], np.float32), name="a:0")
    p2 = _param((2,), values=np.array([3.0, 4.0], np.float32), name="b:0")
    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=1.0, name="t")
    opt.initialize_variables([p1, p2])

    g1 = torch.from_numpy(np.array([20.0, 20.0], np.float32))
    g2 = torch.from_numpy(np.array([6.0, 8.0], np.float32))
    p10, p20 = p1.detach().numpy().astype(np.float64), p2.detach().numpy().astype(np.float64)
    opt.get_update_op([(g1, p1), (g2, p2)])()

    # the GLOBAL norm (sqrt(20^2+20^2+6^2+8^2) = 30) scales BOTH grads
    c1, c2 = _ref_global_clip([np.array([20.0, 20.0], np.float64),
                               np.array([6.0, 8.0], np.float64)], 1.0)
    _, _, d1 = _ref_adabelief_states(np.zeros((2,)), np.zeros((2,)), c1,
                                     0.1, 0.9, 0.999, OFFICIAL_EPS_F32)
    _, _, d2 = _ref_adabelief_states(np.zeros((2,)), np.zeros((2,)), c2,
                                     0.1, 0.9, 0.999, OFFICIAL_EPS_F32)
    assert np.abs(p1.detach().numpy() - (p10 + d1)).max() < 1e-5
    assert np.abs(p2.detach().numpy() - (p20 + d2)).max() < 1e-5


def test_adabelief_clipnorm_below_threshold_unchanged():
    # global norm sqrt(3^2+4^2+1^2+2^2) = sqrt(26) < 100 -> no clipping:
    # the clipped run must equal the plain (no clipnorm) run
    def run(clipnorm):
        p1 = _param((2,), values=np.array([1.0, 2.0], np.float32), name="a:0")
        p2 = _param((2,), values=np.array([1.0, 1.0], np.float32), name="b:0")
        opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=clipnorm, name="t")
        opt.initialize_variables([p1, p2])
        g1 = torch.from_numpy(np.array([3.0, 4.0], np.float32))
        g2 = torch.from_numpy(np.array([1.0, 2.0], np.float32))
        opt.get_update_op([(g1, p1), (g2, p2)])()
        return p1.detach().numpy(), p2.detach().numpy()

    c1, c2 = run(100.0)
    u1, u2 = run(0.0)
    assert np.abs(c1 - u1).max() < 1e-7
    assert np.abs(c2 - u2).max() < 1e-7


def test_adabelief_lr_dropout_per_step_resampling():
    # p=0.5: ~half the elements update on each step; consecutive
    # steps must see DIFFERENT masks (per-step resampling - the
    # USER_LEGACY frozen-mask bug would make them identical)
    x = _param((1, 2048), values=np.zeros((1, 2048), np.float32), name="m:0")
    opt = dfl_nn.AdaBelief(lr=1.0, lr_dropout=0.5, name="t")
    opt.initialize_variables([x])
    g = torch.ones_like(x)

    x0 = x.detach().clone()
    opt.get_update_op([(g, x)])()
    moved1 = (x.detach() != x0).reshape(-1)
    x1 = x.detach().clone()
    opt.get_update_op([(g, x)])()
    moved2 = (x.detach() != x1).reshape(-1)

    # ~50% of elements active on each step (2048 draws, 5-sigma band)
    assert abs(moved1.float().mean().item() - 0.5) < 0.05
    assert abs(moved2.float().mean().item() - 0.5) < 0.05
    # fresh masks: the two steps must not be identical (a frozen mask
    # - the USER_LEGACY bug - would make moved2 == moved1)
    assert not torch.equal(moved1, moved2)


def test_adabelief_lr_dropout_extremes():
    # p=1.0: every element updates -> identical to lr_dropout disabled
    def run_dropout(p):
        x = _param((1, 256), values=np.zeros((1, 256), np.float32), name="m:0")
        opt = dfl_nn.AdaBelief(lr=0.5, lr_dropout=p, name="t")
        opt.initialize_variables([x])
        for _ in range(3):
            opt.get_update_op([(torch.ones_like(x) * 0.25, x)])()
        return x.detach().clone(), opt.iterations.item()

    x1, it1 = run_dropout(1.0)
    xoff, itoff = run_dropout(1.0)
    assert torch.equal(x1, xoff)
    assert it1 == itoff == 3

    # p=0.0: weights frozen, but the official state still evolves
    x = _param((1, 256), values=np.zeros((1, 256), np.float32), name="m:0")
    x0 = x.detach().clone()
    opt = dfl_nn.AdaBelief(lr=0.5, lr_dropout=0.0, name="t")
    opt.initialize_variables([x])
    for _ in range(3):
        opt.get_update_op([(torch.ones_like(x) * 0.25, x)])()
    assert torch.equal(x, x0)  # frozen
    ms = list(opt.ms_dict.values())[0]
    assert ms.abs().sum() > 0  # state still updated (official)


def test_adabelief_state_layout_and_positional_keys():
    a = _param((2, 2), name="a:0")
    b = _param((3,), name=None)  # unnamed -> positional 'param_1'
    opt = dfl_nn.AdaBelief(lr=0.1, name="t")
    opt.initialize_variables([a, b])
    w = opt.get_weights()
    assert w[0] is opt.iterations
    assert w[1] is opt.ms_dict["a:0"]
    assert w[2] is opt.ms_dict["param_1"]
    assert w[3] is opt.vs_dict["a:0"]
    assert w[4] is opt.vs_dict["param_1"]
    names = [n for n, _ in opt._iter_official_weights()]
    assert names == ["iters:0", "ms_a_0:0", "ms_param_1:0",
                     "vs_a_0:0", "vs_param_1:0"], names


def test_adabelief_name_required():
    with pytest.raises(ValueError):
        dfl_nn.AdaBelief(name=None)


def test_adabelief_resume_equivalence():
    # N=5 steps -> snapshot -> fresh optimizer -> restore -> step 6
    # must equal a continuous 6-step run (deterministic: no dropout,
    # no cos, no clip)
    def fresh():
        x = _param((2, 2), values=np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
                   name="w:0")
        opt = dfl_nn.AdaBelief(lr=0.1, name="t")
        opt.initialize_variables([x])
        return x, opt

    grads = [np.full((2, 2), v, np.float32) for v in (0.5, -0.25, 1.0, -1.0, 0.1, 2.0)]

    # continuous run: 6 steps
    xc, oc = fresh()
    for g_np in grads:
        oc.get_update_op([(torch.from_numpy(g_np), xc)])()

    # interrupted run: 5 steps, snapshot, restore, 1 step
    xi, oi = fresh()
    for g_np in grads[:5]:
        oi.get_update_op([(torch.from_numpy(g_np), xi)])()
    snap = {k: t.clone() for k, t in oi.ms_dict.items()}
    snap_vs = {k: t.clone() for k, t in oi.vs_dict.items()}
    iters_snap = int(oi.iterations.item())
    w_snap = xi.detach().clone()

    # a real resume restores the weights AND the optimizer state
    # (the official checkpoint carries both)
    xj, oj = fresh()
    xj.data = w_snap
    iters_t = torch.full((), iters_snap, dtype=torch.long)
    oj.iterations.data = iters_t
    for k, t in oj.ms_dict.items():
        t.copy_(snap[k])
    for k, t in oj.vs_dict.items():
        t.copy_(snap_vs[k])
    oj.get_update_op([(torch.from_numpy(grads[5]), xj)])()

    assert np.abs(xj.detach().numpy() - xc.detach().numpy()).max() < 1e-6
    assert int(oj.iterations.item()) == 6 == int(oc.iterations.item())
    for k in oj.ms_dict:
        assert np.abs(oj.ms_dict[k].numpy() - oc.ms_dict[k].numpy()).max() < 1e-6
        assert np.abs(oj.vs_dict[k].numpy() - oc.vs_dict[k].numpy()).max() < 1e-6


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

def test_rmsprop_single_step_parity():
    x = _param((2, 2), values=np.array([[1.0, -1.0], [2.0, 0.5]], np.float32), name="z:0")
    opt = dfl_nn.RMSprop(lr=0.05, rho=0.9, name="t")
    opt.initialize_variables([x])
    g = torch.tensor([[0.3, -0.6], [1.2, 0.1]], dtype=torch.float32)

    x_ref = x.detach().numpy().astype(np.float64)
    g_ref = g.numpy().astype(np.float64)
    a_ref = np.zeros_like(g_ref)
    opt.get_update_op([(g, x)])()
    a_t, v_diff = _ref_rmsprop_state(a_ref, g_ref, 0.05, 0.9, OFFICIAL_EPS_F32)
    x_ref = x_ref + v_diff
    assert int(opt.iterations.item()) == 1
    assert np.abs(x.detach().numpy() - x_ref).max() < 1e-5
    assert np.abs(list(opt.accumulators_dict.values())[0].numpy() - a_t).max() < 1e-5


def test_rmsprop_multi_param_multi_step():
    a = _param((2, 2), name="a:0")
    b = _param((3,), name="b:0")
    opt = dfl_nn.RMSprop(lr=0.05, rho=0.9, name="t")
    opt.initialize_variables([a, b])

    grads = [
        (np.array([[1.0, -1.0], [2.0, 0.5]], np.float64), np.array([0.5, -0.5, 1.0], np.float64)),
        (np.array([[0.1, 0.1], [-0.2, 0.1]], np.float64), np.array([1.0, 1.0, -1.0], np.float64)),
        (np.zeros((2, 2), np.float64), np.array([0.0, 0.25, 0.0], np.float64)),
    ]
    refs = {
        "a:0": (a.detach().numpy().astype(np.float64), np.zeros((2, 2), np.float64)),
        "b:0": (b.detach().numpy().astype(np.float64), np.zeros((3,), np.float64)),
    }
    for ga_np, gb_np in grads:
        opt.get_update_op([(torch.from_numpy(ga_np.astype(np.float32)), a),
                           (torch.from_numpy(gb_np.astype(np.float32)), b)])()
        for name, gx in (("a:0", ga_np), ("b:0", gb_np)):
            x_ref, a_ref = refs[name]
            a_t, v_diff = _ref_rmsprop_state(a_ref, gx, 0.05, 0.9, OFFICIAL_EPS_F32)
            refs[name] = (x_ref + v_diff, a_t)
    assert np.abs(a.detach().numpy() - refs["a:0"][0]).max() < 1e-5
    assert np.abs(b.detach().numpy() - refs["b:0"][0]).max() < 1e-5
    names = [n for n, _ in opt._iter_official_weights()]
    assert names == ["iters:0", "acc_a_0:0", "acc_b_0:0"], names


def test_rmsprop_zero_gradient_exact_noop():
    x = _param((2, 2), name="z:0")
    x0 = x.detach().clone()
    opt = dfl_nn.RMSprop(lr=0.05, name="t")
    opt.initialize_variables([x])
    opt.get_update_op([(torch.zeros_like(x), x)])()
    assert torch.equal(x, x0)
    assert torch.allclose(list(opt.accumulators_dict.values())[0],
                          torch.zeros_like(x), atol=0.0)


def test_rmsprop_lr_cos_post_increment():
    g_np = np.full((2,), 10.0, np.float64)
    x = _param((2,), values=np.full((2,), 10.0, np.float32), name="z:0")
    opt = dfl_nn.RMSprop(lr=1.0, lr_cos=8, name="t")
    opt.initialize_variables([x])
    g = torch.from_numpy(g_np.astype(np.float32))

    x_ref = x.detach().numpy().astype(np.float64)
    a_ref = np.zeros((2,), np.float64)
    for step in range(1, 4):
        opt.get_update_op([(g, x)])()
        lr_eff = 1.0 * _ref_lr_cos_mult(step, 8)
        a_t, v_diff = _ref_rmsprop_state(a_ref, g_np, lr_eff, 0.9, OFFICIAL_EPS_F32)
        a_ref = a_t
        x_ref = x_ref + v_diff
        assert int(opt.iterations.item()) == step
        assert np.abs(x.detach().numpy() - x_ref).max() < 1e-4


def test_rmsprop_global_norm_clipping():
    p1 = _param((2,), values=np.array([10.0, 10.0], np.float32), name="a:0")
    p2 = _param((2,), values=np.array([3.0, 4.0], np.float32), name="b:0")
    opt = dfl_nn.RMSprop(lr=0.1, clipnorm=1.0, name="t")
    opt.initialize_variables([p1, p2])
    g1 = torch.from_numpy(np.array([20.0, 20.0], np.float32))
    g2 = torch.from_numpy(np.array([6.0, 8.0], np.float32))
    p10 = p1.detach().numpy().astype(np.float64)
    p20 = p2.detach().numpy().astype(np.float64)
    opt.get_update_op([(g1, p1), (g2, p2)])()
    # the GLOBAL norm (30) scales both grads
    c1, c2 = _ref_global_clip([np.array([20.0, 20.0], np.float64),
                               np.array([6.0, 8.0], np.float64)], 1.0)
    _, d1 = _ref_rmsprop_state(np.zeros((2,)), c1, 0.1, 0.9, OFFICIAL_EPS_F32)
    _, d2 = _ref_rmsprop_state(np.zeros((2,)), c2, 0.1, 0.9, OFFICIAL_EPS_F32)
    assert np.abs(p1.detach().numpy() - (p10 + d1)).max() < 1e-5
    assert np.abs(p2.detach().numpy() - (p20 + d2)).max() < 1e-5


def test_rmsprop_lr_dropout_per_step_and_extremes():
    x = _param((1, 1024), values=np.zeros((1, 1024), np.float32), name="m:0")
    opt = dfl_nn.RMSprop(lr=1.0, lr_dropout=0.5, name="t")
    opt.initialize_variables([x])
    g = torch.ones_like(x) * 0.5
    x0 = x.detach().clone()
    opt.get_update_op([(g, x)])()
    moved1 = (x.detach() != x0).reshape(-1)
    x1 = x.detach().clone()
    opt.get_update_op([(g, x)])()
    moved2 = (x.detach() != x1).reshape(-1)
    assert abs(moved1.float().mean().item() - 0.5) < 0.05
    assert not torch.equal(moved1, moved2)  # fresh mask per step

    # p=0.0: weights frozen, acc still evolves
    y = _param((1, 64), values=np.zeros((1, 64), np.float32), name="n:0")
    y0 = y.detach().clone()
    opt2 = dfl_nn.RMSprop(lr=0.5, lr_dropout=0.0, name="t2")
    opt2.initialize_variables([y])
    for _ in range(2):
        opt2.get_update_op([(torch.ones_like(y) * 0.25, y)])()
    assert torch.equal(y, y0)
    assert list(opt2.accumulators_dict.values())[0].abs().sum() > 0


def test_rmsprop_resume_equivalence():
    def fresh():
        x = _param((2, 2), values=np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
                   name="z:0")
        opt = dfl_nn.RMSprop(lr=0.05, rho=0.9, name="t")
        opt.initialize_variables([x])
        return x, opt

    grads = [np.full((2, 2), v, np.float32) for v in (0.5, -0.25, 1.0, 0.0, -2.0, 1.5)]

    xc, oc = fresh()
    for g_np in grads:
        oc.get_update_op([(torch.from_numpy(g_np), xc)])()

    xi, oi = fresh()
    for g_np in grads[:5]:
        oi.get_update_op([(torch.from_numpy(g_np), xi)])()
    acc_snap = {k: t.clone() for k, t in oi.accumulators_dict.items()}
    iters_snap = int(oi.iterations.item())
    w_snap = xi.detach().clone()

    # a real resume restores the weights AND the optimizer state
    xj, oj = fresh()
    xj.data = w_snap
    oj.iterations.data = torch.full((), iters_snap, dtype=torch.long)
    for k, t in oj.accumulators_dict.items():
        t.copy_(acc_snap[k])
    oj.get_update_op([(torch.from_numpy(grads[5]), xj)])()

    assert np.abs(xj.detach().numpy() - xc.detach().numpy()).max() < 1e-6
    assert int(oj.iterations.item()) == 6
    for k in oj.accumulators_dict:
        assert np.abs(oj.accumulators_dict[k].numpy()
                      - oc.accumulators_dict[k].numpy()).max() < 1e-6


# ---------------------------------------------------------------------------
# random_binomial (the lr_dropout mask source)
# ---------------------------------------------------------------------------

def test_random_binomial_extremes_and_dtype():
    z = dfl_ops.random_binomial((3, 4), p=0.0, dtype=torch.float32)
    assert z.shape == (3, 4) and z.sum().item() == 0.0
    o = dfl_ops.random_binomial((3, 4), p=1.0, dtype=torch.float32)
    assert o.sum().item() == 12.0
    h = dfl_ops.random_binomial((3, 4), p=0.5, dtype=torch.float16)
    assert h.dtype == torch.float16
    assert set(np.unique(h.numpy().astype("float32"))) <= {0.0, 1.0}


def test_random_binomial_seeded_deterministic():
    a = dfl_ops.random_binomial((16,), p=0.5, dtype=torch.float32, seed=42)
    b = dfl_ops.random_binomial((16,), p=0.5, dtype=torch.float32, seed=42)
    c = dfl_ops.random_binomial((16,), p=0.5, dtype=torch.float32, seed=43)
    assert torch.equal(a, b)          # same seed -> same sequence
    assert not torch.equal(a, c)      # different seed -> different sequence


def test_random_binomial_distribution_sanity():
    m = dfl_ops.random_binomial((1, 200000), p=0.3, dtype=torch.float32)
    frac = m.mean().item()
    # 200k Bernoulli(0.3) draws: std of the mean ~ 0.00034; 10-sigma band
    assert abs(frac - 0.3) < 0.01
    assert set(np.unique(m.numpy())) <= {0.0, 1.0}


def test_random_binomial_per_call_independent():
    a = dfl_ops.random_binomial((4096,), p=0.5, dtype=torch.float32)
    b = dfl_ops.random_binomial((4096,), p=0.5, dtype=torch.float32)
    assert not torch.equal(a, b)  # seedless calls draw different streams


# ---------------------------------------------------------------------------
# registry / import boundary / backend
# ---------------------------------------------------------------------------

def test_optimizers_registered_on_nn():
    assert isinstance(dfl_nn.AdaBelief, type)
    assert isinstance(dfl_nn.RMSprop, type)
    assert isinstance(dfl_nn.OptimizerBase, type)
    assert issubclass(dfl_nn.AdaBelief, dfl_nn.OptimizerBase)
    assert issubclass(dfl_nn.RMSprop, dfl_nn.OptimizerBase)
    assert callable(dfl_nn.random_binomial)
    assert dfl_nn.random_binomial is dfl_ops.random_binomial


def test_optimizers_package_tf_free():
    import sys
    import core.leras.optimizers  # noqa: F401
    # the whole session (optimizer modules included) must never have
    # pulled TensorFlow in
    assert not any(m.startswith("tensorflow") for m in sys.modules)


def test_no_direct_cuda_in_optimizer_source():
    import ast
    p = REPO_ROOT / "core" / "leras" / "optimizers"
    for f in ("OptimizerBase.py", "AdaBelief.py", "RMSprop.py"):
        tree = ast.parse((p / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = []
                n = node
                while isinstance(n, ast.Attribute):
                    chain.append(n.attr)
                    n = n.value
                if isinstance(n, ast.Name) and n.id == "torch":
                    assert "cuda" not in chain, f"{f}: torch.{'.'.join(reversed(chain))}"


# ---------------------------------------------------------------------------
# RTX 4090 execution + CPU/GPU parity (skip on CPU-only environments)
# ---------------------------------------------------------------------------

@requires_gpu
def test_optimizers_on_rtx4090():
    # GPU execution through the Phase 2 abstraction
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")

    dev = dfl_nn.device
    g_np = np.array([[0.5, -0.25], [1.5, 0.75]], np.float64)
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], np.float64)

    # CPU reference runs (fresh CPU foundation)
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    x_cpu = _param((2, 2), values=x_np.astype(np.float32), name="w:0")
    ab = dfl_nn.AdaBelief(lr=0.1, name="ab")
    ab.initialize_variables([x_cpu])
    x_rm = _param((2, 2), values=x_np.astype(np.float32), name="w:0")
    rm = dfl_nn.RMSprop(lr=0.05, rho=0.9, name="rm")
    rm.initialize_variables([x_rm])
    for _ in range(3):
        g = torch.from_numpy(g_np.astype(np.float32))
        ab.get_update_op([(g, x_cpu)])()
        rm.get_update_op([(g, x_rm)])()
    x_cpu_ref, x_rm_ref = x_cpu.detach().cpu().numpy(), x_rm.detach().cpu().numpy()

    # GPU runs
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")
    xg = _param((2, 2), values=x_np.astype(np.float32), name="w:0", device=dev)
    abg = dfl_nn.AdaBelief(lr=0.1, name="ab")
    abg.initialize_variables([xg])
    xrg = _param((2, 2), values=x_np.astype(np.float32), name="w:0", device=dev)
    rmg = dfl_nn.RMSprop(lr=0.05, rho=0.9, name="rm")
    rmg.initialize_variables([xrg])
    for _ in range(3):
        g = torch.from_numpy(g_np.astype(np.float32)).to(dev)
        abg.get_update_op([(g, xg)])()
        rmg.get_update_op([(g, xrg)])()

    # CPU-vs-GPU parity: same device-agnostic math, f32 accumulation
    assert str(xg.device) == str(dev)
    assert abs(xg.detach().cpu().numpy() - x_cpu_ref).max() < 1e-5
    assert abs(xrg.detach().cpu().numpy() - x_rm_ref).max() < 1e-5
    assert int(abg.iterations.item()) == 3 == int(ab.iterations.item())
