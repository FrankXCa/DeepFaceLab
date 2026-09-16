"""Phase 3E1 acceptance: low-level leras ops with official semantics.

Covers:
- flatten        - the official channel-major (-1, C*H*W) flatten in
                   the NCHW layout (NHWC input is boundary-permuted
                   first; the legacy/A spatial-major flattening of an
                   NHWC input is rejected)
- reshape_4D     - the official (-1, c, h, w) NCHW-layout
                   interpretation of the flat tail (NHWC output under
                   the NHWC data_format); invalid tails fail
                   explicitly (no permissive shape heuristics)
- average_tensor_list - official single-element identity /
                   expand_dims+concat+reduce_mean semantics
- total_variation_mse - the official VERBATIM formula (axis-1 and
                   axis-2 slice differences, squared, SUMMED over
                   axes 1..3 -> per-sample (N,) vector). The official
                   formula is data-format-agnostic: under NCHW the
                   axis-1 term is a CHANNEL-difference term (the
                   official baseline quirk that the SAEHD/AMP GAN
                   loss actually computed - reproduced, not fixed).
                   The test suite explicitly distinguishes this from
                   External A's variant (spatial axes + global scalar
                   MEAN).

Parity labels (vs the independent NumPy implementations of the
official formulas): all inputs use small integer-valued floats so
the sums are exactly representable in float32 and the comparisons
are EXACT (bit-level); CPU-vs-GPU parity is likewise exact for the
deterministic integer cases. TensorFlow runtime parity:
NOT_VERIFIED in this phase (no TF environment in the tested venvs).
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
    reason="Phase 3E1 GPU test: RTX 4090 / CUDA device required",
)


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    """Every test runs with a CPU + NCHW foundation unless it
    initializes something else explicitly."""
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")


def _unique(n, c, h, w, dtype=np.float32):
    """Small integer-valued tensor with unique values per element:
    sums of squared differences stay exact in float32 (|values| <
    2^24), so reference comparisons can be bit-exact."""
    a = np.arange(n * c * h * w, dtype=np.float64).reshape(n, c, h, w)
    return a.astype(dtype)


# ---------------------------------------------------------------------------
# independent NumPy references of the official formulas
# ---------------------------------------------------------------------------

def _ref_flatten_nchw(x_nchw):
    # official: reshape(x, (-1, prod(shape[1:]))) after the NHWC->NCHW
    # boundary transpose (already applied by the caller in the NHWC case)
    return x_nchw.reshape(x_nchw.shape[0], -1)


def _ref_average(tensors):
    # official: reduce_mean(concat([expand_dims(t,0) ...], 0), 0)
    if len(tensors) == 1:
        return tensors[0]
    return np.stack(tensors, axis=0).mean(axis=0)


def _ref_total_variation_mse(images):
    # OFFICIAL formula verbatim (format-agnostic slicing):
    #   dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]
    #   dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]
    #   out  = sum(square(dif1), axes 1..3) + sum(square(dif2), axes 1..3)
    dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
    return (np.square(dif1).sum(axis=(1, 2, 3))
            + np.square(dif2).sum(axis=(1, 2, 3)))


def _ref_total_variation_mse_external_a(images):
    # EXTERNAL A deviation (rejected): spatial axes only (NCHW) +
    # global scalar MEAN, no batch dimension:
    #   dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    #   dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    #   out = mean(dx*dx) + mean(dy*dy)
    dy = images[:, :, 1:, :] - images[:, :, :-1, :]
    dx = images[:, :, :, 1:] - images[:, :, :, :-1]
    return float(np.mean(dx * dx) + np.mean(dy * dy))


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

def test_flatten_exact_placement_nchw():
    x = _unique(2, 3, 4, 5)
    t = torch.from_numpy(x)
    y = dfl_ops.flatten(t)
    assert y.shape == (2, 3 * 4 * 5)
    assert y.dtype == torch.float32
    # channel-major placement, exact (integer values)
    ref = _ref_flatten_nchw(x)
    assert np.array_equal(y.detach().numpy(), ref)


def test_flatten_dtype_preserved():
    x = _unique(1, 2, 3, 4, dtype=np.float64)
    y = dfl_ops.flatten(torch.from_numpy(x))
    assert y.dtype == torch.float64
    assert y.shape == (1, 24)


def test_flatten_nhwc_channel_major():
    # an NHWC input must still flatten channel-major: the official
    # boundary-transposes to NCHW first (legacy torch.flatten(x,
    # start_dim=1) and External A's NCHW-only reshape deviate here)
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NHWC")
    n, h, w, c = 2, 3, 4, 2
    x = _unique(n, c, h, w).transpose(0, 2, 3, 1).astype(np.float32)
    y = dfl_ops.flatten(torch.from_numpy(x))
    assert y.shape == (n, c * h * w)
    # channel-major: y[n, c*h*w + h*w + w] == x[n, h, w, c]
    flat = y.detach().numpy().reshape(n, c, h, w)
    assert np.array_equal(flat, x.transpose(0, 3, 1, 2))


def test_flatten_gradient_flow():
    t = torch.from_numpy(_unique(1, 2, 3, 4)).requires_grad_()
    y = dfl_ops.flatten(t)
    y.sum().backward()
    assert torch.allclose(t.grad, torch.ones_like(t))


# ---------------------------------------------------------------------------
# reshape_4D
# ---------------------------------------------------------------------------

def test_reshape_4d_exact_placement_nchw():
    c, h, w = 3, 4, 5
    x = _unique(2, c, h, w)
    flat = torch.from_numpy(x.reshape(2, -1))
    y = dfl_ops.reshape_4D(flat, w, h, c)
    assert y.shape == (2, c, h, w)
    # channel-major: y[n,c,h,w] == flat[n, c*h*w + h*w + w]
    assert np.array_equal(y.detach().numpy(), x)


def test_reshape_4d_nhwc_output():
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NHWC")
    c, h, w = 2, 3, 5
    x = _unique(2, c, h, w)  # (N, C, H, W) values, channel-major tail
    flat = torch.from_numpy(x.reshape(2, -1))
    y = dfl_ops.reshape_4D(flat, w, h, c)
    assert y.shape == (2, h, w, c)
    assert np.array_equal(y.detach().numpy(), x.transpose(0, 2, 3, 1))


def test_reshape_4d_invalid_tail_rejected():
    # the flat tail must hold exactly c*h*w elements per sample -
    # the op fails explicitly (official tf: InvalidArgumentError)
    flat = torch.from_numpy(np.zeros((2, 10), np.float32))
    with pytest.raises((RuntimeError, ValueError)):
        dfl_ops.reshape_4D(flat, 2, 2, 3)  # needs 12 per sample


def test_reshape_4d_gradient_flow():
    flat = torch.from_numpy(_unique(1, 2, 3, 4).reshape(1, -1)).requires_grad_()
    y = dfl_ops.reshape_4D(flat, 4, 3, 2)
    y.sum().backward()
    assert torch.allclose(flat.grad, torch.ones_like(flat))


# ---------------------------------------------------------------------------
# average_tensor_list
# ---------------------------------------------------------------------------

def test_average_tensor_list_single_identity():
    t = torch.from_numpy(_unique(1, 2, 3, 4))
    out = dfl_ops.average_tensor_list([t])
    assert out is t  # official: single element returned as-is


def test_average_tensor_list_mean_reference():
    a = _unique(2, 3, 4, 5)
    b = (a + 1).astype(np.float32)
    # two integer-valued tensors: (a+b)/2 is exactly representable
    out = dfl_ops.average_tensor_list(
        [torch.from_numpy(a), torch.from_numpy(b)])
    ref = _ref_average([a, b])
    assert out.shape == a.shape
    assert np.array_equal(out.detach().numpy(), ref)


def test_average_tensor_list_signature_parity():
    # the official tf_device_string parameter is accepted (unused
    # under torch: placement comes from the input tensors)
    a = _unique(1, 2, 3, 4)
    out = dfl_ops.average_tensor_list([torch.from_numpy(a)], "cpu")
    assert np.array_equal(out.detach().numpy(), a)


def test_average_tensor_list_gradient_flow():
    a = torch.from_numpy(_unique(2, 2, 3, 4)).requires_grad_()
    b = torch.from_numpy((_unique(2, 2, 3, 4) + 3).astype(np.float32)).requires_grad_()
    loss = dfl_ops.average_tensor_list([a, b]).sum()
    loss.backward()
    assert torch.allclose(a.grad, 0.5 * torch.ones_like(a))
    assert torch.allclose(b.grad, 0.5 * torch.ones_like(b))


# ---------------------------------------------------------------------------
# total_variation_mse
# ---------------------------------------------------------------------------

def test_total_variation_mse_nchw_official_formula():
    x = _unique(2, 3, 4, 5)
    out = dfl_ops.total_variation_mse(torch.from_numpy(x))
    ref = _ref_total_variation_mse(x)
    # per-sample (N,) vector of SUMS, exact (integer values)
    assert out.shape == (2,)
    assert np.array_equal(out.detach().numpy(), ref)


def test_total_variation_mse_distinguishes_external_a_variant():
    # official: (N,) per-sample SUM over the official axis-1/axis-2
    # slices (axis 1 = channel axis under NCHW);
    # External A: scalar global MEAN over spatial slices only.
    # The official reference and the A variant are both computed for
    # the same input and pinned apart explicitly.
    x = _unique(2, 2, 3, 4)
    official = _ref_total_variation_mse(x)
    a_variant = _ref_total_variation_mse_external_a(x)

    out = dfl_ops.total_variation_mse(torch.from_numpy(x))
    assert out.shape == (2,)  # (N,), NOT a scalar
    assert np.array_equal(out.detach().numpy(), official)
    # the A variant must not even match numerically
    assert abs(float(official[0]) - a_variant) > 1.0
    assert abs(float(official[1]) - a_variant) > 1.0


def test_total_variation_mse_nhwc_spatial_terms():
    # under NHWC the same official formula slices the spatial axes
    # (axis 1 = H), i.e. the "intended" spatial total variation
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NHWC")
    n, h, w, c = 2, 4, 5, 3
    x = _unique(n, c, h, w).transpose(0, 2, 3, 1).astype(np.float32)
    out = dfl_ops.total_variation_mse(torch.from_numpy(x))
    ref = _ref_total_variation_mse(x)
    assert out.shape == (n,)
    assert np.array_equal(out.detach().numpy(), ref)


def test_total_variation_mse_zero_input():
    x = np.zeros((2, 3, 4, 5), np.float32)
    out = dfl_ops.total_variation_mse(torch.from_numpy(x))
    assert out.shape == (2,)
    assert np.allclose(out.detach().numpy(), 0.0, atol=0.0)


def test_total_variation_mse_gradient_smoke():
    t = torch.from_numpy(_unique(1, 3, 4, 5)).requires_grad_()
    loss = dfl_ops.total_variation_mse(t).sum()
    loss.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()
    assert t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# aliases / registration
# ---------------------------------------------------------------------------

def test_lowlevel_ops_registered_on_nn_and_module():
    for name in ("flatten", "reshape_4D", "average_tensor_list",
                 "total_variation_mse"):
        assert callable(getattr(dfl_nn, name)), name
        assert getattr(dfl_ops, name) is getattr(dfl_nn, name), name


# ---------------------------------------------------------------------------
# RTX 4090 execution + CPU/GPU parity (skip on CPU-only environments)
# ---------------------------------------------------------------------------

@requires_gpu
def test_lowlevel_ops_on_rtx4090():
    # GPU device picked through the Phase 2 abstraction
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")

    x = _unique(2, 3, 4, 5)
    cpu = dfl_nn.device if str(dfl_nn.device) == "cpu" else torch.device("cpu")

    # flatten / reshape_4D: exact layout ops -> bit-exact CPU/GPU
    y_gpu = dfl_ops.flatten(torch.from_numpy(x).to(dfl_nn.device))
    y_cpu = dfl_ops.flatten(torch.from_numpy(x).to(cpu))
    assert np.array_equal(y_gpu.detach().cpu().numpy(), y_cpu.detach().numpy())

    r_gpu = dfl_ops.reshape_4D(y_gpu, 5, 4, 3)
    r_cpu = dfl_ops.reshape_4D(y_cpu, 5, 4, 3)
    assert np.array_equal(r_gpu.detach().cpu().numpy(), r_cpu.detach().numpy())

    a = _unique(2, 3, 4, 5)
    b = (a + 1).astype(np.float32)
    m_gpu = dfl_ops.average_tensor_list(
        [torch.from_numpy(a).to(dfl_nn.device),
         torch.from_numpy(b).to(dfl_nn.device)])
    m_cpu = dfl_ops.average_tensor_list(
        [torch.from_numpy(a).to(cpu), torch.from_numpy(b).to(cpu)])
    assert np.array_equal(m_gpu.detach().cpu().numpy(), m_cpu.detach().numpy())

    # total_variation_mse: exact integer sums -> bit-exact parity
    t_gpu = dfl_ops.total_variation_mse(torch.from_numpy(x).to(dfl_nn.device))
    t_cpu = dfl_ops.total_variation_mse(torch.from_numpy(x).to(cpu))
    assert np.array_equal(t_gpu.detach().cpu().numpy(), t_cpu.detach().numpy())
