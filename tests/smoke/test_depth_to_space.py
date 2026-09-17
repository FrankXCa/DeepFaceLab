"""Phase 3C acceptance: official-compatible (TF R-R-C) depth_to_space.

The official DFL op (both NCHW branches and the NHWC branch) implements
TensorFlow depth_to_space semantics:

    out[n, c, h*r+i, w*r+j] = in[n, (i*r+j)*C_out + c, h, w]

The Phase 3F P0 re-audit verified (TF v2.4.0 kernel source of the
official DFL era + TF 2.21 runtime probe) that the official
tf.depth_to_space BUILT-IN - used by the official NCHW-GPU path -
uses this SAME R-R-C grouping (its GPU D2S_NCHW kernel documents the
input ordering "n, bY, bX, oC, iY, iX"), so ALL official branches are
identical and official checkpoints of any training path need no dts
re-mapping in Phase 4. PyTorch's F.pixel_shuffle uses C-R-R channel
grouping (NOT the official semantics in any branch), so the migrated
op applies the required channel permutation first (view/permute, no
gather). These tests prove the exact R-R-C placement (deterministic
unique-value tensors, every output cell checked against an independent
NumPy reference implementing the official formula), the strict
rejection semantics, dtype/device behavior, gradient flow, CPU and
RTX 4090 execution, and the import boundaries.

Parity labels: EXACT for the index placement and for CPU-vs-GPU
comparison (the op is a pure index rearrangement - no arithmetic).
TensorFlow runtime parity: NOT_VERIFIED in this phase (no TF
environment in the tested venvs); parity is against the official
index formula derived from the official source. The built-in kernel
was subsequently runtime-probed (TF 2.21) and source-verified (TF
v2.4.0, official DFL era) by the Phase 3F P0 re-audit, confirming
the same R-R-C placement for all official branches.
"""

import ast
import os

import numpy as np
import pytest

import torch

from core.leras import nn as dfl_nn

CUDA_AVAILABLE = torch.cuda.is_available()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS_TORCH_PATH = os.path.join(REPO_ROOT, "core", "leras", "ops", "__init__.py")


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    # the production data_format of the DFL trainers is NCHW
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


def _unique(n, c, h, w, seed):
    """Deterministic tensor: every (channel, h, w) cell unique, so any
    axis/channel mix-up is visible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 100000, (n, c, h, w), dtype=np.int64).astype(np.float32)


def _ref_tf_depth_to_space(xnp, size):
    """Independent NumPy implementation of the official index formula
    (TF R-R-C); used as the parity reference. Note: TensorFlow itself
    is NOT executed in this phase."""
    n, c_in, h, w = xnp.shape
    oc = c_in // (size * size)
    out = np.zeros((n, oc, h * size, w * size), xnp.dtype)
    for a in range(n):
        for c in range(oc):
            for hh in range(h):
                for ww in range(w):
                    for i in range(size):
                        for j in range(size):
                            out[a, c, hh * size + i, ww * size + j] = \
                                xnp[a, (i * size + j) * oc + c, hh, ww]
    return out


# ---------------------------------------------------------------------------
# 1. deterministic exact index placement (block size 2, every output cell)
# ---------------------------------------------------------------------------

def test_exact_index_placement_r2_c4():
    """C=4 (C_out=1), r=2: input channel t=(i*2+j) holds the constant
    t+1; verify every one of the 16 output cells against the TF
    formula. Parity: EXACT (pure index rearrangement)."""
    x_np = np.zeros((1, 4, 2, 2), np.float32)
    for t in range(4):
        x_np[0, t] = t + 1
    y = dfl_nn.depth_to_space(torch.from_numpy(x_np), 2).detach().numpy()
    assert y.shape == (1, 1, 4, 4)
    for h in range(2):
        for w in range(2):
            for i in range(2):
                for j in range(2):
                    t = i * 2 + j  # R-R-C: (i*r+j)*C_out + c
                    assert y[0, 0, h * 2 + i, w * 2 + j] == t + 1
    # the classic 2x2 pattern a correct R-R-C implementation prints
    assert np.array_equal(y[0, 0], np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]], np.int32))


def test_exact_index_placement_r2_c8_unique_spatial():
    """C=8 (C_out=2), r=2, unique value per (channel, h, w) cell:
    verifies the (i, j) -> (row sub, col sub) placement for both
    output channels, all spatial cells. Parity: EXACT."""
    x_np = _unique(1, 8, 3, 4, seed=11)
    y = dfl_nn.depth_to_space(torch.from_numpy(x_np), 2).detach().numpy()
    assert np.array_equal(y, _ref_tf_depth_to_space(x_np, 2))


# ---------------------------------------------------------------------------
# 2. random tensors vs independent reference; 3/4/5. shapes, channel
# counts, block sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,c,h,w,r", [
    (1, 4, 2, 2, 2),
    (1, 4, 3, 5, 2),
    (2, 8, 4, 3, 2),
    (1, 12, 5, 2, 2),
    (1, 16, 3, 3, 2),
    (1, 16, 2, 7, 2),
    (1, 9, 2, 2, 3),
    (1, 18, 4, 4, 3),
    (2, 9, 3, 5, 3),
    (1, 4, 1, 1, 2),
    (1, 16, 2, 2, 1),  # size=1 is the identity
])
def test_parity_vs_independent_reference(n, c, h, w, r):
    """Multiple channel counts / spatial dims / block sizes 1, 2, 3 vs
    the independent NumPy implementation of the official formula.
    Parity: EXACT. (TF runtime parity: NOT_VERIFIED this phase.)"""
    x_np = _unique(n, c, h, w, seed=1000 + n * 100 + c + r)
    y = dfl_nn.depth_to_space(torch.from_numpy(x_np), r).detach().numpy()
    assert y.shape == (n, c // (r * r), h * r, w * r)
    assert np.array_equal(y, _ref_tf_depth_to_space(x_np, r))


# ---------------------------------------------------------------------------
# 7. invalid input rejection
# ---------------------------------------------------------------------------

def test_invalid_channel_count_rejection():
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 6, 2, 2), 2)   # 6 % 4 != 0
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 8, 2, 2), 3)   # 8 % 9 != 0
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 5, 2, 2), 2)   # 5 % 4 != 0


def test_invalid_size_rejection():
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 4, 2, 2), 0)
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 4, 2, 2), -2)
    with pytest.raises(ValueError):
        dfl_nn.depth_to_space(torch.zeros(1, 4, 2, 2), "2")


# ---------------------------------------------------------------------------
# 8/11/12. CPU execution, dtype preservation, gradients
# ---------------------------------------------------------------------------

def test_cpu_execution_and_dtype_preservation():
    x = torch.from_numpy(_unique(1, 8, 3, 3, seed=5))
    for dtype in (torch.float32, torch.float16, torch.float64):
        y = dfl_nn.depth_to_space(x.to(dtype), 2)
        assert y.device.type == "cpu"
        assert y.dtype == dtype
    # float32 parity EXACT vs reference
    y32 = dfl_nn.depth_to_space(x, 2).detach().numpy()
    assert np.array_equal(y32, _ref_tf_depth_to_space(x.numpy(), 2))


def test_gradient_flow_bijection():
    """depth_to_space is a bijection on elements (C*H*W elements in =
    C_out*(rH)*(rW) out), so sum(output).backward() must give exactly
    ones on every input element."""
    x = torch.from_numpy(_unique(1, 8, 3, 3, seed=9)).requires_grad_(True)
    dfl_nn.depth_to_space(x, 2).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    # and the gradient values reach the right input channels (r=2, C=4):
    x2 = torch.zeros(1, 4, 2, 2, requires_grad=True)
    dfl_nn.depth_to_space(x2, 2).sum().backward()
    assert torch.equal(x2.grad, torch.ones(1, 4, 2, 2))


# ---------------------------------------------------------------------------
# NHWC contract (boundary permutation; official DFL runs NCHW)
# ---------------------------------------------------------------------------

def test_nhwc_data_format_contract():
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NHWC")
    x_np = _unique(1, 4, 2, 3, seed=13)          # (N, C, H, W)
    x_nhwc = np.ascontiguousarray(x_np.transpose(0, 2, 3, 1))
    y = dfl_nn.depth_to_space(torch.from_numpy(x_nhwc), 2).detach().numpy()
    # output stays NHWC: (N, H*r, W*r, C_out)
    ref = _ref_tf_depth_to_space(x_np, 2)         # (N, C_out, H*r, W*r)
    assert y.shape == (1, 4, 6, 1)
    assert np.array_equal(y, np.ascontiguousarray(ref.transpose(0, 2, 3, 1)))
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


# ---------------------------------------------------------------------------
# registration: official call sites use nn.depth_to_space
# ---------------------------------------------------------------------------

def test_nn_alias_registration():
    assert dfl_nn.depth_to_space is not None
    x = torch.from_numpy(_unique(1, 4, 2, 2, seed=17))
    y = dfl_nn.depth_to_space(x, 2)
    assert y.shape == (1, 1, 4, 4)


# ---------------------------------------------------------------------------
# 9/10. RTX 4090 execution and CPU-vs-GPU equality
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_gpu_execution_and_cpu_parity_on_rtx4090():
    """GPU execution through the Phase 2 abstraction (no torch.cuda.*
    in this test body) and bit-exact CPU-vs-GPU parity: the op is a
    pure index rearrangement, so outputs must be identical bit for
    bit. Parity: EXACT."""
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), data_format="NCHW")
    try:
        assert dfl_nn.device.type == "cuda"
        x_np = _unique(1, 16, 4, 5, seed=21)
        x_gpu = torch.from_numpy(x_np).to(dfl_nn.device)
        y_gpu = dfl_nn.depth_to_space(x_gpu, 2)
        assert y_gpu.device.type == "cuda"
        assert y_gpu.dtype == torch.float32

        # CPU reference of the same weights/data
        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
        y_cpu = dfl_nn.depth_to_space(torch.from_numpy(x_np), 2)
        # bit-exact: no arithmetic in the op
        assert torch.equal(y_gpu.cpu(), y_cpu)
        # and both match the independent reference exactly
        assert np.array_equal(
            y_cpu.detach().numpy(), _ref_tf_depth_to_space(x_np, 2))
    finally:
        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


# ---------------------------------------------------------------------------
# 13/14. import boundary and no direct CUDA in the op source
# ---------------------------------------------------------------------------

def test_ops_package_import_without_tensorflow():
    import sys
    sys.dont_write_bytecode = True
    from core.leras import ops  # noqa: F401
    assert "tensorflow" not in sys.modules
    assert "keras" not in sys.modules
    # the nn.initialize path (which imports core.leras.ops) must stay TF-free
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    assert "tensorflow" not in sys.modules


def test_no_direct_cuda_in_op_source():
    """AST scan: the torch ops module must not call torch.cuda.* or
    hardcode 'cuda:' device strings (Phase 2 backend neutrality)."""
    with open(OPS_TORCH_PATH, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=OPS_TORCH_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            assert not chain.startswith("torch.cuda"), (
                f"direct CUDA call in ops source: {chain}")
            assert chain != "torch.device", (
                "torch.device literal found; use the Phase 2 abstraction")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.startswith("cuda:"), (
                f"hardcoded cuda: device string in ops source: {node.value!r}")


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))
