"""Phase 1 runtime baseline smoke tests.

Validate the modern Python/PyTorch dependency baseline (IMPLEMENTATION_PLAN.md
section 12 acceptance): Python floor, torch import/version, CUDA availability,
GPU enumeration, CPU fallback, tensor allocation/ops, NumPy version and
compatibility (including NumPy 2 adoption and cross-version checkpoint loading).

Run in the CUDA environment:
    .venv/Scripts/python -m pytest tests/smoke -v
Run in the CPU-only environment (GPU tests skip automatically):
    .venv-cpu/Scripts/python -m pytest tests/smoke -v

These tests exercise the runtime only; no DeepFaceLab source module is
imported (the torch-only core does not exist yet — that is Phase 3).
"""

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from np1_fixture import build_np1_fixture_dict

NP1_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "numpy1_checkpoint_fixture.npy"


def _probe_dir_lifecycle(root):
    """True if a directory in ``root`` can be created (os.makedirs),
    written to, listed, and removed.

    Some sandboxed development environments deny file operations inside
    directories created through ``tempfile``'s open-based creation, and
    deny pytest's Windows ``\\\\?\\`` extended-length basetemp handling
    entirely. This probe exercises the exact lifecycle the fixture uses.
    """
    name = "dfl_probe_%s" % "".join(random.choices("0123456789abcdef", k=8))
    d = os.path.join(root, name)
    try:
        os.makedirs(d)
        probe = os.path.join(d, "probe")
        with open(probe, "wb"):
            pass
        list(os.scandir(d))
        os.unlink(probe)
        os.rmdir(d)
        return True
    except OSError:
        try:
            os.rmdir(d)
        except OSError:
            pass
        return False


@pytest.fixture
def plain_tmp():
    """Temporary directory (makedirs-based) that works in sandboxed
    environments.

    Prefers the platform temp area; falls back to the git-ignored
    ``<repo>/.pytest-tmp`` directory when the temp area is unusable.
    Cleanup is lenient: leftovers in the platform temp area are removed
    by the OS, and workspace leftovers are git-ignored.
    """
    root = tempfile.gettempdir()
    if not _probe_dir_lifecycle(root):
        root = str(Path(__file__).resolve().parents[2] / ".pytest-tmp")
        os.makedirs(root, exist_ok=True)
    name = "dfl_roundtrip_%s" % "".join(random.choices("0123456789abcdef", k=8))
    d = os.path.join(root, name)
    os.makedirs(d)
    yield d
    for entry in os.listdir(d):
        try:
            os.unlink(os.path.join(d, entry))
        except OSError:
            pass
    try:
        os.rmdir(d)
    except OSError:
        pass


def test_python_version_floor():
    # Phase 1 drops Python 3.6/3.8 support; baseline is 3.11+ (tested 3.12).
    assert sys.version_info >= (3, 11)


def test_torch_import_and_version():
    """L0: torch imports and reports a version."""
    import torch

    assert torch.__version__
    # CPU-only wheels report no CUDA runtime; CUDA wheels report e.g. "13.0".
    assert torch.version.cuda is None or isinstance(torch.version.cuda, str)


def test_cuda_enumeration():
    if not torch_cuda_available():
        pytest.skip("no CUDA device in this environment (CPU-only variant)")
    import torch

    assert torch.version.cuda is not None, "CUDA build must report its CUDA runtime"
    assert torch.cuda.device_count() >= 1
    name = torch.cuda.get_device_name(0)
    assert name, "GPU name must be reported"
    capability = torch.cuda.get_device_capability(0)
    assert isinstance(capability, tuple) and len(capability) == 2
    assert capability[0] >= 8, "baseline targets modern (Turing+) GPUs"
    print(f"\ncuda device: {name}, capability sm_{capability[0]}{capability[1]}, "
          f"CUDA runtime {torch.version.cuda}")


def test_cpu_tensor_allocation_and_ops():
    """CPU fallback path: allocation and a basic op must work."""
    import torch

    torch.manual_seed(0)
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    c = a @ b
    assert c.device.type == "cpu"
    assert c.shape == (64, 64)
    assert torch.allclose(c, a @ b)


def test_cuda_tensor_allocation_and_ops():
    if not torch_cuda_available():
        pytest.skip("no CUDA device in this environment (CPU-only variant)")
    import torch

    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.is_cuda and c.shape == (64, 64)
    # cross-device fp32 matmul comparison keeps a generous tolerance
    assert torch.allclose(c.cpu(), a.cpu() @ b.cpu(), atol=1e-4)
    free, total = torch.cuda.mem_get_info()
    print(f"\ncuda tensor ops ok; device memory {total / 2**20:.0f} MiB total, "
          f"{free / 2**20:.0f} MiB free")


def torch_cuda_available():
    import torch

    return torch.cuda.is_available()


def test_numpy_version():
    # Phase 1 explicitly adopts NumPy 2 after the compatibility checks
    # below (and the numpy1 cross-version fixture test) pass.
    assert np.__version__.startswith("2."), (
        f"baseline pins numpy 2.x, found {np.__version__}")


def test_torch_numpy_interop():
    import torch

    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    t = torch.from_numpy(a)
    assert t.dtype == torch.float32
    assert np.array_equal(t.numpy(), a)
    # shared storage: tensor writes are visible in the numpy array
    t[0, 0] = -1.0
    assert a[0, 0] == -1.0


def test_numpy2_checkpoint_pickle_roundtrip(plain_tmp):
    """DFL-style pickled-dict checkpoint survives save/load under NumPy 2."""
    payload = build_np1_fixture_dict()
    path = Path(plain_tmp) / "model_checkpoint.npy"
    np.save(str(path), payload, allow_pickle=True)
    loaded = np.load(str(path), allow_pickle=True).item()
    assert set(loaded) == set(payload)
    for key in payload:
        assert loaded[key].dtype == payload[key].dtype
        assert loaded[key].shape == payload[key].shape
        assert np.array_equal(loaded[key], payload[key])


def test_legacy_numpy1_fixture_load():
    """Checkpoint pickled by numpy 1.x loads byte-exact under NumPy 2.

    The fixture is generated by generate_numpy1_fixture.py in a numpy<2
    environment (.venv-np1). Absent fixture => skipped, not failed: this
    is a cross-version validation artifact, not part of the runtime smoke.
    """
    if not NP1_FIXTURE_PATH.exists():
        pytest.skip(
            "numpy1 fixture absent — run tests/smoke/generate_numpy1_fixture.py "
            "in a numpy<2 environment first")
    loaded = np.load(str(NP1_FIXTURE_PATH), allow_pickle=True).item()
    expected = build_np1_fixture_dict()
    assert set(loaded) == set(expected)
    for key in expected:
        assert np.array_equal(loaded[key], expected[key]), key
