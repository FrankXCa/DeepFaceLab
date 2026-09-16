"""Phase 2 smoke tests: torch-native device layer (core.leras.device).

Covers the required Phase 2 validation areas:
  1. CPU-only initialization        (test_cpu_only_initialization*)
  2. CUDA availability detection    (test_cuda_availability_detection)
  3. RTX 4090 detection             (test_rtx4090_detection)
  4. GPU name reporting             (test_gpu_name_reporting)
  5. GPU index selection            (test_gpu_index_selection)
  6. VRAM reporting                 (test_vram_reporting)
  7. Invalid device index handling  (test_invalid_device_index_handling)
  8. Multi-GPU enumeration logic    (test_multi_gpu_*; MOCK backend,
                                      labeled — no multi-GPU hardware here)
  9. Backend metadata               (test_backend_metadata)
  10. Capability metadata           (test_capability_metadata)
  11. Model-facing allocation without direct CUDA calls
                                    (test_model_facing_allocation)
  12. No TensorFlow import in the device layer (test_no_tensorflow_import)

Plus: environment contract, Devices container API, nn.py alias
compatibility, and the nn.initialize_main_env() entry point.

Run in both validated environments:
  .\\.venv\\Scripts\\python    -m pytest tests/smoke/test_device.py -v   (CUDA)
  .\\.venv-cpu\\Scripts\\python -m pytest tests/smoke/test_device.py -v  (CPU;
  GPU tests are skipped automatically, CPU-only tests are REAL there)
"""

import os
import sys

import pytest
import torch

from core.leras import backends
from core.leras import device as device_layer
from core.leras.device import (
    Device,
    Devices,
    DeviceConfig,
    get_torch_device,
    synchronize,
    memory_info,
)

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

def _clear_device_env():
    for key in list(os.environ):
        if key.startswith('NN_DEVICE') or key == 'NN_DEVICES_INITIALIZED' or key == 'NN_DEVICES_COUNT':
            os.environ.pop(key)
    Devices.all_devices = None


def _ensure_initialized():
    """Real-device initialization (idempotent via NN_DEVICES_INITIALIZED)."""
    Devices.initialize_main_env()
    return Devices.getDevices()


@pytest.fixture(autouse=True)
def _restore_device_env():
    """Every test runs against the real device env; tests that rewrite it
    (CPU-only simulation, mock multi-GPU) restore it in their finally
    blocks, and this fixture re-establishes it after each test."""
    _clear_device_env()
    Devices.initialize_main_env()
    yield
    _clear_device_env()
    Devices.initialize_main_env()


def _mk_mock_info(index, gb):
    return backends.DeviceInfo(
        index=index, backend='mock', family='mockvendor',
        name=('MockGPU Pro' if gb == 8 else 'MockGPU Max'),
        total_mem=gb * 1024 ** 3, free_mem=gb * 1024 ** 3 // 2,
        capability=(7, 5),
    )


def _register_mock_backend():
    """Structural multi-GPU fixture: a fake backend with 3 devices
    (two equal 8 GB 'MockGPU Pro', one 24 GB 'MockGPU Max').
    MOCK — validates enumeration/container logic only, no real hardware."""

    class MockBackend(backends.DeviceBackend):
        name = 'mock'
        family = 'mockvendor'

        def is_available(self):
            return True

        def enumerate_devices(self):
            return [_mk_mock_info(0, 8), _mk_mock_info(1, 8), _mk_mock_info(2, 24)]

        def torch_device(self, index):
            return torch.device('cpu')  # mock never allocates

    backend = MockBackend()
    backends.register(backend)
    return backend


# ---------------------------------------------------------------------------
# 12. no TensorFlow anywhere in the device layer
# ---------------------------------------------------------------------------

def test_no_tensorflow_import():
    # Importing the device layer + backends and running the full
    # initialization/enumeration flow must never import TensorFlow.
    import core.leras.device  # noqa: F401
    import core.leras.backends  # noqa: F401

    assert 'tensorflow' not in sys.modules

    Devices.initialize_main_env()
    Devices.getDevices()

    assert 'tensorflow' not in sys.modules


# ---------------------------------------------------------------------------
# 2. CUDA availability detection
# ---------------------------------------------------------------------------

def test_cuda_availability_detection():
    backend = backends.get_backend('cuda')
    # Neutral API must agree with the torch reference.
    assert backend.is_available() == torch.cuda.is_available()


def test_backend_registry_contents():
    names = {b.name for b in backends.iter_backends()}
    assert {'cuda', 'cpu'} <= names
    # CPU is always available and enumerates no GPU devices (DFL semantics).
    cpu = backends.get_backend('cpu')
    assert cpu.is_available() is True
    assert cpu.enumerate_devices() == []


# ---------------------------------------------------------------------------
# 3./4. RTX 4090 detection + GPU name
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_rtx4090_detection():
    devices = _ensure_initialized()
    assert len(devices) >= 1
    names = [d.name for d in devices]
    assert any('4090' in n for n in names), f"RTX 4090 not detected; got: {names}"
    # primary device representation: BestGPU of a single-GPU machine
    best = DeviceConfig.BestGPU().devices[0]
    assert best.name in names


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_gpu_name_reporting():
    devices = _ensure_initialized()
    for device in devices:
        assert device.name.strip() != ""
        # cross-check against the torch reference API (test-only)
        assert device.name == torch.cuda.get_device_name(device.index).strip()


# ---------------------------------------------------------------------------
# 5. GPU index selection (+ official BestGPU/WorstGPU semantics)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_gpu_index_selection():
    devices = _ensure_initialized()
    assert len(devices) >= 1

    cfg = DeviceConfig.GPUIndexes([0])
    assert not cfg.cpu_only
    assert cfg.devices[0].index == 0
    assert cfg.devices[0].name == devices[0].name

    # selection preserves the requested index order (official behavior)
    if len(devices) > 1:
        cfg2 = DeviceConfig.GPUIndexes([devices[1].index, devices[0].index])
        assert [d.index for d in cfg2.devices] == [devices[1].index, devices[0].index]


# ---------------------------------------------------------------------------
# 6. VRAM reporting
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_vram_reporting():
    devices = _ensure_initialized()
    for device in devices:
        assert device.total_mem > 0
        assert device.free_mem >= 0
        assert device.free_mem <= device.total_mem
        assert device.total_mem_gb == pytest.approx(device.total_mem / 1024 ** 3)

        # cross-check totals against the torch reference (properties)
        props = torch.cuda.get_device_properties(device.index)
        assert device.total_mem == pytest.approx(int(props.total_memory), rel=0.05)


def test_model_facing_memory_info():
    devices = _ensure_initialized()
    if len(devices) == 0:
        # CPU environment: memory_info(None) is best-effort
        info = memory_info(None)
        if info is not None:
            free, total = info
            assert total > 0
        return
    device = devices[0]
    info = memory_info(device)
    assert info is not None
    free, total = info
    assert total > 0
    assert free <= total


# ---------------------------------------------------------------------------
# 7. invalid device index handling (official semantics preserved)
# ---------------------------------------------------------------------------

def test_invalid_device_index_handling():
    devices = _ensure_initialized()

    # container lookup: unknown index -> None (official behavior)
    assert devices.get_device_by_index(9999) is None

    # official DeviceConfig.GPUIndexes maps unknown indexes to the empty
    # device list (== CPU fallback); documented Phase 2 behavior
    cfg = DeviceConfig.GPUIndexes([9999])
    assert len(cfg.devices) == 0
    assert cfg.cpu_only

    # valid and invalid mixed: only valid ones survive (official behavior)
    if len(devices) > 0:
        cfg2 = DeviceConfig.GPUIndexes([devices[0].index, 9999])
        assert [d.index for d in cfg2.devices] == [devices[0].index]


# ---------------------------------------------------------------------------
# 8. multi-GPU enumeration logic — STRUCTURAL MOCK (no multi-GPU hardware)
# ---------------------------------------------------------------------------

def test_multi_gpu_enumeration_logic():
    """MOCK-based: exercises enumeration + container logic with 3 fake
    GPUs. Labeled structural — real multi-GPU enumeration is validated
    by the same registry path on multi-GPU machines (Phase 12)."""
    backend = _register_mock_backend()
    saved_cuda = backends.unregister('cuda')  # isolate the mock set
    try:
        _clear_device_env()
        Devices.initialize_main_env()
        devices = Devices.getDevices()

        assert len(devices) == 3
        assert [d.index for d in devices] == [0, 1, 2]
        assert all(d.backend == 'mock' for d in devices)

        best = devices.get_best_device()
        assert best.index == 2                      # 24 GB MockGPU Max
        worst = devices.get_worst_device()
        assert worst.index in (0, 1)                # 8 GB (first found)
        assert worst.total_mem == 8 * 1024 ** 3

        equal = devices.get_equal_devices(devices[0])
        assert [d.index for d in equal] == [0, 1]   # two identical 'MockGPU Pro'

        at_least = devices.get_devices_at_least_mem(16)
        assert [d.index for d in at_least] == [2]

        # container API
        assert len(devices) == 3
        assert devices[0] is devices.devices[0]
        sliced = devices[1:]
        assert isinstance(sliced, Devices)
        assert [d.index for d in sliced] == [1, 2]
        assert [d.index for d in list(devices)] == [0, 1, 2]

        # selection through DeviceConfig on the mock set
        cfg = DeviceConfig.GPUIndexes([0, 2])
        assert [d.index for d in cfg.devices] == [0, 2]
        assert not cfg.cpu_only
        assert DeviceConfig.CPU().cpu_only
    finally:
        backends.unregister('mock')
        if saved_cuda is not None:
            backends.register(saved_cuda)
        _clear_device_env()


# ---------------------------------------------------------------------------
# 9. backend metadata
# ---------------------------------------------------------------------------

def test_backend_metadata():
    devices = _ensure_initialized()
    if len(devices) == 0:
        # CPU environment: metadata contract still holds (no devices),
        # and the registry exposes the backend identities.
        assert backends.get_backend('cuda').family == 'nvidia'
        assert backends.get_backend('cpu').family == 'cpu'
        return

    for device in devices:
        assert device.backend is not None
        assert device.backend in {b.name for b in backends.iter_backends()}
        assert backends.get_backend(device.backend).family is not None

    # environment contract carries the backend key
    for i in range(len(devices)):
        assert os.environ.get(f'NN_DEVICE_{i}_BACKEND') is not None
    assert os.environ['NN_DEVICE_0_BACKEND'] == devices[0].backend

    if CUDA_AVAILABLE:
        assert devices[0].backend == 'cuda'


# ---------------------------------------------------------------------------
# 10. capability metadata
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_capability_metadata():
    devices = _ensure_initialized()
    for device in devices:
        assert device.capability is not None
        # cross-check against the torch reference API (test-only)
        ref = torch.cuda.get_device_capability(device.index)
        assert (device.capability[0], device.capability[1]) == (int(ref[0]), int(ref[1]))
        # env contract: official 'cc' encoding major*10+minor
        cc = int(os.environ[f'NN_DEVICE_{device.index}_CC'])
        assert cc == device.capability[0] * 10 + device.capability[1]


# ---------------------------------------------------------------------------
# 11. model-facing allocation without direct CUDA calls
# ---------------------------------------------------------------------------

def test_model_facing_allocation():
    """The model-facing API maps devices to torch.device and allocates
    through it — no torch.cuda.* call appears in this test body."""
    devices = _ensure_initialized()

    # CPU side (always valid): None == CPU in DFL semantics
    cpu_device = get_torch_device(None)
    assert cpu_device.type == 'cpu'
    x_cpu = torch.zeros(2, 3, device=cpu_device)
    assert x_cpu.device == cpu_device
    synchronize(None)  # CPU: no-op

    if len(devices) == 0:
        return

    device = DeviceConfig.BestGPU().devices[0]
    tdev = get_torch_device(device)
    assert tdev.type == 'cuda'
    assert tdev.index == device.index

    x = torch.zeros(8, 8, device=tdev)
    y = x * 2 + 1          # allocation + compute through the abstraction
    synchronize(device)    # stream sync through the abstraction
    assert y.device == tdev

    # torch.device passthrough
    assert get_torch_device(tdev) is tdev


# ---------------------------------------------------------------------------
# CPU-only initialization
# ---------------------------------------------------------------------------

def test_cpu_only_initialization():
    """CPU-only mode: with no GPU backend available the device list is
    empty and every DeviceConfig factory falls back to CPU (official
    semantics). In .venv-cpu this is a REAL CPU-only run; in the CUDA
    environment the GPU backend is structurally hidden (labeled)."""
    saved_cuda = backends.unregister('cuda')  # returns the backend instance
    try:
        _clear_device_env()
        Devices.initialize_main_env()
        devices = Devices.getDevices()
        assert len(devices) == 0

        assert DeviceConfig.CPU().cpu_only
        assert DeviceConfig.BestGPU().cpu_only      # empty -> CPU (official)
        assert DeviceConfig.WorstGPU().cpu_only
        assert DeviceConfig.GPUIndexes([0]).cpu_only

        assert get_torch_device(None).type == 'cpu'
        synchronize(None)
    finally:
        if saved_cuda is not None:
            backends.register(saved_cuda)
        _clear_device_env()


# ---------------------------------------------------------------------------
# environment contract + entry point + nn aliases
# ---------------------------------------------------------------------------

def test_env_contract_and_entry_point():
    # the exact entry point main.py uses (Phase 1 bootstrap line)
    from core.leras import nn as dfl_nn
    dfl_nn.initialize_main_env()

    assert os.environ.get('NN_DEVICES_INITIALIZED') == '1'
    devices = Devices.getDevices()
    count = int(os.environ['NN_DEVICES_COUNT'])
    assert count == len(devices)

    for i, device in enumerate(devices):
        assert os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'] == 'GPU'
        assert os.environ[f'NN_DEVICE_{i}_NAME'] == device.name
        assert int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']) == device.total_mem
        assert int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']) == device.free_mem


def test_nn_aliases_keep_call_sites_working():
    # Phase 2 compatibility: nn.DeviceConfig / nn.ask_choose_device_idxs
    # must be the device-layer objects (all mainscripts call nn.*).
    from core.leras import nn as dfl_nn

    assert dfl_nn.DeviceConfig is DeviceConfig
    assert dfl_nn.ask_choose_device_idxs is device_layer.ask_choose_device_idxs
    assert dfl_nn.DeviceConfig.CPU().cpu_only
    cfg = dfl_nn.DeviceConfig.BestGPU()
    assert isinstance(cfg.devices, Devices)


# ---------------------------------------------------------------------------
# Devices container API (official behavior preserved)
# ---------------------------------------------------------------------------

def test_devices_container_api():
    devices = _ensure_initialized()
    if len(devices) == 0:
        # CPU environment: run the container logic on constructed devices
        base = [Device(i, 'GPU', f'Fake{i}', (i + 1) * 1024 ** 3, (i + 1) * 1024 ** 3 // 2,
                       backend='mock', capability=(7, 0))
                for i in range(3)]
    else:
        base = [d for d in devices] + [
            Device(90, 'GPU', 'FakeExtra', 4 * 1024 ** 3, 2 * 1024 ** 3,
                   backend='mock', capability=(7, 0)),
        ]
    ds = Devices(base)

    assert len(ds) == len(base)
    assert ds[0].index == base[0].index
    assert isinstance(ds[1:], Devices)
    assert [d.index for d in ds] == [d.index for d in base]
    assert ds.get_device_by_index(base[0].index) is base[0]
    assert ds.get_device_by_index(12345) is None
    assert [d.index for d in ds.get_devices_from_index_list([base[-1].index])] == [base[-1].index]
    assert ds.get_equal_devices(base[0]) is not None
    big = ds.get_devices_at_least_mem(base[-1].total_mem / 1024 ** 3)
    assert base[-1] in [d for d in big]
