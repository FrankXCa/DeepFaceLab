"""Device layer (Phase 2) — torch-native replacement of the official
TensorFlow-based GPU discovery.

Architecture (IMPLEMENTATION_PLAN_v2.md section 13):

    model / leras code
            |
    Device / Devices / DeviceConfig   (this module, official API)
            |
    backend-neutral registry (core.leras.backends)
            |
    CUDA (today)   CPU (today)   AMD / Intel (future, this layer only)

Preserved official contract:
- ``initialize_main_env()`` publishes ``NN_DEVICES_INITIALIZED`` /
  ``NN_DEVICES_COUNT`` / ``NN_DEVICE_{i}_*`` environment variables so
  spawned child processes can rebuild the device list without re-running
  any discovery.
- ``CUDA_VISIBLE_DEVICES`` is cleared before enumeration (official
  behavior: users select from all physical GPUs).
- CPU is represented by an *empty* device list
  (``DeviceConfig.cpu_only``), exactly like official DFL.
- ``Device`` keeps its official fields/constructor
  (``index, tf_dev_type, name, total_mem, free_mem``) plus Phase 2
  additions: ``backend`` and ``capability``.

Documented deviations from the official TF discovery:
- No TensorFlow and no helper subprocess: torch enumerates devices
  in-process (cheap driver queries, no framework context pinning).
- DirectML (DML) devices are not enumerated: the torch baseline has no
  DirectML backend; production targets are NVIDIA CUDA + CPU.
- ``free_mem`` reports real driver values (torch.cuda.mem_get_info);
  official TF reported the session memory limit (under allow_growth that
  is effectively the total).
- New environment keys ``NN_DEVICE_{i}_BACKEND`` and ``NN_DEVICE_{i}_CC``
  (capability, official 'cc' encoding major*10+minor) are added; the
  official keys are unchanged, so older readers keep working.

Model-facing helpers (use these instead of torch.cuda.* in model code):
``get_torch_device(device)``, ``synchronize(device)``,
``memory_info(device)``.
"""

import os
import sys
from pathlib import Path

import torch

from . import backends


def _log_info(message):
    """Use the official logger when the full app environment is importable
    (core.interact pulls in GUI/opencv dependencies); fall back to a plain
    line so this layer stays importable in the minimal Phase 1 runtime."""
    try:
        from core.interact import interact as io
        io.log_info(message)
    except Exception:
        print(f"[INFO] {message}")


class Device(object):
    def __init__(self, index, tf_dev_type, name, total_mem, free_mem,
                 backend=None, capability=None):
        self.index = index
        # Kept for official compatibility: TF-era model code builds
        # '/{tf_dev_type}:{index}' placement strings from it.
        self.tf_dev_type = tf_dev_type
        self.name = name

        self.total_mem = total_mem
        self.total_mem_gb = total_mem / 1024**3
        self.free_mem = free_mem
        self.free_mem_gb = free_mem / 1024**3

        # Phase 2 metadata:
        #   backend    - registry key, e.g. 'cuda' (future: 'rocm', 'xpu')
        #   capability - (major, minor) e.g. (8, 9) for sm_89; None if unknown
        self.backend = backend
        self.capability = capability

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"


class Devices(object):
    all_devices = None

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append(device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb*(1024**3):
                result.append(device)
        return Devices(result)

    @staticmethod
    def initialize_main_env():
        if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 0:
            return

        # Official behavior: enumerate all physical GPUs.
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')

        # Windows: NVIDIA driver JIT compute cache. This is a driver-level
        # cache (still honored by the torch CUDA runtime), kept from the
        # official behavior. TF-only variables from the official file are
        # intentionally not set.
        if sys.platform[0:3] == 'win':
            appdata = os.environ.get('APPDATA')
            if appdata:
                compute_cache_path = Path(appdata) / 'NVIDIA' / ('ComputeCache_ALL')
                os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
                if not compute_cache_path.exists():
                    _log_info("Caching GPU kernels...")
                    compute_cache_path.mkdir(parents=True, exist_ok=True)

        # Enumerate through the backend-neutral registry (in-process;
        # the official TF subprocess is not needed for torch).
        devices = []
        for backend in backends.available_gpu_backends():
            for info in backend.enumerate_devices():
                devices.append(info)
        devices.sort(key=lambda info: info.index)

        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(devices))

        for i, info in enumerate(devices):
            os.environ[f'NN_DEVICE_{i}_BACKEND'] = info.backend
            # 'GPU' keeps the official key/value for TF-era placement code.
            os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'] = 'GPU'
            os.environ[f'NN_DEVICE_{i}_NAME'] = info.name
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(info.total_mem)
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(info.free_mem)
            if info.capability is not None:
                os.environ[f'NN_DEVICE_{i}_CC'] = str(info.capability_int)

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in main process.")

            devices = []
            for i in range(int(os.environ['NN_DEVICES_COUNT'])):
                cc_env = os.environ.get(f'NN_DEVICE_{i}_CC')
                capability = None
                if cc_env is not None:
                    cc = int(cc_env)
                    capability = (cc // 10, cc % 10)
                devices.append(
                    Device(
                        index=i,
                        tf_dev_type=os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'],
                        name=os.environ[f'NN_DEVICE_{i}_NAME'],
                        total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                        free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']),
                        backend=os.environ.get(f'NN_DEVICE_{i}_BACKEND', 'cuda'),
                        capability=capability,
                    )
                )
            Devices.all_devices = Devices(devices)

        return Devices.all_devices


def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True,
                           suggest_best_multi_gpu=False, suggest_all_gpu=False):
    """Official DFL interactive device prompt, moved here from nn.py
    (Phase 2): DeviceConfig/selection now lives with the device layer.
    Behavior is unchanged (same prompts, defaults, validation loop)."""
    devices = Devices.getDevices()
    if len(devices) == 0:
        return []

    all_devices_indexes = [device.index for device in devices]

    if choose_only_one:
        suggest_best_multi_gpu = False
        suggest_all_gpu = False

    if suggest_all_gpu:
        best_device_indexes = all_devices_indexes
    elif suggest_best_multi_gpu:
        best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device())]
    else:
        best_device_indexes = [devices.get_best_device().index]
    best_device_indexes = ",".join([str(x) for x in best_device_indexes])

    _log_info("")
    if choose_only_one:
        _log_info("Choose one GPU idx.")
    else:
        _log_info("Choose one or several GPU idxs (separated by comma).")
    _log_info("")

    if allow_cpu:
        _log_info("[CPU] : CPU")
    for device in devices:
        _log_info(f"  [{device.index}] : {device.name}")

    _log_info("")

    from core.interact import interact as io

    while True:
        try:
            if choose_only_one:
                choosed_idxs = io.input_str("Which GPU index to choose?", best_device_indexes)
            else:
                choosed_idxs = io.input_str("Which GPU indexes to choose?", best_device_indexes)

            if allow_cpu and choosed_idxs.lower() == "cpu":
                choosed_idxs = []
                break

            choosed_idxs = [int(x) for x in choosed_idxs.split(',')]

            if choose_only_one:
                if len(choosed_idxs) == 1:
                    break
            else:
                if all([idx in all_devices_indexes for idx in choosed_idxs]):
                    break
        except Exception:
            pass
    _log_info("")

    return choosed_idxs


class DeviceConfig():
    """Official device-selection semantics (moved from nn.py in Phase 2;
    nn.DeviceConfig remains an alias so existing call sites are untouched)."""

    @staticmethod
    def ask_choose_device(*args, **kwargs):
        return DeviceConfig.GPUIndexes(ask_choose_device_idxs(*args, **kwargs))

    def __init__(self, devices=None):
        # Official semantics: devices is always a Devices container
        # (model code calls e.g. device_config.devices.get_worst_device()).
        devices = devices or []
        if not isinstance(devices, Devices):
            devices = Devices(devices)

        self.devices = devices
        self.cpu_only = len(devices) == 0

    @staticmethod
    def CPU():
        return DeviceConfig([])

    @staticmethod
    def BestGPU():
        devices = Devices.getDevices()
        if len(devices) == 0:
            return DeviceConfig.CPU()

        return DeviceConfig([devices.get_best_device()])

    @staticmethod
    def WorstGPU():
        devices = Devices.getDevices()
        if len(devices) == 0:
            return DeviceConfig.CPU()

        return DeviceConfig([devices.get_worst_device()])

    @staticmethod
    def GPUIndexes(indexes):
        if len(indexes) != 0:
            devices = Devices.getDevices().get_devices_from_index_list(indexes)
        else:
            devices = []

        return DeviceConfig(devices)


# --- model-facing device API (Phase 2) ---------------------------------
# Model/leras code should use these instead of torch.cuda.is_available()/
# get_device_name()/mem_get_info()/synchronize(); the backend is resolved
# through the registry, so future non-CUDA backends only touch this layer.

def get_backend(device):
    """Resolve the backend for a Device or torch.device.

    ``None`` or a CPU torch.device resolve to the CPU backend (DFL
    semantics: empty device list == CPU).
    """
    if device is None:
        return backends.get_backend('cpu')
    if isinstance(device, Device):
        return backends.get_backend(device.backend)
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            return backends.get_backend('cuda')
        if device.type == 'cpu':
            return backends.get_backend('cpu')
    raise TypeError(f"unsupported device for backend resolution: {device!r}")


def get_torch_device(device):
    """Map a Device (or torch.device, or None == CPU) to a torch.device
    without any CUDA-specific call in the caller."""
    backend = get_backend(device)
    if device is None:
        return backend.torch_device(None)
    if isinstance(device, Device):
        return backend.torch_device(device.index)
    return device  # already a torch.device


def synchronize(device):
    """Synchronize work on the given Device/torch.device (CPU: no-op)."""
    backend = get_backend(device)
    if device is None or isinstance(device, Device):
        index = None if device is None else device.index
        backend.synchronize(backend.torch_device(index))
    else:
        backend.synchronize(device)


def memory_info(device):
    """Return (free_bytes, total_bytes) for the device, or None if the
    backend cannot report it."""
    backend = get_backend(device)
    index = None if device is None else device.index
    return backend.memory_info(index)
