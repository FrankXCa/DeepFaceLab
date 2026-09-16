"""CPU backend (Phase 2) — fallback / smoke path.

In official DFL semantics CPU is not a device entry: CPU mode is the
empty device list (DeviceConfig.cpu_only). The CPU backend therefore
enumerates no devices; it provides the torch.device mapping, no-op
synchronization and best-effort system memory reporting used by the
model-facing helpers in core.leras.device.
"""

import os

import torch

from .base import DeviceBackend, DeviceInfo  # noqa: F401  (DeviceInfo used by the contract)


class CpuBackend(DeviceBackend):
    name = 'cpu'
    family = 'cpu'

    def is_available(self):
        return True

    def enumerate_devices(self):
        return []

    def torch_device(self, index=None):
        return torch.device('cpu')

    def synchronize(self, device):
        return None  # CPU needs no stream synchronization

    def memory_info(self, index=None):
        try:
            total = int(os.sysconf('SC_PAGE_SIZE')) * int(os.sysconf('SC_PHYS_PAGES'))
            return None, total
        except (ValueError, OSError, AttributeError):
            return None
