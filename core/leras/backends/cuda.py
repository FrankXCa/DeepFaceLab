"""NVIDIA CUDA backend (torch.cuda) — the production GPU backend (Phase 2).

Device indices are the physical GPU indices as reported by the driver;
``initialize_main_env`` (core.leras.device) clears CUDA_VISIBLE_DEVICES
before enumeration, matching official DFL behavior, so indices are stable
across main process and child processes.

AMD (ROCm) and Intel (XPU) are NOT implemented here: they remain
architecture-ready targets (IMPLEMENTATION_PLAN_v2.md section 13 backend
support policy) and must land as separate validated features.
"""

import torch

from .base import DeviceBackend, DeviceInfo


class CudaBackend(DeviceBackend):
    name = 'cuda'
    family = 'nvidia'

    def __init__(self):
        # torch is a Phase 1 baseline dependency; import once.
        self._torch = torch

    def is_available(self):
        return self._torch.cuda.is_available()

    def enumerate_devices(self):
        t = self._torch
        if not t.cuda.is_available():
            return []
        result = []
        for i in range(t.cuda.device_count()):
            props = t.cuda.get_device_properties(i)
            total_mem = int(props.total_memory)
            free_mem = total_mem
            try:
                # real driver-reported free/total (better than TF's
                # session memory-limit artifact used by official DFL)
                free_mem, total_mem = (int(v) for v in t.cuda.mem_get_info(i))
            except Exception:
                pass  # keep properties-based values
            try:
                capability = tuple(int(v) for v in t.cuda.get_device_capability(i))
            except Exception:
                capability = None
            result.append(DeviceInfo(
                index=i,
                backend=self.name,
                family=self.family,
                name=props.name.strip(),
                total_mem=total_mem,
                free_mem=free_mem,
                capability=capability,
            ))
        return result

    def torch_device(self, index):
        return self._torch.device('cuda', index)

    def synchronize(self, device):
        self._torch.cuda.synchronize(device)

    def memory_info(self, index):
        free, total = self._torch.cuda.mem_get_info(index)
        return int(free), int(total)
