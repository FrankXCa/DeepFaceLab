"""Device backend registry (Phase 2).

Explicit, code-level registration — there is deliberately no environment
variable switching (no DFL_BACKEND pattern, IMPLEMENTATION_PLAN_v2.md
section 13). Adding a future backend (AMD ROCm, Intel XPU, ...) means:

1. add ``core/leras/backends/<name>.py`` implementing ``DeviceBackend``;
2. register it below (one line);
3. validate it as a separate feature (tests, compatibility status).

No model/leras code outside ``core/leras`` needs to change.
"""

from .base import DeviceBackend, DeviceInfo
from .cpu import CpuBackend
from .cuda import CudaBackend

_BACKENDS = {}


def _register_default_backends():
    register(CudaBackend())
    register(CpuBackend())


def register(backend):
    """Register a backend under ``backend.name`` (replaces on conflict)."""
    if not isinstance(backend, DeviceBackend):
        raise TypeError(f"not a DeviceBackend: {backend!r}")
    _BACKENDS[backend.name] = backend
    return backend


def unregister(name):
    return _BACKENDS.pop(name, None)


def get_backend(name):
    try:
        return _BACKENDS[name]
    except KeyError:
        raise KeyError(f"unknown device backend {name!r}; "
                       f"registered: {sorted(_BACKENDS)}") from None


def iter_backends():
    for backend in _BACKENDS.values():
        yield backend


def gpu_backends():
    """All registered backends that enumerate GPU-style devices."""
    for backend in _BACKENDS.values():
        if backend.name != CpuBackend.name:
            yield backend


def available_gpu_backends():
    """GPU backends usable on this machine right now."""
    for backend in gpu_backends():
        if backend.is_available():
            yield backend


_register_default_backends()

__all__ = [
    'DeviceBackend', 'DeviceInfo',
    'CudaBackend', 'CpuBackend',
    'register', 'unregister', 'get_backend',
    'iter_backends', 'gpu_backends', 'available_gpu_backends',
]
