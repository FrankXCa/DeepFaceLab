"""Backend-neutral device backend contract (Phase 2).

Model/leras code talks to ``Device`` / ``Devices`` / ``DeviceConfig``
(``core.leras.device``) and to the model-facing helpers there; it must not
call ``torch.cuda.*`` directly. Backends implement this contract and are
plugged in through the registry in ``core.leras.backends``.

Today: CUDA (NVIDIA) and CPU are implemented.
Later (IMPLEMENTATION_PLAN_v2.md sections 13 and 38A): AMD (ROCm) and
Intel (XPU) or other maintained GPU backends are added as new modules in
this package, validated as separate features, without touching model code.

Device index convention: the official DFL semantics are preserved —
index 0 of the first GPU backend is the primary device; CPU is not a
GPU device in DFL semantics (it is represented by an empty device list).
With more than one *GPU* backend active in the future, the
``NN_DEVICE_{i}_BACKEND`` environment key already disambiguates
backends; index pairing across multiple GPU backends is a future
contract extension (Phase 12 / 38A).
"""

import abc


class DeviceInfo(object):
    """Plain, picklable device record produced by a backend.

    Carries the backend-neutral metadata required by Phase 2:
    backend identity, vendor family, name, VRAM (bytes) and compute
    capability.
    """

    __slots__ = ('index', 'backend', 'family', 'name',
                 'total_mem', 'free_mem', 'capability')

    def __init__(self, index, backend, family, name, total_mem, free_mem,
                 capability=None):
        self.index = int(index)
        self.backend = backend        # e.g. 'cuda' (future: 'rocm', 'xpu')
        self.family = family          # e.g. 'nvidia', 'amd', 'intel', 'cpu'
        self.name = str(name)
        self.total_mem = int(total_mem)
        self.free_mem = int(free_mem)
        # (major, minor) compute capability, e.g. (8, 9) for sm_89;
        # None when the backend has no capability concept (CPU).
        self.capability = capability

    @property
    def total_mem_gb(self):
        return self.total_mem / 1024 ** 3

    @property
    def free_mem_gb(self):
        return self.free_mem / 1024 ** 3

    @property
    def capability_int(self):
        """Official DFL 'cc' encoding: major*10+minor (e.g. 89 for sm_89)."""
        if self.capability is None:
            return None
        return self.capability[0] * 10 + self.capability[1]

    def __repr__(self):
        return (f"DeviceInfo(index={self.index}, backend={self.backend!r}, "
                f"name={self.name!r}, capability={self.capability})")


class DeviceBackend(abc.ABC):
    """Contract every device backend must implement.

    ``enumerate_devices`` returns GPU-style devices (the things that show
    up in the DFL device list). CPU intentionally returns no devices: in
    official DFL semantics CPU is the *empty* device list, not a device
    entry.
    """

    #: registry key, e.g. 'cuda'
    name = None
    #: vendor family, e.g. 'nvidia'
    family = None

    @abc.abstractmethod
    def is_available(self):
        """True if this backend can be used on this machine right now."""
        raise NotImplementedError

    @abc.abstractmethod
    def enumerate_devices(self):
        """Return [DeviceInfo] for every device of this backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def torch_device(self, index):
        """Map a backend device index to a torch.device."""
        raise NotImplementedError

    def synchronize(self, device):
        """Synchronize work on ``device`` (a torch.device). Default: no-op."""
        return None

    def memory_info(self, index):
        """Return (free_bytes, total_bytes) or None if unknown."""
        return None
