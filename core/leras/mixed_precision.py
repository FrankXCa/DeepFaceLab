"""Phase 8 — mixed-precision training foundation.

The official DFL ``use_fp16`` option is an *export*-time dtype knob
only (export conv dtype + graph boundary casts); official training
is always fp32.  This module adds a modern PyTorch mixed-precision
*training* policy as a NEW feature (IMPLEMENTATION_PLAN_v2 §30 "FP32
master weights / BF16 or FP16 autocast for safe forward operations /
losses in FP32 / GAN-sensitive computations in FP32", §48 "FP16
later, BF16 later" delivered here, §49 Milestone E "no silent
fallback"):

- precision modes ``off | fp16 | bf16`` — a runtime-only
  constructor parameter of ``ModelBase`` (the same channel as
  ``cpu_only`` / ``force_gpu_idxs``); NOT a persisted model option
  (Phase 7 D-6: no new option keys, no extra state file), not an
  io prompt, not in ``data.dat``;
- ``off`` (FP32) is always available and the default; the FP32
  path is the byte-for-byte Phase 6B/7 code path (nullcontext +
  no scaler);
- ``fp16``: ``torch.amp.autocast(dtype=float16)`` over the safe
  forward regions; requires CUDA capability >= (5,3); a
  ``torch.amp.GradScaler`` with the native torch defaults
  (init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
  growth_interval=2000) is REQUIRED; fp16 is not available on
  CPU (torch CPU autocast has no fp16 dtype);
- ``bf16``: ``torch.amp.autocast(dtype=bfloat16)``; requires CUDA
  capability >= (8,0) — the same rule
  ``torch.cuda.is_bf16_supported()`` implements — or a CPU
  device; NO GradScaler (bf16 carries the fp32 exponent range);
- an unsupported mode/device combination raises
  ``PrecisionUnsupportedError`` (a ``ValueError``) explicitly at
  training start — NEVER a silent downgrade (Milestone E; the
  USER_LEGACY silent bf16->fp16->fp32 fallback chain and the
  EXTERNAL_A bf16-default/None->True fallbacks are rejected
  deviations);
- Phase 12 (multi-replica): :func:`resolve_precision_devices`
  validates the requested mode against EVERY selected replica
  device (the replica plan's ordered device list; entry 0 = the
  primary) and raises ``PrecisionUnsupportedError`` naming EVERY
  failing device — an unsupported mode on ANY selected device
  fails training start explicitly (Milestone E: there is no
  silent per-device fallback or downgrade); the returned plan is
  the ONE global ``PrecisionPlan`` of the run (its
  ``device_type`` is the primary device's type; each replica's
  autocast region pins its OWN device type via
  ``PrecisionPlan.autocast_context(device_type=...)``); the
  fp16 plan's native ``GradScaler`` remains the ONE global
  scaler of the run (Phase 12 §6: never one scaler per replica
  or per device);
- FP32 master weights + FP32 optimizer states: parameters are
  never dtype-converted; autocast downcasts forward op math only;
  losses are computed in the FP32 island after a boundary
  ``.to(torch.float32)``; the GAN step stays entirely fp32
  (the official GAN never receives an fp16 treatment);
- checkpoints are mode-independent: NO new persisted state in any
  mode (scaler state is never saved; a fresh native scaler is
  created when training resumes in fp16).

Provenance: NEW_PHASE8_DESIGN (plan v2 §30) with the PyTorch-
native autocast+GradScaler machinery; USER_LEGACY (explicit FP32
loss island, fp16/scaler pairing) and EXTERNAL_A (bf16 autocast
without scaler) corroborate individual choices — the full
per-feature classification lives in docs/PHASE8_PLAN.md §18
(private planning doc; the public contract is this module's
docstring + the plan v2 anchors quoted above).
"""

from contextlib import nullcontext

import torch

MODE_OFF = 'off'
MODE_FP16 = 'fp16'
MODE_BF16 = 'bf16'
VALID_PRECISION_MODES = (MODE_OFF, MODE_FP16, MODE_BF16)

# CUDA compute-capability floors (the torch autocast rules):
# fp16 tensor cores exist from Maxwell (5,3); bf16 (the rule
# torch.cuda.is_bf16_supported() implements) from Ampere (8,0).
_FP16_MIN_CAP = (5, 3)
_BF16_MIN_CAP = (8, 0)


def _cap_ge(cap, floor):
    return cap[0] > floor[0] or (cap[0] == floor[0] and cap[1] >= floor[1])


class PrecisionUnsupportedError(ValueError):
    """Explicit training-start failure for an unsupported
    precision mode/device combination.

    Milestone E (plan v2 §49): there is NO silent fallback — an
    unsupported request fails loudly at start with the mode, the
    device, and the missing capability named."""


class PrecisionPlan(object):
    """The resolved precision policy for one device.

    Created once per training session by :func:`resolve_precision`
    (lazily, on the first training step); immutable afterwards.
    ``off`` plans carry no autocast dtype and no scaler; the
    autocast context manager then degenerates to ``nullcontext``
    and the per-step scaler helpers are no-ops, so the FP32 path
    is literally the unchanged Phase 6B/7 code path.
    """

    def __init__(self, mode, device_type, autocast_dtype=None,
                 scaler_required=False):
        self.mode = mode
        self.device_type = device_type
        self.autocast_dtype = autocast_dtype
        self.scaler_required = scaler_required

    @property
    def enabled(self):
        return self.autocast_dtype is not None

    def autocast_context(self, device_type=None):
        """A context manager wrapping the safe-forward regions in
        this plan's dtype (nullcontext for ``off``).

        ``device_type`` (optional) pins the autocast to one
        replica's device type: in a multi-replica run the RUN-WIDE
        plan's ``device_type`` is the primary's, while each
        replica's forward runs on ITS OWN device — pass the
        replica's device type when it differs from the primary's
        (e.g. a CPU replica under a CUDA-primary bf16 plan).
        Omitted: the plan's own device type (the exact single-
        device Phase 8 behavior)."""
        if device_type is None:
            device_type = self.device_type
        if not self.enabled:
            return nullcontext()
        return torch.amp.autocast(device_type=device_type,
                                  dtype=self.autocast_dtype)

    def make_scaler(self):
        """The native ``torch.amp.GradScaler`` for fp16 (torch
        defaults, no custom configuration — Phase 7 D-6: no new
        option keys); ``None`` for off/bf16 (bf16 forbids a
        scaler)."""
        if not self.scaler_required:
            return None
        return torch.amp.GradScaler(self.device_type)

    def describe(self):
        if not self.enabled:
            return f"{self.mode} (FP32)"
        d = (f"autocast {self.autocast_dtype} on {self.device_type}"
             + (" + GradScaler (torch defaults)"
                if self.scaler_required else " (no scaler)"))
        return f"{self.mode} ({d})"


def _validate_mode(mode):
    """Milestone E: an unknown mode is an explicit error (never a
    silent 'off' / 'auto' interpretation)."""
    if mode not in VALID_PRECISION_MODES:
        raise PrecisionUnsupportedError(
            f"precision {mode!r} is not a valid mode "
            f"(valid: {' | '.join(VALID_PRECISION_MODES)})")


def _device_capability(mode, device):
    """Per-device capability check shared by the single-device and
    all-device resolutions.

    Returns the plan triple ``(device_type, autocast_dtype,
    scaler_required)`` for this device. Raises
    ``PrecisionUnsupportedError`` (naming the mode and device) for
    an unsupported combination — the caller decides whether that
    ends the whole run (single device) or is one of the failing
    devices to report (all-device resolution).
    """
    if mode == MODE_OFF:
        # the FP32 baseline: no capability requirements, no
        # scaler, no autocast — available on every device
        return device.type, None, False

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise PrecisionUnsupportedError(
                f"precision {mode!r} is unsupported on this machine: "
                "CUDA is not available (fp16/bf16 training requires a "
                "CUDA device; use 'off' or 'bf16' on CPU)")
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        if mode == MODE_FP16:
            if _cap_ge(cap, _FP16_MIN_CAP):
                return 'cuda', torch.float16, True
            raise PrecisionUnsupportedError(
                f"precision 'fp16' is unsupported on this CUDA device: "
                f"capability {cap[0]}.{cap[1]} < "
                f"{_FP16_MIN_CAP[0]}.{_FP16_MIN_CAP[1]} (fp16 autocast "
                f"requires Maxwell or newer)")
        # mode == MODE_BF16
        if _cap_ge(cap, _BF16_MIN_CAP):
            return 'cuda', torch.bfloat16, False
        raise PrecisionUnsupportedError(
            f"precision 'bf16' is unsupported on this CUDA device: "
            f"capability {cap[0]}.{cap[1]} < "
            f"{_BF16_MIN_CAP[0]}.{_BF16_MIN_CAP[1]} (bf16 requires "
            f"Ampere or newer — torch.cuda.is_bf16_supported() is False)")

    if device.type == 'cpu':
        if mode == MODE_BF16:
            # torch CPU autocast supports bfloat16 (plan v2 §30
            # "CPU path" capability check) — no scaler
            return 'cpu', torch.bfloat16, False
        raise PrecisionUnsupportedError(
            f"precision 'fp16' is unsupported on CPU: torch CPU "
            f"autocast has no fp16 dtype (use 'off' or 'bf16')")

    raise PrecisionUnsupportedError(
        f"precision {mode!r} is unsupported on device type "
        f"{device.type!r}")


def resolve_precision(mode, device):
    """Resolve a ``(mode, device)`` pair into a :class:`PrecisionPlan`.

    ``device`` is the model's torch device (``nn.device`` after
    ``nn.initialize``).  Raises ``PrecisionUnsupportedError`` for
    an unknown mode or an unsupported mode/device combination —
    the explicit-failure contract of Milestone E.
    """
    _validate_mode(mode)
    device_type, autocast_dtype, scaler_required = _device_capability(
        mode, device)
    return PrecisionPlan(mode, device_type, autocast_dtype, scaler_required)


def resolve_precision_devices(mode, devices):
    """Resolve the requested precision mode against EVERY selected
    device (Phase 12 §6 all-device capability validation).

    ``devices`` is the ordered replica device list (entry 0 = the
    primary replica device = the Phase 2 ``nn.device`` primary
    contract). The requested mode must be supported on EVERY
    selected device: an unsupported mode/device combination raises
    ``PrecisionUnsupportedError`` naming EVERY failing device —
    there is NO silent per-device fallback or downgrade (a
    heterogeneous Pascal+Ampere pair requesting bf16 is an explicit
    start failure, never a per-device downgrade; Milestone E).

    Returns the ONE global :class:`PrecisionPlan` of the run
    (``device_type`` = the primary device's type; each replica's
    autocast region pins its own device type via
    :meth:`PrecisionPlan.autocast_context`); the fp16 plan carries
    the one global ``GradScaler`` requirement (never one scaler per
    replica or per device).

    Raises:
        PrecisionUnsupportedError: unknown mode, or the mode is
            unsupported on any selected device (every failing
            device is named).
        ValueError: empty device list.
        TypeError: a selected device is not a ``torch.device``.
    """
    devices = list(devices)
    if not devices:
        raise ValueError(
            "resolve_precision_devices: devices must be a non-empty list "
            "of torch.device (entry 0 = the primary replica device)")
    for d in devices:
        if not isinstance(d, torch.device):
            raise TypeError(
                f"resolve_precision_devices: every selected device must be "
                f"a torch.device (got {type(d).__name__})")
    _validate_mode(mode)
    failures = []
    for d in devices:
        try:
            _device_capability(mode, d)
        except PrecisionUnsupportedError as err:
            failures.append(f"device {d}: {err}")
    if failures:
        raise PrecisionUnsupportedError(
            f"precision {mode!r} is unsupported on {len(failures)} of the "
            f"{len(devices)} selected device(s) — every selected replica "
            f"device must support the requested mode; there is NO silent "
            f"per-device fallback or downgrade (Milestone E):\n  - "
            + "\n  - ".join(failures))
    device_type, autocast_dtype, scaler_required = _device_capability(
        mode, devices[0])
    return PrecisionPlan(mode, device_type, autocast_dtype, scaler_required)
