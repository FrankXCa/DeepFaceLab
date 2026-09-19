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

    def autocast_context(self):
        """A context manager wrapping the safe-forward regions in
        this plan's dtype (nullcontext for ``off``)."""
        if not self.enabled:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device_type,
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


def resolve_precision(mode, device):
    """Resolve a ``(mode, device)`` pair into a :class:`PrecisionPlan`.

    ``device`` is the model's torch device (``nn.device`` after
    ``nn.initialize``).  Raises ``PrecisionUnsupportedError`` for
    an unknown mode or an unsupported mode/device combination —
    the explicit-failure contract of Milestone E.
    """
    if mode not in VALID_PRECISION_MODES:
        raise PrecisionUnsupportedError(
            f"precision {mode!r} is not a valid mode "
            f"(valid: {' | '.join(VALID_PRECISION_MODES)})")

    if mode == MODE_OFF:
        # the FP32 baseline: no capability requirements, no
        # scaler, no autocast — available on every device
        return PrecisionPlan(MODE_OFF, device.type)

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
                return PrecisionPlan(MODE_FP16, 'cuda',
                                     torch.float16, scaler_required=True)
            raise PrecisionUnsupportedError(
                f"precision 'fp16' is unsupported on this CUDA device: "
                f"capability {cap[0]}.{cap[1]} < "
                f"{_FP16_MIN_CAP[0]}.{_FP16_MIN_CAP[1]} (fp16 autocast "
                f"requires Maxwell or newer)")
        # mode == MODE_BF16
        if _cap_ge(cap, _BF16_MIN_CAP):
            return PrecisionPlan(MODE_BF16, 'cuda', torch.bfloat16)
        raise PrecisionUnsupportedError(
            f"precision 'bf16' is unsupported on this CUDA device: "
            f"capability {cap[0]}.{cap[1]} < "
            f"{_BF16_MIN_CAP[0]}.{_BF16_MIN_CAP[1]} (bf16 requires "
            f"Ampere or newer — torch.cuda.is_bf16_supported() is False)")

    if device.type == 'cpu':
        if mode == MODE_BF16:
            # torch CPU autocast supports bfloat16 (plan v2 §30
            # "CPU path" capability check) — no scaler
            return PrecisionPlan(MODE_BF16, 'cpu', torch.bfloat16)
        raise PrecisionUnsupportedError(
            f"precision 'fp16' is unsupported on CPU: torch CPU "
            f"autocast has no fp16 dtype (use 'off' or 'bf16')")

    raise PrecisionUnsupportedError(
        f"precision {mode!r} is unsupported on device type "
        f"{device.type!r}")
