"""Checkpoint conversion engine (Phase 4).

Centralized, explicit, testable compatibility bridge between the
official DeepFaceLab TF-era checkpoint format and the modernized torch
implementation — the converter foundation that later model phases
(SAEHD/AMP/XSeg/Quick96, Phases 6-8) plug into.

Authoritative source concepts (independent reimplementation — no code
copied; see docs/PHASE4_PLAN.md, Part 1 items 16-19):
- official DFL (GPL-3.0): the file format contract — checkpoint files
  (``*.npy``) contain a pickled ``dict[str, np.ndarray]`` (pickle
  protocol 4) with keys ``{sub_name}:0`` (scope prefix stripped);
- EXTERNAL_A (GPL-3.0): the strict two-pass load concept (validate
  EVERYTHING before applying anything);
- EXTERNAL_B (unlicensed): the component/optimizer spec and
  optimizer-state mapping concepts (reimplemented strictly).
Rejected external/official behaviors (never reintroduced): greedy
``np.reshape`` on shape mismatch, silent re-initialization of missing
keys, silently skipped missing keys, shape-greedy key fallback,
insertion-order optimizer-state fallback, silently initialized
optimizer fallbacks, private ``.pth``-only formats.

Strict policies (IMPLEMENTATION_PLAN_v2.md sections 17-20, 42):
- every conversion item is explainable: source/destination name,
  shape, dtype, declared layout rule, status (``TensorMapping``);
- a conversion is all-or-nothing: all required items are validated
  before any tensor is copied;
- missing required state, unexpected extra keys, shape mismatch
  (including same-element-count wrong shapes — no reshape fallback),
  dtype mismatch, ambiguous/duplicate mappings, corrupt files,
  missing optimizer/iteration state, and unrecognized state names are
  hard errors (``CheckpointLoadError`` with a full report);
- reverse export (torch -> official) rejects any state that is not
  officially representable with ``UnsupportedExportError`` carrying
  the explicit reason — it never silently drops, approximates,
  reshapes, or coerces;
- the conversion layer is device-neutral: it reads/writes CPU/NumPy
  arrays and copies into parameters on their own device;
- no TensorFlow import; the official file format is parsed with
  pickle + NumPy only.

Official-layout rules implemented (declared, reported — item 3 of the
plan's rule priority; the per-layer hooks from Phases 3A/3B are
reused, not duplicated):
1. exact shape identity;
2. the owning layer's explicit ``convert_weight_layout`` hook
   (Conv2D / Conv2DTranspose / DepthwiseConv2D axis maps, Phase 3B);
   for a layer that DECLARES a layout conversion the hook is the
   authoritative rule — a value that already has the torch layout is
   an explicit failure, never a silent identity assumption;
3. ``broadcast_squeeze``: a 1-D torch parameter (bias / BN
   ``weight``/``bias``/``running_mean``/``running_var``) accepting the
   2-D/3-D/4-D singleton-padded broadcast forms found in real
   official files (e.g. S3FD/2DFAN store ``bias:0`` / ``bn*:0`` as
   ``(1,1,1,C)`` while 3DFAN stores ``(C,)``; the official TF
   variables themselves are 1-D, so the 4-D files predate the 1-D
   leras variables and were only loadable there through the greedy
   reshape this project refuses);
4. model-specific explicit rules: RESERVED for the later model
   phases (this phase has no model-specific tables — the generic
   components need none).

Phase 4 scope: this commit lands the conversion engine for the
generic components (weights) — the official pickled-dict format
bridge, name mapping, declared layout rules, strict two-pass
conversion, and the reverse export (with explicit rejection of
non-exportable state). Optimizer-state conversion (iters /
ms_ / vs_ / acc_) and the bidirectional file-level round-trip
tests follow in the next Phase 4 commit; the later model phases
(SAEHD/AMP/XSeg/Quick96, Phases 6-8) plug their model-specific
name/layout tables into this engine.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from core import pathex
from core.leras import checkpoint as ckpt
from core.leras.layers.Saveable import Saveable


# --- formats / error taxonomy ------------------------------------------

OFFICIAL_FORMAT = "official_dfl_pickled_dict_v4"
TORCH_FORMAT = "torch_named_parameters"

OFFICIAL_TO_TORCH = "OFFICIAL_TO_TORCH"
TORCH_TO_OFFICIAL = "TORCH_TO_OFFICIAL"

# report error codes (each entry in ConversionReport.errors is
# '<CODE>: <detail>')
ERR_MISSING_REQUIRED_KEY = "MISSING_REQUIRED_KEY"
ERR_UNEXPECTED_EXTRA_KEY = "UNEXPECTED_EXTRA_KEY"
ERR_SHAPE_MISMATCH = "SHAPE_MISMATCH"
ERR_DTYPE_MISMATCH = "DTYPE_MISMATCH"
ERR_AMBIGUOUS_MAPPING = "AMBIGUOUS_MAPPING"
ERR_DUPLICATE_MAPPING = "DUPLICATE_MAPPING"
ERR_CORRUPT_CHECKPOINT = "CORRUPT_CHECKPOINT"
ERR_MISSING_REQUIRED_STATE = "MISSING_REQUIRED_STATE"
ERR_UNEXPECTED_EXTRA_STATE = "UNEXPECTED_EXTRA_STATE"
ERR_UNRECOGNIZED_STATE_KEY = "UNRECOGNIZED_STATE_KEY"
ERR_INVALID_LAYOUT = "INVALID_LAYOUT"

_WEIGHT_STATE_PREFIXES = ("ms_", "vs_", "acc_")

# np dtype -> torch dtype for the strict dtype policy
_NP_TO_TORCH_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}
# declared exact integer widening for the iteration counter: official
# TF stored ``iters`` as int32, the torch implementation keeps it as
# int64 (torch.long) — value-exact, reported, not a silent cast
_ITERS_INT_WIDENING = {np.dtype(np.int32): torch.int64}


class UnsupportedExportError(Exception):
    """Raised when a torch -> official reverse export is requested for
    state that is NOT representable in the official DFL format.

    The converter never silently drops, approximates, reshapes, or
    coerces such state: the export fails as a whole and this error
    names the offending state and the reason. (Modern Features Mode
    states that have no official representation are kept in the
    torch-local format, never exported as official-compatible files.)
    """


# --- report structures ---------------------------------------------------

@dataclass
class TensorMapping:
    """One explainable conversion item (the unit of the report)."""
    source_name: str            # official sub-name, e.g. 'conv1_1/weight:0'
    destination_name: str       # torch name of the target parameter/buffer
    source_shape: tuple         # shape in the source (before the rule)
    destination_shape: tuple    # shape of the target parameter/buffer
    source_dtype: str           # numpy dtype name of the source value
    destination_dtype: str      # torch dtype name of the target
    rule: str                   # 'identity' | 'layer_layout:<Class>' |
                                # 'broadcast_squeeze' | 'iters_int_widening'
    status: str                 # 'MAPPED_IDENTITY' | 'MAPPED_LAYOUT_CONVERTED'


@dataclass
class ConversionReport:
    """Deterministic structured report for one conversion (item 28).

    Component identities are saveable/optimizer names or class names;
    file paths are never embedded in the report. ``payload`` carries
    the torch -> official output dict (None in the other direction).
    """
    direction: str
    source_format: str
    destination_format: str
    source_component: str
    destination_component: str
    mapped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    payload: dict = None

    @property
    def result(self):
        return "PASS" if not self.errors else "FAIL"

    # --- counted facets (derived, so they cannot drift from `errors`) ---

    def _count(self, code):
        return sum(1 for e in self.errors if e.startswith(code + ":"))

    @property
    def mapped_state_count(self):
        return len(self.mapped)

    @property
    def unmapped_source_count(self):
        return self._count(ERR_UNEXPECTED_EXTRA_KEY) \
            + self._count(ERR_UNEXPECTED_EXTRA_STATE)

    @property
    def missing_required_count(self):
        return self._count(ERR_MISSING_REQUIRED_KEY) \
            + self._count(ERR_MISSING_REQUIRED_STATE)

    @property
    def shape_mismatch_count(self):
        return self._count(ERR_SHAPE_MISMATCH) + self._count(ERR_INVALID_LAYOUT)

    @property
    def dtype_mismatch_count(self):
        return self._count(ERR_DTYPE_MISMATCH)

    @property
    def layout_conversion_count(self):
        return sum(1 for m in self.mapped if m.rule != "identity")

    def to_text(self):
        lines = [
            f"direction: {self.direction}",
            f"source_format: {self.source_format}",
            f"destination_format: {self.destination_format}",
            f"source_component: {self.source_component}",
            f"destination_component: {self.destination_component}",
            f"mapped_state_count: {self.mapped_state_count}",
            f"unmapped_source_count: {self.unmapped_source_count}",
            f"missing_required_count: {self.missing_required_count}",
            f"shape_mismatch_count: {self.shape_mismatch_count}",
            f"dtype_mismatch_count: {self.dtype_mismatch_count}",
            f"layout_conversion_count: {self.layout_conversion_count}",
            "mapped:",
        ]
        if self.mapped:
            for m in self.mapped:
                lines.append(
                    f"  - source_name: {m.source_name}\n"
                    f"    destination_name: {m.destination_name}\n"
                    f"    source_shape: {m.source_shape}\n"
                    f"    destination_shape: {m.destination_shape}\n"
                    f"    source_dtype: {m.source_dtype}\n"
                    f"    destination_dtype: {m.destination_dtype}\n"
                    f"    conversion_rule: {m.rule}\n"
                    f"    status: {m.status}"
                )
        else:
            lines.append("  (none)")
        lines.append("warnings:")
        if self.warnings:
            lines.extend(f"  - {w}" for w in self.warnings)
        else:
            lines.append("  (none)")
        lines.append("errors:")
        if self.errors:
            lines.extend(f"  - {e}" for e in self.errors)
        else:
            lines.append("  (none)")
        lines.append(f"result: {self.result}")
        return "\n".join(lines)


# --- official file format (items 2-5, 19) ---------------------------------

def read_official_checkpoint(path):
    """Read and validate an official DFL checkpoint file.

    The file must contain a pickled ``dict[str, np.ndarray]``
    (protocol 4, written by the official DFL ``save_weights`` and by
    this project's torch ``save_weights``). Anything else — a non-dict
    pickle, a real ``np.save``-format array file, a truncated/corrupt
    pickle — is ``CORRUPT_CHECKPOINT``.
    """
    name = Path(path).name
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise ckpt.CheckpointLoadError(
            f"{ERR_CORRUPT_CHECKPOINT}: {name} cannot be read ({e})"
        ) from e
    try:
        d = pickle.loads(raw)
    except Exception as e:
        raise ckpt.CheckpointLoadError(
            f"{ERR_CORRUPT_CHECKPOINT}: {name} is not a valid official "
            f"DFL checkpoint (pickle parse failed: {e}); the official "
            f"format is a pickled dict of numpy arrays"
        ) from e

    if not isinstance(d, dict):
        raise ckpt.CheckpointLoadError(
            f"{ERR_CORRUPT_CHECKPOINT}: {name} is not an official DFL "
            f"checkpoint (top level is {type(d).__name__}, expected dict)"
        )
    for key, value in d.items():
        if not isinstance(key, str):
            raise ckpt.CheckpointLoadError(
                f"{ERR_CORRUPT_CHECKPOINT}: {name} has a non-string key "
                f"{key!r}"
            )
        if not isinstance(value, np.ndarray):
            raise ckpt.CheckpointLoadError(
                f"{ERR_CORRUPT_CHECKPOINT}: {name} value for '{key}' is "
                f"{type(value).__name__}, expected numpy array"
            )
    return d


def write_official_checkpoint(path, d):
    """Write ``d`` (dict[str, np.ndarray]) as an official DFL
    checkpoint: pickle protocol 4, atomic write — byte-compatible with
    files written by official DeepFaceLab and by this project's
    ``Saveable.save_weights``."""
    if not isinstance(d, dict):
        raise TypeError("write_official_checkpoint expects a dict")
    pathex.write_bytes_safe(Path(path), pickle.dumps(d, 4))


# --- layout rule engine (item 20 + item 23 rule 1) ------------------------

def _declares_layout(owner_cls):
    """True when ``owner_cls`` overrides the identity
    ``convert_weight_layout`` hook (declares an official-layout
    difference)."""
    if owner_cls is None or owner_cls is Saveable:
        return False
    hook = getattr(owner_cls, "convert_weight_layout", None)
    if hook is None:
        return False
    return hook is not Saveable.convert_weight_layout


def _squeeze_candidate(value, param):
    """Rule 3 (generic, any owner): a 1-D torch parameter (bias / BN
    ``weight``/``bias``/``running_mean``/``running_var``) accepting the
    2-D/3-D/4-D singleton-padded broadcast forms found in real
    official files (item 23 rule 1). Returns the squeezed array, or
    None."""
    if (
        param.ndim == 1
        and 2 <= value.ndim <= 4
        and all(s == 1 for s in value.shape[:-1])
        and value.shape[-1] == param.shape[0]
    ):
        return value.reshape(tuple(param.shape))
    return None


def _resolve_layout(saveable, owner_cls, value, param):
    """Resolve the layout of a loaded value onto ``param`` using ONLY
    declared rules. Returns ``(converted, rule)`` or raises
    ``_LayoutError`` (converted into report errors by the callers)."""
    value = np.asarray(value)
    value_shape = tuple(value.shape)
    param_shape = tuple(param.shape)

    if _declares_layout(owner_cls):
        # For a layout-declaring layer (Phase 3B conv family) the
        # explicit axis-map hook is the checkpoint contract: it is
        # trusted whenever it produces the parameter shape.
        try:
            converted = saveable.convert_weight_layout(value, param)
        except Exception:
            converted = None  # the hook cannot express this value
        if converted is not None and \
                tuple(np.asarray(converted).shape) == param_shape:
            rule = ("identity" if converted is value
                    else f"layer_layout:{owner_cls.__name__}")
            return (_as_contig(converted), rule)
        # the hook did not express this value: the generic rules still
        # apply (they concern other parameters of the same layer, e.g.
        # the 1-D bias of a conv that declares a kernel axis map)
        squeezed = _squeeze_candidate(value, param)
        if squeezed is not None:
            return squeezed, "broadcast_squeeze"
        if value_shape == param_shape:
            raise _LayoutError(
                f"{ERR_INVALID_LAYOUT}: value already has the torch "
                f"layout {param_shape}, but {owner_cls.__name__} declares "
                f"an official checkpoint layout conversion (the source is "
                f"not an official-layout file for this layer)"
            )
        raise _LayoutError(
            f"{ERR_SHAPE_MISMATCH}: source shape {value_shape} cannot be "
            f"converted to parameter shape {param_shape} for "
            f"{owner_cls.__name__} (no declared layout rule applies; no "
            f"element-count reshape fallback)"
        )

    # rule 1: exact identity (layers without a declared layout)
    if value_shape == param_shape:
        return value, "identity"

    # rule 3: broadcast_squeeze (official 4-D singleton-padded bias/BN
    # forms, item 23 rule 1)
    squeezed = _squeeze_candidate(value, param)
    if squeezed is not None:
        return squeezed, "broadcast_squeeze"

    raise _LayoutError(
        f"{ERR_SHAPE_MISMATCH}: source shape {value_shape} != parameter "
        f"shape {param_shape} (no declared layout rule applies; no "
        f"element-count reshape fallback)"
    )


class _LayoutError(Exception):
    pass


def _as_contig(arr):
    """NumPy 2.x ``np.ascontiguousarray`` upgrades 0-D arrays to
    ``(1,)`` (NumPy 1.x keeps them 0-D); the official iteration
    counter is a 0-D value, so 0-D inputs pass through untouched —
    identical semantics under both pinned NumPy versions."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr
    return np.ascontiguousarray(arr)


def _dtype_mismatch_text(value_dtype, param_dtype):
    """Strict dtype policy: exact match, or the declared iters
    int32->int64 widening. Returns None on match, else the error text."""
    dt = np.dtype(value_dtype)
    if param_dtype == _NP_TO_TORCH_DTYPE.get(dt):
        return None
    if dt in _ITERS_INT_WIDENING and _ITERS_INT_WIDENING[dt] == param_dtype:
        return None
    return (
        f"{ERR_DTYPE_MISMATCH}: source dtype {dt.name} is not "
        f"representable in the target dtype {param_dtype} (no silent "
        f"coercion)"
    )


# --- official -> torch (weights) ------------------------------------------

def convert_official_to_torch(saveable, d, component=None):
    """Strictly convert an official-format dict ``d`` into the torch
    ``saveable`` (two-pass, all-or-nothing) and return the
    ``ConversionReport``.

    Raises ``ckpt.CheckpointLoadError`` (message = report text) when
    the conversion fails; on failure NOTHING is copied.
    """
    if not isinstance(saveable, torch.nn.Module):
        raise TypeError(
            f"{type(saveable).__name__} is not a torch module: torch "
            f"saveables enumerate registered parameters/buffers"
        )
    component = component or getattr(saveable, "name", None) \
        or type(saveable).__name__
    scope = getattr(saveable, "name", None)

    report = ConversionReport(
        direction=OFFICIAL_TO_TORCH,
        source_format=OFFICIAL_FORMAT,
        destination_format=TORCH_FORMAT,
        source_component=component,
        destination_component=f"{type(saveable).__name__} ({component})",
    )

    items = [(sub, t) for sub, t in saveable._iter_official_weights()
             if sub != ""]
    expected_stripped = {ckpt.strip_zero_suffix(sub) for sub, _ in items}
    if scope:
        # declared alias: keys may also carry the saveable scope prefix
        # (files written by the parent tree); both forms are exact
        # name-derived matches, never guesses
        expected_stripped |= {
            ckpt.strip_zero_suffix(f"{scope}/{sub}") for sub, _ in items
        }

    # duplicate sub-names inside the saveable itself
    seen_subs = {}
    for sub, _ in items:
        if sub in seen_subs:
            report.errors.append(
                f"{ERR_DUPLICATE_MAPPING}: saveable enumerates weight "
                f"'{sub}' more than once (ambiguous target)"
            )
        seen_subs[sub] = True

    # owner classes for the layout engine
    owners = {}
    for module in saveable.modules():
        for _pname, p in module.named_parameters(recurse=False):
            owners[id(p)] = type(module)
        for _bname, b in module.named_buffers(recurse=False):
            owners[id(b)] = type(module)

    def _lookup_sub(sub_name):
        value, matched_key = ckpt._lookup_key(d, sub_name)
        if value is None and scope:
            value, matched_key = ckpt._lookup_key(d, f"{scope}/{sub_name}")
        return value, matched_key

    key_consumers = {}
    planned = []
    for sub_name, param in items:
        value, matched_key = _lookup_sub(sub_name)
        if value is None:
            report.errors.append(
                f"{ERR_MISSING_REQUIRED_KEY}: required weight "
                f"'{sub_name}' is missing from the source (official DFL "
                f"re-initialized missing weights silently; this project "
                f"must not)"
            )
            continue
        if not isinstance(value, np.ndarray):
            report.errors.append(
                f"{ERR_CORRUPT_CHECKPOINT}: value for '{sub_name}' is "
                f"{type(value).__name__}, expected numpy array"
            )
            continue
        if matched_key in key_consumers:
            report.errors.append(
                f"{ERR_AMBIGUOUS_MAPPING}: source key '{matched_key}' is "
                f"required by both '{key_consumers[matched_key]}' and "
                f"'{sub_name}'"
            )
            continue
        key_consumers[matched_key] = sub_name

        dt_err = _dtype_mismatch_text(value.dtype, param.dtype)
        if dt_err is not None:
            report.errors.append(f"{dt_err} (weight '{sub_name}')")
            continue
        try:
            converted, rule = _resolve_layout(saveable, owners.get(id(param)),
                                              value, param)
        except _LayoutError as e:
            report.errors.append(f"{e} (weight '{sub_name}')")
            continue

        report.mapped.append(TensorMapping(
            source_name=matched_key,
            destination_name=sub_name,
            source_shape=tuple(value.shape),
            destination_shape=tuple(param.shape),
            source_dtype=np.dtype(value.dtype).name,
            destination_dtype=str(param.dtype),
            rule=rule,
            status=("MAPPED_IDENTITY" if rule == "identity"
                    else "MAPPED_LAYOUT_CONVERTED"),
        ))
        planned.append((_as_contig(converted), param))

    for key in d:
        if ckpt.strip_zero_suffix(key) not in expected_stripped:
            report.errors.append(
                f"{ERR_UNEXPECTED_EXTRA_KEY}: source key '{key}' does not "
                f"match any weight of this saveable (official DFL ignored "
                f"extra keys silently; this project must not)"
            )

    if report.errors:
        report.warnings = _weight_warnings(report)
        raise ckpt.CheckpointLoadError(report.to_text())

    # pass 2: apply (all-or-nothing)
    for value, param in planned:
        with torch.no_grad():
            param.copy_(torch.from_numpy(value).to(device=param.device,
                                                   dtype=param.dtype))
    report.warnings = _weight_warnings(report)
    return report


def _weight_warnings(report):
    warnings = []
    if any(m.rule == "broadcast_squeeze" for m in report.mapped):
        warnings.append(
            "broadcast_squeeze applied: the official file uses "
            "singleton-padded (1[,1[,1]])C bias/BN shapes; the torch "
            "parameters are 1-D (C,) — declared rule, value-exact"
        )
    return warnings


# --- torch -> official (weights) ------------------------------------------

def convert_torch_to_official(saveable, component=None, non_exportable=None):
    """Convert the torch ``saveable`` to an official-format dict and
    return the ``ConversionReport`` (``report.payload`` is the
    official-layout dict, torch-layout parameters converted through
    the per-layer ``convert_weight_to_official`` hooks).

    ``non_exportable`` maps sub-name -> reason for states that have NO
    official representation (Modern Features Mode state): the export
    fails as a whole with ``UnsupportedExportError`` — the converter
    never silently drops such state.
    """
    if not isinstance(saveable, torch.nn.Module):
        raise TypeError(
            f"{type(saveable).__name__} is not a torch module: torch "
            f"saveables enumerate registered parameters/buffers"
        )
    component = component or getattr(saveable, "name", None) \
        or type(saveable).__name__
    non_exportable = dict(non_exportable or {})

    report = ConversionReport(
        direction=TORCH_TO_OFFICIAL,
        source_format=TORCH_FORMAT,
        destination_format=OFFICIAL_FORMAT,
        source_component=component,
        destination_component=f"{type(saveable).__name__} ({component})",
    )

    d = {}
    for sub_name, w in saveable._iter_official_weights():
        if sub_name == "":
            continue
        reason = non_exportable.get(sub_name) \
            or non_exportable.get(ckpt.strip_zero_suffix(sub_name))
        if reason:
            raise UnsupportedExportError(
                f"UNSUPPORTED_REVERSE_EXPORT: state '{sub_name}' of "
                f"{component} is not representable in the official DFL "
                f"format ({reason}); the official-compatible export is "
                f"refused as a whole — the modern-only state must be kept "
                f"in the torch-local format, never dropped or approximated"
            )
        arr = w.detach().cpu()
        rule = "identity"
        if arr.dtype == torch.bfloat16:
            # declared storage convention (identical to save_weights):
            # numpy has no bfloat16, stored as float32 (exact superset)
            arr = arr.to(torch.float32)
            rule = "bfloat16_stored_as_float32"
            report.warnings.append(
                f"state '{sub_name}' is bfloat16 (no official/numpy "
                f"representation); stored as float32 — exact value, "
                f"declared convention"
            )
        arr = arr.numpy().copy()
        converted = saveable.convert_weight_to_official(arr, w)
        if tuple(np.asarray(converted).shape) != tuple(arr.shape) \
                and rule == "identity":
            rule = f"layer_to_official:{type(saveable).__name__}"
        arr = _as_contig(converted)
        d[sub_name] = arr
        report.mapped.append(TensorMapping(
            source_name=sub_name,
            destination_name=sub_name,
            source_shape=tuple(w.shape),
            destination_shape=tuple(arr.shape),
            source_dtype=str(w.dtype),
            destination_dtype=np.dtype(arr.dtype).name,
            rule=rule,
            status=("MAPPED_IDENTITY" if rule == "identity"
                    else "MAPPED_LAYOUT_CONVERTED"),
        ))
    report.payload = d
    return report
