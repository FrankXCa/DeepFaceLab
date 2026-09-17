"""Checkpoint conversion engine (Phase 4).

Centralized, explicit, testable compatibility bridge between the
official DeepFaceLab TF-era checkpoint format and the modernized torch
implementation — the converter foundation that later model phases
(SAEHD/AMP/XSeg/Quick96, Phases 6-8) plug into.

Authoritative source concepts (independent reimplementation — no code
copied; see docs/PHASE4_PLAN.md, Part 1 items 16-19):
- official DFL (GPL-3.0): the file format contract — checkpoint files
  carry a ``dict[str, np.ndarray]`` whose keys are ``{sub_name}:0``
  (scope prefix stripped). EXACT outer container (verified against
  the official source AND the real artifacts in this repo): the
  official ``Saveable.save_weights`` writes ``pickle.dumps(d, 4)``
  through ``pathex.write_bytes_safe`` — a RAW pickle protocol-4
  stream; the ``.npy`` extension is a misnomer, the file is NOT a
  NumPy ``.npy`` container and ``np.save``/``np.load`` are not
  involved at any level; "protocol 4" is the pickle protocol of the
  whole file (there is no separate outer container). The official
  ``load_weights`` reads it back with ``pickle.loads(file_bytes)``.
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
3. ``channel_broadcast``: an explicit WHITELIST of the known
    official singleton-padded layouts for a 1-D torch parameter of
    size C (conv bias; BN ``weight``/``bias``/``running_mean``/
    ``running_var``; FRNorm ``eps``) — never an unrestricted squeeze:

    - ``(C,)``      identity — the official TF variable shape (all
      1-D variables in the official leras are ``(dim,)``; the 3DFAN
      artifact stores this form);
    - ``(1,1,1,C)`` NHWC 4-D singleton padding — the official
      forward reshape shape ``(1,1,1,dim)`` for ``data_format ==
      'NHWC'`` (official BatchNorm2D/InstanceNorm2D/FRNorm2D); the
      form present in the official S3FD / 2DFAN / FaceEnhancer
      artifacts (conv biases and BN states);
    - ``(1,C,1,1)`` NCHW 4-D singleton padding — the official
      forward reshape shape ``(1,dim,1,1)`` for ``data_format ==
      'NCHW'`` (same official source).

    NO other shape is accepted, even when the element count matches
    and a naive ``np.squeeze()`` would produce the right 1-D length
    (e.g. ``(1,C)``, ``(1,1,C)``, ``(1,C,1)``, ``(C,1)``,
    ``(1,1,C,1)``, ``(C,1,1,1)``, ``(2,1,1,C)`` all fail
    explicitly). The official artifacts in this repo contain ONLY the
    ``(C,)`` and ``(1,1,1,C)`` forms for these tensors (censused over
    all four official facelib files); the official TF variables
    themselves are 1-D, so the 4-D artifacts were only loadable
    officially through the greedy ``np.reshape`` this project
    refuses;
4. model-specific explicit rules: RESERVED for the later model
   phases (this phase has no model-specific tables — the generic
   components need none).

Phase 4 coverage (parity labels, see docs/COMPATIBILITY.md):
- EXACT (weight-level, both directions, file-level round-trip): the
  Phase 3B layers (Conv2D, Conv2DTranspose, DepthwiseConv2D with
  layout rules; Dense / DenseNorm / BatchNorm2D / InstanceNorm2D /
  FRNorm2D with identity + the channel-broadcast whitelist for the
  1-D bias/BN forms found in real official files), the Phase 3F archis
  (DeepFakeArchi Encoder/Inter/Decoder, all official option combos)
  and discriminators (CodeDiscriminator, PatchDiscriminator,
  UNetPatchDiscriminator) as named Saveables, and the Phase 3E2
  optimizers (AdaBelief, RMSprop — iters + ms_/vs_/acc_ states,
  value-exact including the official int32 -> torch int64 iteration
  counter widening);
- EXACT format contract on real official artifacts (facelib/*.npy
  pickled-dict files tracked in the baseline): parsing, key
  conventions, layouts, dtypes (incl. float16 FaceEnhancer);
- NOT_YET_IMPLEMENTED (full model checkpoint compatibility): SAEHD
  (liae/df), AMP, Quick96 and XSeg model classes migrate in Phases
  6-8 on top of this engine (model-specific name/layout tables land
  with those phases); the facelib extractor models (S3FD/2DFAN/3DFAN/
  FaceEnhancer, incl. S3FD's L2Norm 4-D weight rule) migrate in
  Phase 9 — their real checkpoints validate the FORMAT here, not the
  models;
- TF runtime parity for the file format is not required: the format
  is verified against the official artifacts directly (no TF import
  in this module).
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
                                # 'channel_broadcast' | 'iters_int_widening'
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
    this project's torch ``save_weights``). Official builds from the
    modernization era onward additionally store the optimizer
    iteration counter (``iters:0``) as a bare ``int`` instead of a
    0-D array — both spellings are valid official format (a bare
    ``int``/``np.integer`` value is accepted anywhere a state value
    appears and normalized to a 0-D array). Anything else — a
    non-dict pickle, a real ``np.save``-format array file, a
    truncated/corrupt pickle, a list/str/float state value — is
    ``CORRUPT_CHECKPOINT``.
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
        # official newer builds store the optimizer iters counter as
        # a bare int (older TF-era builds: a 0-D array) — both valid
        if not isinstance(value, np.ndarray) and \
                not isinstance(value, (int, np.integer)):
            raise ckpt.CheckpointLoadError(
                f"{ERR_CORRUPT_CHECKPOINT}: {name} value for '{key}' is "
                f"{type(value).__name__}, expected numpy array (or a "
                f"bare int for the official optimizer iters counter)"
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


def _channel_broadcast_candidate(value, param):
    """Rule ``channel_broadcast``: the explicit WHITELIST of known
    official singleton-padded layouts for a 1-D torch parameter of
    size C (conv bias; BN ``weight``/``bias``/``running_mean``/
    ``running_var``; FRNorm ``eps``). Returns the transformed
    ``(C,)`` array, or None when the shape is not a known official
    layout.

    Accepted (C = param.size; the identity form ``(C,)`` is handled
    by rule 1 before this rule is consulted):

    - ``(1,1,1,C)``  NHWC 4-D singleton padding (the official
      forward-reshape placement for ``data_format == 'NHWC'``; the
      form stored by the official S3FD / 2DFAN / FaceEnhancer
      artifacts): the channel axis is last and contiguous in C
      order -> ``value.reshape(C)``;
    - ``(1,C,1,1)``  NCHW 4-D singleton padding (the official
      forward-reshape placement for ``data_format == 'NCHW'``): the
      channel axis is second -> move it last deterministically,
      ``value.transpose(1,0,2,3).reshape(C)``.

    This is a whitelist of known official layouts, NOT an
    unrestricted ``np.squeeze``: no other shape is accepted, even
    when the element count matches and a naive squeeze would give
    the right 1-D length (``(1,C)``, ``(1,1,C)``, ``(1,C,1)``,
    ``(C,1)``, ``(1,1,C,1)``, ``(C,1,1,1)``, ``(2,1,1,C)`` all
    fail explicitly).
    """
    if param.ndim != 1:
        return None
    c = param.shape[0]
    shape = value.shape
    if len(shape) == 4 and shape == (1, 1, 1, c):
        # NHWC placement: trailing axis is contiguous in C order
        return value.reshape(c)
    if len(shape) == 4 and shape == (1, c, 1, 1):
        # NCHW placement: move the channel axis to the end first
        return value.transpose(1, 0, 2, 3).reshape(c)
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
        squeezed = _channel_broadcast_candidate(value, param)
        if squeezed is not None:
            return squeezed, "channel_broadcast"
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

    # rule 3: channel_broadcast (official singleton-padded bias/BN
    # forms, item 23 rule 1)
    squeezed = _channel_broadcast_candidate(value, param)
    if squeezed is not None:
        return squeezed, "channel_broadcast"

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
    if any(m.rule == "channel_broadcast" for m in report.mapped):
        warnings.append(
            "channel_broadcast applied: the official file stores the "
            "1-D bias/BN state singleton-padded ((1,1,1,C) NHWC or "
            "(1,C,1,1) NCHW); the torch parameters are 1-D (C,) — "
            "declared whitelist rule, value-exact"
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


# --- optimizer state (items 14-15) -----------------------------------------

def _state_key_of(param, index):
    """Mirror of OptimizerBase._weight_key (the stable state key)."""
    key = getattr(param, "name", None) or getattr(param, "_dfl_name", None)
    if key is None:
        key = f"param_{index}"
    return key


def _state_param_for(state_sub_name, params_by_name):
    """Official state sub-name -> the tracked parameter it belongs to
    (via the name binding), or None when the slot tracks no bound
    variable (the ``iters`` counter, a well-formed-but-unbound name,
    or a positional ``param_<i>`` key that is itself bound)."""
    for _prefix, varname in _state_varname_candidates(
            ckpt.strip_zero_suffix(state_sub_name)):
        hit = params_by_name.get(varname)
        if hit is not None:
            return hit[0]
    return None


def _state_layout_layer(tracked):
    """The layer whose layout rule applies to a state tensor that
    tracks ``tracked`` (module docstring of OptimizerBase, 'State
    LAYOUT'): the owning layer binding the model code attaches to
    every optimized parameter. None when unbound — the value then
    passes through unchanged (official identity case)."""
    layer = getattr(tracked, "_dfl_owner_layer", None) \
        if tracked is not None else None
    if layer is None or not isinstance(layer, Saveable):
        return None
    return layer


def _state_varname_candidates(state_sub_name):
    """Official state sub-name -> list of (prefix, variable official
    sub-name) candidates, in preference order.

    Official: ``f"{prefix}_{varname}".replace(":", "_") + ":0"`` where
    ``varname`` ends with ``':0'`` — the ONLY ':' turned into '_' is
    the trailing one of the LAST path segment. The last segment of a
    state name that ends in ``'_0'`` is therefore ambiguous: it is
    EITHER the restored ``<param>:0`` (marker interpretation) OR the
    literal name (a torch positional key like ``param_0``). Both
    candidates are returned; the caller binds them against the
    optimizer's actual parameters, so the ambiguity is resolved by
    identity, never guessed.
    """
    base = ckpt.strip_zero_suffix(state_sub_name)
    for prefix in _WEIGHT_STATE_PREFIXES:
        if not base.startswith(prefix):
            continue
        rest = base[len(prefix):]
        head, sep, last = rest.rpartition("/")
        cands = []
        if last.endswith("_0"):
            cands.append((prefix,
                          f"{head}/{last[:-2]}:0" if sep else f"{last[:-2]}:0"))
        cands.append((prefix, rest))  # literal (no ':' marker)
        return cands
    return []


def convert_optimizer_state_official_to_torch(optimizer, d, saveable=None,
                                              component=None):
    """Strictly convert official optimizer-state keys (``iters:0``,
    ``ms_*``/``vs_*``/``acc_*``) from dict ``d`` into the torch
    ``optimizer`` (two-pass, all-or-nothing) and return the report.

    Identity is NAME-driven: each state key embeds the trainable
    variable's official name (``ms_<varname>_0:0``), which resolves to
    the parameter (via the parameter's official name binding,
    ``param.name``/``param._dfl_name``) and then to the optimizer's
    state tensor for that parameter. The ``initialize_variables``
    order is used only to name POSITIONAL (unbound) parameters — never
    as the sole identity. ``saveable`` optionally cross-validates that
    each referenced variable belongs to the given model saveable.
    """
    component = component or getattr(optimizer, "name", None) \
        or type(optimizer).__name__

    report = ConversionReport(
        direction=OFFICIAL_TO_TORCH,
        source_format=OFFICIAL_FORMAT,
        destination_format=TORCH_FORMAT,
        source_component=component,
        destination_component=f"{type(optimizer).__name__} ({component})",
    )

    # torch-side official state names (iters + subclass state order)
    names, tensors = zip(*optimizer._iter_official_weights())
    state_map = dict(zip(names, tensors))  # official name -> tensor

    # variable official name -> (param, index) for THIS optimizer
    weights = list(getattr(optimizer, "_weights", []))
    params_by_name = {}
    for i, p in enumerate(weights):
        params_by_name[_state_key_of(p, i)] = (p, i)

    if saveable is not None:
        saveable_names = {
            ckpt.strip_zero_suffix(sub)
            for sub, _ in saveable._iter_official_weights()
            if sub != ""
        }
    else:
        saveable_names = None

    def _bound_varname(name, context):
        """Bind an official state name to this optimizer's variables.

        Returns (varname, error_text_or_None)."""
        for prefix, varname in _state_varname_candidates(name):
            if varname in params_by_name:
                if saveable_names is not None:
                    # the variable name is relative to the saveable's
                    # PARENT scope; the saveable enumeration is
                    # relative to the saveable scope itself
                    sub = varname
                    sname = getattr(saveable, "name", None)
                    if sname and sub.startswith(sname + "/"):
                        sub = sub[len(sname) + 1:]
                    if ckpt.strip_zero_suffix(sub) not in saveable_names:
                        return None, (
                            f"{ERR_UNRECOGNIZED_STATE_KEY}: state "
                            f"'{name}' references variable '{varname}' "
                            f"which is not part of the given saveable"
                        )
                return varname, None
        # no candidate bound: is the name at least well-formed?
        well_formed = bool(_state_varname_candidates(name))
        if not well_formed:
            return None, (
                f"{ERR_UNRECOGNIZED_STATE_KEY}: '{name}' does not follow "
                f"the official optimizer state naming (iters / ms_ / vs_ / "
                f"acc_ <varname> with the official ':' -> '_' marker)"
            )
        # well-formed but references a variable this optimizer has no
        # state for (unbound parameter name)
        cands = [v for _, v in _state_varname_candidates(name)]
        return None, (
            f"{ERR_MISSING_REQUIRED_STATE}: source state '{name}' "
            f"references variable '{cands[0]}' which is not bound to any "
            f"parameter this optimizer was initialized with (parameters "
            f"need their official name binding, param.name / param._dfl_name)"
        )

    # --- iters ---
    iters_param = state_map.get("iters:0")
    iters_value, iters_key = ckpt._lookup_key(d, "iters:0")
    if iters_param is not None and iters_value is None:
        report.errors.append(
            f"{ERR_MISSING_REQUIRED_STATE}: 'iters:0' (iteration state) "
            f"is missing from the source (official DFL silently started "
            f"from iteration 0; this project must not)"
        )
    elif iters_value is not None:
        # official newer builds store iters as a bare int — normalize
        # to a 0-D array before the standard array validation
        if not isinstance(iters_value, np.ndarray) and \
                isinstance(iters_value, (int, np.integer)):
            iters_value = np.asarray(iters_value)
        if not isinstance(iters_value, np.ndarray):
            report.errors.append(
                f"{ERR_CORRUPT_CHECKPOINT}: value for 'iters:0' is "
                f"{type(iters_value).__name__}, expected numpy array "
                f"(or a bare int, the official newer-build format)"
            )
        else:
            dt_err = _dtype_mismatch_text(iters_value.dtype, iters_param.dtype)
            if dt_err is not None:
                report.errors.append(f"{dt_err} (iters:0)")
            elif iters_value.size != 1:
                report.errors.append(
                    f"{ERR_SHAPE_MISMATCH}: 'iters:0' has shape "
                    f"{iters_value.shape}, expected scalar"
                )
            else:
                widened = (np.dtype(iters_value.dtype) in _ITERS_INT_WIDENING
                           and _ITERS_INT_WIDENING[np.dtype(iters_value.dtype)]
                           == iters_param.dtype)
                report.mapped.append(TensorMapping(
                    source_name=iters_key,
                    destination_name="iters:0",
                    source_shape=() if iters_value.ndim == 0
                    else tuple(iters_value.shape),
                    destination_shape=tuple(iters_param.shape),
                    source_dtype=np.dtype(iters_value.dtype).name,
                    destination_dtype=str(iters_param.dtype),
                    rule="iters_int_widening" if widened else "identity",
                    status=("MAPPED_LAYOUT_CONVERTED" if widened
                            else "MAPPED_IDENTITY"),
                ))

    # --- per-parameter states ---
    for name, tensor in state_map.items():
        if name == "iters:0":
            continue
        value, matched_key = ckpt._lookup_key(d, name)
        if value is None:
            report.errors.append(
                f"{ERR_MISSING_REQUIRED_STATE}: optimizer state '{name}' "
                f"is missing from the source (official DFL silently "
                f"reset missing optimizer state; this project must not)"
            )
            continue
        if not isinstance(value, np.ndarray):
            report.errors.append(
                f"{ERR_CORRUPT_CHECKPOINT}: value for '{name}' is "
                f"{type(value).__name__}, expected numpy array"
            )
            continue

        dt_err = _dtype_mismatch_text(value.dtype, tensor.dtype)
        if dt_err is not None:
            report.errors.append(f"{dt_err} (state '{name}')")
            continue

        # name-driven cross-check: the source key and the torch state
        # slot must bind to the SAME variable (also resolves the
        # tracked variable for the state LAYOUT rule below)
        src_varname, err = _bound_varname(matched_key, "source")
        if err:
            report.errors.append(err)
            continue
        dst_varname, err = _bound_varname(name, "destination")
        if err:
            report.errors.append(err)
            continue
        if src_varname != dst_varname:
            report.errors.append(
                f"{ERR_UNRECOGNIZED_STATE_KEY}: source state "
                f"'{matched_key}' binds to variable '{src_varname}' but "
                f"this optimizer state slot '{name}' belongs to "
                f"'{dst_varname}' (cross-wired optimizer state; refusing)"
            )
            continue

        # State LAYOUT (OptimizerBase module docstring): the official
        # state tensor has the layout of the trained variable it
        # tracks — apply the owning layer's official->torch layout
        # hook BEFORE the shape check (the torch state slot has the
        # parameter's torch layout; without this the strict shape
        # check fails on every official-layout conv kernel state).
        tracked = params_by_name[src_varname][0]
        layer = _state_layout_layer(tracked)
        if layer is not None:
            converted = layer.convert_weight_layout(value, tracked)
            # identity hooks return the input unchanged (same object);
            # a real layout conversion produces a new array
            layout = converted is not value
            value_cmp = converted if layout else value
        else:
            layout = False
            value_cmp = value
        if tuple(np.asarray(value_cmp).shape) != tuple(tensor.shape):
            report.errors.append(
                f"{ERR_SHAPE_MISMATCH}: optimizer state '{name}' has "
                f"shape {tuple(np.asarray(value_cmp).shape)}, expected "
                f"{tuple(tensor.shape)} (the value is converted through "
                f"the tracked variable's layout rule; no reshape)"
            )
            continue
        report.mapped.append(TensorMapping(
            source_name=matched_key,
            destination_name=name,
            source_shape=tuple(np.asarray(value).shape),
            destination_shape=tuple(tensor.shape),
            source_dtype=np.dtype(value.dtype).name,
            destination_dtype=str(tensor.dtype),
            rule="layout_converted" if layout else "identity",
            status="MAPPED_LAYOUT_CONVERTED" if layout
            else "MAPPED_IDENTITY",
        ))

    expected_state_names = {ckpt.strip_zero_suffix(n) for n in names}
    for key in d:
        if ckpt.strip_zero_suffix(key) not in expected_state_names:
            report.errors.append(
                f"{ERR_UNEXPECTED_EXTRA_STATE}: source key '{key}' is not "
                f"a state of this optimizer (official DFL ignored extra "
                f"keys silently; this project must not)"
            )

    # duplicate source keys across the ':0' variant forms: one logical
    # state must not be provided by two different source keys (never
    # silently pick one variant over the other — same rule as the
    # weights path)
    state_owner = {}
    for key in d:
        owner = None
        for n in names:
            if key in (n, ckpt.strip_zero_suffix(n)):
                owner = n
                break
        if owner is None:
            continue
        if owner in state_owner:
            report.errors.append(
                f"{ERR_DUPLICATE_MAPPING}: source keys "
                f"'{state_owner[owner]}' and '{key}' both provide state "
                f"'{owner}' (duplicate ':0' variant forms; refusing to "
                f"pick one)"
            )
        else:
            state_owner[owner] = key

    if report.errors:
        raise ckpt.CheckpointLoadError(report.to_text())

    # pass 2: apply (all-or-nothing)
    for key, value in d.items():
        _, matched_key = ckpt._lookup_key(state_map, key)
        target = state_map[matched_key]
        arr = _as_contig(value)
        if target.dtype == torch.int64 and arr.dtype == np.int32:
            arr = arr.astype(np.int64)  # declared iters widening
        # State LAYOUT: official-layout state value -> torch layout
        # through the tracked variable's owning layer (iters and
        # unbound/plain slots: identity — the shape check in pass 1
        # already validated the converted value).
        if target is not state_map.get("iters:0"):
            tracked = _state_param_for(matched_key, params_by_name)
            layer = _state_layout_layer(tracked)
            if layer is not None:
                arr = np.asarray(layer.convert_weight_layout(arr, tracked))
        with torch.no_grad():
            target.copy_(torch.from_numpy(arr).to(device=target.device,
                                                  dtype=target.dtype))
    return report


def convert_optimizer_state_torch_to_official(optimizer, component=None):
    """Convert the torch ``optimizer`` state to an official-format dict
    (``iters:0`` + the official ``ms_*``/``vs_*``/``acc_*`` sub-names —
    exact on the torch side via ``_iter_official_weights``). The
    iteration counter is stored as int64 (torch.long); the official TF
    implementation used int32 — the value is exact and the official
    loader casts to its variable dtype on load.

    State LAYOUT (OptimizerBase module docstring): the on-disk dict
    is official-layout, so each state tensor is converted through the
    layout hook of the OWNING LAYER of the tracked parameter (the
    in-memory state buffers have the parameter's torch layout) —
    identity for ``iters`` and for parameters without a
    ``_dfl_owner_layer`` binding."""
    d = {}
    weights = list(getattr(optimizer, "_weights", []))
    params_by_name = {
        _state_key_of(p, i): (p, i) for i, p in enumerate(weights)
    }
    for name, tensor in optimizer._iter_official_weights():
        value = tensor.detach().cpu().numpy().copy()
        if name != "iters:0":
            tracked = _state_param_for(name, params_by_name)
            layer = _state_layout_layer(tracked)
            if layer is not None:
                value = layer.convert_weight_to_official(value, tracked)
        d[name] = value
    return d
