"""Checkpoint naming/mapping contract (Phase 3A foundation).

Official DeepFaceLab checkpoints are files (``*.npy``) containing a
pickled ``dict[str, np.ndarray]`` (pickle protocol 4) whose keys are
TensorFlow variable names:

    {scope}/{layer_name}/{param_name}:0

e.g. ``encoder/conv1_1/weight:0``, ``encoder/conv1_1/bias:0``,
``inter_AB/batchnorm_1/running_mean:0``, ``iters:0`` (model level).

The torch leras reproduces the official keys exactly: official DFL
parameter names (``weight`` / ``bias`` / ``running_mean`` /
``running_var``) are identical to the torch parameter names used by the
reconstructed layers, so the mapping is a pure string transform of the
torch (dotted) name:

    torch:  scope.layer_name.param_name
    official: scope/layer_name/param_name:0

This module owns that contract plus the strict-load error type. Phase 4
extends it with per-layout conversion (official conv kernels are
``(H,W,in,out)`` / dense ``(in,out)`` vs torch ``(out,in,H,W)`` /
``(out,in)``) and the TF→torch converter; Phase 3B layers register their
parameters with the official names so that no name translation table is
ever needed.

The official pickled-dict ``.npy`` format is the authoritative
checkpoint representation of this project; no ``.pth``-only path is
introduced (bidirectional Compatibility Mode stays possible:
Original DFL → Modernized DFL → train/save → official-compatible export
→ Original/reference DFL).
"""

import numpy as np
import torch


class CheckpointLoadError(Exception):
    """Raised when a checkpoint cannot be loaded STRICTLY.

    Official DFL silently re-initialized missing weights and greedily
    reshaped mismatches; this project must not reintroduce either
    (IMPLEMENTATION_PLAN_v2.md sections 19 and 42). A strict load is
    all-or-nothing: any missing key, unexpected extra key, or shape
    mismatch raises this exception with a detailed report, and nothing
    is copied into the module.
    """


def strip_zero_suffix(name):
    """'weight:0' -> 'weight'; other names unchanged."""
    if name.endswith(":0"):
        return name[:-2]
    return name


def official_name(torch_dotted_name):
    """torch dotted name -> official DFL variable name.

    'encoder.conv1_1.weight' -> 'encoder/conv1_1/weight:0'
    (a trailing ':0' already present is preserved, not doubled).
    """
    base = torch_dotted_name.replace(".", "/")
    if base.endswith(":0"):
        return base
    return base + ":0"


def torch_name_from_official(official_name):
    """official DFL variable name -> torch dotted name.

    'encoder/conv1_1/weight:0' -> 'encoder.conv1_1.weight'
    """
    base = strip_zero_suffix(official_name).replace("/", ".")
    return base


def _lookup_key(d, key):
    """Find ``key`` (with or without ':0') in dict ``d``.

    Returns (value, matched_key) or (None, None). Used by the strict
    loader so that files written with/without the trailing ':0' both
    work, exactly like the official Saveable's split('/') handling.
    """
    if key in d:
        return d[key], key
    alt = key + ":0" if not key.endswith(":0") else key[:-2]
    if alt in d:
        return d[alt], alt
    return None, None


def matching_alias_keys(d, key):
    """All present spellings of one declared `:0` checkpoint name."""
    alt = key[:-2] if key.endswith(':0') else key + ':0'
    return [candidate for candidate in (key, alt) if candidate in d]


_NP_TO_TORCH_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}


def dtype_mismatch_text(value_dtype, param_dtype, allow_int_widening=False):
    """Exact dtype, except:
    - the declared int32 -> int64 optimizer ``iters``;
    - float -> float pairs (the official ``batch_set_value`` semantics,
      official ``core/leras/ops`` L23: ``np.asarray(value,
      dtype=<target variable dtype>)`` — the target parameter dtype is
      authoritative and the file value is cast to it; in particular an
      fp32 training checkpoint loads into the export-only fp16 archi
      with the official narrowing cast, the DFL 'Export quantized?'
      flow). Cross-kind pairs (e.g. a float file into an integer
      parameter) remain a hard error."""
    dt = np.dtype(value_dtype)
    if _NP_TO_TORCH_DTYPE.get(dt) == param_dtype:
        return None
    if allow_int_widening and dt == np.dtype(np.int32) and param_dtype == torch.int64:
        return None
    if dt.kind == 'f' and param_dtype in (torch.float16, torch.bfloat16,
                                          torch.float32, torch.float64):
        return None
    return (
        f"DTYPE_MISMATCH: source dtype {dt.name} is not representable "
        f"in the target dtype {param_dtype} (cross-kind coercion is "
        f"rejected; the official semantics cast within the float kind)"
    )
