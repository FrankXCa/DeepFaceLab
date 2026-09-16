"""Torch leras ops (Phase 3C start).

Phase 3C migrates exactly one operation from the official TensorFlow
ops module: ``depth_to_space``. The remaining official ops (dssim,
style_loss, gaussian blur, pixel_norm, rgb_to_lab, batch_set_value,
...) are preserved verbatim in ``core/leras/ops/ops_tf.py`` (dead TF
code, never imported by torch paths) and are rebuilt as torch ops in
later Phase 3 subphases. Importing this package never touches
TensorFlow.

depth_to_space - official DeepFaceLab / TensorFlow semantics (R-R-C)
============================================================

The official DFL op (``ops/__init__.py``) uses TensorFlow
``depth_to_space`` semantics in BOTH of its NCHW branches (native
``tf.depth_to_space`` on GPU and DFL's manual CPU fallback) and in its
NHWC branch. The index mapping (derived from the official source,
verified by the exact-placement tests in
``tests/smoke/test_depth_to_space.py``) is:

    official (TF, R-R-C):
        out[n, c, h*r+i, w*r+j] = in[n, (i*r+j)*C_out + c, h, w]

PyTorch's ``torch.nn.functional.pixel_shuffle`` uses a different
channel grouping (C-R-R):

    pixel_shuffle:
        out[n, c, h*r+i, w*r+j] = in[n, c*r*r + i*r+j, h, w]

A bare ``F.pixel_shuffle`` is therefore NOT official-compatible
(External B and the legacy torch bridge both made this mistake:
shape-correct but spatially scrambled output). This implementation
applies the required channel permutation first:

    inP[:, c*r*r + i*r+j] = in[:, (i*r+j)*C_out + c]

built as a pure view/permute (no index tensor, no gather copy):

    in(N, C_in, H, W)
      -> reshape(N, r, r, C_out, H, W)        # t = (i*r+j)*C_out + c
      -> permute(0, 3, 1, 2, 4, 5)            # (N, C_out, r, r, H, W)
      -> reshape(N, C_in, H, W) = inP         # p = c*r*r + i*r + j
      -> F.pixel_shuffle(inP, r)              # official R-R-C output

The permute must leave the spatial axes contiguous as (H, W): the
official manual branch's transpose (0,3,4,1,5,2) targets its own
direct final layout (b, oc, h, i, w, j) and must NOT be copied into
this route, where the reshape to (N, C_in, H, W) would otherwise
silently merge H into the channel axis.

Checkpoint significance: official SAEHD/AMP decoder upsampler
weights (``Upscale.conv1`` -> ``depth_to_space(x, 2)``) were trained
assuming the TF R-R-C semantics, so this op must match TF exactly;
with it, the Phase 3B conv weight layout hooks need no extra channel
permutation (no hidden weight-layout compensation anywhere).

Official deviations (strict policy, plan v2 sections 19/42):
- a channel count not divisible by r*r raises ValueError (the
  official CPU fallback silently truncated via integer division);
- NHWC tensors are handled by boundary permutation through
  ``nn.to_data_format`` (official DFL call sites use NCHW).
"""

import torch
import torch.nn.functional as F

from core.leras import nn


def depth_to_space(x, size):
    """Official DFL / TensorFlow-compatible depth_to_space (R-R-C).

    x: (N, C_in, H, W) NCHW tensor (C_in divisible by size*size), or
    (N, H, W, C_in) when ``nn.data_format`` is "NHWC".
    Returns the same data_format as the input.
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"depth_to_space size must be a positive int, got {size!r}")

    nhwc = nn.data_format == "NHWC"
    if nhwc:
        x = nn.to_data_format(x, "NCHW", "NHWC")

    n, c_in, h, w = x.shape
    if c_in % (size * size) != 0:
        raise ValueError(
            f"depth_to_space: input channels {c_in} must be divisible by "
            f"size*size = {size * size} (the official CPU fallback truncated "
            f"silently; this project fails explicitly)"
        )
    c_out = c_in // (size * size)

    # TF R-R-C -> PyTorch C-R-R channel permutation (pure view/permute):
    # in[:, (i*size+j)*c_out + c] moves to inP[:, c*size*size + i*size + j]
    x = x.reshape(n, size, size, c_out, h, w)
    x = x.permute(0, 3, 1, 2, 4, 5)  # (N, C_out, i, j, H, W): (H, W) stays contiguous
    x = x.reshape(n, c_in, h, w)

    x = F.pixel_shuffle(x, size)

    if nhwc:
        x = nn.to_data_format(x, "NHWC", "NCHW")
    return x


nn.depth_to_space = depth_to_space
