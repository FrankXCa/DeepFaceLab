"""Torch leras ops (Phase 3C/3D progressive migration).

Migrated so far (each op mirrors the official DeepFaceLab
TensorFlow semantics 1:1; the complete official TF module is
preserved verbatim in ``core/leras/ops/ops_tf.py`` as dead
reference code, never imported by torch paths):

- Phase 3C: ``depth_to_space`` (official TF R-R-C semantics via
  channel permutation + ``F.pixel_shuffle``; a bare ``pixel_shuffle``
  is C-R-R and would silently scramble converted official
  checkpoints).
- Phase 3D: ``dssim``, ``gaussian_blur``, ``style_loss``,
  ``pixel_norm`` (the core numerical ops required by the official
  SAEHD/AMP/Quick96/XSeg loss stacks and the SAEHD archi).
- Phase 3E1: ``flatten``, ``reshape_4D`` (tensor-layout ops used by
  the SAEHD/AMP/XSeg archi dense<->map transitions),
  ``average_tensor_list``, ``total_variation_mse`` (reduction/loss
  helpers used by the official model code).
- Phase 3E2: ``random_binomial`` (the official lr_dropout mask
  source for the migrated AdaBelief/RMSprop optimizers).

Still TensorFlow (later Phase 3 subphases / model phases; see
``ops/ops_tf.py``): rgb_to_lab (dead in the official baseline - no
callers; documented deferral), gelu (no baseline callers),
upsample2d (FANExtractor - facelib TF code), resize2d_* (FaceEnhancer
- facelib TF code), max_pool (no nn-namespace callers in the
baseline), space_to_depth (no baseline callers),
tf_gradients/nn.gradients + average_gv_list (-> the model-phase
gradient flow / dedicated multi-GPU phase), batch_set_value/
tf_get_value (session machinery - disappears under the torch
foundation), bilinear_sampler (TanhPolar phase).

Importing this package never touches TensorFlow.

depth_to_space (Phase 3C)
=========================

The official DFL op (``ops/__init__.py``) uses TensorFlow
``depth_to_space`` semantics in BOTH of its NCHW branches (native
``tf.depth_to_space`` on GPU and DFL's manual CPU fallback) and in its
NHWC branch. The Phase 3F P0 re-audit verified (TF v2.4.0 kernel
source of the official DFL era - the GPU ``D2S_NCHW`` kernel documents
its input ordering as ``n, bY, bX, oC, iY, iX`` - plus a TF 2.21
runtime probe) that the native built-in uses this SAME R-R-C
grouping, i.e. ALL official branches are identical and official
checkpoints of any training path need no dts re-mapping. The index
mapping (derived from the official source, verified by the
exact-placement tests in ``tests/smoke/test_depth_to_space.py``) is:

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

dssim / gaussian_blur / style_loss / pixel_norm (Phase 3D)
==========================================================

All four reproduce the official formulas exactly (verified by
``tests/smoke/test_ops_core.py`` against independent NumPy
implementations of the official formulas):

- ``dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5,
  k1=0.01, k2=0.03)``: the official kernel (arange centered at
  (filter_size-1)/2, squared x (-0.5/sigma^2), 2D outer product,
  softmax-normalized, tiled per channel), VALID depthwise
  convolution (External A/B use SAME padding - rejected), the
  official luminance x cs expressions built from the official
  num0/num1/den0/den1 intermediate convs (NOT the algebraically
  equivalent but differently-rounded sigma_xy form), NO epsilon in
  the denominators and NO ssim clamping (both are External A/B
  training-stability hacks), spatial mean over the conv2d spatial
  axes -> (N, C), and the official float32 cast round-trip for
  non-float32 inputs. filter_size is used as given (NOT forced odd,
  as External A/B do - official DFL passes e.g. 22 for resolution
  256).
- ``gaussian_blur(input, radius=2.0)``: ``radius`` is the official
  sigma; kernel_size = max(3, int(2*2*sigma)) forced odd; 1D
  gaussian around mean = floor(0.5*kernel_size); 2D outer product
  normalized to sum 1 (float32, exactly as the official numpy
  construction); symmetric padding kernel_size//2; single-pass
  VALID depthwise convolution (External A/B's OpenCV sigma rule and
  separable two-pass forms produce different kernels/rounding and
  are rejected).
- ``style_loss(target, style, gaussian_blur_radius=0.0,
  loss_weight=1.0, step_size=1)``: the official PER-CHANNEL MOMENTS
  (mean/variance) formulation - NOT a gram-matrix loss (External
  A/B's gram variant is rejected for the compatibility path):
  TF ``tf.nn.moments`` semantics (mean, then mean of squared
  deviations), std = sqrt(var + 1e-5), BOTH loss terms squared,
  summed over all axes except the batch (per-sample vector (N,)),
  scaled by loss_weight / channel_count; raises on channel-count
  mismatch (the legacy bridge forgot to square std_loss - rejected);
  the ``step_size`` parameter is kept for signature parity (unused
  in the official implementation).
- ``pixel_norm(x, axes)``: ``x * rsqrt(mean(x^2, axes, keepdims) +
  1e-06)`` with the official epsilon 1e-6 (External B's 1e-8
  epsilon is rejected) and the official required-``axes`` signature.

flatten / reshape_4D / average_tensor_list / total_variation_mse
(Phase 3E1)
=================================================================

- ``flatten(x)``: the official always flattens in the NCHW layout
  (channel-major ``(-1, C*H*W)``): an NHWC input is boundary-
  permuted to NCHW first so the result is channel-major regardless
  of ``nn.data_format`` (the legacy ``torch.flatten(x, start_dim=1)``
  and External A's NCHW-only reshape flatten an NHWC input
  spatial-major - rejected).
- ``reshape_4D(x, w, h, c)``: the official always reshapes the flat
  tail as the NCHW layout ``(-1, c, h, w)`` (channel-major); an NHWC
  data_format produces an NHWC tensor (boundary permutation). The
  flat tail must contain exactly ``c*h*w`` elements per sample -
  torch's reshape fails explicitly otherwise (no permissive
  heuristics).
- ``average_tensor_list(tensors_list, tf_device_string=None)``:
  single element -> returned as-is; otherwise the mean over a new
  leading axis (official ``expand_dims + concat + reduce_mean``).
  ``tf_device_string`` is kept for official signature parity; under
  torch the placement comes from the input tensors (Phase 2 device
  model).
- ``total_variation_mse(images)``: the official formula VERBATIM -
  ``dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]`` and
  ``dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]``, each squared and
  summed over axes (1,2,3) -> a PER-SAMPLE (N,) vector of SUMS.
  The formula is data-format-agnostic in the official source: with
  NCHW (the format the baseline models run), axis 1 is the channel
  axis, so the first term is a channel-difference term - that is
  exactly what the official baseline GAN loss computed, and this
  project reproduces it rather than "fixing" it to spatial axes
  (the USER_LEGACY NCHW branch and External A both deviate; External
  A additionally returns a global scalar MEAN - rejected on both
  counts: sum not mean, (N,) not scalar).

random_binomial (Phase 3E2)
===========================

The official DFL op (the lr_dropout mask source for AdaBelief/RMSprop
- the ONLY baseline consumers):

    where(random_uniform(shape, seed=...) < p, ones(shape, dtype),
          zeros(shape, dtype))

- P(mask == 1) == p; values are exactly 0/1 in the given ``dtype``.
  The official compares a FLOAT16 random draw (a TF precision quirk);
  the torch implementation draws in float32, distributionally
  equivalent. Exact TensorFlow RNG-stream parity is NOT claimed and
  NOT verified (no TF environment in this project).
- the official ``seed=None`` path picks a fresh random seed per call
  (``np.random.randint(10e6)``); under torch a given ``seed``
  initializes a local generator (deterministic per call) and
  ``seed=None`` draws from the global torch RNG - the optimizers
  rely on the per-call resampling (one fresh mask per optimizer
  step, exactly like the official TF graph re-evaluating its random
  op on every run of the update op).
- the optional ``device`` keyword is a torch-only convenience for
  placing the mask where the optimizers need it; the official
  signature parameters (shape, p, dtype, seed) are all preserved.
"""

import numpy as np

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
            "silently; this project fails explicitly)"
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


# ---------------------------------------------------------------------------
# Phase 3D: dssim / gaussian_blur / style_loss / pixel_norm
# ---------------------------------------------------------------------------

def _make_dssim_kernel(filter_size, filter_sigma, device, dtype):
    """Official DFL DSSIM window, verbatim: arange centered at
    (filter_size-1)/2, squared x (-0.5/sigma^2), 2D outer product,
    softmax-normalized over the flattened (filter_size*filter_size)
    values (tf.nn.softmax == exp/sum for finite values, max-subtract
    stabilized as TF does). Returns a torch (1, 1, fs, fs) tensor."""
    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1) / 2.0
    kernel = kernel ** 2
    kernel *= (-0.5 / (filter_sigma ** 2))
    kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
    kernel = np.reshape(kernel, (-1,))
    e = np.exp(kernel - kernel.max())
    kernel = (e / e.sum()).astype(np.float32)
    kernel = torch.as_tensor(kernel, device=device)
    kernel = kernel.view(1, 1, filter_size, filter_size)
    return kernel.to(dtype)


def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """Official DFL DSSIM (VALID window convolution, per-sample
    (N, C) result). See the module docstring for the exact formula
    and the rejected External A/B deviations (SAME padding, eps,
    clamping, forced-odd filter size, (N,C,1,1) shape)."""
    if img1.dtype != img2.dtype:
        raise ValueError("img1.dtype != img2.dtype")

    not_float32 = img1.dtype != torch.float32
    if not_float32:
        img_dtype = img1.dtype
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)
    else:
        img_dtype = None

    nhwc = nn.data_format == "NHWC"
    if nhwc:
        img1 = nn.to_data_format(img1, "NCHW", "NHWC")
        img2 = nn.to_data_format(img2, "NCHW", "NHWC")

    filter_size = max(1, filter_size)
    channels = img1.shape[1]
    kernel = _make_dssim_kernel(filter_size, filter_sigma, img1.device, img1.dtype)
    kernel = kernel.repeat(channels, 1, 1, 1)  # official: tf.tile over the channel axis

    def reducer(x):
        # official: tf.nn.depthwise_conv2d(..., padding='VALID') - torch conv with
        # padding=0 is exactly VALID (no pre-padding, unlike gaussian_blur)
        return F.conv2d(x, kernel, padding=0, groups=channels)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2))
    # the official code has `c2 *= 1.0 #compensation factor` (a no-op)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # official: tf.reduce_mean(luminance*cs, axis=nn.conv2d_spatial_axes)
    # -> (N, C). The computation above runs in NCHW (NHWC inputs are
    # boundary-permuted), so the spatial axes are (2, 3) here; the
    # resulting (N, C) is identical in both data formats.
    ssim_val = (luminance * cs).mean(dim=(2, 3))
    dssim = (1.0 - ssim_val) / 2.0

    if not_float32:
        dssim = dssim.to(img_dtype)
    return dssim


def _make_gaussian_blur_kernel(radius):
    """Official DFL gaussian blur kernel, verbatim: kernel_size =
    max(3, int(2*2*sigma)) forced odd, 1D gaussian around
    mean = floor(0.5*kernel_size), 2D outer product normalized to
    sum 1 in float32. Returns (kernel, kernel_size)."""
    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

    kernel_size = max(3, int(2 * 2 * radius))
    if kernel_size % 2 == 0:
        kernel_size += 1
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, radius) for x in range(kernel_size)])
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel = np_kernel / np.sum(np_kernel)
    return kernel, kernel_size


def gaussian_blur(input, radius=2.0):
    """Official DFL gaussian blur (``radius`` is the sigma, exactly as
    the official SAEHD/AMP/Quick96 models call it positionally).
    Single-pass VALID depthwise convolution with symmetric
    padding = kernel_size//2. See the module docstring for the
    rejected External A/B kernel deviations."""
    nhwc = nn.data_format == "NHWC"
    if nhwc:
        input = nn.to_data_format(input, "NCHW", "NHWC")

    gauss_kernel, kernel_size = _make_gaussian_blur_kernel(radius)
    channels = input.shape[1]
    k = torch.as_tensor(gauss_kernel, device=input.device)
    k = k.view(1, 1, kernel_size, kernel_size)
    k = k.repeat(channels, 1, 1, 1)  # official: tf.tile over the channel axis

    x = input
    padding = kernel_size // 2
    if padding != 0:
        x = F.pad(x, (padding, padding, padding, padding))
    # official: tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding='VALID')
    x = F.conv2d(x, k, padding=0, groups=channels)

    if nhwc:
        x = nn.to_data_format(x, "NHWC", "NCHW")
    return x


def _tf_moments(x, axes, keepdims=True):
    """TF ``tf.nn.moments`` semantics: mean, then the mean of squared
    deviations from that mean (NOT the algebraically equivalent
    E[x^2]-E[x]^2, which rounds differently in float32)."""
    m = x.mean(dim=axes, keepdim=keepdims)
    v = (x - m).pow(2).mean(dim=axes, keepdim=keepdims)
    return m, v


def style_loss(target, style, gaussian_blur_radius=0.0, loss_weight=1.0, step_size=1):
    """Official DFL style loss: per-channel MOMENTS (mean/variance)
    matching, NOT a gram-matrix loss (see module docstring).
    ``step_size`` is kept for official signature parity; the official
    implementation does not use it."""
    def sd(content, style, loss_weight):
        content_nc = content.shape[nn.conv2d_ch_axis]
        style_nc = style.shape[nn.conv2d_ch_axis]
        if content_nc != style_nc:
            raise Exception("style_loss() content_nc != style_nc")
        c_mean, c_var = _tf_moments(content, nn.conv2d_spatial_axes, keepdims=True)
        s_mean, s_var = _tf_moments(style, nn.conv2d_spatial_axes, keepdims=True)
        c_std, s_std = torch.sqrt(c_var + 1e-5), torch.sqrt(s_var + 1e-5)
        # official: reduce_sum over [1,2,3] = every axis except the batch,
        # in both NCHW (C,H,W) and NHWC (H,W,C) -> per-sample vector (N,)
        axes = tuple(range(1, content.dim()))
        mean_loss = torch.square(c_mean - s_mean).sum(dim=axes)
        std_loss = torch.square(c_std - s_std).sum(dim=axes)
        return (mean_loss + std_loss) * (loss_weight / content_nc)

    if gaussian_blur_radius > 0.0:
        target = gaussian_blur(target, gaussian_blur_radius)
        style = gaussian_blur(style, gaussian_blur_radius)

    return sd(target, style, loss_weight)


def pixel_norm(x, axes):
    """Official DFL pixel norm: ``x * rsqrt(mean(x^2, axes) + 1e-06)``
    with the official epsilon 1e-6 (External B's 1e-8 is rejected)
    and the official required-``axes`` signature (call sites pass
    e.g. axes=-1)."""
    return x * torch.rsqrt(torch.mean(x.pow(2), dim=axes, keepdim=True) + 1e-06)


# ---------------------------------------------------------------------------
# Phase 3E1: flatten / reshape_4D (tensor layout)
# ---------------------------------------------------------------------------

def flatten(x):
    """Official DFL flatten: channel-major ``(-1, C*H*W)`` in the NCHW
    layout - an NHWC input is boundary-permuted first so the result
    is channel-major regardless of ``nn.data_format`` (the SAEHD
    archi, XSeg and the AMP archi rely on this layout for their
    dense<->map transitions)."""
    if nn.data_format == "NHWC":
        x = nn.to_data_format(x, "NCHW", "NHWC")
    return x.reshape(x.shape[0], -1)


def reshape_4D(x, w, h, c):
    """Official DFL reshape_4D: interpret the flat tail of each sample
    as the NCHW layout ``(-1, c, h, w)`` (channel-major, matching the
    official ``flatten``); an NHWC data_format yields an NHWC tensor
    (boundary permutation). The flat tail must hold exactly c*h*w
    elements per sample - torch's reshape fails explicitly otherwise
    (the official tf.reshape raised InvalidArgumentError)."""
    x = x.reshape(-1, c, h, w)
    if nn.data_format == "NHWC":
        x = nn.to_data_format(x, "NHWC", "NCHW")
    return x


# ---------------------------------------------------------------------------
# Phase 3E1: average_tensor_list / total_variation_mse
# ---------------------------------------------------------------------------

def average_tensor_list(tensors_list, tf_device_string=None):
    """Official DFL average over a list of same-shape tensors:
    single element -> returned as-is, otherwise the element-wise
    mean over a new leading axis (official ``expand_dims + concat +
    reduce_mean``). ``tf_device_string`` is kept for official
    signature parity; under torch the placement comes from the input
    tensors (Phase 2 device model)."""
    if len(tensors_list) == 1:
        return tensors_list[0]
    # torch.stack + mean is exactly the official
    # tf.concat([expand_dims(t, 0) ...], 0) + tf.reduce_mean(..., 0)
    return torch.stack(tensors_list, dim=0).mean(dim=0)


def total_variation_mse(images):
    """Official DFL total variation (MSE-difference form, SAEHD/AMP
    GAN term) VERBATIM:

        dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]
        dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]
        out  = sum(square(dif1), axes 1..3) + sum(square(dif2), axes 1..3)

    -> a PER-SAMPLE (N,) vector of SUMS (never a scalar, never a
    mean). The formula is data-format-agnostic in the official
    source: under NCHW (the format the baseline models run) axis 1
    is the channel axis, so the first term is a channel-difference
    term - exactly what the official baseline GAN loss computed.
    Rejected deviations: USER_LEGACY's NCHW branch (spatial axes
    instead) and External A's variant (spatial axes + global scalar
    MEAN, no batch dimension)."""
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
    # official: reduce_sum over [1,2,3] = every axis except the batch,
    # in both NCHW (C,H,W) and NHWC (H,W,C)
    return (torch.square(pixel_dif1).sum(dim=(1, 2, 3))
            + torch.square(pixel_dif2).sum(dim=(1, 2, 3)))


# ---------------------------------------------------------------------------
# Phase 6B: sigmoid_cross_entropy (the official SAEHD/AMP/XSeg DLoss)
# ---------------------------------------------------------------------------

def sigmoid_cross_entropy(labels, logits):
    """Official DFL DLoss (Model_SAEHD/Model_tf.py L499-500) VERBATIM:

        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits),
            axis=[1,2,3])

    -> a PER-SAMPLE (N,) vector: the elementwise sigmoid cross
    entropy ``-[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]``
    reduced by MEAN over axes [1,2,3] (all axes except the batch).
    Elementwise-identical to torch
    ``F.binary_cross_entropy_with_logits``; the manual per-sample
    mean over (1,2,3) is required because torch's ``'mean'``
    reduction would additionally divide by the batch size (the
    official loss keeps a per-sample vector; SAEHD averages it
    with np.mean when reporting, and nn.gradients uses the
    unaveraged vector inside the per-GPU loss). Data-format
    agnostic (axes 1..3 are C,H,W under NCHW and H,W,C under
    NHWC - the official formula is layout-agnostic). Rejected
    deviations: none (no alternative implementation exists in the
    audited references; USER_LEGACY/EXTERNAL_A carry no torch
    equivalent). reduction='none' is REQUIRED: torch's default
    reduction='mean' would collapse the tensor to a scalar before
    the official per-sample (1,2,3) mean, which is not the
    official elementwise DLoss.
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none').mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Phase 3E2: random_binomial (lr_dropout mask source for the optimizers)
# ---------------------------------------------------------------------------

def random_binomial(shape, p=0.0, dtype=None, seed=None, device=None):
    """Official DFL random_binomial: ``where(random_uniform(shape) <
    p, ones(shape, dtype), zeros(shape, dtype))``. See the module
    docstring for the RNG-stream parity limitations (the official
    float16 comparison quirk is a distributionally-equivalent TF
    detail; a given seed uses a local torch generator, so the mask
    sequence is deterministic per seed but NOT the TF stream)."""
    if dtype is None:
        dtype = torch.float32
    dev = device if device is not None else "cpu"
    if seed is None:
        # global RNG stream (official: a fresh random seed per call)
        mask = torch.rand(shape, dtype=torch.float32, device=dev) < p
    else:
        # deterministic per seed: a local generator (NOT the TF stream)
        generator = torch.Generator(device=dev)
        generator.manual_seed(seed)
        mask = torch.rand(shape, dtype=torch.float32, device=dev,
                          generator=generator) < p
    return mask.to(dtype)


nn.depth_to_space = depth_to_space
nn.dssim = dssim
nn.gaussian_blur = gaussian_blur
nn.style_loss = style_loss
nn.pixel_norm = pixel_norm
nn.flatten = flatten
nn.reshape_4D = reshape_4D
nn.average_tensor_list = average_tensor_list
nn.total_variation_mse = total_variation_mse
nn.random_binomial = random_binomial
nn.sigmoid_cross_entropy = sigmoid_cross_entropy
