"""Phase 3D acceptance: core numerical leras ops (dssim, gaussian_blur,
style_loss, pixel_norm) with official DeepFaceLab semantics.

All ops are compared against independent NumPy implementations of the
official formulas (TF parity: NOT_VERIFIED in this phase - no TF
environment in the tested venvs; parity is against the official
source formulas). Known external deviations rejected by these tests:
External A/B SAME-padding dssim + eps/clamp/forced-odd filter size,
External A/B gram-matrix style_loss, External A/B OpenCV-sigma-rule
gaussian blur, External B pixel_norm epsilon 1e-8.
"""

import ast
import os
import sys

import numpy as np
import pytest

import torch

from core.leras import nn as dfl_nn

CUDA_AVAILABLE = torch.cuda.is_available()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS_TORCH_PATH = os.path.join(REPO_ROOT, "core", "leras", "ops", "__init__.py")


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    # the production data_format of the DFL trainers is NCHW
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


def _random(n, c, h, w, seed, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, c, h, w), dtype=np.float32) * scale + offset).astype(np.float32)


# ---------------------------------------------------------------------------
# independent NumPy references (official formulas, float32)
# ---------------------------------------------------------------------------

def _ref_valid_depthwise(x, kernel, pad):
    """Reference VALID (or pre-padded) depthwise 2D convolution,
    explicit index math - no library."""
    n, c, h, w = x.shape
    f = kernel.shape[0]
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = h + 2 * pad - f + 1, w + 2 * pad - f + 1
    out = np.zeros((n, c, ho, wo), np.float32)
    for a in range(n):
        for ch in range(c):
            for i in range(ho):
                for j in range(wo):
                    s = 0.0
                    for di in range(f):
                        for dj in range(f):
                            s += x[a, ch, i + di, j + dj] * kernel[di, dj]
                    out[a, ch, i, j] = s
    return out


def _ref_gaussian_kernel(radius):
    """Official DFL gaussian blur kernel, verbatim (1D gaussian around
    mean = floor(0.5*k), 2D outer product, f32, sum-normalized)."""
    def g(x, mu, s):
        return np.exp(-(x - mu) ** 2 / (2 * s * s))
    ks = max(3, int(2 * 2 * radius))
    if ks % 2 == 0:
        ks += 1
    mean = np.floor(0.5 * ks)
    k1 = np.array([g(x, mean, radius) for x in range(ks)])
    kk = np.outer(k1, k1).astype(np.float32)
    return kk / np.sum(kk), ks


def _ref_gaussian_blur(x, radius):
    k, ks = _ref_gaussian_kernel(radius)
    return _ref_valid_depthwise(x, k, ks // 2)


def _ref_dssim_kernel(filter_size, filter_sigma):
    """Official DFL DSSIM window, verbatim (arange centered at
    (fs-1)/2, squared x -0.5/sigma^2, 2D outer, softmax = exp/sum)."""
    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1) / 2.0
    kernel = kernel ** 2
    kernel *= (-0.5 / (filter_sigma ** 2))
    kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
    flat = np.reshape(kernel, (-1,))
    e = np.exp(flat - flat.max())
    return (e / e.sum()).astype(np.float32).reshape(filter_size, filter_size)


def _ref_dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5,
               k1=0.01, k2=0.03):
    """Official DFL DSSIM formula, verbatim: VALID window convs for
    mean0/mean1, luminance = (2 m0 m1 + c1)/(m0^2 + m1^2 + c1),
    cs = (2 m12 - 2 m0 m1 + c2)/(E[x^2+y^2] - (m0^2+m1^2) + c2),
    spatial mean -> (N, C)."""
    k = _ref_dssim_kernel(filter_size, filter_sigma)

    def red(x):
        return _ref_valid_depthwise(x, k, 0)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    m0, m1 = red(img1), red(img2)
    num0 = m0 * m1 * 2.0
    den0 = m0 ** 2 + m1 ** 2
    lum = (num0 + c1) / (den0 + c1)
    num1 = red(img1 * img2) * 2.0
    den1 = red(img1 ** 2 + img2 ** 2)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim = (lum * cs).mean(axis=(2, 3))
    return ((1.0 - ssim) / 2.0).astype(np.float32)


def _ref_style_loss(target, style, loss_weight=1.0):
    """Official DFL style loss (per-channel TF moments), verbatim."""
    def moments(x):
        m = x.mean(axis=(2, 3), keepdims=True)
        v = ((x - m) ** 2).mean(axis=(2, 3), keepdims=True)
        return m, v
    cm, cv = moments(target)
    sm, sv = moments(style)
    cstd, sstd = np.sqrt(cv + 1e-5), np.sqrt(sv + 1e-5)
    ml = np.square(cm - sm).sum(axis=(1, 2, 3))
    sl = np.square(cstd - sstd).sum(axis=(1, 2, 3))
    return (ml + sl) * (loss_weight / target.shape[1])


# ---------------------------------------------------------------------------
# gaussian_blur
# ---------------------------------------------------------------------------

def test_gaussian_blur_kernel_normalization():
    """The official kernel must be symmetric and sum to 1 (the blur
    preserves constants)."""
    for radius in (2.0, 4.0, 8.0, 32.0, 0.5):
        k, ks = _ref_gaussian_kernel(radius)
        # kernel_size rule: max(3, int(4*sigma)), forced odd
        expected = max(3, int(2 * 2 * radius))
        if expected % 2 == 0:
            expected += 1
        assert ks == expected
        assert np.allclose(k.sum(), 1.0, atol=1e-6)
        assert np.allclose(k, k.T, atol=1e-6)


def test_gaussian_blur_parity_vs_reference():
    """sigma values used by the official models: res/32 (4..8), res/128
    (2), style res//8 (32). CPU parity vs the independent reference:
    WITHIN_TOLERANCE(1e-4) - same formulas, different accumulation."""
    for sigma, (n, c, h, w) in [
        (2.0, (1, 3, 16, 16)),
        (4.0, (1, 3, 20, 12)),
        (8.0, (2, 3, 32, 32)),
        (32.0, (1, 3, 64, 48)),
    ]:
        x = _random(n, c, h, w, seed=100 + int(sigma), scale=2.0)
        y = dfl_nn.gaussian_blur(torch.from_numpy(x), sigma).detach().numpy()
        assert y.shape == (n, c, h, w)  # output size preserved
        r = _ref_gaussian_blur(x, sigma)
        assert np.allclose(y, r, rtol=1e-4, atol=1e-5)


def test_gaussian_blur_impulse_and_constant():
    # impulse in the center -> the (padded) kernel itself appears at the
    # center output pixel
    h = w = 32
    x = np.zeros((1, 1, h, w), np.float32)
    x[0, 0, h // 2, w // 2] = 1.0
    y = dfl_nn.gaussian_blur(torch.from_numpy(x), 4.0).detach().numpy()
    k, ks = _ref_gaussian_kernel(4.0)
    r = _ref_gaussian_blur(x, 4.0)
    assert np.allclose(y, r, rtol=1e-4, atol=1e-6)
    # with the official symmetric padding (pad = (k-1)/2) the impulse
    # response peak stays at the impulse pixel: output(i) = k[2*pad+i0-i],
    # so at i = i0 the kernel center is read
    assert y[0, 0, h // 2, w // 2] == pytest.approx(
        k[ks // 2, ks // 2], abs=1e-5)
    # one pixel away the kernel's off-center tap is read (much smaller)
    assert y[0, 0, h // 2 + 1, w // 2] == pytest.approx(
        k[ks // 2 - 1, ks // 2], abs=1e-6)

    # constant input -> same constant in the INTERIOR (the official op
    # zero-pads the border, so edge pixels darken - that IS the official
    # edge behavior and must not be "fixed")
    xc = np.full((1, 2, 64, 64), 0.37, np.float32)
    yc = dfl_nn.gaussian_blur(torch.from_numpy(xc), 8.0).detach().numpy()
    assert yc.shape == (1, 2, 64, 64)
    assert np.allclose(yc[:, :, 16:48, 16:48], 0.37, atol=1e-5)
    # and the zero-padded border is visibly darker (official behavior)
    assert float(yc[0, 0, 0, 0]) < 0.37 * 0.5


# ---------------------------------------------------------------------------
# dssim
# ---------------------------------------------------------------------------

def test_dssim_parity_vs_reference():
    """Multiple channel counts / spatial sizes / filter sizes (incl. the
    even filter_size=int(res/11.6) values official DFL passes, e.g. 22)
    vs the independent reference. Parity: WITHIN_TOLERANCE(1e-4) vs the
    NumPy reference; EXACT where the inputs make the formula exact."""
    rng = np.random.default_rng(41)
    for (n, c, h, w, fs, seed) in [
        (1, 1, 16, 16, 3, 1),
        (1, 3, 24, 24, 5, 2),
        (2, 3, 24, 16, 11, 3),
        (1, 3, 24, 24, 22, 4),    # even filter_size (res 256 -> int(256/11.6))
        (1, 4, 20, 20, 7, 5),
        (3, 1, 24, 24, 11, 6),
    ]:
        img1 = rng.random((n, c, h, w)).astype(np.float32)
        img2 = rng.random((n, c, h, w)).astype(np.float32)
        y = dfl_nn.dssim(torch.from_numpy(img1), torch.from_numpy(img2),
                         1.0, filter_size=fs).detach().numpy()
        assert y.shape == (n, c)  # official: spatial mean -> (N, C)
        r = _ref_dssim(img1, img2, 1.0, filter_size=fs)
        assert np.allclose(y, r, rtol=1e-4, atol=1e-5)


def test_dssim_identical_inputs():
    x = _random(1, 3, 16, 16, seed=7, scale=1.0)
    y = dfl_nn.dssim(torch.from_numpy(x), torch.from_numpy(x), 1.0,
                     filter_size=3).detach().numpy()
    # identical inputs: ssim = 1 exactly in exact arithmetic
    assert y.shape == (1, 3)
    assert np.allclose(y, 0.0, atol=1e-5)


def test_dssim_clearly_different_inputs():
    x = _random(1, 3, 16, 16, seed=8, scale=1.0)
    y_img = _random(1, 3, 16, 16, seed=9, scale=1.0)
    y = dfl_nn.dssim(torch.from_numpy(x), torch.from_numpy(y_img), 1.0,
                     filter_size=3).detach().numpy()
    # (1 - ssim)/2 with ssim <= 1 -> well separated inputs give a large dssim
    assert y.shape == (1, 3)
    assert float(y.mean()) > 0.1


def test_dssim_dtype_roundtrip():
    """Official behavior: non-float32 inputs are computed in float32 and
    the result is cast back to the input dtype."""
    x = torch.from_numpy(_random(1, 2, 12, 12, seed=10))
    y32 = dfl_nn.dssim(x, x * 0.5, 1.0, filter_size=3)
    assert y32.dtype == torch.float32
    y16 = dfl_nn.dssim(x.to(torch.float16), (x * 0.5).to(torch.float16),
                       1.0, filter_size=3)
    assert y16.dtype == torch.float16
    assert torch.allclose(y16.to(torch.float32), y32, atol=2e-2)


def test_dssim_dtype_mismatch_rejection():
    x = torch.from_numpy(_random(1, 2, 12, 12, seed=11))
    with pytest.raises(ValueError):
        dfl_nn.dssim(x, x.to(torch.float64), 1.0, filter_size=3)


def test_dssim_valid_semantics_shape():
    """VALID window convolution: with H=W=24 and filter 11 the internal
    ssim map is (H-10, W-10) = 14x14; the public op reduces it to
    (N, C). A SAME-padded implementation (External A/B) would differ in
    value at every edge pixel - the reference comparison above catches
    that; here we pin the reduction shape explicitly."""
    x = _random(1, 3, 24, 24, seed=12)
    y = dfl_nn.dssim(torch.from_numpy(x), torch.from_numpy(x * 0.7 + 0.2),
                     1.0, filter_size=11).detach().numpy()
    assert y.shape == (1, 3)
    r = _ref_dssim(x, (x * 0.7 + 0.2).astype(np.float32), 1.0, filter_size=11)
    assert np.allclose(y, r, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# style_loss
# ---------------------------------------------------------------------------

def test_style_loss_parity_vs_reference():
    """Per-batch (N,) results, loss_weight scaling, both terms squared:
    verified against the official per-channel moments formula."""
    for (n, c, h, w, lw, seed) in [
        (1, 3, 16, 16, 1.0, 21),
        (2, 4, 16, 16, 2.5, 22),
        (4, 1, 8, 20, 10000.0, 23),   # SAEHD face_style scaling
        (1, 3, 12, 8, 1.0, 24),       # non-square
    ]:
        t = _random(n, c, h, w, seed=seed, scale=2.0)
        s = _random(n, c, h, w, seed=seed + 100, scale=2.0)
        y = dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(s),
                              loss_weight=lw).detach().numpy()
        assert y.shape == (n,)  # official: per-sample vector
        r = _ref_style_loss(t, s, loss_weight=lw)
        assert np.allclose(y, r, rtol=1e-4, atol=1e-5)


def test_style_loss_identical_inputs_zero():
    t = _random(2, 3, 16, 16, seed=25)
    y = dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(t)).detach().numpy()
    assert y.shape == (2,)
    assert bool((y == 0).all())  # both moment terms are exactly 0


def test_style_loss_shifted_and_scaled():
    # shifted: same variance -> the mean term is nonzero
    t = _random(1, 3, 16, 16, seed=26)
    shifted = (t + 0.5).astype(np.float32)
    y = dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(shifted)).detach().numpy()
    assert y[0] > 0
    # zero-mean data scaled by 2: same mean (0), different std
    rng = np.random.default_rng(27)
    z = (rng.random((1, 3, 16, 16)) - 0.5).astype(np.float32)
    scaled = (z * 2.0).astype(np.float32)
    y2 = dfl_nn.style_loss(torch.from_numpy(z), torch.from_numpy(scaled)).detach().numpy()
    assert y2[0] > 0
    # identical zero-mean data scaled by 1: 0 (sanity of the setup)
    y3 = dfl_nn.style_loss(torch.from_numpy(z), torch.from_numpy(z)).detach().numpy()
    assert bool((y3 == 0).all())


def test_style_loss_not_gram_matrix():
    """Discriminator: two images with IDENTICAL per-channel moments
    (mean 1.5, var 2) but different cross-channel correlation (r=+1
    vs r=-1). The official moments loss is exactly 0; a gram-matrix
    variant (External A/B) would return a large positive value. This
    proves the compatibility path is NOT the gram variant."""
    base = np.tile(np.tile(np.array([0, 1, 2, 3], np.float32), 2).reshape(8, 1), (1, 8))
    # base[i][j] = (i % 4), constant in j: per-channel mean 1.5,
    # E[x^2] 3.5, var 2 - identical moments in content and style
    # (stack along axis 0 of the 2-D arrays -> (2, 8, 8), then batch)
    content = np.stack([base, base.copy()], axis=0)[None]                        # r = +1
    style = np.stack([base, (3.0 - base).astype(np.float32)], axis=0)[None]      # r = -1
    y = dfl_nn.style_loss(torch.from_numpy(content), torch.from_numpy(style)).detach().numpy()
    assert y.shape == (1,)
    assert np.allclose(y, 0.0, atol=1e-5)
    # the gram off-diagonal differs, so a gram-matrix loss would be > 0:
    off = abs(float(content[0, 0].ravel() @ content[0, 1].ravel())
              - float(style[0, 0].ravel() @ style[0, 1].ravel()))
    assert off > 100.0


def test_style_loss_gaussian_blur_path():
    """gaussian_blur_radius > 0 blurs both inputs first (official);
    blurred identical inputs still give 0."""
    t = _random(1, 3, 32, 32, seed=29)
    y = dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(t),
                          gaussian_blur_radius=4.0).detach().numpy()
    assert bool(np.allclose(y, 0.0, atol=1e-5))
    s = _random(1, 3, 32, 32, seed=30)
    y2 = dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(s),
                           gaussian_blur_radius=4.0).detach().numpy()
    # blurred reference (independent) - tolerance is looser after the blur
    tb = _ref_gaussian_blur(t, 4.0)
    sb = _ref_gaussian_blur(s, 4.0)
    r = _ref_style_loss(tb, sb)
    assert np.allclose(y2, r, rtol=1e-3, atol=1e-4)


def test_style_loss_channel_mismatch_rejection():
    t = _random(1, 3, 16, 16, seed=31)
    s = _random(1, 4, 16, 16, seed=32)
    with pytest.raises(Exception, match="style_loss"):
        dfl_nn.style_loss(torch.from_numpy(t), torch.from_numpy(s))


# ---------------------------------------------------------------------------
# pixel_norm
# ---------------------------------------------------------------------------

def test_pixel_norm_deterministic_reference():
    """Official formula x * rsqrt(mean(x^2, axes) + 1e-6); the epsilon
    check discriminates against External B's 1e-8."""
    x = torch.full((1, 4, 3, 3), 1e-3)
    y = dfl_nn.pixel_norm(x, axes=-1).detach().numpy()
    m = (x.numpy() ** 2).mean(axis=-1, keepdims=True)
    ref_1e6 = x.numpy() * (1.0 / np.sqrt(m + 1e-6))
    ref_1e8 = x.numpy() * (1.0 / np.sqrt(m + 1e-8))
    assert np.allclose(y, ref_1e6, rtol=1e-6, atol=1e-9)
    # with 1e-8 the result would differ by > 0.28 here - a hard
    # discriminator of the epsilon value
    assert np.abs(ref_1e6 - ref_1e8).max() > 0.2


def test_pixel_norm_zero_and_near_zero_stability():
    y0 = dfl_nn.pixel_norm(torch.zeros(1, 2, 4, 4), axes=-1).detach().numpy()
    assert bool((y0 == 0).all())  # 0 * rsqrt(0 + 1e-6) = 0, no NaN
    yz = dfl_nn.pixel_norm(torch.full((1, 2, 4, 4), 1e-8), axes=-1).detach().numpy()
    assert bool(np.all(np.isfinite(yz)))


def test_pixel_norm_random_parity():
    x = torch.from_numpy(_random(1, 3, 5, 5, seed=33))
    y = dfl_nn.pixel_norm(x, axes=-1).detach().numpy()
    m = (x.numpy() ** 2).mean(axis=-1, keepdims=True)
    ref = x.numpy() * (1.0 / np.sqrt(m + 1e-6))
    assert np.allclose(y, ref, rtol=1e-5, atol=1e-6)
    # axes over the channel dimension (as SAEHD inter-code uses axes=-1,
    # AMP uses the flattened last axis; both are single-axis here)
    y2 = dfl_nn.pixel_norm(x, axes=1).detach().numpy()
    m2 = (x.numpy() ** 2).mean(axis=1, keepdims=True)
    ref2 = x.numpy() * (1.0 / np.sqrt(m2 + 1e-6))
    assert np.allclose(y2, ref2, rtol=1e-5, atol=1e-6)


def test_pixel_norm_gradient_smoke():
    x = torch.from_numpy(_random(1, 2, 4, 4, seed=34)).requires_grad_(True)
    dfl_nn.pixel_norm(x, axes=-1).sum().backward()
    assert x.grad is not None
    assert bool(torch.all(torch.isfinite(x.grad)))


# ---------------------------------------------------------------------------
# gradient flow for the differentiable loss/blur ops
# ---------------------------------------------------------------------------

def test_dssim_and_gaussian_blur_gradient_smoke():
    x = torch.from_numpy(_random(1, 3, 16, 16, seed=35)).requires_grad_(True)
    y = torch.from_numpy(_random(1, 3, 16, 16, seed=36))
    dfl_nn.dssim(x, y, 1.0, filter_size=3).sum().backward()
    assert x.grad is not None
    assert bool(torch.all(torch.isfinite(x.grad)))
    assert bool((x.grad != 0).any())

    xb = torch.from_numpy(_random(1, 2, 16, 16, seed=37)).requires_grad_(True)
    dfl_nn.gaussian_blur(xb, 4.0).sum().backward()
    assert xb.grad is not None
    assert bool(torch.all(torch.isfinite(xb.grad)))


# ---------------------------------------------------------------------------
# NHWC boundary contract (official DFL call sites use NCHW)
# ---------------------------------------------------------------------------

def test_dssim_nhwc_data_format_contract():
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NHWC")
    try:
        x = _random(1, 3, 16, 16, seed=38)
        y = _random(1, 3, 16, 16, seed=39)
        xn = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        yn = np.ascontiguousarray(y.transpose(0, 2, 3, 1))
        r = dfl_nn.dssim(torch.from_numpy(xn), torch.from_numpy(yn), 1.0,
                         filter_size=5).detach().numpy()
        # the reduced result is (N, C) in both data formats
        assert r.shape == (1, 3)
        # and it equals the NCHW result for the same data
        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
        rn = dfl_nn.dssim(torch.from_numpy(x), torch.from_numpy(y), 1.0,
                          filter_size=5).detach().numpy()
        assert np.allclose(r, rn, rtol=1e-4, atol=1e-5)
    finally:
        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


# ---------------------------------------------------------------------------
# RTX 4090 execution and CPU-vs-GPU parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_gpu_ops_on_rtx4090():
    """GPU execution through the Phase 2 abstraction. Parity vs CPU:
    WITHIN_TOLERANCE(1e-4) for float32 convolution accumulation."""
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), data_format="NCHW")
    try:
        assert dfl_nn.device.type == "cuda"
        a = torch.from_numpy(_random(1, 3, 24, 24, seed=40)).to(dfl_nn.device)
        b = torch.from_numpy(_random(1, 3, 24, 24, seed=41)).to(dfl_nn.device)

        yg = dfl_nn.dssim(a, b, 1.0, filter_size=5)
        assert yg.device.type == "cuda"
        ygc = dfl_nn.gaussian_blur(a, 4.0)
        assert ygc.device.type == "cuda"
        ysc = dfl_nn.style_loss(a, b, gaussian_blur_radius=4.0)
        assert ysc.device.type == "cuda"
        ypn = dfl_nn.pixel_norm(a, axes=-1)
        assert ypn.device.type == "cuda"
        assert ypn.dtype == torch.float32

        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
        ac = torch.from_numpy(_random(1, 3, 24, 24, seed=40))
        bc = torch.from_numpy(_random(1, 3, 24, 24, seed=41))
        assert np.allclose(yg.cpu().numpy(),
                           dfl_nn.dssim(ac, bc, 1.0, filter_size=5).numpy(),
                           rtol=1e-4, atol=1e-4)
        assert np.allclose(ygc.cpu().numpy(), dfl_nn.gaussian_blur(ac, 4.0).numpy(),
                           rtol=1e-4, atol=1e-4)
        assert np.allclose(ysc.cpu().numpy(),
                           dfl_nn.style_loss(ac, bc, gaussian_blur_radius=4.0).numpy(),
                           rtol=1e-4, atol=1e-4)
        assert np.allclose(ypn.cpu().numpy(), dfl_nn.pixel_norm(ac, axes=-1).numpy(),
                           rtol=1e-5, atol=1e-7)
    finally:
        dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")


# ---------------------------------------------------------------------------
# import boundary and no direct CUDA in the ops source
# ---------------------------------------------------------------------------

def test_ops_package_still_tf_free():
    import core.leras.ops  # noqa: F401
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    assert "tensorflow" not in sys.modules
    assert "keras" not in sys.modules


def test_no_direct_cuda_in_op_source():
    """AST scan: the torch ops module must not call torch.cuda.* or
    hardcode 'cuda:' device strings (Phase 2 backend neutrality)."""
    with open(OPS_TORCH_PATH, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=OPS_TORCH_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            assert not chain.startswith("torch.cuda"), (
                f"direct CUDA call in ops source: {chain}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.startswith("cuda:"), (
                f"hardcoded cuda: device string in ops source: {node.value!r}")


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))
