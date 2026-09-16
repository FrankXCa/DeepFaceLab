"""Phase 3B smoke tests: torch concrete leras layers.

Covers the required Phase 3B validation areas for every migrated layer
(Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, DenseNorm,
BatchNorm2D, InstanceNorm2D, FRNorm2D, BlurPool, AdaIN, TLU, ScaleAdd):

- constructor/config state (+ type errors)
- build lifecycle (two-phase, Phase 3A contract)
- parameter names (official checkpoint names) and shapes
- dtype / device (nn.device via the Phase 2 abstraction)
- forward output shape + parity against manual NumPy references
  (deterministic unique-value tensors; parity label per test:
  EXACT or WITHIN_TOLERANCE(1e-4) — float32 accumulation order)
- CPU execution (always) and CUDA execution on the RTX 4090
  (skip-if-CPU-only)
- repeated-construction naming/order stability
- serialization through the Phase 3A Saveable (official-layout files)
- synthetic official-layout -> torch-layout conversion with exact
  index maps (no element-count heuristics)
- invalid-shape rejection (strict)
- no TensorFlow import / no direct CUDA API in layer sources
"""

import os
import pickle
import sys

import pytest
import numpy as np
import torch

from core.leras import nn as dfl_nn
from core.leras import checkpoint as ckpt
from core.leras import device as device_layer
from core.leras.layers import (
    Conv2D,
    Conv2DTranspose,
    DepthwiseConv2D,
    Dense,
    DenseNorm,
    BatchNorm2D,
    InstanceNorm2D,
    FRNorm2D,
    BlurPool,
    AdaIN,
    TLU,
    ScaleAdd,
)

CUDA_AVAILABLE = torch.cuda.is_available()

ALL_LAYER_CLASSES = [
    Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, DenseNorm,
    BatchNorm2D, InstanceNorm2D, FRNorm2D, BlurPool, AdaIN, TLU, ScaleAdd,
]


@pytest.fixture(autouse=True)
def _cpu_foundation_nchw():
    """Every test starts from the CPU foundation in NCHW (the DFL
    data_format used by the official mainscripts). GPU tests
    re-initialize inside their body."""
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    yield


# ---------------------------------------------------------------------------
# deterministic unique-value tensors (inspected values, not random)
# ---------------------------------------------------------------------------

def _seq(n, scale=1.0, offset=0.0):
    return (np.arange(n, dtype=np.float32) * scale + offset)


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

def test_conv2d_config_and_errors():
    c = Conv2D(3, 2, kernel_size=3, strides=1, padding="SAME", dilations=1,
               use_bias=True, use_wscale=False, name="c")
    assert (c.in_ch, c.out_ch, c.kernel_size, c.strides, c.dilations) == (3, 2, 3, 1, 1)
    assert c.padding == 1  # SAME -> int
    assert c.use_bias and not c.use_wscale

    v = Conv2D(3, 2, kernel_size=3, padding="VALID", name="c")
    assert v.padding == 0
    s = Conv2D(3, 2, kernel_size=4, padding="SAME", name="c")
    assert s.padding == 2  # ((4-1)*1+1)//2

    with pytest.raises(ValueError):
        Conv2D(3, 2, kernel_size=3, strides=2.0, name="c")
    with pytest.raises(ValueError):
        Conv2D(3, 2, kernel_size=3, dilations="x", name="c")
    with pytest.raises(ValueError):
        Conv2D(3, 2, kernel_size=3, padding="WRONG", name="c")


def test_conv2d_build_lifecycle_and_names():
    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c")
    assert len(list(c.parameters())) == 0  # phase 1: config only

    c.build_weights()
    assert tuple(c.weight.shape) == (2, 3, 3, 3)  # torch OIHW
    assert tuple(c.bias.shape) == (2,)
    assert c.weight.device == dfl_nn.device
    assert c.weight.dtype == dfl_nn.floatx
    assert c.bias.device == dfl_nn.device
    # official names in deterministic order
    assert [k for k, _ in c._iter_official_weights()] == ["weight:0", "bias:0"]
    # registered initializers: glorot (default), zeros bias
    inits = c.get_param_initializers()
    assert set(inits) == {"weight", "bias"}

    c.init_weights()
    assert bool(torch.all(c.bias == 0))  # zeros bias applied
    # wscale is a non-saved float (official tf.constant)
    assert "wscale" not in [id(p) for p in c.get_weights()]
    assert not hasattr(c, "wscale")  # only created when use_wscale


def test_conv2d_forward_parity_cpu():
    # parity: WITHIN_TOLERANCE(1e-4) vs manual NumPy (float32
    # accumulation order differs between torch and the python loops)
    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c")
    c.build_weights(); c.init_weights()
    x = torch.from_numpy(_seq(108, 7.0, 1.0).reshape(1, 3, 6, 6))
    y = c(x).detach()
    assert tuple(y.shape) == (1, 2, 6, 6)

    refp = np.pad(x.numpy().transpose(0, 2, 3, 1),
                  ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant").astype(np.float32)
    w = c.weight.detach().numpy().transpose(2, 3, 1, 0)  # HWIO
    b = c.bias.detach().numpy()
    out = np.zeros((1, 6, 6, 2), np.float32)
    for h in range(6):
        for wi in range(6):
            for kh in range(3):
                for kw in range(3):
                    out[0, h, wi, :] += (refp[0, h + kh, wi + kw, :][:, None] * w[kh, kw, :, :]).sum(axis=0)
    out += b
    # float32: relative 1e-4 / absolute 1e-4 (accumulation order)
    assert np.allclose(y.numpy().transpose(0, 2, 3, 1), out,
                       rtol=1e-4, atol=1e-4)


def test_conv2d_forward_shapes_stride_dilation_valid():
    a = Conv2D(4, 8, kernel_size=1, padding="VALID", strides=2, name="a")
    a.build_weights(); a.init_weights()
    xa = torch.from_numpy(_seq(4 * 4 * 4, 1.0, 1.0).reshape(1, 4, 4, 4))
    assert tuple(a(xa).shape) == (1, 8, 2, 2)

    b = Conv2D(2, 3, kernel_size=3, padding="SAME", dilations=2, name="b")
    b.build_weights(); b.init_weights()
    assert b.padding == 2
    xb = torch.from_numpy(_seq(2 * 8 * 8, 1.0, 1.0).reshape(1, 2, 8, 8))
    yb = b(xb)
    # effective kernel 5, stride 1, pad 2 -> out 8
    assert tuple(yb.shape) == (1, 3, 8, 8)

    v = Conv2D(2, 3, kernel_size=3, padding="VALID", name="v")
    v.build_weights(); v.init_weights()
    xv = torch.from_numpy(_seq(2 * 6 * 6, 1.0, 1.0).reshape(1, 2, 6, 6))
    assert tuple(v(xv).shape) == (1, 3, 4, 4)

    n = Conv2D(2, 3, kernel_size=3, padding="SAME", use_bias=False, name="n")
    n.build_weights(); n.init_weights()
    assert len(n.get_weights()) == 1


def test_conv2d_layout_conversion_exact():
    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c")
    c.build_weights()
    # official HWIO -> torch OIHW: exact index map
    off = np.arange(3 * 3 * 3 * 2, dtype=np.float32).reshape(3, 3, 3, 2)
    conv = c.convert_weight_layout(off, c.weight)
    assert tuple(conv.shape) == (2, 3, 3, 3)
    assert bool(torch.equal(conv, torch.from_numpy(off.transpose(3, 2, 0, 1))))
    # inverse is lossless
    back = c.convert_weight_to_official(c.weight.detach().cpu().numpy(), c.weight)
    assert back.shape == (3, 3, 3, 2)
    assert bool(torch.equal(c.convert_weight_layout(back, c.weight), c.weight.detach()))


def test_conv2d_saveable_roundtrip_official_layout(plain_tmp):
    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c1")
    c.build_weights(); c.init_weights()
    path = os.path.join(plain_tmp, "c1.npy")
    c.save_weights(path)

    d = pickle.loads(open(path, "rb").read())
    # the file is OFFICIAL layout (HWIO), official keys
    assert set(d) == {"weight:0", "bias:0"}
    assert d["weight:0"].shape == (3, 3, 3, 2)

    c2 = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c1")
    c2.build_weights()
    assert c2.load_weights(path) is True
    assert torch.equal(c2.weight, c.weight)
    assert torch.equal(c2.bias, c.bias)


def test_conv2d_invalid_shape_rejection(plain_tmp):
    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c1")
    c.build_weights(); c.init_weights()

    with pytest.raises(ValueError):
        c.set_weights([torch.zeros((2, 4, 3, 3)), c.bias.detach()])
    with pytest.raises(ValueError):
        c.set_weights([c.weight.detach()])

    # strict load: official-layout file with a wrong kernel shape
    bad = {
        "weight:0": np.zeros((3, 3, 2, 3), dtype=np.float32),  # transposed-ish
        "bias:0": np.zeros((2,), dtype=np.float32),
    }
    p = os.path.join(plain_tmp, "bad.npy")
    open(p, "wb").write(pickle.dumps(bad, 4))
    c3 = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c1")
    c3.build_weights()
    with pytest.raises(ckpt.CheckpointLoadError):
        c3.load_weights(p)


# ---------------------------------------------------------------------------
# Conv2DTranspose
# ---------------------------------------------------------------------------

def test_conv2dtranspose_config_and_build():
    t = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="SAME", name="t")
    assert (t.in_ch, t.out_ch, t.kernel_size, t.strides) == (2, 4, 3, 2)
    assert t.deconv_length(4, 2, 3, "SAME") == 8
    assert t.deconv_length(4, 2, 3, "VALID") == 9
    t.build_weights()
    assert tuple(t.weight.shape) == (2, 4, 3, 3)  # torch (in,out,k,k)
    assert tuple(t.bias.shape) == (4,)
    assert [k for k, _ in t._iter_official_weights()] == ["weight:0", "bias:0"]
    with pytest.raises(ValueError):
        Conv2DTranspose(2, 4, kernel_size=3, strides=2.0, name="t")


def test_conv2dtranspose_forward_parity_cpu():
    # parity: WITHIN_TOLERANCE(1e-4) vs a manual scatter reference
    t = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="SAME", name="t")
    t.build_weights(); t.init_weights()
    x = torch.from_numpy(_seq(32, 3.0, 2.0).reshape(1, 2, 4, 4))
    y = t(x).detach()
    assert tuple(y.shape) == (1, 4, 8, 8)  # SAME: 4*2

    wref = t.weight.detach().numpy()  # (in, out, k, k)
    xref = x.numpy().transpose(0, 2, 3, 1)
    outref = np.zeros((1, 8, 8, 4), np.float32)
    for i in range(2):
        for h in range(4):
            for wi in range(4):
                for kh in range(3):
                    for kw in range(3):
                        oh, ow = 2 * h - 1 + kh, 2 * wi - 1 + kw
                        if 0 <= oh < 8 and 0 <= ow < 8:
                            outref[0, oh, ow, :] += xref[0, h, wi, i] * wref[i, :, kh, kw]
    outref += t.bias.detach().numpy()
    assert np.allclose(y.numpy().transpose(0, 2, 3, 1), outref,
                       rtol=1e-4, atol=1e-4)


def test_conv2dtranspose_forward_valid_and_unreachable_full():
    t = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="VALID", name="t")
    t.build_weights(); t.init_weights()
    x = torch.from_numpy(_seq(32, 1.0, 1.0).reshape(1, 2, 4, 4))
    assert tuple(t(x).shape) == (1, 4, 9, 9)  # VALID: 4*2 + (3-2)

    # FULL convT is unreachable in torch (output_padding >= stride):
    # explicit failure, not a silent repair (unused by official DFL)
    f = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="FULL", name="f")
    f.build_weights(); f.init_weights()
    xf = torch.from_numpy(_seq(32, 1.0, 1.0).reshape(1, 2, 4, 4))
    with pytest.raises(ValueError):
        f(xf)


def test_conv2dtranspose_layout_conversion_exact():
    t = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="SAME", name="t")
    t.build_weights()
    # official (kH, kW, out, in) -> torch (in, out, kH, kW)
    off = np.arange(3 * 3 * 4 * 2, dtype=np.float32).reshape(3, 3, 4, 2)
    conv = t.convert_weight_layout(off, t.weight)
    assert tuple(conv.shape) == (2, 4, 3, 3)
    assert bool(torch.equal(conv, torch.from_numpy(off.transpose(3, 2, 0, 1))))
    back = t.convert_weight_to_official(t.weight.detach().cpu().numpy(), t.weight)
    assert back.shape == (3, 3, 4, 2)
    assert bool(torch.equal(t.convert_weight_layout(back, t.weight), t.weight.detach()))


def test_conv2dtranspose_saveable_roundtrip_official_layout(plain_tmp):
    t = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="SAME", name="t1")
    t.build_weights(); t.init_weights()
    path = os.path.join(plain_tmp, "t1.npy")
    t.save_weights(path)
    d = pickle.loads(open(path, "rb").read())
    assert d["weight:0"].shape == (3, 3, 4, 2)  # official (H,W,out,in)

    t2 = Conv2DTranspose(2, 4, kernel_size=3, strides=2, padding="SAME", name="t1")
    t2.build_weights()
    assert t2.load_weights(path) is True
    assert torch.equal(t2.weight, t.weight)


# ---------------------------------------------------------------------------
# DepthwiseConv2D
# ---------------------------------------------------------------------------

def test_depthwise_conv2d_config_build_and_names():
    d = DepthwiseConv2D(3, kernel_size=3, padding="SAME", depth_multiplier=2,
                        dilations=1, strides=1, name="d")
    assert d.padding == 1
    d.build_weights()
    assert tuple(d.weight.shape) == (6, 1, 3, 3)  # (in*dm, 1, k, k)
    assert tuple(d.bias.shape) == (6,)
    assert [k for k, _ in d._iter_official_weights()] == ["weight:0", "bias:0"]
    with pytest.raises(ValueError):
        DepthwiseConv2D(3, kernel_size=3, strides=1.0, name="d")
    with pytest.raises(ValueError):
        DepthwiseConv2D(3, kernel_size=3, dilations=1.0, name="d")


def test_depthwise_conv2d_forward_parity_cpu():
    # parity: WITHIN_TOLERANCE(1e-4) vs manual depthwise reference
    d = DepthwiseConv2D(3, kernel_size=3, padding="SAME", depth_multiplier=2, name="d")
    d.build_weights(); d.init_weights()
    x = torch.from_numpy(_seq(75, 5.0, 1.0).reshape(1, 3, 5, 5))
    y = d(x).detach()
    assert tuple(y.shape) == (1, 6, 5, 5)

    wd = d.weight.detach().numpy()
    refp = np.pad(x.numpy(), ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant").astype(np.float32)
    out = np.zeros((1, 6, 5, 5), np.float32)
    for c_ in range(3):
        for m in range(2):
            for h in range(5):
                for w_ in range(5):
                    s = 0.0
                    for kh in range(3):
                        for kw in range(3):
                            s += refp[0, c_, h + kh, w_ + kw] * wd[c_ * 2 + m, 0, kh, kw]
                    out[0, c_ * 2 + m, h, w_] = s + d.bias.detach().numpy()[c_ * 2 + m]
    assert np.allclose(y.numpy(), out, rtol=1e-4, atol=1e-4)


def test_depthwise_conv2d_layout_conversion_exact():
    d = DepthwiseConv2D(3, kernel_size=3, padding="SAME", depth_multiplier=2, name="d")
    d.build_weights()
    # official (kH, kW, in, dm) -> torch (in*dm, 1, kH, kW)
    off = np.arange(3 * 3 * 3 * 2, dtype=np.float32).reshape(3, 3, 3, 2)
    conv = d.convert_weight_layout(off, d.weight)
    assert tuple(conv.shape) == (6, 1, 3, 3)
    expected = off.transpose(2, 3, 0, 1).reshape(6, 1, 3, 3)
    assert bool(torch.equal(conv, torch.from_numpy(expected)))
    back = d.convert_weight_to_official(d.weight.detach().cpu().numpy(), d.weight)
    assert back.shape == (3, 3, 3, 2)
    assert bool(torch.equal(d.convert_weight_layout(back, d.weight), d.weight.detach()))


def test_depthwise_conv2d_saveable_roundtrip_official_layout(plain_tmp):
    d = DepthwiseConv2D(3, kernel_size=3, padding="SAME", depth_multiplier=2, name="d1")
    d.build_weights(); d.init_weights()
    path = os.path.join(plain_tmp, "d1.npy")
    d.save_weights(path)
    dd = pickle.loads(open(path, "rb").read())
    assert dd["weight:0"].shape == (3, 3, 3, 2)  # official (H,W,in,dm)

    d2 = DepthwiseConv2D(3, kernel_size=3, padding="SAME", depth_multiplier=2, name="d1")
    d2.build_weights()
    assert d2.load_weights(path) is True
    assert torch.equal(d2.weight, d.weight)


# ---------------------------------------------------------------------------
# Dense / DenseNorm
# ---------------------------------------------------------------------------

def test_dense_config_build_and_parity():
    # weight keeps the OFFICIAL (in, out) layout; parity:
    # WITHIN_TOLERANCE(1e-4) vs x @ W (float32)
    d = Dense(4, 6, name="dn")
    d.build_weights(); d.init_weights()
    assert tuple(d.weight.shape) == (4, 6)
    assert tuple(d.bias.shape) == (6,)
    x = torch.from_numpy(_seq(32, 2.0, 1.0).reshape(8, 4))
    y = d(x).detach()
    assert tuple(y.shape) == (8, 6)
    ref = (x.numpy() @ d.weight.detach().numpy() + d.bias.detach().numpy()[None, :]).astype(np.float32)
    assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_dense_maxout_parity():
    d = Dense(3, 4, maxout_ch=2, name="dm")
    d.build_weights(); d.init_weights()
    assert tuple(d.weight.shape) == (3, 8)  # (in, out*maxout)
    x = torch.from_numpy(_seq(15, 1.5, 0.5).reshape(5, 3))
    y = d(x).detach()
    assert tuple(y.shape) == (5, 4)
    ref = (x.numpy() @ d.weight.detach().numpy()).reshape(5, 4, 2).max(axis=2).astype(np.float32)
    ref += d.bias.detach().numpy()[None, :]
    assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_dense_wscale_and_invalid():
    d = Dense(4, 6, use_wscale=True, name="dw")
    d.build_weights()
    assert d.wscale == pytest.approx(1.0 / float(4 ** 0.5))  # gain 1.0, fan_in = in_ch
    d.init_weights()

    with pytest.raises(ValueError):
        d.set_weights([torch.zeros((6, 4)), d.bias.detach()])  # transposed shape
    dnob = Dense(4, 6, use_bias=False, name="dnob")
    dnob.build_weights()
    assert len(dnob.get_weights()) == 1


def test_densenorm_no_weights_and_parity():
    dn = DenseNorm(name="dnorm")
    dn.build_weights()
    assert len(dn.get_weights()) == 0  # no checkpoint variables (official)
    x = torch.from_numpy(_seq(24, 2.0, 3.0).reshape(4, 6))
    y = dn(x).detach()
    ref = (x.numpy() / np.sqrt(np.mean(x.numpy() ** 2, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
    # parity: EXACT-order ops, tiny float32 tolerance
    assert np.allclose(y.numpy(), ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# BatchNorm2D / InstanceNorm2D / FRNorm2D
# ---------------------------------------------------------------------------

def test_batchnorm2d_build_names_and_parity():
    b = BatchNorm2D(3, name="bn")
    b.build_weights(); b.init_weights()
    # official order: weight, bias, running_mean, running_var
    assert [k for k, _ in b._iter_official_weights()] == [
        "weight:0", "bias:0", "running_mean:0", "running_var:0"
    ]
    assert bool((b.weight.detach().numpy() == 1).all())
    assert bool((b.bias.detach().numpy() == 0).all())
    # official: running_mean AND running_var start at zeros
    assert bool((b.running_mean == 0).all())
    assert bool((b.running_var == 0).all())

    x = torch.from_numpy(_seq(48, 0.5, 0.3).reshape(1, 3, 4, 4))
    y = b(x).detach()
    # inference-only: (x - rm) / sqrt(rv + eps) * w + b
    ref = (x.numpy() / np.sqrt(0.0 + 1e-5) * 1.0).astype(np.float32)
    # parity: EXACT formula; the 1/sqrt(1e-5) gain amplifies float32
    # rounding, so use a relative/absolute tolerance
    assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)
    # running stats are NOT updated by the forward (official: not for training)
    assert bool((b.running_mean == 0).all())


def test_batchnorm2d_roundtrip_including_running_stats(plain_tmp):
    b = BatchNorm2D(3, name="bn")
    b.build_weights(); b.init_weights()
    # simulate trained statistics
    with torch.no_grad():
        b.running_mean.copy_(torch.full((3,), 0.5))
        b.running_var.copy_(torch.full((3,), 2.0))
    path = os.path.join(plain_tmp, "bn.npy")
    b.save_weights(path)
    d = pickle.loads(open(path, "rb").read())
    assert set(d) == {"weight:0", "bias:0", "running_mean:0", "running_var:0"}

    b2 = BatchNorm2D(3, name="bn")
    b2.build_weights(); b2.init_weights()
    assert b2.load_weights(path) is True
    assert torch.equal(b2.weight, b.weight)
    assert torch.equal(b2.running_mean, b.running_mean)
    assert torch.equal(b2.running_var, b.running_var)


def test_instancenorm2d_build_init_and_parity():
    i = InstanceNorm2D(2, name="in")
    i.build_weights(); i.init_weights()
    assert tuple(i.weight.shape) == (2,)
    assert bool((i.bias.detach().numpy() == 0).all())
    # TF 1-D glorot rule: |w| <= sqrt(6/n)
    limit = (6.0 / 2.0) ** 0.5
    assert bool(np.all(np.abs(i.weight.detach().numpy()) <= limit + 1e-6))

    x = torch.from_numpy(_seq(50, 1.3, 0.2).reshape(1, 2, 5, 5))
    y = i(x).detach()
    xni = x.numpy()
    m = xni.mean(axis=(2, 3), keepdims=True)
    s = np.sqrt(xni.var(axis=(2, 3), ddof=0, keepdims=True)) + 1e-5
    ref = ((xni - m) / s * i.weight.detach().numpy().reshape(1, 2, 1, 1)
           + i.bias.detach().numpy().reshape(1, 2, 1, 1)).astype(np.float32)
    assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_frnorn2d_build_eps_and_parity():
    f = FRNorm2D(3, name="frn")
    f.build_weights(); f.init_weights()
    # official: weight ones, bias zeros, eps a (1,) variable at 1e-6
    assert bool((f.weight.detach().numpy() == 1).all())
    assert bool((f.bias.detach().numpy() == 0).all())
    assert tuple(f.eps.shape) == (1,)
    # eps keeps its construction-time value (no initializer registered)
    assert bool((f.eps.detach().numpy() == 1e-6).all())

    x = torch.from_numpy(_seq(48, 0.7, 0.1).reshape(1, 3, 4, 4))
    y = f(x).detach()
    xnf = x.numpy()
    nu2 = np.mean(xnf ** 2, axis=(2, 3), keepdims=True)
    ref = (xnf * (1.0 / np.sqrt(nu2 + np.abs(f.eps.detach().numpy())))
           * f.weight.detach().numpy().reshape(1, 3, 1, 1)
           + f.bias.detach().numpy().reshape(1, 3, 1, 1)).astype(np.float32)
    assert np.allclose(y.numpy(), ref, rtol=1e-5, atol=1e-6)


def test_norm_layers_roundtrips(plain_tmp):
    for layer, name in (
        (InstanceNorm2D(2, name="in"), "in.npy"),
        (FRNorm2D(3, name="frn"), "frn.npy"),
    ):
        layer.build_weights(); layer.init_weights()
        path = os.path.join(plain_tmp, name)
        layer.save_weights(path)
        fresh = type(layer)(**{
            "in_ch": layer.in_ch, "name": layer.name,
        } if isinstance(layer, InstanceNorm2D) else {
            "in_ch": layer.in_ch, "name": layer.name,
        })
        fresh.build_weights(); fresh.init_weights()
        assert fresh.load_weights(path) is True
        vals = {k: p.detach() for k, p in layer._iter_official_weights()}
        vals2 = {k: p.detach() for k, p in fresh._iter_official_weights()}
        for k in vals:
            assert torch.equal(vals[k], vals2[k]), k


# ---------------------------------------------------------------------------
# BlurPool / AdaIN / TLU / ScaleAdd
# ---------------------------------------------------------------------------

def test_blurpool_parity_and_empty_checkpoint(plain_tmp):
    for fs, (p0, p1) in ((2, (0, 1)), (3, (1, 1)), (4, (1, 2))):
        bp = BlurPool(filt_size=fs, stride=2, name=f"bp{fs}")
        bp.build_weights()
        assert len(list(bp.parameters())) == 0
        assert len(list(bp.buffers())) == 0  # no buffers: checkpoint-pure

        x = torch.from_numpy(_seq(3 * 7 * 7, 0.3, 0.1).reshape(1, 3, 7, 7))
        y = bp(x).detach()

        k = np.repeat(bp.a.astype(np.float32)[None, None], 3, axis=0)
        xp = np.pad(x.numpy(), ((0, 0), (0, 0), (p0, p1), (p0, p1)), mode="constant")
        oh = (xp.shape[2] - fs) // 2 + 1
        ow = (xp.shape[3] - fs) // 2 + 1
        ref = np.zeros((1, 3, oh, ow), np.float32)
        for c in range(3):
            for h in range(oh):
                for w in range(ow):
                    ref[0, c, h, w] = (xp[0, c, h * 2:h * 2 + fs, w * 2:w * 2 + fs] * k[0, 0]).sum()
        assert tuple(y.shape) == (1, 3, oh, ow)
        assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)

    # official BlurPool checkpoint file is an EMPTY dict
    bp = BlurPool(filt_size=3, stride=2, name="bp3")
    bp.build_weights()
    path = os.path.join(plain_tmp, "bp.npy")
    bp.save_weights(path)
    assert pickle.loads(open(path, "rb").read()) == {}
    bp2 = BlurPool(filt_size=3, stride=2, name="bp3")
    bp2.build_weights()
    assert bp2.load_weights(path) is True


def test_adain_build_and_parity():
    a = AdaIN(3, 4, name="adin")
    a.build_weights(); a.init_weights()
    assert tuple(a.weight1.shape) == (4, 3)  # (mlp_ch, in_ch) - TF matmul layout
    assert tuple(a.bias1.shape) == (3,)
    assert bool((a.bias1.detach().numpy() == 0).all())
    assert bool((a.bias2.detach().numpy() == 0).all())
    # he_normal: |w| bounded well below 5*std (std = sqrt(2/3) ~ 0.82)
    assert bool(np.all(np.abs(a.weight1.detach().numpy()) < 5 * (2.0 / 3.0) ** 0.5))

    x = torch.from_numpy(_seq(75, 0.4, 0.2).reshape(1, 3, 5, 5))
    mlp = torch.from_numpy(_seq(4, 1.1, 0.3).reshape(1, 4))
    y = a((x, mlp)).detach()

    g = (mlp.numpy() @ a.weight1.detach().numpy() + a.bias1.detach().numpy()).astype(np.float32)
    be = (mlp.numpy() @ a.weight2.detach().numpy() + a.bias2.detach().numpy()).astype(np.float32)
    xna = x.numpy()
    m = xna.mean(axis=(2, 3), keepdims=True)
    s = np.sqrt(xna.var(axis=(2, 3), ddof=0, keepdims=True)) + 1e-5
    ref = ((xna - m) / s * g.reshape(1, 3, 1, 1) + be.reshape(1, 3, 1, 1)).astype(np.float32)
    assert np.allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_tlu_and_scaleadd_parity():
    t = TLU(3, name="tlu")
    t.build_weights(); t.init_weights()
    assert bool((t.tau.detach().numpy() == 0).all())
    x = torch.from_numpy(_seq(48, 0.5, -2.0).reshape(1, 3, 4, 4))
    y = t(x).detach()
    ref = np.maximum(x.numpy(), t.tau.detach().numpy().reshape(1, 3, 1, 1)).astype(np.float32)
    assert bool(np.array_equal(y.numpy(), ref))  # parity: EXACT (max is exact)

    sa = ScaleAdd(3, name="sadd")
    sa.build_weights(); sa.init_weights()
    x0 = torch.from_numpy(_seq(48, 1.0, 1.0).reshape(1, 3, 4, 4))
    x1 = torch.from_numpy(_seq(48, 1.0, 7.0).reshape(1, 3, 4, 4))
    # weight zero: x0 + x1*0 == x0 exactly
    assert bool(torch.equal(sa((x0, x1)), x0))
    sa.set_weights([torch.full((3,), 0.5, dtype=torch.float32)])
    y = sa((x0, x1)).detach()
    ref = (x0.numpy() + x1.numpy() * 0.5).astype(np.float32)
    assert bool(np.array_equal(y.numpy(), ref))  # parity: EXACT


def test_misc_layers_roundtrips(plain_tmp):
    for layer, name in ((AdaIN(3, 4, name="adin"), "adin.npy"),
                        (TLU(3, name="tlu"), "tlu.npy"),
                        (ScaleAdd(3, name="sadd"), "sadd.npy")):
        layer.build_weights(); layer.init_weights()
        path = os.path.join(plain_tmp, name)
        layer.save_weights(path)
        if isinstance(layer, AdaIN):
            fresh = AdaIN(layer.in_ch, layer.mlp_ch, name=layer.name)
        elif isinstance(layer, TLU):
            fresh = TLU(layer.in_ch, name=layer.name)
        else:
            fresh = ScaleAdd(layer.ch, name=layer.name)
        fresh.build_weights(); fresh.init_weights()
        assert fresh.load_weights(path) is True
        for k, p in layer._iter_official_weights():
            p2 = {k2: p2 for k2, p2 in fresh._iter_official_weights()}[k]
            assert torch.equal(p.detach(), p2.detach()), (name, k)


# ---------------------------------------------------------------------------
# cross-layer: naming stability, import boundary, CUDA
# ---------------------------------------------------------------------------

def _construct(layer_cls):
    if layer_cls is Conv2D:
        return Conv2D(3, 2, kernel_size=3, padding="SAME", name="x")
    if layer_cls is Conv2DTranspose:
        return Conv2DTranspose(2, 4, kernel_size=3, padding="SAME", name="x")
    if layer_cls is DepthwiseConv2D:
        return DepthwiseConv2D(3, kernel_size=3, padding="SAME", name="x")
    if layer_cls is Dense:
        return Dense(4, 6, name="x")
    if layer_cls is DenseNorm:
        return DenseNorm(name="x")
    if layer_cls is BatchNorm2D:
        return BatchNorm2D(3, name="x")
    if layer_cls is InstanceNorm2D:
        return InstanceNorm2D(2, name="x")
    if layer_cls is FRNorm2D:
        return FRNorm2D(3, name="x")
    if layer_cls is BlurPool:
        return BlurPool(filt_size=3, stride=2, name="x")
    if layer_cls is AdaIN:
        return AdaIN(3, 4, name="x")
    if layer_cls is TLU:
        return TLU(3, name="x")
    if layer_cls is ScaleAdd:
        return ScaleAdd(3, name="x")
    raise TypeError(layer_cls)


@pytest.mark.parametrize("layer_cls", ALL_LAYER_CLASSES)
def test_layer_naming_stability_across_constructions(layer_cls):
    a = _construct(layer_cls)
    a.build_weights()
    keys_a = [k for k, _ in a._iter_official_weights()]
    b = _construct(layer_cls)
    b.build_weights()
    keys_b = [k for k, _ in b._iter_official_weights()]
    assert keys_a == keys_b
    assert len(set(keys_a)) == len(keys_a)


def test_migrated_layers_import_without_tensorflow():
    import core.leras.layers  # noqa: F401
    import core.leras.nn  # noqa: F401

    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU(), data_format="NCHW")
    assert "tensorflow" not in sys.modules


def test_no_direct_cuda_in_layer_sources():
    """AST-level: no torch.cuda.* attribute access and no 'cuda:...'
    string literal in the migrated layer sources (docstring prose
    ignored; only code counts)."""
    import ast

    import core.leras
    root = os.path.dirname(core.leras.__file__)
    names = [c.__name__ for c in ALL_LAYER_CLASSES]
    for name in names:
        f = os.path.join(root, "layers", name + ".py")
        with open(f, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                target, parts = node, []
                while isinstance(target, ast.Attribute):
                    parts.append(target.attr)
                    target = target.value
                if isinstance(target, ast.Name) and target.id == "torch" and "cuda" in parts:
                    assert False, f"{f}: direct CUDA API call at line {node.lineno}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith("cuda:"), \
                    f"{f}: hardcoded CUDA device string at line {node.lineno}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_gpu_execution_on_rtx4090():
    """GPU construction + forward through the Phase 2 abstraction
    (no torch.cuda.* in this test body). Parity vs CPU:
    WITHIN_TOLERANCE(1e-4) for float32 CUDA accumulation."""
    # official entry point (idempotent): populate the device registry
    # before any BestGPU() selection, as the official mainscripts do
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), data_format="NCHW")
    assert dfl_nn.device.type == "cuda"
    assert dfl_nn.device == device_layer.get_torch_device(
        dfl_nn.getCurrentDeviceConfig().devices[0]
    )

    c = Conv2D(3, 2, kernel_size=3, padding="SAME", name="c")
    c.build_weights(); c.init_weights()
    assert c.weight.device == dfl_nn.device
    x = torch.from_numpy(_seq(108, 7.0, 1.0).reshape(1, 3, 6, 6)).to(dfl_nn.device)
    y = c(x)
    device_layer.synchronize(dfl_nn.device)
    assert y.device == dfl_nn.device
    assert tuple(y.shape) == (1, 2, 6, 6)

    # CPU reference of the same weights (cross-device float32 parity:
    # relative/absolute 1e-4)
    xc = x.cpu()
    yc = c.cpu()(xc)
    assert torch.allclose(y.cpu(), yc, rtol=1e-4, atol=1e-4)

    bp = BlurPool(filt_size=3, stride=2, name="bp")
    bp.build_weights()
    yb = bp(x)
    device_layer.synchronize(dfl_nn.device)
    assert yb.device == dfl_nn.device

    dn = Dense(4, 6, name="dn")
    dn.build_weights(); dn.init_weights()
    xd = torch.from_numpy(_seq(32, 2.0, 1.0).reshape(8, 4)).to(dfl_nn.device)
    yd = dn(xd)
    device_layer.synchronize(dfl_nn.device)
    assert yd.device == dfl_nn.device
