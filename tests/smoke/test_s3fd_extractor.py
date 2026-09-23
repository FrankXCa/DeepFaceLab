"""Phase 9B acceptance: S3FD face-extractor torch port.

Covers the OFFICIAL-verbatim behavior matrix pinned in
docs/PHASE9_STATE.md (S3FD section) against the tracked official
weight file ``facelib/S3FD.npy`` (65 keys, all float32, no scope
prefix — the official DFL ``save_weights`` format):

- import boundary: no TensorFlow in the interpreter, no tf/cuda in
  the ported source (AST scan; docstring prose may mention ``tf``);
- strict official checkpoint load against the REAL tracked file:
  all 65 keys consumed one-to-one; the three declared layout rules
  pinned on real data (Conv2D HWIO->OIHW kernel permute, 1-D bias
  ``channel_broadcast`` from the official ``(1,1,1,C)`` form, L2Norm
  4-D ``(1,1,1,C)`` gain by IDENTITY — the ``channel_broadcast``
  whitelist must NOT apply to it); ``minus`` stays an unregistered
  plain tensor (official ``tf.constant``), excluded from checkpoint
  enumeration;
- strict-load negatives (all-or-nothing): missing key, extra key,
  kernel shape mismatch, value already in torch layout, L2Norm gain
  shape mismatch (no element-count reshape fallback), dtype
  mismatch, duplicate alias (bare + scope-prefixed key), corrupt
  file. On every failure NOTHING is copied (parameter state
  identity asserted);
- L2Norm formula (upstream L18-32): ``x / (sqrt(sum(x^2, axis=-1,
  keepdims=True)) + 1e-10) * weight`` with the 4-D gain;
- ``_max_pool2``: the official raw ``tf.nn.max_pool(x, [1,2,2,1],
  [1,2,2,1], 'VALID')`` — layout-aware 2x2 stride-2, no padding
  (odd input: the last row/column is dropped, VALID semantics);
- ``cls1`` head: the official SINGLE softmax AFTER the max-out on
  the raw conv outputs (background = max(ch0, ch1, ch2)); the
  USER_LEGACY double-softmax deviation must not reappear (the test
  shows the two conventions differ by ~0.27 on a crafted input);
- ``run()`` output contract: 12 maps, 4-D ``(1,H,W,C)`` numpy
  float32 (the official ``((ocls,), (oreg,))`` refine idiom slices
  batch 0 — the batch dim must stay);
- ``extract()`` preprocessing: BGR->RGB flip, the ``d >= 1280 ->
  640`` cap rule, ``input_scale`` + INTER_LINEAR resize;
- decode math of ``refine`` (strides 2**(i+2), prior centers/sizes,
  the 0.1/0.2 box coefficients) and the NMS ``+1`` IoU convention
  (official L248-269 — edge-touching boxes intersect under the
  official ``+1``; a strict-IoU "fix" would change the result);
- opt-in CPU parity vs the frozen official-TF reference
  (``DFL_TEST_S3FD_TFRE_DIR`` = the git-ignored ``docs/_p9_tfref``
  directory; unset -> skip; the private reference is never read
  from a tracked location): olist maps, scored box lists and the
  final ``extract()`` rects for all 5 reference samples.

The frozen reference was generated with the official TF
extractor (TF 2.21.0, CPU, official checkpoint
``e4b7543``); the measured torch-vs-TF f32 kernel noise on this
machine is <= ~9.1e-06 on olist maps and <= ~3.1e-05 on scored
boxes (deterministic single-threaded CPU), so the parity gate
asserts <= 1e-04.
"""

import ast
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn  # noqa: E402
from core.leras import checkpoint as dfl_ckpt  # noqa: E402
from core.leras import convert as cv  # noqa: E402
from facelib.S3FDExtractor import (  # noqa: E402
    L2Norm, S3FD, S3FDExtractor, _max_pool2,
)

# read-only access to the tracked official weight file (the same
# artifact the production extractor loads)
S3FD_OFFICIAL = cv.read_official_checkpoint(
    str(REPO_ROOT / "facelib" / "S3FD.npy"))

S3FD_SOURCE = REPO_ROOT / "facelib" / "S3FDExtractor.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nn_cpu():
    """CPU / NHWC / float32 device environment for the whole module
    (the official extractor CPU-placement semantics; ``nn.initialize``
    is a cheap module-attribute setup)."""
    nn.initialize(nn.DeviceConfig.CPU(), data_format="NHWC")
    yield


@pytest.fixture(autouse=True)
def _nn_ready(nn_cpu):
    yield


@pytest.fixture(scope="module")
def s3fd(nn_cpu):
    """The production path: S3FDExtractor builds the model, strictly
    loads the official checkpoint and prepares ``run()``."""
    return S3FDExtractor(place_model_on_cpu=True)


@pytest.fixture(scope="module")
def bare_model(nn_cpu):
    """A freshly built (unloaded) S3FD model for strict-load negative
    tests and the head-idiom test. A failed strict load must never
    mutate it, so every negative test can reuse the same instance."""
    m = S3FD()
    m.build()
    m.build_for_run([(nn.floatx, nn.get4Dshape(None, None, 3))])
    return m


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _assert_params_unchanged(model, snap):
    for n, p in model.named_parameters():
        assert n in snap and torch.equal(p.detach(), snap[n]), n


def _fail_load(model, d, match, extra_sub=None):
    """Run the strict load on a mutated dict: it must fail with
    ``match`` (and mention ``extra_sub``) and copy NOTHING."""
    snap = _snapshot(model)
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(model, d, component="S3FD")
    assert match in str(ei.value)
    if extra_sub is not None:
        assert extra_sub in str(ei.value)
    _assert_params_unchanged(model, snap)


# ---------------------------------------------------------------------------
# import boundary / source hygiene
# ---------------------------------------------------------------------------

def test_no_tensorflow_import_boundary():
    # importing the ported extractor (and, through facelib/__init__,
    # the rest of the package) must not pull TensorFlow in
    assert "tensorflow" not in sys.modules
    import facelib.S3FDExtractor  # noqa: F401
    assert "tensorflow" not in sys.modules


def test_no_tf_or_cuda_in_s3fd_sources():
    text = S3FD_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("tensorflow"), (a.name,)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").startswith("tensorflow") is False, \
                node.module
        elif isinstance(node, ast.Attribute):
            src = ast.unparse(node)
            assert not src.startswith("torch.cuda"), src
            assert not src.startswith("nn.tf"), src
    # the module docstring documents the official TF provenance and
    # may MENTION tf as text; the AST checks above cover code
    assert '"cuda:"' not in text and "'cuda:'" not in text
    assert ".cuda(" not in text


# ---------------------------------------------------------------------------
# strict official checkpoint load (positive, against the real file)
# ---------------------------------------------------------------------------

def test_strict_load_official_checkpoint(s3fd):
    model = s3fd.model

    # the tracked official file: 65 keys, all float32
    assert len(S3FD_OFFICIAL) == 65
    assert {np.dtype(v.dtype).name for v in S3FD_OFFICIAL.values()} == \
        {"float32"}

    # one-to-one: every registered parameter is a checkpoint key and
    # vice versa (the all-or-nothing engine requires both directions)
    params = dict(model.named_parameters())
    assert len(params) == len(S3FD_OFFICIAL)
    for p in params.values():
        assert p.device.type == "cpu"
        assert p.dtype == torch.float32

    # re-run the engine on the loaded model: every source key is
    # consumed exactly once (a pass-2 re-copy of identical values)
    report = cv.convert_official_to_torch(model, S3FD_OFFICIAL,
                                          component="S3FD")
    assert {m.source_name for m in report.mapped} == set(S3FD_OFFICIAL)
    assert len(report.mapped) == len(S3FD_OFFICIAL)

    # the three declared layout rules, pinned on REAL data:
    # 1. Conv2D kernel: official HWIO -> torch OIHW (axis permute)
    assert torch.equal(
        params["conv1_1.weight"],
        torch.as_tensor(S3FD_OFFICIAL["conv1_1/weight:0"])
        .permute(3, 2, 0, 1))
    # 2. conv bias: the official (1,1,1,C) singleton-padded form is
    #    squeezed to the 1-D parameter (channel_broadcast whitelist)
    assert torch.equal(
        params["conv1_1.bias"],
        torch.as_tensor(S3FD_OFFICIAL["conv1_1/bias:0"]).reshape(64))
    # 3. L2Norm gain: a genuine 4-D (1,1,1,C) parameter — loaded by
    #    EXACT-SHAPE IDENTITY (the channel_broadcast whitelist does
    #    not apply: param.ndim != 1)
    for attr, c in (("conv3_3_norm", 256),
                    ("conv4_3_norm", 512), ("conv5_3_norm", 512)):
        w = params[attr + ".weight"]
        assert isinstance(w, torch.nn.Parameter)
        assert w.shape == (1, 1, 1, c)
        assert not w.requires_grad
        assert torch.equal(
            w, torch.as_tensor(S3FD_OFFICIAL[f"{attr}/weight:0"]))

    # ``minus`` is the official tf.constant([104,117,123]) — a plain
    # unregistered tensor attribute, never checkpoint state (the
    # run() machinery may own internal buffers; ``minus`` is not one)
    assert not isinstance(model.minus, torch.nn.Parameter)
    assert "minus" not in params
    assert "minus" not in dict(model.named_buffers())
    assert model.minus.dtype == torch.float32
    assert np.array_equal(model.minus.detach().numpy(),
                          np.array([104, 117, 123], dtype=np.float32))
    assert not any("minus" in k for k in S3FD_OFFICIAL)


# ---------------------------------------------------------------------------
# strict load negatives (all-or-nothing: on failure nothing is copied)
# ---------------------------------------------------------------------------

def test_strict_load_missing_key(bare_model):
    d = dict(S3FD_OFFICIAL)
    d.pop("conv1_1/weight:0")
    _fail_load(bare_model, d, "MISSING_REQUIRED_KEY", "conv1_1/weight:0")


def test_strict_load_extra_key(bare_model):
    d = dict(S3FD_OFFICIAL)
    d["no_such_layer/weight:0"] = np.zeros((1, 1, 1, 4), dtype=np.float32)
    _fail_load(bare_model, d, "UNEXPECTED_EXTRA_KEY",
               "no_such_layer/weight:0")


def test_strict_load_shape_mismatch_kernel(bare_model):
    # fc7 is a 1x1 conv: official (1,1,Cin,Cout); a wrong kernel
    # order cannot be permuted into the OIHW parameter shape
    d = dict(S3FD_OFFICIAL)
    d["fc7/weight:0"] = np.zeros((1, 1, 1024, 512), dtype=np.float32)
    _fail_load(bare_model, d, "SHAPE_MISMATCH", "fc7/weight:0")


def test_strict_load_value_already_torch_layout(bare_model):
    # a torch-layout (Cout, Cin, kH, kW) value for a layout-declaring
    # conv layer is not an official-layout file: rejected, not
    # silently accepted (fc7 is 1x1: (Cout=1024, Cin=1024, 1, 1))
    d = dict(S3FD_OFFICIAL)
    d["fc7/weight:0"] = np.zeros((1024, 1024, 1, 1), dtype=np.float32)
    _fail_load(bare_model, d, "INVALID_LAYOUT", "fc7/weight:0")


def test_strict_load_shape_mismatch_l2norm_gain(bare_model):
    # the 4-D L2Norm gain has no declared layout hook: the (1,1,C,1)
    # transposition is rejected — there is no element-count reshape /
    # naive squeeze fallback (it would silently corrupt the gain)
    d = dict(S3FD_OFFICIAL)
    d["conv3_3_norm/weight:0"] = np.zeros((1, 1, 256, 1),
                                          dtype=np.float32)
    _fail_load(bare_model, d, "SHAPE_MISMATCH", "conv3_3_norm/weight:0")


def test_strict_load_dtype_mismatch(bare_model):
    d = dict(S3FD_OFFICIAL)
    # Float-kind checkpoint values follow official target-dtype cast
    # semantics; a cross-kind integer value must still fail.
    d["conv1_1/bias:0"] = np.zeros((1, 1, 1, 64), dtype=np.int32)
    _fail_load(bare_model, d, "DTYPE_MISMATCH", "conv1_1/bias:0")


def test_strict_load_duplicate_alias(bare_model):
    # the same weight offered both bare and scope-prefixed: the engine
    # refuses to choose an alias (both are exact name-derived matches)
    d = dict(S3FD_OFFICIAL)
    d["S3FD/conv1_1/weight:0"] = d["conv1_1/weight:0"].copy()
    _fail_load(bare_model, d, "DUPLICATE_MAPPING", "conv1_1/weight:0")


def test_read_official_checkpoint_corrupt(plain_tmp):
    p = Path(plain_tmp) / "trunc.npy"
    p.write_bytes(b"\x80\x02\x95\x00\x00\x00\x00\x00\x00\x00.")
    with pytest.raises(dfl_ckpt.CheckpointLoadError,
                       match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(str(p))

    p = Path(plain_tmp) / "nondict.npy"
    p.write_bytes(pickle.dumps([1, 2, 3], 4))
    with pytest.raises(dfl_ckpt.CheckpointLoadError,
                       match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(str(p))

    p = Path(plain_tmp) / "badvalue.npy"
    p.write_bytes(pickle.dumps({"a:0": [1.0]}, 4))
    with pytest.raises(dfl_ckpt.CheckpointLoadError,
                       match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(str(p))


# ---------------------------------------------------------------------------
# L2Norm layer: formula + 4-D gain
# ---------------------------------------------------------------------------

def test_l2norm_formula_and_4d_gain():
    layer = L2Norm(4, name="test_l2norm")
    layer.build_weights()
    layer.init_weights()

    # the gain is a genuine 4-D (1,1,1,C) parameter, ones-initialized
    w = layer.weight
    assert isinstance(w, torch.nn.Parameter)
    assert w.shape == (1, 1, 1, 4)
    assert w.dtype == torch.float32
    assert torch.allclose(w, torch.ones(1, 1, 1, 4, dtype=torch.float32))

    # deterministic unique-value input (1..50-ish, exact in f32)
    h = np.arange(2 * 3 * 2 * 4, dtype=np.float32) + 1
    x = h.reshape(2, 3, 2, 4)  # (B, H, W, C)
    gain = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    w.data.copy_(torch.as_tensor(gain).reshape(1, 1, 1, 4))

    out = layer(torch.as_tensor(x))  # torch module __call__ contract
    assert out.shape == torch.Size([2, 3, 2, 4])

    # numpy float64 reference of the official formula
    xn = x.astype(np.float64)
    gain_f64 = gain.astype(np.float64).reshape(1, 1, 1, 4)
    ref = xn / (np.sqrt(np.sum(xn ** 2, axis=-1, keepdims=True)) + 1e-10) \
        * gain_f64
    assert np.allclose(out.numpy().astype(np.float64), ref,
                       rtol=1e-5, atol=1e-6)

    # per-channel scaling: out[..., k] = x[..., k] / norm * gain_k
    norm_t = torch.sqrt((torch.as_tensor(x) ** 2).sum(dim=-1, keepdim=True)) \
        + 1e-10
    expected_t = torch.as_tensor(x) / norm_t \
        * torch.as_tensor(gain).reshape(1, 1, 1, 4)
    assert torch.allclose(out, expected_t, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# _max_pool2: official tf.nn.max_pool(..., 'VALID') semantics
# ---------------------------------------------------------------------------

def _valid_pool2d_reference(x, data_format):
    """Explicit numpy reference of ``tf.nn.max_pool(x, [1,2,2,1],
    [1,2,2,1], 'VALID')``: 2x2 windows, stride 2, no padding — odd
    sizes drop the last row/column (floor((n-2)/2)+1 output)."""
    x = np.asarray(x, dtype=np.float64)
    if data_format == "NHWC":
        b, h, w, c = x.shape
        oh, ow = (h - 2) // 2 + 1, (w - 2) // 2 + 1
        out = np.empty((b, oh, ow, c), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                for ci in range(c):
                    out[:, i, j, ci] = x[:, 2 * i:2 * i + 2,
                                        2 * j:2 * j + 2, ci].max(axis=(1, 2))
        return out
    b, c, h, w = x.shape
    oh, ow = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    out = np.empty((b, c, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            for ci in range(c):
                out[:, ci, i, j] = x[:, ci, 2 * i:2 * i + 2,
                                     2 * j:2 * j + 2].max(axis=(1, 2))
    return out


def test_max_pool2_nhwc_valid_semantics(nn_cpu):
    # odd H (5) and even W (4): VALID must drop the last row only
    assert nn.data_format == "NHWC"
    x = np.zeros((1, 5, 4, 2), dtype=np.float32)  # (B, H, W, C)
    x[0, :, :, 0] = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    x[0, :, :, 1] = np.arange(5 * 4, dtype=np.float32).reshape(5, 4) + 100
    t = torch.as_tensor(x)
    out = _max_pool2(t)
    assert out.shape == torch.Size([1, 2, 2, 2])  # (B, C, OH, OW)
    assert np.array_equal(out.numpy(),
                          _valid_pool2d_reference(x, "NHWC").astype(np.float32))


def test_max_pool2_nchw_passthrough(nn_cpu):
    # NCHW data format: F.max_pool2d consumes the tensor natively
    prev = nn.data_format
    try:
        nn.data_format = "NCHW"
        x = np.zeros((1, 2, 5, 4), dtype=np.float32)  # (B, C, H, W)
        x[:, 0, :, :] = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
        x[:, 1, :, :] = np.arange(5 * 4, dtype=np.float32).reshape(5, 4) + 100
        out = _max_pool2(torch.as_tensor(x))
        assert out.shape == torch.Size([1, 2, 2, 2])
        assert np.array_equal(out.numpy(),
                              _valid_pool2d_reference(x, "NCHW")
                              .astype(np.float32))
    finally:
        nn.data_format = prev


# ---------------------------------------------------------------------------
# cls1 head: official single softmax after max-out
# ---------------------------------------------------------------------------

class _ConstantHead(object):
    """Stand-in for the cls1 mbox conv: returns per-channel constants
    of the input's (1,H,W) spatial shape, so the head's raw output is
    known exactly and the test isolates the bmax/canonical-softmax
    idiom on the model's real forward path."""

    def __init__(self, vals):
        self.vals = vals

    def __call__(self, x):
        parts = [torch.full(x.shape[:-1] + (1,), float(v),
                            dtype=x.dtype, device=x.device)
                 for v in self.vals]
        return torch.cat(parts, dim=-1)


def test_cls1_head_single_softmax_idiom(bare_model):
    # raw cls1 channels [0, 1, 2, 10] -> bmax = max(ch0..ch2) = 2 ->
    # concat(bmax, ch3) = [2, 10] -> ONE softmax (official L135,
    # L153-157). The USER_LEGACY double softmax would give
    # softmax(softmax([2,10])) ~ [0.269, 0.731] instead of
    # [~3.35e-4, ~0.99966] — the assertions below pin the official
    # value and show the legacy deviation is far from it.
    # shadow the registered submodule via the instance dict (torch's
    # attribute lookup finds the dict entry before ``_modules``); the
    # original module object is never touched and the entry is popped
    # in ``finally`` so the model is byte-identical for later tests
    bare_model.__dict__["conv3_3_norm_mbox_conf"] = \
        _ConstantHead([0.0, 1.0, 2.0, 10.0])
    try:
        img = (np.arange(64 * 64 * 3, dtype=np.float32) % 7
               ).reshape(1, 64, 64, 3)
        olist = bare_model.run([img])
        cls1 = np.asarray(olist[0], dtype=np.float32)
        assert cls1.ndim == 4 and cls1.shape[0] == 1 and cls1.shape[-1] == 2

        p = torch.softmax(torch.tensor([2.0, 10.0], dtype=torch.float32),
                          dim=-1)
        want = torch.stack([torch.full(cls1.shape[:-1] + (1,), p[0].item()),
                            torch.full(cls1.shape[:-1] + (1,), p[1].item())],
                           dim=-1).numpy().astype(np.float32)
        assert np.allclose(cls1, want, rtol=0, atol=1e-6), \
            "official single-softmax value not reproduced"

        legacy = torch.softmax(p, dim=-1)
        assert abs(float(cls1[0, 0, 0, 0]) - float(legacy[0])) > 0.1, \
            "output matches neither convention — expected the official " \
            "single softmax far from the USER_LEGACY double softmax"
    finally:
        bare_model.__dict__.pop("conv3_3_norm_mbox_conf", None)


# ---------------------------------------------------------------------------
# run() output contract: 12 maps, 4-D (1,H,W,C) float32
# ---------------------------------------------------------------------------

def test_run_output_4d_contract(s3fd):
    # 64x64 input -> head feature maps
    # [16, 8, 4, 6, 3, 2] (f3_3, f4_3, f5_3, ffc7, f6_2, f7_2):
    # the pooled f5_3 is 2x2 and the official fc6 3-pad/VALID-3x3 grows
    # it to 6x6; the two heads are a SEQUENTIAL chain (official DFL
    # variant: f6_2 = conv6_2(ffc7), f7_2 = conv7_2(conv7_1(f6_2))),
    # giving 3x3 (stride-2 SAME on 6x6) and 2x2 (stride-2 SAME on
    # 3x3). The mbox prior strides stay the official uniform
    # 2**(i+2) (kept verbatim).
    img = (np.arange(64 * 64 * 3, dtype=np.float32) % 5).reshape(1, 64, 64, 3)
    olist = s3fd.model.run([img])
    assert len(olist) == 12
    # note: map 0 is the post-maxout cls1 output (bg, face) — the raw
    # 4-channel conv output never leaves forward()
    want_shapes = [
        (1, 16, 16, 2), (1, 16, 16, 4),
        (1, 8, 8, 2), (1, 8, 8, 4),
        (1, 4, 4, 2), (1, 4, 4, 4),
        (1, 6, 6, 2), (1, 6, 6, 4),
        (1, 3, 3, 2), (1, 3, 3, 4),
        (1, 2, 2, 2), (1, 2, 2, 4),
    ]
    for i, (t, want) in enumerate(zip(olist, want_shapes)):
        assert t.dtype == np.float32, i
        assert t.ndim == 4 and t.shape == want, (i, t.shape)


# ---------------------------------------------------------------------------
# extract() preprocessing (resize replica via a stubbed run)
# ---------------------------------------------------------------------------

def test_extract_preprocessing_rules(s3fd, monkeypatch):
    captured = {}

    def fake_run(inputs):
        captured["img"] = inputs[0]
        return []

    monkeypatch.setattr(s3fd.model, "run", fake_run)

    # 300x200 BGR float32: S[y,x,c] = c*100 + x//20 (exact 2x-scale
    # values: the pattern is constant over even 20-px columns)
    xs = np.arange(200, dtype=np.float32) // 20
    src = np.ascontiguousarray(
        np.broadcast_to(xs[None, :, None]
                        + 100 * np.arange(3, dtype=np.float32)[None, None, :],
                        (300, 200, 3)))

    # d = 300 < 1280 -> scale_to = 150, input_scale = 2.0
    # -> (150, 100) resize; is_bgr=True flips the channel order first
    assert s3fd.extract(src, is_bgr=True) == []
    fed = np.asarray(captured["img"], dtype=np.float32)
    assert fed.shape == (1, 150, 100, 3)
    want_row = (np.arange(100, dtype=np.float32)[:, None] // 10
                + 100 * (2 - np.arange(3, dtype=np.float32))[None, :])
    want = np.ascontiguousarray(
        np.broadcast_to(want_row.reshape(1, 100, 3), (150, 100, 3)))
    assert np.allclose(fed[0], want, rtol=0, atol=1e-3), \
        "BGR flip / scale / INTER_LINEAR resize mismatch"

    # is_bgr=False: no channel flip
    captured.clear()
    assert s3fd.extract(src, is_bgr=False) == []
    fed = np.asarray(captured["img"], dtype=np.float32)
    assert fed.shape == (1, 150, 100, 3)
    want_row = (np.arange(100, dtype=np.float32)[:, None] // 10
                + 100 * np.arange(3, dtype=np.float32)[None, :])
    want = np.ascontiguousarray(
        np.broadcast_to(want_row.reshape(1, 100, 3), (150, 100, 3)))
    assert np.allclose(fed[0], want, rtol=0, atol=1e-3)

    # d = 2000 >= 1280 -> 640 cap: input_scale = 3.125
    # -> (new_w, new_h) = (640, 320)
    big = np.zeros((1000, 2000, 3), dtype=np.float32)
    captured.clear()
    assert s3fd.extract(big, is_bgr=True) == []
    fed = np.asarray(captured["img"], dtype=np.float32)
    assert fed.shape == (1, 320, 640, 3)


# ---------------------------------------------------------------------------
# full pipeline on uint8 (official feed dtype) + context manager
# ---------------------------------------------------------------------------

def test_extract_uint8_pipeline(s3fd):
    y, x = np.indices((96, 96))
    img = (((y + x)[..., None] + 30 * np.arange(3)[None, None, :]) % 256
           ).astype(np.uint8)  # BGR, d = 96 -> 48x48 network input
    with s3fd as ext:
        rects = ext.extract(img, is_bgr=True)
    assert isinstance(rects, list)
    for r in rects:
        assert len(r) == 4 and all(type(v) is int for v in r)


# ---------------------------------------------------------------------------
# refine(): decode math (official formula) + thresholding
# ---------------------------------------------------------------------------

def test_refine_decode_math(s3fd):
    # head 0 (stride 4): one candidate at pixel (0,0), score 0.9,
    # zero loc -> prior [2, 2, 16, 16] -> box [-6, -6, 10, 10]
    # head 1 (stride 8): one candidate, score 0.8, loc [5,0,0,0]
    # -> center [20, 4], size [32, 32] -> box [4, -12, 36, 20]
    cls1 = np.zeros((1, 1, 1, 2), dtype=np.float32)
    cls1[0, 0, 0, 1] = 0.9
    reg1 = np.zeros((1, 1, 1, 4), dtype=np.float32)
    cls2 = np.zeros((1, 1, 1, 2), dtype=np.float32)
    cls2[0, 0, 0, 1] = 0.8
    reg2 = np.zeros((1, 1, 1, 4), dtype=np.float32)
    reg2[0, 0, 0, 0] = 5.0

    # the two boxes are far enough apart (official +1 IoU ~ 0.094
    # < 0.3) that NMS keeps both
    got = s3fd.refine([cls1, reg1, cls2, reg2])
    assert [list(map(int, b)) for b in got] == [[-6, -6, 10, 10],
                                                 [4, -12, 36, 20]]

    # nothing above the 0.05 candidacy threshold -> no boxes (the
    # official empty-input sentinel row scores 0 < 0.5 -> filtered)
    cls_empty = np.zeros((1, 1, 1, 2), dtype=np.float32)
    cls_empty[0, 0, 0, 1] = 0.04
    reg_empty = np.zeros((1, 1, 1, 4), dtype=np.float32)
    assert s3fd.refine([cls_empty, reg_empty]) == []


def test_refine_nms_plus_one_convention(s3fd):
    # the official NMS (L248-269) uses the inclusive ``+1``: touching
    # boxes intersect. det A [0,0,1,1] and B [1,0,2,1]: width =
    # min(1,2)-max(0,1)+1 = 1, height = 2 -> IoU = 2/6 ~ 0.333 > 0.3
    # -> B is suppressed (a strict-IoU "fix" would give IoU 0 and keep
    # B — this test pins the official convention)
    dets = np.array([[0, 0, 1, 1, 0.9],
                     [1, 0, 2, 1, 0.8]], dtype=np.float32)
    assert list(s3fd.refine_nms(dets, 0.3)) == [0]

    # identical boxes: IoU 1 -> the lower-scored one is suppressed
    dets = np.array([[0, 0, 3, 3, 0.9],
                     [0, 0, 3, 3, 0.8]], dtype=np.float32)
    assert list(s3fd.refine_nms(dets, 0.3)) == [0]

    # disjoint boxes: both kept
    dets = np.array([[0, 0, 3, 3, 0.9],
                     [10, 10, 13, 13, 0.8]], dtype=np.float32)
    assert list(s3fd.refine_nms(dets, 0.3)) == [0, 1]


# ---------------------------------------------------------------------------
# opt-in CPU parity vs the frozen official-TF reference
# ---------------------------------------------------------------------------

def _scored_refine(self, olist):
    """Test-side replica of the official ``refine`` that KEEPS the
    confidence score the official drops (the frozen reference stores
    the scored rows). The decode lines are the official S3FD formula
    (strides 2**(i+2), prior centers ``(w*s + s/2, h*s + s/2)``,
    sizes ``4s``, the 0.1/0.2 box coefficients, the official
    ``((ocls,), (oreg,))`` batch-0 slicing idiom); ``refine_nms`` is
    the ported instance's own verbatim-official method."""
    bboxlist = []
    for i, ((ocls,), (oreg,)) in enumerate(zip(olist[::2], olist[1::2])):
        stride = 2 ** (i + 2)
        s_d2, s_m4 = stride / 2, stride * 4
        for hindex, windex in zip(*np.where(ocls[..., 1] > 0.05)):
            score = ocls[hindex, windex, 1]
            loc = oreg[hindex, windex, :]
            priors = np.array([windex * stride + s_d2,
                               hindex * stride + s_d2, s_m4, s_m4])
            box = np.concatenate(
                (priors[:2] + loc[:2] * 0.1 * priors[2:],
                 priors[2:] * np.exp(loc[2:] * 0.2)))
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            bboxlist.append([*box, score])
    bboxlist = np.array(bboxlist)
    if len(bboxlist) == 0:
        bboxlist = np.zeros((1, 5))
    bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
    return [[float(v) for v in row] for row in bboxlist]


def test_frozen_tf_reference_parity(s3fd):
    """CPU parity against the frozen official-TF reference run
    (opt-in: ``DFL_TEST_S3FD_TFRE_DIR`` = the git-ignored
    ``docs/_p9_tfref`` directory; unset -> skip). Compares, per
    sample: the 12 olist maps (<= 1e-04, the measured f32
    TF-vs-torch kernel noise cap), the scored post-NMS box lists and
    the final ``extract()`` rects (exact). The reference artifacts
    and their sample images are private (git-ignored); this test
    reads them only from the directory the environment variable
    points at — never a tracked path."""
    tfre_root = os.environ.get("DFL_TEST_S3FD_TFRE_DIR")
    if tfre_root is None:
        pytest.skip("DFL_TEST_S3FD_TFRE_DIR not set")
    tfre_root = Path(tfre_root)
    frozen = tfre_root / "frozen"
    repo = tfre_root.parent.parent
    meta = json.loads((frozen / "meta.json").read_text(encoding="utf-8"))

    # the frozen reference is pinned to the tracked weight file
    md5 = hashlib.md5(
        (REPO_ROOT / "facelib" / "S3FD.npy").read_bytes()).hexdigest()
    assert md5 == meta["weight_md5"]["S3FD.npy"]

    for sample in meta["samples"]:
        stem = sample["stem"]
        ref = json.loads(
            (frozen / f"{stem}.json").read_text(encoding="utf-8"))

        img_path = repo / sample["repo_path"]
        assert hashlib.md5(img_path.read_bytes()).hexdigest() == \
            sample["md5"], f"{stem}: sample image changed"
        img = cv2.imread(str(img_path))
        assert img is not None, img_path

        # resize replica (official extract() rules, L308-315)
        h, w = img.shape[:2]
        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)
        input_scale = d / scale_to
        new_w, new_h = int(w / input_scale), int(h / input_scale)
        fr = ref["s3fd"]["resize"]
        assert (new_w, new_h) == (fr["new_w"], fr["new_h"]), stem
        assert abs(input_scale - fr["input_scale"]) < 1e-9, stem

        rgb = img[:, :, ::-1]  # official BGR->RGB feed
        resized = cv2.resize(rgb, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # 1) olist maps: shapes exact, values within the f32 noise cap
        olist = s3fd.model.run([resized[None, ...]])
        ref_olist = [np.asarray(m, dtype=np.float32)
                     for m in ref["s3fd"]["olist"]]
        assert len(olist) == len(ref_olist) == 12, stem
        worst = 0.0
        for i, (got, want) in enumerate(zip(olist, ref_olist)):
            assert got.dtype == np.float32 and got.shape == want.shape, \
                (stem, i, got.shape)
            worst = max(worst, float(np.abs(got - want).max()))
        assert worst <= 1e-4, f"{stem}: olist worst |d| = {worst:.3e}"

        # 2) scored post-NMS boxes (frozen keeps the score)
        scored = _scored_refine(s3fd, olist)
        ref_scored = ref["s3fd"]["bbox_with_scores"]
        assert len(scored) == len(ref_scored), stem
        worst = max((max(abs(a - b) for a, b in zip(row, rrow))
                     for row, rrow in zip(scored, ref_scored)),
                    default=0.0)
        assert worst <= 1e-4, f"{stem}: scored-box worst |d| = {worst:.3e}"

        # 3) final extract() rects (40px filter, chin +10%, int cast,
        #    area-desc sort) — exact
        got_rects = [list(map(int, r)) for r in s3fd.extract(img)]
        fr_rects = ref["s3fd"]["rects"]
        if fr_rects and isinstance(fr_rects[0], (int, float)):
            fr_rects = [fr_rects]
        want_rects = [list(map(int, r)) for r in fr_rects]
        assert got_rects == want_rects, f"{stem}: {got_rects} vs {want_rects}"
