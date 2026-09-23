"""Phase 9C acceptance: FAN (2D/3D) landmark-extractor torch port.

Covers the OFFICIAL-verbatim behavior matrix pinned in
docs/PHASE9_STATE.md (FAN section) against the tracked official
weight files ``facelib/2DFAN.npy`` / ``facelib/3DFAN.npy`` (945 keys
each, all float32, no scope prefix — the official DFL
``save_weights`` format):

- import boundary: no TensorFlow in the interpreter, no tf/cuda in
  the ported source (AST scan; docstring prose may mention ``tf``);
- strict official checkpoint load against the REAL tracked files:
  all 945 keys consumed one-to-one (577 parameters + 368 buffers);
  the two official 1-D state forms pinned on real data — 2DFAN's
  NHWC ``(1,1,1,C)`` singleton-padded form (channel_broadcast
  squeeze to the 1-D parameter) and 3DFAN's plain ``(C,)`` form
  (exact identity, the official TF variable shape); Conv2D kernels
  via the HWIO->OIHW layout hook, including the nested hourglass
  scopes (``m_0/b1/conv1/weight:0`` -> ``m_0.b1.conv1.weight``);
- strict-load negatives (all-or-nothing): missing key, extra key,
  kernel shape mismatch, value already in torch layout, 1-D state
  shape OUTSIDE the channel_broadcast whitelist (no element-count
  reshape / naive squeeze fallback), dtype mismatch, duplicate
  alias (bare + ``FAN/``-prefixed key). On every failure NOTHING is
  copied (parameter AND buffer state identity asserted);
- ``upsample2d`` (migrated into ``core/leras/ops`` this phase):
  nearest-neighbor 2x = the official
  ``tf.image.resize_nearest_neighbor`` — exact
  ``out[i, j] = in[i // 2, j // 2]`` in both data formats;
- ``_avg_pool2``: the official raw ``tf.nn.avg_pool(x, [1,2,2,1],
  [1,2,2,1], 'VALID')`` — layout-aware 2x2 stride-2, no padding
  (odd input: the last row/column is dropped, VALID semantics);
- module topology: the official list-valued attributes (m, top_m,
  conv_last, bn_end, l, bl, al) keep their elements registered as
  torch modules under the ``m_0..m_3`` ... names (the documented
  ``FAN.build`` torch adaptation), so the checkpoint engine's
  ``named_parameters`` tree reproduces the official dotted scope
  keys; the recursive HourGlass(256, 4) nesting (depth-4 chain of
  ``b2_plus`` ending in a ConvBlock leaf) is asserted;
- ``run()`` output contract: a BARE 4-D ``(1,68,64,64)`` NCHW
  float32 array (single-output model — the official
  ``tf_sess.run``-of-one-tensor contract; the official
  ``self.model.run([img[None, ...]])[0]`` batch-0 idiom and the 3-D
  ``get_pts_from_predict`` work unchanged);
- ``extract()``: BGR flip equivalence (is_bgr=True on BGR ==
  is_bgr=False on RGB, bit-identical feeds — the flip precedes all
  channel-independent crop math), the 256x256x3 float32 in [0,1]
  feed rule, determinism, the empty-rects early return, and a full
  uint8 pipeline run producing a non-null (68, 2) landmark set;
- the official ``transform`` affine (hand-computed: crop (32,32) ->
  image (0,0) for center [0,0] / scale 1 / resolution 64) and the
  ``get_pts_from_predict`` decode (argmax over the flat 64x64 map,
  the ``sign(diff)*0.25`` subpixel correction inside the
  ``0 < p < 63`` window, ``c += 0.5``, the corner-peak no-correction
  case) pinned on crafted heatmaps;
- opt-in CPU parity vs the frozen official-TF reference
  (``DFL_TEST_FAN_TFRE_DIR`` = the git-ignored ``docs/_p9_tfref``
  directory; unset -> skip; the private reference is never read
  from a tracked location): per-rect 68 landmarks of both the 2D
  and 3D FAN on the 4 real samples (the FAN rects are the frozen
  S3FD final rects) at <= 1e-04, plus — for the synthetic sample —
  the full 68x64x64 prediction map at <= 1e-04.

The frozen reference was generated with the official TF extractor
(TF 2.21.0, CPU, official checkpoint ``e4b7543``). Measured
torch-vs-TF on this machine (deterministic single-threaded CPU):
prediction map (synthetic sample) worst |d| = 4.3e-07 and the
per-rect landmarks of both FAN variants EXACT (0.0) on the 4 real
samples — the hard argmax decode is tie-free on face heatmaps, so
the integer landmark sets are bit-identical — hence the parity gate
asserts <= 1e-04. The synthetic
sample's landmark values are NOT compared: its forced rect crops a
seeded-noise image, whose heatmaps are flat enough that the hard
argmax decode ties between TF and torch (a legitimate kernel
difference, not a model discrepancy) — the prediction map is the
continuous parity gate for that sample.
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
from facelib.FANExtractor import FANExtractor, _avg_pool2  # noqa: E402

# read-only access to the tracked official weight files (the same
# artifacts the production extractor loads)
FAN2D_OFFICIAL = cv.read_official_checkpoint(
    str(REPO_ROOT / "facelib" / "2DFAN.npy"))
FAN3D_OFFICIAL = cv.read_official_checkpoint(
    str(REPO_ROOT / "facelib" / "3DFAN.npy"))

FAN_SOURCE = REPO_ROOT / "facelib" / "FANExtractor.py"


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
def fan2d(nn_cpu):
    """The production path: FANExtractor builds the model, strictly
    loads the official 2DFAN checkpoint and prepares ``run()``."""
    return FANExtractor(landmarks_3D=False, place_model_on_cpu=True)


@pytest.fixture(scope="module")
def fan3d(nn_cpu):
    """Same, for the 3D variant (plain 1-D ``(C,)`` official state
    form). The constructor's second ``nn.initialize`` (forced CPU) is
    idempotent with the fixture's — the official pipeline applies the
    SAME device policy to both extractors on a worker."""
    return FANExtractor(landmarks_3D=True, place_model_on_cpu=True)


def _snapshot(model):
    """All parameter AND buffer state (the all-or-nothing engine must
    copy neither on failure)."""
    return ({n: p.detach().clone() for n, p in model.named_parameters()},
            {n: b.detach().clone() for n, b in model.named_buffers()})


def _assert_unchanged(model, snap):
    ps, bs = snap
    for n, p in model.named_parameters():
        assert n in ps and torch.equal(p.detach(), ps[n]), n
    for n, b in model.named_buffers():
        assert n in bs and torch.equal(b.detach(), bs[n]), n


def _fail_load(model, d, match, extra_sub=None):
    """Run the strict load on a mutated dict against the LOADED
    model: it must fail with ``match`` (and mention ``extra_sub``)
    and copy NOTHING (two-pass validation)."""
    snap = _snapshot(model)
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(model, d, component="FAN")
    assert match in str(ei.value)
    if extra_sub is not None:
        assert extra_sub in str(ei.value)
    _assert_unchanged(model, snap)


def _ri2(a, axes):
    """nearest 2x = exact pixel replication on the given axes."""
    for ax in axes:
        a = np.repeat(a, 2, axis=ax)
    return a


def _avg_pool2d_reference(x, data_format):
    """NumPy 2x2 stride-2 VALID mean pool, layout-aware (the official
    ``tf.nn.avg_pool`` honors ``data_format``); output in the input
    layout. Small-integer patterns keep both the NumPy and the torch
    float32 means exact."""
    if data_format == "NCHW":
        x = np.moveaxis(x, 1, -1)
    b, h, w, c = x.shape
    oh, ow = h // 2, w // 2
    out = np.empty((b, oh, ow, c), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            win = x[:, 2*i:2*i+2, 2*j:2*j+2, :].reshape(b, 2, 2, c)
            out[:, i, j, :] = win.mean(axis=(1, 2))
    if data_format == "NCHW":
        out = np.moveaxis(out, -1, 1)
    return out


# ---------------------------------------------------------------------------
# import boundary / source hygiene
# ---------------------------------------------------------------------------

def test_no_tensorflow_import_boundary():
    # importing the ported extractor (and, through facelib/__init__,
    # the rest of the package) must not pull TensorFlow in
    assert "tensorflow" not in sys.modules
    import facelib.FANExtractor  # noqa: F401
    assert "tensorflow" not in sys.modules


def test_no_tf_or_cuda_in_fan_sources():
    text = FAN_SOURCE.read_text(encoding="utf-8")
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
# strict official checkpoint load (positive, against the real files)
# ---------------------------------------------------------------------------

def test_strict_load_2d_official_checkpoint(fan2d):
    model = fan2d.model

    # the tracked official file: 945 keys, all float32
    assert len(FAN2D_OFFICIAL) == 945
    assert {np.dtype(v.dtype).name for v in FAN2D_OFFICIAL.values()} == \
        {"float32"}

    # one-to-one: 577 parameters + 368 buffers = every checkpoint key
    # (conv kernels/biases + BN weight/bias as parameters; BN
    # running_mean/running_var as buffers, per the Saveable contract)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert len(params) + len(buffers) == 945
    for p in params.values():
        assert p.device.type == "cpu"
        assert p.dtype == torch.float32
    for b in buffers.values():
        assert b.device.type == "cpu"
        assert b.dtype == torch.float32

    # re-run the engine on the loaded model: every source key is
    # consumed exactly once (a pass-2 re-copy of identical values)
    report = cv.convert_official_to_torch(model, FAN2D_OFFICIAL,
                                          component="FAN")
    assert {m.source_name for m in report.mapped} == set(FAN2D_OFFICIAL)
    assert len(report.mapped) == 945

    # the declared layout rules, pinned on REAL data. The 2D file's
    # 1-D states are the NHWC (1,1,1,C) singleton-padded form:
    # 1. Conv2D kernel: official HWIO -> torch OIHW (axis permute)
    assert params["conv1.weight"].shape == torch.Size([64, 3, 7, 7])
    assert torch.equal(
        params["conv1.weight"],
        torch.as_tensor(FAN2D_OFFICIAL["conv1/weight:0"])
        .permute(3, 2, 0, 1))
    # 2. conv bias / BN states: (1,1,1,C) squeezed to the 1-D
    #    parameter (channel_broadcast whitelist)
    assert torch.equal(
        params["conv1.bias"],
        torch.as_tensor(FAN2D_OFFICIAL["conv1/bias:0"]).reshape(64))
    # nested hourglass scope: torch m_0.b1.bn1.weight == official
    # m_0/b1/bn1/weight:0
    assert torch.equal(
        params["m_0.b1.bn1.weight"],
        torch.as_tensor(FAN2D_OFFICIAL["m_0/b1/bn1/weight:0"]).reshape(256))
    assert torch.equal(
        params["m_0.b1.conv1.weight"],
        torch.as_tensor(FAN2D_OFFICIAL["m_0/b1/conv1/weight:0"])
        .permute(3, 2, 0, 1))
    assert params["m_0.b1.conv1.weight"].shape == torch.Size([128, 256, 3, 3])
    # BN running states live in the BUFFER registry (1-D (256,))
    assert buffers["m_0.b1.bn1.running_mean"].shape == torch.Size([256])
    assert torch.equal(
        buffers["m_0.b1.bn1.running_mean"],
        torch.as_tensor(FAN2D_OFFICIAL["m_0/b1/bn1/running_mean:0"])
        .reshape(256))


def test_strict_load_3d_official_checkpoint(fan3d):
    model = fan3d.model

    assert len(FAN3D_OFFICIAL) == 945
    assert {np.dtype(v.dtype).name for v in FAN3D_OFFICIAL.values()} == \
        {"float32"}

    # the 3D file stores the 1-D states as PLAIN (C,) — the official
    # TF variable shape (exact-identity rule, no broadcast involved)
    assert FAN3D_OFFICIAL["conv1/bias:0"].shape == (64,)
    assert FAN3D_OFFICIAL["m_0/b1/bn1/weight:0"].shape == (256,)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert len(params) + len(buffers) == 945
    for p in params.values():
        assert p.device.type == "cpu"
        assert p.dtype == torch.float32
    for b in buffers.values():
        assert b.device.type == "cpu"
        assert b.dtype == torch.float32

    report = cv.convert_official_to_torch(model, FAN3D_OFFICIAL,
                                          component="FAN")
    assert {m.source_name for m in report.mapped} == set(FAN3D_OFFICIAL)
    assert len(report.mapped) == 945

    assert params["conv1.bias"].shape == torch.Size([64])
    assert torch.equal(
        params["conv1.bias"],
        torch.as_tensor(FAN3D_OFFICIAL["conv1/bias:0"]))
    assert torch.equal(
        params["m_0.b1.bn1.weight"],
        torch.as_tensor(FAN3D_OFFICIAL["m_0/b1/bn1/weight:0"]))


# ---------------------------------------------------------------------------
# strict load negatives (all-or-nothing: on failure nothing is copied)
# ---------------------------------------------------------------------------

def test_strict_load_missing_key(fan2d):
    d = dict(FAN2D_OFFICIAL)
    d.pop("conv1/weight:0")
    _fail_load(fan2d.model, d, "MISSING_REQUIRED_KEY", "conv1/weight:0")


def test_strict_load_extra_key(fan2d):
    d = dict(FAN2D_OFFICIAL)
    d["no_such_layer/weight:0"] = np.zeros((1, 1, 1, 4), dtype=np.float32)
    _fail_load(fan2d.model, d, "UNEXPECTED_EXTRA_KEY",
               "no_such_layer/weight:0")


def test_strict_load_shape_mismatch_kernel(fan2d):
    # the hourglass conv is 3x3: a 1x1 official kernel cannot be
    # permuted into the (128, 256, 3, 3) parameter shape
    d = dict(FAN2D_OFFICIAL)
    d["m_0/b1/conv1/weight:0"] = np.zeros((1, 1, 256, 128),
                                          dtype=np.float32)
    _fail_load(fan2d.model, d, "SHAPE_MISMATCH", "m_0/b1/conv1/weight:0")


def test_strict_load_value_already_torch_layout(fan2d):
    # a torch-layout (Cout, Cin, kH, kW) value for a layout-declaring
    # conv layer is not an official-layout file: rejected, not
    # silently accepted (conv_last_0 is 1x1: (256, 256, 1, 1))
    d = dict(FAN2D_OFFICIAL)
    d["conv_last_0/weight:0"] = np.zeros((256, 256, 1, 1), dtype=np.float32)
    _fail_load(fan2d.model, d, "INVALID_LAYOUT", "conv_last_0/weight:0")


def test_strict_load_1d_state_outside_whitelist(fan2d):
    # a 1-D state shape outside the channel_broadcast whitelist
    # ((C,), (1,1,1,C), (1,C,1,1)) is rejected — there is no
    # element-count reshape / naive squeeze fallback
    d = dict(FAN2D_OFFICIAL)
    d["bn1/running_mean:0"] = np.zeros((1, 1, 64, 1), dtype=np.float32)
    _fail_load(fan2d.model, d, "SHAPE_MISMATCH", "bn1/running_mean:0")


def test_strict_load_dtype_mismatch(fan2d):
    d = dict(FAN2D_OFFICIAL)
    # Float-kind checkpoint values follow official target-dtype cast
    # semantics; a cross-kind integer value must still fail.
    d["bn1/bias:0"] = np.zeros((1, 1, 1, 64), dtype=np.int32)
    _fail_load(fan2d.model, d, "DTYPE_MISMATCH", "bn1/bias:0")


def test_strict_load_duplicate_alias(fan2d):
    # the same weight offered both bare and scope-prefixed: the engine
    # refuses to choose an alias (both are exact name-derived matches)
    d = dict(FAN2D_OFFICIAL)
    d["FAN/conv1/weight:0"] = d["conv1/weight:0"].copy()
    _fail_load(fan2d.model, d, "DUPLICATE_MAPPING", "conv1/weight:0")


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
    p.write_bytes(pickle.dumps({"k:0": [1, 2, 3]}, 4))
    with pytest.raises(dfl_ckpt.CheckpointLoadError,
                       match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(str(p))


# ---------------------------------------------------------------------------
# upsample2d (migrated op): nearest 2x, both data formats
# ---------------------------------------------------------------------------

def test_upsample2d_nearest_semantics_nhwc(nn_cpu):
    # odd H (3) and even W (4): nearest 2x replicates each pixel;
    # out[i, j] = in[i // 2, j // 2] (top-left aligned)
    assert nn.data_format == "NHWC"
    x = np.zeros((1, 3, 4, 2), dtype=np.float32)  # (B, H, W, C)
    x[0, :, :, 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    x[0, :, :, 1] = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
    out = nn.upsample2d(torch.as_tensor(x))
    assert out.shape == torch.Size([1, 6, 8, 2])
    assert np.array_equal(out.numpy(), _ri2(x, [1, 2]).astype(np.float32))


def test_upsample2d_nchw_passthrough(nn_cpu):
    # NCHW data format: the op resizes natively
    prev = nn.data_format
    try:
        nn.data_format = "NCHW"
        x = np.zeros((1, 2, 3, 4), dtype=np.float32)  # (B, C, H, W)
        x[0, 0, :, :] = np.arange(12, dtype=np.float32).reshape(3, 4)
        x[0, 1, :, :] = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
        out = nn.upsample2d(torch.as_tensor(x))
        assert out.shape == torch.Size([1, 2, 6, 8])
        assert np.array_equal(out.numpy(), _ri2(x, [2, 3]).astype(np.float32))
    finally:
        nn.data_format = prev


# ---------------------------------------------------------------------------
# _avg_pool2: official raw tf.nn.avg_pool VALID 2x2 stride-2
# ---------------------------------------------------------------------------

def test_avg_pool2_nhwc_valid_semantics(nn_cpu):
    # odd H (5) and even W (4): VALID must drop the last row only
    assert nn.data_format == "NHWC"
    x = np.zeros((1, 5, 4, 2), dtype=np.float32)  # (B, H, W, C)
    x[0, :, :, 0] = np.arange(20, dtype=np.float32).reshape(5, 4)
    x[0, :, :, 1] = np.arange(20, dtype=np.float32).reshape(5, 4) + 100
    out = _avg_pool2(torch.as_tensor(x))
    assert out.shape == torch.Size([1, 2, 2, 2])  # (B, OH, OW, C)
    assert np.array_equal(out.numpy(),
                          _avg_pool2d_reference(x, "NHWC").astype(np.float32))


def test_avg_pool2_nchw_passthrough(nn_cpu):
    # NCHW data format: F.avg_pool2d consumes the tensor natively
    prev = nn.data_format
    try:
        nn.data_format = "NCHW"
        x = np.zeros((1, 2, 5, 4), dtype=np.float32)  # (B, C, H, W)
        x[0, 0, :, :] = np.arange(20, dtype=np.float32).reshape(5, 4)
        x[0, 1, :, :] = np.arange(20, dtype=np.float32).reshape(5, 4) + 100
        out = _avg_pool2(torch.as_tensor(x))
        assert out.shape == torch.Size([1, 2, 2, 2])
        assert np.array_equal(out.numpy(),
                              _avg_pool2d_reference(x, "NCHW")
                              .astype(np.float32))
    finally:
        nn.data_format = prev


# ---------------------------------------------------------------------------
# module topology: list-valued attributes + recursive HourGlass
# ---------------------------------------------------------------------------

def test_fan_module_topology(fan2d):
    m = fan2d.model

    # the official list-valued attributes keep their lengths
    assert [len(getattr(m, a)) for a in
            ("m", "top_m", "conv_last", "bn_end", "l")] == [4] * 5
    assert len(m.bl) == 3 and len(m.al) == 3

    # the documented torch adaptation: every list element is the SAME
    # object registered in the module registry under the name the
    # build loop assigned (what makes named_parameters see the
    # hourglass trees and reproduce the official dotted scope keys)
    for attr in ("m", "top_m", "conv_last", "bn_end", "l", "bl", "al"):
        for i, sub in enumerate(getattr(m, attr)):
            assert m._modules[f"{attr}_{i}"] is sub, f"{attr}_{i}"

    # recursive HourGlass(256, 4) nesting: b2_plus chains down to a
    # depth-1 hourglass whose b2_plus is the terminal ConvBlock leaf
    h = m.m[0]                       # depth 4
    for _ in range(3):
        h = h.b2_plus                # depths 3, 2, 1
    leaf = h.b2_plus                 # ConvBlock (terminal)
    assert not hasattr(leaf, "b2_plus")
    assert hasattr(leaf, "conv1")

    # the torch module tree carries the full official dotted scope
    # chain (the checkpoint engine's named_parameters walk)
    names = set(dict(m.named_parameters()).keys())
    assert "m_0.b2_plus.b2_plus.b2_plus.b2_plus.bn1.weight" in names
    assert "m_3.b1.conv3.weight" in names
    assert "al_2.bias" in names
    assert "l_3.bias" in names

    # kernel shapes spot-checks (trunk + heads)
    assert m.conv1.weight.shape == torch.Size([64, 3, 7, 7])
    assert m.conv_last[0].weight.shape == torch.Size([256, 256, 1, 1])
    assert m.l[3].weight.shape == torch.Size([68, 256, 1, 1])
    assert m.al[0].weight.shape == torch.Size([256, 68, 1, 1])
    # ConvBlock convs are use_bias=False: no bias Parameter is
    # registered at all (the attribute is ABSENT, not None); the 1x1
    # heads keep their bias
    assert not hasattr(m.conv2.conv1, "bias")
    assert not hasattr(m.conv2.down_conv1, "bias")
    assert m.conv_last[0].bias is not None


# ---------------------------------------------------------------------------
# run() output contract: bare (1,68,64,64) NCHW float32
# ---------------------------------------------------------------------------

def test_run_output_nchw_contract(fan2d):
    img = (np.random.default_rng(0).integers(0, 256, (1, 256, 256, 3))
           ).astype(np.float32)
    out = fan2d.model.run([img])
    # single-output model: run() returns the BARE 4-D array (the
    # official tf_sess.run-of-one-tensor contract), not a list
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 68, 64, 64)
    assert out.dtype == np.float32
    # the official extract() idiom: [0] takes the batch-0 (68,64,64)
    # NCHW prediction map (trunk: 256 -> 128 -> 64; heads are 64x64)
    assert out[0].shape == (68, 64, 64)


# ---------------------------------------------------------------------------
# extract(): flip / feed rules, determinism, empty rects, uint8 pipeline
# ---------------------------------------------------------------------------

def test_extract_bgr_flip_equivalence_and_feed_rules(fan2d, monkeypatch):
    feeds = []

    def fake_run(inputs):
        feeds.append(inputs[0])
        return np.zeros((1, 68, 64, 64), dtype=np.float32)

    monkeypatch.setattr(fan2d.model, "run", fake_run)

    img = np.empty((120, 160, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 100
    img[:, :, 2] = 200
    rect = [20, 30, 120, 100]

    fan2d.extract(img, [rect], is_bgr=True)
    feed_bgr = feeds[-1]
    fan2d.extract(np.ascontiguousarray(img[:, :, ::-1]), [rect],
                  is_bgr=False)
    feed_rgb = feeds[-1]

    # the feed is the official crop output: 256x256 NHWC float32 in
    # [0, 1] (the /255.0 normalization after the uint8 crop)
    assert feed_bgr.shape == (1, 256, 256, 3)
    assert feed_bgr.dtype == np.float32
    assert float(feed_bgr.min()) >= 0.0 and float(feed_bgr.max()) <= 1.0
    # BGR-in with is_bgr=True == RGB-in with is_bgr=False: the flip
    # precedes all channel-independent crop math -> bit-identical feed
    assert np.array_equal(feed_bgr, feed_rgb)
    # deterministic: same image + rect -> same feed
    fan2d.extract(img, [rect], is_bgr=True)
    assert np.array_equal(feeds[-1], feed_bgr)


def test_extract_empty_rects_and_uint8_pipeline(fan2d):
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    # the official early return: no model call at all
    assert fan2d.extract(img, []) == []

    # one rect covering the image: full pipeline (crop -> /255 ->
    # run -> argmax decode) yields a non-null (68, 2) landmark set
    y, x = np.mgrid[0:96, 0:96]
    img[..., 0] = (x % 256).astype(np.uint8)
    img[..., 1] = (y % 256).astype(np.uint8)
    img[..., 2] = ((x + y) % 256).astype(np.uint8)
    landmarks = fan2d.extract(img, [[0, 0, 96, 96]])
    assert len(landmarks) == 1
    assert landmarks[0] is not None
    assert landmarks[0].shape == (68, 2)
    assert np.all(np.isfinite(landmarks[0]))


# ---------------------------------------------------------------------------
# transform() affine + get_pts_from_predict decode (official math)
# ---------------------------------------------------------------------------

def test_transform_affine_math(fan2d):
    # hand-computed official affine: center [0,0], scale 1,
    # resolution 64 -> m = [[64/200, 0, 32], [0, 64/200, 32],
    # [0, 0, 1]] (inverted in the official code): crop coordinate
    # (32, 32) maps to image (0, 0) and (42.24, 42.24) maps back to
    # crop (32, 32)
    p = fan2d.transform([32, 32], [0, 0], 1.0, 64)
    assert np.allclose(p, [0, 0], atol=1e-6)
    p = fan2d.transform([42.24, 42.24], [0, 0], 1.0, 64)
    assert np.allclose(p, [32, 32], atol=1e-4)


def test_decode_subpixel_math(fan2d):
    # crafted heatmaps: per-channel peak at (h=10, w=20) with pinned
    # neighbours that force the official sign correction
    # ([a[pY,pX+1]-a[pY,pX-1], a[pY+1,pX]-a[pY-1,pX]] = [0.5, -0.5]
    # -> c += sign * 0.25); plus the official +0.5 centering.
    a = np.zeros((68, 64, 64), dtype=np.float32)
    for ch in range(68):
        a[ch, 10, 20] = 1.0
        a[ch, 10, 21] = 0.5   # pX + 1
        a[ch, 9, 20] = 0.5    # pY - 1
    center = np.array([60.0, 60.0])
    scale = 200.0 / 195.0     # the official (r-l+b-t)/195 for this rect

    pts = fan2d.get_pts_from_predict(a, center, scale)
    # decode: argmax -> (x=w=20, y=h=10); subpixel -> (20.25, 9.75);
    # +0.5 -> (20.75, 10.25); transform back to image coordinates
    assert pts.shape == (68, 2)
    want = fan2d.transform([20.75, 10.25], center, scale, 64)
    assert np.allclose(pts, np.tile(want, (68, 1)), atol=1e-5)

    # corner peak (0,0): outside the 0 < p < 63 window -> NO
    # subpixel correction, c = [0.5, 0.5]
    b = np.zeros((68, 64, 64), dtype=np.float32)
    b[:, 0, 0] = 1.0
    pts = fan2d.get_pts_from_predict(b, center, scale)
    want = fan2d.transform([0.5, 0.5], center, scale, 64)
    assert np.allclose(pts, np.tile(want, (68, 1)), atol=1e-5)


# ---------------------------------------------------------------------------
# opt-in CPU parity vs the frozen official (TF) reference
# ---------------------------------------------------------------------------

def test_frozen_tf_reference_parity(fan2d, fan3d):
    tfre_dir = os.environ.get("DFL_TEST_FAN_TFRE_DIR")
    if tfre_dir is None:
        pytest.skip(
            "DFL_TEST_FAN_TFRE_DIR not set "
            "(opt-in frozen-TF parity; points at the git-ignored "
            "docs/_p9_tfref directory)")
    tfre_root = Path(tfre_dir)
    frozen = tfre_root / "frozen"
    repo = tfre_root.parent.parent
    meta = json.loads((frozen / "meta.json").read_text(encoding="utf-8"))

    # weight provenance: the tracked facelib files are byte-identical
    # to the files the frozen reference ran with
    for name, want_md5 in meta["weight_md5"].items():
        got = hashlib.md5((repo / "facelib" / name).read_bytes()).hexdigest()
        assert got == want_md5, f"{name} MD5 {got} != frozen {want_md5}"

    for sample in meta["samples"]:
        stem = sample["stem"]
        ref = json.loads((frozen / f"{stem}.json")
                         .read_text(encoding="utf-8"))
        img_path = repo / ref["image"]["repo_path"]
        raw_md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
        assert raw_md5 == ref["image"]["md5"], stem
        img = cv2.imread(str(img_path))
        assert img is not None and img.shape[2] == 3, stem

        # the harness's FAN rect set: the frozen S3FD final rects;
        # the synthetic sample forces the fixed face rect
        rects_for_fan = [list(map(int, r))
                         for r in ref["s3fd"]["rects"]]
        if stem == "synthetic_640x640" \
                and [128, 128, 512, 512] not in rects_for_fan:
            rects_for_fan = [[128, 128, 512, 512]]

        for fan_name, fan in (("fan2d", fan2d), ("fan3d", fan3d)):
            # 1) official extract() entry point (is_bgr=True default,
            #    second_pass_extractor=None default) — per-rect
            #    (68, 2) landmarks (or None if the official path
            #    swallowed an exception)
            got = fan.extract(img, rects_for_fan)
            ref_lms = ref[fan_name]["landmarks"]
            assert len(got) == len(rects_for_fan), stem
            assert set(ref_lms) == {str(i)
                                    for i in range(len(rects_for_fan))}, stem
            for i, want in ref_lms.items():
                if want is None:
                    assert got[int(i)] is None, (stem, fan_name, i)
                    continue
                if stem == "synthetic_640x640":
                    # hard argmax on the seeded-noise heatmaps ties
                    # between TF and torch (legitimate kernel
                    # difference, not a model discrepancy); the
                    # prediction map below is the continuous gate
                    continue
                got_lm = np.asarray(got[int(i)], dtype=np.float64)
                want_lm = np.asarray(want, dtype=np.float64)
                assert got_lm.shape == want_lm.shape == (68, 2), (
                    stem, fan_name, i)
                worst = np.abs(got_lm - want_lm).max()
                assert worst <= 1e-4, (
                    f"{stem}/{fan_name}[{i}]: worst |d| = {worst:.3e}")

            # 2) full prediction map (synthetic sample, fan2d): the
            #    harness's exact replica lines — crop from the RGB
            #    image, /255 float32, run, batch-0 (68,64,64) map
            if stem == "synthetic_640x640" and fan_name == "fan2d":
                c = np.array([(128 + 512) / 2.0, (128 + 512) / 2.0])
                scale = (512 - 128 + 512 - 128) / 195.0
                crop = fan.crop(img[:, :, ::-1], c, scale)
                crop = crop.astype(np.float32) / 255.0
                pred = fan.model.run([crop[None, ...]])[0]
                ref_map = np.asarray(
                    ref[fan_name]["synthetic_prediction_map"],
                    dtype=np.float32)
                assert pred.shape == ref_map.shape == (68, 64, 64)
                worst = np.abs(pred - ref_map).max()
                assert worst <= 1e-4, (
                    f"{stem}: prediction-map worst |d| = {worst:.3e}")
