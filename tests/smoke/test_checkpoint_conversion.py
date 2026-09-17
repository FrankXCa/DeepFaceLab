"""Phase 4 acceptance: checkpoint compatibility and conversion.

Covers the Phase 4 centralized conversion engine
(``core.leras/convert.py`` — independent reimplementation; concept
sources per docs/PHASE4_PLAN.md items 16-19: official DFL file format
contract (GPL-3.0), EXTERNAL_A strict two-pass load concept (GPL-3.0),
EXTERNAL_B component/optimizer-state mapping concepts (unlicensed —
NOT copied)):

- official file format: the official ``Saveable.save_weights`` writes
  a RAW pickle protocol-4 stream of a ``dict[str, np.ndarray]`` via
  ``pickle.dumps(d, 4)`` + ``pathex.write_bytes_safe`` — the ``.npy``
  extension is a misnomer, the file is NOT a NumPy ``.npy`` container
  (``np.save``/``np.load`` are not involved at any level; "protocol
  4" is the pickle protocol of the whole file, there is no separate
  outer container); the official ``load_weights`` reads it back with
  ``pickle.loads(file_bytes)``. The container contract is pinned
  independently of the converter's own reader (leading pickle bytes
  identical to the real artifacts — raw protocol-4 streams, not
  NumPy ``.npy`` containers — and the pickle module, the official
  load primitive, parses keys/shapes/dtypes/values exactly) and
  against the REAL official artifacts tracked in the baseline
  (``facelib/*.npy`` — S3FD/2DFAN float32 with the NHWC 4-D
  singleton-padded ``(1,1,1,C)`` bias/BN forms, 3DFAN float32 with
  1-D ``(C,)`` forms, FaceEnhancer float16), plus the re-pickle
  protocol-4 round-trip and rejection of corrupt files (truncated
  pickles, non-dict pickles, real ``np.save``-format array files,
  non-string keys, non-ndarray values);
- name mapping: the pure string transform (torch dotted <-> official
  slashed ``:0``) and the ``:0``-variant tolerance, both directions;
- layout conversion with DECLARED rules only (no element-count
  reshape fallback): Conv2D (kH,kW,in,out)<->(out,in,kH,kW),
  Conv2DTranspose (kH,kW,out,in)<->(in,out,kH,kW), DepthwiseConv2D
  (kH,kW,in,dm)<->(in*dm,1,kH,kW), Dense official-layout identity
  (in,out), and the ``channel_broadcast`` WHITELIST for 1-D
  parameters (conv bias / BN weight/bias/running_mean/running_var /
  FRNorm eps): exactly the known official singleton layouts
  ``(C,)`` (identity), ``(1,1,1,C)`` (NHWC padding) and
  ``(1,C,1,1)`` (NCHW padding) — every other shape, including
  same-element-count placements a naive ``np.squeeze()`` would
  "fix" (``(1,C)``, ``(1,1,C)``, ``(1,C,1)``, ``(C,1)``,
  ``(1,1,C,1)``, ``(C,1,1,1)``, ``(2,1,1,C)``), is rejected
  explicitly; layout proofs use UNIQUE-VALUE tensors so any wrong
  axis order is visible (element counts are never an identity
  proof);
- strict two-pass (all-or-nothing) conversion: missing required
  weights, unexpected extra keys, shape mismatch (including
  same-element-count wrong shapes), dtype mismatch (float16 file into
  a float32 module), ambiguous mapping (two sub-names resolving to
  one source key), corrupt values -> ``CheckpointLoadError`` with the
  full structured report; on failure NOTHING is copied;
- reverse export (torch -> official): the official-layout dict via
  the per-layer ``convert_weight_to_official`` hooks; explicit
  rejection with ``UnsupportedExportError`` (never a silent drop /
  approximation / reshape / coercion) when a state is flagged
  non-exportable;
- optimizer state: official ``iters:0`` + ``ms_*``/``vs_*``/``acc_*``
  sub-names mapped NAME-driven (the state key embeds the variable's
  official name; the initialize_variables order only names positional
  parameters — never the sole identity): AdaBelief and RMSprop,
  value-exact resume equivalence (iters + all states ``torch.equal``
  after a fresh optimizer), the declared official int32 -> torch
  int64 iteration-counter widening, and the strict failures (missing
  iters, missing one state, extra state, unrecognized prefix, state
  referencing a variable outside the given saveable);
- file-level round-trips through write_official_checkpoint /
  read_official_checkpoint: torch -> official -> torch and official
  -> torch -> official for layers, archis (canonical option combos)
  and discriminators;
- Saveable.load_weights agreement: the Phase 3A strict loader and the
  Phase 4 converter agree on every value for the same file; the
  0-D ``iters`` counter survives under both pinned NumPy versions
  (the NumPy 2.x ``ascontiguousarray`` 0-D -> (1,) upgrade is
  guarded);
- GPU: device-neutral engine (CPU/NumPy arrays copied onto the
  parameter device), RTX 4090 conversion into a GPU module with
  bit-exact CPU-twin parity (skip on CPU-only environments);
- import boundary: no TensorFlow import and no direct
  ``torch.cuda.*`` in the conversion source (AST).

Parity labels: EXACT for the value/layout/key/dtype round-trips
(pure index rearrangement + value copy — ``torch.equal`` /
``np.array_equal``; the GPU copy is bit-exact); the real facelib
files validate the FORMAT contract (EXACT), not extractor-model
compatibility (Phase 9); SAEHD/AMP/Quick96/XSeg full-model
checkpoint compatibility: NOT_YET_IMPLEMENTED (Phases 6-8).
"""

import ast
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras import checkpoint as dfl_ckpt  # noqa: E402
from core.leras import convert as cv  # noqa: E402
import core.leras.models.PatchDiscriminator  # noqa: F401
# the package __init__ star-imports the class onto the package
# attribute name, shadowing the submodule; sys.modules is authoritative
_PD_mod = sys.modules["core.leras.models.PatchDiscriminator"]

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)


def init_cpu(data_format="NCHW"):
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", data_format)


def init_gpu(data_format="NCHW"):
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", data_format)


def unique_value(shape, dtype=np.float32):
    """Deterministic UNIQUE-VALUE array: every element distinct, so no
    wrong axis order can survive an equality check (element counts
    are never an identity proof)."""
    a = np.arange(int(np.prod(shape)), dtype=np.float64).astype(dtype)
    return a.reshape(shape)


# ---------------------------------------------------------------------------
# Official file format (real official artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,n_keys,dtype", [
    ("facelib/S3FD.npy", 65, np.float32),
    ("facelib/2DFAN.npy", 945, np.float32),
    ("facelib/3DFAN.npy", 945, np.float32),
    ("facelib/FaceEnhancer.npy", 72, np.float16),
])
def test_real_official_files_parse(rel_path, n_keys, dtype):
    d = cv.read_official_checkpoint(str(REPO_ROOT / rel_path))
    assert isinstance(d, dict)
    assert len(d) == n_keys
    for k, v in d.items():
        assert isinstance(k, str) and k.endswith(":0"), k
        assert isinstance(v, np.ndarray), type(v)
    # value dtype: uniform across the file (f32 extractors, f16
    # FaceEnhancer - official use_fp16)
    dtypes = {np.dtype(v.dtype) for v in d.values()}
    assert dtypes == {np.dtype(dtype)}, dtypes


def test_real_official_key_conventions():
    s3fd = cv.read_official_checkpoint(str(REPO_ROOT / "facelib/S3FD.npy"))
    # official HWIO conv layout (kH, kW, in, out)
    assert s3fd["conv1_1/weight:0"].shape == (3, 3, 3, 64)
    # official 4-D singleton-padded bias form (S3FD era)
    assert s3fd["conv1_1/bias:0"].shape == (1, 1, 1, 64)
    dfan = cv.read_official_checkpoint(str(REPO_ROOT / "facelib/2DFAN.npy"))
    assert dfan["conv1/weight:0"].shape == (7, 7, 3, 64)
    assert dfan["bn1/running_mean:0"].shape == (1, 1, 1, 64)
    assert dfan["bn1/running_var:0"].shape == (1, 1, 1, 64)
    dfan3 = cv.read_official_checkpoint(str(REPO_ROOT / "facelib/3DFAN.npy"))
    # the 3DFAN file stores the same logical state 1-D (C,)
    assert dfan3["bn1/running_mean:0"].shape == (64,)
    fe = cv.read_official_checkpoint(str(REPO_ROOT / "facelib/FaceEnhancer.npy"))
    assert np.dtype(fe["conv1/weight:0"].dtype) == np.float16


def test_real_official_files_protocol4_repickle_roundtrip():
    d = cv.read_official_checkpoint(str(REPO_ROOT / "facelib/S3FD.npy"))
    again = pickle.loads(pickle.dumps(d, 4))
    assert set(again) == set(d)
    for k in d:
        assert np.array_equal(again[k], d[k]), k


def test_corrupt_files_rejected(plain_tmp):
    def _w(name, blob):
        p = Path(plain_tmp) / name
        p.write_bytes(blob)
        return str(p)

    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(_w("trunc.npy", b"\x80\x02\x95\x00\x00\x00\x00\x00\x00\x00."))
    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(_w("nondict.npy", pickle.dumps([1, 2, 3], 4)))
    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(_w("nondictobj.npy", pickle.dumps(object(), 4)))
    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(_w("listval.npy", pickle.dumps({"a:0": [1.0]}, 4)))
    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(_w("key.npy", pickle.dumps({1: np.zeros(2)}, 4)))
    # a real np.save-format array file is NOT the official dict format
    p = Path(plain_tmp) / "array.npy"
    np.save(str(p), np.zeros(3))
    with pytest.raises(dfl_ckpt.CheckpointLoadError, match="CORRUPT_CHECKPOINT"):
        cv.read_official_checkpoint(str(p))


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

def test_official_outer_container_contract_independent(plain_tmp):
    """The OUTER file contract, verified without convert.py's reader:
    direct byte inspection + the pickle module (the official load
    primitive — official ``Saveable.load_weights`` is exactly
    ``pickle.loads(Path.read_bytes())``).

    The official writer is ``pickle.dumps(d, 4)`` through
    ``pathex.write_bytes_safe``: a RAW pickle protocol-4 stream. The
    ``.npy`` extension is a misnomer — the file is NOT a NumPy ``.npy``
    container (``np.save``/``np.load`` play no role), so "protocol 4"
    is the pickle protocol of the whole file, not an inner payload.
    """
    import pickle as _pickle

    d = {
        "weight:0": unique_value((3, 3, 2, 4)),   # official HWIO kernel
        "bias:0": unique_value((4,)),
        "running_mean:0": unique_value((1, 1, 1, 4)),  # NHWC padded form
        "iters:0": np.zeros((), np.int64),
    }
    p = str(Path(plain_tmp) / "contract.npy")
    cv.write_official_checkpoint(p, d)

    raw = Path(p).read_bytes()
    # outer container: raw pickle protocol-4 stream — NOT an npy file
    assert raw[:2] == b"\x80\x04"
    assert raw[:6] != b"\x93NUMPY"

    # structural comparison with a REAL official artifact: identical
    # container framing (PROTO 4 + FRAME opcodes)
    with open(str(REPO_ROOT / "facelib" / "S3FD.npy"), "rb") as f:
        real_head = f.read(3)
    assert real_head == b"\x80\x04\x95"
    assert raw[:3] == real_head  # same outer container as the real file

    # the official load primitive (not the converter's reader) parses it
    d2 = _pickle.loads(raw)
    assert isinstance(d2, dict)
    assert list(d2) == list(d)  # key names + order preserved exactly
    for k, v in d.items():
        assert isinstance(d2[k], np.ndarray), k
        assert d2[k].shape == v.shape, k
        assert np.dtype(d2[k].dtype) == np.dtype(v.dtype), k
        assert np.array_equal(d2[k], v), k

    # container pin: both our file and the real official artifact are
    # raw pickle streams, not NumPy .npy containers (the NUMPY magic
    # is absent). Whether numpy's legacy pickle path in np.load reads
    # such files is a NumPy implementation detail across versions —
    # the official reader is pickle.loads, and that is what this
    # contract pins.
    with open(str(REPO_ROOT / "facelib" / "S3FD.npy"), "rb") as f:
        real_head = f.read(6)
    assert raw[:6] != b"\x93NUMPY"
    assert real_head != b"\x93NUMPY"
    assert real_head[:2] == b"\x80\x04"


def test_name_mapping_roundtrip():
    cases = [
        "weight", "bias", "running_mean", "running_var",
        "conv1_1/weight", "encoder/conv1_1/bias",
        "inter_AB/batchnorm_1/running_mean", "GAN/convs_0/weight",
    ]
    for official in cases:
        dotted = dfl_ckpt.torch_name_from_official(official + ":0")
        assert dfl_ckpt.official_name(dotted) == official + ":0"
    # :0 already present is preserved, not doubled
    assert dfl_ckpt.official_name("a/b:0") == "a/b:0"
    # both :0 variants resolve
    d = {"weight": np.zeros(2, np.float32)}
    v, k = dfl_ckpt._lookup_key(d, "weight:0")
    assert v is not None and k == "weight"
    d2 = {"weight:0": np.zeros(2, np.float32)}
    v2, k2 = dfl_ckpt._lookup_key(d2, "weight")
    assert v2 is not None and k2 == "weight:0"


# ---------------------------------------------------------------------------
# Layout conversion proofs (unique values; declared rules only)
# ---------------------------------------------------------------------------

def test_conv2d_layout_both_directions():
    init_cpu()
    # non-square CHANNELS (in != out): any wrong axis order yields a
    # different shape or different values (unique values)
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, strides=1, padding='SAME', name="c")
    c.build_weights()
    src = unique_value((3, 3, 3, 5))          # official (kH, kW, in, out)
    rep = cv.convert_official_to_torch(c, {"weight:0": src,
                                           "bias:0": unique_value((5,))},
                                       component="c")
    assert rep.result == "PASS"
    w_map = [m for m in rep.mapped if m.destination_name == "weight:0"][0]
    assert w_map.rule == "layer_layout:Conv2D"
    assert w_map.status == "MAPPED_LAYOUT_CONVERTED"
    # torch (out, in, kH, kW) == official (kH, kW, out, in) transposed
    assert torch.equal(c.weight, torch.from_numpy(np.ascontiguousarray(
        src.transpose(3, 2, 0, 1))))
    # reverse: torch -> official
    rep_r = cv.convert_torch_to_official(c, component="c")
    assert rep_r.result == "PASS"
    assert np.array_equal(rep_r.payload["weight:0"], src)


def test_conv2dtranspose_layout_both_directions():
    init_cpu()
    c = dfl_nn.Conv2DTranspose(3, 5, kernel_size=3, strides=1, padding='SAME', name="c")
    c.build_weights()
    # official (kH, kW, out, in)
    src = unique_value((3, 3, 5, 3))
    rep = cv.convert_official_to_torch(c, {"weight:0": src,
                                           "bias:0": unique_value((5,))},
                                       component="c")
    assert rep.result == "PASS"
    # torch (in, out, kH, kW)
    assert torch.equal(c.weight, torch.from_numpy(np.ascontiguousarray(
        src.transpose(3, 2, 0, 1))))
    rep_r = cv.convert_torch_to_official(c, component="c")
    assert np.array_equal(rep_r.payload["weight:0"], src)


def test_depthwise_conv_layout_both_directions():
    init_cpu()
    c = dfl_nn.DepthwiseConv2D(3, 3, strides=1, padding='SAME',
                               depth_multiplier=2, name="c")
    c.build_weights()
    # official (kH, kW, in, dm)  ->  torch (in*dm, 1, kH, kW)
    src = unique_value((3, 3, 3, 2))
    rep = cv.convert_official_to_torch(c, {"weight:0": src,
                                           "bias:0": unique_value((6,))},
                                       component="c")
    assert rep.result == "PASS"
    ref = np.ascontiguousarray(
        src.transpose(2, 3, 0, 1).reshape(6, 3, 3)[:, None, :, :])
    assert torch.equal(c.weight, torch.from_numpy(ref))
    rep_r = cv.convert_torch_to_official(c, component="c")
    assert rep_r.payload["weight:0"].shape == (3, 3, 3, 2)
    assert np.array_equal(rep_r.payload["weight:0"], src)


def test_dense_official_layout_identity():
    init_cpu()
    d = dfl_nn.Dense(3, 5, name="d")
    d.build_weights()
    # the project's Dense KEEPS the official (in, out) layout in torch
    src = unique_value((3, 5))
    rep = cv.convert_official_to_torch(d, {"weight:0": src,
                                           "bias:0": unique_value((5,))},
                                       component="d")
    assert rep.result == "PASS"
    assert [m for m in rep.mapped if m.destination_name == "weight:0"][0].rule == "identity"
    assert torch.equal(d.weight, torch.from_numpy(src))
    rep_r = cv.convert_torch_to_official(d, component="d")
    assert np.array_equal(rep_r.payload["weight:0"], src)


def test_channel_broadcast_whitelist_forms():
    # exactly the known official singleton layouts for a 1-D parameter
    # of size C=5: (C,) identity, (1,1,1,C) NHWC, (1,C,1,1) NCHW
    init_cpu()
    for form, expected_rule in ((5,), "identity"), \
                                ((1, 1, 1, 5), "channel_broadcast"), \
                                ((1, 5, 1, 1), "channel_broadcast"):
        c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
        c.build_weights()
        src = unique_value((3, 3, 3, 5))
        bias = unique_value((5,)).reshape(form)
        rep = cv.convert_official_to_torch(
            c, {"weight:0": src, "bias:0": bias}, component="c")
        assert rep.result == "PASS", form
        rule = [m for m in rep.mapped if m.destination_name == "bias:0"][0].rule
        assert rule == expected_rule, (form, rule)
        # value-exact: the channel axis values land in order
        assert torch.equal(c.bias, torch.from_numpy(unique_value((5,)))), form


def test_channel_broadcast_rejects_malformed_same_count_shapes():
    # malformed placements that np.squeeze() would "fix" to the right
    # 1-D length but that are NOT known official layouts — the
    # whitelist must reject them explicitly (never a disguised
    # element-count reshape)
    init_cpu()
    rejected = [
        (1, 5),        # 2-D: channel last but rank 2 (not official)
        (5, 1),        # 2-D: channel first
        (1, 1, 5),     # 3-D (not official — official forms are 1-D/4-D)
        (1, 5, 1),     # 3-D middle placement
        (5, 1, 1),     # 3-D channel first
        (1, 1, 5, 1),  # 4-D trailing singleton (not an official layout)
        (5, 1, 1, 1),  # 4-D channel first
        (2, 1, 1, 5),  # leading dim not a singleton
        (1, 5, 1, 5),  # channel duplicated (not a singleton pattern)
    ]
    for form in rejected:
        c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
        c.build_weights()
        before = {n: p.detach().clone() for n, p in
                  list(c.named_parameters()) + list(c.named_buffers())}
        src = unique_value((3, 3, 3, 5))
        # distinct values (some malformed forms carry a different
        # element count, e.g. (2,1,1,5) — the content is irrelevant:
        # the conversion must fail on the shape alone)
        bias = np.arange(int(np.prod(form)), dtype=np.float32).reshape(form)
        with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
            cv.convert_official_to_torch(
                c, {"weight:0": src, "bias:0": bias}, component="c")
        text = ei.value.args[0]
        assert ("SHAPE_MISMATCH" in text or "INVALID_LAYOUT" in text), form
        # all-or-nothing: nothing was copied on failure
        after = {n: p.detach().cpu() for n, p in
                 list(c.named_parameters()) + list(c.named_buffers())}
        assert all(torch.equal(before[k], after[k]) for k in before), form


def test_bn_states_squeeze_and_identity():
    init_cpu()
    for cls in (dfl_nn.BatchNorm2D, dfl_nn.InstanceNorm2D, dfl_nn.FRNorm2D):
        layer = cls(4, name="bn")
        layer.build_weights()
        # only the state this class actually enumerates (BN carries the
        # running statistics; instance/FR norm differ), each value with
        # its parameter's own shape (e.g. FRNorm's 1-D eps)
        items = list(layer._iter_official_weights())
        subs = [dfl_ckpt.strip_zero_suffix(s) for s, _ in items]
        shapes = {dfl_ckpt.strip_zero_suffix(s): tuple(t.shape)
                  for s, t in items}
        vals = {f"{s}:0": unique_value(shapes[s]) for s in subs}
        rep = cv.convert_official_to_torch(layer, vals, component="bn")
        assert rep.result == "PASS", cls.__name__
        # and the 4-D singleton-padded official forms
        layer2 = cls(4, name="bn")
        layer2.build_weights()
        vals4 = {f"{s}:0": unique_value(shapes[s]).reshape(
            (1,) * (4 - len(shapes[s])) + shapes[s]) for s in subs}
        rep4 = cv.convert_official_to_torch(layer2, vals4, component="bn")
        assert rep4.result == "PASS", cls.__name__
        for s in subs:
            p = None
            for _n, _p in list(layer2.named_parameters()) + \
                    list(layer2.named_buffers()):
                if _n == s:
                    p = _p
                    break
            assert p is not None
            assert torch.equal(p, torch.from_numpy(unique_value(shapes[s]))), (
                cls.__name__, s)


# ---------------------------------------------------------------------------
# Strict failure behavior (weights)
# ---------------------------------------------------------------------------

def _fresh_conv():
    init_cpu()
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
    c.build_weights()
    return c


def _snapshot(module):
    return {n: p.detach().cpu().clone() for n, p in
            list(module.named_parameters()) + list(module.named_buffers())}


def test_missing_required_key_fails_and_copies_nothing():
    c = _fresh_conv()
    before = _snapshot(c)
    src = unique_value((3, 3, 3, 5))
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(c, {"weight:0": src}, component="c")
    assert "MISSING_REQUIRED_KEY" in ei.value.args[0]
    assert "bias:0" in ei.value.args[0]
    after = _snapshot(c)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_unexpected_extra_key_fails():
    c = _fresh_conv()
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(
            c, {"weight:0": unique_value((3, 3, 3, 5)),
                "bias:0": unique_value((5,)),
                "ghost:0": np.zeros(2, np.float32)}, component="c")
    assert "UNEXPECTED_EXTRA_KEY" in ei.value.args[0]
    assert "ghost:0" in ei.value.args[0]


def test_same_element_count_wrong_shape_fails():
    c = _fresh_conv()
    # (5, 3, 3, 3) has the same 225 elements as (3, 3, 3, 5) but is the
    # TORCH layout for this conv: a declared-layout layer must reject
    # it explicitly (INVALID_LAYOUT), never silently accept it
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(
            c, {"weight:0": unique_value((5, 3, 3, 3)),
                "bias:0": unique_value((5,))}, component="c")
    text = ei.value.args[0]
    assert ("INVALID_LAYOUT" in text) or ("SHAPE_MISMATCH" in text)
    # and a non-declared layer with an equal-count wrong shape: Dense
    # (in, out) = (3, 5) vs a (5, 3) file
    d = dfl_nn.Dense(3, 5, name="d")
    d.build_weights()
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as e2:
        cv.convert_official_to_torch(
            d, {"weight:0": unique_value((5, 3)),
                "bias:0": unique_value((5,))}, component="d")
    assert "SHAPE_MISMATCH" in e2.value.args[0]


def test_dtype_mismatch_fails():
    c = _fresh_conv()
    # float16 official file (FaceEnhancer convention) into a float32
    # module: strict - no silent coercion
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(
            c, {"weight:0": unique_value((3, 3, 3, 5)).astype(np.float16),
                "bias:0": unique_value((5,))}, component="c")
    assert "DTYPE_MISMATCH" in ei.value.args[0]
    assert "float16" in ei.value.args[0]


def test_ambiguous_mapping_fails():
    # a saveable whose enumeration contains BOTH ':0' variants of the
    # same logical key; one source key serves both -> ambiguous
    init_cpu()

    class _VariantSaveable(torch.nn.Module, dfl_nn.Saveable):
        def __init__(self, name=None):
            super().__init__()
            dfl_nn.Saveable.__init__(self, name)
            self.pa = torch.nn.Parameter(torch.zeros(2, device=dfl_nn.device,
                                                    dtype=dfl_nn.floatx),
                                         requires_grad=True)
            self.pb = torch.nn.Parameter(torch.zeros(2, device=dfl_nn.device,
                                                    dtype=dfl_nn.floatx),
                                         requires_grad=True)

        def _iter_official_weights(self):
            return [("weight:0", self.pa), ("weight", self.pb)]

    s = _VariantSaveable(name="s")
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_official_to_torch(s, {"weight:0": np.zeros(2, np.float32)},
                                     component="s")
    assert "AMBIGUOUS_MAPPING" in ei.value.args[0]


def test_report_structure_counts():
    c = _fresh_conv()
    with pytest.raises(dfl_ckpt.CheckpointLoadError):
        cv.convert_official_to_torch(
            c, {"weight:0": unique_value((3, 3, 3, 5))}, component="c")
    # passing case: counts derived from the report
    c2 = _fresh_conv()
    rep = cv.convert_official_to_torch(
        c2, {"weight:0": unique_value((3, 3, 3, 5)),
             "bias:0": unique_value((5,)).reshape(1, 1, 1, 5)}, component="c")
    assert rep.result == "PASS"
    assert rep.mapped_state_count == 2
    assert rep.layout_conversion_count == 2
    assert rep.missing_required_count == 0
    assert rep.dtype_mismatch_count == 0
    text = rep.to_text()
    for token in ("direction: OFFICIAL_TO_TORCH",
                  "source_format: official_dfl_pickled_dict_v4",
                  "destination_format: torch_named_parameters",
                  "conversion_rule:", "result: PASS"):
        assert token in text, token


# ---------------------------------------------------------------------------
# Reverse export (torch -> official)
# ---------------------------------------------------------------------------

def test_reverse_export_official_layout():
    init_cpu()
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
    c.build_weights()
    src = unique_value((3, 3, 3, 5))
    cv.convert_official_to_torch(c, {"weight:0": src,
                                     "bias:0": unique_value((5,))}, component="c")
    rep = cv.convert_torch_to_official(c, component="c")
    assert rep.result == "PASS"
    assert sorted(rep.payload) == ["bias:0", "weight:0"]
    # the exported file carries the OFFICIAL layout (official-readable)
    assert rep.payload["weight:0"].shape == (3, 3, 3, 5)
    assert np.array_equal(rep.payload["weight:0"], src)


def test_reverse_export_rejects_non_exportable_state():
    init_cpu()
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
    c.build_weights()
    with pytest.raises(cv.UnsupportedExportError) as ei:
        cv.convert_torch_to_official(
            c, component="c",
            non_exportable={"weight:0": "modern-only state (test fixture)"})
    assert "UNSUPPORTED_REVERSE_EXPORT" in ei.value.args[0]
    assert "weight:0" in ei.value.args[0]
    assert "never dropped" in ei.value.args[0]
    # nothing is written: the export failed as a whole
    assert not hasattr(cv.convert_torch_to_official, "payload")


# ---------------------------------------------------------------------------
# Optimizer state conversion
# ---------------------------------------------------------------------------

def _bound_conv(name="encoder"):
    c = dfl_nn.Conv2D(3, 4, kernel_size=5, padding='SAME', name=name)
    c.build_weights()
    ps = list(c.parameters())
    ps[0]._dfl_name = f"{name}/weight:0"
    ps[1]._dfl_name = f"{name}/bias:0"
    return c, ps


def _two_steps(opt, params):
    # per-parameter constant gradients (deterministic, no RNG)
    with torch.no_grad():
        opt.get_update_op([(torch.full_like(p, 0.5), p) for p in params])()
        opt.get_update_op([(torch.full_like(p, -0.25), p) for p in params])()


def test_adabelief_optimizer_state_roundtrip():
    init_cpu()
    _, ps = _bound_conv()
    opt = dfl_nn.AdaBelief(name="opt", lr=0.01)
    opt.initialize_variables(ps, vars_on_cpu=True)
    _two_steps(opt, ps)
    od = cv.convert_optimizer_state_torch_to_official(opt)
    # official names, exactly
    assert sorted(od) == sorted(["iters:0",
                                 "ms_encoder/weight_0:0",
                                 "ms_encoder/bias_0:0",
                                 "vs_encoder/weight_0:0",
                                 "vs_encoder/bias_0:0"])
    assert int(od["iters:0"]) == 2
    assert np.dtype(od["iters:0"].dtype) == np.int64

    # fresh optimizer + matching saveable: official -> torch
    saveable2, ps2 = _bound_conv()
    opt2 = dfl_nn.AdaBelief(name="opt2", lr=0.01)
    opt2.initialize_variables(ps2, vars_on_cpu=True)
    rep = cv.convert_optimizer_state_official_to_torch(opt2, od,
                                                       saveable=saveable2,
                                                       component="opt2")
    assert rep.result == "PASS"
    assert int(opt2.iterations) == 2  # value-exact resume
    assert all(torch.equal(a, b) for a, b in
               zip(opt.ms_dict.values(), opt2.ms_dict.values()))
    assert all(torch.equal(a, b) for a, b in
               zip(opt.vs_dict.values(), opt2.vs_dict.values()))


def test_rmsprop_optimizer_state_roundtrip():
    init_cpu()
    _, ps = _bound_conv()
    opt = dfl_nn.RMSprop(name="opt", lr=0.001, rho=0.9)
    opt.initialize_variables(ps, vars_on_cpu=True)
    _two_steps(opt, ps)
    od = cv.convert_optimizer_state_torch_to_official(opt)
    assert sorted(od) == sorted(["iters:0",
                                 "acc_encoder/weight_0:0",
                                 "acc_encoder/bias_0:0"])
    _, ps2 = _bound_conv()
    opt2 = dfl_nn.RMSprop(name="opt2", lr=0.001, rho=0.9)
    opt2.initialize_variables(ps2, vars_on_cpu=True)
    rep = cv.convert_optimizer_state_official_to_torch(opt2, od,
                                                       component="opt2")
    assert rep.result == "PASS"
    assert int(opt2.iterations) == 2
    assert all(torch.equal(a, b) for a, b in
               zip(opt.accumulators_dict.values(),
                   opt2.accumulators_dict.values()))


def test_iters_int32_official_widening():
    # official TF stored iters as int32; the torch counter is int64 —
    # declared exact widening, reported (not a silent cast)
    init_cpu()
    _, ps = _bound_conv()
    opt = dfl_nn.AdaBelief(name="opt", lr=0.01)
    opt.initialize_variables(ps, vars_on_cpu=True)
    _two_steps(opt, ps)
    od = {k: (v.astype(np.int32) if k == "iters:0" else v)
          for k, v in cv.convert_optimizer_state_torch_to_official(opt).items()}
    assert np.dtype(od["iters:0"].dtype) == np.int32
    _, ps2 = _bound_conv()
    opt2 = dfl_nn.AdaBelief(name="opt2", lr=0.01)
    opt2.initialize_variables(ps2, vars_on_cpu=True)
    rep = cv.convert_optimizer_state_official_to_torch(opt2, od, component="opt2")
    assert rep.result == "PASS"
    iters_map = [m for m in rep.mapped if m.destination_name == "iters:0"][0]
    assert iters_map.rule == "iters_int_widening"
    assert int(opt2.iterations) == 2


def test_optimizer_state_strict_failures():
    init_cpu()
    _, ps = _bound_conv()
    opt = dfl_nn.AdaBelief(name="opt", lr=0.01)
    opt.initialize_variables(ps, vars_on_cpu=True)
    _two_steps(opt, ps)
    od = cv.convert_optimizer_state_torch_to_official(opt)

    _, ps2 = _bound_conv()
    opt2 = dfl_nn.AdaBelief(name="opt2", lr=0.01)
    opt2.initialize_variables(ps2, vars_on_cpu=True)

    def _load(d):
        with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
            cv.convert_optimizer_state_official_to_torch(opt2, d, component="x")
        return ei.value.args[0]

    # missing iteration state
    t = _load({k: v for k, v in od.items() if k != "iters:0"})
    assert "MISSING_REQUIRED_STATE" in t and "iters:0" in t
    # missing one state (no silent optimizer reset)
    t = _load({k: v for k, v in od.items() if k != "vs_encoder/weight_0:0"})
    assert "MISSING_REQUIRED_STATE" in t and "vs_encoder/weight_0:0" in t
    # unexpected extra state (no silent ignore)
    t = _load({**od, "zz_w_0:0": np.zeros((), np.int64)})
    assert "UNEXPECTED_EXTRA_STATE" in t and "zz_w_0:0" in t
    # duplicate ':0' variant forms for one state (never pick one variant
    # silently)
    t = _load({**od, "iters": od["iters:0"]})
    assert "DUPLICATE_MAPPING" in t and "iters" in t
    # a state referencing a variable outside the given saveable
    other = dfl_nn.Conv2D(2, 2, kernel_size=3, padding='SAME', name="other")
    other.build_weights()
    with pytest.raises(dfl_ckpt.CheckpointLoadError) as ei:
        cv.convert_optimizer_state_official_to_torch(opt2, od, saveable=other,
                                                     component="x")
    assert "not part of the given saveable" in ei.value.args[0]


# ---------------------------------------------------------------------------
# Archi / discriminator conversion (Phase 3F components)
# ---------------------------------------------------------------------------

def test_archi_stack_conversion_both_directions():
    init_cpu()
    torch.manual_seed(0)
    archi = dfl_nn.DeepFakeArchi(64, opts="")
    enc = archi.Encoder(in_ch=3, e_ch=4, name="encoder")
    enc.init_weights()
    dec = archi.Decoder(in_ch=4, d_ch=4, d_mask_ch=2, name="decoder_src")
    dec.init_weights()

    # official -> torch (build an identical fresh stack and convert in)
    torch.manual_seed(123)
    archi2 = dfl_nn.DeepFakeArchi(64, opts="")
    enc2 = archi2.Encoder(in_ch=3, e_ch=4, name="encoder")
    enc2.init_weights()
    d_enc = cv.convert_torch_to_official(enc, component="encoder").payload
    rep = cv.convert_official_to_torch(enc2, d_enc, component="encoder")
    assert rep.result == "PASS"
    for (n1, p1), (n2, p2) in zip(enc.named_parameters(), enc2.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), n1
    assert rep.layout_conversion_count > 0  # conv kernels were converted

    # torch -> official -> torch file-level round trip
    tmp = os.path.join(str(REPO_ROOT), "phase4_test_archi_tmp")
    os.makedirs(tmp, exist_ok=True)
    try:
        p = os.path.join(tmp, "encoder.npy")
        cv.write_official_checkpoint(p, d_enc)
        back = cv.read_official_checkpoint(p)
        assert set(back) == set(d_enc)
        assert all(np.array_equal(back[k], d_enc[k]) for k in back)
    finally:
        os.remove(os.path.join(tmp, "encoder.npy"))
        os.rmdir(tmp)


@pytest.mark.parametrize("res,opts", [(64, "td"), (96, "ud"), (256, "d")])
def test_archi_option_combos_keysets_and_conversion(res, opts):
    init_cpu()
    torch.manual_seed(0)
    archi = dfl_nn.DeepFakeArchi(res, opts=opts)
    enc = archi.Encoder(in_ch=3, e_ch=4, name="encoder")
    enc.init_weights()
    keys = [sub for sub, _ in enc._iter_official_weights()]
    assert all(k.endswith(":0") for k in keys)
    # every key converts (official -> identical fresh stack)
    torch.manual_seed(7)
    archi2 = dfl_nn.DeepFakeArchi(res, opts=opts)
    enc2 = archi2.Encoder(in_ch=3, e_ch=4, name="encoder")
    enc2.init_weights()
    d = cv.convert_torch_to_official(enc, component="encoder").payload
    rep = cv.convert_official_to_torch(enc2, d, component="encoder")
    assert rep.result == "PASS"
    assert rep.mapped_state_count == len(keys)


def test_discriminator_conversion():
    init_cpu()
    disc = dfl_nn.PatchDiscriminator(patch_size=8, in_ch=3, name="GAN")
    disc.init_weights()
    d = cv.convert_torch_to_official(disc, component="GAN").payload
    assert any(k.startswith("convs_") for k in d)
    disc2 = dfl_nn.PatchDiscriminator(patch_size=8, in_ch=3, name="GAN")
    disc2.init_weights()
    rep = cv.convert_official_to_torch(disc2, d, component="GAN")
    assert rep.result == "PASS"
    for (n1, p1), (n2, p2) in zip(disc.named_parameters(),
                                   disc2.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), n1

    unet = _PD_mod.UNetPatchDiscriminator(patch_size=16, in_ch=3, base_ch=16,
                                          name="D_src")
    unet.init_weights()
    d2 = cv.convert_torch_to_official(unet, component="D_src").payload
    assert any(k.startswith("upconvs_") for k in d2)
    unet2 = _PD_mod.UNetPatchDiscriminator(patch_size=16, in_ch=3, base_ch=16,
                                           name="D_src")
    unet2.init_weights()
    rep2 = cv.convert_official_to_torch(unet2, d2, component="D_src")
    assert rep2.result == "PASS"

    code = dfl_nn.CodeDiscriminator(4, code_res=4, name="D_code")
    code.init_weights()
    d3 = cv.convert_torch_to_official(code, component="D_code").payload
    code2 = dfl_nn.CodeDiscriminator(4, code_res=4, name="D_code")
    code2.init_weights()
    rep3 = cv.convert_official_to_torch(code2, d3, component="D_code")
    assert rep3.result == "PASS"


# ---------------------------------------------------------------------------
# File-level round trips
# ---------------------------------------------------------------------------

def test_file_level_roundtrip_both_directions(plain_tmp):
    init_cpu()
    torch.manual_seed(0)
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="conv1")
    c.build_weights()
    src_w = unique_value((3, 3, 3, 5))
    src_b = unique_value((1, 1, 1, 5)).reshape(5)  # official 4-D form on file
    cv.convert_official_to_torch(
        c, {"weight:0": src_w, "bias:0": unique_value((1, 1, 1, 5))},
        component="conv1")
    d_out = cv.convert_torch_to_official(c, component="conv1").payload

    p = str(Path(plain_tmp) / "conv1.npy")
    cv.write_official_checkpoint(p, d_out)
    d_back = cv.read_official_checkpoint(p)
    assert set(d_back) == {"weight:0", "bias:0"}
    assert np.array_equal(d_back["weight:0"], d_out["weight:0"])

    # official -> torch -> official: key sets, shapes, dtypes, values
    c2 = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="conv1")
    c2.build_weights()
    rep_in = cv.convert_official_to_torch(c2, d_back, component="conv1")
    assert rep_in.result == "PASS"
    d_out2 = cv.convert_torch_to_official(c2, component="conv1").payload
    assert set(d_out2) == set(d_out)
    for k in d_out:
        assert d_out2[k].shape == d_out[k].shape
        assert np.dtype(d_out2[k].dtype) == np.dtype(d_out[k].dtype)
        assert np.array_equal(d_out2[k], d_out[k]), k
    assert torch.equal(c.weight, c2.weight)
    assert torch.equal(c.bias, c2.bias)


def test_saveable_loader_and_converter_agree(plain_tmp):
    # the Phase 3A strict loader (Saveable.load_weights) and the Phase 4
    # converter must agree on every value for the same file
    init_cpu()
    torch.manual_seed(0)
    c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="conv1")
    c.build_weights()
    p = str(Path(plain_tmp) / "agree.npy")
    c.save_weights(p)

    d = cv.read_official_checkpoint(p)
    c_conv = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="conv1")
    c_conv.build_weights()
    cv.convert_official_to_torch(c_conv, d, component="conv1")

    c_load = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="conv1")
    c_load.build_weights()
    assert c_load.load_weights(p) is True
    for (n1, p1), (n2, p2) in zip(c.named_parameters(), c_conv.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), n1
    for (n1, p1), (n2, p2) in zip(c.named_parameters(), c_load.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), n1


def test_zero_d_iters_survives_numpy2(plain_tmp):
    # the 0-D iters counter through the Phase 3A save/load path under
    # NumPy 2.x (ascontiguousarray 0-D -> (1,) upgrade guarded)
    init_cpu()
    _, ps = _bound_conv()
    opt = dfl_nn.AdaBelief(name="opt", lr=0.01)
    opt.initialize_variables(ps, vars_on_cpu=True)
    _two_steps(opt, ps)
    p = str(Path(plain_tmp) / "opt.npy")
    opt.save_weights(p)
    raw = pickle.loads(Path(p).read_bytes())
    assert np.dtype(raw["iters:0"].dtype) in (np.int32, np.int64)
    assert raw["iters:0"].ndim in (0, 1)

    _, ps2 = _bound_conv()
    opt2 = dfl_nn.AdaBelief(name="opt2", lr=0.01)
    opt2.initialize_variables(ps2, vars_on_cpu=True)
    assert opt2.load_weights(p) is True
    assert int(opt2.iterations) == int(opt.iterations)
    assert all(torch.equal(a, b) for a, b in
               zip(opt.ms_dict.values(), opt2.ms_dict.values()))
    # and the converter path agrees
    _, ps3 = _bound_conv()
    opt3 = dfl_nn.AdaBelief(name="opt3", lr=0.01)
    opt3.initialize_variables(ps3, vars_on_cpu=True)
    rep = cv.convert_optimizer_state_official_to_torch(opt3, raw, component="opt3")
    assert rep.result == "PASS"
    assert int(opt3.iterations) == int(opt.iterations)


# ---------------------------------------------------------------------------
# Device neutrality / GPU
# ---------------------------------------------------------------------------

@requires_gpu
def test_gpu_conversion_bitexact_with_cpu_twin():
    # device-neutral engine: the conversion runs on CPU/NumPy copies and
    # copies onto the parameter device — bit-exact against the CPU twin
    init_cpu()
    torch.manual_seed(0)
    c_ref = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
    c_ref.build_weights()
    c_ref.init_weights()
    init_gpu()
    c_gpu = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
    c_gpu.build_weights()
    d = cv.convert_torch_to_official(c_ref, component="c").payload
    rep = cv.convert_official_to_torch(c_gpu, d, component="c")
    assert rep.result == "PASS"
    assert c_gpu.weight.device.type == "cuda"
    assert torch.equal(c_gpu.weight.cpu(), c_ref.weight.cpu())
    assert torch.equal(c_gpu.bias.cpu(), c_ref.bias.cpu())


# ---------------------------------------------------------------------------
# Boundaries / labels
# ---------------------------------------------------------------------------

def test_no_tf_no_cuda_in_converter():
    src = (REPO_ROOT / "core" / "leras" / "convert.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("tensorflow"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tensorflow"), node.module
    assert "torch.cuda" not in src


def test_coverage_labels_present():
    doc = cv.__doc__ or ""
    # explicit labels: generic EXACT, models NOT_YET_IMPLEMENTED
    assert "EXACT" in doc
    assert "NOT_YET_IMPLEMENTED" in doc
    for model in ("SAEHD", "AMP", "Quick96", "XSeg"):
        assert model in doc
    # the file format is the official pickled dict, not .pth
    assert "official_dfl_pickled_dict_v4" in cv.OFFICIAL_FORMAT
