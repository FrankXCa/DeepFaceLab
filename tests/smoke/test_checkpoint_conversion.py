"""Phase 4 acceptance: checkpoint compatibility and conversion (1/2).

Covers the Phase 4 centralized conversion engine
(``core/leras/convert.py`` — independent reimplementation; concept
sources per docs/PHASE4_PLAN.md items 16-19: official DFL file format
contract (GPL-3.0), EXTERNAL_A strict two-pass load concept
(GPL-3.0), EXTERNAL_B component mapping concepts (unlicensed — NOT
copied)) for the generic-component (weights) direction:

- official file format (a pickled ``dict[str, np.ndarray]``, protocol
  4, keys ``{sub}:0`` — parsed with pickle + NumPy, NO TensorFlow):
  validation against the REAL official artifacts tracked in the
  baseline (``facelib/*.npy`` — official TF-era pickled dicts written
  by official DFL code: S3FD/2DFAN float32 with 4-D singleton-padded
  bias/BN forms, 3DFAN float32 with 1-D forms, FaceEnhancer float16),
  re-pickle protocol-4 round-trip, and rejection of corrupt files
  (truncated pickles, non-dict pickles, real ``np.save``-format
  array files, non-string keys, non-ndarray values);
- name mapping: the pure string transform (torch dotted <-> official
  slashed ``:0``) and the ``:0``-variant tolerance, both directions;
- layout conversion with DECLARED rules only (no element-count
  reshape fallback): Conv2D (kH,kW,in,out)<->(out,in,kH,kW),
  Conv2DTranspose (kH,kW,out,in)<->(in,out,kH,kW), DepthwiseConv2D
  (kH,kW,in,dm)<->(in*dm,1,kH,kW), Dense official-layout identity
  (in,out), and the ``broadcast_squeeze`` rule for 1-D parameters
  (bias / BN weight/bias/running_mean/running_var) accepting the
  2-D/3-D/4-D singleton-padded official forms; layout proofs use
  UNIQUE-VALUE tensors so any wrong axis order is visible (element
  counts are never an identity proof);
- strict two-pass (all-or-nothing) conversion: missing required
  weights, unexpected extra keys, shape mismatch (including
  same-element-count wrong shapes), dtype mismatch (float16 file into
  a float32 module), ambiguous mapping (two sub-names resolving to
  one source key), corrupt values -> ``CheckpointLoadError`` with the
  full structured report; on failure NOTHING is copied;
- reverse export (torch -> official): the official-layout dict via
  the per-layer ``convert_weight_to_official`` hooks (module-tree
  cascading — archi/discriminator files carry official layouts);
  explicit rejection with ``UnsupportedExportError`` (never a silent
  drop / approximation / reshape / coercion) when a state is flagged
  non-exportable;
- archis (canonical option combos) and discriminators converted as
  named Saveables; file-level round-trips through
  write_official_checkpoint / read_official_checkpoint;
  Saveable.load_weights agreement with the converter;
- GPU: device-neutral engine (CPU/NumPy arrays copied onto the
  parameter device), RTX 4090 conversion into a GPU module with
  bit-exact CPU-twin parity (skip on CPU-only environments);
- import boundary: no TensorFlow import and no direct
  ``torch.cuda.*`` in the conversion source (AST).

The next Phase 4 commit (2/2) adds the optimizer-state conversion
(iters / ms_ / vs_ / acc_, value-exact resume equivalence, the
declared int32 -> int64 iteration-counter widening, optimizer
strict-failure paths), the file-level bidirectional round-trip
tests, the 0-D iters counter under NumPy 2.x, and the coverage-label
checks (SAEHD/AMP/Quick96/XSeg full-model checkpoint compatibility:
NOT_YET_IMPLEMENTED, Phases 6-8; facelib extractor models: Phase 9 —
their real files validate the FORMAT contract here).

Parity labels: EXACT for the value/layout/key/dtype round-trips
(pure index rearrangement + value copy — ``torch.equal`` /
``np.array_equal``; the GPU copy is bit-exact); the real facelib
files validate the FORMAT contract (EXACT), not extractor-model
compatibility.
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


def test_broadcast_squeeze_forms():
    init_cpu()
    for form in ((5,), (1, 5), (1, 1, 5), (1, 1, 1, 5)):
        c = dfl_nn.Conv2D(3, 5, kernel_size=3, padding='SAME', name="c")
        c.build_weights()
        src = unique_value((3, 3, 3, 5))
        bias = unique_value((5,)).reshape(form)
        rep = cv.convert_official_to_torch(
            c, {"weight:0": src, "bias:0": bias}, component="c")
        assert rep.result == "PASS", form
        rule = [m for m in rep.mapped if m.destination_name == "bias:0"][0].rule
        expected = "identity" if form == (5,) else "broadcast_squeeze"
        assert rule == expected, (form, rule)
        assert torch.equal(c.bias, torch.from_numpy(unique_value((5,))))


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
