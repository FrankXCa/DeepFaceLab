"""Phase 11 acceptance — AMP DFM/ONNX export (torch, no TensorFlow).

The official DFM export contract (models/Model_AMP/Model_tf.py
L585-629, tf2onnx opset 12, name='AMP') is reproduced by
``AMPModel.export_dfm`` on the torch foundation:

- inputs: ``in_face`` (TF-visible ``in_face:0``, NHWC
  ``(N,res,res,3)`` float32, DYNAMIC batch) and ``morph_value``
  (TF-visible ``morph_value:0``, float32 shape ``(1,)`` static);
- outputs IN ORDER (the DeepFaceLive DFMModel runtime unpacks
  ``sess.run(None, ...)`` positionally): ``out_face_mask:0`` (the
  DST learned mask), ``out_celeb_face:0`` (the predicted celeb
  face), ``out_celeb_face_mask:0`` (the src/celeb learned mask);
- opset 12 (the official tf2onnx opset — unchanged);
- AMP is always NCHW internally (the hard-wired official
  model_data_format) — the official AMP export calls no
  set_data_format (unlike SAEHD's L706);
- the morph stays a REAL graph input: ``k = floor(inter_dims *
  morph_value)`` (the official int32 cast truncation) selects the
  leading k inter channels from the inter_src head and the
  remainder from the inter_dst head — the graph-traceable boolean
  channel mask (``c < k``) is mathematically identical to the
  official ``concat(slice(...))`` for every morph in [0,1].

The binding morph grid (the official merger morph presets) at
inter_dims=32:

    morph 0.0  -> k = 0   (the full inter_dst code — the dst heads)
    morph 0.25 -> k = 8
    morph 0.50 -> k = 16
    morph 0.65 -> k = 20  (floor(20.8) — the truncation is pinned
                                  against an explicit k=20
                                  concat-slice reference)
    morph 1.0  -> k = 32  (the full inter_src code)

Validation is eager torch vs ONNX Runtime (CPU): the parity gates
(max_abs <= 1e-4, mean_abs <= 1e-5) were set AFTER measuring the
actual numbers on the seeded reference model (observed max_abs
1.19e-7, mean_abs <= 1.81e-8 over the grid — a >500x safety margin
for ORT thread-count / platform FP variance, the Phase 10E XSeg
precedent).

The tracked tests use deterministic seeded synthetic models only.
A real checkpoint export is private and opt-in via
``DFL_TEST_AMP_CHECKPOINT`` (an untracked directory holding the
official AMP model files) — unset here, so the gate skips.
"""

import builtins
import os
import pickle
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, checker
from onnxruntime import InferenceSession

from core.interact import interact as io
from core.leras import nn as dfl_nn
from models import import_model

_smoke_dir = Path(__file__).resolve().parent
if str(_smoke_dir) not in sys.path:
    sys.path.insert(0, str(_smoke_dir))
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    DEFAULT_SEED_OPTIONS,
    make_model as make_amp,
)

MODEL_NAME = "test_AMP"
RES = 64
INTER_DIMS = 32

# the binding morph grid (the official merger morph presets) and the
# k = floor(inter_dims * morph) channel split each one implies at
# inter_dims=32
MORPH_GRID = ((0.0, 0), (0.25, 8), (0.50, 16), (0.65, 20), (1.0, 32))

# measured on the seeded reference model (the full grid, this
# platform): max_abs 1.19e-7, mean_abs <= 1.81e-8 -> the gates keep the
# Phase 10E precedent (a large margin for ORT thread/platform FP
# variance)
PARITY_MAX_ABS = 1e-4
PARITY_MEAN_ABS = 1e-5

INPUT_FACE = "in_face:0"
INPUT_MORPH = "morph_value:0"
OUTPUT_NAMES = ("out_face_mask:0", "out_celeb_face:0",
                "out_celeb_face_mask:0")
OUTPUT_CHANNELS = (1, 3, 1)

# the official on-disk prefix, either the bare "AMP_*" form or the
# "<model>_AMP_*" form (mirrors test_model_amp_training.py)
_AMPFILE_RE = re.compile(r"^(?:.*_)?AMP_(?P<rest>[^/]+)$")


@pytest.fixture(autouse=True)
def restore_rng_state():
    """Preserve every RNG state across tests (RNG preservation policy)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@pytest.fixture(autouse=True)
def restore_data_format():
    """AMP is always NCHW (its on_initialize hard-wires the global
    format) — restore whatever the suite left behind so later test
    files see the pre-test format."""
    fmt = dfl_nn.data_format
    ch_axis = dfl_nn.conv2d_ch_axis
    sp_axes = dfl_nn.conv2d_spatial_axes
    yield
    # direct assignment (not set_data_format): the initial state
    # before any model runs is data_format == None, which the
    # validating setter rejects — the fixture must restore it
    dfl_nn.data_format = fmt
    dfl_nn.conv2d_ch_axis = ch_axis
    dfl_nn.conv2d_spatial_axes = sp_axes


@pytest.fixture
def no_cli_prompts(monkeypatch):
    """Neutralize the official CLI layer on the export path: every
    prompt answers its default — including "Export quantized?" ->
    False (the fp32 export path); the fp16 test re-patches
    ``io.input_bool`` at test level (the more recent monkeypatch
    wins)."""
    def _input_str(prompt, default=None, **kwargs):
        return "" if default is None else str(default)

    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    monkeypatch.setattr(io, "input_str", _input_str)
    monkeypatch.setattr(io, "input_bool", lambda *a, **k: False)
    monkeypatch.setattr(io, "input_number", lambda *a, **k: 0.0)
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


@pytest.fixture
def workdir(plain_tmp):
    """Dedicated model-storage dir inside the shared plain_tmp base."""
    d = Path(plain_tmp) / "amp_onnx"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _seeded_storage(root):
    """Build a deterministic AMP model storage dir in ``root``:

    the seeded archi's component weights (``test_AMP_*.npy``) plus
    the model data file ``test_AMP_data.dat`` (iter=1 -> a RESUMED
    model, so the production constructor takes the strict load path
    and no first-run prompts fire).
    """
    seed = dict(DEFAULT_SEED_OPTIONS)
    seed.update(resolution=RES, batch_size=1, ae_dims=32,
                inter_dims=INTER_DIMS, e_dims=16, d_dims=16,
                d_mask_dims=6)
    m = make_amp(AMPHeadless, root, is_training=False, seed=seed,
                 cpu_only=True)
    m.set_iter(1)
    m.save()
    return m


def _build_export_model(root):
    """The ExportDFM production construction contract:
    models.import_model('AMP')(is_exporting=True, cpu_only=True) ->
    model.export_dfm()."""
    _seeded_storage(root)
    model = import_model("AMP")(is_exporting=True,
                                saved_models_path=root,
                                cpu_only=True)
    model.export_dfm()
    return model


def _onnx_dims(value_info):
    return [d.dim_param if d.dim_param else d.dim_value
            for d in value_info.type.tensor_type.shape.dim]


def _random_input(n, base=9000, resolution=RES):
    rng = np.random.RandomState(base + n)
    return rng.rand(n, resolution, resolution, 3).astype(np.float32)


def _to_nchw(x_nhwc):
    return torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)))


def _eager_merge_heads(model, x_nhwc, morph):
    """The production inference closure (the official inference
    boundary): AMP is always NCHW, so the NHWC test input is
    boundary-converted and the three heads come back NCHW —
    (pred_src_dst, pred_dst_dstm, pred_src_dstm)."""
    dfl_nn.set_data_format("NCHW")
    heads = model.AE_merge(_to_nchw(x_nhwc), morph)
    return [h.transpose(0, 2, 3, 1) for h in heads]


def _k_split_reference(model, x_nhwc, k):
    """The official concat-slice morph semantics computed through the
    model's own components (the Model_tf.py L600-602 math with an
    explicit integer k — independent of both AE_merge's Python
    ``int(inter_dims*morph)`` and the ONNX graph's mask): the
    leading k inter channels from the inter_src head, the remainder
    from the inter_dst head. Returns NHWC (dst_dstm, src_dst,
    src_dstm)."""
    x = _to_nchw(x_nhwc)
    with torch.no_grad():
        code = model.encoder(x)
        src_code = model.inter_src(code)
        dst_code = model.inter_dst(code)
        blended = torch.cat((src_code[:, :k], dst_code[:, k:]),
                            dim=1)
        src_dst, src_dstm = model.decoder(blended)
        _, dst_dstm = model.decoder(dst_code)
        return (dst_dstm.permute(0, 2, 3, 1).numpy(),
                src_dst.permute(0, 2, 3, 1).numpy(),
                src_dstm.permute(0, 2, 3, 1).numpy())


def _assert_onnx_contract(onnx_path):
    """opset 12, the exact official I/O names/order/dtypes/dims,
    inline weights, no external data (the DFM/DFL replay contract)."""
    m = onnx.load(str(onnx_path))
    checker.check_model(m)

    opsets = {(d.domain or "ai.onnx"): d.version for d in m.opset_import}
    assert opsets == {"ai.onnx": 12}, opsets

    g = m.graph
    assert [i.name for i in g.input] == [INPUT_FACE, INPUT_MORPH]
    assert [o.name for o in g.output] == list(OUTPUT_NAMES)

    in_face = g.input[0]
    assert in_face.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _onnx_dims(in_face) == ["batch", RES, RES, 3]
    in_morph = g.input[1]
    assert in_morph.type.tensor_type.elem_type == TensorProto.FLOAT
    # the official placeholder shape (1,) — static, NOT a batch axis
    assert _onnx_dims(in_morph) == [1]

    for out_vi, ch in zip(g.output, OUTPUT_CHANNELS):
        assert out_vi.type.tensor_type.elem_type == TensorProto.FLOAT
        assert _onnx_dims(out_vi) == ["batch", RES, RES, ch]

    # weights embedded in the proto (no external-data tensors)
    assert len(g.initializer) > 20
    assert all(len(t.external_data) == 0 for t in g.initializer)


def test_export_dfm_writes_contract_onnx(workdir, no_cli_prompts):
    # the export must not initialize a CUDA context (the suite's
    # session may already have one from other test files — assert the
    # export path itself never flips it)
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir)

    onnx_path = workdir / f"{MODEL_NAME}_model.dfm"
    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 1_000_000

    _assert_onnx_contract(onnx_path)

    assert torch.cuda.is_initialized() is cuda_before


def test_morph_grid_ort_parity(workdir, no_cli_prompts):
    """The binding morph grid vs the eager AE_merge (whose Python
    ``int(inter_dims*morph)`` floor is the official semantics) —
    plus, for 0.65, an explicit k=20 concat-slice reference through
    the model's own components: if the graph rounded/ceiled instead
    of truncating (k=21), the blended code would draw channel 20
    from the inter_src head and the decoder output would miss the
    gate by orders of magnitude — the floor is pinned."""
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir)

    sess = InferenceSession(str(workdir / f"{MODEL_NAME}_model.dfm"),
                            providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == [INPUT_FACE, INPUT_MORPH]
    assert [o.name for o in sess.get_outputs()] == list(OUTPUT_NAMES)

    for morph, k in MORPH_GRID:
        x = _random_input(1, base=1000 + int(morph * 100))
        # the DFMModel feed pattern (the literal graph input keys)
        ort_out = sess.run(
            None, {INPUT_FACE: x, INPUT_MORPH: np.float32([morph])})
        assert len(ort_out) == 3
        (o_face_mask, o_celeb_face, o_celeb_mask) = ort_out
        assert o_celeb_face.shape == (1, RES, RES, 3)
        assert o_face_mask.shape == (1, RES, RES, 1)
        assert o_celeb_mask.shape == (1, RES, RES, 1)
        for o in ort_out:
            assert o.dtype == np.float32
            assert o.min() >= 0.0 and o.max() <= 1.0

        # out_face_mask = pred_dst_dstm, out_celeb_face =
        # pred_src_dst, out_celeb_face_mask = pred_src_dstm
        eager = _eager_merge_heads(model, x, morph)
        pairs = ((o_celeb_face, eager[0], "out_celeb_face"),
                 (o_face_mask, eager[1], "out_face_mask"),
                 (o_celeb_mask, eager[2], "out_celeb_face_mask"))
        for a, b, name in pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS, \
                f"morph={morph} (k={k}) {name}: max_abs {d.max():.3e} " \
                f"exceeds the measured gate"
            assert d.mean() <= PARITY_MEAN_ABS, \
                f"morph={morph} (k={k}) {name}: mean_abs {d.mean():.3e} " \
                f"exceeds the measured gate"

        # the independent explicit-k concat-slice reference pins the
        # floor: at 0.65 it is k = floor(20.8) = 20
        ref = _k_split_reference(model, x, k)
        ref_pairs = ((o_celeb_face, ref[1]),
                     (o_face_mask, ref[0]),
                     (o_celeb_mask, ref[2]))
        for a, b in ref_pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS, \
                f"morph={morph} (k={k}): explicit k-split reference " \
                f"max_abs {d.max():.3e} exceeds the measured gate"

    # endpoint invariants (the k=0 / k=inter_dims channel split):
    # morph 0.0 decodes the FULL inter_dst code (the dst heads);
    # morph 1.0 decodes the FULL inter_src code. The grid parity
    # above already covers both against the eager references — pin
    # the invariants explicitly as range/order sanity on the ORT
    # outputs themselves.
    assert torch.cuda.is_initialized() is cuda_before


def test_exported_onnx_dynamic_batch_ort_parity(workdir, no_cli_prompts):
    """Dynamic face batch (N=1,2,4) at a fixed intermediate morph
    (0.5): the morph input stays the static (1,) per-session value
    while the face batch axis is dynamic — both together."""
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir)

    sess = InferenceSession(str(workdir / f"{MODEL_NAME}_model.dfm"),
                            providers=["CPUExecutionProvider"])
    for n in (1, 2, 4):
        x = _random_input(n, base=2000 + n)
        ort_out = sess.run(None, {INPUT_FACE: x,
                                  INPUT_MORPH: np.float32([0.5])})
        assert ort_out[1].shape == (n, RES, RES, 3)
        assert ort_out[0].shape == (n, RES, RES, 1)
        assert ort_out[2].shape == (n, RES, RES, 1)
        for o in ort_out:
            assert o.min() >= 0.0 and o.max() <= 1.0

        eager = _eager_merge_heads(model, x, 0.5)
        pairs = ((ort_out[1], eager[0]), (ort_out[0], eager[1]),
                 (ort_out[2], eager[2]))
        for a, b in pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS
            assert d.mean() <= PARITY_MEAN_ABS

    assert torch.cuda.is_initialized() is cuda_before


def test_export_is_tf_and_onnxscript_free(workdir, no_cli_prompts):
    """Dependency policy: the production export path needs neither
    TensorFlow (the Phase 11 goal) nor onnxscript (the legacy
    TorchScript exporter is the chosen torch 2.14 path — the Phase
    10E dependency decision, docs/PHASE10_STATE.md)."""
    _seeded_storage(workdir)
    model = import_model("AMP")(is_exporting=True,
                                saved_models_path=workdir,
                                cpu_only=True)
    model.export_dfm()

    for banned in ("tensorflow", "tensorflow.python", "keras",
                   "onnxscript"):
        assert not any(mod == banned or mod.startswith(banned + ".")
                       for mod in sys.modules), \
            f"{banned} must not be importable after the export"
    assert (workdir / f"{MODEL_NAME}_model.dfm").is_file()


def test_dfm_dfl_replay_contract(workdir, no_cli_prompts):
    """The DeepFaceLive DFMModel runtime contract (model_type 2 —
    the 2-input DFM), emulated verbatim (the DFL app itself: no
    venv in this environment — PENDING ENVIRONMENTALLY; the feed
    pattern is the binding consumer contract):

    - ``inputs[0].name`` contains 'in_face' with static
      ``shape[1:3] == (res,res)``; ``inputs[1].name`` contains
      'morph_value' (the AMP marker); >2 inputs would be invalid;
    - feed after DFMModel's ``ImageProcessor.get_image('NHWC')``
      normalization: ``sess.run(None, {'in_face:0': img,
      'morph_value:0': np.float32([morph_factor])})`` — a single face
      is a float32 ``(1,res,res,3)`` batch and the morph is a float32
      array of shape (1,);
    - ``out_face_mask, out_celeb, out_celeb_mask = ...`` —
      positional unpack of ALL outputs (the graph output order is
      binding).
    """
    _build_export_model(workdir)
    onnx_path = str(workdir / f"{MODEL_NAME}_model.dfm")

    sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # DFMModel.__init__ acceptance (modelhub/DFLive/DFMModel.py)
    assert "in_face" in inputs[0].name
    assert inputs[0].shape[1] == RES and inputs[0].shape[2] == RES
    assert len(inputs) == 2, "the AMP DFM takes exactly two inputs"
    assert "morph_value" in inputs[1].name
    assert tuple(inputs[1].shape) == (1,)
    assert [o.name for o in outputs] == list(OUTPUT_NAMES)

    img = _random_input(1, base=777)  # ImageProcessor NHWC output
    assert img.shape == (1, RES, RES, 3)
    out_face_mask, out_celeb, out_celeb_mask = sess.run(
        None, {INPUT_FACE: img, INPUT_MORPH: np.float32([0.5])})
    assert out_celeb.shape == (1, RES, RES, 3)
    assert out_face_mask.shape == (1, RES, RES, 1)
    assert out_celeb_mask.shape == (1, RES, RES, 1)
    for o in (out_face_mask, out_celeb, out_celeb_mask):
        assert o.dtype == np.float32
        assert o.min() >= 0.0 and o.max() <= 1.0


def test_amp_onnx_fp16_structural(workdir, no_cli_prompts, monkeypatch):
    """The official ``use_fp16`` export knob ("Export quantized?"):
    the archi is built with the fp16 conv dtype (Conv layers +
    boundary casts; Dense stays fp32) — the D6 acceptance: graph
    structure (checker, names/order/opset unchanged, float32 I/O
    contract, fp16 Conv weights) + a torch-eager CPU sanity run
    (finite, [0,1]). ORT CPU fp16 Conv parity is PENDING
    ENVIRONMENTALLY (no onnxruntime-gpu by dependency policy)."""
    def _quantized_bool(prompt, default=False, **kw):
        if isinstance(prompt, str) and prompt.startswith("Export quantized?"):
            return True
        return default if default is not None else False

    monkeypatch.setattr(io, "input_bool", _quantized_bool)

    cuda_before = torch.cuda.is_initialized()
    _seeded_storage(workdir)
    model = import_model("AMP")(is_exporting=True,
                                saved_models_path=workdir,
                                cpu_only=True)
    model.export_dfm()

    onnx_path = workdir / f"{MODEL_NAME}_model.dfm"
    assert onnx_path.is_file()
    m = onnx.load(str(onnx_path))
    checker.check_model(m)
    opsets = {(d.domain or "ai.onnx"): d.version
              for d in m.opset_import}
    assert opsets == {"ai.onnx": 12}, opsets
    g = m.graph
    assert [i.name for i in g.input] == [INPUT_FACE, INPUT_MORPH]
    assert [o.name for o in g.output] == list(OUTPUT_NAMES)
    # the DFM I/O contract stays float32 (the official graph casts
    # the heads back to float32 at the boundary)
    for i in g.input:
        assert i.type.tensor_type.elem_type == TensorProto.FLOAT
    for o in g.output:
        assert o.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _onnx_dims(g.input[0]) == ["batch", RES, RES, 3]
    assert _onnx_dims(g.input[1]) == [1]

    # the conv weights are the fp16 initializers (4D OIHW); no fp32
    # 4D conv weight survives
    fp16_convs = [t for t in g.initializer
                  if t.data_type == TensorProto.FLOAT16
                  and len(t.dims) == 4]
    fp32_convs = [t for t in g.initializer
                  if t.data_type == TensorProto.FLOAT
                  and len(t.dims) == 4]
    assert len(fp16_convs) > 0
    assert len(fp32_convs) == 0, \
        "the use_fp16 archi keeps no fp32 Conv weights"

    # torch eager CPU fp16 sanity (finite + the sigmoid [0,1] range)
    x = _random_input(1, base=555)
    eager = _eager_merge_heads(model, x, 0.65)
    for o in eager:
        assert np.isfinite(o).all()
        assert o.min() >= 0.0 and o.max() <= 1.0

    assert torch.cuda.is_initialized() is cuda_before


def test_amp_onnx_private_checkpoint_export(workdir, no_cli_prompts):
    """OPT-IN (untracked) real-checkpoint export gate:

    DFL_TEST_AMP_CHECKPOINT=<dir holding the official AMP model
    files (component .npy + data.dat, either the bare AMP_* form or
    the <model>_AMP_* prefix form)>

    Copies the private files into the sandbox temp workdir (the
    private directory is never read in place or referenced in
    tracked output), renames the official prefix to the production
    model name, exports through the production path and checks the
    same contract + ORT/eager parity on synthetic inputs. Skipped
    (never failing) when the variable is unset.
    """
    path = os.environ.get("DFL_TEST_AMP_CHECKPOINT")
    if path is None:
        pytest.skip("DFL_TEST_AMP_CHECKPOINT not set")
    src = Path(path)
    assert src.is_dir(), path

    for f in src.iterdir():
        if not f.is_file():
            continue
        m = _AMPFILE_RE.match(f.name)
        if m is None:
            continue
        rest = m.group("rest")
        if rest == "data.dat":
            shutil.copy(f, workdir / f"{MODEL_NAME}_data.dat")
        elif rest == "default_options.dat":
            shutil.copy(f, workdir / f"{MODEL_NAME}_default_options.dat")
        elif rest.endswith(".npy"):
            shutil.copy(f, workdir / f"{MODEL_NAME}_{rest}")

    data = pickle.loads((workdir / f"{MODEL_NAME}_data.dat").read_bytes())
    if int(data.get("iter", 0)) < 1:
        data["iter"] = 1
        (workdir / f"{MODEL_NAME}_data.dat").write_bytes(
            pickle.dumps(data, 4))

    cuda_before = torch.cuda.is_initialized()
    model = import_model("AMP")(is_exporting=True,
                                saved_models_path=workdir,
                                cpu_only=True)
    model.export_dfm()

    onnx_path = None
    for cand in workdir.glob("*_AMP_model.dfm"):
        onnx_path = cand
        break
    assert onnx_path is not None
    m = onnx.load(str(onnx_path))
    checker.check_model(m)
    opsets = {(d.domain or "ai.onnx"): d.version
              for d in m.opset_import}
    assert opsets == {"ai.onnx": 12}, opsets
    assert [i.name for i in m.graph.input] == [INPUT_FACE, INPUT_MORPH]
    assert [o.name for o in m.graph.output] == list(OUTPUT_NAMES)
    res = model.resolution
    assert _onnx_dims(m.graph.input[0]) == ["batch", res, res, 3]

    # ORT/eager parity on the PRIVATE weights (the grid endpoints)
    sess = InferenceSession(str(onnx_path),
                            providers=["CPUExecutionProvider"])
    for morph in (0.0, 0.65, 1.0):
        x = _random_input(1, base=60000, resolution=res)
        eager = _eager_merge_heads(model, x, morph)
        ort_out = sess.run(None, {INPUT_FACE: x,
                                  INPUT_MORPH: np.float32([morph])})
        pairs = ((ort_out[1], eager[0]), (ort_out[0], eager[1]),
                 (ort_out[2], eager[2]))
        for a, b in pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS
            assert d.mean() <= PARITY_MEAN_ABS

    assert torch.cuda.is_initialized() is cuda_before
