"""Phase 11 acceptance — SAEHD DFM/ONNX export (torch, no TensorFlow).

The official DFM export contract (models/Model_SAEHD/Model_tf.py
L700-750, tf2onnx opset 12, name='SAEHD') is reproduced by
``SAEHDModel.export_dfm`` on the torch foundation:

- input ``in_face`` (TF-visible ``in_face:0``): NHWC ``(N,res,res,3)``
  float32 with a DYNAMIC batch axis (N=1 and N>1 both run);
- outputs IN ORDER (the DeepFaceLive DFMModel runtime unpacks
  ``sess.run(None, ...)`` positionally):
  ``out_face_mask:0`` (the DST learned mask), ``out_celeb_face:0``
  (the predicted celeb face), ``out_celeb_face_mask:0`` (the
  src/celeb learned mask) — all NHWC float32, dynamic batch;
- opset 12 (the official tf2onnx opset — unchanged; the XSeg model
  keeps its separate Phase 10 opset 13);
- both archi bases: ``df`` (encoder/inter/decoder_src/decoder_dst)
  and ``liae`` (encoder/inter_B/inter_AB/decoder);
- the official ``nn.set_data_format('NCHW')`` inside export_dfm
  (Model_tf.py L706 — the SAEHD on_initialize rule has no
  is_exporting clause);
- the official ``use_fp16`` export knob ("Export quantized?") is the
  archi conv-dtype construction (Conv layers fp16 + boundary casts;
  Dense stays fp32) — NOT the Phase 8 autocast policy.

Validation here is eager torch vs ONNX Runtime (CPU): the parity
gates (max_abs <= 1e-4, mean_abs <= 1e-5) were set AFTER measuring
the actual numbers on the seeded reference models (observed
max_abs 1.19e-7, mean_abs <= 1.91e-8 for N=1..4 — a >500x safety
margin for ORT thread-count / platform FP variance, the Phase 10E
XSeg precedent).

DeepFaceLive replay: ``DFMModel`` (the local DeepFaceLive install
sibling to this repository) loads the .dfm through ONNX Runtime and
normalizes a single face through ``ImageProcessor.get_image('NHWC')``
to a one-image NHWC float32 batch before feeding the literal graph
input key ``sess.run(None, {'in_face:0': img})``, unpacking the outputs
positionally —
``test_dfm_dfl_replay_contract`` emulates that acceptance + feed
verbatim (the full DFL GUI app has no venv in this environment —
PENDING ENVIRONMENTALLY, the feed-pattern emulation is the accepted
substitute).

The tracked tests use deterministic seeded synthetic models only.
A real checkpoint export is private and opt-in via
``DFL_TEST_SAEHD_CHECKPOINT`` (an untracked directory holding the
official SAEHD model files) — unset here, so the gate skips.
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
from Model_SAEHDTest.Model import (  # noqa: E402
    DEFAULT_SEED_OPTIONS,
    SAEHDHeadless,
    make_model as make_saehd,
)

MODEL_NAME = "test_SAEHD"
RES = 64

# the official on-disk prefix, either the bare "SAEHD_*" form or the
# "<model>_SAEHD_*" form (mirrors test_model_saehd.py)
_SAEHDFILE_RE = re.compile(r"^(?:.*_)?SAEHD_(?P<rest>[^/]+)$")

# measured on the seeded reference models (N=1..4, this platform):
# max_abs 1.19e-7, mean_abs <= 1.91e-8 -> the gates keep the Phase 10E
# precedent (a large margin for ORT thread/platform FP variance)
PARITY_MAX_ABS = 1e-4
PARITY_MEAN_ABS = 1e-5

INPUT_NAME = "in_face:0"
OUTPUT_NAMES = ("out_face_mask:0", "out_celeb_face:0",
                "out_celeb_face_mask:0")
OUTPUT_CHANNELS = (1, 3, 1)


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
    """export_dfm (SAEHD) switches the global leras data format to
    NCHW (the official L706 behavior) — restore whatever the suite
    left behind so later test files see the pre-test format."""
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
    d = Path(plain_tmp) / "saehd_onnx"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _seeded_storage(root, archi):
    """Build a deterministic SAEHD model storage dir in ``root``:

    the seeded archi's component weights (``test_SAEHD_*.npy``) plus
    the model data file ``test_SAEHD_data.dat`` (iter=1 -> a RESUMED
    model, so the production constructor takes the strict load path
    and no first-run prompts fire).
    """
    seed = dict(DEFAULT_SEED_OPTIONS)
    seed.update(resolution=RES, archi=archi, batch_size=1)
    m = make_saehd(SAEHDHeadless, root, is_training=False, seed=seed,
                   cpu_only=True)
    m.set_iter(1)
    m.save()
    return m


def _build_export_model(root, archi):
    """The ExportDFM production construction contract:
    models.import_model('SAEHD')(is_exporting=True, cpu_only=True) ->
    model.export_dfm()."""
    _seeded_storage(root, archi)
    model = import_model("SAEHD")(is_exporting=True,
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


def _assert_onnx_contract(onnx_path, archi):
    """opset 12, the exact official I/O names/order/dtypes/dims,
    inline weights, no external data (the DFM/DFL replay contract)."""
    m = onnx.load(str(onnx_path))
    checker.check_model(m)

    opsets = {(d.domain or "ai.onnx"): d.version for d in m.opset_import}
    assert opsets == {"ai.onnx": 12}, opsets

    g = m.graph
    assert [i.name for i in g.input] == [INPUT_NAME]
    assert [o.name for o in g.output] == list(OUTPUT_NAMES)

    in_vi = g.input[0]
    assert in_vi.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _onnx_dims(in_vi) == ["batch", RES, RES, 3]

    for out_vi, ch in zip(g.output, OUTPUT_CHANNELS):
        assert out_vi.type.tensor_type.elem_type == TensorProto.FLOAT
        assert _onnx_dims(out_vi) == ["batch", RES, RES, ch]

    # weights embedded in the proto (no external-data tensors)
    assert len(g.initializer) > 20
    assert all(len(t.external_data) == 0 for t in g.initializer)


def _eager_merge_heads(model, x_nhwc):
    """The production inference closure (the official inference
    boundary) on the model as left by export_dfm: the SAEHD
    on-CPU model_data_format is NHWC, so the global format is
    restored to NHWC first and the three heads come back NHWC —
    (pred_src_dst, pred_dst_dstm, pred_src_dstm)."""
    dfl_nn.set_data_format("NHWC")
    return model.AE_merge(x_nhwc)


@pytest.mark.parametrize("archi", ["df", "liae-ud"])
def test_export_dfm_writes_contract_onnx(workdir, no_cli_prompts, archi):
    # the export must not initialize a CUDA context (the suite's
    # session may already have one from other test files — assert the
    # export path itself never flips it)
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir, archi)

    onnx_path = workdir / f"{MODEL_NAME}_model.dfm"
    assert onnx_path.is_file()
    size = onnx_path.stat().st_size
    # the weights are embedded in the proto (no external data) — the
    # seeded 64px archis alone are multi-MB on disk
    assert size > 1_000_000

    _assert_onnx_contract(onnx_path, archi)

    # the export ran on the CPU net (cpu_only construction)
    assert torch.cuda.is_initialized() is cuda_before


@pytest.mark.parametrize("archi", ["df", "liae-ud"])
def test_exported_onnx_dynamic_batch_ort_parity(workdir, no_cli_prompts,
                                                archi):
    """Dynamic batch (N=1 and N>1) with eager-torch vs ONNX Runtime
    parity on the seeded reference model (CPU, both sides)."""
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir, archi)

    sess = InferenceSession(str(workdir / f"{MODEL_NAME}_model.dfm"),
                            providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == [INPUT_NAME]
    assert [o.name for o in sess.get_outputs()] == list(OUTPUT_NAMES)

    for n in (1, 2, 4):
        x = _random_input(n)
        # the official output order vs the AE_merge head order:
        # out_face_mask = pred_dst_dstm, out_celeb_face =
        # pred_src_dst, out_celeb_face_mask = pred_src_dstm
        eager = _eager_merge_heads(model, x)
        assert len(eager) == 3
        (e_src_dst, e_dst_dstm, e_src_dstm) = eager
        assert e_src_dst.shape == (n, RES, RES, 3)
        assert e_dst_dstm.shape == (n, RES, RES, 1)
        assert e_src_dstm.shape == (n, RES, RES, 1)

        ort_out = sess.run(None, {INPUT_NAME: x})
        assert len(ort_out) == 3
        (o_face_mask, o_celeb_face, o_celeb_mask) = ort_out
        assert o_celeb_face.shape == (n, RES, RES, 3)
        assert o_face_mask.shape == (n, RES, RES, 1)
        assert o_celeb_mask.shape == (n, RES, RES, 1)
        for o in ort_out:
            assert o.dtype == np.float32

        pairs = ((o_celeb_face, e_src_dst, "out_celeb_face"),
                 (o_face_mask, e_dst_dstm, "out_face_mask"),
                 (o_celeb_mask, e_src_dstm, "out_celeb_face_mask"))
        for a, b, name in pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS, \
                f"N={n} {archi} {name}: max_abs {d.max():.3e} " \
                f"exceeds the measured gate"
            assert d.mean() <= PARITY_MEAN_ABS, \
                f"N={n} {archi} {name}: mean_abs {d.mean():.3e} " \
                f"exceeds the measured gate"

        # the sigmoid heads keep every output in [0,1] (the masks are
        # probabilities, NOT logits — both sides agree in range)
        for o in ort_out:
            assert o.min() >= 0.0 and o.max() <= 1.0

    # the export + ORT session never initialized a CUDA context
    assert torch.cuda.is_initialized() is cuda_before


def test_export_is_tf_and_onnxscript_free(workdir, no_cli_prompts):
    """Dependency policy: the production export path needs neither
    TensorFlow (the Phase 11 goal) nor onnxscript (the legacy
    TorchScript exporter is the chosen torch 2.14 path — the Phase
    10E dependency decision, docs/PHASE10_STATE.md)."""
    _seeded_storage(workdir, "df")
    model = import_model("SAEHD")(is_exporting=True,
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
    """The DeepFaceLive DFMModel runtime contract, emulated verbatim
    (DFL app itself: no venv in this environment — PENDING
    ENVIRONMENTALLY; the feed pattern is the binding consumer
    contract):

    - ``inputs[0].name`` contains 'in_face' and
      ``inputs[0].shape[1:3]`` are the static (res,res) ints;
    - exactly 1 input (SAEHD — a 2nd input must carry 'morph_value',
      AMP's model_type 2);
    - feed: ``sess.run(None, {'in_face:0': img})`` after DFMModel's
      ``ImageProcessor.get_image('NHWC')`` normalization, which makes
      a single face a float32 ``(1,res,res,3)`` batch;
    - ``out_face_mask, out_celeb, out_celeb_mask = ...`` — positional
      unpack of ALL outputs (the graph output order is binding).
    """
    model = _build_export_model(workdir, "df")
    onnx_path = str(workdir / f"{MODEL_NAME}_model.dfm")

    sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # DFMModel.__init__ acceptance (modelhub/DFLive/DFMModel.py)
    assert "in_face" in inputs[0].name
    assert inputs[0].shape[1] == RES and inputs[0].shape[2] == RES
    assert len(inputs) == 1, "SAEHD DFM takes exactly one input"
    assert [o.name for o in outputs] == list(OUTPUT_NAMES)

    img = _random_input(1, base=777)  # ImageProcessor NHWC output
    assert img.shape == (1, RES, RES, 3)
    out_face_mask, out_celeb, out_celeb_mask = sess.run(
        None, {INPUT_NAME: img})
    assert out_celeb.shape == (1, RES, RES, 3)
    assert out_face_mask.shape == (1, RES, RES, 1)
    assert out_celeb_mask.shape == (1, RES, RES, 1)
    for o in (out_face_mask, out_celeb, out_celeb_mask):
        assert o.dtype == np.float32
        assert o.min() >= 0.0 and o.max() <= 1.0


def test_saehd_onnx_fp16_structural(workdir, no_cli_prompts, monkeypatch):
    """The official ``use_fp16`` export knob ("Export quantized?"):
    the archi is built with the fp16 conv dtype (Conv layers +
    boundary casts; Dense stays fp32) — the D6 acceptance: graph
    structure (checker, names/order/opset unchanged, float32 I/O
    contract, fp16 Conv weights) + a torch-eager CPU sanity run
    (finite, [0,1]). ORT CPU fp16 Conv parity is PENDING
    ENVIRONMENTALLY (no onnxruntime-gpu by dependency policy)."""
    # "Export quantized?" answered True, every other bool prompt
    # stays False (the fixture's default)
    def _quantized_bool(prompt, default=False, **kw):
        if isinstance(prompt, str) and prompt.startswith("Export quantized?"):
            return True
        return default if default is not None else False

    monkeypatch.setattr(io, "input_bool", _quantized_bool)

    cuda_before = torch.cuda.is_initialized()
    _seeded_storage(workdir, "df")
    model = import_model("SAEHD")(is_exporting=True,
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
    assert [i.name for i in g.input] == [INPUT_NAME]
    assert [o.name for o in g.output] == list(OUTPUT_NAMES)
    # the DFM I/O contract stays float32 (the official graph casts
    # the heads back to float32 at the boundary)
    assert g.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    for o in g.output:
        assert o.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _onnx_dims(g.input[0]) == ["batch", RES, RES, 3]

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
    eager = _eager_merge_heads(model, x)
    for o in eager:
        assert np.isfinite(o).all()
        assert o.min() >= 0.0 and o.max() <= 1.0

    assert torch.cuda.is_initialized() is cuda_before


def test_saehd_onnx_private_checkpoint_export(workdir, no_cli_prompts):
    """OPT-IN (untracked) real-checkpoint export gate:

    DFL_TEST_SAEHD_CHECKPOINT=<dir holding the official SAEHD model
    files (component .npy + data.dat, either the bare SAEHD_* form or
    the <model>_SAEHD_* prefix form)>

    Copies the private files into the sandbox temp workdir (the
    private directory is never read in place or referenced in
    tracked output), renames the official prefix to the production
    model name, exports through the production path and checks the
    same contract + ORT/eager parity on synthetic inputs. Skipped
    (never failing) when the variable is unset.
    """
    path = os.environ.get("DFL_TEST_SAEHD_CHECKPOINT")
    if path is None:
        pytest.skip("DFL_TEST_SAEHD_CHECKPOINT not set")
    src = Path(path)
    assert src.is_dir(), path

    for f in src.iterdir():
        if not f.is_file():
            continue
        m = _SAEHDFILE_RE.match(f.name)
        if m is None:
            continue
        rest = m.group("rest")
        if rest == "data.dat":
            shutil.copy(f, workdir / f"{MODEL_NAME}_data.dat")
        elif rest == "default_options.dat":
            shutil.copy(f, workdir / f"{MODEL_NAME}_default_options.dat")
        elif rest.endswith(".npy"):
            shutil.copy(f, workdir / f"{MODEL_NAME}_{rest}")

    # the export context loads the archi component files registered
    # by the is_exporting construction — the checkpoint's own
    # data.dat (iter >= 1) gates the resume path
    data = pickle.loads((workdir / f"{MODEL_NAME}_data.dat").read_bytes())
    if int(data.get("iter", 0)) < 1:
        data["iter"] = 1
        (workdir / f"{MODEL_NAME}_data.dat").write_bytes(
            pickle.dumps(data, 4))

    cuda_before = torch.cuda.is_initialized()
    model = import_model("SAEHD")(is_exporting=True,
                                  saved_models_path=workdir,
                                  cpu_only=True)
    model.export_dfm()

    onnx_path = None
    for cand in workdir.glob("*_SAEHD_model.dfm"):
        onnx_path = cand
        break
    assert onnx_path is not None
    m = onnx.load(str(onnx_path))
    checker.check_model(m)
    opsets = {(d.domain or "ai.onnx"): d.version
              for d in m.opset_import}
    assert opsets == {"ai.onnx": 12}, opsets
    assert [i.name for i in m.graph.input] == [INPUT_NAME]
    assert [o.name for o in m.graph.output] == list(OUTPUT_NAMES)
    res = model.resolution
    assert _onnx_dims(m.graph.input[0]) == ["batch", res, res, 3]

    # ORT/eager parity on the PRIVATE weights
    sess = InferenceSession(str(onnx_path),
                            providers=["CPUExecutionProvider"])
    for n in (1, 2):
        x = _random_input(n, base=60000, resolution=res)
        eager = _eager_merge_heads(model, x)
        ort_out = sess.run(None, {INPUT_NAME: x})
        pairs = ((ort_out[1], eager[0]), (ort_out[0], eager[1]),
                 (ort_out[2], eager[2]))
        for a, b in pairs:
            d = np.abs(a - b)
            assert d.max() <= PARITY_MAX_ABS
            assert d.mean() <= PARITY_MEAN_ABS

    assert torch.cuda.is_initialized() is cuda_before
