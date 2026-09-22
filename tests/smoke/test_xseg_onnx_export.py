"""Phase 10E acceptance — XSeg ONNX export (torch, no TensorFlow).

The official DFM export contract (models/Model_XSeg/Model_tf.py L254-281,
tf2onnx opset 13) is reproduced by ``XSegModel.export_dfm`` on the torch
foundation:

- input ``in_face`` (TF-visible ``in_face:0``): NHWC ``(N,256,256,3)``
  float32 with a DYNAMIC batch axis (N=1 and N>1 both run);
- output ``out_mask`` (TF-visible ``out_mask:0``): NHWC
  ``(N,256,256,1)`` float32 — the official ``_, pred`` flow-tuple
  selection (the sigmoid, NOT the raw logits);
- opset 13 (the official tf2onnx opset — unchanged);
- the exporter is the legacy torch TorchScript ONNX exporter
  (``dynamo=False`` — torch 2.14's dynamo path requires onnxscript,
  which the project deliberately does not install: the Phase 10E
  dependency decision in docs/PHASE10_STATE.md).

Validation here is eager torch vs ONNX Runtime (CPU): the parity
gates (max_abs <= 1e-4, mean_abs <= 1e-5) were set AFTER measuring
the actual numbers on the seeded reference model (observed
max_abs ~1.3e-5, mean_abs ~1.4e-6 for N=1..4 — a ~7x safety margin
for ORT thread-count / platform FP variance).

The tracked tests use deterministic seeded synthetic models only.
A real checkpoint export is private and opt-in via the
``DFL_TEST_XSEG_ONNX_CHECKPOINT`` environment variable (an untracked
directory holding a trained ``XSeg_256.npy`` + ``XSeg_data.dat``).
"""

import os
import pickle
import random
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
from facelib import XSegNet
from models import import_model

REPO_ROOT = Path(__file__).resolve().parents[2]

RES = 256
SEED = 1234

PARITY_MAX_ABS = 1e-4
PARITY_MEAN_ABS = 1e-5
PARITY_RANGE_ABS = 1e-4


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


@pytest.fixture
def no_cli_prompts(monkeypatch):
    """Neutralize the official CLI prompts on the export path."""
    monkeypatch.setattr(io, "input_str", lambda *a, **k: "")
    monkeypatch.setattr(io, "input_bool", lambda *a, **k: False)


@pytest.fixture
def workdir(plain_tmp):
    """Dedicated model-storage dir inside the shared plain_tmp base."""
    import shutil

    d = Path(plain_tmp) / "xseg_onnx"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _seeded_storage(root):
    """Build a deterministic XSeg model storage dir in ``root``:

    the seeded net's weights (XSeg_256.npy + XSeg_256_opt.npy, the
    XSegNet save_weights layout) plus the model data file
    ``XSeg_data.dat`` (iter=1 -> a RESUMED model, so the production
    constructor takes the strict load_weights path and no first-run
    prompts fire).
    """
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    net = XSegNet(name="XSeg",
                  resolution=RES,
                  load_weights=False,
                  weights_file_root=str(root),
                  training=True,
                  place_model_on_cpu=True,
                  optimizer=dfl_nn.RMSprop(lr=0.0001, lr_dropout=0.3,
                                           name="opt"),
                  data_format=dfl_nn.data_format)
    net.save_weights()

    (root / "XSeg_data.dat").write_bytes(pickle.dumps(
        {"iter": 1,
         "options": {"face_type": "f", "pretrain": False,
                     "batch_size": 4}}))


def _build_export_model(root, no_cli_prompts):
    """The ExportDFM production construction contract:
    models.import_model('XSeg')(is_exporting=True, cpu_only=True)."""
    model = import_model("XSeg")(is_exporting=True,
                                 saved_models_path=root,
                                 cpu_only=True)
    model.export_dfm()
    return model


def _onnx_dims(value_info):
    return [d.dim_param if d.dim_param else d.dim_value
            for d in value_info.type.tensor_type.shape.dim]


def _random_input(n, base=9000):
    rng = np.random.RandomState(base + n)
    return rng.rand(n, RES, RES, 3).astype(np.float32)


def _eager_mask(net, x_nhwc):
    """The official flow contract: NHWC in -> NCHW net (the
    is_exporting data-format rule) -> sigmoid out -> NHWC."""
    x = torch.from_numpy(x_nhwc).permute(0, 3, 1, 2)
    with torch.no_grad():
        _, pred = net(x)
    return pred.permute(0, 2, 3, 1).numpy()


def test_export_dfm_writes_contract_onnx(workdir, no_cli_prompts):
    _seeded_storage(workdir)
    # the export must not initialize a CUDA context (the suite's
    # session may already have one from other test files — assert the
    # export path itself never flips it)
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir, no_cli_prompts)

    onnx_path = workdir / "XSeg_model.onnx"
    assert onnx_path.is_file()
    size = onnx_path.stat().st_size
    # the weights are embedded in the proto (no external data) — the
    # 17.5M-param fp32 net alone is ~70MB on disk
    assert size > 60_000_000

    m = onnx.load(str(onnx_path))
    checker.check_model(m)

    opsets = {(d.domain or "ai.onnx"): d.version for d in m.opset_import}
    assert opsets == {"ai.onnx": 13}, opsets

    g = m.graph
    assert [i.name for i in g.input] == ["in_face"]
    assert [o.name for o in g.output] == ["out_mask"]

    in_vi, out_vi = g.input[0], g.output[0]
    assert in_vi.type.tensor_type.elem_type == TensorProto.FLOAT
    assert out_vi.type.tensor_type.elem_type == TensorProto.FLOAT
    # dynamic batch + the exact official static contract
    assert _onnx_dims(in_vi) == ["batch", 256, 256, 3]
    assert _onnx_dims(out_vi) == ["batch", 256, 256, 1]

    # weights are embedded in the graph (no external-data tensors):
    # the exporter constant-folds the layer parameters down to 55
    # distinct initializers for this net (measured); the 60MB+ file
    # size below is the real "weights are inside the proto" check
    assert len(g.initializer) > 20
    # external_data is a repeated field (0 entries == fully inline)
    assert all(len(t.external_data) == 0 for t in g.initializer)
    # the export ran on the CPU net (cpu_only construction)
    assert torch.cuda.is_initialized() is cuda_before


def test_exported_onnx_dynamic_batch_ort_parity(workdir, no_cli_prompts):
    """Dynamic batch (N=1 and N>1) with eager-torch vs ONNX Runtime
    parity on the seeded reference model (CPU, both sides)."""
    _seeded_storage(workdir)
    cuda_before = torch.cuda.is_initialized()
    model = _build_export_model(workdir, no_cli_prompts)

    sess = InferenceSession(str(workdir / "XSeg_model.onnx"),
                            providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["in_face"]
    assert [o.name for o in sess.get_outputs()] == ["out_mask"]

    net = model.model.model
    for n in (1, 2, 4):
        x = _random_input(n)
        eager = _eager_mask(net, x)
        ort_out = sess.run(["out_mask"], {"in_face": x})[0]

        assert ort_out.shape == (n, RES, RES, 1)
        assert ort_out.dtype == np.float32
        assert eager.shape == (n, RES, RES, 1)
        assert eager.dtype == np.float32

        d = np.abs(ort_out - eager)
        assert d.max() <= PARITY_MAX_ABS, \
            f"N={n}: max_abs {d.max():.3e} exceeds the measured gate"
        assert d.mean() <= PARITY_MEAN_ABS, \
            f"N={n}: mean_abs {d.mean():.3e} exceeds the measured gate"
        # the sigmoid contract keeps both outputs in [0,1]; the
        # measured ranges agree well inside fp32 noise
        assert abs(ort_out.min() - eager.min()) <= PARITY_RANGE_ABS
        assert abs(ort_out.max() - eager.max()) <= PARITY_RANGE_ABS
        assert ort_out.min() >= 0.0 and ort_out.max() <= 1.0

    # the export + ORT session never initialized a CUDA context
    assert torch.cuda.is_initialized() is cuda_before


def test_export_is_tf_and_onnxscript_free(workdir, no_cli_prompts):
    """Dependency policy: the production export path needs neither
    TensorFlow (the 10E goal) nor onnxscript (the legacy exporter
    is the chosen torch 2.14 path — see PHASE10_STATE.md)."""
    _seeded_storage(workdir)
    model = _build_export_model(workdir, no_cli_prompts)

    for banned in ("tensorflow", "tensorflow.python", "keras",
                   "onnxscript"):
        assert not any(mod == banned or mod.startswith(banned + ".")
                       for mod in sys.modules), \
            f"{banned} must not be importable after the export"
    assert (workdir / "XSeg_model.onnx").is_file()


def test_xseg_onnx_private_checkpoint_export(workdir):
    """OPT-IN (untracked) real-checkpoint export gate:

    DFL_TEST_XSEG_ONNX_CHECKPOINT=<dir holding a trained XSeg_256.npy
    + XSeg_256_opt.npy + XSeg_data.dat>

    Exports the PRIVATE trained checkpoint through the production
    path and checks the same ORT/eager parity on synthetic inputs.
    Skipped (never failing) when the variable is unset, so the
    tracked suite stays green on machines without the checkpoint.
    """
    ckpt = os.environ.get("DFL_TEST_XSEG_ONNX_CHECKPOINT")
    if not ckpt:
        pytest.skip("DFL_TEST_XSEG_ONNX_CHECKPOINT not set (private "
                    "opt-in gate)")

    root = Path(ckpt)
    missing = [f for f in ("XSeg_256.npy", "XSeg_data.dat")
               if not (root / f).is_file()]
    if missing:
        pytest.skip(f"checkpoint dir {root} missing {missing}")

    # copy into the workdir so the strict loader + export write
    # against a private temp copy (never the user's original dir)
    import shutil

    for f in os.listdir(root):
        if f.startswith("XSeg") and f.endswith((".npy", ".dat")):
            shutil.copy(str(root / f), str(workdir / f))

    model = import_model("XSeg")(is_exporting=True,
                                 saved_models_path=workdir,
                                 cpu_only=True)
    model.export_dfm()

    m = onnx.load(str(workdir / "XSeg_model.onnx"))
    checker.check_model(m)
    assert [i.name for i in m.graph.input] == ["in_face"]
    assert [o.name for o in m.graph.output] == ["out_mask"]

    sess = InferenceSession(str(workdir / "XSeg_model.onnx"),
                            providers=["CPUExecutionProvider"])
    net = model.model.model
    for n in (1, 2):
        x = _random_input(n, base=4242)
        eager = _eager_mask(net, x)
        ort_out = sess.run(["out_mask"], {"in_face": x})[0]
        d = np.abs(ort_out - eager)
        assert d.max() <= PARITY_MAX_ABS
        assert d.mean() <= PARITY_MEAN_ABS
