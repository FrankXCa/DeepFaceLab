"""Phase 5 — training lifecycle (``models.ModelBase``) tests.

Covers the top-level official training lifecycle on the torch
foundation (the official ``models/ModelBase.py`` is backend-neutral and
runs unchanged except the documented USER_LEGACY hardenings — see the
module docstring and ``docs/PHASE5_PLAN.md``). The test vehicle is the
test-only dummy model in ``tests/smoke/Model_Dummy/`` (the package name
follows the official ``Model_<Class>`` folder convention that
``models.ModelBase.__init__`` derives ``model_class_name`` from):

- construction (forced model name — no interactive prompt; first-run
  vs resume semantics: ``iter``/``options``/``loss_history`` restored
  from ``data.dat`` only when ``iter != 0``; class-level
  ``default_options.dat`` snapshot on first run — official behavior
  kept as the default of the adopted ``disable_default_options_autosave()``
  hook);
- deterministic component registration through the model-owned
  ``on_initialize`` (leras container + AdaBelief optimizer + sample
  generator, the official calling pattern);
- the model-owned training step (official pattern: the MODEL consumes
  its own samples in ``onTrainOneIter``; native torch autograd —
  ``loss.backward()`` + the optimizer's ``get_update_op`` — no TF
  session/placeholder/feed_dict, no ``nn.tf.gradients`` stub);
- iteration bookkeeping: the model-level ``iter`` (persisted in
  ``data.dat`` — drives UI/backup/preview/target) and each optimizer's
  own ``iters`` counter (persisted in the optimizer file — drives
  lr_cos) are TWO distinct counters with different roles, advanced in
  lockstep by the official lifecycle and restored from their own files;
- save: summary text + ``onSave`` (official ``[[model, filename], ...]``
  pairs through the Phase 4 Saveable — raw pickle protocol-4 streams)
  + ``data.dat`` bookkeeping pickle + the 24-slot autobackup ring
  (``create_backup``);
- resume equivalence (the required A/B test): Run A = init + 3 steps;
  Run B = identical init + 2 steps + save + recreate + load + 1 step;
  parameters, optimizer state (iters + ms_/vs_), iteration state and
  loss values are EXACT (zero RNG: constant initializers, deterministic
  weight fill, fixed sample batches, deterministic update math);
- strict failures: a required component file missing on resume fails
  explicitly (Phase 4 all-or-nothing policy — the official
  ``exists()``-guard silent re-initialization is NOT reproduced on
  resume); a corrupted/malformed component or bookkeeping file fails
  the load explicitly, nothing partially applied;
- CPU lifecycle (default environment) and the RTX 4090 lifecycle
  (skip on CPU-only environments);
- AST hygiene: no TensorFlow import, no direct ``torch.cuda.*`` /
  ``'cuda:'`` literals in the lifecycle foundation sources.

Parity: EXACT on CPU; the GPU test compares the same A/B runs on the
CUDA device (f32, same device both runs — bit-exact in this
environment, as established by the Phase 3B/3F GPU suites).

Official DFL has NO train/eval mode switch (its BatchNorm is explicitly
"not for training"; preview reuses the training forward) — the base
lifecycle therefore performs no mode toggling, and the inference
boundary is the leras container's ``run()`` (``torch.no_grad``; see
``test_leras_modelbase.py``). The model phases own any layer-level
``train()/eval()`` need.
"""

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from core.leras import nn as dfl_nn
import core.leras.models  # noqa: F401  (binds nn.ModelBase)

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_Dummy.Model import (  # noqa: E402
    DummyModel, DummyModelBadGen, DummyModelNoGen,
    DummyModelNoSnapshot, make_model, snapshot)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# the official lifecycle distinguishes two names (models/ModelBase.py):
# model_class_name = "Dummy" (derived from the Model_Dummy test package
# folder — prefixes the class-level default_options file) and
# model_name = "dummy_Dummy" (forced via force_model_class_name in the
# test — prefixes every per-model file)
MODEL_CLASS = "dummy_Dummy"
CLASS_NAME = "Dummy"


def construct(model_class, tmpdir, **device_kwargs):
    return make_model(model_class, tmpdir, **device_kwargs)


# --- construction / first run ---------------------------------------------

def test_first_run_construction_cpu(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)

    assert model.is_first_run() is True
    assert model.get_iter() == 0
    assert model.get_model_name() == MODEL_CLASS
    assert model.options['dummy_alpha'] == 0.25
    assert model.get_batch_size() == 2
    # CPU device selection (no GPU configured)
    assert len(model.device_config.devices) == 0
    # components built and registered by on_initialize
    assert model.net.built is True
    assert [l.name for l in model.net.layers] == ["conv1", "conv2"]
    assert len(model.dummy_opt.ms_dict) == 4  # one ms + one vs per parameter
    # parameters live on the training device (CPU here)
    for p in model.net.parameters():
        assert p.device.type == "cpu"
    # official first-run behavior: the class-level default snapshot IS
    # written (the adopted autosave hook defaults to the official True)
    default_path = Path(plain_tmp) / f"{CLASS_NAME}_default_options.dat"
    assert default_path.exists()
    assert pickle.loads(default_path.read_bytes())['dummy_alpha'] == 0.25


def test_autosave_disable_hook_skips_snapshot(plain_tmp):
    model = construct(DummyModelNoSnapshot, plain_tmp, cpu_only=True)
    assert not (Path(plain_tmp) / f"{CLASS_NAME}_default_options.dat").exists()


# --- training step / iteration ---------------------------------------------

def test_training_step_updates_state_cpu(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    fill = {
        tuple(p.shape): p.detach().clone()
        for p in model.net.parameters()
    }

    it, _ = model.train_one_iter()
    assert it == 1
    it, _ = model.train_one_iter()
    assert it == 2
    assert model.get_iter() == 2
    # two loss entries, finite positive scalars
    assert len(model.loss_history) == 2
    for entry in model.loss_history:
        assert len(entry) == 1 and np.isfinite(entry[0]) and entry[0] > 0.0
    # the optimizer step actually ran (post-increment counter, states)
    assert int(model.dummy_opt.iterations) == 2
    assert any(not torch.equal(
        st, torch.zeros_like(st)) for st in model.dummy_opt._states())
    # and the weights moved off the deterministic fill
    moved = False
    for p in model.net.parameters():
        if not torch.equal(p, fill[tuple(p.shape)]):
            moved = True
    assert moved


def test_set_iter_truncates_loss_history(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    for _ in range(3):
        model.train_one_iter()
    assert len(model.loss_history) == 3
    model.set_iter(1)
    assert model.get_iter() == 1
    assert len(model.loss_history) == 1


# --- save / file contract ----------------------------------------------------

def test_save_creates_official_files_cpu(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    model.train_one_iter()
    model.save()

    root = Path(plain_tmp)
    data_path = root / f"{MODEL_CLASS}_data.dat"
    net_path = root / f"{MODEL_CLASS}_dummy_net.npy"
    opt_path = root / f"{MODEL_CLASS}_dummy_opt.npy"
    summary_path = root / f"{MODEL_CLASS}_summary.txt"
    for f in (data_path, net_path, opt_path, summary_path):
        assert f.exists(), f

    # Phase 4 outer contract: the component files are raw pickle
    # protocol-4 streams (the .npy extension is a misnomer)
    for f in (net_path, opt_path):
        raw = f.read_bytes()
        assert raw[:3] == b"\x80\x04\x95"
        assert b"\x93NUMPY" not in raw[:64]

    data = pickle.loads(data_path.read_bytes())
    assert data['iter'] == 1
    assert data['options']['dummy_alpha'] == 0.25
    assert len(data['loss_history']) == 1
    # optimizer file carries the official state names (Phase 4 mapping)
    opt_d = pickle.loads(opt_path.read_bytes())
    assert "iters:0" in opt_d
    assert any(k.startswith("ms_") for k in opt_d)
    assert any(k.startswith("vs_") for k in opt_d)


def test_create_backup_copies_component_files(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    model.train_one_iter()
    model.save()
    model.create_backup()  # empty preview list -> no multiprocessing writer

    slot = Path(plain_tmp) / f"{MODEL_CLASS}_autobackups" / "01"
    assert slot.is_dir()
    names = {p.name for p in slot.iterdir()}
    assert {f"{MODEL_CLASS}_dummy_net.npy", f"{MODEL_CLASS}_dummy_opt.npy",
            f"{MODEL_CLASS}_summary.txt", f"{MODEL_CLASS}_data.dat"} <= names


# --- resume -------------------------------------------------------------------

def test_resume_restores_state_cpu(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    model.train_one_iter()
    model.train_one_iter()
    before = snapshot(model)
    model.save()

    resumed = construct(DummyModel, plain_tmp, cpu_only=True)  # same dir

    assert resumed.is_first_run() is False
    assert resumed.get_iter() == 2
    assert resumed.options['dummy_alpha'] == 0.25
    assert [list(x) for x in resumed.loss_history] == before["losses"]

    after = snapshot(resumed)
    for a, b in zip(before["params"], after["params"]):
        assert torch.equal(a, b)
    for a, b in zip(before["states"], after["states"]):
        assert torch.equal(a, b)


def test_resume_equivalence_cpu(plain_tmp):
    """Required A/B test: A = init + 3 steps; B = identical init + 2
    steps + save + recreate + load + 1 step. With zero RNG involved the
    two histories must be EXACT: parameters, optimizer state (iters +
    ms_/vs_), iteration state and loss values."""
    dir_a = str(Path(plain_tmp) / "run_a")
    dir_b = str(Path(plain_tmp) / "run_b")
    # the lifecycle writes into these directories (as in the GUI, the
    # directories pre-exist)
    Path(dir_a).mkdir(parents=True, exist_ok=True)
    Path(dir_b).mkdir(parents=True, exist_ok=True)

    model_a = construct(DummyModel, dir_a, cpu_only=True)
    for _ in range(3):
        model_a.train_one_iter()
    snap_a = snapshot(model_a)

    model_b = construct(DummyModel, dir_b, cpu_only=True)
    model_b.train_one_iter()
    model_b.train_one_iter()
    model_b.save()

    model_b = construct(DummyModel, dir_b, cpu_only=True)  # recreate + load
    model_b.train_one_iter()
    snap_b = snapshot(model_b)

    assert snap_a["iter"] == snap_b["iter"] == 3
    assert snap_a["losses"] == snap_b["losses"]
    assert len(snap_a["params"]) == len(snap_b["params"])
    for a, b in zip(snap_a["params"], snap_b["params"]):
        assert torch.equal(a, b)
    for a, b in zip(snap_a["states"], snap_b["states"]):
        assert torch.equal(a, b)


def test_missing_required_file_on_resume_fails(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    model.train_one_iter()
    model.save()

    victim = Path(plain_tmp) / f"{MODEL_CLASS}_dummy_opt.npy"
    victim.unlink()

    with pytest.raises(FileNotFoundError, match="required component file missing on resume"):
        construct(DummyModel, plain_tmp, cpu_only=True)


def test_corrupted_component_file_fails_explicitly(plain_tmp):
    model = construct(DummyModel, plain_tmp, cpu_only=True)
    model.train_one_iter()
    model.save()

    net_path = Path(plain_tmp) / f"{MODEL_CLASS}_dummy_net.npy"
    raw = net_path.read_bytes()
    net_path.write_bytes(raw[: len(raw) // 2])  # truncated pickle

    with pytest.raises(Exception):
        construct(DummyModel, plain_tmp, cpu_only=True)


def test_malformed_data_dat_fails_explicitly(plain_tmp):
    # a bookkeeping file that exists but cannot be parsed must fail the
    # construction explicitly (no silent first-run reinterpretation):
    # a truncated pickle frame header is a guaranteed parse failure
    data_path = Path(plain_tmp) / f"{MODEL_CLASS}_data.dat"
    data_path.write_bytes(b"\x80\x04\x95")

    with pytest.raises(Exception):
        construct(DummyModel, plain_tmp, cpu_only=True)


# --- strict generator validation ------------------------------------------------

def test_unset_generators_fail_explicitly(plain_tmp):
    # official quirk preserved: an UNSET generator_list raises
    # AttributeError (the ValueError branch requires the attribute to
    # exist as None — a set-but-nonconforming list gets it below)
    with pytest.raises((AttributeError, ValueError)):
        construct(DummyModelNoGen, plain_tmp, cpu_only=True)


def test_non_sample_generator_rejected(plain_tmp):
    with pytest.raises(ValueError, match="SampleGeneratorBase"):
        construct(DummyModelBadGen, plain_tmp, cpu_only=True)


# --- RTX 4090 lifecycle ----------------------------------------------------------

@requires_gpu
def test_gpu_lifecycle_and_resume(plain_tmp):
    dfl_nn.initialize_main_env()
    dir_a = str(Path(plain_tmp) / "gpu_a")
    dir_b = str(Path(plain_tmp) / "gpu_b")
    Path(dir_a).mkdir(parents=True, exist_ok=True)
    Path(dir_b).mkdir(parents=True, exist_ok=True)

    model_a = construct(DummyModel, dir_a, force_gpu_idxs=[0])
    # Phase 2 Device API: the registry backend key ('cuda'), not a .type
    assert model_a.device_config.devices[0].backend == "cuda"
    for _ in range(3):
        model_a.train_one_iter()
    for p in model_a.net.parameters():
        assert p.device.type == "cuda"
    snap_a = snapshot(model_a)

    model_b = construct(DummyModel, dir_b, force_gpu_idxs=[0])
    model_b.train_one_iter()
    model_b.train_one_iter()
    model_b.save()
    model_b = construct(DummyModel, dir_b, force_gpu_idxs=[0])
    model_b.train_one_iter()
    snap_b = snapshot(model_b)

    assert snap_a["iter"] == snap_b["iter"] == 3
    assert snap_a["losses"] == snap_b["losses"]
    for a, b in zip(snap_a["params"], snap_b["params"]):
        assert torch.equal(a, b)
    for a, b in zip(snap_a["states"], snap_b["states"]):
        assert torch.equal(a, b)
    for p in model_b.net.parameters():
        assert p.device.type == "cuda"


# --- hygiene -------------------------------------------------------------------

def test_no_tf_or_cuda_in_lifecycle_sources():
    files = [
        REPO_ROOT / "models" / "ModelBase.py",
        REPO_ROOT / "models" / "__init__.py",
        REPO_ROOT / "samplelib" / "SampleGeneratorBase.py",
        REPO_ROOT / "core" / "leras" / "models" / "ModelBase.py",
    ]
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("tensorflow"), (f, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").startswith("tensorflow") is False, (
                    f, node.module)
            elif isinstance(node, ast.Attribute):
                src = ast.unparse(node)
                assert not src.startswith("torch.cuda"), (f, src)
        text = f.read_text(encoding="utf-8")
        assert '"cuda:"' not in text and "'cuda:'" not in text, f
