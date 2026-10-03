"""Phase-13 external/user-supplied pretraining-data contract.

The tests use temporary synthetic facesets only.  They never discover or
depend on a developer-local pretraining pack and they forbid network access
while exercising the positive path.
"""

import multiprocessing
import pickle
import runpy
import socket
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import osex
from core.interact import interact as io
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceType
from mainscripts import Trainer
import models
from models import ModelBase, PretrainingDataError
from samplelib import PackedFaceset, SampleLoader, SampleType
from samplelib.SampleLoader import PackedFacesetDataError


_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_SAEHDTest.Model import make_packed_faceset  # noqa: E402


def _validator(path):
    model = object.__new__(ModelBase)
    model.pretraining_data_path = path
    return model


def _remove_packed_landmarks(samples_path, sample_index):
    """Mutate one synthetic helper-built pack without adding fixture bytes."""
    packed_path = Path(samples_path) / "faceset.pak"
    packed = packed_path.read_bytes()
    version, config_size = struct.unpack("QQ", packed[:16])
    configs = pickle.loads(packed[16:16 + config_size])
    table_size = 8 * (len(configs) + 1)
    payload = packed[16 + config_size + table_size:]
    offset_table = packed[16 + config_size:16 + config_size + table_size]
    configs[sample_index]["landmarks"] = None
    config_bytes = pickle.dumps(configs, 4)
    packed_path.write_bytes(
        struct.pack("QQ", version, len(config_bytes))
        + config_bytes + offset_table + payload
    )


def test_pretraining_mode_requires_explicit_path(monkeypatch):
    def unexpected_loader(*_args, **_kwargs):
        raise AssertionError("missing input must not trigger fallback discovery")

    monkeypatch.setattr(SampleLoader, "load", unexpected_loader)
    with pytest.raises(PretrainingDataError, match="Pretraining mode is enabled") as exc:
        _validator(None).validate_pretraining_data()
    assert '--pretraining-data-dir "<PRETRAIN_DATA_DIR>"' in str(exc.value)


@pytest.mark.parametrize("kind", ["missing", "file", "empty", "nested-only"])
def test_structurally_invalid_pretraining_paths_fail_before_loader(
        plain_tmp, monkeypatch, kind):
    root = Path(plain_tmp) / "external pretraining fixture"
    if kind == "file":
        root.write_bytes(b"not a directory")
    elif kind == "empty":
        root.mkdir()
    elif kind == "nested-only":
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "face.jpg").write_bytes(b"candidate")

    def unexpected_loader(*_args, **_kwargs):
        raise AssertionError("structural preflight must run before SampleLoader")

    monkeypatch.setattr(SampleLoader, "load", unexpected_loader)
    with pytest.raises(PretrainingDataError) as exc:
        _validator(root).validate_pretraining_data()
    message = str(exc.value)
    assert "--pretraining-data-dir" in message
    if kind == "missing":
        assert "does not exist" in message
    elif kind == "file":
        assert "not a directory" in message
    else:
        assert "no candidate face samples" in message


def test_candidate_images_without_valid_dfl_metadata_are_rejected(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "candidate images"
    root.mkdir()
    (root / "face.jpg").write_bytes(b"synthetic non-DFL candidate")
    calls = []

    def no_valid_samples(sample_type, samples_path, **_kwargs):
        calls.append((sample_type, samples_path))
        return []

    monkeypatch.setattr(SampleLoader, "load", no_valid_samples)
    with pytest.raises(PretrainingDataError, match="no valid DFL face samples"):
        _validator(root).validate_pretraining_data()
    assert calls and calls[0][1] == root


def test_unexpected_loader_error_is_not_reclassified(plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "unexpected loader defect"
    root.mkdir()
    (root / "face.jpg").write_bytes(b"candidate")

    def programming_error(*_args, **_kwargs):
        raise RuntimeError("unexpected loader defect")

    monkeypatch.setattr(SampleLoader, "load", programming_error)
    with pytest.raises(RuntimeError, match="unexpected loader defect"):
        _validator(root).validate_pretraining_data()


def test_malformed_packed_faceset_is_typed_pretraining_failure(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "malformed packed data"
    root.mkdir()
    (root / "faceset.pak").write_bytes(b"truncated")
    raw_errors = []
    monkeypatch.setattr(
        io, "log_err", lambda message, **_kwargs: raw_errors.append(str(message))
    )

    with pytest.raises(PretrainingDataError, match="malformed faceset.pak") as exc:
        _validator(root).validate_pretraining_data()

    assert isinstance(exc.value.__cause__, PackedFacesetDataError)
    assert "--pretraining-data-dir" in str(exc.value)
    assert raw_errors == []


def test_strict_packed_loader_propagates_programming_runtime_error(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "packed programming defect"
    root.mkdir()
    (root / "faceset.pak").write_bytes(b"candidate")

    def programming_error(_samples_path):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(PackedFaceset, "load", programming_error)
    with pytest.raises(RuntimeError, match="sentinel"):
        _validator(root).validate_pretraining_data()


def test_sample_loader_default_retains_legacy_packed_fallback(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "legacy loader fallback"
    root.mkdir()
    (root / "faceset.pak").write_bytes(b"candidate")
    messages = []

    monkeypatch.setattr(
        PackedFaceset, "load",
        lambda _path: (_ for _ in ()).throw(RuntimeError("legacy sentinel")),
    )
    monkeypatch.setattr(SampleLoader, "load_face_samples", lambda _paths: [])
    monkeypatch.setattr(
        io, "log_err", lambda message, **_kwargs: messages.append(str(message))
    )

    samples = SampleLoader.load(SampleType.FACE, root)
    assert len(samples) == 0
    assert any("legacy sentinel" in message for message in messages)


def test_packed_sample_without_landmarks_is_rejected(plain_tmp):
    root = Path(plain_tmp) / "packed missing landmarks"
    make_packed_faceset(root, n_samples=2, size=64)
    _remove_packed_landmarks(root, sample_index=1)

    with pytest.raises(PretrainingDataError, match="usable 68x2 face landmarks") as exc:
        _validator(root).validate_pretraining_data()

    assert "1 sample(s)" in str(exc.value)
    assert "first sample index: 1" in str(exc.value)


def test_unpacked_dfl_sample_with_invalid_landmarks_is_rejected(plain_tmp):
    root = Path(plain_tmp) / "unpacked invalid landmarks"
    root.mkdir()
    face_path = root / "face.jpg"
    assert cv2.imwrite(str(face_path), np.zeros((64, 64, 3), dtype=np.uint8))
    dflimg = DFLIMG.load(face_path)
    assert dflimg is not None
    dflimg.set_face_type(FaceType.toString(FaceType.FULL))
    dflimg.set_landmarks([])
    dflimg.save()

    with pytest.raises(PretrainingDataError, match="usable 68x2 face landmarks"):
        _validator(root).validate_pretraining_data()


def test_missing_explicit_path_never_probes_fallback_filesystem(
        plain_tmp, monkeypatch):
    tmp_root = Path(plain_tmp)
    explicit = tmp_root / "configured missing faceset"
    sentinel_paths = {
        tmp_root / "legacy-dfl",
        tmp_root / "Downloads",
        tmp_root / "AppData",
        tmp_root / "sibling-repo",
    }
    original_exists = Path.exists
    inspected = []

    def guarded_exists(path):
        candidate = Path(path)
        inspected.append(candidate)
        if candidate in sentinel_paths or candidate != explicit:
            raise AssertionError(f"unexpected fallback filesystem probe: {candidate}")
        return original_exists(candidate)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    with pytest.raises(PretrainingDataError, match="does not exist"):
        _validator(explicit).validate_pretraining_data()
    assert inspected == [explicit]


def test_main_train_cli_normalizes_and_forwards_spaced_pretraining_path(
        plain_tmp, monkeypatch):
    cli_root = Path(plain_tmp) / "cli working directory"
    cli_root.mkdir()
    captured = []

    monkeypatch.chdir(cli_root)
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.setattr(Trainer, "main", lambda **kwargs: captured.append(kwargs) or 0)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "train",
        "--training-data-src-dir", "source faces",
        "--training-data-dst-dir", "destination faces",
        "--pretraining-data-dir", "external pretraining faces",
        "--model-dir", "saved model",
        "--model", "SAEHD",
        "--no-preview",
    ])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(__file__).resolve().parents[2] / "main.py"),
                       run_name="__main__")

    assert exc.value.code == 0
    assert len(captured) == 1
    assert captured[0]["pretraining_data_path"] == (
        cli_root / "external pretraining faces"
    ).resolve()
    assert captured[0]["training_data_src_path"] == (
        cli_root / "source faces"
    ).resolve()


def test_synthetic_packed_faceset_path_with_spaces_is_loaded_offline(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "external pretraining data with spaces"
    make_packed_faceset(root, n_samples=2, size=64)
    messages = []

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("pretraining-data validation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(io, "log_info", lambda message, **_kwargs: messages.append(str(message)))

    samples = _validator(root).validate_pretraining_data()
    assert len(samples) == 2
    assert any("2 valid face samples" in message for message in messages)
    assert any(str(root) in message for message in messages)


def _trainer_kwargs(root):
    return {
        "model_class_name": "SAEHD",
        "saved_models_path": root / "model",
        "training_data_src_path": root / "src",
        "training_data_dst_path": root / "dst",
        "pretraining_data_path": None,
        "pretrained_model_path": None,
        "no_preview": True,
        "force_model_name": None,
        "force_gpu_idxs": None,
        "cpu_only": True,
        "silent_start": True,
        "execute_programs": [],
        "debug": False,
    }


def test_trainer_returns_actionable_failure_for_malformed_packed_data(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp)
    pretraining_data = root / "malformed trainer pack"
    pretraining_data.mkdir()
    (pretraining_data / "faceset.pak").write_bytes(b"truncated")

    class ValidatingModel:
        def __init__(self, pretraining_data_path=None, **_kwargs):
            _validator(pretraining_data_path).validate_pretraining_data()

    messages = []
    tracebacks = []
    monkeypatch.setattr(models, "import_model", lambda _name: ValidatingModel)
    monkeypatch.setattr(io, "log_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        io, "log_err", lambda message, **_kwargs: messages.append(str(message))
    )
    monkeypatch.setattr(
        Trainer.traceback, "print_exc", lambda: tracebacks.append(True)
    )
    kwargs = _trainer_kwargs(root)
    kwargs["pretraining_data_path"] = pretraining_data

    rc = Trainer.main(**kwargs)

    assert rc == 1
    assert len(messages) == 1
    assert "malformed faceset.pak" in messages[0]
    assert "--pretraining-data-dir" in messages[0]
    assert tracebacks == []


def test_trainer_returns_failure_for_typed_pretraining_data_error(
        plain_tmp, monkeypatch):
    class FailingModel:
        def __init__(self, **_kwargs):
            raise PretrainingDataError("typed fixture failure")

    messages = []
    monkeypatch.setattr(models, "import_model", lambda _name: FailingModel)
    monkeypatch.setattr(io, "log_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(io, "log_err", lambda message, **_kwargs: messages.append(str(message)))

    rc = Trainer.main(**_trainer_kwargs(Path(plain_tmp)))
    assert rc == 1
    assert messages == ["Pretraining data error: typed fixture failure"]


def test_trainer_keeps_unexpected_errors_diagnosable(plain_tmp, monkeypatch):
    class BrokenModel:
        def __init__(self, **_kwargs):
            raise RuntimeError("unexpected model defect")

    tracebacks = []
    resource_messages = []
    monkeypatch.setattr(models, "import_model", lambda _name: BrokenModel)
    monkeypatch.setattr(io, "log_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(io, "log_err", lambda message, **_kwargs: resource_messages.append(str(message)))
    monkeypatch.setattr(Trainer.traceback, "print_exc", lambda: tracebacks.append(True))

    rc = Trainer.main(**_trainer_kwargs(Path(plain_tmp)))
    assert rc == 1
    assert tracebacks == [True]
    assert resource_messages == []
