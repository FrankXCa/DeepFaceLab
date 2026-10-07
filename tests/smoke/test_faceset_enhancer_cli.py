"""FacesetEnhancer worker, file, device, and real argparse contracts."""

import multiprocessing
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import facelib
from DFLIMG import DFLIMG
from core import osex, pathex
from core.interact import interact as io
from core.leras import nn
from core.leras.device import Device, DeviceConfig, Devices
from mainscripts import FacesetEnhancer


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
OMITTED = object()


def _run_cli(monkeypatch, tmp_path, force_gpu_idxs=OMITTED, cpu_only=False):
    captured = []
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.setattr(
        FacesetEnhancer, "process_folder",
        lambda *args, **kwargs: captured.append((args, kwargs)))
    argv = [
        "main.py", "facesettool", "enhance", "--input-dir",
        str(tmp_path / "aligned faces"),
    ]
    if cpu_only:
        argv.append("--cpu-only")
    if force_gpu_idxs is not OMITTED:
        argv.extend(("--force-gpu-idxs", force_gpu_idxs))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MAIN_PY), run_name="__main__")
    return exc_info.value.code, captured


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", [0]), ("0,1", [0, 1]), ("0, 1", [0, 1]),
     (" 0 , 1 ", [0, 1]), ("-1", [-1]), ("0,0", [0, 0])],
)
def test_faceset_cli_normalizes_gpu_indexes(
        monkeypatch, tmp_path, raw, expected):
    code, captured = _run_cli(monkeypatch, tmp_path, raw)
    assert code == 0
    assert len(captured) == 1
    value = captured[0][1]["force_gpu_idxs"]
    assert value == expected
    assert type(value) is list
    assert all(type(index) is int for index in value)


@pytest.mark.parametrize("raw", ["", ",", "0,", ",1", "abc", "0,abc"])
def test_faceset_cli_rejects_malformed_gpu_indexes(
        monkeypatch, tmp_path, capsys, raw):
    code, captured = _run_cli(monkeypatch, tmp_path, raw)
    output = capsys.readouterr()
    assert code == 2
    assert captured == []
    assert "comma-separated integer GPU indexes" in output.err


def test_faceset_cli_omitted_and_cpu_only_boundaries(monkeypatch, tmp_path):
    code, captured = _run_cli(monkeypatch, tmp_path)
    assert code == 0
    assert captured[0][1]["force_gpu_idxs"] is None
    code, captured = _run_cli(
        monkeypatch, tmp_path, force_gpu_idxs="0", cpu_only=True)
    assert code == 0
    assert captured[0][1]["cpu_only"] is True
    assert captured[0][1]["force_gpu_idxs"] == [0]


def _fake_devices(monkeypatch):
    devices = Devices([
        Device(0, "GPU", "Mock GPU 0", 8 * 1024**3, 8 * 1024**3),
        Device(1, "GPU", "Mock GPU 1", 2 * 1024**3, 2 * 1024**3),
    ])
    monkeypatch.setattr(Devices, "all_devices", devices)
    return devices


def test_faceset_worker_passes_explicit_local_device_without_initialize(
        monkeypatch, plain_tmp):
    _fake_devices(monkeypatch)
    captured = []

    class FakeEnhancer:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def enhance(self, image, **_kwargs):
            return image

    monkeypatch.setattr(facelib, "FaceEnhancer", FakeEnhancer)
    monkeypatch.setattr(
        nn, "initialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FacesetEnhancer must not initialize Leras")))
    client = FacesetEnhancer.FacesetEnhancerSubprocessor.Cli.__new__(
        FacesetEnhancer.FacesetEnhancerSubprocessor.Cli)
    client.log_info = lambda *_args, **_kwargs: None
    client.on_initialize({
        "device_idx": 1,
        "device_type": "GPU",
        "device_name": "Mock GPU 1",
        "output_dirpath": Path(plain_tmp),
        "nn_initialize_mp_lock": None,
    })
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["run_on_cpu"] is False
    assert kwargs["place_model_on_cpu"] is True
    assert [device.index for device in kwargs["device_config"].devices] == [1]


def _image(size=32):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    return np.stack((x / size, y / size, (x + y) / (2 * size)),
                    axis=-1).astype(np.float32)


def _make_dfl(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.clip(_image() * 255, 0, 255).astype(np.uint8))
    dfl = DFLIMG.load(path)
    assert dfl is not None
    dfl.set_dict({"face_type": "full_face", "source_filename": marker})
    dfl.save()
    return dfl.get_dict()


def test_faceset_worker_preserves_metadata_names_quality_and_invalid_input(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "faceset with spaces"
    output = root.parent / "faceset with spaces_enhanced"
    output.mkdir(parents=True)
    source = root / "face one.jpg"
    metadata = _make_dfl(source, "synthetic-marker")
    invalid = root / "invalid.jpg"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(invalid), np.zeros((32, 32, 3), np.uint8))

    class IdentityEnhancer:
        @staticmethod
        def enhance(image, **_kwargs):
            return image

    errors = []
    write_parameters = []
    real_imwrite = FacesetEnhancer.cv2_imwrite

    def recording_imwrite(path, image, params=None):
        write_parameters.append(params)
        return real_imwrite(path, image, params)

    monkeypatch.setattr(FacesetEnhancer, "cv2_imwrite", recording_imwrite)
    client = FacesetEnhancer.FacesetEnhancerSubprocessor.Cli.__new__(
        FacesetEnhancer.FacesetEnhancerSubprocessor.Cli)
    client.output_dirpath = output
    client.fe = IdentityEnhancer()
    client.log_err = lambda message: errors.append(str(message))
    valid_result = client.process_data(source)
    invalid_result = client.process_data(invalid)
    assert valid_result[0] == 1
    assert valid_result[2].name == source.name
    assert valid_result[2].parent == output
    assert invalid_result == (0, invalid, None)
    assert write_parameters == [[int(cv2.IMWRITE_JPEG_QUALITY), 100]]
    assert any("not a dfl image" in message for message in errors)
    restored = DFLIMG.load(valid_result[2])
    assert restored.has_data()
    assert restored.get_dict() == metadata
    encoded = valid_result[2].read_bytes()
    assert encoded[:2] == b"\xff\xd8" and encoded[-2:] == b"\xff\xd9"


@pytest.mark.parametrize("merge", [False, True])
def test_process_folder_output_name_replacement_and_cleanup(
        monkeypatch, plain_tmp, merge):
    root = Path(plain_tmp) / "parent with spaces" / "aligned faces"
    first = root / "first.jpg"
    second = root / "second.jpg"
    first_meta = _make_dfl(first, "first")
    second_meta = _make_dfl(second, "second")

    class FakeSubprocessor:
        def __init__(self, image_paths, output_dirpath, device_config):
            self.image_paths = image_paths
            self.output = output_dirpath
            self.device_config = device_config

        def run(self):
            result = []
            for source in self.image_paths:
                target = self.output / source.name
                target.write_bytes(source.read_bytes())
                result.append((source, target))
            return result

    monkeypatch.setattr(
        FacesetEnhancer, "FacesetEnhancerSubprocessor", FakeSubprocessor)
    monkeypatch.setattr(io, "input_bool", lambda *_args, **_kwargs: merge)
    monkeypatch.setattr(io, "log_info", lambda *_args, **_kwargs: None)
    FacesetEnhancer.process_folder(root, cpu_only=True, force_gpu_idxs=[0])
    enhanced = root.parent / "aligned faces_enhanced"
    if merge:
        assert not enhanced.exists()
        assert DFLIMG.load(first).get_dict() == first_meta
        assert DFLIMG.load(second).get_dict() == second_meta
    else:
        assert enhanced.is_dir()
        assert sorted(path.name for path in enhanced.glob("*.jpg")) == [
            "first.jpg", "second.jpg"]


def test_faceset_device_selection_cpu_selected_forced_and_fallback(
        monkeypatch, plain_tmp):
    _fake_devices(monkeypatch)
    root = Path(plain_tmp) / "device-selection"
    root.mkdir()
    monkeypatch.setattr(pathex, "get_image_paths", lambda *_args: [])
    monkeypatch.setattr(io, "input_bool", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(io, "log_info", lambda *_args, **_kwargs: None)
    captured = []

    class FakeSubprocessor:
        def __init__(self, _paths, _output, device_config):
            captured.append(device_config)

        def run(self):
            return []

    monkeypatch.setattr(
        FacesetEnhancer, "FacesetEnhancerSubprocessor", FakeSubprocessor)
    FacesetEnhancer.process_folder(root, cpu_only=True, force_gpu_idxs=[0])
    FacesetEnhancer.process_folder(root, cpu_only=False, force_gpu_idxs=[1])
    FacesetEnhancer.process_folder(root, cpu_only=False, force_gpu_idxs=[0])
    FacesetEnhancer.process_folder(root, cpu_only=False, force_gpu_idxs=[99])
    assert captured[0].cpu_only
    assert [device.index for device in captured[1].devices] == [1]
    assert [device.index for device in captured[2].devices] == [0]
    assert captured[3].cpu_only


@pytest.mark.parametrize("use_cuda", [False, True])
def test_real_faceenhancer_faceset_worker_cpu_and_cuda(
        plain_tmp, use_cuda, monkeypatch):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("physical CUDA required")
    root = Path(plain_tmp) / f"real worker {'cuda' if use_cuda else 'cpu'}"
    output = root.parent / f"real worker {'cuda' if use_cuda else 'cpu'}_enhanced"
    source = root / "synthetic face.jpg"
    metadata = _make_dfl(source, "production-integration")
    output.mkdir(parents=True)
    if use_cuda:
        monkeypatch.setattr(Devices, "all_devices", Devices([
            Device(0, "GPU", torch.cuda.get_device_name(0),
                   24 * 1024**3, 24 * 1024**3),
        ]))
    config = (DeviceConfig.GPUIndexes([0]) if use_cuda
              else DeviceConfig.CPU())

    client = FacesetEnhancer.FacesetEnhancerSubprocessor.Cli.__new__(
        FacesetEnhancer.FacesetEnhancerSubprocessor.Cli)
    client.log_info = lambda *_args, **_kwargs: None
    client.log_err = lambda message: pytest.fail(str(message))
    client.on_initialize({
        "device_idx": 0,
        "device_type": "GPU" if use_cuda else "CPU",
        "device_name": "physical CUDA" if use_cuda else "CPU0",
        "output_dirpath": output,
        "nn_initialize_mp_lock": None,
    })
    # on_initialize resolves its own equivalent worker-local config; the
    # explicit construction above proves the selected physical index exists.
    assert config.cpu_only is (not use_cuda)
    result = client.process_data(source)
    assert result[0] == 1
    assert result[2].name == source.name
    restored = DFLIMG.load(result[2])
    assert restored.has_data()
    assert restored.get_dict() == metadata
    assert client.fe.compute_device.type == ("cuda" if use_cuda else "cpu")
