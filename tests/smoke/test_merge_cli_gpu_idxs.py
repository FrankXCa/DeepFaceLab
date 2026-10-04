"""Focused CLI contract for merger forced-GPU index selection."""

import multiprocessing
import runpy
import sys
from pathlib import Path

import pytest

from core import osex
from core.leras import nn
from core.leras.device import Device, DeviceConfig, Devices
from mainscripts import Merger


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
OMITTED = object()


def _run_merge_cli(monkeypatch, tmp_path, force_gpu_idxs=OMITTED,
                   cpu_only=False):
    """Run the real merge subparser while replacing only merger execution."""
    captured = []

    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.setattr(
        Merger, "main", lambda **kwargs: captured.append(kwargs))

    argv = [
        "main.py", "merge",
        "--input-dir", str(tmp_path / "input"),
        "--output-dir", str(tmp_path / "output"),
        "--output-mask-dir", str(tmp_path / "mask"),
        "--aligned-dir", str(tmp_path / "aligned"),
        "--model-dir", str(tmp_path / "model"),
        "--model", "SAEHD",
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
    ("raw_value", "expected"),
    [
        ("0", [0]),
        ("0,1", [0, 1]),
        ("0, 1", [0, 1]),
        (" 0 , 1 ", [0, 1]),
        ("-1", [-1]),
        ("0,0", [0, 0]),
        ("99", [99]),
        ("0,99", [0, 99]),
    ],
)
def test_merge_cli_normalizes_forced_gpu_indexes(
        monkeypatch, tmp_path, raw_value, expected):
    exit_code, captured = _run_merge_cli(
        monkeypatch, tmp_path, force_gpu_idxs=raw_value)

    assert exit_code == 0
    assert len(captured) == 1
    value = captured[0]["force_gpu_idxs"]
    assert value == expected
    assert type(value) is list
    assert all(type(index) is int for index in value)
    assert captured[0]["model_class_name"] == "SAEHD"


@pytest.mark.parametrize("raw_value", ["", ",", "0,", ",1", "abc", "0,abc"])
def test_merge_cli_rejects_malformed_forced_gpu_indexes(
        monkeypatch, tmp_path, capsys, raw_value):
    exit_code, captured = _run_merge_cli(
        monkeypatch, tmp_path, force_gpu_idxs=raw_value)
    output = capsys.readouterr()

    assert exit_code == 2
    assert captured == []
    assert "--force-gpu-idxs" in output.err
    assert "comma-separated integer GPU indexes" in output.err
    assert "Traceback" not in output.out + output.err


def test_merge_cli_omitted_forced_gpu_indexes_remain_none(
        monkeypatch, tmp_path):
    exit_code, captured = _run_merge_cli(monkeypatch, tmp_path)

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0]["force_gpu_idxs"] is None


def test_merge_cli_preserves_cpu_only_with_valid_forced_indexes(
        monkeypatch, tmp_path):
    exit_code, captured = _run_merge_cli(
        monkeypatch, tmp_path, force_gpu_idxs="0", cpu_only=True)

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0]["cpu_only"] is True
    assert captured[0]["force_gpu_idxs"] == [0]


def test_merge_cli_rejects_malformed_forced_indexes_with_cpu_only(
        monkeypatch, tmp_path, capsys):
    exit_code, captured = _run_merge_cli(
        monkeypatch, tmp_path, force_gpu_idxs="abc", cpu_only=True)
    output = capsys.readouterr()

    assert exit_code == 2
    assert captured == []
    assert "comma-separated integer GPU indexes" in output.err
    assert "Traceback" not in output.out + output.err


def test_normalized_indexes_preserve_existing_device_selection_semantics(
        monkeypatch, tmp_path):
    fake_devices = Devices([
        Device(0, "GPU", "Mock GPU 0", 8 * 1024**3, 8 * 1024**3),
        Device(1, "GPU", "Mock GPU 1", 8 * 1024**3, 8 * 1024**3),
    ])
    monkeypatch.setattr(Devices, "all_devices", fake_devices)

    cases = [
        ("0,1", [0, 1]),
        ("0,0", [0]),
        ("-1", []),
        ("99", []),
        ("0,99", [0]),
    ]
    for raw_value, expected_devices in cases:
        exit_code, captured = _run_merge_cli(
            monkeypatch, tmp_path, force_gpu_idxs=raw_value)
        assert exit_code == 0
        indexes = captured[0]["force_gpu_idxs"]
        assert all(type(index) is int for index in indexes)

        config = DeviceConfig.GPUIndexes(indexes)
        assert [device.index for device in config.devices] == expected_devices
        assert config.cpu_only is (len(expected_devices) == 0)
