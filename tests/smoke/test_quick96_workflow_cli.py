"""Real argparse boundary checks for the curated Quick96 workflows."""

import multiprocessing
import runpy
import sys
from pathlib import Path

import pytest

from core import osex
from core.leras import nn
from mainscripts import Merger, Trainer


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MAIN_PY), run_name="__main__")

    assert exc_info.value.code == 0


def test_quick96_train_launcher_arguments_reach_trainer(monkeypatch, tmp_path):
    workspace = tmp_path / "installation with spaces" / "workspace"
    captured = []
    monkeypatch.setattr(
        Trainer, "main", lambda **kwargs: captured.append(kwargs) or 0)
    monkeypatch.setattr(
        Merger, "main", lambda **_kwargs: pytest.fail("Merger.main called"))

    _run_main(monkeypatch, [
        "train",
        "--training-data-src-dir", str(workspace / "data_src" / "aligned"),
        "--training-data-dst-dir", str(workspace / "data_dst" / "aligned"),
        "--model-dir", str(workspace / "model"),
        "--model", "Quick96",
        "--no-preview",
        "--silent-start",
    ])

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["model_class_name"] == "Quick96"
    assert kwargs["training_data_src_path"] == (
        workspace / "data_src" / "aligned").resolve()
    assert kwargs["training_data_dst_path"] == (
        workspace / "data_dst" / "aligned").resolve()
    assert kwargs["saved_models_path"] == (workspace / "model").resolve()
    assert kwargs["no_preview"] is True
    assert kwargs["silent_start"] is True
    assert kwargs["pretrained_model_path"] is None
    assert kwargs["force_gpu_idxs"] is None


def test_quick96_merge_launcher_arguments_reach_merger(monkeypatch, tmp_path):
    workspace = tmp_path / "installation with spaces" / "workspace"
    captured = []
    monkeypatch.setattr(
        Merger, "main", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(
        Trainer, "main", lambda **_kwargs: pytest.fail("Trainer.main called"))

    _run_main(monkeypatch, [
        "merge",
        "--input-dir", str(workspace / "data_dst"),
        "--output-dir", str(workspace / "data_dst" / "merged"),
        "--output-mask-dir", str(workspace / "data_dst" / "merged_mask"),
        "--aligned-dir", str(workspace / "data_dst" / "aligned"),
        "--model-dir", str(workspace / "model"),
        "--model", "Quick96",
    ])

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs == {
        "model_class_name": "Quick96",
        "saved_models_path": (workspace / "model").resolve(),
        "force_model_name": None,
        "input_path": (workspace / "data_dst").resolve(),
        "output_path": (workspace / "data_dst" / "merged").resolve(),
        "output_mask_path": (
            workspace / "data_dst" / "merged_mask").resolve(),
        "aligned_path": (workspace / "data_dst" / "aligned").resolve(),
        "force_gpu_idxs": None,
        "cpu_only": False,
    }
    assert not any("export" in key or "optimizer" in key for key in kwargs)


def test_quick96_generic_merge_normalizes_forced_gpu_indexes(
        monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    captured = []
    monkeypatch.setattr(
        Merger, "main", lambda **kwargs: captured.append(kwargs))

    _run_main(monkeypatch, [
        "merge",
        "--input-dir", str(workspace / "data_dst"),
        "--output-dir", str(workspace / "data_dst" / "merged"),
        "--output-mask-dir", str(workspace / "data_dst" / "merged_mask"),
        "--aligned-dir", str(workspace / "data_dst" / "aligned"),
        "--model-dir", str(workspace / "model"),
        "--model", "Quick96",
        "--force-gpu-idxs", "0",
    ])

    assert len(captured) == 1
    assert captured[0]["model_class_name"] == "Quick96"
    assert captured[0]["force_gpu_idxs"] == [0]
    assert type(captured[0]["force_gpu_idxs"]) is list
    assert type(captured[0]["force_gpu_idxs"][0]) is int
