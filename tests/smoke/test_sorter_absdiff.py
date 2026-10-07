"""Torch-native absolute-difference sorter contracts."""

import multiprocessing
import runpy
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from DFLIMG import DFLIMG
from core import osex
from core.interact import interact as io
from core.leras import nn
from core.leras.device import DeviceConfig
from mainscripts import Sorter


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"


def _write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


def _images(root, values, shape=(3, 5, 3)):
    return [
        _write(root / f"{index:02d}.png",
               np.full(shape, value, dtype=np.uint8))
        for index, value in enumerate(values)
    ]


def _numpy_scores(i_images, j_images):
    return np.asarray([
        [np.sum(np.abs(i.astype(np.float32) - j.astype(np.float32)),
                dtype=np.float32) for i in i_images]
        for j in j_images
    ], dtype=np.float32)


def _state():
    values = (
        nn.current_DeviceConfig, nn.device, nn.data_format,
        nn.conv2d_ch_axis, nn.conv2d_spatial_axes, nn.floatx,
    )
    return tuple(list(value) if isinstance(value, list) else value
                 for value in values)


def _cpu_order(paths, similar=True, block_size=2):
    Sorter._validate_absdiff_images(paths)
    return Sorter._sort_absdiff_image_paths(
        paths, similar, torch.device("cpu"), torch, block_size=block_size)


def test_score_formula_exact_float32_and_no_uint8_wraparound():
    zero = np.zeros((3, 5, 3), np.uint8)
    uniform = np.full_like(zero, 10)
    sparse = zero.copy()
    sparse[1, 2] = (255, 2, 1)
    channel = zero.copy()
    channel[..., 2] = 7
    maximum = np.full_like(zero, 255)
    images = [zero, uniform, sparse, channel, maximum]
    actual = Sorter._score_absdiff_blocks(
        images, [zero], torch.device("cpu"), torch)[0]
    expected = _numpy_scores(images, [zero])[0]
    assert np.array_equal(actual, expected)
    assert actual.tolist() == [0.0, 450.0, 258.0, 105.0, 11475.0]


def test_score_formula_odd_shape_and_multiple_reference_rows():
    rng = np.random.default_rng(1401)
    images = [rng.integers(0, 256, (3, 5, 4), dtype=np.uint8)
              for _ in range(4)]
    actual = Sorter._score_absdiff_blocks(
        images[:3], images[2:], torch.device("cpu"), torch)
    expected = _numpy_scores(images[:3], images[2:])
    assert np.array_equal(actual, expected)


def test_similar_and_dissimilar_are_greedy_from_first_path(plain_tmp):
    paths = _images(Path(plain_tmp), [0, 10, 11, 30])
    assert _cpu_order(paths, True) == [0, 1, 2, 3]
    assert _cpu_order(paths, False) == [0, 3, 1, 2]


def test_ties_characterize_current_argsort_without_stability_claim(plain_tmp):
    paths = _images(Path(plain_tmp), [0, 10, 10, 20])
    first = _cpu_order(paths, True)
    second = _cpu_order(paths, True)
    assert first == second == [0, 1, 2, 3]


def test_multiple_blocks_and_symmetric_zero_diagonal_storage(
        plain_tmp, monkeypatch):
    paths = _images(Path(plain_tmp), [0, 2, 5, 9, 14])
    seen = []
    real_score = Sorter._score_absdiff_blocks

    def recording_score(i_images, j_images, device, torch_module):
        result = real_score(i_images, j_images, device, torch_module)
        seen.append((len(i_images), len(j_images), result.copy()))
        return result

    monkeypatch.setattr(Sorter, "_score_absdiff_blocks", recording_score)
    assert _cpu_order(paths, True, block_size=2) == [0, 1, 2, 3, 4]
    assert len(seen) == 6
    assert all(i <= 2 and j <= 2 for i, j, _ in seen)
    assert all(block.dtype == np.float32 for _, _, block in seen)


def test_empty_and_single_input_are_trivial_without_device_prompt(
        plain_tmp, monkeypatch):
    monkeypatch.setattr(
        nn.DeviceConfig, "ask_choose_device",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("prompted")))
    root = Path(plain_tmp)
    assert Sorter.sort_by_absdiff(root) == ([], [])
    only = _images(root, [7])[0]
    assert Sorter.sort_by_absdiff(root) == ([(only,)], [])


@pytest.mark.parametrize("kind", ("corrupt", "rank", "shape"))
def test_invalid_complete_input_fails_before_rename_or_temp(
        plain_tmp, monkeypatch, kind):
    root = Path(plain_tmp)
    baseline = _state()
    first = Path(_images(root, [0])[0])
    if kind == "corrupt":
        bad = root / "01.png"
        bad.write_bytes(b"not an image")
    elif kind == "rank":
        bad = Path(_write(root / "01.png", np.zeros((3, 5), np.uint8)))
    else:
        bad = Path(_write(root / "01.png", np.zeros((4, 5, 3), np.uint8)))
    before = sorted(path.name for path in root.iterdir())
    monkeypatch.setattr(
        Sorter.tempfile, "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("temporary storage created before validation")))
    with pytest.raises(ValueError, match="absdiff|decode"):
        Sorter.sort_by_absdiff(root)
    assert sorted(path.name for path in root.iterdir()) == before
    assert first.exists() and bad.exists()
    assert _state() == baseline


def test_temp_memmap_is_removed_on_success_scoring_and_selection_failure(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp)
    paths = _images(root / "images", [0, 3, 8])
    cache = root / "cache"
    cache.mkdir()
    real_mkstemp = tempfile.mkstemp
    created = []

    def local_mkstemp(**kwargs):
        fd, name = real_mkstemp(dir=cache, **kwargs)
        created.append(Path(name))
        return fd, name

    monkeypatch.setattr(Sorter.tempfile, "mkstemp", local_mkstemp)
    baseline = _state()
    assert _cpu_order(paths) == [0, 1, 2]
    assert created and all(not path.exists() for path in created)
    assert _state() == baseline

    real_score = Sorter._score_absdiff_blocks
    monkeypatch.setattr(
        Sorter, "_score_absdiff_blocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        _cpu_order(paths)
    assert all(not path.exists() for path in created)
    assert _state() == baseline

    monkeypatch.setattr(Sorter, "_score_absdiff_blocks", real_score)
    monkeypatch.setattr(
        Sorter, "_greedy_absdiff_order",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("selection boom")))
    with pytest.raises(RuntimeError, match="selection boom"):
        _cpu_order(paths)
    assert all(not path.exists() for path in created)
    assert _state() == baseline


def test_sort_renames_only_and_preserves_bytes_and_dfl_metadata(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "aligned faces"
    original = {}
    for name, value in (("a.jpg", 0), ("b.jpg", 20), ("c.jpg", 21)):
        path = Path(_write(root / name, np.full((16, 16, 3), value, np.uint8)))
        dfl = DFLIMG.load(path)
        dfl.set_dict({"face_type": "full_face", "source_filename": name})
        dfl.save()
        original[name] = path.read_bytes()

    monkeypatch.setattr(io, "input_bool", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        nn.DeviceConfig, "ask_choose_device",
        lambda **_kwargs: DeviceConfig.CPU())
    baseline = _state()
    Sorter.main(root, "absdiff")
    assert _state() == baseline
    outputs = sorted(root.glob("*.jpg"))
    assert [path.name for path in outputs] == [
        "00000.jpg", "00001.jpg", "00002.jpg"]
    assert {path.read_bytes() for path in outputs} == set(original.values())
    assert {DFLIMG.load(path).get_source_filename() for path in outputs} == {
        "a.jpg", "b.jpg", "c.jpg"}


def test_real_argparse_dispatch_reaches_production_sorter(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "cli images"
    _images(root, [0, 8, 9])
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.setattr(io, "input_bool", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        nn.DeviceConfig, "ask_choose_device",
        lambda **_kwargs: DeviceConfig.CPU())
    monkeypatch.setattr(
        sys, "argv", ["main.py", "sort", "--input-dir", str(root),
                      "--by", "absdiff"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MAIN_PY), run_name="__main__")
    assert exc_info.value.code == 0
    assert [path.name for path in sorted(root.glob("*.png"))] == [
        "00000.png", "00001.png", "00002.png"]


def test_source_has_no_tensorflow_h5py_or_leras_initialize_dependency():
    source = Path(Sorter.__file__).read_text(encoding="utf-8")
    block = source.split("def sort_by_absdiff", 1)[1].split(
        "def final_process", 1)[0]
    for forbidden in ("nn.tf", "tf_sess", "tf.placeholder", "import h5py",
                      "nn.initialize"):
        assert forbidden not in block


def test_unrelated_black_sort_mode_is_unchanged(plain_tmp):
    root = Path(plain_tmp)
    paths = _images(root, [0, 255])
    ordered, trash = Sorter.sort_by_black(root)
    assert [Path(item[0]).name for item in ordered] == ["01.png", "00.png"]
    assert trash == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_numerics_determinism_and_cpu_order_agreement(plain_tmp):
    root = Path(plain_tmp)
    paths = _images(root, [0, 10, 11, 30], shape=(7, 9, 3))
    decoded = [Sorter._read_absdiff_image(path) for path in paths]
    expected = _numpy_scores(decoded, decoded)
    cuda = torch.device("cuda:0")
    baseline = _state()
    first = Sorter._score_absdiff_blocks(decoded, decoded, cuda, torch)
    second = Sorter._score_absdiff_blocks(decoded, decoded, cuda, torch)
    max_deviation = float(np.max(np.abs(first - expected)))
    assert max_deviation == 0.0
    assert np.array_equal(first, second)
    for similar in (True, False):
        cpu_order = _cpu_order(paths, similar)
        cuda_order = Sorter._sort_absdiff_image_paths(
            paths, similar, cuda, torch, block_size=2)
        assert cuda_order == cpu_order
    assert _state() == baseline
