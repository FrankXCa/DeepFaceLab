"""Extractor-only cuDNN TF32 scope and CPU/CUDA parity regression."""

import gc
import multiprocessing
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from core.leras import nn
from DFLIMG import DFLJPG
from facelib.FANExtractor import FANExtractor
from facelib.S3FDExtractor import S3FDExtractor
from facelib._extractor_precision import cudnn_fp32_for_extractor


ROOT = Path(__file__).resolve().parents[2]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SAMPLES = ("mini_tutorial.jpg", "deage_0_1.jpg", "political_speech1.jpg",
           "meme1.jpg")
TOL = 1e-4  # Existing Phase 9 parity limit; do not loosen.


@pytest.fixture(autouse=True)
def _preserve_rng_state():
    """Strict-load constructors initialize weights before replacing them.

    Keep those draws from shifting later training-test initialization when
    the full smoke suite runs in its normal order.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [0] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _init(device):
    nn.initialize_main_env()
    config = (nn.DeviceConfig.CPU() if device == "cpu"
              else nn.DeviceConfig.GPUIndexes([0]))
    nn.initialize(config, data_format="NHWC")


def _images():
    images = {}
    for name in SAMPLES:
        image = cv2.imread(str(ROOT / "doc" / name))
        assert image is not None, name
        images[name] = image
    images["synthetic"] = np.random.default_rng(20240501).integers(
        0, 256, size=(640, 640, 3), dtype=np.uint8)
    return images


def _s3fd_output(ext, image):
    h, w = image.shape[:2]
    d = max(w, h)
    scale = d / max(64, 640 if d >= 1280 else d / 2)
    resized = cv2.resize(image[:, :, ::-1], (int(w / scale), int(h / scale)),
                         interpolation=cv2.INTER_LINEAR)
    maps = ext.model.run([resized[None, ...]])
    boxes = []
    for i, ((cls,), (reg,)) in enumerate(zip(maps[::2], maps[1::2])):
        stride = 2 ** (i + 2)
        for row, col in zip(*np.where(cls[..., 1] > 0.05)):
            loc = reg[row, col, :]
            prior = np.array([col * stride + stride / 2,
                              row * stride + stride / 2, 4 * stride, 4 * stride])
            box = np.concatenate((prior[:2] + loc[:2] * .1 * prior[2:],
                                  prior[2:] * np.exp(loc[2:] * .2)))
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            boxes.append([*box, cls[row, col, 1]])
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    if len(boxes) == 0:
        boxes = np.zeros((1, 5), dtype=np.float64)
    scored = boxes[ext.refine_nms(boxes, 0.3), :]
    rects = np.asarray(ext.extract(image), dtype=np.int64).reshape(-1, 4)
    return maps, scored, rects


def _fan_output(ext, image, rect):
    left, top, right, bottom = rect
    center = np.asarray([(left + right) / 2, (top + bottom) / 2])
    scale = (right - left + bottom - top) / 195.0
    crop = ext.crop(image[:, :, ::-1], center, scale).astype(np.float32) / 255.0
    pred = ext.model.run([crop[None, ...]])[0]
    decoded = ext.get_pts_from_predict(pred, center, scale)
    final = ext.extract(image, [rect])[0]
    assert final is not None
    return pred, decoded, final


def test_precision_scope_restores_state_on_exception():
    class CudaLike:
        is_cuda = True

    original = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    enabled = torch.backends.cudnn.enabled
    try:
        for external in (True, False):
            torch.backends.cudnn.allow_tf32 = external
            with pytest.raises(RuntimeError, match="sentinel"):
                with cudnn_fp32_for_extractor(CudaLike()):
                    assert torch.backends.cudnn.allow_tf32 is False
                    raise RuntimeError("sentinel")
            assert torch.backends.cudnn.allow_tf32 is external
            assert torch.backends.cuda.matmul.allow_tf32 is matmul
            assert torch.backends.cudnn.enabled is enabled
    finally:
        torch.backends.cudnn.allow_tf32 = original


def test_cpu_inference_ignores_cudnn_tf32_setting():
    image = _images()["mini_tutorial.jpg"]
    original = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        _init("cpu")
        models = (S3FDExtractor(place_model_on_cpu=True),
                  FANExtractor(landmarks_3D=False, place_model_on_cpu=True),
                  FANExtractor(landmarks_3D=True, place_model_on_cpu=True))
        rect = [506, 100, 676, 348]
        for external in (True, False):
            torch.backends.cudnn.allow_tf32 = external
            outputs = (models[0].extract(image),
                       models[1].extract(image, [rect])[0],
                       models[2].extract(image, [rect])[0])
            assert torch.backends.cudnn.allow_tf32 is external
            if external:
                first = outputs
            else:
                assert first[0] == outputs[0]
                np.testing.assert_array_equal(first[1], outputs[1])
                np.testing.assert_array_equal(first[2], outputs[2])
        assert torch.backends.cuda.matmul.allow_tf32 is matmul
    finally:
        torch.backends.cudnn.allow_tf32 = original


@CUDA
def test_s3fd_cuda_parity_and_scoped_tf32():
    images = _images()
    _init("cpu")
    cpu = S3FDExtractor(place_model_on_cpu=True)
    expected = {name: _s3fd_output(cpu, image)
                for name, image in images.items()}
    del cpu
    _init("cuda")
    baseline_allocated = torch.cuda.memory_allocated()
    gpu = S3FDExtractor()
    prior = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        for name, image in images.items():
            maps, scored, rects = _s3fd_output(gpu, image)
            cpu_maps, cpu_scored, cpu_rects = expected[name]
            assert len(maps) == len(cpu_maps) == 12
            for got, want in zip(maps, cpu_maps):
                assert got.dtype == want.dtype == np.float32
                assert got.shape == want.shape
                np.testing.assert_allclose(got, want, rtol=0, atol=TOL)
            assert scored.shape == cpu_scored.shape
            np.testing.assert_allclose(scored, cpu_scored, rtol=0, atol=TOL)
            np.testing.assert_array_equal(rects, cpu_rects)
            assert torch.backends.cudnn.allow_tf32 is True
            assert torch.backends.cuda.matmul.allow_tf32 is matmul
    finally:
        torch.backends.cudnn.allow_tf32 = prior
        del gpu
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() <= baseline_allocated


@pytest.mark.parametrize("landmarks_3d", (False, True), ids=("2DFAN", "3DFAN"))
@CUDA
def test_fan_cuda_parity_and_scoped_tf32(landmarks_3d):
    images = _images()
    rects = {"mini_tutorial.jpg": [506, 100, 676, 348],
             "deage_0_1.jpg": [100, 0, 396, 365],
             "political_speech1.jpg": [846, 216, 1005, 420],
             "synthetic": [128, 128, 512, 512]}
    _init("cpu")
    cpu = FANExtractor(landmarks_3D=landmarks_3d, place_model_on_cpu=True)
    expected = {name: _fan_output(cpu, images[name], rect)
                for name, rect in rects.items()}
    del cpu
    _init("cuda")
    baseline_allocated = torch.cuda.memory_allocated()
    gpu = FANExtractor(landmarks_3D=landmarks_3d)
    prior = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        for name, rect in rects.items():
            pred, decoded, final = _fan_output(gpu, images[name], rect)
            cpu_pred, cpu_decoded, cpu_final = expected[name]
            assert pred.shape == cpu_pred.shape == (68, 64, 64)
            assert pred.dtype == cpu_pred.dtype == np.float32
            np.testing.assert_allclose(pred, cpu_pred, rtol=0, atol=TOL)
            np.testing.assert_array_equal(decoded, cpu_decoded)
            np.testing.assert_array_equal(final, cpu_final)
            assert torch.backends.cudnn.allow_tf32 is True
            assert torch.backends.cuda.matmul.allow_tf32 is matmul
    finally:
        torch.backends.cudnn.allow_tf32 = prior
        del gpu
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() <= baseline_allocated


def _used_gpu_mib():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return int(raw.splitlines()[0].strip())


@CUDA
def test_cuda_pipeline_reaps_worker_and_releases_vram(tmp_path):
    inp, out = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    out.mkdir()
    (inp / "mini_tutorial.jpg").write_bytes(
        (ROOT / "doc" / "mini_tutorial.jpg").read_bytes())
    baseline_children = len(multiprocessing.active_children())
    baseline_vram = _used_gpu_mib()
    cmd = [sys.executable, str(ROOT / "main.py"), "extract",
           "--detector", "s3fd", "--input-dir", str(inp),
           "--output-dir", str(out), "--face-type", "full_face",
           "--max-faces-from-image", "0", "--image-size", "256",
           "--jpeg-quality", "90", "--no-output-debug",
           "--force-gpu-idxs", "0", "--gpu-worker-count", "1",
           "--final-worker-count", "1"]
    result = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "Faces detected:      1" in result.stdout
    files = list(out.glob("mini_tutorial_*.jpg"))
    assert len(files) == 1
    dfl = DFLJPG.load(str(files[0]))
    assert dfl.has_data()
    assert dfl.get_source_rect().tolist() == [506, 100, 676, 348]
    assert dfl.get_landmarks().shape == (68, 2)
    assert len(multiprocessing.active_children()) == baseline_children
    deadline = time.monotonic() + 15
    while _used_gpu_mib() > baseline_vram + 32 and time.monotonic() < deadline:
        time.sleep(.5)
    assert _used_gpu_mib() <= baseline_vram + 32
