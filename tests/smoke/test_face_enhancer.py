"""Authenticated FaceEnhancer Torch backend and parity contracts."""

import hashlib
import importlib
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.leras import nn
from facelib.FaceEnhancer import (
    CHECKPOINT_SHA256, FaceEnhancer, legacy_resize2d,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/parity/fixtures/face_enhancer_tf_reference.npz"
CHECKPOINT_PATH = ROOT / "facelib/FaceEnhancer.npy"
face_enhancer_module = importlib.import_module("facelib.FaceEnhancer")
NN_STATE = (
    "current_DeviceConfig", "device", "floatx", "data_format",
    "conv2d_ch_axis", "conv2d_spatial_axes",
)


def _fixture():
    fixture = np.load(FIXTURE_PATH, allow_pickle=False)
    return fixture, json.loads(str(fixture["metadata"]))


def _state():
    return {name: getattr(nn, name) for name in NN_STATE}


def _assert_state_unchanged(before):
    for name, value in before.items():
        actual = getattr(nn, name)
        if isinstance(value, (str, int, float, type(None), tuple)):
            assert actual == value, name
        else:
            assert actual is value, name


def _parameter_snapshot(enhancer):
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in enhancer.model.state_dict().items()
    }


def _array_digest(value):
    value = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": value.dtype.name, "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def cpu_enhancer():
    before = _state()
    enhancer = FaceEnhancer(place_model_on_cpu=True, run_on_cpu=True)
    _assert_state_unchanged(before)
    assert enhancer.model_device == torch.device("cpu")
    assert enhancer.compute_device == torch.device("cpu")
    assert {parameter.device.type for parameter in
            enhancer.model.parameters()} == {"cpu"}
    return enhancer


def test_checkpoint_identity_inventory_mapping_and_strict_shapes():
    raw = CHECKPOINT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CHECKPOINT_SHA256
    values = pickle.loads(raw)
    tensors = FaceEnhancer._validate_checkpoint_values(values)
    assert len(values) == 72
    assert len(tensors) == 72
    assert sum(key.endswith("/bias:0") for key in values) == 35
    assert sum(key.endswith("/weight:0") and not key.startswith("dense")
               for key in values) == 35
    assert sum(key.startswith("dense") for key in values) == 2
    assert all(isinstance(value, np.ndarray) for value in values.values())
    assert all(value.dtype == np.float16 for value in values.values())
    for key, value in values.items():
        if key.endswith("/bias:0"):
            assert value.ndim == 4 and value.shape[:3] == (1, 1, 1)
            assert tensors[key].shape == (value.shape[-1],)
        elif key.startswith("dense"):
            assert value.shape == (1, 64)
            assert tensors[key].shape == (64, 1)
        else:
            assert tensors[key].shape == (
                value.shape[3], value.shape[2], value.shape[0], value.shape[1])

    bias_key = "conv1/bias:0"
    for wrong_shape in ((64,), (1, 64), (1, 1, 64), (64, 1, 1, 1)):
        bad = dict(values)
        bad[bias_key] = np.zeros(wrong_shape, np.float16)
        with pytest.raises(RuntimeError, match="invalid checkpoint bias"):
            FaceEnhancer._validate_checkpoint_values(bad)

    for mutation, match in (
        (lambda item: item.pop("conv1/bias:0"), "key inventory"),
        (lambda item: item.update({"extra:0": np.zeros(1, np.float16)}),
         "key inventory"),
        (lambda item: item.update({bias_key: np.zeros((1, 1, 1, 64),
                                                     np.float32)}),
         "invalid checkpoint bias"),
        (lambda item: item.update({bias_key: [0] * 64}),
         "non-array checkpoint value"),
    ):
        bad = dict(values)
        mutation(bad)
        with pytest.raises(RuntimeError, match=match):
            FaceEnhancer._validate_checkpoint_values(bad)


def test_checkpoint_authentication_precedes_pickle(monkeypatch):
    called = False

    def forbidden(_raw):
        nonlocal called
        called = True
        raise AssertionError("pickle must not run")

    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(RuntimeError, match="before deserialization"):
        FaceEnhancer._decode_checkpoint(b"not the checkpoint")
    assert called is False


@pytest.mark.parametrize(
    ("key", "non_finite"),
    [
        ("conv1/weight:0", np.nan),
        ("conv1/bias:0", np.nan),
        ("dense1/weight:0", np.nan),
        ("e0_conv0/weight:0", np.inf),
        ("dense2/weight:0", -np.inf),
    ],
)
def test_checkpoint_rejects_non_finite_tensor_values(key, non_finite):
    values = pickle.loads(CHECKPOINT_PATH.read_bytes())
    original = values[key]
    malformed = original.copy()
    malformed.flat[0] = non_finite
    values[key] = malformed

    assert malformed.dtype == original.dtype == np.float16
    assert malformed.shape == original.shape
    with pytest.raises(RuntimeError, match=rf"{key}.*non-finite"):
        FaceEnhancer._validate_checkpoint_values(values)


def test_failed_reload_does_not_mutate_live_model(cpu_enhancer, plain_tmp):
    before = _parameter_snapshot(cpu_enhancer)
    invalid = Path(plain_tmp) / "invalid-face-enhancer.npy"
    invalid.write_bytes(CHECKPOINT_PATH.read_bytes() + b"invalid")
    with pytest.raises(RuntimeError, match="authentication failed"):
        cpu_enhancer.load_checkpoint(invalid)
    assert _parameter_snapshot(cpu_enhancer) == before


def test_non_finite_reload_does_not_mutate_live_model(
        cpu_enhancer, plain_tmp, monkeypatch):
    before_model = cpu_enhancer.model
    before_parameters = _parameter_snapshot(cpu_enhancer)
    before_devices = {
        name: value.device
        for name, value in cpu_enhancer.model.state_dict().items()
    }
    values = pickle.loads(CHECKPOINT_PATH.read_bytes())
    key = "conv1/weight:0"
    values[key] = values[key].copy()
    values[key].flat[0] = np.nan
    payload = pickle.dumps(values)
    invalid = Path(plain_tmp) / "non-finite-face-enhancer.npy"
    invalid.write_bytes(payload)

    # Reach the post-authentication strict validator with a controlled test
    # digest; production loading continues to require CHECKPOINT_SHA256.
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    decode = FaceEnhancer._decode_checkpoint.__func__

    def decode_test_checkpoint(cls, raw):
        return decode(cls, raw, expected_sha256=expected_sha256)

    monkeypatch.setattr(
        FaceEnhancer, "_decode_checkpoint", classmethod(decode_test_checkpoint))
    with pytest.raises(RuntimeError, match=rf"{key}.*non-finite"):
        cpu_enhancer.load_checkpoint(invalid)

    assert cpu_enhancer.model is before_model
    assert _parameter_snapshot(cpu_enhancer) == before_parameters
    assert {
        name: value.device
        for name, value in cpu_enhancer.model.state_dict().items()
    } == before_devices


def test_all_legacy_resize_fixture_cases():
    fixture, metadata = _fixture()
    maxima = []
    try:
        for record in metadata["resize_cases"]:
            name = record["name"]
            source = torch.from_numpy(fixture[f"{name}_input"])[None, None]
            actual = legacy_resize2d(
                source, size=record["requested_size"])[0, 0].numpy()
            expected = fixture[f"{name}_output"]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7,
                                       err_msg=name)
            maxima.append(float(np.max(np.abs(actual - expected))))
    finally:
        fixture.close()
    assert len(maxima) == 27
    assert max(maxima) == 1.1920928955078125e-7


def test_raw_cpu_parity_and_global_state(cpu_enhancer):
    fixture, _ = _fixture()
    before = _state()
    started = time.perf_counter()
    try:
        actual = cpu_enhancer.infer_raw(fixture["raw_model_input"])
        expected = fixture["raw_model_output"]
        difference = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-5)
    finally:
        fixture.close()
    _assert_state_unchanged(before)
    assert actual.shape == (1, 768, 768, 3)
    assert actual.dtype == np.float32
    print({"cpu_raw_max_abs": difference,
           "cpu_raw_seconds": time.perf_counter() - started})


def test_raw_model_uses_exact_nine_resize_stages(cpu_enhancer, monkeypatch):
    fixture, _ = _fixture()
    calls = []
    original = face_enhancer_module.legacy_resize2d

    def recording_resize(value, *args, **kwargs):
        output = original(value, *args, **kwargs)
        calls.append((list(value.shape), list(output.shape)))
        return output

    monkeypatch.setattr(face_enhancer_module, "legacy_resize2d",
                        recording_resize)
    try:
        cpu_enhancer.infer_raw(fixture["raw_model_input"])
    finally:
        fixture.close()
    assert calls == [
        ([1, 512, 6, 6], [1, 512, 12, 12]),
        ([1, 512, 12, 12], [1, 512, 24, 24]),
        ([1, 512, 24, 24], [1, 512, 48, 48]),
        ([1, 288, 48, 48], [1, 288, 96, 96]),
        ([1, 160, 96, 96], [1, 160, 192, 192]),
        ([1, 96, 192, 192], [1, 96, 384, 384]),
        ([1, 3, 192, 192], [1, 3, 384, 384]),
        ([1, 72, 384, 384], [1, 72, 768, 768]),
        ([1, 3, 384, 384], [1, 3, 768, 768]),
    ]


def test_public_pipeline_cpu_parity(cpu_enhancer):
    fixture, metadata = _fixture()
    before = _state()
    measurements = {}
    try:
        for case in metadata["pipeline_cases"]:
            name = case["name"]
            result = cpu_enhancer._run_pipeline(
                fixture[f"{name}_input"], is_tanh=case["is_tanh"],
                preserve_size=case["preserve_size"])
            assert result["padding"] == tuple(
                case["padding_top_bottom_left_right"])
            assert result["x_coordinates"] == case["x_coordinates"]
            assert result["y_coordinates"] == case["y_coordinates"]
            np.testing.assert_array_equal(
                result["preprocessed"], fixture[f"{name}_preprocessed"])
            np.testing.assert_array_equal(
                result["padded"], fixture[f"{name}_padded"])
            final_diff = float(np.max(np.abs(
                result["final"] - fixture[f"{name}_final"])))
            final_atol = 2e-5
            np.testing.assert_allclose(
                result["final"], fixture[f"{name}_final"],
                rtol=0, atol=final_atol, err_msg=f"{name} final")
            field_differences = {"final": final_diff}
            for field in ("accumulated", "divisor", "stitched", "cropped",
                          "preserved"):
                flat = result[field].reshape(-1)
                probes = np.asarray(
                    [flat[0], flat[len(flat) // 2], flat[-1]], np.float64)
                expected_probes = fixture[f"{name}_{field}_probes"]
                tolerance = 5e-5 if field in ("stitched", "cropped") else 5e-5
                np.testing.assert_allclose(
                    probes, expected_probes, rtol=0, atol=tolerance,
                    err_msg=f"{name} {field} probes")
                field_differences[field] = float(np.max(
                    np.abs(probes - expected_probes)))
            if f"{name}_preserved" in fixture.files:
                np.testing.assert_allclose(
                    result["preserved"], fixture[f"{name}_preserved"],
                    rtol=0, atol=2e-5, err_msg=f"{name} preserve-size")
            measurements[name] = field_differences
        # Exercise the public boundary independently of the private details
        # return and prove it is exactly the same final array.
        public = cpu_enhancer.enhance(fixture["exact_normal_input"],
                                      is_tanh=False, preserve_size=False)
        np.testing.assert_allclose(
            public, fixture["exact_normal_final"], rtol=0, atol=2e-5)
    finally:
        fixture.close()
    _assert_state_unchanged(before)
    print({"cpu_pipeline_max_abs": measurements})


def test_device_selection_fallback_and_cpu_authority(monkeypatch):
    selected = SimpleNamespace(devices=[SimpleNamespace(index=0)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert FaceEnhancer._resolve_compute_device(False, selected).type == "cpu"
    assert FaceEnhancer._resolve_compute_device(True, selected).type == "cpu"
    assert FaceEnhancer._resolve_compute_device(False, None).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA required")
def test_cuda_same_device_split_placement_parity_and_determinism():
    selected = SimpleNamespace(devices=[SimpleNamespace(index=0)])
    fixture, _ = _fixture()
    before = _state()
    try:
        same = FaceEnhancer(
            place_model_on_cpu=False, run_on_cpu=False,
            device_config=selected)
        assert same.model_device == torch.device("cuda", 0)
        assert same.compute_device == torch.device("cuda", 0)
        assert {p.device.type for p in same.model.parameters()} == {"cuda"}
        started = time.perf_counter()
        first = same.infer_raw(fixture["raw_model_input"])
        second = same.infer_raw(fixture["raw_model_input"])
        torch.cuda.synchronize(0)
        same_seconds = time.perf_counter() - started
        same_difference = float(np.max(np.abs(
            first - fixture["raw_model_output"])))
        assert first.shape == fixture["raw_model_output"].shape
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)

        del same
        torch.cuda.empty_cache()
        split = FaceEnhancer(
            place_model_on_cpu=True, run_on_cpu=False,
            device_config=selected)
        assert split.model_device == torch.device("cpu")
        assert split.compute_device == torch.device("cuda", 0)
        assert {p.device.type for p in split.model.parameters()} == {"cpu"}
        started = time.perf_counter()
        split_actual = split.infer_raw(fixture["raw_model_input"])
        torch.cuda.synchronize(0)
        split_seconds = time.perf_counter() - started
        assert split_actual.shape == fixture["raw_model_output"].shape
        assert split_actual.dtype == np.float32
        assert np.isfinite(split_actual).all()
        assert {p.device.type for p in split.model.parameters()} == {"cpu"}
        print({
            "cuda_same_raw_max_abs": same_difference,
            "cuda_same_repeated_spread": float(np.max(np.abs(first-second))),
            "cuda_same_two_runs_seconds": same_seconds,
            "cuda_split_raw_max_abs": float(np.max(np.abs(
                split_actual - fixture["raw_model_output"]))),
            "cuda_split_seconds": split_seconds,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
        })
    finally:
        fixture.close()
    _assert_state_unchanged(before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA required")
def test_cuda_public_pipeline_measurements_are_stable():
    selected = SimpleNamespace(devices=[SimpleNamespace(index=0)])
    fixture, metadata = _fixture()
    enhancer = FaceEnhancer(
        place_model_on_cpu=False, run_on_cpu=False, device_config=selected)
    measurements = {}
    try:
        for case in metadata["pipeline_cases"]:
            name = case["name"]
            result = enhancer._run_pipeline(
                fixture[f"{name}_input"], is_tanh=case["is_tanh"],
                preserve_size=case["preserve_size"])
            actual = result["final"]
            expected = fixture[f"{name}_final"]
            assert actual.shape == expected.shape
            assert actual.dtype == np.float32
            assert np.isfinite(actual).all()
            measurements[name] = float(np.max(np.abs(actual - expected)))
        first = enhancer.enhance(
            fixture["exact_normal_input"], preserve_size=False)
        second = enhancer.enhance(
            fixture["exact_normal_input"], preserve_size=False)
        assert np.array_equal(first, second)
    finally:
        fixture.close()
    print({"cuda_pipeline_final_max_abs": measurements,
           "cuda_pipeline_repeated_spread": 0.0})


@pytest.mark.parametrize("use_cuda", [False, True])
def test_real_faceenhancer_masked_merger_integration(
        plain_tmp, use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("physical CUDA required")
    from tests.smoke.test_merge_contracts import (
        _const_predictor, _masked_cfg, _merge_frame,
    )

    selected = (SimpleNamespace(devices=[SimpleNamespace(index=0)])
                if use_cuda else None)
    enhancer = FaceEnhancer(
        place_model_on_cpu=not use_cuda,
        run_on_cpu=not use_cuda,
        device_config=selected)
    calls = []

    def enhance(*args, **kwargs):
        calls.append((kwargs["is_tanh"], kwargs["preserve_size"]))
        return enhancer.enhance(*args, **kwargs)

    root = Path(plain_tmp) / f"real-faceenhancer-merge-{'cuda' if use_cuda else 'cpu'}"
    root.mkdir()
    output, _ = _merge_frame(
        root, _const_predictor(face_value=0.4),
        _masked_cfg(mask_mode=0, super_res=25), enhancer=enhance)
    assert calls == [(True, False)]
    assert output.shape == (256, 256, 4)
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()
    expected_device = "cuda" if use_cuda else "cpu"
    assert {parameter.device.type for parameter in
            enhancer.model.parameters()} == {expected_device}


def test_quick96_faceenhancer_cpu_interleaving(plain_tmp):
    from tests.smoke.test_model_quick96 import (
        _make_merge_checkpoint, _make_model,
    )

    # Import only the established Quick96 checkpoint builders; inference is
    # the real accepted model/predictor path and the FaceEnhancer is production.
    root = Path(plain_tmp) / "quick96-faceenhancer-interleave"
    quick_fixture, _, expected = _make_merge_checkpoint(root)
    quick = _make_model(root, is_training=False)
    enhancer_fixture, _ = _fixture()
    try:
        enhancer = FaceEnhancer(place_model_on_cpu=True, run_on_cpu=True)
        before_state = _state()
        before_devices = {
            component.name: tuple(value.device for value in component.get_weights())
            for component in quick._quick96_components
        }
        before_parameters = {
            component.name: [hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for value in component.get_weights()]
            for component in quick._quick96_components
        }
        assert before_parameters == expected
        quick_input = quick_fixture["warped_dst"][0]
        enhancer_input = enhancer_fixture["exact_normal_input"]

        q1 = quick.predictor_func(quick_input)
        e1 = enhancer.enhance(enhancer_input, preserve_size=False)
        q2 = quick.predictor_func(quick_input)
        e2 = enhancer.enhance(enhancer_input, preserve_size=False)
        q3 = quick.predictor_func(quick_input)
        e3 = enhancer.enhance(enhancer_input, preserve_size=False)
        for left, right in zip(q1, q2):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(q1, q3):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(e1, e3)
        assert before_devices == {
            component.name: tuple(value.device for value in component.get_weights())
            for component in quick._quick96_components
        }
        assert before_parameters == {
            component.name: [hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for value in component.get_weights()]
            for component in quick._quick96_components
        }
        assert {p.device.type for p in enhancer.model.parameters()} == {"cpu"}
        _assert_state_unchanged(before_state)
    finally:
        enhancer_fixture.close()
        quick_fixture.close()
        quick.finalize()
