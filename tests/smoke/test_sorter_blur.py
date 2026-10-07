"""Authenticated blur-sort compatibility and fatal-error contracts."""

import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from DFLIMG import DFLIMG
from core import osex
from core.imagelib import estimate_sharpness
from core.interact import interact as io
from core.leras import nn
from mainscripts import Sorter


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
FIXTURE = ROOT / "tests" / "parity" / "fixtures" / \
    "sharpness_skimage0142_reference.npz"
GENERATOR = ROOT / "tests" / "parity" / \
    "generate_sharpness_skimage0142_reference.py"
FIXTURE_SHA256 = \
    "f5dd8f4ce8f65c8ae1427c6d6c0ad0e788e522e9b1c5d7e0591dfb494c56ff79"


sharpness_module = importlib.import_module(
    "core.imagelib.estimate_sharpness")


def _fixture_metadata():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return json.loads(str(fixture["metadata"]))


def _compatibility_intermediates(gray):
    kernel = np.array(sharpness_module._HSOBEL_WEIGHTS)
    kernel /= np.sum(abs(kernel))
    response = ndi.convolve(gray, kernel.T)
    strength_raw = np.square(response)
    threshold = np.float64(2 * np.sqrt(np.mean(strength_raw)))
    strength_thresholded = strength_raw.copy()
    strength_thresholded[strength_thresholded <= threshold] = 0
    thinned = sharpness_module._simple_thinning(strength_thresholded)
    canny = sharpness_module._canny(gray)
    widths = sharpness_module.marziliano_method(thinned, gray)
    probability_map = np.zeros(gray.shape, np.float64)
    histogram = np.zeros(101, np.float64)
    qualified = 0
    for row in range(int(gray.shape[0] / sharpness_module.BLOCK_HEIGHT)):
        for column in range(
                int(gray.shape[1] / sharpness_module.BLOCK_WIDTH)):
            rows = slice(64 * row, 64 * (row + 1))
            columns = slice(64 * column, 64 * (column + 1))
            if sharpness_module.is_edge_block(
                    canny[rows, columns], sharpness_module.THRESHOLD):
                block_widths = widths[rows, columns]
                nonzero = block_widths != 0
                contrast = sharpness_module.get_block_contrast(
                    gray[rows, columns])
                jnb = sharpness_module.WIDTH_JNB[contrast]
                probabilities = 1 - np.exp(
                    -abs(block_widths[nonzero] / jnb) **
                    sharpness_module.BETA)
                target = probability_map[rows, columns]
                target[nonzero] = probabilities
                for probability in probabilities:
                    histogram[int(round(probability * 100))] += 1
                    qualified += 1
    if qualified:
        histogram /= qualified
    return {
        "canny": canny,
        "sobel_response": response,
        "sobel_strength2_raw": strength_raw,
        "sobel_threshold": threshold,
        "sobel_strength2_thresholded": strength_thresholded,
        "sobel_thinned": thinned,
        "edge_widths": widths,
        "pblur_map": probability_map,
        "pblur_histogram": histogram,
        "qualified_edge_count": np.int64(qualified),
    }


def _write_dfl_jpg(path, image, landmarks):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    dfl = DFLIMG.load(path)
    assert dfl is not None
    height, width = image.shape[:2]
    dfl.set_dict({
        "face_type": "full_face",
        "landmarks": np.asarray(landmarks, np.float32),
        "source_filename": path.name,
        "source_rect": np.asarray([0, 0, width - 1, height - 1]),
    })
    dfl.save()
    return path


def _sort_fixture_image(index, path):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return _write_dfl_jpg(
            path, fixture[f"sort_{index:03d}_decoded"],
            fixture[f"sort_{index:03d}_landmarks"])


def test_fixture_identity_schema_and_integrity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    spec = importlib.util.spec_from_file_location("sharpness_oracle", GENERATOR)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        payload = {key: fixture[key] for key in fixture.files}
    metadata = json.loads(str(payload["metadata"]))
    validator.validate_payload(np, payload, metadata)
    assert metadata["schema"] == "sharpness-skimage0142-reference-v1"
    assert metadata["schema_version"] == 1
    assert len(payload) == 321
    assert len(metadata["subgroup_sha256"]) == 10


@pytest.mark.parametrize("case_index", range(21))
def test_all_historical_cases_match_exactly(case_index):
    prefix = f"case_{case_index:03d}_"
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        image = fixture[prefix + "input"]
        gray = fixture[prefix + "gray_float64"]
        expected_score = fixture[prefix + "score"]
        expected = {
            name: fixture[prefix + name]
            for name in (
                "canny", "sobel_response", "sobel_strength2_raw",
                "sobel_threshold", "sobel_strength2_thresholded",
                "sobel_thinned", "edge_widths", "pblur_map",
                "pblur_histogram", "qualified_edge_count")
        }
    actual = _compatibility_intermediates(gray)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name
    assert np.float64(estimate_sharpness(image)).tobytes() == \
        np.float64(expected_score).tobytes()


def test_controlled_blur_sequence_is_exact():
    metadata = _fixture_metadata()
    scores = {record["name"]: record["score"] for record in metadata["cases"]}
    assert [scores[name] for name in (
        "edge_rich_base_rank2", "edge_rich_blur_low",
        "edge_rich_blur_medium", "edge_rich_blur_high")
    ] == [1.0, 0.054310344827586204, 0.021310602024507193, 0.0]


def test_public_and_internal_sort_evidence_is_exact():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata"]))
        count = len(metadata["sort_cases"])
        public_scores = np.asarray([
            estimate_sharpness(fixture[f"sort_{index:03d}_public_input"])
            for index in range(count)], np.float64)
        final_scores = np.asarray([
            estimate_sharpness(fixture[f"sort_{index:03d}_final_input"])
            for index in range(count)], np.float64)
        assert np.array_equal(public_scores, fixture["sort_public_scores"])
        assert np.array_equal(
            final_scores, fixture["sort_best_blur_preselection_scores"])
        public_order = sorted(
            range(count), key=lambda index: public_scores[index], reverse=True)
        final_order = sorted(
            range(count), key=lambda index: final_scores[index], reverse=True)
        assert public_order == fixture["sort_public_order"].tolist()
        assert final_order == \
            fixture["sort_best_blur_preselection_order"].tolist()
        assert metadata["sort_contract"]["public_tie_order"][
            "classification"] == "OBSERVED_REFERENCE_ORDER"


def test_supported_input_forms_and_small_image_contract():
    metadata = _fixture_metadata()
    by_name = {record["name"]: record for record in metadata["cases"]}
    required = (
        "constant_black_even_rank2", "constant_white_odd_rank2",
        "single_channel_rank3", "bgr_rank3", "bgra_rank3")
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for name in required:
            record = by_name[name]
            prefix = f"case_{record['index']:03d}_"
            assert estimate_sharpness(fixture[prefix + "input"]) == \
                float(fixture[prefix + "score"])
    assert estimate_sharpness(np.zeros((63, 63), np.uint8)) == 0.0


def test_blur_worker_failure_is_tagged_and_parent_raises(
        plain_tmp, monkeypatch):
    path = _sort_fixture_image(0, Path(plain_tmp) / "worker.jpg")
    client = Sorter.BlurEstimatorSubprocessor.Cli.__new__(
        Sorter.BlurEstimatorSubprocessor.Cli)
    client.estimate_motion_blur = False
    monkeypatch.setattr(
        Sorter, "estimate_sharpness",
        lambda _image: (_ for _ in ()).throw(RuntimeError("injected blur")))
    result = client.process_data((str(path), []))
    assert result[:4] == [
        Sorter.FATAL_SCORER_FAILURE, str(path), "RuntimeError",
        "injected blur"]

    parent = Sorter.BlurEstimatorSubprocessor.__new__(
        Sorter.BlurEstimatorSubprocessor)
    parent.img_list, parent.trash_img_list, parent.fatal_errors = [], [], []
    monkeypatch.setattr(io, "progress_bar_inc", lambda *_args: None)
    parent.on_result({}, (str(path), []), result)
    with pytest.raises(Sorter.SharpnessScoringError, match="injected blur"):
        parent.get_result()


def test_final_loader_failure_is_tagged_and_parent_raises(
        plain_tmp, monkeypatch):
    path = _sort_fixture_image(0, Path(plain_tmp) / "final.jpg")
    client = Sorter.FinalLoaderSubprocessor.Cli.__new__(
        Sorter.FinalLoaderSubprocessor.Cli)
    client.faster = False
    monkeypatch.setattr(
        Sorter, "estimate_sharpness",
        lambda _image: (_ for _ in ()).throw(RuntimeError("injected final")))
    result = client.process_data([str(path)])
    assert result[:4] == [
        Sorter.FATAL_SCORER_FAILURE, str(path), "RuntimeError",
        "injected final"]

    parent = Sorter.FinalLoaderSubprocessor.__new__(
        Sorter.FinalLoaderSubprocessor)
    parent.result, parent.result_trash, parent.fatal_errors = [], [], []
    monkeypatch.setattr(io, "progress_bar_inc", lambda *_args: None)
    parent.on_result({}, [str(path)], result)
    with pytest.raises(Sorter.SharpnessScoringError, match="injected final"):
        parent.get_result()


def test_non_dfl_remains_nonfatal(plain_tmp):
    path = Path(plain_tmp) / "ordinary.jpg"
    assert cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
    blur = Sorter.BlurEstimatorSubprocessor.Cli.__new__(
        Sorter.BlurEstimatorSubprocessor.Cli)
    blur.estimate_motion_blur = False
    blur.log_err = lambda _message: None
    assert blur.process_data((str(path), [])) == [str(path), 0]
    final = Sorter.FinalLoaderSubprocessor.Cli.__new__(
        Sorter.FinalLoaderSubprocessor.Cli)
    final.faster = False
    final.log_err = lambda _message: None
    assert final.process_data([str(path)]) == [1, [str(path)]]


def test_fatal_sort_never_reaches_final_process_or_mutates_files(
        plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "partial activity"
    paths = [_sort_fixture_image(index, root / f"{index}.jpg")
             for index in range(2)]
    before = {path.name: path.read_bytes() for path in paths}

    def failing_sort(_input_path):
        estimate_sharpness(np.zeros((64, 64), np.uint8))
        raise Sorter.SharpnessScoringError("injected after activity")

    monkeypatch.setitem(
        Sorter.sort_func_methods, "blur", ("blur", failing_sort))
    monkeypatch.setattr(
        Sorter, "final_process",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("final_process reached")))
    with pytest.raises(
            Sorter.SharpnessScoringError, match="after activity"):
        Sorter.main(root, "blur")
    assert {path.name: path.read_bytes() for path in paths} == before
    assert not (root.parent / (root.name + "_trash")).exists()


def test_real_argparse_blur_dispatch_with_spaces(plain_tmp, monkeypatch):
    root = Path(plain_tmp) / "unrelated cwd" / "aligned faces"
    paths = [_sort_fixture_image(index, root / f"face {index}.jpg")
             for index in range(4)]
    expected_scores = {}
    for path in paths:
        dfl = DFLIMG.load(path)
        image = cv2.imread(str(path))
        mask = Sorter.LandmarksProcessor.get_image_hull_mask(
            image.shape, dfl.get_landmarks())
        expected_scores[path.read_bytes()] = estimate_sharpness(
            (image * mask).astype(np.uint8))
    expected = [item[0] for item in sorted(
        expected_scores.items(), key=lambda item: item[1], reverse=True)]

    def one_worker(self):
        yield "CPU0", {}, {"estimate_motion_blur": self.estimate_motion_blur}

    monkeypatch.setattr(
        Sorter.BlurEstimatorSubprocessor, "process_info_generator", one_worker)
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nn, "initialize_main_env", lambda: None)
    monkeypatch.setattr(osex, "set_process_lowest_prio", lambda: None)
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "sort", "--input-dir", str(root),
                      "--by", "blur"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MAIN_PY), run_name="__main__")
    assert exc_info.value.code == 0
    outputs = sorted(root.glob("*.jpg"))
    assert [path.name for path in outputs] == [
        "00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
    assert [path.read_bytes() for path in outputs] == expected


def test_supported_blur_path_has_no_live_skimage_or_new_heavy_dependency():
    scorer_source = Path(sharpness_module.__file__).read_text(encoding="utf-8")
    sorter_source = Path(Sorter.__file__).read_text(encoding="utf-8")
    assert "from skimage" not in scorer_source
    assert "import skimage" not in scorer_source
    for forbidden in ("import tensorflow", "import h5py"):
        assert forbidden not in scorer_source
        assert forbidden not in sorter_source
    assert importlib.util.find_spec("skimage") is None


def test_final_fast_entrypoint_is_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(
        Sorter, "sort_best", lambda path, faster=False: seen.append(
            (path, faster)) or ([], []))
    marker = Path("synthetic")
    assert Sorter.sort_best_faster(marker) == ([], [])
    assert seen == [(marker, True)]


def test_final_fast_loader_error_classification_is_unchanged(
        plain_tmp, monkeypatch):
    path = _sort_fixture_image(0, Path(plain_tmp) / "fast.jpg")
    client = Sorter.FinalLoaderSubprocessor.Cli.__new__(
        Sorter.FinalLoaderSubprocessor.Cli)
    client.faster = True
    client.log_err = lambda _message: None
    monkeypatch.setattr(Sorter, "cv2_imread", lambda _path: None)
    assert client.process_data([str(path)]) == [1, [str(path)]]
