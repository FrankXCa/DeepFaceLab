r"""Generate the authenticated historical blur-sort sharpness oracle.

This program must be executed by the Python interpreter contained in the
official packaged reference.  It authenticates all historical Python inputs
before importing the scorer.  The resulting NPZ contains synthetic data only.

Example (PowerShell)::

    <reference-root>\_internal\python-3.6.8\python.exe `
      tests\parity\generate_sharpness_skimage0142_reference.py `
      --reference-root <reference-root> `
      --output tests\parity\fixtures\sharpness_skimage0142_reference.npz
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.abc
import importlib.util
import io
import json
import marshal
import os
import pickle
import platform
import py_compile
import re
import struct
import sys
import tempfile
import types
import zipfile
from collections import OrderedDict
from pathlib import Path


SCHEMA = "sharpness-skimage0142-reference-v1"
SCHEMA_VERSION = 1
GENERATOR_VERSION = 2
UPSTREAM_COMMIT = "e4b7543ffa1d73b26fce1e31852727f658ba490c"
EXPECTED_ENVIRONMENT = OrderedDict([
    ("python", "3.6.8"),
    ("numpy", "1.19.3"),
    ("scipy", "1.4.1"),
    ("scikit-image", "0.14.2"),
    ("opencv", "4.1.0"),
])
EXPECTED_CANNY_CONTRACT = {
    "sigma": 1.0,
    "low_threshold": None,
    "high_threshold": None,
    "effective_float64_low_threshold": 0.1,
    "effective_float64_high_threshold": 0.2,
    "mask": None,
    "use_quantiles": False,
    "gaussian_boundary_mode": "constant",
    "nonmaximum_suppression": "scikit-image 0.14.2",
    "hysteresis_connectivity": 8,
    "border_exclusion_pixels": 1,
}
EXPECTED_CPBD_CONTRACT = {
    "historical_hsobel_weights": [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    "historical_edges_module_divisor": 4.0,
    "scorer_second_normalization_sum_abs": 2.0,
    "effective_kernel_divisor": 8.0,
    "block_shape": [64, 64],
    "edge_block_threshold": 0.002,
    "beta": 3.6,
    "width_jnb_source": "historical_estimate_sharpness.WIDTH_JNB",
    "score_histogram_buckets_included": [0, 63],
}
EXPECTED_PRIVACY = {
    "synthetic_inputs_only": True,
    "user_media": False,
    "absolute_paths_serialized": False,
    "reference_paths_serialized": False,
    "credentials_or_private_urls_serialized": False,
}
EXPECTED_CASE_NAMES = (
    "constant_black_even_rank2", "constant_white_odd_rank2",
    "single_pixel_impulse", "horizontal_hard_edge_even",
    "vertical_hard_edge_even", "diagonal_hard_edge_odd",
    "checkerboard_cell_02", "checkerboard_cell_08",
    "checkerboard_cell_16", "repeated_bars_controlled_widths",
    "smooth_gradient_odd", "edge_rich_base_rank2",
    "edge_rich_blur_low", "edge_rich_blur_medium",
    "edge_rich_blur_high", "face_like_base_odd",
    "face_like_blur_low_odd", "face_like_blur_high_odd",
    "single_channel_rank3", "bgr_rank3", "bgra_rank3",
)
EXPECTED_SORT_FILENAMES = (
    "sort_even_sharp.jpg", "sort_odd_blur_low.jpg",
    "sort_even_blur_medium.jpg", "sort_odd_blur_high.jpg",
    "sort_tie_a.jpg", "sort_tie_b.jpg",
)
EXPECTED_SELF_TEST_KEYS = (
    "wrong_python_version_rejected", "wrong_numpy_version_rejected",
    "wrong_scipy_version_rejected", "wrong_scikit_image_version_rejected",
    "wrong_opencv_version_rejected", "wrong_source_digest_rejected",
    "substituted_pyc_bypass_rejected", "altered_canny_subgroup_rejected",
    "altered_cpbd_subgroup_rejected", "altered_final_score_subgroup_rejected",
    "altered_public_sort_score_rejected", "altered_public_sort_order_rejected",
    "altered_public_sort_mapping_rejected",
    "altered_internal_final_score_rejected",
    "altered_internal_final_order_rejected",
    "altered_internal_mapping_rejected", "altered_source_metadata_rejected",
    "altered_environment_metadata_rejected", "altered_schema_rejected",
    "altered_schema_version_rejected", "altered_generator_self_test_rejected",
    "altered_canny_sigma_rejected", "altered_canny_threshold_rejected",
    "altered_canny_boundary_rejected", "altered_canny_hysteresis_rejected",
    "altered_cpbd_block_size_rejected", "altered_cpbd_beta_rejected",
    "altered_cpbd_sobel_rejected", "altered_cpbd_threshold_rejected",
    "altered_score_positive_count_rejected",
    "altered_score_distinct_count_rejected", "altered_score_min_rejected",
    "altered_score_max_rejected", "altered_scalar_tolerance_rejected",
    "altered_privacy_metadata_rejected", "fixed_zip_metadata",
)
AUTHORITATIVE_METADATA_KEYS = (
    "schema", "schema_version", "generator_version", "upstream_commit",
    "environment", "opencv_version", "authenticated_sources",
    "authentication_order", "canny_contract", "cpbd_contract", "cases",
    "sort_cases", "sort_public_order_filenames",
    "sort_best_blur_preselection_order_filenames", "sort_contract",
    "score_summary", "scalar_compatibility_tolerances", "privacy",
    "generator_self_tests", "subgroup_sha256",
)
AUTHORITATIVE_METADATA_COVERAGE = {
    "source_environment_metadata": (
        "schema", "schema_version", "generator_version", "upstream_commit",
        "environment", "opencv_version", "authenticated_sources",
        "authentication_order"),
    "evidence_contract_metadata": (
        "canny_contract", "cpbd_contract", "sort_contract",
        "scalar_compatibility_tolerances"),
    "evidence_inventory_metadata": (
        "cases", "sort_cases", "sort_public_order_filenames",
        "sort_best_blur_preselection_order_filenames"),
    "evidence_summary_metadata": (
        "score_summary", "privacy", "generator_self_tests"),
    "independently_recomputed_integrity_manifest": ("subgroup_sha256",),
}
SOURCE_SPECS = OrderedDict([
    ("estimate_sharpness", {
        "relative_path": "_internal/DeepFaceLab/core/imagelib/estimate_sharpness.py",
        "upstream_path": "core/imagelib/estimate_sharpness.py",
        "sha256": "08b3eea0a30a9f41df7c0bf63d8f9387483c159c2a53276fb908d51e160fbc60",
        "git_blob_sha1": "e4b3e2dce92cc55cf7bccea633f548db0557d40d",
    }),
    ("skimage_feature_canny", {
        "relative_path": "_internal/python-3.6.8/Lib/site-packages/skimage/feature/_canny.py",
        "upstream_path": "skimage/feature/_canny.py",
        "sha256": "3646016e5b56d60a94c3fec1d41625df16c5c6db71373859a88cdf94b5c74d15",
        "git_blob_sha1": "1d685f202806eb286f6e1ff8166a6d123c887b6b",
    }),
    ("skimage_filters_edges", {
        "relative_path": "_internal/python-3.6.8/Lib/site-packages/skimage/filters/edges.py",
        "upstream_path": "skimage/filters/edges.py",
        "sha256": "95adf833c66fe1fadf2efd6fabf0092d408881665e1db1c89c33d138bae5724a",
        "git_blob_sha1": "471edad657afb75cf9959fdf12b586dd572a313e",
    }),
])


class AuthenticationError(RuntimeError):
    pass


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value):
    # The Windows package stores these text files with CRLF.  Git's text
    # normalization hashes their authoritative upstream LF representation.
    normalized = value.replace(b"\r\n", b"\n")
    header = ("blob %d\0" % len(normalized)).encode("ascii")
    return hashlib.sha1(header + normalized).hexdigest()


def locate_reference(reference_root):
    root = Path(reference_root).resolve()
    runtime = root / "_internal" / "python-3.6.8" / "python.exe"
    site_packages = (root / "_internal" / "python-3.6.8" / "Lib" /
                     "site-packages")
    dfl_root = root / "_internal" / "DeepFaceLab"
    for path in (runtime, site_packages, dfl_root):
        if not path.exists():
            raise AuthenticationError("missing reference component: %s" % path.name)
    if Path(sys.executable).resolve() != runtime.resolve():
        raise AuthenticationError(
            "wrong interpreter: invoke the packaged python.exe from --reference-root")
    return root, runtime, site_packages, dfl_root


def authenticate_sources(root):
    raw_inputs = OrderedDict()
    for name, spec in SOURCE_SPECS.items():
        path = root / Path(spec["relative_path"])
        if not path.is_file():
            raise AuthenticationError("missing authenticated source: %s" % name)
        raw_inputs[name] = path.read_bytes()
    identities = OrderedDict(
        (name, calculate_source_identity(raw))
        for name, raw in raw_inputs.items())
    authenticated = OrderedDict()
    for name, spec in SOURCE_SPECS.items():
        actual_sha256, actual_blob = verify_source_identity(
            name, identities[name])
        authenticated[name] = {
            "sha256": actual_sha256,
            "git_blob_sha1": actual_blob,
            "upstream_path": spec["upstream_path"],
        }
    return authenticated, raw_inputs


def calculate_source_identity(raw):
    return sha256_bytes(raw), git_blob_sha1(raw)


def verify_source_identity(name, identity):
    spec = SOURCE_SPECS[name]
    actual_sha256, actual_blob = identity
    if actual_sha256 != spec["sha256"]:
        raise AuthenticationError(
            "%s SHA-256 mismatch: expected %s, got %s" %
            (name, spec["sha256"], actual_sha256))
    if actual_blob != spec["git_blob_sha1"]:
        raise AuthenticationError(
            "%s Git blob mismatch: expected %s, got %s" %
            (name, spec["git_blob_sha1"], actual_blob))
    return actual_sha256, actual_blob


def verify_source_bytes(name, raw):
    return verify_source_identity(name, calculate_source_identity(raw))


def distribution_version_from_metadata(site_packages, prefixes):
    candidates = []
    for child in site_packages.iterdir():
        lower = child.name.lower()
        if child.is_dir() and lower.endswith((".dist-info", ".egg-info")):
            normalized = lower.replace("_", "-")
            if any(normalized.startswith(prefix) for prefix in prefixes):
                candidates.append(child)
    if len(candidates) != 1:
        raise AuthenticationError(
            "expected one installed metadata directory for %s, found %d" %
            (prefixes[0], len(candidates)))
    metadata = candidates[0] / "METADATA"
    if not metadata.is_file():
        metadata = candidates[0] / "PKG-INFO"
    raw = metadata.read_bytes().decode("utf-8", "strict")
    versions = [line.split(":", 1)[1].strip() for line in raw.splitlines()
                if line.startswith("Version:")]
    if len(versions) != 1:
        raise AuthenticationError("invalid package metadata: %s" % candidates[0].name)
    return versions[0]


def verify_environment_metadata(site_packages):
    actual = OrderedDict()
    actual["python"] = platform.python_version()
    actual["numpy"] = distribution_version_from_metadata(
        site_packages, ("numpy-",))
    actual["scipy"] = distribution_version_from_metadata(
        site_packages, ("scipy-",))
    actual["scikit-image"] = distribution_version_from_metadata(
        site_packages, ("scikit-image-", "scikit_image-"))
    require_environment_versions(actual, include_opencv=False)
    return actual


def require_environment_versions(actual, include_opencv=True):
    for name, expected in EXPECTED_ENVIRONMENT.items():
        if name == "opencv" and not include_opencv:
            continue
        if actual[name] != expected:
            raise AuthenticationError(
                "%s version mismatch: expected %s, got %s" %
                (name, expected, actual[name]))


class AuthenticatedSourceLoader(importlib.abc.Loader):
    """Compile reviewed bytes directly; never consult filesystem bytecode."""

    def __init__(self, fullname, source, origin):
        self.fullname = fullname
        self.source = source
        self.origin = str(origin)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.origin
        module.__loader__ = self
        module.__authenticated_source_sha256__ = sha256_bytes(self.source)
        code = compile(self.source, self.origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class AuthenticatedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        item = self.sources.get(fullname)
        if item is None:
            return None
        source, origin = item
        loader = AuthenticatedSourceLoader(fullname, source, origin)
        return importlib.util.spec_from_loader(fullname, loader, origin=str(origin))


def import_authenticated_runtime(root, environment, raw_inputs):
    import numpy as np
    import scipy
    import cv2

    runtime_versions = {"numpy": np.__version__, "scipy": scipy.__version__,
                        "opencv": cv2.__version__}
    for name, value in runtime_versions.items():
        if value != EXPECTED_ENVIRONMENT[name]:
            raise AuthenticationError(
                "%s version mismatch: expected %s, got %s" %
                (name, EXPECTED_ENVIRONMENT[name], value))
    environment["opencv"] = cv2.__version__
    require_environment_versions(environment)

    scorer_path = root / Path(SOURCE_SPECS["estimate_sharpness"]["relative_path"])
    canny_path = root / Path(SOURCE_SPECS["skimage_feature_canny"]["relative_path"])
    edges_path = root / Path(SOURCE_SPECS["skimage_filters_edges"]["relative_path"])
    module_names = (
        "authenticated_historical_estimate_sharpness",
        "skimage.feature._canny", "skimage.filters.edges")
    for name in module_names:
        sys.modules.pop(name, None)
    sources = {
        module_names[0]: (raw_inputs["estimate_sharpness"], scorer_path),
        module_names[1]: (raw_inputs["skimage_feature_canny"], canny_path),
        module_names[2]: (raw_inputs["skimage_filters_edges"], edges_path),
    }
    finder = AuthenticatedSourceFinder(sources)
    sys.meta_path.insert(0, finder)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import skimage
        if skimage.__version__ != environment["scikit-image"]:
            raise AuthenticationError(
                "scikit-image imported version differs from authenticated metadata")
        import skimage.feature
        import skimage.feature._canny as authenticated_canny
        import skimage.filters.edges as authenticated_edges
        spec = importlib.util.find_spec(module_names[0])
        scorer = importlib.util.module_from_spec(spec)
        sys.modules[module_names[0]] = scorer
        spec.loader.exec_module(scorer)
        loaded = (
            ("estimate_sharpness", scorer),
            ("skimage_feature_canny", authenticated_canny),
            ("skimage_filters_edges", authenticated_edges),
        )
        for source_name, module in loaded:
            expected = SOURCE_SPECS[source_name]["sha256"]
            if not isinstance(module.__loader__, AuthenticatedSourceLoader):
                raise AuthenticationError(
                    "%s was not loaded by authenticated source loader" % source_name)
            if module.__authenticated_source_sha256__ != expected:
                raise AuthenticationError(
                    "%s executed-source digest mismatch" % source_name)
        if skimage.feature.canny is not authenticated_canny.canny:
            raise AuthenticationError("skimage.feature.canny bypassed authenticated source")
    finally:
        sys.dont_write_bytecode = old_dont_write
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
    return np, scipy, skimage, cv2, scorer


def array_digest(np, value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def named_arrays_digest(np, items):
    digest = hashlib.sha256()
    for name, value in items:
        encoded = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(array_digest(np, value).encode("ascii"))
    return digest.hexdigest()


def canonical_json_array(np, value):
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def subgroup_items(np, payload, metadata):
    case_input_metadata = []
    for record in metadata["cases"]:
        case_input_metadata.append(OrderedDict(
            (key, record[key]) for key in (
                "index", "name", "source_representation", "input_shape",
                "input_dtype", "gray_shape", "gray_dtype", "input_sha256")))
    public_sort_metadata = []
    internal_sort_metadata = []
    for record in metadata["sort_cases"]:
        public_sort_metadata.append(OrderedDict(
            (key, record[key]) for key in (
                "index", "filename", "shape", "gaussian_sigma",
                "public_blur_score", "preselection_candidate",
                "synthetic_yaw_bin")))
        internal_sort_metadata.append(OrderedDict(
            (key, record[key]) for key in (
                "index", "filename", "shape", "gaussian_sigma",
                "sort_best_faster_false_score", "preselection_candidate",
                "synthetic_yaw_bin")))

    groups = OrderedDict((name, []) for name in (
        "input_evidence", "canny_evidence", "cpbd_evidence",
        "final_score_evidence", "public_sort_evidence",
        "internal_final_blur_evidence", "source_environment_metadata",
        "evidence_contract_metadata", "evidence_inventory_metadata",
        "evidence_summary_metadata"))
    for record in metadata["cases"]:
        prefix = "case_%03d_" % record["index"]
        for field in ("input", "gray_float64"):
            groups["input_evidence"].append((prefix + field, payload[prefix + field]))
        groups["canny_evidence"].append(
            (prefix + "canny", payload[prefix + "canny"]))
        for field in (
                "sobel_response", "sobel_strength2_raw", "sobel_threshold",
                "sobel_strength2_thresholded", "sobel_thinned", "edge_widths",
                "pblur_map", "pblur_histogram", "qualified_edge_count"):
            groups["cpbd_evidence"].append(
                (prefix + field, payload[prefix + field]))
        groups["final_score_evidence"].append(
            (prefix + "score", payload[prefix + "score"]))
    groups["input_evidence"].append(
        ("case_metadata", canonical_json_array(np, case_input_metadata)))
    groups["final_score_evidence"].append(("scores", payload["scores"]))

    for record in metadata["sort_cases"]:
        prefix = "sort_%03d_" % record["index"]
        for field in ("decoded", "landmarks", "mask", "public_input",
                      "public_score"):
            groups["public_sort_evidence"].append(
                (prefix + field, payload[prefix + field]))
        for field in ("final_input", "final_score"):
            groups["internal_final_blur_evidence"].append(
                (prefix + field, payload[prefix + field]))
        groups["final_score_evidence"].append(
            (prefix + "public_score", payload[prefix + "public_score"]))
        groups["final_score_evidence"].append(
            (prefix + "final_score", payload[prefix + "final_score"]))
    groups["public_sort_evidence"].extend([
        ("sort_public_scores", payload["sort_public_scores"]),
        ("sort_public_order", payload["sort_public_order"]),
        ("public_sort_metadata", canonical_json_array(np, public_sort_metadata)),
        ("sort_public_order_filenames", canonical_json_array(
            np, metadata["sort_public_order_filenames"])),
        ("public_tie_contract", canonical_json_array(
            np, metadata["sort_contract"]["public_tie_order"])),
    ])
    groups["internal_final_blur_evidence"].extend([
        ("sort_best_blur_preselection_scores",
         payload["sort_best_blur_preselection_scores"]),
        ("sort_best_blur_preselection_order",
         payload["sort_best_blur_preselection_order"]),
        ("internal_sort_metadata", canonical_json_array(
            np, internal_sort_metadata)),
        ("sort_best_blur_preselection_order_filenames", canonical_json_array(
            np, metadata["sort_best_blur_preselection_order_filenames"])),
        ("internal_tie_contract", canonical_json_array(
            np, metadata["sort_contract"]["internal_tie_order"])),
    ])
    source_environment = OrderedDict([
        ("schema", metadata["schema"]),
        ("schema_version", metadata["schema_version"]),
        ("generator_version", metadata["generator_version"]),
        ("upstream_commit", metadata["upstream_commit"]),
        ("environment", metadata["environment"]),
        ("opencv_version", metadata["opencv_version"]),
        ("authenticated_sources", metadata["authenticated_sources"]),
        ("authentication_order", metadata["authentication_order"]),
    ])
    groups["source_environment_metadata"].append(
        ("source_environment_metadata",
         canonical_json_array(np, source_environment)))
    contract_metadata = OrderedDict(
        (key, metadata[key])
        for key in AUTHORITATIVE_METADATA_COVERAGE["evidence_contract_metadata"])
    inventory_metadata = OrderedDict(
        (key, metadata[key])
        for key in AUTHORITATIVE_METADATA_COVERAGE["evidence_inventory_metadata"])
    summary_metadata = OrderedDict(
        (key, metadata[key])
        for key in AUTHORITATIVE_METADATA_COVERAGE["evidence_summary_metadata"])
    groups["evidence_contract_metadata"].append(
        ("evidence_contract_metadata",
         canonical_json_array(np, contract_metadata)))
    groups["evidence_inventory_metadata"].append(
        ("evidence_inventory_metadata",
         canonical_json_array(np, inventory_metadata)))
    groups["evidence_summary_metadata"].append(
        ("evidence_summary_metadata",
         canonical_json_array(np, summary_metadata)))
    return groups


def compute_subgroup_hashes(np, payload, metadata):
    return OrderedDict(
        (name, named_arrays_digest(np, items))
        for name, items in subgroup_items(np, payload, metadata).items())


def save_deterministic_npz(np, output, payload):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(output), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payload.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(
                "%s.npy" % name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def checkerboard(np, height, width, cell):
    yy, xx = np.indices((height, width))
    return (((yy // cell + xx // cell) & 1) * 255).astype(np.uint8)


def edge_rich(np, cv2, height, width):
    image = np.zeros((height, width), np.uint8)
    for x in range(5, width, 13):
        image[:, x:min(x + 5, width)] = 220 if (x // 13) % 2 else 55
    for y in range(7, height, 19):
        cv2.line(image, (0, y), (width - 1, y), int((y * 17) % 256), 2)
    cv2.rectangle(image, (width // 8, height // 8),
                  (width * 7 // 8, height * 7 // 8), 255, 3)
    cv2.circle(image, (width // 2, height // 2),
               max(5, min(height, width) // 4), 20, 4)
    return image


def face_like(np, cv2, height, width):
    image = np.full((height, width), 18, np.uint8)
    center = (width // 2, height // 2)
    axes = (max(8, width * 3 // 8), max(8, height * 7 // 16))
    cv2.ellipse(image, center, axes, 0, 0, 360, 175, -1)
    cv2.ellipse(image, center, axes, 0, 0, 360, 245, 3)
    eye_y = height * 43 // 100
    for eye_x in (width * 35 // 100, width * 65 // 100):
        cv2.ellipse(image, (eye_x, eye_y),
                    (max(3, width // 13), max(2, height // 35)),
                    0, 0, 360, 25, -1)
        cv2.circle(image, (eye_x, eye_y), max(1, width // 55), 250, -1)
    cv2.line(image, (width // 2, height * 47 // 100),
             (width * 47 // 100, height * 62 // 100), 70, 3)
    cv2.ellipse(image, (width // 2, height * 72 // 100),
                (width // 7, max(2, height // 30)), 0, 0, 180, 30, 3)
    for offset in range(-2, 3):
        cv2.line(image, (width // 5, height // 3 + offset * 11),
                 (width * 4 // 5, height // 3 + offset * 11),
                 205 if offset & 1 else 110, 1)
    return image


def synthetic_cases(np, cv2):
    cases = OrderedDict()
    cases["constant_black_even_rank2"] = np.zeros((128, 128), np.uint8)
    cases["constant_white_odd_rank2"] = np.full((129, 131), 255, np.uint8)
    impulse = np.zeros((129, 129), np.uint8)
    impulse[64, 64] = 255
    cases["single_pixel_impulse"] = impulse
    horizontal = np.zeros((128, 130), np.uint8)
    horizontal[64:] = 255
    cases["horizontal_hard_edge_even"] = horizontal
    vertical = np.zeros((130, 128), np.uint8)
    vertical[:, 64:] = 255
    cases["vertical_hard_edge_even"] = vertical
    diagonal = np.zeros((129, 131), np.uint8)
    yy, xx = np.indices(diagonal.shape)
    diagonal[xx > yy] = 255
    cases["diagonal_hard_edge_odd"] = diagonal
    for cell in (2, 8, 16):
        cases["checkerboard_cell_%02d" % cell] = checkerboard(
            np, 128, 128, cell)
    bars = np.zeros((192, 192), np.uint8)
    x = 0
    for width in (1, 2, 3, 5, 8, 13, 21):
        for value in (235, 35):
            bars[:, x:min(192, x + width)] = value
            x += width
    cases["repeated_bars_controlled_widths"] = bars
    gradient = np.tile(np.linspace(0, 255, 130).astype(np.uint8), (129, 1))
    cases["smooth_gradient_odd"] = gradient
    base = edge_rich(np, cv2, 192, 192)
    cases["edge_rich_base_rank2"] = base
    for label, sigma in (("low", 0.8), ("medium", 1.6), ("high", 3.2)):
        cases["edge_rich_blur_%s" % label] = cv2.GaussianBlur(
            base, (0, 0), sigmaX=sigma, sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101)
    face = face_like(np, cv2, 193, 191)
    cases["face_like_base_odd"] = face
    cases["face_like_blur_low_odd"] = cv2.GaussianBlur(face, (0, 0), 1.0)
    cases["face_like_blur_high_odd"] = cv2.GaussianBlur(face, (0, 0), 2.8)
    cases["single_channel_rank3"] = base[..., None]
    bgr = np.stack((base, np.roll(base, 3, axis=1),
                    np.roll(base, 5, axis=0)), axis=2)
    cases["bgr_rank3"] = bgr
    alpha = np.full(base.shape, 173, np.uint8)
    cases["bgra_rank3"] = np.dstack((bgr, alpha))
    return cases


def historical_gray(np, cv2, image):
    if image.ndim == 3:
        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image[..., 0]
    return image.astype(np.float64)


def cpbd_intermediates(np, scipy, skimage, scorer, gray):
    from skimage.filters.edges import HSOBEL_WEIGHTS
    h1 = np.array(HSOBEL_WEIGHTS)
    h1 /= np.sum(abs(h1))
    response = scipy.ndimage.convolve(gray, h1.T)
    strength_raw = np.square(response)
    threshold = np.float64(2 * np.sqrt(np.mean(strength_raw)))
    strength_thresholded = strength_raw.copy()
    strength_thresholded[strength_thresholded <= threshold] = 0
    thinned = scorer._simple_thinning(strength_thresholded)
    canny = skimage.feature.canny(gray)
    widths = scorer.marziliano_method(thinned, gray)
    pblur_map = np.zeros(gray.shape, np.float64)
    hist = np.zeros(101, np.float64)
    total = 0
    block_h, block_w = scorer.BLOCK_HEIGHT, scorer.BLOCK_WIDTH
    for i in range(int(gray.shape[0] / block_h)):
        for j in range(int(gray.shape[1] / block_w)):
            rows = slice(block_h * i, block_h * (i + 1))
            cols = slice(block_w * j, block_w * (j + 1))
            if scorer.is_edge_block(canny[rows, cols], scorer.THRESHOLD):
                block_widths = widths[rows, cols]
                nz = block_widths != 0
                contrast = scorer.get_block_contrast(gray[rows, cols])
                jnb = scorer.WIDTH_JNB[contrast]
                probabilities = 1 - np.exp(
                    -abs(block_widths[nz] / jnb) ** scorer.BETA)
                target = pblur_map[rows, cols]
                target[nz] = probabilities
                for probability in probabilities:
                    hist[int(round(probability * 100))] += 1
                    total += 1
    if total:
        hist /= total
    derived_score = np.sum(hist[:64])
    official_score = np.float64(scorer._calculate_sharpness_metric(
        gray, canny, widths))
    if official_score.tobytes() != np.float64(derived_score).tobytes():
        raise RuntimeError("CPBD audit reconstruction differs from official score")
    return OrderedDict([
        ("canny", canny),
        ("sobel_response", response),
        ("sobel_strength2_raw", strength_raw),
        ("sobel_threshold", threshold),
        ("sobel_strength2_thresholded", strength_thresholded),
        ("sobel_thinned", thinned),
        ("edge_widths", widths),
        ("pblur_map", pblur_map),
        ("pblur_histogram", hist),
        ("qualified_edge_count", np.int64(total)),
        ("score", official_score),
    ])


def normalized_landmarks(np, height, width):
    points = []
    for index in range(17):
        angle = np.pi * (index / 16.0)
        points.append((width * (0.5 - 0.38 * np.cos(angle)),
                       height * (0.48 + 0.40 * np.sin(angle))))
    points.extend([
        (width*x, height*y) for x, y in (
            (.24,.31),(.30,.27),(.37,.27),(.43,.31),(.46,.34),
            (.54,.34),(.57,.31),(.63,.27),(.70,.27),(.76,.31),
            (.50,.34),(.50,.42),(.50,.50),(.50,.57),
            (.42,.58),(.46,.61),(.50,.62),(.54,.61),(.58,.58),
            (.29,.39),(.34,.36),(.40,.37),(.43,.41),(.39,.43),(.33,.43),
            (.57,.41),(.60,.37),(.66,.36),(.71,.39),(.67,.43),(.61,.43),
            (.36,.69),(.42,.66),(.47,.65),(.50,.66),(.53,.65),(.58,.66),
            (.64,.69),(.58,.75),(.53,.78),(.50,.79),(.47,.78),(.42,.75),
            (.39,.70),(.47,.69),(.50,.70),(.53,.69),(.61,.70),(.53,.73),
            (.50,.74),(.47,.73))])
    result = np.asarray(points, np.float32)
    if result.shape != (68, 2):
        raise RuntimeError("synthetic landmarks must contain 68 points")
    return result


def historical_hull_mask(np, cv2, shape, landmarks):
    lmrks = np.array(landmarks.copy(), dtype=int)
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))
    lmrks[17:22] = lmrks[17:22] + 0.5 * (lmrks[17:22] - bot_l)
    lmrks[22:27] = lmrks[22:27] + 0.5 * (lmrks[22:27] - bot_r)
    mask = np.zeros(shape[:2] + (1,), np.float32)
    parts = [
        (lmrks[0:9], lmrks[17:18]), (lmrks[8:17], lmrks[26:27]),
        (lmrks[17:20], lmrks[8:9]), (lmrks[24:27], lmrks[8:9]),
        (lmrks[19:25], lmrks[8:9]),
        (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9]),
        (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9]),
        (lmrks[27:31], lmrks[31:36]),
    ]
    for part in parts:
        merged = np.concatenate(part)
        cv2.fillConvexPoly(mask, cv2.convexHull(merged), (1,))
    return mask


def insert_dfl_metadata(jpeg, metadata):
    if jpeg[:2] != b"\xff\xd8":
        raise RuntimeError("OpenCV did not produce JPEG data")
    payload = pickle.dumps(metadata, protocol=4)
    if len(payload) + 2 > 65535:
        raise RuntimeError("synthetic DFL metadata is too large")
    marker = b"\xff\xef" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:2] + marker + jpeg[2:]


def sorter_evidence(np, cv2, scorer):
    records = []
    definitions = [
        ("sort_even_sharp.jpg", 192, 192, 0.0),
        ("sort_odd_blur_low.jpg", 193, 191, 0.9),
        ("sort_even_blur_medium.jpg", 192, 192, 1.8),
        ("sort_odd_blur_high.jpg", 193, 191, 3.4),
        ("sort_tie_a.jpg", 192, 192, 1.8),
        ("sort_tie_b.jpg", 192, 192, 1.8),
    ]
    with tempfile.TemporaryDirectory(prefix="dfl-sharpness-evidence-") as temporary:
        for filename, height, width, sigma in definitions:
            gray = edge_rich(np, cv2, height, width)
            if sigma:
                gray = cv2.GaussianBlur(gray, (0, 0), sigma)
            bgr = np.dstack((gray, np.roll(gray, 2, 1), np.roll(gray, 3, 0)))
            ok, encoded = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError("unable to encode synthetic JPEG")
            landmarks = normalized_landmarks(np, height, width)
            metadata = {
                "face_type": "full_face", "landmarks": landmarks,
                "source_filename": filename,
                "source_rect": [0, 0, width - 1, height - 1],
            }
            dfl_bytes = insert_dfl_metadata(encoded.tobytes(), metadata)
            path = Path(temporary) / filename
            path.write_bytes(dfl_bytes)
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2] != (height, width):
                raise RuntimeError("synthetic DFL-compatible JPEG did not decode")
            if b"\xff\xef" not in dfl_bytes[:4096]:
                raise RuntimeError("synthetic DFL-compatible JPEG lacks APP15")
            mask = historical_hull_mask(np, cv2, decoded.shape, landmarks)
            public_input = (decoded * mask).astype(np.uint8)
            decoded_gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            final_input = (decoded_gray[..., None] * mask).astype(np.uint8)
            records.append({
                "filename": filename, "height": height, "width": width,
                "sigma": sigma, "decoded": decoded,
                "landmarks": landmarks, "mask": mask,
                "public_input": public_input,
                "public_score": np.float64(scorer.estimate_sharpness(public_input)),
                "final_input": final_input,
                "final_score": np.float64(scorer.estimate_sharpness(final_input)),
            })
    public_order = sorted(range(len(records)),
                          key=lambda index: records[index]["public_score"],
                          reverse=True)
    final_order = sorted(range(len(records)),
                         key=lambda index: records[index]["final_score"],
                         reverse=True)
    return records, public_order, final_order


def expected_authenticated_sources():
    return OrderedDict(
        (name, {"sha256": spec["sha256"],
                "git_blob_sha1": spec["git_blob_sha1"],
                "upstream_path": spec["upstream_path"]})
        for name, spec in SOURCE_SPECS.items())


def expected_generator_self_tests():
    return OrderedDict((key, True) for key in EXPECTED_SELF_TEST_KEYS)


def recompute_score_summary(np, payload, metadata):
    scores = np.asarray(payload["scores"])
    positive = scores[scores > 0]
    return {
        "case_count": len(metadata["cases"]),
        "positive_case_count": int(len(positive)),
        "distinct_positive_score_count": int(len(np.unique(positive))),
        "positive_min": float(np.min(positive)),
        "positive_max": float(np.max(positive)),
        "overall_min": float(np.min(scores)),
        "overall_max": float(np.max(scores)),
    }


def validate_authoritative_metadata_structure(metadata):
    if set(metadata) != set(AUTHORITATIVE_METADATA_KEYS):
        missing = sorted(set(AUTHORITATIVE_METADATA_KEYS) - set(metadata))
        unexpected = sorted(set(metadata) - set(AUTHORITATIVE_METADATA_KEYS))
        raise RuntimeError(
            "authoritative metadata keys differ: missing=%r unexpected=%r" %
            (missing, unexpected))
    covered = []
    for keys in AUTHORITATIVE_METADATA_COVERAGE.values():
        covered.extend(keys)
    if len(covered) != len(set(covered)):
        raise RuntimeError("trusted authoritative metadata coverage overlaps")
    if set(covered) != set(AUTHORITATIVE_METADATA_KEYS):
        raise RuntimeError("trusted authoritative metadata coverage is incomplete")


def iter_string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            for result in iter_string_values(child):
                yield result
    elif isinstance(value, (list, tuple)):
        for child in value:
            for result in iter_string_values(child):
                yield result


def validate_privacy(np, payload, metadata):
    if metadata["privacy"] != EXPECTED_PRIVACY:
        raise RuntimeError("privacy metadata binding failed")
    if tuple(record["name"] for record in metadata["cases"]) != EXPECTED_CASE_NAMES:
        raise RuntimeError("synthetic case inventory differs from trusted inventory")
    if tuple(record["filename"] for record in metadata["sort_cases"]) != EXPECTED_SORT_FILENAMES:
        raise RuntimeError("synthetic sorter inventory differs from trusted inventory")
    strings = list(iter_string_values(metadata))
    for key, value in payload.items():
        if key == "metadata":
            continue
        array = np.asarray(value)
        if array.dtype.kind in ("U", "S"):
            strings.extend(str(item) for item in array.reshape(-1).tolist())
    username = os.environ.get("USERNAME", "")
    forbidden_fragments = (
        "AI_RUN", "file://", "http://", "https://", "api_key", "access_token",
        "password", "credential", "private_url")
    for value in strings:
        lowered = value.lower()
        if (re.search(r"[a-zA-Z]:[\\/]", value) or value.startswith(("/", "\\\\")) or
                (username and username.lower() in lowered) or
                any(fragment.lower() in lowered for fragment in forbidden_fragments)):
            raise RuntimeError("private path, credential, or URL leaked into evidence")


def validate_order(np, payload, metadata, order_key, scores_key,
                   filenames_key, score_field):
    count = len(metadata["sort_cases"])
    order = payload[order_key]
    scores = payload[scores_key]
    if order.dtype != np.int64 or order.shape != (count,):
        raise RuntimeError("%s must be an int64 candidate vector" % order_key)
    if scores.dtype != np.float64 or scores.shape != (count,):
        raise RuntimeError("%s must be a float64 candidate vector" % scores_key)
    if sorted(order.tolist()) != list(range(count)):
        raise RuntimeError("%s is not a complete permutation" % order_key)
    expected_scores = np.asarray(
        [record[score_field] for record in metadata["sort_cases"]], np.float64)
    if scores.tobytes() != expected_scores.tobytes():
        raise RuntimeError("%s differs from candidate metadata" % scores_key)
    expected_filenames = [metadata["sort_cases"][i]["filename"]
                          for i in order.tolist()]
    if metadata[filenames_key] != expected_filenames:
        raise RuntimeError("%s differs from order mapping" % filenames_key)
    ordered_scores = scores[order]
    if np.any(ordered_scores[:-1] < ordered_scores[1:]):
        raise RuntimeError("%s is not descending by score" % order_key)


def validate_payload(np, payload, metadata):
    if "metadata" in payload:
        embedded_metadata = json.loads(str(np.asarray(payload["metadata"])))
        if embedded_metadata != metadata:
            raise RuntimeError("embedded metadata differs from validated metadata")
    validate_authoritative_metadata_structure(metadata)
    if metadata["schema"] != SCHEMA or metadata["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("schema self-validation failed")
    if metadata["generator_version"] != GENERATOR_VERSION:
        raise RuntimeError("generator version binding failed")
    if metadata["upstream_commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("upstream commit binding failed")
    if metadata["environment"] != dict(EXPECTED_ENVIRONMENT):
        raise RuntimeError("environment metadata binding failed")
    if metadata["opencv_version"] != EXPECTED_ENVIRONMENT["opencv"]:
        raise RuntimeError("OpenCV metadata binding failed")
    if metadata["authenticated_sources"] != expected_authenticated_sources():
        raise RuntimeError("authenticated source metadata binding failed")
    if metadata["canny_contract"] != EXPECTED_CANNY_CONTRACT:
        raise RuntimeError("Canny contract binding failed")
    if metadata["cpbd_contract"] != EXPECTED_CPBD_CONTRACT:
        raise RuntimeError("CPBD contract binding failed")
    if metadata["scalar_compatibility_tolerances"] != "UNFINALIZED":
        raise RuntimeError("scalar compatibility tolerance status changed")
    if metadata["generator_self_tests"] != expected_generator_self_tests():
        raise RuntimeError("generator self-test record binding failed")
    if metadata["authentication_order"] != [
            "locate_reference_runtime", "read_source_files_as_raw_bytes",
            "sha256_authenticated_inputs", "compare_reviewed_digests",
            "verify_exact_environment_versions_including_opencv",
            "construct_authenticated_source_execution",
            "execute_authenticated_scorer"]:
        raise RuntimeError("authentication order binding failed")
    if not metadata["cases"]:
        raise RuntimeError("case inventory is empty")
    case_indices = [record["index"] for record in metadata["cases"]]
    if case_indices != list(range(len(metadata["cases"]))):
        raise RuntimeError("case indices are not contiguous and ordered")
    case_names = [record["name"] for record in metadata["cases"]]
    if len(set(case_names)) != len(case_names):
        raise RuntimeError("case names are not unique")
    case_scores = []
    for record in metadata["cases"]:
        prefix = "case_%03d_" % record["index"]
        image = payload[prefix + "input"]
        gray = payload[prefix + "gray_float64"]
        canny = payload[prefix + "canny"]
        score = payload[prefix + "score"]
        if list(image.shape) != record["input_shape"] or image.dtype.name != record["input_dtype"]:
            raise RuntimeError("input representation metadata binding failed")
        if list(gray.shape) != record["gray_shape"] or gray.dtype.name != record["gray_dtype"]:
            raise RuntimeError("grayscale representation metadata binding failed")
        if canny.dtype != np.bool_ or canny.shape != gray.shape:
            raise RuntimeError("Canny evidence must be Boolean and geometry-matched")
        if score.dtype != np.float64 or score.shape != () or not np.isfinite(score):
            raise RuntimeError("score evidence must be finite scalar float64")
        if score.tobytes() != np.float64(record["score"]).tobytes():
            raise RuntimeError("case score metadata binding failed")
        if array_digest(np, image) != record["input_sha256"]:
            raise RuntimeError("synthetic case metadata binding failed")
        case_scores.append(score)
    expected_case_scores = np.asarray(case_scores, np.float64)
    if (payload["scores"].dtype != np.float64 or
            payload["scores"].shape != expected_case_scores.shape or
            payload["scores"].tobytes() != expected_case_scores.tobytes()):
        raise RuntimeError("aggregate case score vector binding failed")
    if metadata["score_summary"] != recompute_score_summary(np, payload, metadata):
        raise RuntimeError("score summary recomputation failed")

    sort_indices = [record["index"] for record in metadata["sort_cases"]]
    sort_filenames = [record["filename"] for record in metadata["sort_cases"]]
    if sort_indices != list(range(len(sort_indices))) or len(set(sort_filenames)) != len(sort_filenames):
        raise RuntimeError("sort candidate mapping is invalid")
    for record in metadata["sort_cases"]:
        index = record["index"]
        prefix = "sort_%03d_" % index
        if record["preselection_candidate"] is not True:
            raise RuntimeError("sort candidate metadata is invalid")
        if not isinstance(record["synthetic_yaw_bin"], int):
            raise RuntimeError("synthetic yaw-bin metadata is invalid")
        for field, metadata_field in (
                ("public_score", "public_blur_score"),
                ("final_score", "sort_best_faster_false_score")):
            value = payload[prefix + field]
            if value.dtype != np.float64 or value.shape != () or not np.isfinite(value):
                raise RuntimeError("sort score must be finite scalar float64")
            if value.tobytes() != np.float64(record[metadata_field]).tobytes():
                raise RuntimeError("sort score metadata binding failed")
        if list(payload[prefix + "decoded"].shape) != record["shape"]:
            raise RuntimeError("sort candidate shape metadata binding failed")

    validate_order(
        np, payload, metadata, "sort_public_order", "sort_public_scores",
        "sort_public_order_filenames", "public_blur_score")
    validate_order(
        np, payload, metadata, "sort_best_blur_preselection_order",
        "sort_best_blur_preselection_scores",
        "sort_best_blur_preselection_order_filenames",
        "sort_best_faster_false_score")
    for value in payload.values():
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise RuntimeError("object arrays are forbidden")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise RuntimeError("numeric evidence must be finite")
    validate_privacy(np, payload, metadata)
    recomputed = compute_subgroup_hashes(np, payload, metadata)
    if metadata["subgroup_sha256"] != recomputed:
        raise RuntimeError("subgroup digest validation failed")


def prove_pyc_bypass_blocked():
    authenticated = b"VALUE = 'authenticated'\n"
    substituted = b"VALUE = 'substituted'\n"
    with tempfile.TemporaryDirectory(prefix="dfl-pyc-bypass-test-") as temporary:
        source_path = Path(temporary) / "probe.py"
        source_path.write_bytes(authenticated)
        cache_path = Path(importlib.util.cache_from_source(str(source_path)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(source_path), cfile=str(cache_path), doraise=True)
        cached = cache_path.read_bytes()
        header_size = 12 if sys.version_info < (3, 7) else 16
        substituted_code = compile(substituted, str(source_path), "exec")
        cache_path.write_bytes(cached[:header_size] + marshal.dumps(substituted_code))

        normal_name = "_dfl_substituted_pyc_probe"
        normal_spec = importlib.util.spec_from_file_location(normal_name, str(source_path))
        normal_module = importlib.util.module_from_spec(normal_spec)
        normal_spec.loader.exec_module(normal_module)

        authenticated_name = "_dfl_authenticated_source_probe"
        loader = AuthenticatedSourceLoader(
            authenticated_name, authenticated, source_path)
        authenticated_spec = importlib.util.spec_from_loader(
            authenticated_name, loader, origin=str(source_path))
        authenticated_module = importlib.util.module_from_spec(authenticated_spec)
        authenticated_spec.loader.exec_module(authenticated_module)
        return (normal_module.VALUE == "substituted" and
                authenticated_module.VALUE == "authenticated")


def self_test_fail_closed(root, site_packages, environment, np, payload, metadata):
    tests = OrderedDict()
    for name in EXPECTED_ENVIRONMENT:
        wrong = OrderedDict(environment)
        wrong[name] = "0.0.0"
        try:
            require_environment_versions(wrong)
        except AuthenticationError:
            tests["wrong_%s_version_rejected" % name.replace("-", "_")] = True
        else:
            tests["wrong_%s_version_rejected" % name.replace("-", "_")] = False
    source = root / Path(SOURCE_SPECS["estimate_sharpness"]["relative_path"])
    try:
        verify_source_bytes("estimate_sharpness", source.read_bytes() + b"altered")
    except AuthenticationError:
        tests["wrong_source_digest_rejected"] = True
    else:
        tests["wrong_source_digest_rejected"] = False
    tests["substituted_pyc_bypass_rejected"] = prove_pyc_bypass_blocked()

    def rejected(name, altered_payload=None, altered_metadata=None):
        try:
            validate_payload(
                np, payload if altered_payload is None else altered_payload,
                metadata if altered_metadata is None else altered_metadata)
        except (RuntimeError, KeyError, TypeError, ValueError):
            tests[name] = True
        else:
            tests[name] = False

    altered = OrderedDict(payload)
    altered["case_000_canny"] = payload["case_000_canny"].copy()
    altered["case_000_canny"][1, 1] = ~altered["case_000_canny"][1, 1]
    rejected("altered_canny_subgroup_rejected", altered_payload=altered)
    altered = OrderedDict(payload)
    altered["case_000_sobel_threshold"] = np.asarray(
        float(payload["case_000_sobel_threshold"]) + 1.0, np.float64)
    rejected("altered_cpbd_subgroup_rejected", altered_payload=altered)
    altered = OrderedDict(payload)
    altered["case_011_score"] = np.asarray(
        float(payload["case_011_score"]) - 0.01, np.float64)
    rejected("altered_final_score_subgroup_rejected", altered_payload=altered)
    altered = OrderedDict(payload)
    altered["sort_000_public_score"] = np.asarray(-1.0, np.float64)
    rejected("altered_public_sort_score_rejected", altered_payload=altered)
    altered = OrderedDict(payload)
    altered["sort_public_order"] = np.zeros(
        payload["sort_public_order"].shape, np.int64)
    rejected("altered_public_sort_order_rejected", altered_payload=altered)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["sort_cases"][0]["filename"] = "altered.jpg"
    rejected("altered_public_sort_mapping_rejected", altered_metadata=altered_meta)
    altered = OrderedDict(payload)
    altered["sort_000_final_score"] = np.asarray(-1.0, np.float64)
    rejected("altered_internal_final_score_rejected", altered_payload=altered)
    altered = OrderedDict(payload)
    altered["sort_best_blur_preselection_order"] = np.zeros(
        payload["sort_best_blur_preselection_order"].shape, np.int64)
    rejected("altered_internal_final_order_rejected", altered_payload=altered)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["sort_cases"][0]["synthetic_yaw_bin"] = 99
    rejected("altered_internal_mapping_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["authenticated_sources"]["estimate_sharpness"]["sha256"] = "0" * 64
    rejected("altered_source_metadata_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["environment"]["opencv"] = "0.0.0"
    rejected("altered_environment_metadata_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["schema"] = "altered"
    rejected("altered_schema_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["schema_version"] = 999
    rejected("altered_schema_version_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["generator_self_tests"]["wrong_opencv_version_rejected"] = False
    rejected("altered_generator_self_test_rejected", altered_metadata=altered_meta)
    for test_name, field, value in (
            ("altered_canny_sigma_rejected", "sigma", 99.0),
            ("altered_canny_threshold_rejected", "low_threshold", 0.25),
            ("altered_canny_boundary_rejected", "gaussian_boundary_mode", "reflect"),
            ("altered_canny_hysteresis_rejected", "hysteresis_connectivity", 4)):
        altered_meta = json.loads(json.dumps(metadata))
        altered_meta["canny_contract"][field] = value
        rejected(test_name, altered_metadata=altered_meta)
    for test_name, field, value in (
            ("altered_cpbd_block_size_rejected", "block_shape", [32, 32]),
            ("altered_cpbd_beta_rejected", "beta", 9.9),
            ("altered_cpbd_sobel_rejected", "historical_hsobel_weights",
             [[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
            ("altered_cpbd_threshold_rejected", "edge_block_threshold", 0.5)):
        altered_meta = json.loads(json.dumps(metadata))
        altered_meta["cpbd_contract"][field] = value
        rejected(test_name, altered_metadata=altered_meta)
    for test_name, field, value in (
            ("altered_score_positive_count_rejected", "positive_case_count", 999),
            ("altered_score_distinct_count_rejected",
             "distinct_positive_score_count", 999),
            ("altered_score_min_rejected", "positive_min", -1.0),
            ("altered_score_max_rejected", "overall_max", 999.0)):
        altered_meta = json.loads(json.dumps(metadata))
        altered_meta["score_summary"][field] = value
        rejected(test_name, altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["scalar_compatibility_tolerances"] = {"rtol": 1.0}
    rejected("altered_scalar_tolerance_rejected", altered_metadata=altered_meta)
    altered_meta = json.loads(json.dumps(metadata))
    altered_meta["privacy"]["synthetic_inputs_only"] = False
    rejected("altered_privacy_metadata_rejected", altered_metadata=altered_meta)
    info_a = zipfile.ZipInfo("x.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info_b = zipfile.ZipInfo("x.npy", date_time=(1980, 1, 1, 0, 0, 0))
    tests["fixed_zip_metadata"] = (
        info_a.date_time == info_b.date_time == (1980, 1, 1, 0, 0, 0))
    if not all(tests.values()):
        raise RuntimeError("generator fail-closed self-test failed: %r" % tests)
    return tests


def generate(args):
    # Required order: locate, raw-read/hash, digest comparison, version check,
    # and only then import or execute the historical scorer.
    root, runtime, site_packages, dfl_root = locate_reference(args.reference_root)
    authentication, raw_inputs = authenticate_sources(root)
    environment = verify_environment_metadata(site_packages)
    np, scipy, skimage, cv2, scorer = import_authenticated_runtime(
        root, environment, raw_inputs)

    cases = synthetic_cases(np, cv2)
    payload = OrderedDict()
    case_records = []
    input_items, canny_items, cpbd_items, score_items = [], [], [], []
    for index, (name, image) in enumerate(cases.items()):
        prefix = "case_%03d_" % index
        gray = historical_gray(np, cv2, image)
        evidence = cpbd_intermediates(np, scipy, skimage, scorer, gray)
        direct_score = np.float64(scorer.estimate_sharpness(image))
        if direct_score.tobytes() != evidence["score"].tobytes():
            raise RuntimeError("direct scorer differs from intermediate audit: %s" % name)
        payload[prefix + "input"] = image
        payload[prefix + "gray_float64"] = gray
        for field, value in evidence.items():
            payload[prefix + field] = value
        record = {
            "index": index, "name": name,
            "source_representation": (
                "rank2" if image.ndim == 2 else
                ("rank3_one_channel" if image.shape[2] == 1 else
                 ("BGRA" if image.shape[2] == 4 else "BGR"))),
            "input_shape": list(image.shape), "input_dtype": image.dtype.name,
            "gray_shape": list(gray.shape), "gray_dtype": gray.dtype.name,
            "score_dtype": evidence["score"].dtype.name,
            "score": float(evidence["score"]),
            "input_sha256": array_digest(np, image),
        }
        case_records.append(record)
        input_items.append((prefix + "input", image))
        canny_items.append((prefix + "canny", evidence["canny"]))
        score_items.append((prefix + "score", evidence["score"]))
        for field in ("sobel_response", "sobel_strength2_raw",
                      "sobel_strength2_thresholded", "sobel_thinned",
                      "edge_widths", "pblur_map", "pblur_histogram"):
            cpbd_items.append((prefix + field, evidence[field]))

    sort_records, public_order, final_order = sorter_evidence(np, cv2, scorer)
    sort_meta = []
    for index, record in enumerate(sort_records):
        prefix = "sort_%03d_" % index
        for field in ("decoded", "landmarks", "mask", "public_input",
                      "public_score", "final_input", "final_score"):
            payload[prefix + field] = record[field]
        sort_meta.append({
            "index": index, "filename": record["filename"],
            "shape": [record["height"], record["width"], 3],
            "gaussian_sigma": record["sigma"],
            "public_blur_score": float(record["public_score"]),
            "sort_best_faster_false_score": float(record["final_score"]),
            "preselection_candidate": True,
            "synthetic_yaw_bin": 0,
        })
    payload["sort_public_scores"] = np.asarray(
        [record["public_score"] for record in sort_records], np.float64)
    payload["sort_best_blur_preselection_scores"] = np.asarray(
        [record["final_score"] for record in sort_records], np.float64)
    payload["sort_public_order"] = np.asarray(public_order, np.int64)
    payload["sort_best_blur_preselection_order"] = np.asarray(final_order, np.int64)
    payload["scores"] = np.asarray(
        [record["score"] for record in case_records], np.float64)

    positive = payload["scores"][payload["scores"] > 0]
    if len(np.unique(positive)) < 3:
        raise RuntimeError("synthetic cases did not produce several distinct positive scores")
    metadata = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "environment": environment,
        "opencv_version": cv2.__version__,
        "authenticated_sources": authentication,
        "authentication_order": [
            "locate_reference_runtime", "read_source_files_as_raw_bytes",
            "sha256_authenticated_inputs", "compare_reviewed_digests",
            "verify_exact_environment_versions_including_opencv",
            "construct_authenticated_source_execution",
            "execute_authenticated_scorer"],
        "canny_contract": json.loads(json.dumps(EXPECTED_CANNY_CONTRACT)),
        "cpbd_contract": json.loads(json.dumps(EXPECTED_CPBD_CONTRACT)),
        "cases": case_records, "sort_cases": sort_meta,
        "sort_public_order_filenames": [sort_meta[i]["filename"] for i in public_order],
        "sort_best_blur_preselection_order_filenames": [
            sort_meta[i]["filename"] for i in final_order],
        "sort_contract": {
            "temporary_inputs": "synthetic DFL-compatible JPEG with APP15 metadata",
            "public_blur_path": "decoded BGR * landmark hull, then estimate_sharpness",
            "sort_best_faster_false_path": "decoded gray * landmark hull, then estimate_sharpness",
            "sort_direction": "descending",
            "non_tied_ordering": "HARD_PARITY_TARGET",
            "tie_behavior": "GENERATOR_OBSERVED_TIE_ORDER",
            "public_tie_order": {
                "classification": "OBSERVED_REFERENCE_ORDER",
                "statement": (
                    "Observed order for this authenticated generator execution; "
                    "not a general filename-order or stable multiprocessing tie contract."),
            },
            "internal_tie_order": {
                "classification": "OBSERVED_REFERENCE_ORDER",
                "statement": (
                    "Observed order for this authenticated generator execution; "
                    "not a general filename-order or stable multiprocessing tie contract."),
            },
            "jpeg_files_persisted": False,
        },
        "score_summary": {},
        "scalar_compatibility_tolerances": "UNFINALIZED",
        "privacy": dict(EXPECTED_PRIVACY),
    }
    metadata["score_summary"] = recompute_score_summary(np, payload, metadata)
    # Predeclare the exact expected record so metadata subgroup hashes can be
    # exercised by the mutation self-tests.  Generation still runs every test
    # below and refuses output unless the independently produced record agrees.
    metadata["generator_self_tests"] = expected_generator_self_tests()
    metadata["subgroup_sha256"] = compute_subgroup_hashes(np, payload, metadata)
    # Validate metadata bindings before embedding it.
    tests = self_test_fail_closed(
        root, site_packages, environment, np, payload, metadata)
    if tests != expected_generator_self_tests():
        raise RuntimeError("generator self-test record differs from trusted structure")
    metadata["generator_self_tests"] = tests
    payload["metadata"] = np.asarray(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")))
    validate_payload(np, payload, metadata)
    save_deterministic_npz(np, args.output, payload)
    print(json.dumps({
        "output": str(args.output), "schema": SCHEMA,
        "key_count": len(payload), "case_count": len(case_records),
        "positive_count": int(len(positive)),
        "min_score": float(np.min(payload["scores"])),
        "max_score": float(np.max(payload["scores"])),
        "sha256": sha256_bytes(Path(args.output).read_bytes()),
        "size": Path(args.output).stat().st_size,
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
