r"""Generate the authenticated TensorFlow FaceEnhancer reference fixture.

The TensorFlow path must run against an exact, clean checkout of official
DeepFaceLab commit 14cc9d4e5ffc062856a739c8e64707654780774e.  The fixture uses
only the tracked official checkpoint and deterministic synthetic images.

Examples (PowerShell)::

    .venv-tf\Scripts\python.exe tests\parity\generate_face_enhancer_tf_reference.py `
        --official-root .cache\phase14-faceenhancer-official-a `
        --checkpoint facelib\FaceEnhancer.npy `
        --output tests\parity\fixtures\face_enhancer_tf_reference.npz

    .venv\Scripts\python.exe tests\parity\generate_face_enhancer_tf_reference.py `
        --compare-torch --checkpoint facelib\FaceEnhancer.npy `
        --fixture tests\parity\fixtures\face_enhancer_tf_reference.npz `
        --comparison-output .cache\face_enhancer_torch_comparison.json
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


OFFICIAL_COMMIT = "14cc9d4e5ffc062856a739c8e64707654780774e"
SCHEMA = "face-enhancer-tf-reference-v1"
COMPARISON_SCHEMA = "face-enhancer-torch-comparison-v2"
COMPARISON_VERSION = 2
EXPECTED_CHECKPOINT_SHA256 = (
    "254958f67c9adfe97a0c9fc7b3c343ba490a1519c01862a50945fa228875476a")
SEED = 140014
PATCH_SIZE = 192
UPSCALE = 4
OVERLAP = 96
CONV_NAMES = (
    "conv1", "e0_conv0", "e0_conv1", "e1_conv0", "e1_conv1",
    "e2_conv0", "e2_conv1", "e3_conv0", "e3_conv1", "e4_conv0",
    "e4_conv1", "center_conv0", "center_conv1", "center_conv2",
    "center_conv3", "d4_conv0", "d4_conv1", "d3_conv0", "d3_conv1",
    "d2_conv0", "d2_conv1", "d1_conv0", "d1_conv1", "d0_conv0",
    "d0_conv1", "out1x_conv0", "out1x_conv1", "dec2x_conv0",
    "dec2x_conv1", "out2x_conv0", "out2x_conv1", "dec4x_conv0",
    "dec4x_conv1", "out4x_conv0", "out4x_conv1",
)
SOURCE_FILES = (
    "facelib/FaceEnhancer.py",
    "core/leras/ops/__init__.py",
    "core/leras/models/ModelBase.py",
    "core/leras/layers/Conv2D.py",
    "core/leras/layers/Dense.py",
)
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}


class OfficialSourceAuthenticationError(RuntimeError):
    """The supplied oracle is not the reviewed official Git tree."""


def _git(root, *args):
    command = ["git", "-C", str(root), *args]
    result = subprocess.run(
        command, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise OfficialSourceAuthenticationError(
            f"official source authentication failed: {' '.join(command)}: "
            f"{detail}")
    return result.stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": value.dtype.name, "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def json_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def authenticate_official_root(root, expected_commit=OFFICIAL_COMMIT):
    root = Path(root).resolve()
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if root != top:
        raise OfficialSourceAuthenticationError(
            f"--official-root must be the Git root: {root} != {top}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise OfficialSourceAuthenticationError(
            f"official source HEAD mismatch: expected {expected_commit}, "
            f"got {commit}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise OfficialSourceAuthenticationError(
            f"official source has tracked modifications:\n{dirty}")
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise OfficialSourceAuthenticationError(
            f"official source is incomplete: {missing}")
    return {
        "commit": commit,
        "tracked_state": "clean",
        "source_sha256": {
            name: sha256_file(root / name) for name in SOURCE_FILES
        },
        "root": root,
    }


def configure_determinism():
    if "tensorflow" in sys.modules:
        raise RuntimeError("TensorFlow imported before determinism controls")
    os.environ.update(ENVIRONMENT)
    random.seed(SEED)
    np.random.seed(SEED)


def save_deterministic_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for name, value in payload.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def load_checkpoint(path, expected_sha256=EXPECTED_CHECKPOINT_SHA256):
    path = Path(path).resolve()
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "FaceEnhancer checkpoint authentication failed before "
            f"deserialization: expected {expected_sha256}, "
            f"got {actual_sha256}")
    values = pickle.loads(raw)
    if not isinstance(values, dict):
        raise RuntimeError("FaceEnhancer checkpoint is not a dictionary")
    records = []
    conv_entries = dense_entries = 0
    for key, value in values.items():
        value = np.asarray(value)
        if value.dtype != np.float16:
            raise RuntimeError(f"unexpected checkpoint dtype for {key}")
        if key.startswith("dense"):
            dense_entries += 1
        else:
            conv_entries += 1
        records.append({
            "key": key, "dtype": value.dtype.name,
            "shape": list(value.shape),
        })
    if len(records) != 72 or conv_entries != 70 or dense_entries != 2:
        raise RuntimeError(
            "checkpoint inventory mismatch: expected 72 total / 70 conv / "
            f"2 dense, got {len(records)} / {conv_entries} / {dense_entries}")
    expected_keys = {
        f"{name}/{field}:0" for name in CONV_NAMES
        for field in ("weight", "bias")
    } | {"dense1/weight:0", "dense2/weight:0"}
    actual_keys = set(values)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "checkpoint key inventory mismatch; missing="
            f"{sorted(expected_keys - actual_keys)}, extra="
            f"{sorted(actual_keys - expected_keys)}")
    for name in CONV_NAMES:
        weight = np.asarray(values[f"{name}/weight:0"])
        bias = np.asarray(values[f"{name}/bias:0"])
        if weight.ndim != 4 or bias.shape != (1, 1, 1, weight.shape[3]):
            raise RuntimeError(
                f"convolution weight/bias shape mismatch for {name}: "
                f"{weight.shape} / {bias.shape}")
    for name in ("dense1", "dense2"):
        if np.asarray(values[f"{name}/weight:0"]).shape != (1, 64):
            raise RuntimeError(f"dense shape mismatch for {name}")
    weights = [r for r in records if r["key"].endswith("/weight:0")]
    biases = [r for r in records if r["key"].endswith("/bias:0")]
    if len(weights) != 37 or len(biases) != 35:
        raise RuntimeError("checkpoint weight/bias inventory mismatch")
    if any(len(r["shape"]) != 4 or r["shape"][:3] != [1, 1, 1]
           for r in biases):
        raise RuntimeError("legacy convolution bias shape mismatch")
    return values, {
        "file_size": path.stat().st_size,
        "sha256": actual_sha256,
        "key_count": len(records),
        "conv_entry_count": conv_entries,
        "dense_entry_count": dense_entries,
        "dtype_inventory": ["float16"],
        "entries": records,
        "future_mapping": {
            "conv_weight": "HWIO -> OIHW",
            "conv_bias": "(1,1,1,C) -> (C,)",
            "dense_weight": "official (input,output) ordering retained",
        },
    }


def resize_patterns(size):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    ramp = (x + y * np.float32(0.37)) / np.float32(max(1, size - 1))
    checker = ((x.astype(np.int32) + y.astype(np.int32)) % 2).astype(np.float32)
    impulse = np.zeros((size, size), np.float32)
    impulse[size // 2, (size - 1) // 2] = np.float32(1.0)
    random_like = np.mod(
        x * np.float32(0.173) + y * np.float32(0.319) +
        x * y * np.float32(0.071), np.float32(1.0))
    return OrderedDict(
        ramp=ramp, checkerboard=checker, impulse=impulse,
        random_like=random_like.astype(np.float32))


def synthetic_image(height, width, phase):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    channels = (
        np.mod(x * np.float32(0.013) + y * np.float32(0.007) + phase,
               np.float32(1.0)),
        np.mod(x * y * np.float32(0.00037) + y * np.float32(0.019) +
               phase * np.float32(0.7), np.float32(1.0)),
        np.mod(x * np.float32(0.031) - y * np.float32(0.011) +
               phase * np.float32(1.3), np.float32(1.0)),
    )
    return np.stack(channels, axis=-1).astype(np.float32)


def patch_coordinates(length):
    limit = length - PATCH_SIZE + 1
    result = []
    position = 0
    while position < limit:
        result.append(position)
        if position == limit - 1:
            break
        position = min(position + OVERLAP, limit - 1)
    return result


def patch_mask():
    line = np.concatenate([
        np.linspace(0, 1, PATCH_SIZE // 2 * UPSCALE),
        np.linspace(1, 0, PATCH_SIZE // 2 * UPSCALE),
    ])
    xx, yy = np.meshgrid(line, line)
    return (xx * yy)[..., None]


def _padding(height, width):
    top = bottom = left = right = 0
    if height < PATCH_SIZE:
        top = (PATCH_SIZE - height) // 2
        bottom = PATCH_SIZE - height - top
    if width < PATCH_SIZE:
        left = (PATCH_SIZE - width) // 2
        right = PATCH_SIZE - width - left
    return top, bottom, left, right


def run_pipeline(model_run, image, is_tanh, preserve_size, cv2):
    original = np.asarray(image, np.float32)
    processed = (original if is_tanh else
                 np.clip(original * 2 - 1, -1, 1).astype(np.float32))
    ih, iw, channels = processed.shape
    top, bottom, left, right = _padding(ih, iw)
    padded = np.pad(
        processed, ((top, bottom), (left, right), (0, 0)),
        mode="constant").astype(np.float32)
    height, width, _ = padded.shape
    ys = patch_coordinates(height)
    xs = patch_coordinates(width)
    mask = patch_mask()
    accumulated = np.zeros(
        (height * UPSCALE, width * UPSCALE, channels), np.float32)
    divisor = np.zeros(
        (height * UPSCALE, width * UPSCALE, 1), np.float32)
    patch_records = []
    for y in ys:
        for x in xs:
            patch = padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            output = model_run(patch)
            oy, ox = y * UPSCALE, x * UPSCALE
            accumulated[oy:oy + PATCH_SIZE * UPSCALE,
                        ox:ox + PATCH_SIZE * UPSCALE] += output * mask
            divisor[oy:oy + PATCH_SIZE * UPSCALE,
                    ox:ox + PATCH_SIZE * UPSCALE] += mask
            patch_records.append({
                "x": x, "y": y,
                "input_sha256": array_digest(patch),
                "output_sha256": array_digest(output),
            })
    divisor_safe = divisor.copy()
    divisor_safe[divisor_safe == 0] = 1.0
    stitched = accumulated / divisor_safe
    cropped = stitched[
        top * UPSCALE:(height - bottom) * UPSCALE,
        left * UPSCALE:(width - right) * UPSCALE]
    preserved = (cv2.resize(cropped, (iw, ih), interpolation=cv2.INTER_LANCZOS4)
                 if preserve_size else cropped)
    final = (preserved if is_tanh else
             np.clip(preserved / 2 + 0.5, 0, 1))
    return {
        "input": original,
        "preprocessed": processed,
        "padded": padded,
        "accumulated": accumulated,
        "divisor": divisor,
        "stitched": stitched,
        "cropped": cropped,
        "preserved": preserved,
        "final": final,
        "padding": [top, bottom, left, right],
        "x_coordinates": xs,
        "y_coordinates": ys,
        "patches": patch_records,
    }


def _selected_evidence(payload, prefix, result, full_intermediates):
    # Full contract arrays that are reasonably sized; large blending internals
    # are frozen by digest plus deterministic edge/center probes.
    for field in ("input", "preprocessed", "padded", "final"):
        payload[f"{prefix}_{field}"] = result[field]
    if result["preserved"].shape != result["cropped"].shape:
        payload[f"{prefix}_preserved"] = result["preserved"]
    compact_fields = ("accumulated", "divisor", "stitched", "cropped",
                      "preserved")
    if full_intermediates:
        for field in ("stitched", "cropped", "preserved"):
            payload[f"{prefix}_{field}"] = result[field]
    for field in compact_fields:
        value = result[field]
        payload[f"{prefix}_{field}_digest"] = np.asarray(array_digest(value))
        payload[f"{prefix}_{field}_probes"] = np.asarray([
            value.reshape(-1)[0],
            value.reshape(-1)[len(value.reshape(-1)) // 2],
            value.reshape(-1)[-1],
        ], np.float64)


EXPECTED_MODEL_RESIZE_STAGES = (
    ([None, 6, 6, 512], [None, 12, 12, 512]),
    ([None, 12, 12, 512], [None, 24, 24, 512]),
    ([None, 24, 24, 512], [None, 48, 48, 512]),
    ([None, 48, 48, 288], [None, 96, 96, 288]),
    ([None, 96, 96, 160], [None, 192, 192, 160]),
    ([None, 192, 192, 96], [None, 384, 384, 96]),
    ([None, 192, 192, 3], [None, 384, 384, 3]),
    ([None, 384, 384, 72], [None, 768, 768, 72]),
    ([None, 384, 384, 3], [None, 768, 768, 3]),
)


def authenticated_model_resize_inventory(tf, model):
    """Return ResizeBilinear nodes connected to the official model output."""
    outputs = model.run_output
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    ancestors = set()
    pending = [tensor.op for tensor in outputs]
    while pending:
        op = pending.pop()
        if op in ancestors:
            continue
        ancestors.add(op)
        pending.extend(tensor.op for tensor in op.inputs)
    operations = [
        op for op in tf.get_default_graph().get_operations()
        if op in ancestors and op.type == "ResizeBilinear"
    ]
    if len(operations) != len(EXPECTED_MODEL_RESIZE_STAGES):
        raise RuntimeError(
            "official FaceEnhancer model resize inventory mismatch: expected "
            f"{len(EXPECTED_MODEL_RESIZE_STAGES)}, got {len(operations)}")
    records = []
    for index, (op, expected_stage) in enumerate(
            zip(operations, EXPECTED_MODEL_RESIZE_STAGES)):
        input_shape = op.inputs[0].shape.as_list()
        output_shape = op.outputs[0].shape.as_list()
        record = {
            "site_index": index,
            "op_name": op.name,
            "op_type": op.type,
            "source_tensor": op.inputs[0].name,
            "source_op_type": op.inputs[0].op.type,
            "output_tensor": op.outputs[0].name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_dtype": op.inputs[0].dtype.name,
            "align_corners": bool(op.get_attr("align_corners")),
            "half_pixel_centers": bool(op.get_attr("half_pixel_centers")),
        }
        if (input_shape, output_shape) != expected_stage:
            raise RuntimeError(
                f"official FaceEnhancer resize stage {index} mismatch: "
                f"{input_shape} -> {output_shape}")
        if (record["op_type"] != "ResizeBilinear" or
                record["input_dtype"] != "float32" or
                record["align_corners"] or
                record["half_pixel_centers"]):
            raise RuntimeError(
                f"official FaceEnhancer resize contract mismatch: {record}")
        records.append(record)
    return {"count": len(records), "nodes": records}


def pipeline_reference_digest(payload, pipeline_records):
    identity = OrderedDict()
    identity["case_contracts"] = pipeline_records
    for record in pipeline_records:
        name = record["name"]
        keys = [
            f"{name}_{field}"
            for field in ("input", "preprocessed", "padded", "final")
        ] + [
            f"{name}_{field}_{suffix}"
            for field in ("accumulated", "divisor", "stitched", "cropped",
                          "preserved")
            for suffix in ("digest", "probes")
        ]
        identity[name] = {
            key: array_digest(payload[key]) for key in keys
        }
    identity["patch_mask_line"] = array_digest(payload["patch_mask_line"])
    return json_digest(identity)


def reference_bindings(provenance, checkpoint, payload, pipeline_records,
                       resize_contract, model_resize_inventory):
    resize_identity = {
        "synthetic_contract": resize_contract,
        "official_model_inventory": model_resize_inventory,
    }
    return {
        "official_commit": provenance["official_commit"],
        "checkpoint_sha256": checkpoint["sha256"],
        "face_enhancer_source_sha256": provenance["official_source_sha256"][
            "facelib/FaceEnhancer.py"],
        "reference_schema": SCHEMA,
        "raw_input_sha256": array_digest(payload["raw_model_input"]),
        "raw_output_sha256": array_digest(payload["raw_model_output"]),
        "pipeline_reference_sha256": pipeline_reference_digest(
            payload, pipeline_records),
        "resize_contract_sha256": json_digest(resize_identity),
    }


def validate_comparison(comparison, expected_bindings):
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise RuntimeError("Torch comparison schema mismatch")
    if comparison.get("version") != COMPARISON_VERSION:
        raise RuntimeError("Torch comparison version mismatch")
    actual_bindings = comparison.get("bindings")
    if actual_bindings != expected_bindings:
        names = sorted(set(expected_bindings) | set(actual_bindings or {}))
        mismatches = [
            name for name in names
            if (actual_bindings or {}).get(name) != expected_bindings.get(name)
        ]
        raise RuntimeError(
            f"Torch comparison reference binding mismatch: {mismatches}")


def generate_tensorflow(args):
    authentication = authenticate_official_root(args.official_root)
    official_root = authentication["root"]
    official_checkpoint = official_root / "facelib" / "FaceEnhancer.npy"
    official_checkpoint_sha256 = sha256_file(official_checkpoint)
    if official_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "authenticated official checkpoint digest mismatch: expected "
            f"{EXPECTED_CHECKPOINT_SHA256}, got {official_checkpoint_sha256}")
    checkpoint_values, checkpoint_meta = load_checkpoint(
        args.checkpoint, expected_sha256=official_checkpoint_sha256)

    configure_determinism()
    sys.path.insert(0, str(official_root))
    from core.leras import nn  # noqa: E402

    nn.initialize(nn.DeviceConfig.CPU(), data_format="NHWC")
    tf = nn.tf
    tf.set_random_seed(SEED)
    import cv2  # noqa: E402
    from facelib.FaceEnhancer import FaceEnhancer  # noqa: E402

    payload = OrderedDict()
    resize_records = []
    resize_ops = []

    def tf_resize(name, value, target):
        tensor = tf.constant(value[None, ..., None], dtype=tf.float32,
                             name=f"{name}_input")
        resized = tf.image.resize(
            tensor, target, method=tf.image.ResizeMethod.BILINEAR,
            name=f"{name}_resize")
        resize_ops.extend([
            op for op in tf.get_default_graph().get_operations()
            if op.type == "ResizeBilinear" and op not in resize_ops])
        return nn.tf_sess.run(resized)[0, ..., 0]

    target_by_size = {2: (3, 4), 3: (6, 5), 4: (7, 8),
                      5: (10, 9), 6: (12, 12)}
    for size, target in target_by_size.items():
        for pattern_name, value in resize_patterns(size).items():
            name = f"resize_{size}x{size}_{pattern_name}_{target[0]}x{target[1]}"
            output = tf_resize(name, value, target)
            payload[f"{name}_input"] = value
            payload[f"{name}_output"] = output
            resize_records.append({
                "name": name, "pattern": pattern_name,
                "input_shape": list(value.shape),
                "requested_size": list(target),
                "output_shape": list(output.shape),
                "dtype": output.dtype.name,
            })
    for source, target in ((6, 12), (12, 24), (24, 48), (48, 96),
                           (96, 192), (192, 384), (384, 768)):
        value = resize_patterns(source)["random_like"]
        name = f"resize_stage_{source}_to_{target}"
        output = tf_resize(name, value, (target, target))
        payload[f"{name}_input"] = value
        payload[f"{name}_output"] = output
        resize_records.append({
            "name": name, "pattern": "random_like",
            "input_shape": list(value.shape),
            "requested_size": [target, target],
            "output_shape": list(output.shape),
            "dtype": output.dtype.name,
            "face_enhancer_stage": True,
        })

    op_contracts = []
    for op in resize_ops:
        op_contracts.append({
            "op_type": op.type,
            "align_corners": bool(op.get_attr("align_corners")),
            "half_pixel_centers": bool(op.get_attr("half_pixel_centers")),
            "T": op.get_attr("T").name,
        })
    unique_contracts = [dict(items) for items in {
        tuple(sorted(item.items())) for item in op_contracts
    }]
    if unique_contracts != [{
            "T": "float32", "align_corners": False,
            "half_pixel_centers": False, "op_type": "ResizeBilinear"}]:
        raise RuntimeError(f"unexpected resize op contract: {unique_contracts}")

    enhancer = FaceEnhancer(place_model_on_cpu=True, run_on_cpu=True)
    model_resize_inventory = authenticated_model_resize_inventory(
        tf, enhancer.model)
    run_cache = {}

    def model_run(patch):
        key = array_digest(patch)
        if key not in run_cache:
            result = enhancer.model.run([
                patch[None, ...], [np.array([0.2])],
                [np.array([1.0])]])[0]
            run_cache[key] = np.asarray(result, np.float32)
        return run_cache[key]

    cases = OrderedDict([
        ("exact_normal", (synthetic_image(192, 192, np.float32(0.11)),
                          False, False)),
        ("small_tanh", (synthetic_image(127, 181, np.float32(0.23)),
                        True, False)),
        ("large_overlap_normal", (
            synthetic_image(192, 289, np.float32(0.37)), False, False)),
        ("last_pin_tanh", (synthetic_image(192, 193, np.float32(0.41)),
                           True, False)),
        ("nonsquare_preserve_normal", (
            synthetic_image(131, 177, np.float32(0.53)), False, True)),
    ])
    pipeline_records = []
    original_run = enhancer.model.run
    for name, (image, is_tanh, preserve_size) in cases.items():
        result = run_pipeline(model_run, image, is_tanh, preserve_size, cv2)
        _selected_evidence(
            payload, name, result,
            full_intermediates=args.comparison_json is None)

        # Exercise the official public method itself while serving its already
        # frozen raw patch results from the cache.  This proves the instrumented
        # pipeline is byte-for-byte faithful without repeating convolutions.
        def cached_official_run(inputs):
            patch = np.asarray(inputs[0][0], np.float32)
            return model_run(patch)[None, ...]

        enhancer.model.run = cached_official_run
        official_final = enhancer.enhance(
            image.copy(), is_tanh=is_tanh, preserve_size=preserve_size)
        enhancer.model.run = original_run
        public_max_abs = float(np.max(np.abs(official_final - result["final"])))
        if public_max_abs != 0.0:
            raise RuntimeError(
                f"instrumented pipeline differs from official method: "
                f"{name}: {public_max_abs}")
        pipeline_records.append({
            "name": name,
            "input_shape": list(image.shape),
            "is_tanh": is_tanh,
            "preserve_size": preserve_size,
            "padding_top_bottom_left_right": result["padding"],
            "x_coordinates": result["x_coordinates"],
            "y_coordinates": result["y_coordinates"],
            "patches": result["patches"],
            "shapes": {
                field: list(result[field].shape) for field in
                ("preprocessed", "padded", "stitched", "cropped",
                 "preserved", "final")
            },
            "dtypes": {
                field: result[field].dtype.name for field in
                ("preprocessed", "padded", "stitched", "cropped",
                 "preserved", "final")
            },
            "official_public_method_max_abs": public_max_abs,
        })

    exact = cases["exact_normal"][0]
    normalized = np.clip(exact * 2 - 1, -1, 1).astype(np.float32)
    raw = model_run(normalized)
    payload["raw_model_input"] = normalized[None, ...]
    payload["raw_model_output"] = raw[None, ...]
    merger = next(record for record in pipeline_records
                  if record["name"] == "last_pin_tanh")
    mask = patch_mask()
    payload["patch_mask_line"] = mask[:, mask.shape[1] // 2, 0]

    provenance = {
        "classification": "UPSTREAM",
        "official_commit": authentication["commit"],
        "official_tracked_state": authentication["tracked_state"],
        "official_source_sha256": authentication["source_sha256"],
        "checkpoint_matches_official": True,
        "private_media": False,
        "synthetic_inputs_only": True,
    }
    expected_bindings = reference_bindings(
        provenance, checkpoint_meta, payload, pipeline_records,
        unique_contracts[0], model_resize_inventory)
    comparison = None
    if args.comparison_json:
        comparison = json.loads(Path(args.comparison_json).read_text("utf-8"))
        validate_comparison(comparison, expected_bindings)

    metadata = {
        "schema": SCHEMA,
        "generator_version": 2,
        "provenance": provenance,
        "historical_environment": {
            "requirements_source": "requirements-cuda.txt at official commit",
            "tensorflow": "tensorflow-gpu==2.4.0",
            "numpy": "numpy==1.19.3",
            "decision": "not practically reproducible on the available "
                        "Python 3.12 runtime; use the accepted modern "
                        "TensorFlow reference environment",
        },
        "environment": {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "device": "CPU",
            "data_format": "NHWC",
            "tensorflow_eager": bool(tf.executing_eagerly()),
            "environment": ENVIRONMENT,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "omp_threads": 1,
            "seed": SEED,
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "tensorflow_random_seeded": True,
        },
        "checkpoint": checkpoint_meta,
        "resize_op_contract": unique_contracts[0],
        "official_model_resize_inventory": model_resize_inventory,
        "resize_cases": resize_records,
        "raw_model": {
            "input_shape": [1, 192, 192, 3], "input_dtype": "float32",
            "output_shape": [1, 768, 768, 3], "output_dtype": "float32",
            "param": 0.2, "param1": 1.0,
        },
        "pipeline_contract": {
            "patch_size": PATCH_SIZE, "overlap": OVERLAP,
            "upscale": UPSCALE, "patch_mask_shape": list(mask.shape),
            "patch_mask_dtype": mask.dtype.name,
            "patch_mask_sha256": array_digest(mask),
            "final_coordinate_pinning": True,
            "padding": "symmetric zero; odd remainder on bottom/right",
            "blend": "sum(output * linear-ramp mask) / sum(mask)",
            "crop": "padding coordinates multiplied by 4",
            "preserve_size_interpolation": "cv2.INTER_LANCZOS4",
        },
        "pipeline_cases": pipeline_records,
        "normalization": {
            "is_tanh_false_input": "clip(input*2-1,-1,1)",
            "is_tanh_false_final": "clip(output/2+0.5,0,1)",
            "is_tanh_true_input": "identity",
            "is_tanh_true_final": "identity (no FaceEnhancer clipping)",
        },
        "merger_style": {
            "source_case": merger["name"],
            "input_key": "last_pin_tanh_input",
            "output_key": "last_pin_tanh_final",
            "input_clipped_0_1": True,
            "is_tanh": True, "preserve_size": False,
            "spatial_scale": 4,
        },
        "torch_comparison": comparison,
    }
    payload["metadata"] = np.asarray(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")))
    save_deterministic_npz(args.output, payload)
    print(args.output)
    print(json.dumps({
        "schema": SCHEMA,
        "keys": len(payload),
        "checkpoint_sha256": checkpoint_meta["sha256"],
        "raw_output_sha256": array_digest(payload["raw_model_output"]),
    }, sort_keys=True))


def _torch_model(checkpoint):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as functional

    class Model(tnn.Module):
        def __init__(self):
            super().__init__()
            specs = OrderedDict([
                ("conv1", (3, 64)), ("e0_conv0", (64, 64)),
                ("e0_conv1", (64, 64)), ("e1_conv0", (64, 112)),
                ("e1_conv1", (112, 112)), ("e2_conv0", (112, 192)),
                ("e2_conv1", (192, 192)), ("e3_conv0", (192, 336)),
                ("e3_conv1", (336, 336)), ("e4_conv0", (336, 512)),
                ("e4_conv1", (512, 512)),
                ("center_conv0", (512, 512)),
                ("center_conv1", (512, 512)),
                ("center_conv2", (512, 512)),
                ("center_conv3", (512, 512)),
                ("d4_conv0", (1024, 512)), ("d4_conv1", (512, 512)),
                ("d3_conv0", (848, 512)), ("d3_conv1", (512, 512)),
                ("d2_conv0", (704, 288)), ("d2_conv1", (288, 288)),
                ("d1_conv0", (400, 160)), ("d1_conv1", (160, 160)),
                ("d0_conv0", (224, 96)), ("d0_conv1", (96, 96)),
                ("out1x_conv0", (96, 48)), ("out1x_conv1", (48, 3)),
                ("dec2x_conv0", (96, 96)), ("dec2x_conv1", (96, 96)),
                ("out2x_conv0", (96, 48)), ("out2x_conv1", (48, 3)),
                ("dec4x_conv0", (96, 72)), ("dec4x_conv1", (72, 72)),
                ("out4x_conv0", (72, 36)), ("out4x_conv1", (36, 3)),
            ])
            for name, (inputs, outputs) in specs.items():
                setattr(self, name, tnn.Conv2d(inputs, outputs, 3, padding=1))
            self.dense1 = tnn.Linear(1, 64, bias=False)
            self.dense2 = tnn.Linear(1, 64, bias=False)
            with torch.no_grad():
                for name in specs:
                    layer = getattr(self, name)
                    layer.weight.copy_(torch.from_numpy(
                        np.asarray(checkpoint[f"{name}/weight:0"], np.float32)
                        .transpose(3, 2, 0, 1).copy()))
                    layer.bias.copy_(torch.from_numpy(
                        np.asarray(checkpoint[f"{name}/bias:0"], np.float32)
                        .reshape(-1).copy()))
                for name in ("dense1", "dense2"):
                    getattr(self, name).weight.copy_(torch.from_numpy(
                        np.asarray(checkpoint[f"{name}/weight:0"], np.float32)
                        .T.copy()))

        @staticmethod
        def act(value):
            return functional.leaky_relu(value, negative_slope=0.1)

        @staticmethod
        def resize(value):
            return _torch_legacy_resize(value, scale_factor=2)

        def forward(self, image, param, param1):
            x = self.conv1(image)
            x = self.act(x + self.dense1(param)[:, :, None, None] +
                         self.dense2(param1)[:, :, None, None])
            x = self.act(self.e0_conv0(x)); e0 = self.act(self.e0_conv1(x))
            x = functional.avg_pool2d(e0, 2, 2)
            x = self.act(self.e1_conv0(x)); e1 = self.act(self.e1_conv1(x))
            x = functional.avg_pool2d(e1, 2, 2)
            x = self.act(self.e2_conv0(x)); e2 = self.act(self.e2_conv1(x))
            x = functional.avg_pool2d(e2, 2, 2)
            x = self.act(self.e3_conv0(x)); e3 = self.act(self.e3_conv1(x))
            x = functional.avg_pool2d(e3, 2, 2)
            x = self.act(self.e4_conv0(x)); e4 = self.act(self.e4_conv1(x))
            x = functional.avg_pool2d(e4, 2, 2)
            x = self.act(self.center_conv0(x)); x = self.act(self.center_conv1(x))
            x = self.act(self.center_conv2(x)); x = self.act(self.center_conv3(x))
            x = torch.cat((self.resize(x), e4), 1)
            x = self.act(self.d4_conv0(x)); x = self.act(self.d4_conv1(x))
            x = torch.cat((self.resize(x), e3), 1)
            x = self.act(self.d3_conv0(x)); x = self.act(self.d3_conv1(x))
            x = torch.cat((self.resize(x), e2), 1)
            x = self.act(self.d2_conv0(x)); x = self.act(self.d2_conv1(x))
            x = torch.cat((self.resize(x), e1), 1)
            x = self.act(self.d1_conv0(x)); x = self.act(self.d1_conv1(x))
            x = torch.cat((self.resize(x), e0), 1)
            x = self.act(self.d0_conv0(x)); d0 = self.act(self.d0_conv1(x))
            x = self.act(self.out1x_conv0(d0)); x = self.out1x_conv1(x)
            out1x = image + torch.tanh(x)
            x = self.act(self.dec2x_conv0(d0)); x = self.act(self.dec2x_conv1(x))
            d2x = self.resize(x)
            x = self.act(self.out2x_conv0(d2x)); x = self.out2x_conv1(x)
            out2x = self.resize(out1x) + torch.tanh(x)
            x = self.act(self.dec4x_conv0(d2x)); x = self.act(self.dec4x_conv1(x))
            d4x = self.resize(x)
            x = self.act(self.out4x_conv0(d4x)); x = self.out4x_conv1(x)
            return self.resize(out2x) + torch.tanh(x)

    return Model().eval()


def _torch_legacy_resize(value, size=None, scale_factor=None):
    """TensorFlow ResizeBilinear false/false asymmetric coordinates."""
    import torch

    input_height, input_width = value.shape[-2:]
    if size is None:
        output_height = int(input_height * scale_factor)
        output_width = int(input_width * scale_factor)
    else:
        output_height, output_width = size
    y = (torch.arange(output_height, device=value.device,
                      dtype=value.dtype) *
         (float(input_height) / float(output_height)))
    x = (torch.arange(output_width, device=value.device,
                      dtype=value.dtype) *
         (float(input_width) / float(output_width)))
    y0 = torch.floor(y).to(torch.long)
    x0 = torch.floor(x).to(torch.long)
    y1 = torch.clamp(y0 + 1, max=input_height - 1)
    x1 = torch.clamp(x0 + 1, max=input_width - 1)
    wy = (y - y0.to(value.dtype))[None, None, :, None]
    wx = (x - x0.to(value.dtype))[None, None, None, :]
    rows = value[:, :, y0, :] * (1 - wy) + value[:, :, y1, :] * wy
    return rows[:, :, :, x0] * (1 - wx) + rows[:, :, :, x1] * wx


def compare_torch(args):
    import torch
    import torch.nn.functional as functional
    import cv2

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint, checkpoint_meta = load_checkpoint(args.checkpoint)
    model = _torch_model(checkpoint)
    fixture = np.load(args.fixture, allow_pickle=False)
    metadata = json.loads(str(fixture["metadata"]))
    if metadata["schema"] != SCHEMA:
        raise RuntimeError("fixture schema mismatch")
    if metadata["checkpoint"]["sha256"] != checkpoint_meta["sha256"]:
        raise RuntimeError("fixture/checkpoint digest mismatch")
    bindings = reference_bindings(
        metadata["provenance"], metadata["checkpoint"], fixture,
        metadata["pipeline_cases"], metadata["resize_op_contract"],
        metadata["official_model_resize_inventory"])

    resize_results = []
    for record in metadata["resize_cases"]:
        name = record["name"]
        value = fixture[f"{name}_input"]
        expected = fixture[f"{name}_output"]
        tensor = torch.from_numpy(value)[None, None]
        target = tuple(record["requested_size"])
        candidates = {}
        for align in (False, True):
            actual = functional.interpolate(
                tensor, size=target, mode="bilinear", align_corners=align)
            actual = actual[0, 0].detach().numpy()
            candidates[str(align).lower()] = {
                "max_abs": float(np.max(np.abs(actual - expected))),
                "shape_equal": actual.shape == expected.shape,
                "dtype_equal": actual.dtype == expected.dtype,
            }
        actual = _torch_legacy_resize(tensor, size=target)[0, 0].numpy()
        candidates["legacy_asymmetric"] = {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "shape_equal": actual.shape == expected.shape,
            "dtype_equal": actual.dtype == expected.dtype,
        }
        resize_results.append({"name": name, "candidates": candidates})

    inference_cache = {}

    def model_run(patch):
        key = array_digest(patch)
        if key not in inference_cache:
            tensor = torch.from_numpy(
                np.ascontiguousarray(patch.transpose(2, 0, 1)))[None]
            with torch.inference_mode():
                result = model(
                    tensor, torch.tensor([[0.2]], dtype=torch.float32),
                    torch.tensor([[1.0]], dtype=torch.float32))
            inference_cache[key] = result[0].numpy().transpose(1, 2, 0)
        return inference_cache[key]

    case_results = []
    for record in metadata["pipeline_cases"]:
        name = record["name"]
        result = run_pipeline(
            model_run, fixture[f"{name}_input"], record["is_tanh"],
            record["preserve_size"], cv2)
        fields = {}
        for field in ("cropped", "preserved", "final"):
            expected = fixture[f"{name}_{field}"]
            fields[field] = float(np.max(np.abs(result[field] - expected)))
        expected_stitched = fixture[f"{name}_stitched"]
        fields["stitched_max_abs"] = float(np.max(np.abs(
            result["stitched"] - expected_stitched)))
        case_results.append({"name": name, "max_abs": fields})

    raw_input = fixture["raw_model_input"][0]
    raw_actual = model_run(raw_input)
    raw_expected = fixture["raw_model_output"][0]
    raw_max_abs = float(np.max(np.abs(raw_actual - raw_expected)))
    resize_false_max = max(
        r["candidates"]["false"]["max_abs"] for r in resize_results)
    resize_true_max = max(
        r["candidates"]["true"]["max_abs"] for r in resize_results)
    resize_legacy_max = max(
        r["candidates"]["legacy_asymmetric"]["max_abs"]
        for r in resize_results)
    result = {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "bindings": bindings,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "device": "CPU", "threads": 1,
        },
        "resize": {
            "cases": resize_results,
            "align_corners_false_overall_max_abs": resize_false_max,
            "align_corners_true_overall_max_abs": resize_true_max,
            "legacy_asymmetric_overall_max_abs": resize_legacy_max,
        },
        "raw_model_max_abs": raw_max_abs,
        "pipeline_cases": case_results,
        "cuda_measured": False,
    }
    output = json.dumps(result, sort_keys=True, separators=(",", ":"))
    Path(args.comparison_output).write_text(output + "\n", encoding="utf-8")
    print(args.comparison_output)
    print(json.dumps({
        "resize_false_max_abs": resize_false_max,
        "resize_true_max_abs": resize_true_max,
        "resize_legacy_max_abs": resize_legacy_max,
        "raw_model_max_abs": raw_max_abs,
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--comparison-json", type=Path)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    if args.compare_torch:
        if args.fixture is None or args.comparison_output is None:
            parser.error("--compare-torch requires --fixture and "
                         "--comparison-output")
    elif args.official_root is None or args.output is None:
        parser.error("TensorFlow generation requires --official-root and "
                     "--output")
    return args


def main():
    args = parse_args()
    if args.compare_torch:
        compare_torch(args)
    else:
        generate_tensorflow(args)


if __name__ == "__main__":
    main()
