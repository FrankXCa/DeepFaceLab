"""Generate the frozen synthetic official-TensorFlow Quick96 reference.

This script must be run against an isolated checkout/archive of official DFL
commit 14cc9d4e5ffc062856a739c8e64707654780774e.  It contains no user media and
does not read pretrained weights: inputs and component values are deterministic
synthetic arrays generated below.

Example (paths intentionally repository-relative)::

    <TF_PYTHON> tests/parity/generate_quick96_tf_reference.py \
        --official-root .cache/phase14-official-14cc \
        --output tests/parity/fixtures/quick96_tf_reference.npz
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


OFFICIAL_COMMIT = "14cc9d4e5ffc062856a739c8e64707654780774e"
DETERMINISM_SEED = 140096


class OfficialSourceAuthenticationError(RuntimeError):
    """The supplied oracle source is not the reviewed official Git tree."""


def _git(root, *args):
    command = ["git", "-C", str(root), *args]
    result = subprocess.run(
        command, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise OfficialSourceAuthenticationError(
            f"official source authentication command failed: "
            f"{' '.join(command)}: {detail}")
    return result.stdout.strip()


def authenticate_official_root(root, expected_commit=OFFICIAL_COMMIT):
    """Require an exact, clean Git worktree before importing oracle code.

    Exact HEAD plus an entirely clean tracked worktree is deliberately
    stronger than checking only Quick96's Model.py: the oracle imports the
    architecture, layers, ops, optimizer, and Saveable implementation from
    this same tree.
    """
    root = Path(root).resolve()
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise OfficialSourceAuthenticationError(
            f"--official-root must be the Git worktree root: {root} != {top}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise OfficialSourceAuthenticationError(
            f"official source HEAD mismatch: expected {expected_commit}, "
            f"got {commit}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise OfficialSourceAuthenticationError(
            "official source has tracked modifications; refusing a "
            f"contaminated oracle:\n{dirty}")
    return {"commit": commit, "tracked_state": "clean", "root": root}


def configure_determinism():
    """Fix execution controls before the official tree imports TensorFlow."""
    if "tensorflow" in sys.modules:
        raise RuntimeError(
            "TensorFlow was imported before determinism controls were set")
    environment = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_DETERMINISTIC_OPS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    os.environ.update(environment)
    random.seed(DETERMINISM_SEED)
    np.random.seed(DETERMINISM_SEED)
    return environment


def save_deterministic_npz(path, payload):
    """Write an NPZ with stable member order, timestamps, and attributes."""
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


def _state_records(names, uninterrupted, resumed):
    """Canonical named state evidence for exact continuation comparison."""
    if not (len(names) == len(uninterrupted) == len(resumed)):
        raise RuntimeError("resume state inventory length mismatch")
    records = []
    max_abs = 0.0
    for name, expected, actual in zip(names, uninterrupted, resumed):
        expected = np.ascontiguousarray(np.asarray(expected))
        actual = np.ascontiguousarray(np.asarray(actual))
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise RuntimeError(f"resume state metadata mismatch for {name}")
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"resume continuation mismatch for {name}")
        if expected.size:
            max_abs = max(max_abs, float(np.max(np.abs(
                expected.astype(np.float64) - actual.astype(np.float64)))))
        header = json.dumps(
            {"name": name, "dtype": expected.dtype.name,
             "shape": list(expected.shape)},
            sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_digest = hashlib.sha256(
            header + b"\0" + expected.tobytes(order="C")).hexdigest()
        actual_digest = hashlib.sha256(
            header + b"\0" + actual.tobytes(order="C")).hexdigest()
        records.append({
            "name": name,
            "dtype": expected.dtype.name,
            "shape": list(expected.shape),
            "uninterrupted_sha256": expected_digest,
            "resumed_sha256": actual_digest,
        })
    return records, max_abs


def _stable_number(name):
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4],
                          "little")


def synthetic_value(name, shape):
    """Small, non-symmetric deterministic official-layout tensor."""
    size = int(np.prod(shape, dtype=np.int64))
    base = np.arange(size, dtype=np.float32)
    phase = _stable_number(name) % 251
    value = (((base + phase) % 251.0) - 125.0) * np.float32(2e-5)
    return value.reshape(shape)


def synthetic_inputs(batch=2):
    pixels = batch * 96 * 96 * 3
    masks = batch * 96 * 96
    p = np.arange(pixels, dtype=np.float32).reshape(batch, 96, 96, 3)
    m = np.arange(masks, dtype=np.float32).reshape(batch, 96, 96, 1)
    return {
        "warped_src": ((p * 0.013 + 0.11) % 1.0).astype(np.float32),
        "target_src": ((p * 0.017 + 0.23) % 1.0).astype(np.float32),
        "warped_dst": ((p * 0.019 + 0.37) % 1.0).astype(np.float32),
        "target_dst": ((p * 0.023 + 0.41) % 1.0).astype(np.float32),
        "target_srcm": ((m * 0.029 + 0.07) % 1.0).astype(np.float32),
        "target_dstm": ((m * 0.031 + 0.13) % 1.0).astype(np.float32),
    }


def _sub_name(variable, scope):
    prefix = scope + "/"
    if not variable.name.startswith(prefix):
        raise RuntimeError(f"{variable.name} is outside {scope}")
    return variable.name[len(prefix):]


def _metadata(saveable, scope):
    return [
        {"key": _sub_name(v, scope),
         "shape": v.shape.as_list(),
         "dtype": v.dtype.as_numpy_dtype.__name__}
        for v in saveable.get_weights()
    ]


def _probes(values):
    result = []
    for value in values:
        flat = np.asarray(value).reshape(-1)
        indexes = sorted({0, len(flat) // 2, len(flat) - 1})
        result.append([float(flat[i]) for i in indexes])
    return np.asarray(result, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    authentication = authenticate_official_root(args.official_root)
    official_root = authentication["root"]
    expected = official_root / "models" / "Model_Quick96" / "Model.py"
    if not expected.is_file():
        raise SystemExit("--official-root is not an official DFL source tree")
    determinism_environment = configure_determinism()
    sys.path.insert(0, str(official_root))

    from core.leras import nn  # noqa: E402

    nn.initialize(nn.DeviceConfig.CPU(), data_format="NHWC")
    tf = nn.tf
    tf.set_random_seed(DETERMINISM_SEED)
    resolution = 96
    archi = nn.DeepFakeArchi(resolution, opts="ud")
    encoder = archi.Encoder(3, 64, name="encoder")
    encoder_out_ch = encoder.get_out_ch() * encoder.get_out_res(96) ** 2
    inter = archi.Inter(encoder_out_ch, 128, 128, name="inter")
    decoder_src = archi.Decoder(inter.get_out_ch(), 64, 16,
                                name="decoder_src")
    decoder_dst = archi.Decoder(inter.get_out_ch(), 64, 16,
                                name="decoder_dst")
    components = (encoder, inter, decoder_src, decoder_dst)

    shape_bgr = nn.get4Dshape(96, 96, 3)
    shape_mask = nn.get4Dshape(96, 96, 1)
    warped_src = tf.placeholder(nn.floatx, shape_bgr)
    target_src = tf.placeholder(nn.floatx, shape_bgr)
    target_srcm = tf.placeholder(nn.floatx, shape_mask)
    warped_dst = tf.placeholder(nn.floatx, shape_bgr)
    target_dst = tf.placeholder(nn.floatx, shape_bgr)
    target_dstm = tf.placeholder(nn.floatx, shape_mask)

    encoder_src = encoder(warped_src)
    encoder_dst = encoder(warped_dst)
    inter_src = inter(encoder_src)
    inter_dst = inter(encoder_dst)
    pred_src_src, pred_src_srcm = decoder_src(inter_src)
    pred_dst_dst, pred_dst_dstm = decoder_dst(inter_dst)
    pred_src_dst, pred_src_dstm = decoder_src(inter_dst)

    srcm_blur = nn.gaussian_blur(target_srcm, 3)
    dstm_blur = nn.gaussian_blur(target_dstm, 3)
    target_src_masked = target_src * srcm_blur
    pred_src_masked = pred_src_src * srcm_blur
    target_dst_masked = target_dst * dstm_blur
    pred_dst_masked = pred_dst_dst * dstm_blur
    filter_size = int(96 / 11.6)
    src_loss = tf.reduce_mean(
        10 * nn.dssim(target_src_masked, pred_src_masked,
                      max_val=1.0, filter_size=filter_size), axis=[1])
    src_loss += tf.reduce_mean(
        10 * tf.square(target_src_masked - pred_src_masked),
        axis=[1, 2, 3])
    src_loss += tf.reduce_mean(
        10 * tf.square(target_srcm - pred_src_srcm), axis=[1, 2, 3])
    dst_loss = tf.reduce_mean(
        10 * nn.dssim(target_dst_masked, pred_dst_masked,
                      max_val=1.0, filter_size=filter_size), axis=[1])
    dst_loss += tf.reduce_mean(
        10 * tf.square(target_dst_masked - pred_dst_masked),
        axis=[1, 2, 3])
    dst_loss += tf.reduce_mean(
        10 * tf.square(target_dstm - pred_dst_dstm), axis=[1, 2, 3])

    trainable = sum((component.get_weights() for component in components), [])

    # Controlled lr_dropout mask: exactly the same deterministic mask is used
    # by the Torch comparison. This is deliberately separate from the fresh-
    # random-mask statistical test; cross-framework RNG identity is not a
    # checkpoint or parity requirement.
    mask_index = {"value": 0}
    original_random_binomial = nn.random_binomial

    def controlled_binomial(shape, p, dtype):
        index = mask_index["value"]
        mask_index["value"] += 1
        size = int(np.prod(shape.as_list(), dtype=np.int64))
        phase = _stable_number(f"mask-{index}") % 10
        values = (((np.arange(size) + phase) % 10) < 3).astype(np.float32)
        return tf.constant(values.reshape(shape.as_list()), dtype=dtype)

    nn.random_binomial = controlled_binomial
    optimizer = nn.RMSprop(
        lr=2e-4, rho=0.9, lr_dropout=0.3, name="src_dst_opt")
    optimizer.initialize_variables(trainable, vars_on_cpu=True)
    nn.random_binomial = original_random_binomial

    controlled_count = 4
    controlled_gv = [
        (tf.ones_like(v) * np.float32((i + 1) * 1e-3), v)
        for i, v in enumerate(trainable)]
    update = optimizer.get_update_op(controlled_gv)

    nn.tf_sess.run(tf.global_variables_initializer())
    for component in components:
        values = [
            synthetic_value(v.name, v.shape.as_list())
            for v in component.get_weights()]
        component.set_weights(values)

    inputs = synthetic_inputs()
    feed = {
        warped_src: inputs["warped_src"],
        target_src: inputs["target_src"],
        target_srcm: inputs["target_srcm"],
        warped_dst: inputs["warped_dst"],
        target_dst: inputs["target_dst"],
        target_dstm: inputs["target_dstm"],
    }
    fetches = [
        encoder_src, inter_src, pred_src_src, pred_src_srcm,
        pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm,
        srcm_blur, dstm_blur, src_loss, dst_loss]
    result = nn.tf_sess.run(fetches, feed_dict=feed)

    before = nn.tf_sess.run(trainable[:controlled_count])
    nn.tf_sess.run(update)
    after = nn.tf_sess.run(trainable[:controlled_count])
    accumulators = nn.tf_sess.run(
        list(optimizer.accumulators_dict.values())[:controlled_count])

    checkpoint_meta = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for component in components:
            path = tmp / f"{component.name}.npy"
            component.save_weights(path)
            payload = pickle.loads(path.read_bytes())
            checkpoint_meta[path.name] = [
                {"key": key, "shape": list(value.shape),
                 "dtype": value.dtype.name}
                for key, value in payload.items()]
        opt_path = tmp / "src_dst_opt.npy"
        optimizer.save_weights(opt_path)
        opt_payload = pickle.loads(opt_path.read_bytes())
        checkpoint_meta[opt_path.name] = [
            {"key": key, "shape": list(np.asarray(value).shape),
             "dtype": np.asarray(value).dtype.name}
            for key, value in opt_payload.items()]

        saved_outputs = nn.tf_sess.run(
            [pred_src_src, pred_src_srcm], feed_dict=feed)
        nn.batch_set_value([(v, np.zeros(v.shape.as_list(), np.float32))
                            for v in trainable[:controlled_count]])
        for component in components:
            component.load_weights(tmp / f"{component.name}.npy")
        optimizer.load_weights(opt_path)
        loaded_outputs = nn.tf_sess.run(
            [pred_src_src, pred_src_srcm], feed_dict=feed)
        save_load_max_abs = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(saved_outputs, loaded_outputs))

        # The checkpoint above is the interrupted point (after step 1).
        nn.tf_sess.run(update)
        uninterrupted_model = nn.tf_sess.run(trainable)
        uninterrupted_optimizer = nn.tf_sess.run(optimizer.get_weights())
        for component in components:
            component.load_weights(tmp / f"{component.name}.npy")
        optimizer.load_weights(opt_path)
        nn.tf_sess.run(update)
        resumed_model = nn.tf_sess.run(trainable)
        resumed_optimizer = nn.tf_sess.run(optimizer.get_weights())

        model_names = [v.name for v in trainable]
        optimizer_names = [v.name for v in optimizer.get_weights()]
        resume_model_records, model_resume_max_abs = _state_records(
            model_names, uninterrupted_model, resumed_model)
        resume_optimizer_records, optimizer_resume_max_abs = _state_records(
            optimizer_names, uninterrupted_optimizer, resumed_optimizer)
        resume_max_abs = max(model_resume_max_abs, optimizer_resume_max_abs)

    metadata = {
        "provenance": {
            "classification": "UPSTREAM",
            "official_commit": authentication["commit"],
            "official_tracked_state": authentication["tracked_state"],
            "quick96_source_sha256": hashlib.sha256(
                expected.read_bytes()).hexdigest(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "data_format": "NHWC",
            "private_media": False,
            "pretrained_bytes": False,
        },
        "determinism": {
            "seed": DETERMINISM_SEED,
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "tensorflow_random_seeded": True,
            "tensorflow_eager": bool(tf.executing_eagerly()),
            "device": "CPU",
            "environment": determinism_environment,
        },
        "architecture": {
            "resolution": 96, "opts": "ud", "e_dims": 64,
            "ae_dims": 128, "d_dims": 64, "d_mask_dims": 16,
        },
        "components": {
            component.name: _metadata(component, component.name)
            for component in components
        },
        "optimizer": {
            "lr": 2e-4, "rho": 0.9, "lr_dropout": 0.3,
            "epsilon": "np.finfo(float32).resolution",
            "controlled_parameter_names": [v.name for v in
                                               trainable[:controlled_count]],
            "iters_dtype": optimizer.iterations.dtype.as_numpy_dtype.__name__,
            "state_count": len(optimizer.get_weights()),
        },
        "checkpoint": checkpoint_meta,
        "save_load_max_abs": save_load_max_abs,
        "resume_max_abs": resume_max_abs,
        "resume": {
            "model_count": len(resume_model_records),
            "optimizer_count": len(resume_optimizer_records),
            "optimizer_iteration_name": optimizer_names[0],
            "model": resume_model_records,
            "optimizer": resume_optimizer_records,
        },
        "predictor_order": ["pred_src_dst", "pred_src_dstm",
                            "pred_dst_dstm"],
    }

    names = [
        "encoder_src", "inter_src", "decoder_src_face",
        "decoder_src_mask", "decoder_dst_face", "decoder_dst_mask",
        "ae_merge_face", "ae_merge_src_mask", "src_mask_blur",
        "dst_mask_blur", "src_loss", "dst_loss"]
    payload = {name: value for name, value in zip(names, result)}
    payload.update(inputs)
    payload.update(
        ae_view_src=result[2], ae_view_dst=result[4],
        ae_view_dst_mask=result[5], ae_view_swap=result[6],
        ae_view_swap_mask=result[7],
        ae_merge_dst_mask=result[5],
        parameter_before_probes=_probes(before),
        parameter_after_probes=_probes(after),
        accumulator_probes=_probes(accumulators),
        optimizer_iters=np.asarray(1, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    save_deterministic_npz(args.output, payload)
    print(args.output)
    print(json.dumps({
        "save_load_max_abs": save_load_max_abs,
        "resume_max_abs": resume_max_abs,
        "optimizer_state_count": len(optimizer.get_weights()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
