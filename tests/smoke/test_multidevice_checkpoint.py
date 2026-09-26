"""Phase 12 Commit 6: cross-count checkpoint and resume proof.

The suite keeps three claims deliberately separate:

* STRUCTURAL_INVARIANCE compares independently initialized N == 1 and
  SIMULATED_MULTI_REPLICA N == 2 checkpoint schemas, never values.
* INJECTED_CANONICAL_STATE_IDENTITY compares parsed semantic values only
  after the same controlled canonical state is installed in both models.
  It makes no byte-identity claim about pickle streams.
* RESUME_COHERENCE saves state produced by a real production training
  iteration, resumes under the other replica count, proves canonical
  model/optimizer restoration and mirror reconstruction, then executes a
  second production iteration under the new count's semantics.

N > 1 rows use two logical replicas on one physical device (CPU except for
the actual CUDA FP16 scaler-policy row) and are therefore
SIMULATED_MULTI_REPLICA.  They are not physical multi-GPU proof.

Provenance: INDEPENDENT_REIMPLEMENTATION; no external code was copied.
"""

import builtins
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(__file__).resolve().parent
for path in (REPO_ROOT, SMOKE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import core.leras.models  # noqa: F401,E402  (binds nn.ModelBase)
from core.leras import nn as dfl_nn  # noqa: E402
from core.leras.device import Device, DeviceConfig, Devices  # noqa: E402
from core.leras.layers.Saveable import Saveable  # noqa: E402
from core.leras.multidevice import ReplicaPlan  # noqa: E402
from core.leras.optimizers.OptimizerBase import OptimizerBase  # noqa: E402
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    make_model as make_amp,
    make_training_dirs,
)
from Model_SAEHDTest.Model import (  # noqa: E402
    SAEHDHeadless,
    make_model as make_saehd,
)
from test_model_amp_training import synth_samples_nchw  # noqa: E402
from test_model_saehd_training import synth_samples  # noqa: E402
from test_multidevice_amp import seed as amp_seed  # noqa: E402
from test_multidevice_saehd import cpu_seed as saehd_seed  # noqa: E402


CPU = torch.device("cpu")
CUDA = torch.device("cuda:0")
MODEL_KINDS = ("saehd", "amp")
PERSISTED_METADATA_FIELDS = (
    "iter", "options", "loss_history", "sample_for_preview",
    "choosed_gpu_indexes",
)
FORBIDDEN_KEY = re.compile(
    r"(^|[/._:-])(?:"
    r"(?:module|replica|mirror)(?=$|[/._:-])|"
    r"(?:cuda|gpu|device)(?:(?:[_:-]?\d+)(?=$|[/._:-])|(?=$|[./:]))"
    r")",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def _foundation_reset():
    dfl_nn.initialize(DeviceConfig.CPU(), "float32", "NCHW")
    yield
    dfl_nn.initialize(DeviceConfig.CPU(), "float32", "NCHW")


@pytest.fixture
def headless_io(monkeypatch):
    """Keep all real model lifecycle prompts deterministic and headless."""
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "")
    from core.interact import interact as io
    monkeypatch.setattr(io, "input_in_time", lambda seconds, timeout: False)


def _options(kind, optional=False):
    if kind == "saehd":
        return saehd_seed(
            archi="df", batch_size=2, random_warp=False,
            true_face_power=0.01 if optional else 0.0,
            gan_power=0.01 if optional else 0.0,
        )
    return amp_seed(batch_size=2, gan_power=0.1 if optional else 0.0)


def _expected_filenames(kind, optional=False):
    if kind == "saehd":
        names = ["encoder.npy", "inter.npy", "decoder_src.npy",
                 "decoder_dst.npy"]
        if optional:
            names += ["code_discriminator.npy", "GAN.npy"]
        names += ["src_dst_opt.npy"]
        if optional:
            names += ["D_code_opt.npy", "GAN_opt.npy"]
        return names
    names = ["encoder.npy", "inter_src.npy", "inter_dst.npy",
             "decoder.npy", "src_dst_opt.npy"]
    if optional:
        names += ["GAN.npy", "GAN_opt.npy"]
    return names


def _patch_plan(monkeypatch, device, count, events=None):
    devices = [device] * count

    def factory(cls, config):
        plan = ReplicaPlan.from_torch_devices(device, devices)
        if events is not None:
            real_add = plan.add_component
            real_sync = plan.initial_sync

            def add_component(component):
                result = real_add(component)
                events.append(f"mirror_created:{component.name}")
                return result

            def initial_sync():
                result = real_sync()
                events.append("initial_sync_done")
                return result

            plan.add_component = add_component
            plan.initial_sync = initial_sync
        return plan

    monkeypatch.setattr(
        dfl_nn.ReplicaPlan, "from_device_config", classmethod(factory))


def _prepare_root(root):
    root = Path(root)
    make_training_dirs(root)
    return root


def _construct(kind, root, count, monkeypatch, *, optional=False,
               precision="off", events=None, gpu=False):
    if count > 1:
        _patch_plan(monkeypatch, CUDA if gpu else CPU, count, events)
    maker, cls = ((make_saehd, SAEHDHeadless)
                  if kind == "saehd" else (make_amp, AMPHeadless))
    device_args = ({"force_gpu_idxs": [0]} if gpu else {"cpu_only": True})
    return maker(
        cls, root, is_training=True, seed=_options(kind, optional),
        debug=True, precision=precision, **device_args)


def _samples(kind, model, batch=2, seed_no=700):
    if kind == "saehd":
        return synth_samples(
            model.resolution, batch=batch,
            data_format=model.model_data_format, seed_no=seed_no)
    return synth_samples_nchw(model.resolution, batch=batch, seed_no=seed_no)


def _stub_fetch(model, arrays):
    def fetch():
        return ((arrays[0], arrays[1], arrays[2], arrays[3]),
                (arrays[4], arrays[5], arrays[6], arrays[7]))
    model.generate_next_samples = fetch


def _logical_payloads(model):
    payloads = {}
    for _, filename in model.model_filename_list:
        path = Path(model.get_strpath_storage_for_file(filename))
        payload = pickle.loads(path.read_bytes())
        assert isinstance(payload, dict)
        payloads[filename] = payload
    return payloads


def _actual_checkpoint_filenames(model):
    prefix = f"{model.get_model_name()}_"
    return sorted(
        path.name[len(prefix):]
        for path in model.saved_models_path.glob(f"{prefix}*.npy"))


def _contains_scaler_key(value):
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and "scaler" in key.lower())
            or _contains_scaler_key(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_scaler_key(item) for item in value)
    return False


def _schema(value):
    if isinstance(value, np.ndarray):
        return ("array", tuple(value.shape), value.dtype.str)
    if isinstance(value, dict):
        return ("dict", tuple((key, _schema(val))
                              for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_schema(val) for val in value))
    return (type(value).__name__,)


def _assert_semantically_identical(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert np.array_equal(left, right, equal_nan=True)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_semantically_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            _assert_semantically_identical(l_item, r_item)
    else:
        assert left == right


def _assert_canonical_keys(payloads):
    for filename, payload in payloads.items():
        assert payload, filename
        for key in payload:
            assert isinstance(key, str)
            assert FORBIDDEN_KEY.search(key) is None, (filename, key)


def _snapshot_persisted_metadata(model):
    """Value snapshot through the same pickle-compatible type domain."""
    values = {
        "iter": model.iter,
        "options": model.options,
        "loss_history": model.loss_history,
        "sample_for_preview": model.sample_for_preview,
        "choosed_gpu_indexes": model.choosed_gpu_indexes,
    }
    return pickle.loads(pickle.dumps(values, protocol=4))


def _install_resume_metadata_sentinels(kind, model, source_count):
    """Install valid, distinctive values for all five ModelBase fields."""
    seed_options = _options(kind, optional=True)
    assert "target_iter" not in seed_options
    assert "autobackup_hour" not in seed_options

    model.iter = 37 + source_count
    model.options["target_iter"] = 424_200 + source_count
    model.options["autobackup_hour"] = 19 + source_count
    model.loss_history = [
        [101.25 + source_count, 202.5 + source_count],
        [303.75 + source_count, 404.0 + source_count],
        [505.125 + source_count, 606.25 + source_count],
    ]
    arrays = _samples(
        kind, model, batch=2, seed_no=930 + source_count)
    model.sample_for_preview = (
        (arrays[0], arrays[1], arrays[2], arrays[3]),
        (arrays[4], arrays[5], arrays[6], arrays[7]),
    )
    model.choosed_gpu_indexes = [7, 2 ** 30]

    snapshot = _snapshot_persisted_metadata(model)
    assert snapshot["iter"] > 1
    assert snapshot["options"]["target_iter"] > 400_000
    assert len(snapshot["loss_history"]) == 3
    assert snapshot["sample_for_preview"] is not None
    assert snapshot["choosed_gpu_indexes"] == [7, 2 ** 30]
    return snapshot


def _runtime_snapshot(model):
    return {
        filename: [tensor.detach().cpu().clone()
                   for tensor in saveable.get_weights()]
        for saveable, filename in model.model_filename_list
    }


def _assert_runtime_snapshot(model, expected):
    assert [name for _, name in model.model_filename_list] == list(expected)
    for saveable, filename in model.model_filename_list:
        actual = saveable.get_weights()
        wanted = expected[filename]
        assert len(actual) == len(wanted), filename
        assert all(torch.equal(a.detach().cpu(), b)
                   for a, b in zip(actual, wanted)), filename


def _inject_canonical_state(model):
    """Install deterministic state through real canonical Saveable tensors."""
    with torch.no_grad():
        for file_index, (saveable, _) in enumerate(model.model_filename_list):
            for tensor_index, tensor in enumerate(saveable.get_weights()):
                if tensor.dtype.is_floating_point:
                    tensor.fill_(
                        (file_index + 1) * 1.0e-4
                        + (tensor_index + 1) * 1.0e-7)
                else:
                    tensor.fill_(17 + file_index)
    model.iter = 6
    model.loss_history = [[1.25, 2.5]]
    model.sample_for_preview = None
    model.choosed_gpu_indexes = [0, 9999]


def _mirror_tensor_ids(model):
    plan = model.replica_plan
    if plan is None or not plan.is_multi:
        return set()
    return {
        id(tensor)
        for component in plan.components
        for mirror in component.mirrors
        for tensor in list(mirror.parameters()) + list(mirror.buffers())
    }


def _assert_mirrors_synced(model):
    plan = model.replica_plan
    assert plan is not None and plan.is_multi
    for component in plan.components:
        canonical = (list(component.canonical.parameters())
                     + list(component.canonical.buffers()))
        for mirror in component.mirrors:
            mirrored = list(mirror.parameters()) + list(mirror.buffers())
            assert len(canonical) == len(mirrored)
            assert all(torch.equal(a.detach().cpu(), b.detach().cpu())
                       for a, b in zip(canonical, mirrored))


def _optimizers(model):
    return [(filename, saveable)
            for saveable, filename in model.model_filename_list
            if isinstance(saveable, OptimizerBase)]


def _expected_optimizer_params(kind, model, filename):
    if filename == "src_dst_opt.npy":
        return (model.src_dst_saveable_weights if kind == "saehd"
                else model.G_weights)
    if kind == "saehd" and filename == "D_code_opt.npy":
        return model.code_discriminator.get_weights()
    if kind == "saehd" and filename == "GAN_opt.npy":
        return model.D_src.get_weights()
    if kind == "amp" and filename == "GAN_opt.npy":
        return model.GAN.get_weights()
    raise AssertionError(f"unexpected optimizer checkpoint role: {filename}")


def _sync_role_params(kind, model):
    """Production sync order and the canonical parameters for each role."""
    if kind == "saehd":
        return {
            "G": list(model.src_dst_trainable_weights),
            "D_code": list(model.code_discriminator.get_weights()),
            "GAN": list(model.D_src.get_weights()),
        }
    return {
        "G": list(model.G_weights),
        "GAN": list(model.GAN.get_weights()),
    }


def _snapshot_role_params(role_params):
    return {
        role: [param.detach().clone() for param in params]
        for role, params in role_params.items()
    }


def _role_changed(role, role_params, snapshots):
    return any(
        not torch.equal(param.detach(), before)
        for param, before in zip(role_params[role], snapshots[role])
    )


def _assert_optimizer_attachment(kind, model):
    mirror_ids = _mirror_tensor_ids(model)
    for filename, optimizer in _optimizers(model):
        canonical_ids = {id(param) for param in optimizer._weights}
        intended_ids = {
            id(param)
            for param in _expected_optimizer_params(kind, model, filename)}
        owner_ids = {id(param) for param in optimizer._state_owner.values()}
        assert canonical_ids, filename
        assert canonical_ids == intended_ids, filename
        assert owner_ids == canonical_ids, filename
        assert canonical_ids.isdisjoint(mirror_ids), filename
        assert {id(state) for state in optimizer._states()}.isdisjoint(
            mirror_ids), filename


def _track_checkpoint_loads(monkeypatch, events):
    real_load = Saveable.load_weights

    def tracked(saveable, filename):
        result = real_load(saveable, filename)
        events.append(f"load_done:{Path(filename).name}")
        return result

    monkeypatch.setattr(Saveable, "load_weights", tracked)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_structural_invariance_n1_vs_simulated_n2(
        kind, plain_tmp, monkeypatch, headless_io):
    """STRUCTURAL_INVARIANCE; independently initialized values ignored."""
    base = Path(plain_tmp) / kind / "structural"
    schemas = []
    for count in (1, 2):
        root = _prepare_root(base / f"n{count}")
        with monkeypatch.context() as scoped:
            model = _construct(kind, root, count, scoped)
            model.iter = 1
            model.save()
            payloads = _logical_payloads(model)
            assert list(payloads) == _expected_filenames(kind)
            assert _actual_checkpoint_filenames(model) == sorted(
                _expected_filenames(kind))
            _assert_canonical_keys(payloads)
            names = tuple(payloads)
            schema = {name: _schema(payload) for name, payload in payloads.items()}
            metadata = pickle.loads(model.model_data_path.read_bytes())
            schemas.append((names, schema, _schema(metadata)))

            if count == 1:
                assert model.replica_plan is not None
                assert not model.replica_plan.is_multi
                assert model.replica_plan.components == []
            else:
                checkpoint_objects = {
                    id(saveable) for saveable, _ in model.model_filename_list}
                assert all(id(component.canonical) in checkpoint_objects
                           for component in model.replica_plan.components)
                assert not (_mirror_tensor_ids(model) & {
                    id(tensor)
                    for saveable, _ in model.model_filename_list
                    for tensor in saveable.get_weights()})
            model.finalize()

    assert schemas[0] == schemas[1]


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_injected_canonical_state_is_semantically_identical(
        kind, plain_tmp, monkeypatch, headless_io):
    """INJECTED_CANONICAL_STATE_IDENTITY: SEMANTICALLY_IDENTICAL only."""
    base = Path(plain_tmp) / kind / "identity"
    saved = []
    for count in (1, 2):
        root = _prepare_root(base / f"n{count}")
        with monkeypatch.context() as scoped:
            model = _construct(kind, root, count, scoped, optional=True)
            _inject_canonical_state(model)
            if count == 2:
                model.replica_plan.sync_from_canonical()
            model.save()
            payloads = _logical_payloads(model)
            assert list(payloads) == _expected_filenames(kind, optional=True)
            assert _actual_checkpoint_filenames(model) == sorted(
                _expected_filenames(kind, optional=True))
            _assert_canonical_keys(payloads)
            metadata = pickle.loads(model.model_data_path.read_bytes())
            saved.append((payloads, metadata))
            model.finalize()

    _assert_semantically_identical(saved[0][0], saved[1][0])
    for field in ("iter", "options", "loss_history",
                  "sample_for_preview", "choosed_gpu_indexes"):
        _assert_semantically_identical(saved[0][1][field], saved[1][1][field])


@pytest.mark.parametrize("kind", MODEL_KINDS)
@pytest.mark.parametrize("source_count,target_count", ((1, 2), (2, 1)))
def test_cross_count_resume_restores_and_continues_production_training(
        kind, source_count, target_count, plain_tmp, monkeypatch,
        headless_io):
    """RESUME_COHERENCE with real pre-save and post-load train steps."""
    root = _prepare_root(
        Path(plain_tmp) / kind / f"resume_{source_count}_to_{target_count}")

    with monkeypatch.context() as source_patch:
        source = _construct(
            kind, root, source_count, source_patch, optional=True)
        _stub_fetch(source, _samples(kind, source, batch=2, seed_no=810))
        source.train_one_iter()
        assert source.iter == 1
        assert all(math.isfinite(value) for value in source.loss_history[-1])
        before_iterations = {
            filename: int(optimizer.iterations.item())
            for filename, optimizer in _optimizers(source)}
        assert before_iterations
        assert all(value == 1 for value in before_iterations.values())
        expected = _runtime_snapshot(source)
        expected_metadata = _install_resume_metadata_sentinels(
            kind, source, source_count)
        source.save()
        source_payloads = _logical_payloads(source)
        assert list(source_payloads) == _expected_filenames(
            kind, optional=True)
        assert _actual_checkpoint_filenames(source) == sorted(
            _expected_filenames(kind, optional=True))
        _assert_canonical_keys(source_payloads)
        assert not any("scaler" in path.name.lower()
                       for path in root.iterdir())
        source.finalize()

    events = []
    with monkeypatch.context() as target_patch:
        _track_checkpoint_loads(target_patch, events)
        target = _construct(
            kind, root, target_count, target_patch, optional=True,
            events=events)
        _assert_runtime_snapshot(target, expected)
        restored_metadata = _snapshot_persisted_metadata(target)
        assert set(restored_metadata) == set(PERSISTED_METADATA_FIELDS)
        for field in PERSISTED_METADATA_FIELDS:
            _assert_semantically_identical(
                restored_metadata[field], expected_metadata[field])
        assert [name for _, name in target.model_filename_list] == \
            _expected_filenames(kind, optional=True)
        _assert_optimizer_attachment(kind, target)
        assert {
            filename: int(optimizer.iterations.item())
            for filename, optimizer in _optimizers(target)
        } == before_iterations

        load_positions = [i for i, event in enumerate(events)
                          if event.startswith("load_done:")]
        assert load_positions
        role_params = _sync_role_params(kind, target)
        role_snapshots = _snapshot_role_params(role_params)
        sync_observations = []
        if target_count == 2:
            mirror_positions = [i for i, event in enumerate(events)
                                if event.startswith("mirror_created:")]
            sync_position = events.index("initial_sync_done")
            assert mirror_positions
            assert max(load_positions) < min(mirror_positions)
            assert max(mirror_positions) < sync_position
            _assert_mirrors_synced(target)

            def observe_forward(event):
                def delegated(module, args):
                    events.append(event)
                return delegated

            encoder_set = next(
                component for component in target.replica_plan.components
                if component.canonical is target.encoder)
            forward_events = ["forward_r0"] + [
                f"forward_r{replica}"
                for replica in range(1, target.replica_plan.num_replicas)]
            encoder_modules = [target.encoder] + list(encoder_set.mirrors)
            assert len(encoder_modules) == len(forward_events)
            hooks = [
                module.register_forward_pre_hook(observe_forward(event))
                for module, event in zip(encoder_modules, forward_events)
            ]

            real_sync = target.replica_plan.sync_from_canonical
            sync_roles = list(role_params)

            def tracked_sync_from_canonical():
                call_index = len(sync_observations)
                assert call_index < len(sync_roles)
                role = sync_roles[call_index]
                changed = _role_changed(
                    role, role_params, role_snapshots)
                sync_observations.append((role, changed))
                events.append(f"sync_entry:{role}:changed={changed}")
                return real_sync()

            target.replica_plan.sync_from_canonical = \
                tracked_sync_from_canonical
        else:
            assert target.replica_plan is not None
            assert not target.replica_plan.is_multi
            assert target.replica_plan.components == []
            hooks = []

        _stub_fetch(target, _samples(kind, target, batch=2, seed_no=811))
        resumed_iter, _ = target.train_one_iter()
        if hooks:
            for hook in hooks:
                hook.remove()
            for event in forward_events:
                assert event in events
                assert events.index("initial_sync_done") < events.index(event)

        assert resumed_iter == expected_metadata["iter"] + 1
        assert all(math.isfinite(value) for value in target.loss_history[-1])
        after_iterations = {
            filename: int(optimizer.iterations.item())
            for filename, optimizer in _optimizers(target)}
        assert after_iterations == {
            filename: value + 1
            for filename, value in before_iterations.items()}
        if target_count == 2:
            assert sync_observations == [
                (role, True) for role in role_params]
            _assert_mirrors_synced(target)
        else:
            assert all(
                _role_changed(role, role_params, role_snapshots)
                for role in role_params)
        assert target._mp_plan is not None
        assert target._mp_scaler is None  # precision='off', freshly resolved
        target.finalize()


def test_restored_gpu_indexes_compose_with_drop_and_cpu_fallback(
        monkeypatch):
    """ModelBase stores metadata; DeviceConfig applies it when composed."""
    missing = 2 ** 30
    available = Device(
        index=7, tf_dev_type="GPU", name="structural-test-device",
        total_mem=8 * 1024 ** 3, free_mem=4 * 1024 ** 3,
        backend="mock", capability=(8, 0))
    monkeypatch.setattr(
        Devices, "getDevices", staticmethod(lambda: Devices([available])))

    restored_metadata = [7, missing]
    selected = DeviceConfig.GPUIndexes(restored_metadata)
    assert [device.index for device in selected.devices] == [7]
    assert not selected.cpu_only

    assert DeviceConfig.GPUIndexes([missing]).cpu_only
    assert len(DeviceConfig.GPUIndexes([missing]).devices) == 0


def test_forbidden_checkpoint_namespace_token_boundaries():
    must_reject = (
        "module.weight:0",
        "replica.weight:0",
        "mirror.weight:0",
        "cuda.weight:0",
        "gpu.weight:0",
        "device.weight:0",
        "cuda0.weight:0",
        "cuda:0/weight:0",
        "cuda_0.weight:0",
        "gpu1.weight:0",
        "gpu_1.weight:0",
        "device0.weight:0",
        "device_0.weight:0",
    )
    must_accept = (
        "weight:0",
        "encoder/down1/conv1/weight:0",
        "ms_encoder/down1/conv1/weight_0:0",
        "some_gpuish_feature:0",
        "device_norm_like_name:0",
        "encoder/mycuda0_feature:0",
    )
    assert all(FORBIDDEN_KEY.search(key) is not None for key in must_reject)
    assert all(FORBIDDEN_KEY.search(key) is None for key in must_accept)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for real GradScaler recreation")
def test_fp16_scaler_is_runtime_only_and_recreated_on_n1_to_simulated_n2(
        plain_tmp, monkeypatch, headless_io):
    """Actual GradScaler state is absent from files and starts fresh."""
    dfl_nn.initialize_main_env()
    root = _prepare_root(Path(plain_tmp) / "amp" / "fp16_scaler")

    with monkeypatch.context() as source_patch:
        source = _construct(
            "amp", root, 1, source_patch, precision="fp16", gpu=True)
        source._mp_ensure_resolved()
        assert source._mp_scaler is not None
        source._mp_scaler.scale(torch.ones((), device=CUDA))
        source._mp_scaler.update(new_scale=2048.0)
        assert source._mp_scaler.get_scale() == 2048.0
        source.iter = 1
        source.save()
        payloads = _logical_payloads(source)
        assert all(not _contains_scaler_key(payload)
                   for payload in payloads.values())
        metadata = pickle.loads(source.model_data_path.read_bytes())
        assert not _contains_scaler_key(metadata)
        assert not any("scaler" in path.name.lower()
                       for path in root.iterdir())
        source.finalize()

    with monkeypatch.context() as target_patch:
        target = _construct(
            "amp", root, 2, target_patch, precision="fp16", gpu=True)
        assert target.replica_plan.is_multi
        assert target._mp_scaler is not None
        assert target._mp_scaler.get_scale() == 2.0 ** 16
        assert target._mp_scaler.get_scale() != 2048.0
        _assert_mirrors_synced(target)
        target.finalize()
