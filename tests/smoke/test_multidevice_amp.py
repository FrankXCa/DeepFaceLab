"""Phase 12 Commit 5: AMP multi-replica acceptance.

All training tests drive the real ``models.Model_AMP.AMPModel`` through
the headless option seam.  N > 1 rows use logical replicas on one device
and are therefore SIMULATED_MULTI_REPLICA, never a physical multi-GPU
claim.  Physical 2-GPU remains PENDING_ENVIRONMENTALLY / NOT_VERIFIED;
physical FP16 multi-GPU remains EXPERIMENTAL / NOT_VERIFIED.

Provenance: INDEPENDENT_REIMPLEMENTATION of the official DFL semantics.
"""

import builtins
import copy
import importlib
import inspect
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SMOKE = Path(__file__).resolve().parent
if str(SMOKE) not in sys.path:
    sys.path.insert(0, str(SMOKE))

import core.leras.models  # noqa: F401,E402
from core.leras import nn as dfl_nn  # noqa: E402
from core.leras.multidevice import ReplicaPlan  # noqa: E402
amp_module = importlib.import_module("models.Model_AMP.Model")  # noqa: E402
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    DEFAULT_SEED_OPTIONS,
    make_model as make_amp,
    make_training_dirs,
)
from test_model_amp_training import synth_samples_nchw  # noqa: E402


CPU = torch.device("cpu")
CUDA = torch.device("cuda:0")
CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="Phase 12 AMP GPU tier requires CUDA")
FP16_INIT_SCALE = 2.0 ** 16


@pytest.fixture(autouse=True)
def _foundation_reset():
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")


@pytest.fixture
def headless_io(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    from core.interact import interact as io
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


def seed(**overrides):
    out = dict(DEFAULT_SEED_OPTIONS)
    out.update(dict(
        resolution=64, batch_size=2, ae_dims=32, inter_dims=32,
        e_dims=16, d_dims=16, d_mask_dims=6, morph_factor=1.0,
        random_warp=False, gan_power=0.0, gan_patch_size=8, gan_dims=8,
        models_opt_on_gpu=True, lr_dropout="n", clipgrad=False,
        random_src_flip=False, random_dst_flip=False,
    ))
    out.update(overrides)
    return out


def patch_plan(monkeypatch, device, n):
    devices = [device] * n

    def fake(cls, cfg):
        return ReplicaPlan.from_torch_devices(device, devices)

    monkeypatch.setattr(
        dfl_nn.ReplicaPlan, "from_device_config", classmethod(fake))


def build_cpu(tmp, monkeypatch, n=1, cls=AMPHeadless, **options):
    root = Path(tmp) / f"amp_{n}"
    if n > 1:
        patch_plan(monkeypatch, CPU, n)
    make_training_dirs(root)
    return make_amp(
        cls, root, is_training=True, seed=seed(**options), debug=True,
        cpu_only=True)


_gpu_env_ready = False


def build_gpu(tmp, monkeypatch, cls, precision, **options):
    global _gpu_env_ready
    if not _gpu_env_ready:
        dfl_nn.initialize_main_env()
        _gpu_env_ready = True
    root = Path(tmp) / f"amp_gpu_{precision}"
    patch_plan(monkeypatch, CUDA, 2)
    make_training_dirs(root)
    return make_amp(
        cls, root, is_training=True, seed=seed(**options), debug=True,
        force_gpu_idxs=[0], precision=precision)


def stub_fetch(model, arrays):
    def fetch():
        return ((arrays[0], arrays[1], arrays[2], arrays[3]),
                (arrays[4], arrays[5], arrays[6], arrays[7]))
    model.generate_next_samples = fetch


def marked8(res, n):
    """NCHW rows with distinguishable src/dst/sample signatures."""
    arrays = list(synth_samples_nchw(res, batch=n, seed_no=410))
    for i in range(n):
        src_level = 0.08 + 0.12*i
        dst_level = 0.55 + 0.08*i
        arrays[0][i] = src_level
        arrays[1][i] = src_level + 0.03
        arrays[4][i] = dst_level
        arrays[5][i] = min(0.98, dst_level + 0.03)
        arrays[2][i] = 0.0
        arrays[2][i, :, :, :8 + 8*i] = 1.0
        arrays[6][i] = 0.0
        arrays[6][i, :, :12 + 6*i, :] = 1.0
        arrays[3][i] = float(i % 2)
        arrays[7][i] = float((i + 1) % 2)
    return tuple(np.ascontiguousarray(x) for x in arrays)


def module_tensors(module):
    return list(module.parameters()) + list(module.buffers())


def assert_synced(model):
    plan = model.replica_plan
    assert plan is not None and plan.is_multi
    for cset in plan.components:
        for mirror in cset.mirrors:
            ca = [x.detach().cpu() for x in module_tensors(cset.canonical)]
            ma = [x.detach().cpu() for x in module_tensors(mirror)]
            assert len(ca) == len(ma)
            assert all(torch.equal(a, b) for a, b in zip(ca, ma))


def component_names(model):
    return {c.canonical.name for c in model.replica_plan.components}


def mirror_ids(plan):
    return {id(p) for c in plan.components for row in c.mirror_params.values()
            for p in row}


def copy_components(src, dst):
    for name in ("encoder", "inter_src", "inter_dst", "decoder"):
        getattr(dst, name).load_state_dict(
            copy.deepcopy(getattr(src, name).state_dict()))
    if hasattr(src, "GAN") and hasattr(dst, "GAN"):
        dst.GAN.load_state_dict(copy.deepcopy(src.GAN.state_dict()))


class _ZeroBatchAMP(AMPHeadless):
    def on_initialize(self):
        self.set_batch_size(0)
        return super().on_initialize()


class _BatchAMP(AMPHeadless):
    requested_batch = 1

    def on_initialize(self):
        self.set_batch_size(type(self).requested_batch)
        return super().on_initialize()


class _HookedAMP(AMPHeadless):
    def _init_hooks(self):
        self.counts = {"backward": 0, "unscale": 0, "step": 0,
                       "update": 0, "stepped": []}
        self.inject_class_b = False
        self.retry_cleanup = None

    def _mp_backward(self, loss):
        self.counts["backward"] += 1
        if (self.inject_class_b and self.counts["backward"] == 2
                and self.counts["step"] == 0):
            loss = loss * 1.0e30
        if self.inject_class_b and self.counts["backward"] == 3:
            states = []
            for cset in self.replica_plan.components:
                states += [p.grad is None for p in cset.canonical.parameters()]
                for mirror in cset.mirrors:
                    states += [p.grad is None for p in mirror.parameters()]
            self.retry_cleanup = bool(states) and all(states)
        return super()._mp_backward(loss)

    def _mp_unscale_opt(self, opt, active_weights=None):
        self.counts["unscale"] += 1
        return super()._mp_unscale_opt(opt, active_weights)

    def _mp_opt_step(self, opt, grads_vars):
        result = super()._mp_opt_step(opt, grads_vars)
        self.counts["step"] += 1
        self.counts["stepped"].append(result)
        return result

    def _mp_scaler_update(self):
        self.counts["update"] += 1
        return super()._mp_scaler_update()


class _OrderAMP(AMPHeadless):
    events = None

    def set_training_data_generators(self, generators):
        result = super().set_training_data_generators(generators)
        type(self).events.append("generators_done")
        return result

    def _mp_ensure_resolved(self):
        result = super()._mp_ensure_resolved()
        if (self._mp_plan is not None
                and "precision_resolved" not in type(self).events):
            type(self).events.append("precision_resolved")
        return result


def bind_test_module(cls, name):
    module = types.ModuleType(name)
    module.__file__ = str(SMOKE / "Model_AMPTest" / f"{name}.py")
    sys.modules[name] = module
    cls.__module__ = name


bind_test_module(_ZeroBatchAMP, "_amp_zero_batch")
bind_test_module(_BatchAMP, "_amp_batch_rows")
bind_test_module(_HookedAMP, "_amp_hooked_gpu")
bind_test_module(_OrderAMP, "_amp_order")


def test_n1_normal_invariance_and_no_mirrors(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=1, morph_factor=1.0)
    assert m.replica_plan is not None and not m.replica_plan.is_multi
    assert m.replica_plan.components == []
    assert m._mp_plan is None  # Phase-8 lazy resolution is preserved.
    a8 = synth_samples_nchw(64, batch=1, seed_no=1)
    src, dst = m.train(*a8)
    assert src.shape == dst.shape == (1,)
    assert int(m.src_dst_opt.iterations.item()) == 1
    m.finalize()


def test_b0_n1_official_correction_executes_production(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=1, cls=_ZeroBatchAMP)
    assert m.get_batch_size() == m.options["batch_size"] == 1
    # Both generators are constructed after the corrected write-back.
    assert all(g.batch_size == 1 for g in m.generator_list)
    m.finalize()


@pytest.mark.parametrize("batch,n,expected", [
    (7, 2, 6), (2, 3, 3), (1, 2, 2), (0, 2, 2),
])
def test_official_batch_formula_rows_execute_production(
        tmp_path, monkeypatch, batch, n, expected):
    patch_plan(monkeypatch, CPU, n)
    real_get = dfl_nn.getCurrentDeviceConfig
    real_cfg = real_get()
    fake_cfg = types.SimpleNamespace(devices=[CPU] * n)
    monkeypatch.setattr(dfl_nn, "getCurrentDeviceConfig", lambda: fake_cfg)
    _BatchAMP.requested_batch = batch
    root = tmp_path / f"b{batch}_n{n}"
    make_training_dirs(root)
    m = make_amp(_BatchAMP, root, is_training=True, seed=seed(),
                 debug=True, cpu_only=True)
    assert m.get_batch_size() == m.options["batch_size"] == expected
    # Debug generators intentionally force runtime batches to one, but
    # their constructor receives the production-adjusted value; the
    # write-back above is the production batch contract.
    assert real_cfg is not None
    m.finalize()


def test_simulated_n2_full_iteration_gan_off(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, morph_factor=1.0)
    assert component_names(m) == {"encoder", "inter_src", "inter_dst", "decoder"}
    a8 = marked8(64, 4)
    stub_fetch(m, a8)
    it, _ = m.train_one_iter()
    assert it == 1
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert all(math.isfinite(x) for x in m.loss_history[-1])
    assert_synced(m)
    m.finalize()


def test_component_and_optimizer_ownership(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, gan_power=0.1)
    assert component_names(m) == {
        "encoder", "inter_src", "inter_dst", "decoder", "GAN"}
    owned = set(m.src_dst_opt._weight_keys)
    expected_g = ({id(p) for p in m.encoder.get_weights()}
                  | {id(p) for p in m.decoder.get_weights()})
    assert owned == expected_g
    assert not ({id(p) for p in m.inter_src.get_weights()} & owned)
    assert not ({id(p) for p in m.inter_dst.get_weights()} & owned)
    assert not (mirror_ids(m.replica_plan) & owned)
    assert not (mirror_ids(m.replica_plan) & set(m.GAN_opt._weight_keys))
    assert set(m.GAN_opt._weight_keys) == {
        id(p) for p in m.GAN.get_weights()}
    m.finalize()


def test_contiguous_b4_n2_and_anti_rebinding(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, morph_factor=1.0)
    a8 = marked8(64, 4)
    enc_set = next(c for c in m.replica_plan.components
                   if c.canonical is m.encoder)
    seen = {0: [], 1: []}

    def hook_for(replica):
        def hook(module, args):
            x = args[0].detach().cpu().numpy()
            seen[replica].append([float(row.mean()) for row in x])
        return hook

    h0 = m.encoder.register_forward_pre_hook(hook_for(0))
    h1 = enc_set.mirrors[0].register_forward_pre_hook(hook_for(1))
    src, dst = m.train(*a8)
    h0.remove(); h1.remove()
    assert src.shape == dst.shape == (4,)
    # Each replica sees src then dst; each call retains two rows.  A
    # rebound/shortened source would produce an empty or one-row call.
    assert [len(x) for x in seen[0]] == [2, 2]
    assert [len(x) for x in seen[1]] == [2, 2]
    assert seen[0][0][0] < seen[0][0][1] < seen[1][0][0] < seen[1][0][1]
    assert seen[0][1][0] < seen[0][1][1] < seen[1][1][0] < seen[1][1][1]


def test_all_eight_inputs_same_shard_and_loss_order(tmp_path, monkeypatch):
    multi = build_cpu(tmp_path / "multi", monkeypatch, n=2, morph_factor=1.0)
    # Restore the production plan factory before constructing N==1.
    monkeypatch.undo()
    single = build_cpu(tmp_path / "single", monkeypatch, n=1, morph_factor=1.0)
    copy_components(multi, single)
    multi.replica_plan.sync_from_canonical()
    a8 = marked8(64, 4)

    # Suppress only the optimizer mutation; the real production
    # prepare/forward/loss/backward/aggregation path still runs.
    multi.src_dst_opt.get_update_op = lambda gv: (lambda: None)
    single.src_dst_opt.get_update_op = lambda gv: (lambda: None)
    msrc, mdst = multi.train(*a8)
    rsrc, rdst = [], []
    for i in range(4):
        row = tuple(x[i:i+1] for x in a8)
        s, d = single.train(*row)
        rsrc.append(s.detach()); rdst.append(d.detach())
    # Batched convolution vs four one-row convolutions can differ by a
    # few fp32 ulps; the distinguishable rows make any reorder obvious.
    assert torch.allclose(msrc.cpu(), torch.cat(rsrc).cpu(), atol=5e-4, rtol=0)
    assert torch.allclose(mdst.cpu(), torch.cat(rdst).cpu(), atol=5e-4, rtol=0)


def test_initialization_binding_order(tmp_path, monkeypatch):
    events = []
    _OrderAMP.events = events
    devices = [CPU, CPU]

    def factory(cls, cfg):
        plan = ReplicaPlan.from_torch_devices(CPU, devices)
        events.append("plan_created")
        real_add = plan.add_component
        real_sync = plan.initial_sync

        def add(component):
            if "mirror_built" not in events:
                events.append("mirror_built")
            return real_add(component)

        def initial_sync():
            events.append("initial_sync")
            return real_sync()

        plan.add_component = add
        plan.initial_sync = initial_sync
        return plan

    monkeypatch.setattr(
        dfl_nn.ReplicaPlan, "from_device_config", classmethod(factory))
    m = build_cpu(tmp_path, monkeypatch, n=1, cls=_OrderAMP)
    hook = m.encoder.register_forward_pre_hook(
        lambda module, args: events.append("first_forward"))
    m.train(*synth_samples_nchw(64, batch=2, seed_no=19))
    hook.remove()
    positions = {event: events.index(event) for event in (
        "plan_created", "precision_resolved", "generators_done",
        "mirror_built", "initial_sync", "first_forward")}
    assert positions["plan_created"] < positions["precision_resolved"]
    assert positions["precision_resolved"] < positions["generators_done"]
    assert positions["generators_done"] < positions["mirror_built"]
    assert positions["mirror_built"] < positions["initial_sync"]
    assert positions["initial_sync"] < positions["first_forward"]


@pytest.mark.parametrize("morph", [0.25, 1.0])
def test_morph_factor_shared_and_exact_k_per_replica(
        tmp_path, monkeypatch, morph):
    calls = []
    real = amp_module.exact_k_morph_mask

    def wrapped(batch, dims, factor, **kwargs):
        out = real(batch, dims, factor, **kwargs)
        calls.append((batch, dims, factor, out.detach().cpu()))
        return out

    monkeypatch.setattr(amp_module, "exact_k_morph_mask", wrapped)
    m = build_cpu(tmp_path, monkeypatch, n=2, morph_factor=morph)
    # ModelBase's construction-time preview rendering performs several
    # canonical one-row AE_view calls.  This test observes only the
    # direct multi-replica training closure below.
    calls.clear()
    m.train(*marked8(64, 4))
    assert len(calls) == 2
    for batch, dims, factor, mask in calls:
        assert batch == 2 and dims == 32 and factor == morph
        assert torch.equal(mask.flatten(1).sum(1),
                           torch.full((2,), int(32*morph), dtype=mask.dtype))


def test_loss_concat_and_history_mean(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, morph_factor=1.0)
    a8 = marked8(64, 4)
    src, dst = m.train(*a8)
    assert src.shape == dst.shape == (4,)
    # Use a fresh model for the lifecycle/history assertion.
    m2 = build_cpu(tmp_path / "history", monkeypatch, n=2, morph_factor=1.0)
    stub_fetch(m2, a8)
    captured = {}
    real_train = m2.train

    def wrapped(*args):
        out = real_train(*args)
        captured["loss"] = tuple(x.detach().clone() for x in out)
        return out

    m2.train = wrapped
    m2.train_one_iter()
    assert m2.loss_history[-1][0] == pytest.approx(
        float(captured["loss"][0].mean()))
    assert m2.loss_history[-1][1] == pytest.approx(
        float(captured["loss"][1].mean()))


def test_gan_on_one_step_post_g_and_order(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, gan_power=0.1,
                  morph_factor=1.0)
    a8 = marked8(64, 4)
    stub_fetch(m, a8)
    pre = next(m.encoder.parameters()).detach().clone()
    events = []
    real_g, real_d = m.train, m.GAN_train

    def g(*args):
        events.append("G")
        out = real_g(*args)
        events.append("G_done")
        return out

    def d(*args):
        events.append("GAN")
        assert not torch.equal(pre, next(m.encoder.parameters()).detach())
        return real_d(*args)

    m.train, m.GAN_train = g, d
    m.train_one_iter()
    assert events == ["G", "G_done", "GAN"]
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.GAN_opt.iterations.item()) == 1
    assert_synced(m)


def test_checkpoint_preview_export_structural_isolation(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2, gan_power=0.1)
    names = [name for _, name in m.model_filename_list]
    assert names == ["encoder.npy", "inter_src.npy", "inter_dst.npy",
                     "decoder.npy", "src_dst_opt.npy", "GAN.npy",
                     "GAN_opt.npy"]
    assert all("mirror" not in name.lower() and "replica" not in name.lower()
               for name in names)
    a8 = synth_samples_nchw(64, batch=1, seed_no=22)
    before = m.AE_view(a8[1], a8[5], 0.25)
    enc_set = next(c for c in m.replica_plan.components
                   if c.canonical is m.encoder)
    with torch.no_grad():
        next(enc_set.mirrors[0].parameters()).add_(10.0)
    after = m.AE_view(a8[1], a8[5], 0.25)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    source = inspect.getsource(type(m).export_dfm)
    assert "replica_plan" not in source and "mirrors" not in source


def test_merge_surface_unchanged_and_teardown(tmp_path, monkeypatch):
    m = build_cpu(tmp_path, monkeypatch, n=2)
    plan = m.replica_plan
    method_source = inspect.getsource(type(m).predictor_func)
    config_source = inspect.getsource(type(m).get_MergerConfig)
    assert "replica" not in method_source and "replica" not in config_source
    m.finalize()
    assert m.replica_plan is None and plan.is_disposed
    with pytest.raises(Exception):
        plan.sync_from_canonical()
    # Canonical inference remains usable after mirror disposal.
    a8 = synth_samples_nchw(64, batch=1, seed_no=23)
    assert len(m.AE_view(a8[1], a8[5], 1.0)) == 5


@requires_gpu
def test_gpu_bf16_simulated_n2_success(tmp_path, monkeypatch):
    m = build_gpu(tmp_path, monkeypatch, _HookedAMP, "bf16")
    m._init_hooks()
    stub_fetch(m, synth_samples_nchw(64, batch=2, seed_no=31))
    m.train_one_iter()
    assert m._mp_scaler is None
    assert m.counts == {"backward": 2, "unscale": 1, "step": 1,
                        "update": 1, "stepped": [True]}
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert_synced(m)


@requires_gpu
def test_gpu_fp16_simulated_n2_success(tmp_path, monkeypatch):
    m = build_gpu(tmp_path, monkeypatch, _HookedAMP, "fp16")
    m._init_hooks()
    stub_fetch(m, synth_samples_nchw(64, batch=2, seed_no=32))
    m.train_one_iter()
    assert isinstance(m._mp_scaler, torch.amp.GradScaler)
    attempts = m.counts["update"]
    assert attempts >= 1
    assert m.counts["backward"] == 2*attempts
    assert m.counts["unscale"] == m.counts["step"] == attempts
    assert m.counts["stepped"][-1] is True
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert_synced(m)


@requires_gpu
def test_gpu_fp16_class_b_same_samples_no_gan_failed_attempt(
        tmp_path, monkeypatch):
    m = build_gpu(tmp_path, monkeypatch, _HookedAMP, "fp16", gan_power=0.1)
    m._init_hooks(); m.inject_class_b = True
    a8 = synth_samples_nchw(64, batch=2, seed_no=33)
    fetches = {"count": 0}

    def fetch():
        fetches["count"] += 1
        return ((a8[0], a8[1], a8[2], a8[3]),
                (a8[4], a8[5], a8[6], a8[7]))
    m.generate_next_samples = fetch
    m.train_one_iter()
    assert fetches["count"] == 1
    assert m.counts["stepped"][0] is False
    assert m.counts["stepped"][-1] is True
    assert all(v is False for v in m.counts["stepped"][:-1])
    assert m.counts["update"] == len(m.counts["stepped"])
    assert m.retry_cleanup is True
    assert m._mp_scaler.get_scale() == (
        FP16_INIT_SCALE * (0.5 ** m.counts["stepped"].count(False)))
    assert int(m.src_dst_opt.iterations.item()) == 1
    assert int(m.GAN_opt.iterations.item()) == 1


@requires_gpu
def test_gpu_fp16_class_a_hard_failure(tmp_path, monkeypatch):
    m = build_gpu(tmp_path, monkeypatch, _HookedAMP, "fp16", gan_power=0.1)
    m._init_hooks()
    a8 = list(synth_samples_nchw(64, batch=2, seed_no=34))
    a8[4] = a8[4].copy(); a8[4][1] += 1.0e6
    stub_fetch(m, tuple(a8))
    with pytest.raises(FloatingPointError):
        m.train_one_iter()
    assert m.counts["backward"] == 1
    assert m.counts["unscale"] == m.counts["step"] == 0
    assert m.counts["update"] == 0
    assert int(m.src_dst_opt.iterations.item()) == 0
    assert int(m.GAN_opt.iterations.item()) == 0
    assert m._mp_scaler.get_scale() == FP16_INIT_SCALE
    for cset in m.replica_plan.components:
        assert all(p.grad is None for p in cset.canonical.parameters())
        for mirror in cset.mirrors:
            assert all(p.grad is None for p in mirror.parameters())
