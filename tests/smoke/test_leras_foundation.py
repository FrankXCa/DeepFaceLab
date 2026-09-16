"""Phase 3A smoke tests: torch-only leras foundation contracts.

Covers the required Phase 3A validation areas:
  1. foundation imports without TensorFlow (3A scope)
  2. LayerBase / module construction (official two-phase lifecycle)
  3. deterministic parameter / official-name enumeration
  4. stable naming across repeated construction
  5. device placement through the Phase 2 abstraction
  6. CPU construction
  7. CUDA construction on the RTX 4090 (skip-if-CPU-only)
  8. save/load foundation behavior for a minimal synthetic module
  9. expected failure behavior for invalid / missing state (strict)
  10. no direct CUDA dependency in foundation sources
  11. no TensorFlow import in the tested 3A path (after a full cycle)

The synthetic modules below are test-only constructs standing in for
the concrete layers that Phase 3B rebuilds; they use the official
parameter names (weight/bias) and the official two-phase
build_weights/init_weights lifecycle exactly as production layers will.
"""

import os
import pickle
import sys

import pytest
import torch

from core.leras import nn as dfl_nn
from core.leras import checkpoint as ckpt
from core.leras import device as device_layer
from core.leras.layers import LayerBase, Saveable
from core.leras.initializers import initializers

CUDA_AVAILABLE = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# synthetic stand-ins for the production layers (test-only, Phase 3B scope)
# ---------------------------------------------------------------------------

class _SynthLayer(LayerBase):
    """Minimal layer: a 2-D 'weight' (3 x 2) and a 'bias' (2,), created
    in build_weights with the official parameter names."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        w = torch.empty((3, 2), device=dfl_nn.device, dtype=dfl_nn.floatx)
        b = torch.empty((2,), device=dfl_nn.device, dtype=dfl_nn.floatx)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(b)
        self.register_param_initializer("weight", initializers.random_normal)
        self.register_param_initializer("bias", initializers.zeros)

    def forward(self, x):
        return x @ self.weight + self.bias


class _SynthModel(torch.nn.Module, Saveable):
    """Module tree: two named sub-layers, the Phase 3B archis shape."""

    def __init__(self, name=None):
        super().__init__()
        Saveable.__init__(self, name)
        self.layer1 = _SynthLayer(name="conv1_1")
        self.layer2 = _SynthLayer(name="conv1_2")
        self.layer1.build_weights()
        self.layer2.build_weights()

    def build_weights(self):
        pass

    def init_weights(self):
        # Composite modules cascade initialization to their children —
        # exactly like the official archis do (the foundation's
        # init_weights applies only a module's DIRECT parameters).
        self.layer1.init_weights()
        self.layer2.init_weights()

    def forward(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _cpu_foundation():
    """Every test starts from the CPU foundation state (deterministic);
    tests needing the GPU re-initialize inside their body."""
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU())
    yield


# ---------------------------------------------------------------------------
# 1. / 11. TensorFlow independence of the 3A path
# ---------------------------------------------------------------------------

def test_foundation_imports_without_tensorflow():
    import core.leras.nn  # noqa: F401
    import core.leras.device  # noqa: F401
    import core.leras.backends  # noqa: F401
    import core.leras.layers  # noqa: F401
    import core.leras.initializers  # noqa: F401
    import core.leras.checkpoint  # noqa: F401

    assert "tensorflow" not in sys.modules

    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU())
    assert "tensorflow" not in sys.modules


def test_no_tensorflow_after_full_cycle(plain_tmp):
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU())
    model = _SynthModel(name="encoder")
    model.init_weights()
    path = os.path.join(plain_tmp, "encoder.npy")
    model.save_weights(path)

    fresh = _SynthModel(name="encoder")
    fresh.load_weights(path)

    assert "tensorflow" not in sys.modules


# ---------------------------------------------------------------------------
# 2. LayerBase / module construction (official two-phase lifecycle)
# ---------------------------------------------------------------------------

def test_layerbase_construction():
    layer = _SynthLayer(name="conv1_1")
    # phase 1: config only, no weights yet
    assert len(list(layer.parameters())) == 0

    # phase 2: weights created by build_weights on nn.device/nn.floatx
    layer.build_weights()
    params = list(layer.parameters())
    assert [p for p in params]  # two parameters
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert isinstance(layer.bias, torch.nn.Parameter)
    assert layer.weight.device == dfl_nn.device
    assert layer.weight.dtype == dfl_nn.floatx
    assert layer.bias.device == dfl_nn.device

    # official get_weights contract: parameters in creation order
    weights = layer.get_weights()
    assert [p is layer.weight for p in weights] or len(weights) == 2
    assert weights[0] is layer.weight
    assert weights[1] is layer.bias

    # torch module forward contract
    x = torch.zeros((4, 3), device=dfl_nn.device, dtype=dfl_nn.floatx)
    y = layer(x)
    assert y.shape == (4, 2)

    # official aliases on the nn class (call-site compatibility)
    assert dfl_nn.LayerBase is LayerBase
    assert dfl_nn.Saveable is Saveable
    assert dfl_nn.initializers is initializers


def test_initialization_lifecycle():
    layer = _SynthLayer(name="conv1_1")
    layer.build_weights()
    assert layer.get_param_initializers().keys() == {"weight", "bias"}

    pre = layer.weight.detach().clone()
    layer.init_weights()
    # registered initializers were applied (values changed from empty())
    assert not torch.equal(layer.weight.detach(), pre)
    # zeros initializer for bias
    assert torch.all(layer.bias == 0)


# ---------------------------------------------------------------------------
# 3./4. deterministic enumeration + stable naming
# ---------------------------------------------------------------------------

def _model_keys(model):
    return [key for key, _ in model._iter_official_weights()]


def test_deterministic_parameter_enumeration():
    model = _SynthModel(name="encoder")
    keys = _model_keys(model)
    # registration order: parameters before buffers, layer1 before layer2
    assert keys == [
        "layer1/weight:0",
        "layer1/bias:0",
        "layer2/weight:0",
        "layer2/bias:0",
    ]
    # repeated enumeration of the same module is identical
    assert _model_keys(model) == keys
    # torch dotted <-> official mapping is lossless
    for key in keys:
        assert ckpt.official_name(ckpt.torch_name_from_official(key)) == key


def test_stable_naming_across_repeated_construction():
    keys_a = _model_keys(_SynthModel(name="encoder"))
    keys_b = _model_keys(_SynthModel(name="encoder"))
    assert keys_a == keys_b
    assert len(set(keys_a)) == len(keys_a)


# ---------------------------------------------------------------------------
# 5./6./7. device placement through the Phase 2 abstraction
# ---------------------------------------------------------------------------

def test_cpu_construction_and_placement():
    # CPU foundation (the autouse fixture); also a REAL CPU-only run in
    # the .venv-cpu environment.
    assert dfl_nn.device.type == "cpu"
    assert dfl_nn.device == device_layer.get_torch_device(None)

    model = _SynthModel(name="encoder")
    for param in model.parameters():
        assert param.device == dfl_nn.device
        assert param.device.type == "cpu"

    x = torch.zeros((2, 3), device=dfl_nn.device, dtype=dfl_nn.floatx)
    y = model.layer1(x)  # forward on CPU through the abstraction
    assert y.device == dfl_nn.device


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available in this environment")
def test_cuda_construction_on_rtx4090():
    # device placement resolved through the Phase 2 device abstraction:
    # no torch.cuda.* call appears in this test body
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU())
    assert dfl_nn.device.type == "cuda"

    best = dfl_nn.getCurrentDeviceConfig().devices[0]
    assert dfl_nn.device == device_layer.get_torch_device(best)
    assert dfl_nn.device.index == best.index

    model = _SynthModel(name="encoder")
    for param in model.parameters():
        assert param.device == dfl_nn.device

    x = torch.zeros((2, 3), device=dfl_nn.device, dtype=dfl_nn.floatx)
    y = model.layer1(x)
    device_layer.synchronize(dfl_nn.device)
    assert y.device == dfl_nn.device


# ---------------------------------------------------------------------------
# 8. save/load foundation for a minimal synthetic module
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(plain_tmp):
    dfl_nn.initialize(dfl_nn.DeviceConfig.CPU())
    model = _SynthModel(name="encoder")
    model.init_weights()

    path = os.path.join(plain_tmp, "encoder.npy")
    model.save_weights(path)
    assert os.path.exists(path)

    # official format: pickled dict[str, np.ndarray] with ':0' keys
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    assert set(d.keys()) == {
        "layer1/weight:0", "layer1/bias:0",
        "layer2/weight:0", "layer2/bias:0",
    }
    import numpy as np
    assert all(isinstance(v, np.ndarray) for v in d.values())
    assert d["layer1/weight:0"].shape == (3, 2)

    # load into a fresh module (uninitialized values must be replaced)
    fresh = _SynthModel(name="encoder")
    assert fresh.load_weights(path) is True
    for sub, param in model._iter_official_weights():
        fresh_sub = {k: p for k, p in fresh._iter_official_weights()}[sub]
        assert torch.equal(param.detach().cpu(), fresh_sub.detach().cpu())


# ---------------------------------------------------------------------------
# 9. expected failure behavior (strict, all-or-nothing)
# ---------------------------------------------------------------------------

def _write(path, d):
    with open(path, "wb") as fh:
        fh.write(pickle.dumps(d, 4))


def test_strict_load_missing_key_fails(plain_tmp):
    model = _SynthModel(name="encoder")
    model.init_weights()
    d = {k: v.detach().cpu().numpy() for k, v in model._iter_official_weights()}
    del d["layer2/weight:0"]
    path = os.path.join(plain_tmp, "bad_missing.npy")
    _write(path, d)

    fresh = _SynthModel(name="encoder")
    with pytest.raises(ckpt.CheckpointLoadError):
        fresh.load_weights(path)
    # all-or-nothing: nothing was copied (values still from empty())
    assert not torch.equal(fresh.layer2.weight, model.layer2.weight)


def test_strict_load_extra_key_fails(plain_tmp):
    model = _SynthModel(name="encoder")
    model.init_weights()
    d = {k: v.detach().cpu().numpy() for k, v in model._iter_official_weights()}
    d["rogue/weight:0"] = torch.zeros((1,)).numpy()
    path = os.path.join(plain_tmp, "bad_extra.npy")
    _write(path, d)

    fresh = _SynthModel(name="encoder")
    with pytest.raises(ckpt.CheckpointLoadError):
        fresh.load_weights(path)


def test_strict_load_shape_mismatch_fails(plain_tmp):
    model = _SynthModel(name="encoder")
    model.init_weights()
    d = {k: v.detach().cpu().numpy() for k, v in model._iter_official_weights()}
    d["layer1/weight:0"] = torch.zeros((4, 2, 2, 2)).numpy()
    path = os.path.join(plain_tmp, "bad_shape.npy")
    _write(path, d)

    fresh = _SynthModel(name="encoder")
    with pytest.raises(ckpt.CheckpointLoadError):
        fresh.load_weights(path)


def test_load_absent_file_returns_false(plain_tmp):
    model = _SynthModel(name="encoder")
    assert model.load_weights(os.path.join(plain_tmp, "absent.npy")) is False


def test_set_weights_strict():
    layer = _SynthLayer(name="conv1_1")
    layer.build_weights()
    layer.init_weights()

    # matching shapes copy (tensor and ndarray sources)
    layer.set_weights([layer.weight.detach().clone(), layer.bias.detach().cpu().numpy()])

    # shape mismatch raises (official greedy reshape removed)
    with pytest.raises(ValueError):
        layer.set_weights([torch.zeros((4, 2), device=dfl_nn.device), layer.bias.detach()])

    # list length mismatch raises (official contract)
    with pytest.raises(ValueError):
        layer.set_weights([torch.zeros_like(layer.weight)])


def test_save_requires_name():
    anonymous = _SynthModel()
    anonymous.name = None
    with pytest.raises(Exception):
        anonymous.save_weights(os.path.join(os.environ.get("TEMP", "."), "x.npy"))


# ---------------------------------------------------------------------------
# 10. no direct CUDA dependency in foundation sources
# ---------------------------------------------------------------------------

def test_no_direct_cuda_in_foundation_sources():
    """AST-level check: no ``torch.cuda.*`` attribute access and no
    ``'cuda:...'`` device string literal in the 3A foundation sources
    (docstring prose is ignored; only code counts)."""
    import ast

    import core.leras
    root = os.path.dirname(core.leras.__file__)
    files = [
        os.path.join(root, "nn.py"),
        os.path.join(root, "checkpoint.py"),
        os.path.join(root, "layers", "Saveable.py"),
        os.path.join(root, "layers", "LayerBase.py"),
        os.path.join(root, "initializers", "__init__.py"),
    ]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # resolve the 'torch.cuda...' attribute chain
                target, parts = node, []
                while isinstance(target, ast.Attribute):
                    parts.append(target.attr)
                    target = target.value
                if isinstance(target, ast.Name) and target.id == "torch" and "cuda" in parts:
                    assert False, \
                        f"{f}: direct CUDA API call at line {node.lineno}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith("cuda:"), \
                    f"{f}: hardcoded CUDA device string at line {node.lineno}"
