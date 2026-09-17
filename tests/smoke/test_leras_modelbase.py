"""Phase 5 — torch leras model container (``nn.ModelBase``) tests.

Covers the torch replacement of the official ``core/leras/models/
ModelBase.py`` container (the official TF source is preserved verbatim
in ``core/leras/models/ModelBase_tf.py``; see ``tests/smoke/README.md``,
Phase 5 section):

- deterministic component registration: the official attribute-discovery
  build loop (``xor_list`` over ``vars(self)``, generator ``on_build``
  support) assigns sub-layers/models by attribute name — registration
  order and ``layers_by_name`` are insertion-order deterministic (no
  object ids, no hash order); nested ``ModelBase`` containers register
  and flatten through ``get_layers()``;
- ``get_weights()`` concatenates the registered components' weights in
  registration order (the official order — the checkpoint key order the
  Phase 4 engine consumes);
- ``__call__`` auto-builds and runs ``forward`` (torch eager);
- ``build_for_run``/``run``: the official inference API with the TF
  session concepts removed — recorded input contract instead of
  placeholders, ``torch.no_grad()`` forward instead of
  ``tf_sess.run(feed_dict=...)``, NumPy in (torch tensors or arrays,
  converted to ``nn.device``/the declared dtype) and NumPy out (the
  official caller contract); explicit failures preserved (not built ->
  the official exception message; input-count mismatch -> the official
  ``ValueError``);
- the container is a Phase 4 Saveable: ``save_weights``/``load_weights``
  round-trip the official raw pickle protocol-4 stream (leading bytes
  pinned) through the strict engine, all-or-nothing;
- GPU: the same ``run()`` inference path on the RTX 4090 (skip on
  CPU-only environments);
- AST hygiene: no TensorFlow import and no direct ``torch.cuda.*`` /
  ``'cuda:'`` literals in the foundation sources.

Parity: EXACT (deterministic constant initializers, zero RNG; values
compared with torch.equal / np.array_equal; the GPU test is the same
no-grad forward on the CUDA device).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from core.leras import nn as dfl_nn
import core.leras.models  # noqa: F401  (binds nn.ModelBase before the class definitions below)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _init_cpu(data_format="NHWC"):
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", data_format)


# --- test vehicles (small, deterministic: constant initializers, no RNG) ---

class TwoLayerNet(dfl_nn.ModelBase):
    """Conv2D(3->4, 3x3, SAME) -> Dense(4->2); kernel zeros + bias ones
    (unique per component so the weight order is observable)."""

    def on_build(self):
        self.conv1 = dfl_nn.Conv2D(
            3, 4, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.ones,
            name="conv1")
        self.dense = dfl_nn.Dense(
            4, 2, use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.ones,
            name="dense")

    def forward(self, x):
        return self.dense(self.conv1(x))


class NamelessSubNet(dfl_nn.ModelBase):
    """Sub-layers created WITHOUT explicit names: the official build
    assigns the attribute name (the container's scoping contract)."""

    def on_build(self):
        self.first = dfl_nn.Conv2D(
            3, 4, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros)
        self.second = dfl_nn.Dense(
            4, 2, use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros)

    def forward(self, x):
        return self.second(self.first(x))


class NestedSubNet(dfl_nn.ModelBase):
    """4-channel sub-net: Conv2D(4->4) -> Dense(4->2)."""

    def on_build(self):
        self.first = dfl_nn.Conv2D(
            4, 4, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros)
        self.second = dfl_nn.Dense(
            4, 2, use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros)

    def forward(self, x):
        return self.second(self.first(x))


class NestedContainer(dfl_nn.ModelBase):
    """A layer plus a nested ModelBase container (official: containers
    may compose containers; get_layers flattens down to LayerBase)."""

    def on_build(self):
        self.head = dfl_nn.Conv2D(
            3, 4, kernel_size=3, padding="SAME", use_bias=False,
            kernel_initializer=dfl_nn.initializers.zeros)
        self.tail = NestedSubNet()

    def forward(self, x):
        return self.tail(self.head(x))


# --- registration ------------------------------------------------------

def test_registration_order_and_names():
    _init_cpu()
    net = TwoLayerNet(name="net")
    out = net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    assert tuple(out.shape) == (1, 8, 8, 2)
    assert [l.name for l in net.layers] == ["conv1", "dense"]
    assert set(net.layers_by_name) == {"conv1", "dense"}
    assert net.get_layer_by_name("conv1") is net.conv1
    assert net.get_layer_by_name("missing") is None
    assert net.built is True


def test_nameless_layers_get_attribute_names():
    _init_cpu()
    net = NamelessSubNet(name="unnamed")
    net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    assert [l.name for l in net.layers] == ["first", "second"]
    assert net.get_layer_by_name("first") is net.first


def test_nested_container_registration_and_flattening():
    _init_cpu()
    model = NestedContainer(name="nest")
    model(torch.zeros(1, 8, 8, 3, dtype=torch.float32))

    # registration: the direct components, nested container included
    assert [l.name for l in model.layers] == ["head", "tail"]

    # flattening: only LayerBase components, nested ones expanded
    layers = model.get_layers()
    assert [type(l).__name__ for l in layers] == [
        "Conv2D", "Conv2D", "Dense"]
    assert [l.name for l in layers] == ["head", "first", "second"]


def test_generator_on_build_passes():
    _init_cpu()

    class Yielding(dfl_nn.ModelBase):
        def on_build(self):
            self.a = dfl_nn.Conv2D(
                3, 4, kernel_size=3, padding="SAME", use_bias=False,
                kernel_initializer=dfl_nn.initializers.zeros)
            yield
            self.b = dfl_nn.Conv2D(
                4, 2, kernel_size=3, padding="SAME", use_bias=False,
                kernel_initializer=dfl_nn.initializers.zeros)

        def forward(self, x):
            return self.b(self.a(x))

    m = Yielding(name="gen")
    out = m(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    assert tuple(out.shape) == (1, 8, 8, 2)
    # both build passes registered their layers, in order
    assert [l.name for l in m.layers] == ["a", "b"]


# --- weights / forward ---------------------------------------------------

def test_get_weights_registration_order_and_values():
    _init_cpu()
    net = TwoLayerNet(name="w")
    net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))

    weights = net.get_weights()
    # official order: conv1 kernel, conv1 bias, dense kernel, dense bias
    assert [tuple(w.shape) for w in weights] == [
        (4, 3, 3, 3), (4,), (4, 2), (2,)]
    # zero kernels, ones biases (the initializers above)
    assert torch.equal(weights[0], torch.zeros_like(weights[0]))
    assert torch.equal(weights[1], torch.ones_like(weights[1]))
    assert torch.equal(weights[2], torch.zeros_like(weights[2]))
    assert torch.equal(weights[3], torch.ones_like(weights[3]))


def test_call_autobuilds():
    _init_cpu()
    net = TwoLayerNet(name="lazy")
    assert net.built is False
    out = net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    assert net.built is True
    assert tuple(out.shape) == (1, 8, 8, 2)
    # determinism: constant initializers, no RNG anywhere
    out2 = net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    assert torch.equal(out, out2)


def test_forward_is_grad_capable():
    _init_cpu()
    _init_cpu()
    net = TwoLayerNet(name="grad")
    x = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
    x.requires_grad_(True)
    out = net(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # every trainable parameter got a gradient (loss touches them all)
    for p in net.parameters():
        assert p.grad is not None


# --- build_for_run / run (the TF session concepts removed) ---------------

def test_build_for_run_records_contract_and_builds():
    _init_cpu()
    net = TwoLayerNet(name="run")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])
    assert net.built is True
    assert net.run_placeholders == [(torch.float32, (1, 8, 8, 3))]

    with pytest.raises(ValueError, match="shapes_list must be a list"):
        net.build_for_run((torch.float32, (1, 8, 8, 3)))


def test_run_numpy_in_numpy_out_matches_torch_forward():
    _init_cpu()
    net = TwoLayerNet(name="run")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])

    x = np.random.RandomState(7).rand(1, 8, 8, 3).astype(np.float32)
    result = net.run([x])

    # official caller contract: NumPy out (TF session outputs were NumPy)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 8, 8, 2)
    assert result.dtype == np.float32

    # exact agreement with the same forward run as a torch tensor
    ref = net(torch.from_numpy(x)).detach().cpu().numpy()
    assert np.array_equal(result, ref)


def test_run_no_grad_boundary():
    _init_cpu()
    net = TwoLayerNet(name="no_grad")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])
    x = np.random.RandomState(11).rand(1, 8, 8, 3).astype(np.float32)
    net.run([x])
    # the inference path created no gradient bookkeeping
    for p in net.parameters():
        assert p.grad is None


def test_run_torch_tensor_input():
    _init_cpu()
    net = TwoLayerNet(name="t_in")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])
    x = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
    result = net.run([x])
    assert isinstance(result, np.ndarray)
    ref = net(x).detach().cpu().numpy()
    assert np.array_equal(result, ref)


def test_run_explicit_failures():
    _init_cpu()
    x = np.zeros((1, 8, 8, 3), dtype=np.float32)

    unbuilt = TwoLayerNet(name="never_built")
    with pytest.raises(Exception, match="Model didn't build for run."):
        unbuilt.run([x])

    net = TwoLayerNet(name="fail")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])
    with pytest.raises(ValueError, match="len.inputs"):
        net.run([x, x])
    with pytest.raises(TypeError, match="torch.Tensor or a numpy array"):
        net.run([3.14])


@requires_gpu
def test_run_gpu_inference_path():
    dfl_nn.initialize_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NHWC")
    assert dfl_nn.device.type == "cuda"

    net = TwoLayerNet(name="gpu_run")
    net.build_for_run([(torch.float32, (1, 8, 8, 3))])
    x = np.random.RandomState(3).rand(1, 8, 8, 3).astype(np.float32)
    result = net.run([x])  # CPU NumPy input is placed on nn.device

    assert isinstance(result, np.ndarray)
    ref = net(torch.from_numpy(x).to(dfl_nn.device)).detach().cpu().numpy()
    assert np.array_equal(result, ref)
    # parameters live on the training device
    for p in net.parameters():
        assert p.device.type == "cuda"


# --- Phase 4 Saveable integration ----------------------------------------

def test_saveable_roundtrip_official_format(plain_tmp):
    _init_cpu()
    net = TwoLayerNet(name="saved_net")
    net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    path = str(Path(plain_tmp) / "saved_net.npy")
    net.save_weights(path)

    # the file is the official raw pickle protocol-4 stream (Phase 4
    # contract — the .npy extension is a misnomer, pinned here too)
    raw = Path(path).read_bytes()
    assert raw[:3] == b"\x80\x04\x95"
    assert b"\x93NUMPY" not in raw[:64]

    twin = TwoLayerNet(name="saved_net")
    # official flow: the model is built (weights created) before the
    # checkpoint load; an unbuilt target has no parameters to load into
    twin.build()
    twin.load_weights(path)
    for a, b in zip(net.get_weights(), twin.get_weights()):
        assert torch.equal(a.cpu(), b.cpu())


def test_load_missing_file_returns_false(plain_tmp):
    _init_cpu()
    net = TwoLayerNet(name="gone")
    # official Saveable contract: a missing file returns False (the model
    # code decides the policy — first-run fresh init vs resume failure);
    # a file that EXISTS but cannot be loaded strictly raises
    assert net.load_weights(str(Path(plain_tmp) / "does_not_exist.npy")) is False
    assert net.built is False


def test_load_corrupt_file_fails_explicitly(plain_tmp):
    _init_cpu()
    net = TwoLayerNet(name="corrupt")
    net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    path = Path(plain_tmp) / "corrupt.npy"
    # a truncated pickle: exists, but cannot be parsed
    net.save_weights(str(path))
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    weights_before = [w.detach().clone() for w in net.get_weights()]
    with pytest.raises(Exception):
        net.load_weights(str(path))
    # all-or-nothing: nothing was copied on failure
    for before, after in zip(weights_before, net.get_weights()):
        assert torch.equal(before, after)


def test_summary_table():
    _init_cpu()
    net = TwoLayerNet(name="summarized")
    net(torch.zeros(1, 8, 8, 3, dtype=torch.float32))
    net.summary()  # official text table; must not raise


# --- hygiene ---------------------------------------------------------------

def test_no_tf_or_cuda_in_foundation_sources():
    files = [
        REPO_ROOT / "core" / "leras" / "models" / "ModelBase.py",
        REPO_ROOT / "core" / "leras" / "models" / "__init__.py",
        REPO_ROOT / "models" / "ModelBase.py",
        REPO_ROOT / "models" / "__init__.py",
    ]
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("tensorflow"), (f, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").startswith("tensorflow") is False, (
                    f, node.module)
            elif isinstance(node, ast.Attribute):
                src = ast.unparse(node)
                assert not src.startswith("torch.cuda"), (f, src)
        text = f.read_text(encoding="utf-8")
        assert '"cuda:"' not in text and "'cuda:'" not in text, f


def test_nn_modelbase_bound_on_nn_class():
    _init_cpu()
    import importlib
    module = importlib.import_module("core.leras.models.ModelBase")
    # the class bound on the nn class IS the torch module's class
    assert dfl_nn.ModelBase is module.ModelBase
    assert issubclass(dfl_nn.ModelBase, dfl_nn.Saveable)
