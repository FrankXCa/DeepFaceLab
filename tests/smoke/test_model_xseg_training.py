"""Phase 10C acceptance for the Torch-only XSeg training model."""
import ast
import builtins
import gc
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import core.leras.models  # noqa: E402,F401
from core.interact import interact as io  # noqa: E402
from core.leras import nn  # noqa: E402
from samplelib import SampleGeneratorBase  # noqa: E402

xseg_module = importlib.import_module("models.Model_XSeg.Model")
XSegModel = xseg_module.XSegModel
RES = 256
BS = 2


@pytest.fixture(autouse=True)
def preserve_rng_and_interaction(monkeypatch):
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    monkeypatch.setattr(io, "input_in_time", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def feeds(fmt, batch=BS):
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32)
    image = np.empty((batch, RES, RES, 3), dtype=np.float32)
    mask = np.empty((batch, RES, RES, 1), dtype=np.float32)
    for i in range(batch):
        image[i, ..., 0] = 0.1 + 0.7 * xx / RES
        image[i, ..., 1] = 0.2 + 0.6 * yy / RES
        image[i, ..., 2] = 0.3 + 0.4 * ((xx + yy + i * 11) % 53) / 53
        mask[i, ..., 0] = (((xx - 128) ** 2 + (yy - 140) ** 2)
                           < (72 + i) ** 2)
    if fmt == "NCHW":
        image = np.transpose(image, (0, 3, 1, 2))
        mask = np.transpose(mask, (0, 3, 1, 2))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class DummyXSegGenerator(SampleGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(False, kwargs.get("batch_size", BS))
        self.fmt = kwargs.get("data_format", nn.data_format)

    def __next__(self):
        return list(feeds(self.fmt, self.batch_size))


class DummyFaceGenerator(SampleGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(False, kwargs.get("batch_size", BS))
        self.fmt = kwargs.get("output_sample_types", [{}])[0].get(
            "data_format", nn.data_format)
        self.outputs = len(kwargs.get("output_sample_types", []))

    def __next__(self):
        image, mask = feeds(self.fmt, self.batch_size)
        return [image, mask] if self.outputs == 2 else [image]


@pytest.fixture
def dummy_generators(monkeypatch):
    monkeypatch.setattr(xseg_module, "SampleGeneratorFaceXSeg",
                        DummyXSegGenerator)
    monkeypatch.setattr(xseg_module, "SampleGeneratorFace",
                        DummyFaceGenerator)


class HeadlessXSeg(XSegModel):
    seed_pretrain = False

    def on_initialize_options(self):
        self.options.update({"face_type": "f",
                             "pretrain": bool(self.seed_pretrain),
                             "batch_size": BS})
        self.batch_size = BS
        self.pretrain_just_disabled = False


class DisablePretrainXSeg(XSegModel):
    def on_initialize_options(self):
        was_pretrain = self.load_or_def_option("pretrain", False)
        self.options["face_type"] = self.load_or_def_option("face_type", "f")
        self.options["pretrain"] = False
        self.options["batch_size"] = self.load_or_def_option("batch_size", BS)
        self.batch_size = self.options["batch_size"]
        self.pretrain_just_disabled = bool(was_pretrain)


# ModelBase derives the model family from the defining module's parent
# directory.  Keep these option-only test subclasses in the production XSeg
# family without copying production lifecycle code into a test package.
HeadlessXSeg.__module__ = xseg_module.__name__
DisablePretrainXSeg.__module__ = xseg_module.__name__


def build(cls, root, pretrain_path=None, gpu=False):
    if gpu:
        nn.initialize_main_env()
    return cls(is_training=True, saved_models_path=root,
               training_data_src_path=Path(root) / "src",
               training_data_dst_path=Path(root) / "dst",
               pretraining_data_path=pretrain_path,
               cpu_only=not gpu, force_gpu_idxs=[0] if gpu else None,
               debug=False)


def snap(model):
    weights = [p.detach().cpu().numpy().copy()
               for p in model.model.get_weights()]
    opt = [p.detach().cpu().numpy().copy()
           for p in model.model.opt.get_weights()]
    return weights, opt


def exact_lists(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y)
                                    for x, y in zip(a, b))


def expected_loss(model, image, target):
    x = torch.from_numpy(image).to(nn.device, dtype=nn.floatx)
    t = torch.from_numpy(target).to(nn.device, dtype=nn.floatx)
    with torch.no_grad(), xseg_module._xseg_training_precision(x):
        logits, pred = model.model.flow(x, pretrain=model.pretrain)
        if model.pretrain:
            return (5 * nn.dssim(t, pred, max_val=1.0,
                                  filter_size=int(RES / 11.6)).mean(dim=1)
                    + 5 * nn.dssim(t, pred, max_val=1.0,
                                   filter_size=int(RES / 23.2)).mean(dim=1)
                    + 10 * torch.square(t - pred).mean(dim=(1, 2, 3))) \
                .cpu().numpy()
        return nn.sigmoid_cross_entropy(t, logits).cpu().numpy()


def test_xseg_model_cpu_lifecycle_transition_and_preview(
        plain_tmp, dummy_generators):
    root = Path(plain_tmp) / "xseg_training"
    root.mkdir()
    pretrain_path = root / "pretrain"
    HeadlessXSeg.seed_pretrain = True
    model = build(HeadlessXSeg, root, pretrain_path)
    assert model.model_data_format == "NHWC" and model.pretrain
    image, target = feeds("NHWC")
    expected = expected_loss(model, image, target)
    before, _ = snap(model)
    loss = model.train(image, target)
    np.testing.assert_allclose(loss, expected, rtol=0, atol=2e-6)
    after, _ = snap(model)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    assert int(model.model.opt.iterations.item()) == 1

    previews = model.onGetPreview([(image, target)])
    assert [name for name, _ in previews] == ["XSeg training faces"]
    assert previews[0][1].shape == (BS * RES, 2 * RES, 3)
    assert np.isfinite(previews[0][1]).all()

    model.set_iter(1)
    model.save()
    saved_w, saved_o = snap(model)
    model.finalize()
    del model
    gc.collect()

    resumed = build(XSegModel, root, pretrain_path)
    rw, ro = snap(resumed)
    assert resumed.get_iter() == 1 and resumed.pretrain
    assert exact_lists(saved_w, rw) and exact_lists(saved_o, ro)
    resumed.finalize()
    del resumed
    gc.collect()

    transitioned = build(DisablePretrainXSeg, root, pretrain_path)
    tw, to = snap(transitioned)
    assert transitioned.pretrain_just_disabled
    assert transitioned.get_iter() == 0 and not transitioned.pretrain
    assert exact_lists(saved_w, tw) and exact_lists(saved_o, to)

    normal_expected = expected_loss(transitioned, image, target)
    normal_loss = transitioned.train(image, target)
    np.testing.assert_allclose(normal_loss, normal_expected, rtol=0, atol=2e-6)
    assert int(transitioned.model.opt.iterations.item()) == 2
    normal_previews = transitioned.onGetPreview(
        [(image, target), (image,), (image,)])
    assert [name for name, _ in normal_previews] == [
        "XSeg training faces", "XSeg src faces", "XSeg dst faces"]
    assert all(frame.shape == (BS * RES, 3 * RES, 3)
               for _name, frame in normal_previews)

    iteration, _elapsed = transitioned.train_one_iter()
    assert iteration == 1
    transitioned.save()
    final_w, final_o = snap(transitioned)
    transitioned.finalize()
    del transitioned
    gc.collect()

    final = build(XSegModel, root, pretrain_path)
    fw, fo = snap(final)
    assert final.get_iter() == 1 and not final.pretrain
    assert exact_lists(final_w, fw) and exact_lists(final_o, fo)
    final.finalize()


def test_xseg_model_batch_sum_backward(
        plain_tmp, dummy_generators, monkeypatch):
    root = Path(plain_tmp) / "batch_sum"
    root.mkdir()
    HeadlessXSeg.seed_pretrain = False
    model = build(HeadlessXSeg, root)
    image, target = feeds("NHWC", batch=1)
    image = np.repeat(image, 2, axis=0)
    target = np.repeat(target, 2, axis=0)
    captured = {}
    original_backward = torch.autograd.backward

    def capture_backward(tensors, grad_tensors=None, *args, **kwargs):
        captured["loss"] = tensors.detach().cpu().clone()
        captured["upstream"] = grad_tensors.detach().cpu().clone()
        return original_backward(tensors, grad_tensors, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", capture_backward)
    before, _ = snap(model)
    loss = model.train(image, target)
    after, _ = snap(model)

    assert loss.shape == (2,)
    np.testing.assert_allclose(loss[0], loss[1], rtol=0, atol=1e-7)
    assert captured["loss"].shape == (2,)
    torch.testing.assert_close(captured["upstream"],
                               torch.ones(2), rtol=0, atol=0)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    assert int(model.model.opt.iterations.item()) == 1
    model.finalize()


def test_xseg_training_has_no_tensorflow_import():
    tree = ast.parse(Path(xseg_module.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "tensorflow" or name.startswith("tensorflow.")
                   for name in imports)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_xseg_model_cuda_step_and_precision_restoration(
        plain_tmp, dummy_generators):
    root = Path(plain_tmp) / "cuda_xseg_training"
    root.mkdir()
    HeadlessXSeg.seed_pretrain = False
    model = build(HeadlessXSeg, root, gpu=True)
    assert model.model_data_format == "NCHW" and nn.device.type == "cuda"
    image, target = feeds("NCHW")
    before, _ = snap(model)
    cudnn_before = torch.backends.cudnn.allow_tf32
    matmul_before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        loss = model.train(image, target)
        assert np.isfinite(loss).all()
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 == matmul_before
        after, _ = snap(model)
        assert any(not np.array_equal(a, b) for a, b in zip(before, after))

        marker = torch.zeros(1, device="cuda")
        with pytest.raises(RuntimeError, match="restore probe"):
            with xseg_module._xseg_training_precision(marker):
                assert torch.backends.cudnn.allow_tf32 is False
                raise RuntimeError("restore probe")
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_before
        model.finalize()
