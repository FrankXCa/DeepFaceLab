"""Phase 10B acceptance: XSeg checkpoint strict load + XSegNet lifecycle.

Covers the Phase 10B XSeg model/wrapper acceptance on the torch
foundation (architecture ground truth: official baseline ``e4b7543``).
The frozen TensorFlow parity artifacts are private and are not read here,
so this file runs standalone on the tracked tree:

- weight inventory: the official 222-variable layout
  (per conv block ``conv/weight:0`` + ``conv/bias:0`` + ``frn/weight:0``
  + ``frn/bias:0`` + ``frn/eps:0`` + ``tlu/tau:0``; ``dense1`` /
  ``dense2`` weights + bias; BlurPool has no variables) and the
  17,569,317-parameter total, both data formats;
- strict checkpoint round-trip (Phase 4 engine, ``Saveable.load_weights``
  / ``save_weights`` — official raw pickle protocol-4 streams in the
  official on-disk layouts): deterministic unique-value weight fill
  (layout proofs — any wrong axis order is visible), save, fresh-model
  strict load, re-save, value-EXACT dict equality;
- cross-format strict load: the official on-disk layout is
  data-format-INDEPENDENT (conv kernels (kH,kW,in,out) in NHWC and NCHW
  alike; 1-D/2-D weights format-independent) — an NHWC-saved file loads
  into a fresh NCHW model and vice versa, value-exact (this is the real
  property: official files travel between data formats; the Phase 10B
  scratch-driver layout bug that assumed OIHW-on-disk for NCHW is the
  exact failure this must not reproduce);
- strict negatives (all-or-nothing pass 1): missing key, extra key,
  same-element-count shape mismatch (the exact failure mode of the
  Phase 10B scratch-driver layout bug), dtype mismatch ->
  ``CheckpointLoadError`` and ZERO mutation (every parameter
  bit-identical to the pre-attempt snapshot);
- ``facelib.XSegNet`` (torch port of the official wrapper,
  ``XSegNet_tf.py`` preserved verbatim) lifecycle on ``plain_tmp``
  weight roots (NO private paths): first-run initialization,
  ``save_weights`` file contract (training mode: ``{name}_{res}_opt.npy``
  + ``{name}_{res}.npy``; inference mode: model file only), strict
  resume with value-exact weights/optimizer state, the one-step
  RMSprop resume equivalence (the official
  ``RMSprop(lr=0.0001, lr_dropout=0.3)`` state — ``iters`` +
  ``acc_*`` — survives save/strict-load exactly), the ``extract``
  contract (3-D single image / 4-D batch ONLY — 2-D inputs fail;
  clip to [0,1]; the official noise gate ``result[result < 0.1] = 0``;
  NumPy in, NumPy out), and the Phase 10B strict policy: a missing
  REQUIRED file on a resume fails explicitly with
  ``FileNotFoundError`` (documented deviation from the official silent
  re-initialization / 0.5-ones inference fallback — v2 section 19,
  Phase 6 SAEHD precedent; pinned here as the port's contract);
- CPU only (the smoke default environment); the explicit
  ``DeviceConfig([])`` initialization before every construction pins
  the Phase 2 device abstraction to CPU (a bare ``nn.initialize``
  without a prior session would fall back to ``BestGPU``).

Parity labels: EXACT (value/layout round-trips — ``np.array_equal`` /
``torch.equal``; deterministic fills + single-thread CPU per
``conftest.py``). No tolerance: any diff is a defect.
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras import checkpoint as dfl_ckpt  # noqa: E402

FMTS = ["NHWC", "NCHW"]
RES = 256


@pytest.fixture(autouse=True)
def restore_rng_state():
    """Keep XSeg initialization from changing later smoke-test streams."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def init_fmt(fmt):
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", fmt)


def build_xseg(fmt):
    init_fmt(fmt)
    m = dfl_nn.XSeg(3, 32, 1, name="XSeg")
    m.build()
    return m


def fill_weights(model, seed=1234):
    """Deterministic unique-value fill of every parameter (layout
    proofs, the test_checkpoint_conversion convention): value =
    (flat_position % 1000)/1000 - 0.5, so any axis reorder of the
    stored pattern is visible after a save/load round-trip."""
    for p in model.get_weights():
        n = p.numel()
        t = (torch.arange(n, dtype=torch.float32) % 1000) / 1000.0 - 0.5
        with torch.no_grad():
            p.copy_(t.reshape(p.shape))


def save_dict(path):
    """A model's save_weights file as an in-memory official-layout
    dict (the raw protocol-4 pickle container, parsed back)."""
    d = pickle.loads(Path(path).read_bytes())
    assert isinstance(d, dict)
    return d


def dicts_equal(a, b):
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def to_fmt(arr, fmt):
    if fmt == "NCHW":
        if arr.ndim == 4:
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr)


def synth_image(fmt, batch=1, base=0.35):
    """Deterministic 256x256x3 float32 content (no private samples):
    per-channel structured gradients so the forward outputs are
    sample-specific, plus a fixed channel mix."""
    img = np.full((RES, RES, 3), base, dtype=np.float32)
    yy, xx = np.mgrid[0:RES, 0:RES]
    img[..., 0] += (0.25 * (xx % 64) / 64.0).astype(np.float32)
    img[..., 1] += (0.25 * (yy % 64) / 64.0).astype(np.float32)
    img[..., 2] = (0.5 + 0.4 * (((xx * 7 + yy * 13) % 32) / 32.0)).astype(np.float32)
    img = np.repeat(img[None, ...], batch, axis=0)
    return to_fmt(img, fmt).astype(np.float32)


# --- weight inventory -------------------------------------------------------

@pytest.mark.parametrize("fmt", FMTS)
def test_xseg_forward_shape_sigmoid_and_pretrain_skip(fmt):
    init_fmt(fmt)
    torch.manual_seed(12345)
    model = dfl_nn.XSeg(3, 2, 1, name="XSeg")
    model.build()
    model.init_weights()
    captured = []
    hook = model.uconv53.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone()))
    x = torch.from_numpy(synth_image(fmt, batch=2))
    expected = (2, RES, RES, 1) if fmt == "NHWC" else (2, 1, RES, RES)
    try:
        with torch.no_grad():
            logits, mask = model(x, pretrain=False)
            pre_logits, pre_mask = model(x, pretrain=True)
    finally:
        hook.remove()
    assert logits.shape == mask.shape == expected
    assert pre_logits.shape == pre_mask.shape == expected
    torch.testing.assert_close(mask, torch.sigmoid(logits), rtol=0, atol=0)
    torch.testing.assert_close(pre_mask, torch.sigmoid(pre_logits), rtol=0, atol=0)
    assert len(captured) == 2
    ch_axis = -1 if fmt == "NHWC" else 1
    normal_skip = captured[0].narrow(ch_axis, 8, 16)
    pretrain_skip = captured[1].narrow(ch_axis, 8, 16)
    assert torch.count_nonzero(normal_skip) > 0
    assert torch.count_nonzero(pretrain_skip) == 0


@pytest.mark.parametrize("fmt", FMTS)
def test_xseg_weight_inventory(fmt):
    m = build_xseg(fmt)
    weights = m.get_weights()
    assert len(weights) == 222
    total = sum(p.numel() for p in weights)
    assert total == 17_569_317
    # the full official variable names are bound (optimizer-state
    # naming contract, Phase 10B)
    names = [p._dfl_name for p in weights]
    assert names[0] == "XSeg/conv01/conv/weight:0"
    assert "XSeg/conv01/frn/eps:0" in names
    assert len(set(names)) == 222


# --- strict round-trip (same format) ----------------------------------------

@pytest.mark.parametrize("fmt", FMTS)
def test_xseg_strict_roundtrip_exact(plain_tmp, fmt):
    a = build_xseg(fmt)
    fill_weights(a)
    f1 = str(Path(plain_tmp) / "xseg_a.npy")
    a.save_weights(f1)

    b = build_xseg(fmt)
    assert b.load_weights(f1) is True
    f2 = str(Path(plain_tmp) / "xseg_b.npy")
    b.save_weights(f2)
    assert dicts_equal(save_dict(f1), save_dict(f2))


# --- cross-format strict load -----------------------------------------------

def test_xseg_crossformat_strict_load(plain_tmp):
    # NHWC-saved -> NCHW-model (and the reverse direction)
    a = build_xseg("NHWC")
    fill_weights(a)
    f_nhwc_saved = str(Path(plain_tmp) / "xseg_nhwc.npy")
    a.save_weights(f_nhwc_saved)

    b = build_xseg("NCHW")
    assert b.load_weights(f_nhwc_saved) is True
    fb = str(Path(plain_tmp) / "xseg_b.npy")
    b.save_weights(fb)
    assert dicts_equal(save_dict(f_nhwc_saved), save_dict(fb))

    c = build_xseg("NCHW")
    fill_weights(c)
    f_nchw = str(Path(plain_tmp) / "xseg_nchw.npy")
    c.save_weights(f_nchw)
    d = build_xseg("NHWC")
    assert d.load_weights(f_nchw) is True
    fd = str(Path(plain_tmp) / "xseg_d.npy")
    d.save_weights(fd)
    assert dicts_equal(save_dict(f_nchw), save_dict(fd))


# --- strict negatives (all-or-nothing + zero mutation) -----------------------

@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("corrupt", ["missing", "extra", "conv_shape",
                                     "dense_shape", "dtype", "duplicate_alias"])
def test_xseg_strict_load_negatives(plain_tmp, fmt, corrupt):
    m = build_xseg(fmt)
    fill_weights(m)
    f = str(Path(plain_tmp) / "xseg_ok.npy")
    m.save_weights(f)
    snapshot = [p.detach().clone() for p in m.get_weights()]

    d = save_dict(f)
    target = "conv01/conv/weight:0"
    if corrupt == "missing":
        del d[target]
    elif corrupt == "extra":
        d["bogus/weight:0"] = np.zeros((2,), dtype=np.float32)
    elif corrupt == "conv_shape":
        # same element count, wrong placement: (3,3,3,32) -> (3,3,32,3)
        # (the exact failure class of the Phase 10B scratch-driver
        # layout bug — the engine's layout hook + shape check must
        # reject it, never "fix" it)
        w = d[target].astype(np.float32)
        d[target] = w.reshape(3, 3, 32, 3).copy()
    elif corrupt == "dense_shape":
        w = d["dense1/weight:0"]
        d["dense1/weight:0"] = w.reshape(w.shape[::-1]).copy()
    elif corrupt == "dtype":
        d["dense1/weight:0"] = d["dense1/weight:0"].astype(np.float16)
    elif corrupt == "duplicate_alias":
        d[target[:-2]] = d[target]

    f_bad = str(Path(plain_tmp) / "xseg_bad.npy")
    with open(f_bad, "wb") as fh:
        fh.write(pickle.dumps(d, 4))

    with pytest.raises(dfl_ckpt.CheckpointLoadError):
        m.load_weights(f_bad)

    for p, s in zip(m.get_weights(), snapshot):
        assert torch.equal(p.detach(), s), "mutation after a failed strict load"


# --- XSegNet lifecycle (plain_tmp roots, no private paths) -------------------

@pytest.mark.parametrize("fmt", FMTS)
def test_xsegnetwork_inference_lifecycle(plain_tmp, fmt):
    # first run: no weight files -> official initialization branch
    init_fmt(fmt)
    torch.manual_seed(12345)
    from facelib.XSegNet import XSegNet

    net = XSegNet("XSeg", RES, load_weights=False,
                  weights_file_root=plain_tmp, training=False,
                  data_format=fmt)
    assert net.initialized is True
    assert [f for _, f in net.model_filename_list] == [f"XSeg_{RES}.npy"]
    assert not (Path(plain_tmp) / f"XSeg_{RES}.npy").exists()

    img3 = synth_image(fmt, batch=1)[0]             # 3-D single
    img4 = synth_image(fmt, batch=2)                # 4-D batch

    out3 = net.extract(img3)
    # the official extract output shape is data-format dependent
    # (C=1 in the net's data format; the batch dim is stripped for the
    # 3-D single-image contract)
    exp3 = (RES, RES, 1) if fmt == "NHWC" else (1, RES, RES)
    exp4 = (2, RES, RES, 1) if fmt == "NHWC" else (2, 1, RES, RES)
    assert out3.shape == exp3
    assert out3.dtype == np.float32
    out4 = net.extract(img4)
    assert out4.shape == exp4
    # OneDNN may choose different f32 kernels for batch 1 and batch 2.
    # Keep the comparison well below the frozen TF inference bound.
    np.testing.assert_allclose(out3, out4[0], rtol=0, atol=1e-5)
    # official extract contract: clip [0,1] + noise gate <0.1 -> 0
    for out in (out3, out4):
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert np.all((out == 0) | (out >= 0.1))

    # save (inference mode: model file only)
    net.save_weights()
    assert (Path(plain_tmp) / f"XSeg_{RES}.npy").exists()

    # resume: strict load of the saved file; identical deterministic
    # outputs (single-thread CPU)
    net2 = XSegNet("XSeg", RES, load_weights=True,
                   weights_file_root=plain_tmp, training=False,
                   data_format=fmt)
    assert net2.initialized is True
    np.testing.assert_array_equal(net2.extract(img3), out3)
    np.testing.assert_array_equal(net2.extract(img4), out4)


@pytest.mark.parametrize("fmt", FMTS)
def test_xsegnetwork_training_lifecycle_one_step(plain_tmp, fmt):
    # training mode: the official optimizer state file accompanies the
    # model file; one RMSprop step must survive save / strict resume
    # exactly (value-exact weights + iters + acc_* state).
    init_fmt(fmt)
    from facelib.XSegNet import XSegNet

    opt = dfl_nn.RMSprop(lr=0.0001, lr_dropout=0.3, name="opt")
    net = XSegNet("XSeg", RES, load_weights=False,
                  weights_file_root=plain_tmp, training=True,
                  optimizer=opt, data_format=fmt)
    assert [f for _, f in net.model_filename_list] == [
        f"XSeg_{RES}_opt.npy", f"XSeg_{RES}.npy"]

    x = synth_image(fmt, batch=2)
    target_shape = (2, RES, RES, 1) if fmt == "NHWC" else (2, 1, RES, RES)
    t = np.full(target_shape, 0.6, dtype=np.float32)
    xt = torch.from_numpy(x).to(device=dfl_nn.device, dtype=dfl_nn.floatx)
    tt = torch.from_numpy(t).to(device=dfl_nn.device, dtype=dfl_nn.floatx)
    logits, _ = net.flow(xt)
    loss = dfl_nn.sigmoid_cross_entropy(tt, logits)  # per-sample (B,)
    net.model.zero_grad()
    loss.backward(torch.ones_like(loss))  # official batch-SUM semantics
    grads = [(p.grad, p) for p in net.model_weights]
    net.opt.get_update_op(grads)()
    assert int(net.opt.iterations.item()) == 1

    net.save_weights()
    assert (Path(plain_tmp) / f"XSeg_{RES}_opt.npy").exists()
    assert (Path(plain_tmp) / f"XSeg_{RES}.npy").exists()
    d_opt = save_dict(Path(plain_tmp) / f"XSeg_{RES}_opt.npy")
    assert "iters:0" in d_opt
    n_acc = sum(1 for k in d_opt if k.startswith("acc_"))
    assert n_acc == 222

    # resume with a FRESH optimizer: strict load restores the exact
    # state (iters + acc_*) and weights
    opt2 = dfl_nn.RMSprop(lr=0.0001, lr_dropout=0.3, name="opt")
    net2 = XSegNet("XSeg", RES, load_weights=True,
                   weights_file_root=plain_tmp, training=True,
                   optimizer=opt2, data_format=fmt)
    assert int(net2.opt.iterations.item()) == 1
    w_a = [p.detach().cpu().numpy() for p in net.model_weights]
    w_b = [p.detach().cpu().numpy() for p in net2.model_weights]
    for a, b in zip(w_a, w_b):
        assert np.array_equal(a, b)
    for full_name in [p._dfl_name for p in net.model_weights]:
        a = net.opt.accumulators_dict[full_name].detach().cpu().numpy()
        b = net2.opt.accumulators_dict[full_name].detach().cpu().numpy()
        assert np.array_equal(a, b)

    (Path(plain_tmp) / f"XSeg_{RES}_opt.npy").unlink()
    with pytest.raises(FileNotFoundError):
        XSegNet("XSeg", RES, load_weights=True,
                weights_file_root=plain_tmp, training=True,
                optimizer=dfl_nn.RMSprop(lr=0.0001, lr_dropout=0.3,
                                         name="opt"), data_format=fmt)


def test_xsegnetwork_missing_required_file(plain_tmp):
    # the Phase 10B strict policy (documented deviation, v2 section 19 /
    # Phase 6 SAEHD precedent): a missing REQUIRED file on a resume
    # fails explicitly instead of the official silent re-initialization
    # (the official inference 0.5-ones fallback is therefore
    # unreachable on this port — pinned as a contract here)
    init_fmt("NHWC")
    from facelib.XSegNet import XSegNet

    with pytest.raises(FileNotFoundError):
        XSegNet("XSeg", RES, load_weights=True,
                weights_file_root=plain_tmp, training=False,
                data_format="NHWC")


def test_xsegnetwork_extract_2d_contract(plain_tmp):
    # the official extract contract is 3-D single / 4-D batch ONLY —
    # a 2-D row input must fail, not be silently reshaped
    init_fmt("NHWC")
    from facelib.XSegNet import XSegNet

    net = XSegNet("XSeg", RES, load_weights=False,
                  weights_file_root=plain_tmp, training=False,
                  data_format="NHWC")
    with pytest.raises(RuntimeError):
        net.extract(np.zeros((RES, RES), dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_xsegnetwork_cuda_inference_precision_scope(plain_tmp):
    from facelib.XSegNet import XSegNet

    init_fmt("NCHW")
    torch.manual_seed(12345)
    cpu = XSegNet("XSeg", RES, load_weights=False,
                  weights_file_root=plain_tmp, data_format="NCHW")
    x = synth_image("NCHW", batch=1)[0]
    y_cpu = cpu.extract(x)
    cpu.save_weights()

    previous = torch.backends.cudnn.allow_tf32
    try:
        dfl_nn.initialize_main_env()
        dfl_nn.initialize(dfl_nn.DeviceConfig.BestGPU(), "float32", "NCHW")
        assert dfl_nn.device.type == "cuda"
        gpu = XSegNet("XSeg", RES, load_weights=True,
                      weights_file_root=plain_tmp, data_format="NCHW")
        torch.backends.cudnn.allow_tf32 = True
        y_gpu = gpu.extract(x)
        assert torch.backends.cudnn.allow_tf32 is True
        assert y_gpu.shape == y_cpu.shape
        assert np.max(np.abs(y_cpu - y_gpu)) <= 5e-5
    finally:
        torch.backends.cudnn.allow_tf32 = previous
        init_fmt("NCHW")
