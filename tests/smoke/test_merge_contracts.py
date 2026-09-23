"""Phase 11 acceptance — merge contract tests (validation only).

The entire merge surface (``merger/Merger.py``, ``MergerScreen.py``,
``MergerConfig.py``, ``MergeMasked.py``, ``MergeAvatar.py``,
``InteractiveMergerSubprocessor.py``, ``mainscripts/Merger.py``,
``core/imagelib/**``) is LINE-IDENTICAL to the official DeepFaceLab
(verified by full-file diff in the Phase 11 audit — 0 diff lines) and
TF-free (pure numpy/OpenCV). NO merge production code is changed in
Phase 11; these tests pin the current official-identical behavior so
any future regression is caught:

- predictor callbacks: SAEHD/AMP ``predictor_func`` (NHWC float32 in;
  (face, celeb_mask, dst_mask) out; shapes/dtypes/[0,1] ranges;
  bit-deterministic on re-run) and the ``get_MergerConfig`` triple
  (func, (res,res,3), MergerConfigMasked(face_type, 'overlay')) —
  including AMP's morph prompt -> ``predictor_morph`` wrapper matching
  the direct ``predictor_func(face, morph)`` call, the 0..1 clip, and
  the binding morph grid 0.0/0.25/0.50/0.65/1.0;
- ``MergeMaskedFace`` mask modes 0..5 analytically (full / dst /
  learned-prd / learned-dst / product / clipped sum) with controlled
  synthetic predictors, plus XSeg modes 6..9 through a deterministic
  stub extractor (real XSeg inference is deliberately NOT pulled in);
- the official noise removal: mask values ``< 1/255`` are zeroed;
- blur / sharpen: box + gaussian sharpen, negative + positive
  blursharpen amounts, and the background outside the effective face
  mask stays unchanged (``out = img*(1-m) + out*m`` with m == 0);
- the color-transfer HANDOFF to the existing ``core.imagelib``
  functions (rct / hist-match) — the test calls the same imagelib
  functions as the independent reference, it does NOT re-implement
  the algorithms;
- the output contract: uint8 ``(H,W,4)``, alpha in [0,255];
- the official multi-face combine formula:
  ``final = final*(1-m) + img*m``, ``final_mask =
  clip(final_mask + m, 0, 1)``.
"""

import builtins
import importlib
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from core.interact import interact as io
from core.leras import nn as dfl_nn
from core import imagelib
from facelib import FaceType
import merger  # noqa: E402  (loads the package; its __init__ shadows the MergeMasked SUBMODULE attribute with the same-named FUNCTION, so the module must be loaded via importlib, not `import ... as`)
merge_masked = importlib.import_module("merger.MergeMasked")
from merger.MergerConfig import MergerConfigMasked, ctm_str_dict
from models import import_model

_smoke_dir = Path(__file__).resolve().parent
if str(_smoke_dir) not in sys.path:
    sys.path.insert(0, str(_smoke_dir))
from Model_SAEHDTest.Model import (  # noqa: E402
    DEFAULT_SEED_OPTIONS as SAEHD_DEFAULTS,
    SAEHDHeadless,
    _face_landmarks,
    make_model as make_saehd,
)
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    DEFAULT_SEED_OPTIONS as AMP_DEFAULTS,
    make_model as make_amp,
)

SAEHD_NAME = "test_SAEHD"
AMP_NAME = "test_AMP"
RES = 64
FRAME = 256
MORPH_GRID = (0.0, 0.25, 0.50, 0.65, 1.0)


@pytest.fixture(autouse=True)
def restore_rng_state():
    """Preserve every RNG state across tests (RNG preservation policy)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@pytest.fixture(autouse=True)
def restore_data_format():
    fmt = dfl_nn.data_format
    ch_axis = dfl_nn.conv2d_ch_axis
    sp_axes = dfl_nn.conv2d_spatial_axes
    yield
    # direct assignment (not set_data_format): the initial state
    # before any model runs is data_format == None, which the
    # validating setter rejects — the fixture must restore it
    dfl_nn.data_format = fmt
    dfl_nn.conv2d_ch_axis = ch_axis
    dfl_nn.conv2d_spatial_axes = sp_axes


@pytest.fixture
def no_cli_prompts(monkeypatch):
    """Neutralize the official CLI layer (the AMP morph prompt
    answers 0.0; tests re-patch at test level for other values — the
    more recent monkeypatch wins)."""
    def _input_str(prompt, default=None, **kwargs):
        return "" if default is None else str(default)

    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    monkeypatch.setattr(io, "input_str", _input_str)
    monkeypatch.setattr(io, "input_bool", lambda *a, **k: False)
    monkeypatch.setattr(io, "input_number", lambda *a, **k: 0.0)
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


@pytest.fixture
def workdir(plain_tmp):
    d = Path(plain_tmp) / "merge_contracts"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --- deterministic synthetic fixtures -------------------------------------

def _face_image_f32(size, variant=0):
    """A smooth deterministic BGR field in [0,1] (float32, NHWC)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    v = variant * 0.1
    img = np.stack([
        0.35 + 0.2 * np.sin(x / 41.0) + v,
        0.55 + 0.15 * np.cos(y / 57.0) + v,
        0.45 + 0.2 * np.sin((x + y) / 67.0) + v,
    ], axis=-1)
    return np.clip(img, 0.0, 1.0)


def _write_frame(root, size=FRAME, variant=0):
    """A deterministic uint8 BGR frame on disk (the merger reads it
    through the official cv2_imread path)."""
    img_u8 = (_face_image_f32(size, variant) * 255.0).astype(np.uint8)
    path = root / f"frame_{variant}.png"
    cv2.imwrite(str(path), img_u8)
    return path


def _landmarks(variant=0):
    """68-point FULL-face landmarks (the official 2DFAN layout,
    non-degenerate hull groups — the Model_SAEHDTest helper)."""
    lm = _face_landmarks(FRAME, variant=variant)
    assert lm.shape == (68, 2)
    return lm


def _frame_info(root, n_faces=1):
    """The minimal frame_info the official MergeMasked driver needs:
    one frame file (the official driver reads the SAME frame for
    every face) + per-face landmark sets (+ the motion fields)."""
    _write_frame(root, variant=0)
    if n_faces == 1:
        landmarks = [_landmarks(0)]
    else:
        # two faces with UNAMBIGUOUS left/right centroids (the base
        # landmark set is centered exactly at FRAME/2, which would
        # fall on the fake's decision boundary): face 1 shifted
        # slightly left, face 2 shifted far to the right
        lm1 = _landmarks(0).astype(np.float32)
        lm1[:, 0] = np.clip(lm1[:, 0] - 0.03 * FRAME, 0, FRAME - 1)
        lm2 = _landmarks(0).astype(np.float32)
        lm2[:, 0] = np.clip(lm2[:, 0] + 0.45 * FRAME, 0, FRAME - 1)
        landmarks = [lm1, lm2]
    return SimpleNamespace(filepath=str(root / "frame_0.png"),
                           landmarks_list=landmarks,
                           motion_power=0.0,
                           motion_deg=0.0)


def _const_predictor(face_value=0.5, prd_mask=1.0, dst_mask=1.0):
    """A deterministic synthetic predictor with CONSTANT outputs
    (shape-checked on the warped input): the merger sees
    (face, celeb_mask, dst_mask) — the official (predicted[0],
    predicted[1], predicted[2]) contract."""
    def predictor(face):
        h, w = face.shape[0], face.shape[1]
        face_out = np.full((h, w, 3), face_value, dtype=np.float32)
        prd = np.full((h, w), prd_mask, dtype=np.float32)
        dst = np.full((h, w), dst_mask, dtype=np.float32)
        return face_out, prd, dst
    return predictor


def _field_predictor(prd_field, dst_field, face_value=0.5):
    """A deterministic predictor returning per-pixel masks (fields
    sized like the warped input, resampled on the fly)."""
    def predictor(face):
        h, w = face.shape[0], face.shape[1]
        face_out = np.full((h, w, 3), face_value, dtype=np.float32)
        prd = cv2.resize(prd_field, (w, h))
        dst = cv2.resize(dst_field, (w, h))
        return face_out, prd, dst
    return predictor


def _texture_face(h, w, c=3):
    """A deterministic NON-constant BGR texture (channel-shifted
    sines in [0.1, 0.9]): blur/sharpen and color transfer actually
    change such a face, unlike a constant one."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    chans = []
    for k in range(c):
        v = 0.5 + 0.25 * np.sin(2.0 * np.pi * (x / 16.0 + y / 24.0)
                                 * (k + 1.0) + k * 0.7)
        chans.append(np.clip(v, 0.1, 0.9))
    return np.stack(chans, axis=-1)


def _texture_predictor(prd_mask=1.0, dst_mask=1.0):
    """A deterministic predictor with a TEXTURED face (see
    _texture_face) and constant masks."""
    def predictor(face):
        h, w = face.shape[0], face.shape[1]
        face_out = _texture_face(h, w, 3)
        prd = np.full((h, w), prd_mask, dtype=np.float32)
        dst = np.full((h, w), dst_mask, dtype=np.float32)
        return face_out, prd, dst
    return predictor


def _no_enhancer(face, is_tanh=False, preserve_size=True):
    return face


def _zero_xseg(face_256):
    return np.zeros((256, 256, 1), dtype=np.float32)


def _masked_cfg(mode="overlay", mask_mode=4, ctm=ctm_str_dict[None],
                erode=0, blur=0, sharpen=0, amount=0,
                super_res=0, scale=0):
    # ctm: the official ctm_str_dict keys the "None" (no transfer)
    # mode under the None value (ctm 0), and the named modes under
    # their string names (rct/lct/mkl/...)
    return MergerConfigMasked(
        face_type=FaceType.FULL,
        default_mode="overlay",
        mode=mode,
        masked_hist_match=True,
        hist_match_threshold=238,
        mask_mode=mask_mode,
        erode_mask_modifier=erode,
        blur_mask_modifier=blur,
        motion_blur_power=0,
        output_face_scale=scale,
        super_resolution_power=super_res,
        color_transfer_mode=ctm,
        image_denoise_power=0,
        bicubic_degrade_power=0,
        color_degrade_power=0,
        sharpen_mode=sharpen,
        blursharpen_amount=amount,
    )


def _merge_frame(root, predictor, cfg, n_faces=1):
    """The official multi-face driver on a synthetic frame. Returns
    (the uint8 (H,W,4) frame, the frame_info) — the official
    ``MergeMasked`` returns exactly the uint8 array
    (``(final_img*255).astype(np.uint8)``)."""
    frame_info = _frame_info(root, n_faces)
    out_u8 = merge_masked.MergeMasked(
        predictor, (RES, RES, 3), _no_enhancer, _zero_xseg,
        cfg, frame_info)
    assert out_u8.dtype == np.uint8
    return out_u8, frame_info


# --- predictor callbacks ----------------------------------------------------

def test_saehd_predictor_func_and_merger_config(workdir, no_cli_prompts):
    """SAEHD: get_MergerConfig triple + predictor_func contract
    (NHWC float32 in; (face, celeb_mask, dst_mask) out; shapes,
    dtypes, [0,1] ranges; bit-deterministic on re-run)."""
    seed = dict(SAEHD_DEFAULTS)
    seed.update(resolution=RES, archi="df", batch_size=1)
    m = make_saehd(SAEHDHeadless, workdir, is_training=False, seed=seed,
                   cpu_only=True)
    m.set_iter(1)
    m.save()
    model = import_model("SAEHD")(is_exporting=False,
                                  saved_models_path=workdir,
                                  cpu_only=True)

    predictor_func, input_shape, cfg = model.get_MergerConfig()
    assert input_shape == (RES, RES, 3)
    assert isinstance(cfg, MergerConfigMasked)
    assert cfg.face_type == model.face_type == FaceType.FULL
    assert cfg.default_mode == "overlay"

    face = np.random.RandomState(42).rand(RES, RES, 3).astype(np.float32)
    out = predictor_func(face)
    assert len(out) == 3
    face_out, celeb_mask, dst_mask = out
    assert face_out.shape == (RES, RES, 3)
    assert celeb_mask.shape == (RES, RES)
    assert dst_mask.shape == (RES, RES)
    for x in out:
        assert x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0

    # the inference path has no RNG: re-running is bit-identical
    out2 = predictor_func(face)
    for a, b in zip(out, out2):
        assert np.array_equal(a, b)


def test_amp_predictor_func_and_merger_config(workdir, no_cli_prompts,
                                              monkeypatch):
    """AMP: same contract + the morph prompt -> predictor_morph
    wrapper (the fixed-morph wrapper matches the direct
    predictor_func(face, morph) call bit-for-bit), the official
    0..1 clip of the prompted value, and the binding morph grid with
    the endpoint semantics (0.0 -> the dst-code heads, 1.0 -> the
    src-code heads)."""
    seed = dict(AMP_DEFAULTS)
    seed.update(resolution=RES, batch_size=1, ae_dims=32, inter_dims=32,
                e_dims=16, d_dims=16, d_mask_dims=6)
    m = make_amp(AMPHeadless, workdir, is_training=False, seed=seed,
                 cpu_only=True)
    m.set_iter(1)
    m.save()
    model = import_model("AMP")(is_exporting=False,
                                saved_models_path=workdir,
                                cpu_only=True)

    # the morph prompt answered 0.5 -> the merger gets the
    # predictor_morph wrapper bound to 0.5
    monkeypatch.setattr(io, "input_number", lambda *a, **k: 0.5)
    predictor_morph, input_shape, cfg = model.get_MergerConfig()
    assert input_shape == (RES, RES, 3)
    assert isinstance(cfg, MergerConfigMasked)
    assert cfg.face_type == FaceType.FULL
    assert cfg.default_mode == "overlay"

    face = np.random.RandomState(43).rand(RES, RES, 3).astype(np.float32)

    # the official morph clip: the prompt value is clipped to [0,1]
    # before the wrapper binds it
    monkeypatch.setattr(io, "input_number", lambda *a, **k: 1.7)
    predictor_morph_clipped, _, _ = model.get_MergerConfig()
    clipped = predictor_morph_clipped(face)
    direct_clipped = model.predictor_func(face, 1.0)
    for a, b in zip(clipped, direct_clipped):
        assert np.array_equal(a, b)

    out = predictor_morph(face)
    assert len(out) == 3
    face_out, celeb_mask, dst_mask = out
    assert face_out.shape == (RES, RES, 3)
    assert celeb_mask.shape == (RES, RES)
    assert dst_mask.shape == (RES, RES)
    for x in out:
        assert x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0

    # the fixed-morph wrapper is exactly the direct call at 0.5
    direct = model.predictor_func(face, 0.5)
    for a, b in zip(out, direct):
        assert np.array_equal(a, b)

    # the binding grid: shapes/dtypes/ranges at every morph, plus
    # the head mapping pinned against the model's own AE_merge,
    # whose Python int(inter_dims*morph) floor is the official
    # semantics (predictor order (face, prd_mask, dst_mask) vs the
    # internal NCHW AE_merge order (src_dst, dst_dstm, src_dstm))
    face_nchw = np.expand_dims(face.transpose(2, 0, 1), 0).copy()
    for mv in MORPH_GRID:
        grid_out = model.predictor_func(face, mv)
        assert len(grid_out) == 3
        for x in grid_out:
            assert x.dtype == np.float32
            assert x.min() >= 0.0 and x.max() <= 1.0
        heads = model.AE_merge(face_nchw, mv)
        assert heads[0].shape == (1, 3, RES, RES)
        assert heads[1].shape == (1, 1, RES, RES)
        assert heads[2].shape == (1, 1, RES, RES)
        assert np.allclose(grid_out[0], heads[0][0].transpose(1, 2, 0),
                           atol=1e-6)
        assert np.allclose(grid_out[1], heads[2][0, 0], atol=1e-6)
        assert np.allclose(grid_out[2], heads[1][0, 0], atol=1e-6)


# --- MergeMaskedFace mask modes + output contract ---------------------------

def test_merge_mask_modes_and_output_contract(workdir, no_cli_prompts):
    """The official mask-mode arithmetic (0..5) with controlled
    constant predictors, the official XSeg-mode handoff through a
    stub extractor (6..9), the output uint8 (H,W,4) contract, and
    background preservation.

    Official pipeline semantics pinned here (MergeMasked.py):
    - the work mask is processed at mask_subres_size = 4x the
      predict size (L24) and warped back with the 4x-space matrix,
      so the mask DOMAIN is slightly wider than the face-content
      warp domain: a few-pixel edge band where alpha >= 0.9 but the
      face content is the warp border (dark). Assertions on the
      face content therefore use the DEEP INTERIOR (the central
      box), where alpha == 1.0 and the content is exactly the
      warped predicted face;
    - with a CONSTANT predicted face (0.5) and full masks, the
      interior is the constant (0.5*255 = 127.5 -> 127/128) and
      outside the warped square (alpha == 0) the original frame
      survives within the uint8 round-trip.
    """
    out_full, frame_info = _merge_frame(
        workdir, _const_predictor(face_value=0.5, prd_mask=1.0,
                                  dst_mask=1.0), _masked_cfg(mask_mode=0))
    assert out_full.dtype == np.uint8
    assert out_full.shape == (FRAME, FRAME, 4)
    assert out_full[..., 3].min() >= 0 and out_full[..., 3].max() <= 255

    alpha = out_full[..., 3].astype(np.float32) / 255.0
    orig_u8 = cv2.imread(str(frame_info.filepath), cv2.IMREAD_COLOR)
    assert alpha.max() > 0.99

    # DEEP INTERIOR (central 70x70 box around the canvas center,
    # well inside the warped face square for this landmark layout):
    # alpha saturated and the frame is the constant predicted face
    interior = np.zeros((FRAME, FRAME), dtype=bool)
    interior[80:176, 80:176] = True
    assert alpha[interior].min() > 0.99
    face_region = out_full[..., :3][interior]
    assert np.abs(face_region.astype(np.float32) - 127.5).max() <= 2.0

    # outside the warped square (alpha == 0) the original frame
    # survives within the uint8 round-trip
    outside = alpha == 0
    assert outside.sum() > 100
    diff_out = np.abs(out_full[..., :3][outside].astype(np.float32)
                      - orig_u8[outside].astype(np.float32))
    assert diff_out.max() <= 1.0, "the background must survive"

    # mask_mode sweep 0..5: each mode runs, produces a valid frame
    # (both constant masks are 1.0, so every mode saturates the
    # same way here — the mode-specific arithmetic is exercised in
    # the partial-mask check below)
    for mask_mode in range(6):
        out, _ = _merge_frame(workdir,
                              _const_predictor(prd_mask=1.0, dst_mask=1.0),
                              _masked_cfg(mask_mode=mask_mode))
        assert out.dtype == np.uint8
        assert out.shape == (FRAME, FRAME, 4)
        assert out[..., 3].min() >= 0 and out[..., 3].max() <= 255

    # mode-specific arithmetic with PARTIAL masks: learned-prd=0.6,
    # learned-dst=0.4 -> the mode-4 (product) alpha inside the face
    # center is ~0.24 and mode-5 (clipped sum) ~1.0 (the region
    # gate at 0.1 is passed by both, so both merge)
    partial = _const_predictor(prd_mask=0.6, dst_mask=0.4)
    out_prod, _ = _merge_frame(workdir, partial,
                               _masked_cfg(mask_mode=4))
    out_sum, _ = _merge_frame(workdir, partial,
                              _masked_cfg(mask_mode=5))
    cy, cx = 128, 128  # canvas center = deep interior
    a_prod = out_prod[cy, cx, 3]
    a_sum = out_sum[cy, cx, 3]
    assert 0 < a_prod < a_sum <= 255, \
        f"product alpha ({a_prod}) must be below clipped-sum alpha " \
        f"({a_sum}) at the face center"

    # XSeg modes 6..9 through the deterministic stub extractor
    # (all-zero masks -> the region gate (L161-167) finds no
    # >= 0.1 region -> nothing is merged -> the frame is the
    # original within the uint8 round-trip): this pins that the
    # modes route through xseg_256_extract_func
    for mask_mode in range(6, 10):
        out, _ = _merge_frame(
            workdir, _const_predictor(prd_mask=1.0, dst_mask=1.0),
            _masked_cfg(mask_mode=mask_mode))
        assert out.dtype == np.uint8
        assert out.shape == (FRAME, FRAME, 4)
        diff = np.abs(out[..., :3].astype(np.float32)
                      - orig_u8.astype(np.float32))
        assert diff.max() <= 1.0, \
            f"mask_mode {mask_mode} with the all-zero XSeg stub " \
            f"must not merge anything"


def test_merge_noise_removal_pin(workdir, no_cli_prompts):
    """The official mask thresholds, pinned on a learned-prd mask
    (mask_mode=2) with constant values:
    - values < 1/255 are ZEROED by the official noise removal
      (MergeMasked.py L95 + L133): the merging mask comes out
      all-zero;
    - values in [1/255, 0.1) survive the noise removal but the
      official region gate (L161-167: no region of >= 0.1) skips
      all processing -> the frame passes through UNCHANGED while
      the (small) merging mask is still reported;
    - values >= 0.1 open the region gate: the face is actually
      merged (the interior shows the predicted face, alpha 255).
    """
    # (a) just UNDER the 1/255 threshold (0.3/255 ~ 0.00117
    # < 0.00392): zeroed -> the merging mask is all-zero
    out_under, frame_info = _merge_frame(
        workdir, _const_predictor(prd_mask=0.3 / 255.0),
        _masked_cfg(mask_mode=2))
    orig_u8 = cv2.imread(str(frame_info.filepath), cv2.IMREAD_COLOR)
    diff = np.abs(out_under[..., :3].astype(np.float32)
                  - orig_u8.astype(np.float32))
    assert diff.max() <= 1.0, "sub-threshold masks are zeroed"
    assert out_under[..., 3].max() == 0, \
        "the zeroed mask must leave the merging mask empty"

    # (b) just ABOVE the threshold but BELOW the 0.1 region gate
    # (2.0/255 ~ 0.00784): the frame stays unmerged, yet the mask
    # is reported (small non-zero merging mask)
    out_mid, _ = _merge_frame(
        workdir, _const_predictor(prd_mask=2.0 / 255.0),
        _masked_cfg(mask_mode=2))
    diff_mid = np.abs(out_mid[..., :3].astype(np.float32)
                      - orig_u8.astype(np.float32))
    assert diff_mid.max() <= 1.0, "sub-0.1 masks do not trigger merging"
    assert 0 < out_mid[..., 3].max() <= 2, \
        f"the sub-0.1 mask must be reported as a small merging mask " \
        f"(got max {out_mid[..., 3].max()})"

    # (c) comfortably ABOVE the region gate (1.0 — the alpha
    # channel reports the mask VALUE, so a saturated mask is needed
    # for the saturated-alpha assertion): the face is truly merged
    # and the interior shows the constant predicted face
    out_above, _ = _merge_frame(
        workdir, _const_predictor(prd_mask=1.0),
        _masked_cfg(mask_mode=2))
    interior = np.zeros((FRAME, FRAME), dtype=bool)
    interior[80:176, 80:176] = True
    assert out_above[128, 128, 3] == 255, \
        "super-gate masks must merge (saturated alpha)"
    reg = out_above[..., :3][interior]
    assert np.abs(reg.astype(np.float32) - 127.5).max() <= 2.0, \
        "the merged interior must show the predicted face"


def test_merge_blur_sharpen_background_invariance(workdir, no_cli_prompts):
    """Box/gaussian sharpen, negative (blur) and positive (sharpen)
    blursharpen amounts, and the mask blur — with the background
    OUTSIDE the effective face mask staying unchanged (the final
    combine is ``img*(1-m) + out*m`` with m == 0 there).

    The predicted face is a NON-constant texture (a constant face
    is invariant to blur/sharpen and would make the effect
    assertions vacuous)."""
    # a full predict-space learned-prd mask so the face region is
    # substantial; textured face so the effects are visible
    predictor = _texture_predictor(prd_mask=1.0, dst_mask=1.0)

    baseline, frame_info = _merge_frame(
        workdir, predictor,
        _masked_cfg(mask_mode=2, blur=0, sharpen=0, amount=0))
    orig_u8 = cv2.imread(str(frame_info.filepath), cv2.IMREAD_COLOR)

    results = {}
    for label, kw in (
            ("box_sharpen", dict(sharpen=1, amount=50)),
            ("gaussian_sharpen", dict(sharpen=2, amount=50)),
            ("negative_amount", dict(sharpen=2, amount=-50)),
            ("mask_blur", dict(blur=8)),
    ):
        out, _ = _merge_frame(
            workdir, predictor,
            _masked_cfg(mask_mode=2, **kw))
        assert out.dtype == np.uint8
        results[label] = out

    # background invariance, per variant: wherever THAT variant's
    # alpha is 0, the combine is the identity on the original frame
    for label, out in results.items():
        background = out[..., 3] == 0
        assert background.sum() > 100
        diff = np.abs(out[..., :3][background].astype(np.float32)
                      - orig_u8[background].astype(np.float32))
        assert diff.max() <= 1.0, f"{label}: background changed"

    # the face-region output differs from the baseline for each
    # active effect (the effect is real, not a no-op) — checked in
    # the deep interior, away from the warp edge band
    interior = np.zeros((FRAME, FRAME), dtype=bool)
    interior[80:176, 80:176] = True
    base_face = baseline[..., :3][interior].astype(np.float32)
    for label in ("box_sharpen", "gaussian_sharpen", "negative_amount"):
        assert np.abs(results[label][..., :3][interior]
                      .astype(np.float32) - base_face).max() > 1.0, \
            f"{label}: no visible effect in the face interior"
    # the mask blur changes the reported merging mask (boundary
    # softening) without altering the saturated interior content
    assert (results["mask_blur"][..., 3]
            != baseline[..., 3]).sum() > 100, \
        "mask blur must change the merging mask"
    assert np.abs(results["mask_blur"][..., :3][interior]
                  .astype(np.float32) - base_face).max() <= 1.0, \
        "mask blur must not alter the saturated interior content"


def test_merge_color_transfer_handoff(workdir, no_cli_prompts):
    """The color-transfer HANDOFF to the existing imagelib functions
    (not a re-implementation): with a TEXTURED predicted face (a
    constant face is invariant to rct and would make the handoff
    invisible) and a full mask, the production overlay output in
    the face interior must match the direct imagelib calls on the
    same warped inputs — rct via
    ``imagelib.reinhard_color_transfer`` and hist-match via
    ``imagelib.color_hist_match`` with the official white-channel
    masking glue (MergeMasked.py L190-204, incl. its
    ``hist_match_2[hist_match_1 > 1.0]`` quirk, a no-op on this
    full-mask data); overlay with ctm=none must NOT alter the
    predicted face beyond the warp round-trips.

    Comparisons are restricted to the deep interior (alpha >=
    0.99): the official mask domain is slightly wider than the face
    warp domain, so the edge band blends frame+face and must not
    contaminate the handoff check. The uint8 round-trip costs
    <= 1/255 ~ 0.004; the gate is 0.02 (a >1000x-worse
    regression — e.g. a wrong handoff argument order — would
    show up as an O(0.05+) mismatch).
    """
    from facelib import LandmarksProcessor

    out_none, frame_info = _merge_frame(
        workdir, _texture_predictor(prd_mask=1.0, dst_mask=1.0),
        _masked_cfg(mode="overlay", mask_mode=0, ctm=ctm_str_dict[None]))
    lm = frame_info.landmarks_list[0]

    frame_f = cv2.imread(str(frame_info.filepath),
                         cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    input_size = RES
    face_mat = LandmarksProcessor.get_transform_mat(
        lm, input_size, face_type=FaceType.FULL)
    dst_face_bgr = np.clip(cv2.warpAffine(
        frame_f, face_mat, (input_size, input_size),
        flags=cv2.INTER_CUBIC), 0, 1)
    # the same textured predicted face the predictor returned
    prd_tex = _texture_face(input_size, input_size, 3)
    # the official wrk_face_mask_area_a in the full-mask case:
    # ones where the (full) work mask > 0 — the entire square
    full_mask = np.ones((input_size, input_size, 1), dtype=np.float32)
    face_output_mat = LandmarksProcessor.get_transform_mat(
        lm, input_size, face_type=FaceType.FULL, scale=1.0)

    alpha = out_none[..., 3].astype(np.float32) / 255.0
    interior = np.zeros((FRAME, FRAME), dtype=bool)
    interior[80:176, 80:176] = True
    assert alpha[interior].min() > 0.99

    # (a) rct: the production output face interior must equal the
    # direct reinhard handoff warped back
    cfg_rct = _masked_cfg(mode="overlay", mask_mode=0,
                          ctm=ctm_str_dict["rct"])
    out_rct, _ = _merge_frame(workdir,
                              _texture_predictor(prd_mask=1.0,
                                                 dst_mask=1.0), cfg_rct)
    # the handoff must be real: rct changes the interior vs ctm=0
    assert np.abs(out_rct[..., :3][interior].astype(np.float32)
                  - out_none[..., :3][interior].astype(np.float32)
                  ).max() > 1.0, "rct must alter the face"
    ref_rct = imagelib.reinhard_color_transfer(
        prd_tex, dst_face_bgr, target_mask=full_mask,
        source_mask=full_mask)
    ref_warped = cv2.warpAffine(
        ref_rct, face_output_mat, (FRAME, FRAME),
        np.empty_like(frame_f), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    diff = np.abs(out_rct[..., :3][interior].astype(np.float32) / 255.0
                  - np.clip(ref_warped, 0, 1)[interior])
    assert diff.max() <= 0.02, \
        f"rct handoff mismatch: max {diff.max():.4f}"

    # (b) hist-match: the production output differs from the
    # ctm=none overlay (the handoff happened) and matches the
    # direct color_hist_match reference
    cfg_hist = _masked_cfg(mode="hist-match", mask_mode=0,
                           ctm=ctm_str_dict[None])
    out_hist, _ = _merge_frame(workdir,
                               _texture_predictor(prd_mask=1.0,
                                                  dst_mask=1.0),
                               cfg_hist)
    assert np.abs(out_hist[..., :3][interior].astype(np.float32)
                  - out_none[..., :3][interior].astype(np.float32)
                  ).max() > 1.0, \
        "hist-match must differ from the plain overlay"

    # the official white-channel glue (MergeMasked.py L191-204):
    # masked_hist_match=True with a full mask -> hist_mask_a == 1
    # everywhere -> white == 0
    hm1 = np.minimum(prd_tex * full_mask + (1.0 - full_mask)
                     * np.ones_like(prd_tex), 1.0)
    hm2 = np.minimum(dst_face_bgr * full_mask + (1.0 - full_mask)
                     * np.ones_like(prd_tex), 1.0)
    ref_hist = imagelib.color_hist_match(hm1, hm2, 238
                                         ).astype(np.float32)
    ref_hist_warped = cv2.warpAffine(
        ref_hist, face_output_mat, (FRAME, FRAME),
        np.empty_like(frame_f), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    diff_hist = np.abs(out_hist[..., :3][interior].astype(np.float32) / 255.0
                       - np.clip(ref_hist_warped, 0, 1)[interior])
    assert diff_hist.max() <= 0.02, \
        f"hist-match handoff mismatch: max {diff_hist.max():.4f}"


def test_merge_multi_face_combine_formula(workdir, no_cli_prompts,
                                          monkeypatch):
    """The official multi-face combine formula (merger/MergeMasked.py
    L333-344):
        final = final*(1-m) + img*m
        final_mask = clip(final_mask + m, 0, 1)
    with the (img, mask) pairs controlled by a test-local
    MergeMaskedFace stand-in (the formula under test is the official
    combine loop itself)."""
    def fake_merge_face(predictor_func, predictor_input_shape,
                        face_enhancer_func, xseg_256_extract_func,
                        cfg, frame_info, img_bgr_uint8, img_bgr,
                        img_face_landmarks):
        # deterministic per-face outputs: the face whose landmark
        # centroid is left of center gets the 0.25 constant + a
        # centered mask; the right-shifted face gets 0.75 + a lower
        # mask (no overlap)
        centroid_x = float(np.mean(img_face_landmarks[:, 0]))
        img = np.zeros((FRAME, FRAME, 3), dtype=np.float32)
        mask = np.zeros((FRAME, FRAME, 1), dtype=np.float32)
        if centroid_x < FRAME / 2:
            img[:, :] = 0.25
            mask[40:120, 40:120] = 1.0
        else:
            img[:, :] = 0.75
            mask[120:200, 148:228] = 1.0
        return img, mask

    monkeypatch.setattr(merge_masked, "MergeMaskedFace", fake_merge_face)

    out, frame_info = _merge_frame(workdir, _const_predictor(),
                                   _masked_cfg(), n_faces=2)

    assert out.dtype == np.uint8
    assert out.shape == (FRAME, FRAME, 4)

    # the official formula, recomputed on the controlled inputs
    img1 = np.full((FRAME, FRAME, 3), 0.25, dtype=np.float32)
    m1 = np.zeros((FRAME, FRAME, 1), dtype=np.float32)
    m1[40:120, 40:120] = 1.0
    img2 = np.full((FRAME, FRAME, 3), 0.75, dtype=np.float32)
    m2 = np.zeros((FRAME, FRAME, 1), dtype=np.float32)
    m2[120:200, 148:228] = 1.0
    final_img = img1 * (1 - m2) + img2 * m2
    final_mask = np.clip(m1 + m2, 0, 1)
    expected = (np.concatenate([final_img, final_mask], -1) * 255
                ).astype(np.uint8)
    assert np.array_equal(out, expected), \
        "the combine formula must match the official arithmetic"

    # spot checks: face-1 region -> the 0.25 face (0.25*255 = 63.75
    # -> 63 by truncation); face-2 region -> the 0.75 face
    # (0.75*255 = 191.25 -> 191); alpha accumulates to 1 in both
    # regions; the background stays 0
    assert out[80, 80, 0] == 63
    assert out[160, 188, 0] == 191
    assert out[80, 80, 3] == 255
    assert out[160, 188, 3] == 255
    assert out[20, 20, 3] == 0
