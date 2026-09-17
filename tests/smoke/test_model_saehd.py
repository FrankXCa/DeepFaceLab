"""Phase 6A — SAEHD structural skeleton (options, archi parsing,
components, discriminators, optimizers, checkpoint registration) tests.

Covers the official SAEHD model migrated to the torch foundation
(``models/Model_SAEHD/Model.py``; the verbatim official TF source is
preserved in ``Model_tf.py`` and never imported by torch paths):

- option parsing: stored ``data.dat`` options restored on resume, the
  legacy ``eyes_prio`` pop, the official ``lr_dropout`` bool->'y'/'n'
  backward compatibility, the true-face power zeroing warning for
  non-'df' archis (official first-run/override prompt block), and the
  pretrain structural overrides (gan_power 0.0 / warp off / flips on /
  styles 0 / uniform_yaw on / pretraining data path swap);
- the two official detection flags (official Model_tf.py L180-185,
  reproduced verbatim): ``pretrain_just_disabled`` (stored pretrain
  True -> now False through the override prompt -> inter re-init rule
  + ``set_iter(0)``) and ``gan_model_changed`` (stored gan archi
  differs from the overridden value -> D_src re-init rule), both
  exercised headless through scripted ``builtins.input`` answers + a
  forced ``input_in_time``;
- archi parsing: ``df`` / ``liae`` bases, the ``u``/``d``/``t``/``c``
  modifier subsets, and the explicit ``ValueError`` rejections
  (``xseg-ud`` / ``df-`` / ``df-xyz`` / ``df-ud-l`` / ``SAEHD`` /
  ``liae-x`` — the official interactive re-prompt loop has no headless
  torch equivalent, so a malformed stored archi fails at
  ``on_initialize`` — documented hardening);
- component construction & registration under the official logical
  names and file names: df ``encoder``/``inter``/``decoder_src``/
  ``decoder_dst`` (+ ``dis``/``code_discriminator.npy`` for
  true_face != 0), liae ``encoder``/``inter_AB``/``inter_B``/
  ``decoder``, ``D_src``/``GAN.npy`` for gan_power != 0, the three
  optimizers (``src_dst_opt``/``D_code_opt``/``GAN_opt``) and their
  files;
- the official lr=5e-5 optimizer wiring: AdaBelief vs RMSprop
  (``adabelief`` option), ``clipnorm = 1.0 if clipgrad else 0.0``,
  the lr_cos/lr_dropout coupling (y/cpu -> 500/0.3, else 0/1.0), the
  official src_dst saveable/trainable weight lists (df: all four
  components; liae: encoder+inter_AB+inter_B+decoder saveable,
  trainable = saveable with random_warp, encoder+inter_B+decoder
  without — official L333-338 keeps the encoder in the no-warp set)
  and the ``random_warp`` effect on the trainable list;
- the official optimizer-state naming integration: before
  ``initialize_variables`` the model binds every optimized
  parameter to its full official DFL variable name
  ``<component>/<sub_name>:0``, so the optimizer state checkpoint
  keys are the official ``ms_/vs_<full_varname>_0:0`` (RMSprop:
  ``acc_...``) instead of a positional ``param_<i>`` fallback —
  what makes the official optimizer state files strict-loadable
  (df: all four components under ``src_dst_opt``; liae: the
  saveable list, ``inter_AB`` included when ``random_warp`` is off;
  ``D_code_opt`` over the ``dis`` scope; ``GAN_opt`` over the
  ``D_src`` scope);
- forward shapes for every modifier combo at the official
  ``get_out_res``/``get_out_ch`` contract (encoder ``e*8`` ch,
  ``res//16`` (``res//32`` with 't'); inter ``lowest_dense_res``
  (``res//32`` with 'd') x2 (or x1 with 't'); decoder BGR 3ch + mask
  1ch full resolution) at resolutions 64/128/256, the ``D_src``
  ``(center_out, x)`` contract and the ``CodeDiscriminator`` 1x1
  single-channel output;
- backward readiness (no loss composition — Phase 6B): the encoder ->
  inter -> decoder_src chain, the code discriminator and the patch
  discriminator participate in autograd and produce finite parameter
  gradients; unused components (decoder_dst in that graph) stay
  grad-free;
- the headless lifecycle: the test-only ``Model_SAEHDTest`` package
  (``SAEHDHeadless`` — the model_class_name is derived from the
  folder name, mirroring the Phase 5 ``Model_Dummy`` convention; it
  overrides ONLY ``on_initialize_options`` with a direct option seed,
  the real ``on_initialize`` runs) constructs with real packed
  facesets in-process (``debug=True`` — the official in-process
  ``ThisThreadGenerator``; the subprocess generator is
  sandbox-hostile under pytest), and the official sample pipeline
  (SampleProcessor mask paths — the numpy-2 ``np.int`` fix in
  ``facelib/LandmarksProcessor.py`` is what unblocked it) runs
  end-to-end;
- strict save/load round-trip EXACT: the headless model (first run,
  all nine official component/optimizer files for df + true_face +
  GAN) saves through the official ``onSave``; the REAL
  ``SAEHDModel`` resumes from the same directory (class name
  ``SAEHD`` vs the test class's ``SAEHDTest`` — the per-model files
  are shared through the forced ``test_SAEHD`` model name) and every
  component weight and every optimizer state tensor (iters + ms_/vs_
  + the lr_dropout masks) is restored EXACT;
- strict failures on resume: a missing required component file
  fails explicitly (Phase 4/5 policy — the official silent
  re-initialization is NOT reproduced), a truncated component
  pickle fails, an archi mismatch (stored df files, liae options)
  fails, a shape-mismatched weight fails with the strict
  ``CheckpointLoadError``;
- device placement: the CPU-only construction (``cpu_only=True``)
  with parameters/forwards on CPU, and the RTX 4090 construction
  (``force_gpu_idxs=[0]``) with parameters/forwards on CUDA plus a
  full 9-file save/load round-trip EXACT on GPU
  (skip-if-CPU-only);
- the TensorFlow import boundary (no ``tensorflow`` in
  ``sys.modules`` after the torch SAEHD import) and the AST/regex
  hygiene over ``models/Model_SAEHD/Model.py`` (no tensorflow
  import, no ``torch.cuda.*`` attribute, no ``"cuda:"``/``'cuda:'``
  literals — ``Model_tf.py`` is the verbatim official reference and
  is excluded by design);
- the OPTIONAL real official checkpoint validation, gated on the
  ``DFL_TEST_SAEHD_CHECKPOINT`` environment variable (path to a
  directory holding the official SAEHD model files; unset -> the test
  skips, the normal suite never touches the private path). The
  checkpoint files are COPIED into the test temp dir and the official
  on-disk prefix (the bare ``SAEHD_`` form or the ``<model>_SAEHD_``
  form) is renamed to the test model name; the options are taken from
  the checkpoint's own ``data.dat`` with the documented test-only
  substitution ``face_type 'wf' -> 'f'`` (the fixture facesets carry
  full-face 2DFAN landmarks, not XSeg masks). The real model resumes
  in the TRAINING context (the GAN discriminator and optimizer-state
  files are registered and strictly loaded in-place; the packed
  faceset fixture supplies the sample data), and each official
  component and optimizer-state file is additionally validated
  through the Phase 4 conversion engine (zero missing-REQUIRED keys,
  zero shape/dtype mismatches are hard gates; benign legacy extra
  keys are reported, not failed). Labels: REAL CHECKPOINT
  STRUCTURAL MAPPING TESTED (PASS/FAIL) — TRAINING PARITY
  NOT_VERIFIED (Phase 6B).
"""

import ast
import builtins
import os
import pickle
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from core.leras import nn as dfl_nn
import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import checkpoint as ckpt
from core.leras import convert
from core.interact import interact as io  # the interact singleton

_SMOKE_DIR = Path(__file__).resolve().parent
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))
from Model_SAEHDTest.Model import (  # noqa: E402
    DEFAULT_SEED_OPTIONS,
    SAEHDHeadless,
    make_model as make_saehd,
    make_packed_faceset,
    make_training_dirs,
)

from models.Model_SAEHD import Model as SAEHDModel  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA (RTX 4090) environment required"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# every construction (test subclass AND the real model) uses this
# forced model name, so the per-model files the headless bootstrap
# writes are exactly the files the real-model resume loads
MODEL_NAME = "test_SAEHD"

# official on-disk SAEHD file prefix: the bare "SAEHD_*" form or the
# "<model>_-prefixed" form (e.g. "<model>_SAEHD_*")
_SAEHDFILE_RE = re.compile(r"^(?:.*_)?SAEHD_(?P<rest>[^/]+)$")

# the prompt sequence of the official first-run/override flow for a
# resume with a non-zero stored gan power and face_type 'f' (the
# official Model_tf.py L147-178 order, mirrored by the port): the
# lifecycle asks first (ModelBase L153-159), then the
# eyes/mouth/yaw/mask block (Model L223-229, the masked_training
# prompt is wf/head-only so it does not appear for 'f'), then the
# model options (Model L235+) — the scripted tests answer one value
# per prompt, "" meaning "keep the stored default". The
# write_preview_history second prompt (choose_preview_history) and
# the gan patch/dims prompts are conditional (off in the test seeds).
PROMPT_IDX = {
    "autobackup_hour": 0, "write_preview_history": 1, "target_iter": 2,
    "random_src_flip": 3, "random_dst_flip": 4, "batch_size": 5,
    "eyes_mouth_prio": 6, "uniform_yaw": 7, "blur_out_mask": 8,
    "models_opt_on_gpu": 9, "adabelief": 10, "lr_dropout": 11,
    "random_warp": 12, "random_hsv_power": 13, "gan_power": 14,
    "gan_patch_size": 15, "gan_dims": 16, "true_face_power": 17,
    "face_style_power": 18, "bg_style_power": 19, "ct_mode": 20,
    "clipgrad": 21, "pretrain": 22,
}
N_PROMPTS = 23


# --- helpers -----------------------------------------------------------------

def seed(**overrides):
    """DEFAULT_SEED_OPTIONS with per-test overrides (the test-only
    bootstrap seed; the real-model tests pre-seed data.dat instead)."""
    s = dict(DEFAULT_SEED_OPTIONS)
    s.update(overrides)
    return s


def construct(model_class, tmpdir, is_training=False, seed=None,
              debug=False, **device_kwargs):
    """Construct ``model_class`` through the official lifecycle
    constructor. ``seed`` is only honored by the test-only subclass
    (``SAEHDHeadless``); ``is_training=True`` builds the packed
    facesets and requires ``debug=True`` (in-process generator)."""
    if is_training:
        make_training_dirs(Path(tmpdir))
    return make_saehd(model_class, tmpdir, is_training=is_training,
                      seed=seed, debug=debug, **device_kwargs)


def bootstrap_saved(root, seed_overrides, is_training=True, **device_kwargs):
    """Build + save the headless bootstrap model whose saved file set
    matches the given seed (archi/dims/gan/true_face included); its
    in-memory weights are exactly the saved files (the later
    real-model resume loads them). If the seed carries a
    ``batch_size`` it is patched back into the saved ``data.dat``: the
    lifecycle pre-loads ``batch_size`` BEFORE ``on_initialize_options``
    (ModelBase L220, first-run options are empty -> 1) and re-writes
    the lifecycle value after ``on_initialize`` (L238), which would
    overwrite the seeded value in the saved options."""
    m = construct(SAEHDHeadless, root, is_training=is_training,
                  seed=seed(**seed_overrides),
                  debug=is_training, **device_kwargs)
    m.set_iter(1)
    m.save()
    bs = seed_overrides.get("batch_size")
    if bs is not None:
        data_path = Path(root) / f"{MODEL_NAME}_data.dat"
        data = pickle.loads(data_path.read_bytes())
        data["options"]["batch_size"] = bs
        data_path.write_bytes(pickle.dumps(data, 4))
    return m


def preseed_data_dat(root, options, it=1, sample_for_preview=None):
    """Write the ``data.dat`` bookkeeping file the real-model resume
    tests are built on (iter >= 1 => options restored, no prompts)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "iter": it,
        "options": dict(options),
        "loss_history": [],
        "sample_for_preview": sample_for_preview,
    }
    (root / f"{MODEL_NAME}_data.dat").write_bytes(pickle.dumps(data, 4))


def construct_real(tmpdir, options=None, is_training=False, it=1,
                   pretraining_data_path=None, debug=False,
                   sample_for_preview=None, **device_kwargs):
    """Construct the REAL ``SAEHDModel`` in resume mode: pre-seeded
    ``data.dat`` (+ the caller's component files / fixture dirs). A
    ``data.dat`` already present in the directory (bootstrap save or
    caller modification) is kept as-is."""
    root = Path(tmpdir)
    if not (root / f"{MODEL_NAME}_data.dat").exists():
        preseed_data_dat(root, options or {}, it=it,
                         sample_for_preview=sample_for_preview)
    if is_training:
        make_training_dirs(root)
    return SAEHDModel(
        is_training=is_training,
        saved_models_path=root,
        training_data_src_path=root / "src",
        training_data_dst_path=root / "dst",
        pretraining_data_path=(Path(pretraining_data_path)
                               if pretraining_data_path else None),
        pretrained_model_path=None,
        force_model_class_name=MODEL_NAME,
        debug=debug,
        **device_kwargs,
    )


class InputScript:
    """Answer provider for the official override prompts (one value
    per prompt in the official order; "" = keep the stored default)."""

    def __init__(self, overrides):
        self.answers = [""] * N_PROMPTS
        for idx, answer in overrides.items():
            self.answers[idx] = answer
        self.pos = 0

    def __call__(self, _prompt):
        if self.pos >= len(self.answers):
            return ""
        a = self.answers[self.pos]
        self.pos += 1
        return a


def to_fmt(x, to, source=None):
    """Phase 2 format conversion with the from-format filled in (the
    official API takes both formats explicitly)."""
    source = source or dfl_nn.data_format
    return dfl_nn.to_data_format(x, to, source)


def weights_of(model_component):
    return [w.detach().cpu().clone() for w in model_component.get_weights()]


@pytest.fixture
def headless_io(monkeypatch):
    """Pin the official interactive layer to deterministic headless
    semantics for the real-model tests: ``builtins.input`` answers
    every prompt with "" (the prompt returns the stored default) and
    the official ``input_in_time`` override poll never triggers.
    Override tests re-patch ``input_in_time`` (and, when a non-default
    answer is required, ``builtins.input``) AFTER this fixture —
    monkeypatch applies the most recent setattr first on undo, so the
    test-level patch wins while the test runs. TTY-independent."""
    monkeypatch.setattr(builtins, "input", InputScript({}))
    monkeypatch.setattr(io, "input_in_time", lambda s, t: False)


def check_weights_exact(a, b, what):
    wa, wb = weights_of(a), weights_of(b)
    assert len(wa) == len(wb), (what, len(wa), len(wb))
    for x, y in zip(wa, wb):
        assert x.shape == y.shape, (what, x.shape, y.shape)
        assert torch.equal(x, y), what


# --- option parsing -----------------------------------------------------------

def test_stored_options_restored_on_resume_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    opts = {
        "archi": "df-d", "ae_dims": 64, "e_dims": 32, "d_dims": 32,
        "d_mask_dims": 16, "adabelief": False, "clipgrad": True,
        "lr_dropout": "y", "random_warp": False, "batch_size": 3,
        "gan_power": 0.0, "true_face_power": 0.0,
    }
    bootstrap_saved(root, opts, is_training=False, cpu_only=True)
    model = construct_real(root, options=None, is_training=False,
                           cpu_only=True)

    assert model.is_first_run() is False
    for k, v in opts.items():
        assert model.options[k] == v, k
    assert model.archi_type == "df"
    # the stored archi drove the df component set
    assert [m.name for m, _ in model.model_filename_list][:4] == \
        ["encoder", "inter", "decoder_src", "decoder_dst"]
    assert not hasattr(model, "D_src")
    assert not hasattr(model, "code_discriminator")
    # the lifecycle-level batch size option is loaded pre-options
    assert model.batch_size == 3


def test_eyes_prio_removal_on_resume_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    # the legacy official key rides along in the stored options
    bootstrap_saved(root, {"archi": "df", "eyes_prio": True},
                    is_training=False, cpu_only=True)
    model = construct_real(root, options=None, is_training=False,
                           cpu_only=True)
    # official on_initialize pops the legacy key before use
    assert "eyes_prio" not in model.options
    assert model.options.get("eyes_mouth_prio") is False


def test_lr_dropout_bool_backward_compat_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    # the official legacy bool form in the stored options
    bootstrap_saved(root, {"archi": "df", "lr_dropout": True},
                    is_training=False, cpu_only=True)
    model = construct_real(root, options=None, is_training=False,
                           cpu_only=True)
    assert model.options["lr_dropout"] == "y"


def test_true_face_zeroed_for_non_df_cpu(plain_tmp, monkeypatch, headless_io):
    # the official zeroing happens in the first-run/override prompt
    # block; exercise it through the headless override path
    monkeypatch.setattr(io, "input_in_time", lambda s, t: True)

    root = Path(plain_tmp) / "m"
    # bootstrap the directory with a saved liae model so the real-model
    # resume below has the matching component files to load (training
    # context: ask_override — the official zeroing branch — only runs
    # on training resumes)
    m0 = construct(SAEHDHeadless, root, is_training=True,
                   seed=seed(archi="liae-ud", true_face_power=0.0),
                   debug=True, cpu_only=True)
    m0.set_iter(1)
    m0.save()
    # the stored options get a non-zero true face power (the override
    # prompt keeps it; the official warning branch must zero it back)
    data_path = root / f"{MODEL_NAME}_data.dat"
    data = pickle.loads(data_path.read_bytes())
    data["options"]["true_face_power"] = 0.5
    data_path.write_bytes(pickle.dumps(data, 4))

    model = construct_real(root, options=None, is_training=True,
                           debug=True, cpu_only=True)

    # the official warning branch zeroed it (non-df archi)
    assert model.options["true_face_power"] == 0.0
    assert not hasattr(model, "code_discriminator")


def test_pretrain_structural_overrides_cpu(plain_tmp, headless_io):
    pre = Path(plain_tmp) / "pre"
    # pretrain mode feeds the pretraining path ITSELF (as both src and
    # dst) to the sample generators — a single faceset.pak directly in
    # ``pre``, not a src/dst layout
    make_packed_faceset(pre)
    root = Path(plain_tmp) / "m"
    bootstrap_saved(root, {
        "archi": "liae-ud", "pretrain": False, "gan_power": 0.5,
        "lr_dropout": "y", "random_warp": True,
        "random_hsv_power": 0.2, "face_style_power": 2.0,
        "bg_style_power": 1.0, "uniform_yaw": False,
    }, is_training=True, cpu_only=True)
    # the resume flips the stored pretrain to True
    data_path = root / f"{MODEL_NAME}_data.dat"
    data = pickle.loads(data_path.read_bytes())
    data["options"]["pretrain"] = True
    data_path.write_bytes(pickle.dumps(data, 4))

    model = construct_real(root, options=None, is_training=True,
                           pretraining_data_path=pre, debug=True,
                           cpu_only=True)

    assert model.pretrain is True
    # official L230-239: the effective values are forced, the stored
    # options themselves are kept
    assert model.gan_power == 0.0
    assert model.options["gan_power"] == 0.5
    assert model.options_show_override["lr_dropout"] == "n"
    assert model.options_show_override["random_warp"] is False
    assert model.options_show_override["gan_power"] == 0.0
    assert model.options_show_override["uniform_yaw"] is True
    # the forced gan_power=0 suppresses D_src entirely
    assert not hasattr(model, "D_src")
    # optimizers still exist (pretrain trains the AE with lr_dropout off)
    assert model.src_dst_opt.lr_cos == 0
    assert model.src_dst_opt.lr_dropout == 1.0
    # the pretraining data path drove the sample generators
    assert model.generator_list is not None


def test_pretrain_requires_pretraining_path_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    opts = {"archi": "liae-ud", "pretrain": True}
    with pytest.raises(Exception, match="pretraining_data_path"):
        construct_real(root, options=opts, is_training=False, cpu_only=True)


# --- detection flags (official L180-185) ---------------------------------------

def _bootstrap_saved_model(root, seed_overrides):
    """Headless bootstrap: build + save the model files that the
    real-model override resume below loads."""
    m = construct(SAEHDHeadless, root, is_training=True,
                  seed=seed(archi="df", gan_power=0.1,
                            **seed_overrides),
                  debug=True, cpu_only=True)
    m.set_iter(1)
    m.save()
    return m


def test_pretrain_just_disabled_detection_cpu(plain_tmp, monkeypatch, headless_io):
    monkeypatch.setattr(io, "input_in_time", lambda s, t: True)

    root = Path(plain_tmp) / "m"
    m1 = _bootstrap_saved_model(root, {"pretrain": False})
    # flip the stored pretrain to True: the override prompt below
    # answers 'n' -> the official detection fires
    data_path = root / f"{MODEL_NAME}_data.dat"
    data = pickle.loads(data_path.read_bytes())
    data["options"]["pretrain"] = True
    data_path.write_bytes(pickle.dumps(data, 4))

    monkeypatch.setattr(builtins, "input",
                        InputScript({PROMPT_IDX["pretrain"]: "n"}))
    m2 = construct_real(root, options=None, is_training=True, debug=True,
                        cpu_only=True)

    # truthiness (not `is True`): the official flag is an `and` of `==`
    # on the (numpy) option scalars, so it may be a numpy bool
    assert bool(m2.pretrain_just_disabled) is True
    # official on_initialize: set_iter(0)
    assert m2.get_iter() == 0
    # the official re-init rule: the df inter component is re-
    # initialized (NOT loaded from the file), the encoder IS loaded
    inter_differs = any(
        not torch.equal(a, b)
        for a, b in zip(weights_of(m1.inter), weights_of(m2.inter)))
    assert inter_differs, "inter should have been re-initialized"
    enc_same = all(
        torch.equal(a, b)
        for a, b in zip(weights_of(m1.encoder), weights_of(m2.encoder)))
    assert enc_same, "encoder must be loaded, not re-initialized"
    assert m2.options["pretrain"] is False


def test_gan_model_changed_detection_cpu(plain_tmp, monkeypatch, headless_io):
    monkeypatch.setattr(io, "input_in_time", lambda s, t: True)

    root = Path(plain_tmp) / "m"
    m1 = _bootstrap_saved_model(root, {"gan_patch_size": 16})
    assert m1.options["gan_patch_size"] == 16

    # the override prompt changes the GAN patch size (16 -> 32)
    monkeypatch.setattr(builtins, "input",
                        InputScript({PROMPT_IDX["gan_patch_size"]: "32"}))
    m2 = construct_real(root, options=None, is_training=True, debug=True,
                        cpu_only=True)

    # truthiness (not `is True`): the official flag is an `or` of `!=`
    # on the (numpy) option scalars, so it is a numpy bool after an
    # override prompt (np.clip -> numpy int)
    assert bool(m2.gan_model_changed) is True
    assert m2.options["gan_patch_size"] == 32
    # official 637-657 rule: D_src is re-initialized (NOT loaded)
    d_differs = any(
        not torch.equal(a, b)
        for a, b in zip(weights_of(m1.D_src), weights_of(m2.D_src)))
    assert d_differs, "D_src should have been re-initialized"
    enc_same = all(
        torch.equal(a, b)
        for a, b in zip(weights_of(m1.encoder), weights_of(m2.encoder)))
    assert enc_same, "encoder must be loaded, not re-initialized"


def test_plain_resume_flags_stay_false_cpu(plain_tmp, headless_io):
    # the regression that the unconditional flag assignment fixes:
    # a plain resume (no override) must construct with both flags
    # present and False — the original conditional assignment left
    # pretrain_just_disabled unset and crashed on_initialize
    root = Path(plain_tmp) / "m"
    # gan_power=0.1 is the _bootstrap_saved_model default (passed
    # explicitly, so it must not be repeated in the overrides)
    m1 = _bootstrap_saved_model(root, {"pretrain": False})
    # a training-context plain resume (no override prompt: headless_io
    # pins input_in_time to False) so all components — D_src included —
    # are present for the weight comparison
    m2 = construct_real(root, options=None, is_training=True,
                        debug=True, cpu_only=True)
    # truthiness (not `is False`): the official flags are `or`/`and` of
    # `!=`/`==` on the (numpy) option scalars, so they may be numpy bools
    assert bool(m2.gan_model_changed) is False
    assert bool(m2.pretrain_just_disabled) is False
    assert m2.get_iter() == 1
    check_weights_exact(m1.encoder, m2.encoder, "encoder (plain resume)")
    check_weights_exact(m1.inter, m2.inter, "inter (plain resume)")
    check_weights_exact(m1.D_src, m2.D_src, "D_src (plain resume)")


# --- archi parsing --------------------------------------------------------------

INVALID_ARCHIS = [
    ("xseg-ud", "base must be"),
    ("df-", "empty modifier list"),
    ("df-xyz", "subset of u/d/t/c"),
    ("df-ud-l", "expected 'df' or 'liae'"),
    ("SAEHD", "base must be"),
    ("liae-x", "subset of u/d/t/c"),
]

@pytest.mark.parametrize("archi,match", INVALID_ARCHIS)
def test_invalid_archi_rejected_cpu(plain_tmp, archi, match):
    with pytest.raises(ValueError, match=match):
        construct(SAEHDHeadless, Path(plain_tmp) / archi.replace("-", "_"),
                  is_training=False, seed=seed(archi=archi), cpu_only=True)


VALID_ARCHIS = [
    "df", "df-u", "df-d", "df-t", "df-c",
    "df-ut", "df-ud", "df-td", "df-uc",
    "df-utd", "df-udt", "df-uct",
    "liae", "liae-u", "liae-d", "liae-t", "liae-c",
    "liae-ut", "liae-ud", "liae-td", "liae-uc",
    "liae-utd", "liae-udt", "liae-uct",
]

@pytest.mark.parametrize("archi", VALID_ARCHIS)
def test_valid_archi_parses_cpu(plain_tmp, archi):
    model = construct(SAEHDHeadless, Path(plain_tmp) / archi.replace("-", "_"),
                      is_training=False, seed=seed(archi=archi),
                      cpu_only=True)
    expected_base = "df" if archi.startswith("df") else "liae"
    assert model.archi_type == expected_base
    assert model.options["archi"] == archi


# --- component construction / registration ---------------------------------------

def test_df_components_names_files_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1),
                      debug=True, cpu_only=True)
    pairs = [(m, f) for m, f in model.model_filename_list]
    names = {m.name: f for m, f in pairs}
    assert names["encoder"] == "encoder.npy"
    assert names["inter"] == "inter.npy"
    assert names["decoder_src"] == "decoder_src.npy"
    assert names["decoder_dst"] == "decoder_dst.npy"
    assert names["dis"] == "code_discriminator.npy"
    assert names["D_src"] == "GAN.npy"
    assert names["src_dst_opt"] == "src_dst_opt.npy"
    assert names["D_code_opt"] == "D_code_opt.npy"
    assert names["GAN_opt"] == "GAN_opt.npy"
    # the true-face discriminator takes the inter code resolution
    assert model.code_discriminator.name == "dis"
    assert model.D_src.name == "D_src"


def test_liae_components_names_files_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "liae",
                      is_training=True, seed=seed(archi="liae-ud",
                                                  gan_power=0.1),
                      debug=True, cpu_only=True)
    names = {m.name: f for m, f in model.model_filename_list}
    assert names["encoder"] == "encoder.npy"
    assert names["inter_AB"] == "inter_AB.npy"
    assert names["inter_B"] == "inter_B.npy"
    assert names["decoder"] == "decoder.npy"
    assert names["D_src"] == "GAN.npy"
    assert names["src_dst_opt"] == "src_dst_opt.npy"
    assert names["GAN_opt"] == "GAN_opt.npy"
    # liae has no true-face code discriminator (official df-only path)
    assert not hasattr(model, "code_discriminator")
    assert "D_code_opt" not in names


def test_gan_disabled_drops_discriminators_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "nog",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.0,
                                gan_power=0.0),
                      debug=True, cpu_only=True)
    assert not hasattr(model, "D_src")
    assert not hasattr(model, "code_discriminator")
    names = {m.name for m, _ in model.model_filename_list}
    assert names == {"encoder", "inter", "decoder_src", "decoder_dst",
                     "src_dst_opt"}


def _check_registration_order(model, expected):
    """Exact official order + entry KIND: [component, file] LIST pairs
    for components, (optimizer, file) TUPLES for the optimizers."""
    entries = model.model_filename_list
    assert len(entries) == len(expected), \
        (model.options.get('archi'), [e[1] for e in entries])
    for entry, (name, filename, kind) in zip(entries, expected):
        component, fname = entry
        assert component.name == name, (component.name, name)
        assert fname == filename
        assert type(entry) is kind, (name, type(entry))


def test_model_filename_list_registration_order_cpu(plain_tmp):
    # df + true_face + GAN: the official nine-entry registration
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df_full",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1),
                      debug=True, cpu_only=True)
    _check_registration_order(model, [
        ("encoder", "encoder.npy", list),
        ("inter", "inter.npy", list),
        ("decoder_src", "decoder_src.npy", list),
        ("decoder_dst", "decoder_dst.npy", list),
        ("dis", "code_discriminator.npy", list),
        ("D_src", "GAN.npy", list),
        ("src_dst_opt", "src_dst_opt.npy", tuple),
        ("D_code_opt", "D_code_opt.npy", tuple),
        ("GAN_opt", "GAN_opt.npy", tuple),
    ])

    # liae + GAN: the official seven-entry registration
    model = construct(SAEHDHeadless, Path(plain_tmp) / "liae_g",
                      is_training=True, seed=seed(archi="liae-ud",
                                                  gan_power=0.1),
                      debug=True, cpu_only=True)
    _check_registration_order(model, [
        ("encoder", "encoder.npy", list),
        ("inter_AB", "inter_AB.npy", list),
        ("inter_B", "inter_B.npy", list),
        ("decoder", "decoder.npy", list),
        ("D_src", "GAN.npy", list),
        ("src_dst_opt", "src_dst_opt.npy", tuple),
        ("GAN_opt", "GAN_opt.npy", tuple),
    ])


# --- optimizer wiring ---------------------------------------------------------------

def test_optimizers_adabelief_lr_clip_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1, adabelief=True,
                                clipgrad=False),
                      debug=True, cpu_only=True)
    for opt in (model.src_dst_opt, model.D_code_opt, model.D_src_dst_opt):
        assert isinstance(opt, dfl_nn.AdaBelief)
        assert opt.lr == 5e-5
        assert opt.clipnorm == 0.0
        assert opt.lr_cos == 0          # lr_dropout 'n' (the seed default)
        assert opt.lr_dropout == 1.0


def test_optimizers_rmsprop_clipgrad_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True,
                      seed=seed(archi="df", adabelief=False,
                                clipgrad=True),
                      debug=True, cpu_only=True)
    for opt in (model.src_dst_opt,):
        assert isinstance(opt, dfl_nn.RMSprop)
        assert opt.lr == 5e-5
        assert opt.clipnorm == 1.0


def test_optimizer_lr_dropout_coupling_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True, seed=seed(archi="df",
                                                  lr_dropout="y"),
                      debug=True, cpu_only=True)
    assert model.src_dst_opt.lr_cos == 500
    assert model.src_dst_opt.lr_dropout == 0.3


def test_optimizer_lr_dropout_cpu_variant_cpu(plain_tmp):
    # 'cpu' enables the same lr_cos/lr_dropout coupling as 'y'
    # (official: lr_dropout in ['y', 'cpu'] and not pretrain)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True, seed=seed(archi="df",
                                                  lr_dropout="cpu"),
                      debug=True, cpu_only=True)
    assert model.src_dst_opt.lr_cos == 500
    assert model.src_dst_opt.lr_dropout == 0.3


def test_optimizer_state_official_names_cpu(plain_tmp):
    # Phase 3E2/6A state-naming integration: BEFORE
    # initialize_variables the model binds every optimized parameter
    # to its full official DFL variable name
    # '<component>/<sub_name>:0' (the component's checkpoint scope +
    # its official sub-name), so the optimizer state checkpoint keys
    # are the official 'ms_/vs_<full_varname>_0:0' (RMSprop:
    # 'acc_...') — NOT the positional 'param_<i>' fallback. The
    # official DFL optimizer names its state after the trained
    # variables, which is what makes the official optimizer state
    # files (src_dst_opt.npy / D_code_opt.npy / GAN_opt.npy)
    # strict-loadable.
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df_full",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1),
                      debug=True, cpu_only=True)

    def state_names(opt):
        return dict(opt._iter_official_weights())

    def expected_state(comps):
        names = {"iters:0"}
        for comp in comps:
            for sub, _p in comp._iter_official_weights():
                base = f"{comp.name}/{sub}".replace(":", "_")
                names.add(f"ms_{base}:0")
                names.add(f"vs_{base}:0")
        return names

    # 1) every registered parameter of every optimized component
    #    carries its full official DFL variable name
    df_comps = (model.encoder, model.inter, model.decoder_src,
                model.decoder_dst)
    for comp in df_comps + (model.code_discriminator, model.D_src):
        for sub, p in comp._iter_official_weights():
            assert p._dfl_name == f"{comp.name}/{sub}", \
                (comp.name, sub, p._dfl_name)

    # 2) src_dst_opt: iters + ms_/vs_ under the official names of
    #    every saveable parameter (the four df components); the state
    #    tensors pair with their parameters (shape + dtype)
    expected = expected_state(df_comps)
    names = state_names(model.src_dst_opt)
    assert next(iter(names)) == "iters:0"
    assert not any("param_" in n for n in names)
    assert set(names) == expected, sorted(set(names) ^ expected)
    for comp in df_comps:
        for sub, p in comp._iter_official_weights():
            base = f"{comp.name}/{sub}".replace(":", "_")
            for prefix in ("ms", "vs"):
                s = names[f"{prefix}_{base}:0"]
                assert tuple(s.shape) == tuple(p.shape)
                assert s.dtype == p.dtype

    # 3) D_code_opt (scope 'dis') / GAN_opt (scope 'D_src'): the same
    #    official naming over their single component's parameters
    assert set(state_names(model.D_code_opt)) == \
        expected_state([model.code_discriminator])
    assert set(state_names(model.D_src_dst_opt)) == \
        expected_state([model.D_src])

    # 4) liae + random_warp off: the src_dst_opt state still covers
    #    the SAVEABLE list (official L333-338: inter_AB stays in the
    #    optimizer state even though it is untrainable without warp)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "liae_nowarp",
                      is_training=True,
                      seed=seed(archi="liae", random_warp=False,
                                gan_power=0.1),
                      debug=True, cpu_only=True)
    liae_comps = (model.encoder, model.inter_AB, model.inter_B,
                  model.decoder)
    assert next(iter(state_names(model.src_dst_opt))) == "iters:0"
    assert set(state_names(model.src_dst_opt)) == \
        expected_state(liae_comps)
    assert set(state_names(model.D_src_dst_opt)) == \
        expected_state([model.D_src])


def test_optimizer_state_layout_official_cpu(plain_tmp):
    # Phase 4 state LAYOUT integration: official optimizer state
    # tensors are stored in the OFFICIAL layout of the tracked
    # variable (the NHWC conv kernel on disk) and converted back to
    # the torch layout on load — the state buffer holds the tracked
    # parameter's torch layout. An identity-hook fallback would write
    # and read the torch layout and would fail strict load of a real
    # official state file (real-checkpoint evidence: 62 NHWC/NCHW
    # shape mismatches on a 320-liae SAEHD src_dst_opt.npy).
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True, seed=seed(archi="df"),
                      debug=True, cpu_only=True)
    opt = model.src_dst_opt

    # layout-sensitive NON-zero state (zeros would round-trip even
    # through a broken conversion): deterministic per-element values
    g = torch.Generator().manual_seed(1234)
    for name, t in opt._iter_official_weights():
        if name == "iters:0":
            t.fill_(7)
        else:
            t.copy_(torch.randn(t.shape, generator=g, dtype=t.dtype))

    def conv4_of(component, transpose=False):
        for sub, p in component._iter_official_weights():
            owner = getattr(p, "_dfl_owner_layer", None)
            if p.dim() != 4:
                continue
            if transpose and not isinstance(owner, dfl_nn.Conv2DTranspose):
                continue
            if not transpose and isinstance(owner, dfl_nn.Conv2DTranspose):
                continue
            return component, sub, p
        return None

    # one plain conv kernel (encoder) and, if the archi builds any,
    # one conv-transpose kernel (decoder upsample stage) — both
    # official orientations
    picks = [p for p in (conv4_of(model.encoder),
                         conv4_of(model.decoder_src, transpose=True))
             if p is not None]
    assert picks, "no 4-D conv weight found in the df encoder"

    # --- save: the on-disk file is OFFICIAL-layout -------------------
    path = Path(plain_tmp) / "opt_state.npy"
    opt.save_weights(str(path))
    d = pickle.loads(path.read_bytes())
    # torch-layout state values (the in-memory buffers), by official name
    state = dict(opt._iter_official_weights())
    for comp, sub, p in picks:
        t = tuple(p.shape)  # torch layout (c1, c2, kH, kW)
        key = f"ms_{comp.name}/{sub}".replace(":", "_") + ":0"
        arr = d[key]
        # official conv / conv-transpose kernel orientation:
        # (kH, kW, c1, c2) — the spatial pair swapped with the
        # channel pair
        assert arr.shape == (t[3], t[2], t[1], t[0]), (arr.shape, t)
        # the on-disk VALUE is the state buffer transposed to the
        # official layout (not the parameter value — the state is its
        # own tensor, here non-zero and layout-sensitive)
        assert np.array_equal(
            arr, np.asarray(state[key].detach().cpu()).transpose(2, 3, 1, 0)), sub

    # --- load: a fresh optimizer strict-loads the official-layout
    # file and restores the EXACT torch-layout state
    opt2 = dfl_nn.AdaBelief(lr=5e-5, lr_dropout=1.0, lr_cos=0,
                            clipnorm=0.0, name="src_dst_opt")
    opt2.initialize_variables(list(model.src_dst_saveable_weights))
    assert opt2.load_weights(str(path)) is True
    for (n1, t1), (n2, t2) in zip(opt._iter_official_weights(),
                                  opt2._iter_official_weights()):
        assert n1 == n2, (n1, n2)
        assert torch.equal(t1, t2), n1


def test_src_dst_weight_lists_cpu(plain_tmp):
    # df: saveable == trainable == all four components (official L330-332)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df",
                      is_training=True, seed=seed(archi="df"),
                      debug=True, cpu_only=True)
    expected = (model.encoder.get_weights() + model.inter.get_weights()
                + model.decoder_src.get_weights()
                + model.decoder_dst.get_weights())
    assert list(model.src_dst_saveable_weights) == list(expected)
    assert list(model.src_dst_trainable_weights) == list(expected)

    # liae + random_warp: trainable == saveable (official L333-336)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "liae_w",
                      is_training=True,
                      seed=seed(archi="liae", random_warp=True),
                      debug=True, cpu_only=True)
    expected = (model.encoder.get_weights() + model.inter_AB.get_weights()
                + model.inter_B.get_weights() + model.decoder.get_weights())
    assert list(model.src_dst_saveable_weights) == list(expected)
    assert list(model.src_dst_trainable_weights) == list(expected)

    # liae + random_warp off: official L338 keeps the encoder,
    # drops inter_AB (the src path is frozen)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "liae_now",
                      is_training=True,
                      seed=seed(archi="liae", random_warp=False),
                      debug=True, cpu_only=True)
    expected = (model.encoder.get_weights() + model.inter_B.get_weights()
                + model.decoder.get_weights())
    assert list(model.src_dst_saveable_weights) != list(expected)
    assert list(model.src_dst_trainable_weights) == list(expected)


# --- forward shapes ------------------------------------------------------------------

def _archi_opts(archi):
    return set(archi.split("-")[1]) if "-" in archi else set()


@pytest.mark.parametrize("archi", [
    "df", "df-u", "df-d", "df-t", "df-ut", "df-ud", "df-td", "df-udt",
    "df-c", "df-uc", "df-ct", "df-dc", "df-tc",
    "liae", "liae-u", "liae-d", "liae-t", "liae-ut", "liae-ud",
    "liae-td", "liae-udt", "liae-c", "liae-uc",
    "liae-ct", "liae-dc", "liae-tc",
])
def test_forward_shapes_cpu(plain_tmp, archi):
    res = 128
    model = construct(SAEHDHeadless, Path(plain_tmp) / archi.replace("-", "_"),
                      is_training=False, seed=seed(archi=archi),
                      cpu_only=True)
    opts = _archi_opts(archi)
    is_df = model.archi_type == "df"

    x = torch.rand(2, res, res, 3)
    x = to_fmt(x, model.model_data_format, "NHWC")

    # the official encoder output is a FLAT vector (N, e*8*res_e^2) —
    # the 2D code has no data-format axis, so no format conversion
    e_res = res // 32 if "t" in opts else res // 16
    enc = model.encoder(x)
    assert tuple(enc.shape) == (2, model.options["e_dims"] * 8
                                * e_res * e_res), archi

    x_model = to_fmt(x, model.model_data_format, "NHWC")
    enc_model = model.encoder(x_model)
    if is_df:
        inter_out = to_fmt(model.inter(enc_model), "NHWC", model.model_data_format)
        low = res // 32 if "d" in opts else res // 16
        i_res = low if "t" in opts else low * 2
        ae_out = model.options["ae_dims"]
        assert tuple(inter_out.shape) == (2, i_res, i_res, ae_out), archi
        code_model = model.inter(enc_model)
        bgr, mask = model.decoder_src(code_model)
    else:
        ab_out = to_fmt(model.inter_AB(enc_model), "NHWC", model.model_data_format)
        b_out = to_fmt(model.inter_B(enc_model), "NHWC", model.model_data_format)
        low = res // 32 if "d" in opts else res // 16
        i_res = low if "t" in opts else low * 2
        ae_out = model.options["ae_dims"] * 2
        assert tuple(ab_out.shape) == (2, i_res, i_res, ae_out), archi
        assert tuple(b_out.shape) == (2, i_res, i_res, ae_out), archi
        code_model = torch.concat(
            [model.inter_B(enc_model), model.inter_AB(enc_model)],
            dim=dfl_nn.conv2d_ch_axis)
        bgr, mask = model.decoder(code_model)

    # the 'd' modifier doubles the INTERNAL code density (lowest_dense_res
    # res//32 instead of res//16) while the depth_to_space x-head brings
    # the BGR/mask pair back to the input resolution — the output comes
    # out at `res` for every archi variant (the official '-d' semantics:
    # "doubling the resolution using the same computation cost")
    bgr = to_fmt(bgr, "NHWC", model.model_data_format)
    mask = to_fmt(mask, "NHWC", model.model_data_format)
    out_res = res
    assert tuple(bgr.shape) == (2, out_res, out_res, 3), archi
    assert tuple(mask.shape) == (2, out_res, out_res, 1), archi
    assert torch.isfinite(bgr).all()
    assert torch.isfinite(mask).all()


@pytest.mark.parametrize("archi,res", [
    ("df", 64), ("df", 256), ("df-udt", 256), ("liae-udt", 64),
])
def test_forward_shapes_other_resolutions_cpu(plain_tmp, archi, res):
    model = construct(SAEHDHeadless,
                      Path(plain_tmp) / f"{archi}_{res}",
                      is_training=False,
                      seed=seed(archi=archi, resolution=res),
                      cpu_only=True)
    x = torch.rand(1, res, res, 3)
    x = to_fmt(x, model.model_data_format, "NHWC")
    # flat encoder output (2D code — no data-format conversion)
    e_res = res // 32 if "t" in _archi_opts(archi) else res // 16
    enc = model.encoder(x)
    assert tuple(enc.shape) == (1, model.options["e_dims"] * 8
                                * e_res * e_res)
    # the full decoder round trip: every archi variant (the 'd' head
    # included — it doubles the half-res inter map back to full res)
    # outputs at the input resolution
    enc_x = model.encoder(x)
    if model.archi_type == "df":
        code = model.inter(enc_x)
        bgr, mask = model.decoder_src(code)
    else:
        # the liae decoder consumes the concatenated inter codes
        code = torch.concat([model.inter_B(enc_x), model.inter_AB(enc_x)],
                            dim=dfl_nn.conv2d_ch_axis)
        bgr, mask = model.decoder(code)
    bgr = to_fmt(bgr, "NHWC", model.model_data_format)
    out_res = res
    assert tuple(bgr.shape) == (1, out_res, out_res, 3)


def test_decoder_dim_propagation_cpu(plain_tmp):
    # d_dims / d_mask_dims reach the decoder conv stacks. The Upscale
    # layers keep their conv weights with the x4 out-channel expansion
    # (the official depth_to_space factor is folded out of the weight),
    # so the first up-conv of a branch with base channel count b
    # carries weight out-ch 32*b — a value the other branch of these
    # configs cannot produce (b=16 -> 512, b=32 -> 1024).
    model = construct(SAEHDHeadless, Path(plain_tmp) / "dm",
                      is_training=False,
                      seed=seed(archi="df", d_dims=16, d_mask_dims=32),
                      cpu_only=True)
    mask_outs = [w.shape[-1] for w in model.decoder_src.get_weights()]
    assert 32 * 32 in mask_outs, \
        "d_mask_dims must reach the mask conv stack"

    model = construct(SAEHDHeadless, Path(plain_tmp) / "dd",
                      is_training=False,
                      seed=seed(archi="df", d_dims=32, d_mask_dims=16),
                      cpu_only=True)
    bgr_outs = [w.shape[-1] for w in model.decoder_src.get_weights()]
    assert 32 * 32 in bgr_outs, "d_dims must reach the BGR conv stack"


def test_discriminator_shapes_cpu(plain_tmp):
    res = 128
    patch = 16
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1, gan_patch_size=patch),
                      debug=True, cpu_only=True)
    fmt = model.model_data_format

    # D_src: (center_out, x) — both 1-channel, x at the input size
    x = torch.rand(2, patch, patch, 3)
    x = to_fmt(x, fmt, "NHWC")
    center_out, x_out = model.D_src(x)
    center_out = to_fmt(center_out, "NHWC", fmt)
    x_out = to_fmt(x_out, "NHWC", fmt)
    assert center_out.shape[-1] == 1
    assert x_out.shape[-1] == 1
    assert tuple(x_out.shape) == (2, patch, patch, 1)
    assert torch.isfinite(center_out).all()

    # CodeDiscriminator: inter code -> downsampled single-channel map.
    # Phase 3F contract (official discriminators_tf.py L29, reproduced
    # by the Phase 3F port): 1 + code_res//8 stride-2 convs — the first
    # with kernel 4, the rest with kernel 3 — using the official DFL
    # SAME->int conversion (Phase 3B Conv2D: explicit symmetric padding
    # ((k-1)*d+1)//2 = k//2 per side, then VALID; even-kernel stride-2
    # convs map x -> x/2+1, which is the official behavior, not TF true
    # SAME), then the 1x1 VALID out_conv. Walk that exact chain: at
    # code_res 16 (res 128) it lands on 3x3, at code_res 32 (res 256)
    # on 2x2.
    code_res = model.inter.get_out_res()
    code = torch.rand(2, code_res, code_res, model.options["ae_dims"])
    code = to_fmt(code, fmt, "NHWC")
    dis_out = to_fmt(model.code_discriminator(code), "NHWC", fmt)
    out_res = code_res
    for i in range(1 + code_res // 8):
        k = 4 if i == 0 else 3
        out_res = (out_res + 2 * (k // 2) - k) // 2 + 1
    assert tuple(dis_out.shape) == (2, out_res, out_res, 1)


# --- backward readiness (no loss — Phase 6B) -----------------------------------------

def test_backward_reaches_all_components_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1),
                      debug=True, cpu_only=True)
    res = model.resolution
    patch = model.options["gan_patch_size"]

    x_src = torch.rand(2, res, res, 3)
    x_src = to_fmt(x_src, model.model_data_format, "NHWC")

    src_code = model.encoder(x_src)
    inter_code = model.inter(src_code)
    bgr, mask = model.decoder_src(inter_code)
    dis_out = model.code_discriminator(inter_code)
    patch_crop = to_fmt(x_src, "NHWC", model.model_data_format)[:, :patch, :patch, :]
    patch_crop = to_fmt(patch_crop, model.model_data_format, "NHWC")
    center_out, _ = model.D_src(patch_crop)

    loss = inter_code.sum() + bgr.sum() + mask.sum() \
        + dis_out.sum() + center_out.sum()
    loss.backward()

    for comp in (model.encoder, model.inter, model.decoder_src,
                 model.D_src, model.code_discriminator):
        grads = [p.grad for p in comp.parameters() if p.grad is not None]
        assert len(grads) > 0, comp.name
        for g in grads[:3]:
            assert torch.isfinite(g).all(), comp.name
    # decoder_dst is not in this graph: no gradients (strict autograd)
    assert all(p.grad is None for p in model.decoder_dst.parameters())


def test_ae_merge_inference_boundary_cpu(plain_tmp):
    # the official non-training inference closure: NumPy in / NumPy out
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df",
                      is_training=False, seed=seed(archi="df"),
                      cpu_only=True)
    assert hasattr(model, "AE_merge")
    x = np.random.rand(1, 128, 128, 3).astype(np.float32)
    out = model.AE_merge(x)
    assert isinstance(out, list) and len(out) == 3
    bgr, mask, other = out
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == (1, 128, 128, 3)
    assert mask.shape == (1, 128, 128, 1)


def test_predictor_func_and_merger_config_cpu(plain_tmp):
    # the thin official merge-boundary wrappers (plan item 1):
    # predictor_func is a NumPy-in/NumPy-out AE_merge wrapper
    # (bgr + both mask channels, batch squeezed); get_MergerConfig
    # returns the official (func, in_size, MergerConfigMasked) triple
    # (the merger package import is TF-free under the torch venv)
    model = construct(SAEHDHeadless, Path(plain_tmp) / "df",
                      is_training=False, seed=seed(archi="df"),
                      cpu_only=True)
    face = np.random.rand(128, 128, 3).astype(np.float32)
    bgr, mask_src, mask_dst = model.predictor_func(face)
    assert isinstance(bgr, np.ndarray) and bgr.shape == (128, 128, 3)
    assert isinstance(mask_src, np.ndarray) and mask_src.shape == (128, 128)
    assert isinstance(mask_dst, np.ndarray) and mask_dst.shape == (128, 128)

    import merger
    func, in_size, config = model.get_MergerConfig()
    # a bound method of THIS model (bound-method objects are not
    # identity-stable across accesses — compare the underlying function)
    assert getattr(func, "__self__", None) is model
    assert func.__func__ is type(model).predictor_func
    assert in_size == (128, 128, 3)
    assert isinstance(config, merger.MergerConfigMasked)
    assert config.face_type is model.face_type


# --- strict save/load round-trip -------------------------------------------------------

def _round_trip_files(root):
    return [f for f in sorted((Path(root)).iterdir())
            if f.name.startswith(MODEL_NAME + "_") and f.suffix == ".npy"]


def test_save_load_round_trip_exact_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    m1 = construct(SAEHDHeadless, root, is_training=True,
                   seed=seed(archi="df", true_face_power=0.1,
                             gan_power=0.1, lr_dropout="y"),
                   debug=True, cpu_only=True)
    m1.set_iter(1)
    m1.save()

    # the official outer container of every component file
    saved = {f.stem[len(MODEL_NAME) + 1:]: f for f in _round_trip_files(root)}
    for name in ("encoder", "inter", "decoder_src", "decoder_dst",
                 "code_discriminator", "GAN", "src_dst_opt",
                 "D_code_opt", "GAN_opt"):
        assert name in saved, name
        raw = saved[name].read_bytes()
        assert raw[:3] == b"\x80\x04\x95", name
        assert b"\x93NUMPY" not in raw[:64], name
    # data.dat bookkeeping
    data = pickle.loads((root / f"{MODEL_NAME}_data.dat").read_bytes())
    assert data["iter"] == 1
    assert data["options"]["archi"] == "df"
    assert data["options"]["lr_dropout"] == "y"
    assert data["sample_for_preview"] is not None

    # the REAL model resumes from the same directory (is_training=True
    # so its component set — discriminators + optimizers — matches the
    # bootstrap's; the restored sample_for_preview skips re-generation)
    m2 = construct_real(root, options=None, is_training=True,
                        debug=True, cpu_only=True)
    assert m2.is_first_run() is False
    assert m2.get_iter() == 1
    assert m2.options["archi"] == "df"
    assert m2.options["true_face_power"] == 0.1
    assert m2.options["gan_power"] == 0.1

    # every component weight and optimizer state restored EXACT
    for na in ("encoder", "inter", "decoder_src", "decoder_dst",
               "code_discriminator", "D_src",
               "src_dst_opt", "D_code_opt", "D_src_dst_opt"):
        check_weights_exact(getattr(m1, na), getattr(m2, na), na)
    # the summary file is part of the official save
    assert (root / f"{MODEL_NAME}_summary.txt").exists()


def test_save_load_round_trip_liae_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    m1 = construct(SAEHDHeadless, root, is_training=True,
                   seed=seed(archi="liae-udt", lr_dropout="y"),
                   debug=True, cpu_only=True)
    m1.set_iter(1)
    m1.save()
    m2 = construct_real(root, options=None, is_training=True,
                        debug=True, cpu_only=True)
    assert m2.options["archi"] == "liae-udt"
    assert m2.archi_type == "liae"
    for na in ("encoder", "inter_AB", "inter_B", "decoder",
               "src_dst_opt"):
        check_weights_exact(getattr(m1, na), getattr(m2, na), na)


# --- strict failures --------------------------------------------------------------------

def test_missing_required_file_on_resume_fails_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    m = construct(SAEHDHeadless, root, is_training=True,
                  seed=seed(archi="df"), debug=True, cpu_only=True)
    m.set_iter(1)
    m.save()

    victim = root / f"{MODEL_NAME}_inter.npy"
    victim.unlink()
    with pytest.raises(FileNotFoundError,
                       match="required component file missing on resume"):
        construct_real(root, options=None, is_training=False, cpu_only=True)


def test_corrupted_component_file_fails_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    m = construct(SAEHDHeadless, root, is_training=True,
                  seed=seed(archi="df"), debug=True, cpu_only=True)
    m.set_iter(1)
    m.save()

    victim = root / f"{MODEL_NAME}_inter.npy"
    raw = victim.read_bytes()
    victim.write_bytes(raw[: len(raw) // 2])  # truncated pickle
    with pytest.raises(Exception):
        construct_real(root, options=None, is_training=False, cpu_only=True)


def test_malformed_data_dat_fails_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{MODEL_NAME}_data.dat").write_bytes(b"\x80\x04\x95")
    with pytest.raises(Exception):
        construct_real(root, options=None, is_training=False, cpu_only=True)


def test_archi_mismatch_missing_files_fail_cpu(plain_tmp, headless_io):
    # df component files saved, but the stored options say liae: the
    # required liae files are absent -> explicit failure (the official
    # silent re-init is NOT reproduced on resume)
    root = Path(plain_tmp) / "m"
    m = construct(SAEHDHeadless, root, is_training=True,
                  seed=seed(archi="df"), debug=True, cpu_only=True)
    m.set_iter(1)
    m.save()
    data_path = root / f"{MODEL_NAME}_data.dat"
    data = pickle.loads(data_path.read_bytes())
    data["options"]["archi"] = "liae-ud"
    data_path.write_bytes(pickle.dumps(data, 4))

    with pytest.raises(FileNotFoundError,
                       match="required component file missing on resume"):
        construct_real(root, options=None, is_training=False, cpu_only=True)


def test_shape_mismatch_fails_strictly_cpu(plain_tmp, headless_io):
    root = Path(plain_tmp) / "m"
    m = construct(SAEHDHeadless, root, is_training=True,
                  seed=seed(archi="df"), debug=True, cpu_only=True)
    m.set_iter(1)
    m.save()

    victim = root / f"{MODEL_NAME}_inter.npy"
    d = pickle.loads(victim.read_bytes())
    # flatten one weight: same element count, wrong ndim -> the strict
    # shape check must reject it (the official greedy reshape is gone)
    key = next(iter(d))
    d[key] = d[key].reshape(-1)
    victim.write_bytes(pickle.dumps(d, 4))

    with pytest.raises(ckpt.CheckpointLoadError,
                       match="strict load"):
        construct_real(root, options=None, is_training=False, cpu_only=True)


# --- device placement --------------------------------------------------------------------

def test_cpu_only_construction_cpu(plain_tmp):
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=True,
                      seed=seed(archi="df", true_face_power=0.1,
                                gan_power=0.1),
                      debug=True, cpu_only=True)
    assert len(model.device_config.devices) == 0
    assert dfl_nn.device.type == "cpu"
    for comp in (model.encoder, model.inter, model.decoder_src,
                 model.decoder_dst, model.D_src, model.code_discriminator):
        for p in comp.parameters():
            assert p.device.type == "cpu"
    # the optimizer placement kwarg (Phase 3E2 signature parity)
    # resolved to CPU: no GPU devices exist, so vars_on_cpu is True
    # regardless of the stored models_opt_on_gpu
    assert model.device_config.cpu_only is True

    x = torch.rand(1, 128, 128, 3)
    x = to_fmt(x, model.model_data_format, "NHWC")
    out = model.encoder(x)
    assert out.device.type == "cpu"


@requires_gpu
def test_gpu_construction_forward_cpu(plain_tmp):
    dfl_nn.initialize_main_env()
    model = construct(SAEHDHeadless, Path(plain_tmp) / "m",
                      is_training=False, seed=seed(archi="df"),
                      force_gpu_idxs=[0])
    assert model.device_config.devices[0].backend == "cuda"
    for p in model.encoder.parameters():
        assert p.device.type == "cuda"
    x = torch.rand(1, 128, 128, 3, device="cuda")
    x = to_fmt(x, model.model_data_format, "NHWC")
    out = model.encoder(x)
    assert out.device.type == "cuda"
    assert torch.isfinite(out.detach().cpu()).all()


@requires_gpu
def test_gpu_round_trip_exact(plain_tmp, headless_io):
    dfl_nn.initialize_main_env()
    root = Path(plain_tmp) / "m"
    m1 = construct(SAEHDHeadless, root, is_training=True,
                   seed=seed(archi="df", true_face_power=0.1,
                             gan_power=0.1, lr_dropout="y"),
                   debug=True, force_gpu_idxs=[0])
    m1.set_iter(1)
    m1.save()
    for p in m1.encoder.parameters():
        assert p.device.type == "cuda"

    m2 = construct_real(root, options=None, is_training=True, debug=True,
                        force_gpu_idxs=[0])
    assert m2.device_config.devices[0].backend == "cuda"
    for na in ("encoder", "inter", "decoder_src", "decoder_dst",
               "code_discriminator", "D_src",
               "src_dst_opt", "D_code_opt", "D_src_dst_opt"):
        a, b = getattr(m1, na), getattr(m2, na)
        wa = [w.detach().cpu() for w in a.get_weights()]
        wb = [w.detach().cpu() for w in b.get_weights()]
        assert len(wa) == len(wb), na
        for x, y in zip(wa, wb):
            assert torch.equal(x, y), na
    for p in m2.encoder.parameters():
        assert p.device.type == "cuda"


# --- import boundary / hygiene -----------------------------------------------------------

def test_no_tensorflow_import_boundary_cpu():
    # importing the torch SAEHD model (done at module import above)
    # must not pull TensorFlow into the interpreter
    assert "tensorflow" not in sys.modules
    import models.Model_SAEHD  # noqa: F401
    assert "tensorflow" not in sys.modules


def test_no_tf_or_cuda_in_saehd_sources():
    files = [
        REPO_ROOT / "models" / "Model_SAEHD" / "Model.py",
        REPO_ROOT / "models" / "Model_SAEHD" / "__init__.py",
    ]
    for f in files:
        text = f.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("tensorflow"), (f, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").startswith("tensorflow") is False, \
                    (f, node.module)
            elif isinstance(node, ast.Attribute):
                src = ast.unparse(node)
                assert not src.startswith("torch.cuda"), (f, src)
        # (the module docstring documents the device policy and may
        # MENTION torch.cuda as text; the AST checks above cover code)
        assert '"cuda:"' not in text and "'cuda:'" not in text, f
        assert ".cuda(" not in text, f


# --- the optional real official checkpoint validation --------------------------------------

def test_real_official_checkpoint_mapping(plain_tmp, headless_io):
    """Validates a REAL official SAEHD checkpoint (opt-in via
    DFL_TEST_SAEHD_CHECKPOINT; unset -> skip). The files are copied
    into the test temp dir (the private directory is never read in
    place or referenced); the official on-disk prefix may be the bare
    ``SAEHD_`` form or the ``<model>_SAEHD_`` form (e.g. distributed
    checkpoint names) and is renamed to the test model name either
    way. The options come from the checkpoint's own ``data.dat``
    (documented test-only substitution face_type 'wf'->'f'; the
    fixture facesets carry full-face 2DFAN landmarks, not XSeg masks).
    The real SAEHDModel resumes in the TRAINING context (the GAN
    discriminator + optimizer-state files are only registered — and
    thus strictly loaded in-place — on the training path; the packed
    faceset fixture provides the workable sample data), and every
    component/optimizer file is additionally re-validated through the
    Phase 4 conversion engine: zero missing-REQUIRED keys, zero
    shape/dtype mismatches are hard gates; unexpected extra (legacy)
    keys are reported, not failed. Structural mapping only:
    REAL CHECKPOINT STRUCTURAL MAPPING TESTED — TRAINING PARITY
    NOT_VERIFIED (Phase 6B)."""
    path = os.environ.get("DFL_TEST_SAEHD_CHECKPOINT")
    if path is None:
        pytest.skip("DFL_TEST_SAEHD_CHECKPOINT not set")
    src = Path(path)
    assert src.is_dir(), path

    root = Path(plain_tmp) / "m"
    root.mkdir(parents=True, exist_ok=True)

    # copy the SAEHD files; rename the official prefix to the test
    # model name. Both on-disk layouts are accepted: the bare
    # "SAEHD_*" form and the official "<model>_SAEHD_*" prefix form;
    # non-SAEHD artifacts in the directory (other models) are ignored.
    for f in src.iterdir():
        if not f.is_file():
            continue
        m = _SAEHDFILE_RE.match(f.name)
        if m is None:
            continue
        rest = m.group("rest")
        if rest == "data.dat":
            shutil.copy(f, root / f"{MODEL_NAME}_data.dat")
        elif rest == "default_options.dat":
            shutil.copy(f, root / f"{MODEL_NAME}_default_options.dat")
        elif rest.endswith(".npy"):
            shutil.copy(f, root / f"{MODEL_NAME}_{rest}")
        # everything else (summary.txt, ...) is not part of the load

    # options from the checkpoint's own data.dat + the documented
    # test-only face_type substitution (the fixture facesets carry
    # full-face landmarks, not XSeg masks)
    data = pickle.loads((root / f"{MODEL_NAME}_data.dat").read_bytes())
    opts = data["options"]
    if opts.get("face_type") == "wf":
        opts["face_type"] = "f"
    preseed_data_dat(root, opts, it=max(int(data.get("iter", 0)), 1),
                     sample_for_preview=None)

    # training context + in-process sample generation: registers the
    # GAN discriminator + optimizer-state files so the strict resume
    # load applies them in-place too (component-file-only mapping via
    # is_training=False would leave those files unregistered)
    model = construct_real(root, options=None, is_training=True,
                           debug=True, cpu_only=True)
    print(f"REAL CHECKPOINT STRUCTURAL MAPPING: resumed "
          f"archi={model.options.get('archi')} "
          f"res={model.options.get('resolution')} "
          f"components={[comp.name for comp, _ in model.model_filename_list]}")

    # per-file conversion-engine validation (independent of the
    # in-place strict load the resume already performed). The mapping
    # is FILE-driven: the D_src component is saved under the official
    # file name "GAN.npy", so component NAMES cannot key the table.
    file_to_component = {f[:-len(".npy")]: comp
                         for comp, f in model.model_filename_list}
    # optimizer-state files: the state keys are name-driven (ms_/vs_
    # embed the owning component's official parameter names) and
    # resolve through the parameter bindings; ``saveable`` cross-
    # validates the reference (None = several components)
    optimizer_saveables = {"src_dst_opt": None,
                           "D_code_opt": getattr(model,
                                                 "code_discriminator",
                                                 None),
                           "GAN_opt": getattr(model, "D_src", None)}
    for f in sorted(root.glob(f"{MODEL_NAME}_*.npy")):
        suffix = f.name[len(MODEL_NAME) + 1:]
        key = suffix[: -len(".npy")]
        comp = file_to_component.get(key)
        if comp is None:
            # a file the migrated model does not register (e.g. an
            # official-only bookkeeping artifact) is not part of the
            # structural mapping under test
            print(f"REAL CHECKPOINT: skipping unregistered file {suffix}")
            continue
        d = convert.read_official_checkpoint(f)
        if key in optimizer_saveables:
            report = convert.convert_optimizer_state_official_to_torch(
                comp, d, saveable=optimizer_saveables[key], component=key)
        else:
            report = convert.convert_official_to_torch(comp, d,
                                                       component=comp.name)
        # hard gates: zero REQUIRED mapping errors (a missing required
        # key/state, a shape or dtype mismatch). Unexpected extra
        # (legacy) keys are reported, not failed.
        assert report.missing_required_count == 0, \
            f"{suffix}: {report.errors}"
        assert report.shape_mismatch_count == 0, \
            f"{suffix}: {report.errors}"
        assert report.dtype_mismatch_count == 0, \
            f"{suffix}: {report.errors}"
        if report.unmapped_source_count:
            print(f"REAL CHECKPOINT: {suffix}: {report.unmapped_source_count} "
                  f"unmapped extra (legacy) keys (reported): "
                  f"{report.errors[:5]}")
    print("REAL CHECKPOINT STRUCTURAL MAPPING TESTED: PASS "
          "(TRAINING PARITY NOT_VERIFIED)")


def test_real_faceset_sample_generator_cpu(plain_tmp, headless_io):
    # OPTIONAL real-artifact validation (gated by
    # DFL_TEST_FACESET_PAK = path to an official faceset.pak or its
    # parent directory; unset/absent -> skip, never hardcoded). The
    # full sample-generator lifecycle on REAL data: real images +
    # landmarks through SampleLoader/PackedFaceset, the in-process
    # ThisThreadGenerator under the DSH sandbox, the uniform-yaw
    # index host, and the SampleProcessor output contract (warped /
    # unwarped BGR + FULL_FACE / EYES_MOUTH masks — exercising the
    # LandmarksProcessor mask paths on real landmark sets).
    path = os.environ.get("DFL_TEST_FACESET_PAK")
    if not path or not Path(path).exists():
        pytest.skip("DFL_TEST_FACESET_PAK not set (real faceset)")
    pak_file = Path(path)
    if pak_file.is_dir():
        pak_file = pak_file / "faceset.pak"
        if not pak_file.exists():
            pytest.skip("no faceset.pak under DFL_TEST_FACESET_PAK dir")

    # face type of the real faceset -> official option string; head
    # requires XSeg (out of Phase 6A scope)
    from facelib import FaceType
    from samplelib import PackedFaceset
    samples = PackedFaceset.load(pak_file.parent)
    if not samples:
        pytest.skip("real faceset is empty")
    ft_of = {FaceType.HALF: "h", FaceType.MID_FULL: "mf",
             FaceType.FULL: "f", FaceType.WHOLE_FACE: "wf",
             FaceType.HEAD: "head"}
    ft = ft_of.get(samples[0].face_type)
    if ft is None or ft == "head":
        pytest.skip("real faceset face type not usable by SAEHD here")

    root = Path(plain_tmp) / "real"
    for side in ("src", "dst"):
        (root / side).mkdir(parents=True)
        shutil.copy2(pak_file, root / side / "faceset.pak")

    # raw lifecycle construction: make_model does NOT (re)create the
    # synthetic training dirs — the real facesets above are the data
    model = make_saehd(SAEHDHeadless, root, is_training=True,
                       seed=seed(archi="df", face_type=ft,
                                 batch_size=1, uniform_yaw=True),
                       debug=True, cpu_only=True)
    gens = model.get_training_data_generators()
    assert len(gens) == 2
    res = int(model.options["resolution"])  # seed default 128
    batch = next(gens[0])
    assert len(batch) == 4
    warped, unwarped, mask_full, mask_em = batch
    assert warped.shape == (1, res, res, 3)
    assert unwarped.shape == (1, res, res, 3)
    assert mask_full.shape == (1, res, res, 1)
    assert mask_em.shape == (1, res, res, 1)
    # real content: the BGR frame and both masks carry signal
    assert float(np.max(np.asarray(warped))) > 0.0
    assert float(np.max(np.asarray(mask_full))) > 0.0
    assert float(np.max(np.asarray(mask_em))) > 0.0
    # the dst generator pair produces the same output contract
    assert len(next(gens[1])) == 4
    print(f"REAL FACESET SAMPLE GENERATOR LIFECYCLE: PASS "
          f"({len(samples)} real samples, face_type={ft}, res={res})")
