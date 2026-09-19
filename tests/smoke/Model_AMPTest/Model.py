"""Test-only AMP bootstrap package for the Phase 7 test suite.

NOT production code: it exists so the headless smoke tests can drive
the REAL ``models.Model_AMP.AMPModel`` through the official Phase 5
lifecycle without any interactive prompt (the same role
``tests/smoke/Model_Dummy/`` and ``tests/smoke/Model_SAEHDTest/`` play
for the Phase 5 / Phase 6A-6B test suites).

Two vehicles, both defined here (never imported by production code):

- ``AMPHeadless``: an ``AMPModel`` subclass that overrides ONLY
  ``on_initialize_options`` (the interactive option-prompt layer) with
  a deterministic direct seed of ``self.options``. Everything else —
  ``on_initialize`` (component construction through the Phase 7
  ``nn.AMPArchi`` factory, the official two-optimizer construction,
  ``model_filename_list``, the official load/init loop, the
  sample-generator wiring, the Phase 7 train closures) and
  ``onSave`` / ``get_model_filename_list`` / ``predictor_func`` /
  ``get_MergerConfig`` / the ``export_dfm`` deferral stub — is the
  real production code. The class attribute ``test_seed_options``
  (set per-test, ``None`` = built-in default seed) controls the
  seeded option set.
- ``make_training_dirs``: re-exported from the ``Model_SAEHDTest``
  package — the deterministic packed-faceset builder is model-agnostic
  (the official ``PackedFaceset`` VERSION-1 layout with two tiny FULL
  ('f') face samples).

Construction notes (mirroring the Phase 5/6A dummy pattern):

- ``make_model(..., cpu_only=True)`` / ``force_gpu_idxs=[0]`` selects
  the device through the normal lifecycle (the constructor's own
  ``nn.initialize``); tests must not pre-initialize ``nn``.
- ``is_training=True`` constructions should pass ``debug=True`` so
  the model's ``SampleGeneratorFace`` uses the in-process
  ``ThisThreadGenerator`` (no multiprocessing subprocess spawn under
  pytest). Note: ``debug=True`` forces the sample generator to
  batch_size 1 (the official SampleGeneratorBase behavior), so the
  onTrainOneIter driver runs N=1 batches; the train closures
  themselves accept any N and are driven with synthetic batches of
  N > 1 directly in the tests (the Q11 batch-sum pin).
- The real ``AMPModel`` is used directly (not this subclass) for
  resume-mode tests that must exercise the real
  ``on_initialize_options`` headlessly (stored options are restored
  from a pre-seeded ``*_data.dat`` with ``iter >= 1``; no first-run
  prompts; ``ask_override`` is a bounded non-blocking poll).
"""

import sys
from pathlib import Path

_SMOKE_DIR = Path(__file__).resolve().parents[1]
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

from Model_SAEHDTest.Model import (  # noqa: E402
    make_packed_faceset,
    make_training_dirs,
)

from models.Model_AMP import Model as AMPModel  # noqa: E402


# --- the test-only headless AMP model --------------------------------------

#: default seed (used when ``test_seed_options is None``): the tiny
#: valid AMP configuration for fast CPU structural / training tests
#: (64px, batch 1, the smallest practical dims; d_mask_dims is the
#: official derived default for d_dims=16: 16//3=5 rounded even -> 6)
DEFAULT_SEED_OPTIONS = {
    'resolution': 64,
    'face_type': 'f',
    'models_opt_on_gpu': True,
    'ae_dims': 32,
    'inter_dims': 32,
    'e_dims': 16,
    'd_dims': 16,
    'd_mask_dims': 6,
    'morph_factor': 0.5,
    'uniform_yaw': False,
    'blur_out_mask': False,
    'lr_dropout': 'n',
    'random_warp': True,
    'ct_mode': 'none',
    'clipgrad': False,
    'gan_power': 0.0,
    'gan_patch_size': 8,
    'gan_dims': 16,
    'batch_size': 1,
    'random_src_flip': False,
    'random_dst_flip': True,
}


class AMPHeadless(AMPModel):
    """Test-only ``AMPModel``: the official interactive
    ``on_initialize_options`` prompt layer is replaced by a direct
    deterministic seed of ``self.options``; the REAL
    ``on_initialize`` (and everything else) runs unchanged.

    Per-test seed selection: set the class attribute
    ``test_seed_options`` to an options dict (merged over
    ``DEFAULT_SEED_OPTIONS``) or ``None`` for the default. The test
    seed WINS over the lifecycle pre-loads (``self.options.update``).
    ``test_seed_options`` is a test-only hook and is never defined on
    the production ``AMPModel``.
    """

    #: test-only option seed (None = DEFAULT_SEED_OPTIONS)
    test_seed_options = None

    def on_initialize_options(self):
        user_seed = self.test_seed_options
        if user_seed is None:
            merged = DEFAULT_SEED_OPTIONS
        else:
            merged = dict(DEFAULT_SEED_OPTIONS)
            merged.update(user_seed)
        # the test seed WINS over the lifecycle pre-loads: the
        # lifecycle loads batch_size into options before
        # on_initialize_options (ModelBase), so a "fill missing
        # only" merge would silently keep the lifecycle value
        self.options.update(merged)
        # the real on_initialize_options computes this from the
        # default-options comparison; a test seed has no stored
        # defaults, so the detection flag stays False unless the
        # test sets it explicitly on the instance
        self.gan_model_changed = False


def make_model(model_class, tmpdir, is_training=False, seed=None,
               debug=False, **device_kwargs):
    """Construct ``model_class`` under ``tmpdir`` through the official
    lifecycle constructor (``cpu_only=True`` / ``force_gpu_idxs=[0]``
    select the device; the lifecycle's own ``nn.initialize`` runs —
    tests must not pre-initialize ``nn``). ``debug=True`` is required
    for ``is_training=True`` constructions (in-process
    ``ThisThreadGenerator``; see the module docstring). The ``seed``
    is only applied when ``model_class`` owns the test-only
    ``test_seed_options`` hook (``AMPHeadless``); it is never written
    onto the production ``AMPModel`` class.
    """
    if hasattr(model_class, 'test_seed_options'):
        model_class.test_seed_options = None if seed is None else dict(seed)
    root = Path(tmpdir)
    root.mkdir(parents=True, exist_ok=True)
    return model_class(
        is_training=is_training,
        saved_models_path=root,
        training_data_src_path=root / 'src',
        training_data_dst_path=root / 'dst',
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name='test_AMP',
        debug=debug,
        **device_kwargs,
    )


class AMPModelGanChanged(AMPModel):
    """Test-only ``AMPModel``: runs the REAL
    ``on_initialize_options`` (the resume-mode stored-options
    restore, headless: no first-run prompts, ``ask_override`` is a
    bounded non-blocking poll) and then forces the official
    ``gan_model_changed`` detection flag — a headless resume can
    never trigger the official L101 stored-default-vs-reprompted
    comparison on its own, and in-test subclassing is impossible
    (the official ``model_class_name`` derivation requires the
    class's module directory name to contain an underscore — this
    package directory 'Model_AMPTest' satisfies it; the derived
    class name 'AMPTest' only affects the first-run
    default-options snapshot name, which a resume never touches).
    The re-init rule under test is the official load-loop
    behavior (GAN + GAN_opt re-initialized on the changed flag,
    every other component strictly loaded)."""

    def on_initialize_options(self):
        super().on_initialize_options()
        self.gan_model_changed = True


__all__ = [
    'DEFAULT_SEED_OPTIONS',
    'AMPHeadless',
    'AMPModelGanChanged',
    'make_packed_faceset',
    'make_training_dirs',
    'make_model',
]
