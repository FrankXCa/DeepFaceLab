"""Test-only SAEHD bootstrap package for the Phase 6A test suite.

NOT production code: it exists so the headless smoke tests can drive the
REAL ``models.Model_SAEHD.SAEHDModel`` through the official Phase 5
lifecycle without any interactive prompt (the same role
``tests/smoke/Model_Dummy/`` plays for the Phase 5 lifecycle tests).

Two vehicles, both defined here (never imported by production code):

- ``SAEHDHeadless``: a ``SAEHDModel`` subclass that overrides ONLY
  ``on_initialize_options`` (the interactive option-prompt layer) with a
  deterministic direct seed of ``self.options``. Everything else —
  ``on_initialize`` (archi parsing + validation, component construction
  through ``nn.DeepFakeArchi``, the official discriminator creation
  rules, the three official optimizers with the official weight lists,
  ``model_filename_list``, the official 637-657 load/init loop, the
  sample-generator wiring), ``onSave`` / ``get_model_filename_list`` /
  ``predictor_func`` / ``get_MergerConfig`` and the Phase 6B/11
  deferral stubs — is the real production code. The class attribute
  ``test_seed_options`` (set per-test, ``None`` = built-in default seed)
  controls the seeded option set.
- ``make_packed_faceset`` / ``make_training_dirs``: build a
  deterministic test-only packed faceset (the official
  ``PackedFaceset`` VERSION-1 ``faceset.pak`` layout) with two tiny
  FULL (``'f'``) face samples — 68 deterministic landmarks (jaw
  ellipse, brows, nose, eyes, outer/inner lips) and a decodable
  PNG — so an ``is_training=True`` construction gets workable
  ``SampleGeneratorFace`` data with NO XSeg/TF dependency. It exists
  only to allow structural training-context construction; the tests
  never assert sample-VALUE parity.

Construction notes (mirroring the Phase 5 dummy pattern):

- ``make_model(..., cpu_only=True)`` / ``force_gpu_idxs=[0]`` selects the
  device through the normal lifecycle (the constructor's own
  ``nn.initialize``); tests must not pre-initialize ``nn``.
- ``is_training=True`` constructions should pass ``debug=True`` so the
  model's ``SampleGeneratorFace`` uses the in-process
  ``ThisThreadGenerator`` (no multiprocessing subprocess spawn under
  pytest). Side effect: ``model_data_format`` is NHWC on GPU as well
  (a structural no-op — components are device-placed the same way).
- The real ``SAEHDModel`` is used directly (not this subclass) for
  resume-mode tests that must exercise the real
  ``on_initialize_options`` headlessly (stored options are restored
  from a pre-seeded ``*_data.dat`` with ``iter >= 1``; no first-run
  prompts; ``ask_override`` is a bounded 2s non-blocking poll).
"""

import math
import pickle
import struct
from pathlib import Path

import cv2
import numpy as np

from facelib import FaceType
from models.Model_SAEHD import Model as SAEHDModel
from samplelib import Sample, SampleType


# --- deterministic test faceset fixture ------------------------------------

def _face_landmarks(size, variant=0):
    """A deterministic 68-point landmark set (the official 2DFAN layout:
    jaw 0-16, brows 17-28, nose 29-34, eyes 35-47, outer lips 48-59,
    inner lips 60-67) for a ``size``x``size`` image. Every mask group
    used by ``facelib.LandmarksProcessor`` (hull/eye/mouth convex hulls)
    is non-degenerate (>= 3 distinct, non-collinear points), so the
    ``SampleProcessor`` mask path works headlessly. ``variant`` shifts
    the face slightly so two samples are distinct."""
    s = float(size)
    dx = 0.0 if variant == 0 else 0.02 * s
    pts = []
    # jaw contour: 17 points along a half-ellipse, right temple -> chin
    # -> left temple
    for i in range(17):
        t = i / 16.0
        ang = math.pi * (1.0 - t)
        pts.append([0.5 * s + dx + 0.28 * s * math.cos(ang),
                    0.78 * s - 0.30 * s * math.sin(ang)])
    # right brow 17-21, left brow 22-26, nose bridge top 27
    for i in range(5):
        pts.append([0.30 * s + dx + 0.02 * s * i,
                    0.36 * s - 0.015 * s * math.sin(math.pi * i / 4.0)])
    for i in range(5):
        pts.append([0.70 * s + dx - 0.02 * s * i,
                    0.36 * s - 0.015 * s * math.sin(math.pi * i / 4.0)])
    pts.append([0.50 * s + dx, 0.38 * s])
    # nose: ridge 28-30, base 31-35
    pts.append([0.50 * s + dx, 0.44 * s])
    pts.append([0.50 * s + dx, 0.50 * s])
    pts.append([0.50 * s + dx, 0.55 * s])
    pts.append([0.42 * s + dx, 0.56 * s])
    pts.append([0.465 * s + dx, 0.585 * s])
    pts.append([0.50 * s + dx, 0.59 * s])
    pts.append([0.535 * s + dx, 0.585 * s])
    pts.append([0.58 * s + dx, 0.56 * s])
    # right eye 36-41, left eye 42-47 (6-point arcs)
    pts.append([0.30 * s + dx, 0.43 * s])
    pts.append([0.34 * s + dx, 0.41 * s])
    pts.append([0.375 * s + dx, 0.415 * s])
    pts.append([0.41 * s + dx, 0.435 * s])
    pts.append([0.375 * s + dx, 0.45 * s])
    pts.append([0.34 * s + dx, 0.455 * s])
    pts.append([0.59 * s + dx, 0.435 * s])
    pts.append([0.625 * s + dx, 0.415 * s])
    pts.append([0.66 * s + dx, 0.41 * s])
    pts.append([0.70 * s + dx, 0.43 * s])
    pts.append([0.66 * s + dx, 0.455 * s])
    pts.append([0.625 * s + dx, 0.45 * s])
    # outer lips 48-59
    pts.append([0.37 * s + dx, 0.67 * s])
    pts.append([0.43 * s + dx, 0.65 * s])
    pts.append([0.465 * s + dx, 0.655 * s])
    pts.append([0.50 * s + dx, 0.65 * s])
    pts.append([0.535 * s + dx, 0.655 * s])
    pts.append([0.57 * s + dx, 0.65 * s])
    pts.append([0.63 * s + dx, 0.67 * s])
    pts.append([0.57 * s + dx, 0.715 * s])
    pts.append([0.535 * s + dx, 0.74 * s])
    pts.append([0.50 * s + dx, 0.745 * s])
    pts.append([0.465 * s + dx, 0.74 * s])
    pts.append([0.43 * s + dx, 0.715 * s])
    # inner lips 60-67
    pts.append([0.435 * s + dx, 0.675 * s])
    pts.append([0.475 * s + dx, 0.668 * s])
    pts.append([0.50 * s + dx, 0.665 * s])
    pts.append([0.525 * s + dx, 0.668 * s])
    pts.append([0.565 * s + dx, 0.675 * s])
    pts.append([0.525 * s + dx, 0.72 * s])
    pts.append([0.50 * s + dx, 0.725 * s])
    pts.append([0.475 * s + dx, 0.72 * s])
    return np.array(pts, dtype=np.float32)


def _face_image(size, variant=0):
    """Deterministic BGR face-like image (a smooth per-pixel field plus a
    skin-tone face region) — the content is irrelevant to the structural
    tests; it only needs to decode through ``cv2.imdecode`` and warp
    through ``SampleProcessor``."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    v = variant * 0.15
    img = np.stack([
        0.35 + 0.25 * np.sin(x / 37.0) + v,
        0.55 + 0.20 * np.cos(y / 53.0) + v,
        0.45 + 0.25 * np.sin((x + y) / 61.0) + v,
    ], axis=-1)
    # skin-tone face disc so the BGR samples are not a pure gradient
    cx, cy = size * (0.5 + (0.02 if variant else 0.0)), size * 0.55
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size * 0.32)
    face = np.clip(1.0 - r, 0.0, 1.0)[..., None]
    img = img * (1.0 - face) + np.array([0.55, 0.68, 0.78], dtype=np.float32) * face
    return np.clip(img, 0.0, 1.0)


def make_packed_faceset(samples_path, n_samples=2, size=256):
    """Write ``<samples_path>/faceset.pak`` — the official
    ``PackedFaceset`` VERSION-1 layout (8-byte version + 8-byte config
    length + protocol-4 pickled sample configs + 8*(n+1) offset table +
    raw PNG bytes) with ``n_samples`` deterministic FULL ('f') face
    samples. Test-only: it builds samplelib contract data, it does not
    modify or copy anything from a real faceset."""
    samples_path = Path(samples_path)
    samples_path.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(n_samples):
        samples.append(Sample(
            sample_type=SampleType.FACE,
            filename='face_%d.png' % i,
            face_type=FaceType.FULL,
            shape=(size, size, 3),
            landmarks=_face_landmarks(size, variant=i),
        ))

    samples_configs = [s.get_config() for s in samples]
    samples_bytes = pickle.dumps(samples_configs, 4)

    of = open(samples_path / 'faceset.pak', 'wb')
    of.write(struct.pack('Q', 1))  # PackedFaceset.VERSION
    of.write(struct.pack('Q', len(samples_bytes)))
    of.write(samples_bytes)
    data_table_offset = of.tell()
    of.write(bytes(8 * (len(samples) + 1)))
    data_start_offset = of.tell()
    offsets = []
    for i in range(len(samples)):
        ok, png = cv2.imencode('.png', _face_image(size, variant=i))
        if not ok:
            raise RuntimeError('test fixture: PNG encoding failed')
        offsets.append(of.tell() - data_start_offset)
        of.write(bytes(png))
    offsets.append(of.tell())
    of.seek(data_table_offset, 0)
    for offset in offsets:
        of.write(struct.pack('Q', offset))
    of.close()


def make_training_dirs(tmpdir):
    """Build deterministic packed facesets for the src and dst training
    data directories of a model under ``tmpdir`` (the layout
    ``make_model`` passes to the lifecycle constructor)."""
    root = Path(tmpdir)
    make_packed_faceset(root / 'src')
    make_packed_faceset(root / 'dst')


# --- the test-only headless SAEHD model ------------------------------------

#: default seed (used when ``test_seed_options is None``): a small valid
#: liae-ud configuration for fast CPU structural tests
DEFAULT_SEED_OPTIONS = {
    'resolution': 128,
    'face_type': 'f',
    'models_opt_on_gpu': True,
    'archi': 'liae-ud',
    'ae_dims': 32,
    'e_dims': 16,
    'd_dims': 16,
    'd_mask_dims': 16,
    'masked_training': True,
    'eyes_mouth_prio': False,
    'uniform_yaw': False,
    'blur_out_mask': False,
    'adabelief': True,
    'lr_dropout': 'n',
    'random_warp': True,
    'random_hsv_power': 0.0,
    'true_face_power': 0.0,
    'face_style_power': 0.0,
    'bg_style_power': 0.0,
    'ct_mode': 'none',
    'clipgrad': False,
    'pretrain': False,
    'gan_power': 0.0,
    'gan_patch_size': 16,
    'gan_dims': 16,
    'batch_size': 2,
    'random_src_flip': False,
    'random_dst_flip': True,
}


class SAEHDHeadless(SAEHDModel):
    """Test-only ``SAEHDModel``: the official interactive
    ``on_initialize_options`` prompt layer is replaced by a direct
    deterministic seed of ``self.options``; the REAL ``on_initialize``
    (and everything else) runs unchanged.

    Per-test seed selection: set the class attribute
    ``test_seed_options`` to an options dict (merged over
    ``DEFAULT_SEED_OPTIONS``) or ``None`` for the default. Keys already
    present in ``self.options`` (restored from ``*_data.dat`` on a
    resume) are never overwritten. ``test_seed_options`` is a
    test-only hook and is never defined on the production
    ``SAEHDModel``.
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
        # on_initialize_options (ModelBase L220), so a "fill missing
        # only" merge would silently keep the lifecycle value
        self.options.update(merged)
        # the real on_initialize_options computes these from the
        # default-options comparison; a test seed has no stored
        # defaults, so both detection flags stay False unless the test
        # sets them explicitly on the instance after construction
        self.gan_model_changed = False
        self.pretrain_just_disabled = False
        self.pretrain = self.options['pretrain']


def make_model(model_class, tmpdir, is_training=False, seed=None,
               debug=False, **device_kwargs):
    """Construct ``model_class`` under ``tmpdir`` through the official
    lifecycle constructor (``cpu_only=True`` / ``force_gpu_idxs=[0]``
    select the device; the lifecycle's own ``nn.initialize`` runs —
    tests must not pre-initialize ``nn``). ``debug=True`` is required
    for ``is_training=True`` constructions (in-process
    ``ThisThreadGenerator``; see the module docstring). The ``seed`` is
    only applied when ``model_class`` owns the test-only
    ``test_seed_options`` hook (``SAEHDHeadless``); it is never written
    onto the production ``SAEHDModel`` class.
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
        force_model_class_name='test_SAEHD',
        debug=debug,
        **device_kwargs,
    )
