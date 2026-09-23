"""Phase 10D acceptance: classic XSeg editor compatibility.

The classic PyQt5 XSeg editor (``XSegEditor/``) is intentionally NOT
rewritten by the modernization: it consumes DFLIMG XSeg masks
(``get_xseg_mask``) and polygon labels (``get_seg_ie_polys``) written by
the extractor, ``xseg apply`` (the torch XSeg inference), and the editor
itself — and it must import and run against the Torch foundation with
NO TensorFlow anywhere in its import chain.  This file verifies:

- static import scan: the editor package (``XSegEditor/*.py``) imports
  no TensorFlow/keras and no model-runtime code (no ``facelib`` /
  ``core.leras`` — the editor works on saved DFLIMG metadata only); the
  restored official ``core/qtex/`` Qt helper package (byte-identical
  dead reference, required by ``from core.qtex import *``) is likewise
  TF-free;
- live import: ``XSegEditor.XSegEditor`` imports cleanly in this
  (Torch, headless) environment.  PyQt5 is not a runtime dependency of
  the modernized tree (the headless CLI is the product; the GUI toolkit
  is an optional add-on), so when PyQt5 is absent the test installs a
  minimal PyQt5 stand-in (dummy Qt classes) solely to exercise the full
  non-Qt import chain — the assertion is that the chain completes with
  ZERO tensorflow/keras modules loaded.  With real PyQt5 installed the
  same import runs against it;
- DFLIMG XSeg mask metadata round-trips: binary masks are stored on the
  lossless PNG path (buffer magic checked) and come back value-exact;
  fractional masks round-trip within the 8-bit quantization bound
  (1/255); ``set_xseg_mask(None)`` (the ``remove_xseg`` write path)
  clears the metadata; masks coexist with face_type and polygon labels
  through save/reload;
- polygon label compatibility: ``SegIEPolys`` (INCLUDE/EXCLUDE
  polygons) round-trip value-exact through ``set_seg_ie_polys`` /
  save / ``get_seg_ie_polys`` (the editor's read path), the legacy
  list-form load (old facesets) still works, and ``overlay_mask``
  paints include/exclude deterministically (the editor's label
  preview);
- CLI workflow contracts (``mainscripts/XSegUtil.py`` — the same
  functions ``main.py xseg ...`` dispatches): ``remove_xseg`` clears
  masks without touching labels/other metadata and skips non-DFLIMG
  files; ``remove_xseg_labels`` clears labels without touching masks;
  ``fetch_xseg`` copies exactly the labeled faces into
  ``<input>_xseg/`` and (with the delete prompt answered "no") keeps
  the originals;
- ``apply_xseg`` end-to-end on a deterministic Torch XSeg checkpoint:
  the written mask is binary float32 (256,256,1) in [0,1] — the exact
  shape/dtype/contract the editor reads via ``get_xseg_mask`` — other
  metadata (face_type) survives, the run is deterministic (two runs
  with the same weights produce identical masks), and the
  apply -> remove loop closes.  The Torch foundation keeps the process
  TensorFlow-free throughout.

Manual GUI smoke requirement (documented, NOT automated — GUI
automation is impractical and faking it is forbidden):
  1. In a venv with PyQt5 installed: ``python main.py xseg
     editor --input-dir <faceset>`` on a faceset that went through
     ``xseg apply``; the trained mask must render in VIEW_XSEG /
     VIEW_XSEG_OVERLAY mode and labeled polygons must render in
     VIEW_BAKED mode.
  2. Draw one INCLUDE polygon, save (the editor writes the DFLIMG
     metadata itself); re-open — the polygon persists; run
     ``xseg fetch`` — the face lands in ``<faceset>_xseg/``.
  Environment note: the torch venv does not ship PyQt5 (optional GUI
  dependency) and the pinned NumPy 2.x baseline does not provide
  ``np.int`` (used once in the restored official ``core/qtex/qtex.py``
  ``QPoint_from_np``); both are GUI-runtime-only gaps, recorded in
  ``docs/PHASE10_STATE.md`` section 10D — the automated coverage above
  exercises everything the editor consumes.

CPU only (the smoke default environment).  RNG state (Python / NumPy /
Torch CPU / Torch CUDA) is preserved by the autouse fixture.
"""

import ast
import gc
import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.interact import interact as io  # noqa: E402
from core.leras import nn as dfl_nn  # noqa: E402
from core.imagelib import SegIEPolys, SegIEPolyType  # noqa: E402
from DFLIMG import DFLIMG  # noqa: E402
from facelib import FaceType, XSegNet  # noqa: E402
from mainscripts import XSegUtil  # noqa: E402

EDITOR_DIR = REPO_ROOT / "XSegEditor"
QTEX_DIR = REPO_ROOT / "core" / "qtex"
RES = 256

# DFLIMG files store face_type as the FaceType.toString() full string
# ('whole_face'); XSeg_data.dat carries the model's SHORT code ('wf') —
# both formats are exercised here, exactly as the production code reads
# them (DFLJPG.get_face_type -> FaceType.fromString; apply_xseg maps
# the dat short code through its own table).
FT_WF = FaceType.toString(FaceType.WHOLE_FACE)
FT_FULL = FaceType.toString(FaceType.FULL)

# TensorFlow modules that must NEVER appear in a process that only
# imported the editor chain (the point of Phase 10 for the editor).
FORBIDDEN_MODULES = ("tensorflow", "keras")


def _forbidden_loaded():
    return sorted(
        m for m in sys.modules
        if m in FORBIDDEN_MODULES or m.startswith("tensorflow.")
        or m.startswith("keras."))


# --- PyQt5 stand-in (only when PyQt5 is not installed) ---------------


class _QtStubMeta(type):
    # class-attribute access (Qt.AlignCenter, QPalette.Window, ...)
    def __getattr__(cls, name):
        return _QtStub


class _QtStub(metaclass=_QtStubMeta):
    # every Qt name the editor chain references at import time resolves
    # to this dummy class: instantiable, attribute-accessible, callable
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _QtStub

    def __call__(self, *args, **kwargs):
        return _QtStub()


_QT_STUB_NAMES = [
    "Qt", "QAction", "QActionGroup", "QApplication", "QBrush",
    "QButtonGroup", "QColor", "QCursor", "QDialog", "QFileDialog",
    "QFont", "QFontDatabase", "QFrame", "QGraphicsOpacityEffect",
    "QGridLayout", "QHBoxLayout", "QIcon", "QImage", "QLabel",
    "QLightEffect", "QLineEdit", "QMenu", "QMessageBox", "QPainter",
    "QPainterPath", "QPalette", "QPen", "QPixmap", "QPoint",
    "QKeySequence", "QProgressBar", "QPushButton", "QRect",
    "QScrollArea", "QSize", "QSizePolicy", "QSlider", "QTabWidget",
    "QThread", "QTimer", "QToolButton", "QVBoxLayout", "QWidget",
    "QGroupBox", "QGuiApplication", "QShortcut", "QStyle",
    "QAbstractButton", "Signal", "Slot",
]


def _install_pyqt5_stub():
    pkg = types.ModuleType("PyQt5")
    pkg.__path__ = []
    for sub in ("QtCore", "QtGui", "QtWidgets"):
        m = types.ModuleType("PyQt5." + sub)
        for name in _QT_STUB_NAMES:
            m.__dict__[name] = _QtStub
        pkg.__dict__[sub] = m
        sys.modules["PyQt5." + sub] = m
    sys.modules["PyQt5"] = pkg
    return [pkg] + [sys.modules["PyQt5." + s]
                    for s in ("QtCore", "QtGui", "QtWidgets")]


@pytest.fixture(autouse=True)
def restore_rng_state():
    """Keep XSeg initialization from changing later smoke-test streams
    (Python / NumPy / Torch CPU / Torch CUDA)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@pytest.fixture
def tmp_root(plain_tmp):
    """A clean per-test subdirectory of the shared ``plain_tmp`` dir
    (recursive teardown, so the fixture's fresh same-name re-creation
    for the next test never collides)."""
    import shutil

    d = Path(plain_tmp) / "work"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def no_cli_prompts(monkeypatch):
    """Answer every XSegUtil interactive prompt deterministically:
    confirmations proceed, the fetch "delete originals?" prompt is
    answered NO (originals are asserted to survive)."""
    monkeypatch.setattr(io, "input_str",
                        lambda *args, **kwargs: "")
    monkeypatch.setattr(io, "input_bool",
                        lambda *args, **kwargs: False)


# --- fixtures ---------------------------------------------------------


def synth_face(w=RES, h=RES):
    """Deterministic uint8 BGR face stand-in (gradients only)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([
        np.clip(127 + 127 * xx / w, 0, 255),
        np.clip(127 - 127 * yy / h, 0, 255),
        np.clip(((xx + yy) % 251) * 255 / 251, 0, 255),
    ]).astype(np.uint8)
    return img


def binary_mask(w=RES, h=RES):
    """Deterministic binary (256,256,1) float32 mask: left half 0,
    right half 1 — the shape the editor consumes."""
    m = np.zeros((h, w, 1), dtype=np.float32)
    m[:, w // 2:, 0] = 1.0
    return m


def gradient_mask(w=RES, h=RES):
    """Deterministic fractional mask in [0,1): every row is a horizontal
    8-bit quantization ramp — PNG-encodes to a few KB, staying on the
    lossless path."""
    ramp = np.linspace(0.0, w - 1, w, dtype=np.float32) / w
    return ramp[None, :].repeat(h, axis=0)[:, :, None]


def make_labeled_polys():
    """INCLUDE + EXCLUDE polygons; all coordinates are exactly
    representable in float32 (integers and halves/quarters), so the
    round-trip is value-exact."""
    polys = SegIEPolys()
    include = polys.add_poly(SegIEPolyType.INCLUDE)
    for x, y in ((30.0, 40.5), (200.0, 55.0), (240.0, 240.0), (60.0, 220.25)):
        include.add_pt(x, y)
    exclude = polys.add_poly(SegIEPolyType.EXCLUDE)
    for x, y in ((100.0, 100.0), (150.0, 120.5), (120.0, 160.0)):
        exclude.add_pt(x, y)
    return polys


def make_faceset(dirpath, with_mask=True, with_labels=True):
    """Four-file faceset: DFLIMGs with mask / labels / bare metadata,
    plus one non-DFLIMG file that the CLI loops must skip untouched."""
    dirpath.mkdir(parents=True, exist_ok=True)
    specs = {
        "mask.jpg": {"face_type": FT_WF, "mask": binary_mask() if with_mask else None},
        "labels.jpg": {"face_type": FT_WF, "polys": make_labeled_polys() if with_labels else None},
        "plain.jpg": {"face_type": FT_FULL},
    }
    for name, meta in specs.items():
        p = dirpath / name
        cv2.imwrite(str(p), synth_face())
        dfl = DFLIMG.load(p)
        assert dfl is not None  # load() guarantees dfl_dict = {} when plain
        dfl.set_face_type(meta["face_type"])
        if meta.get("mask") is not None:
            dfl.set_xseg_mask(meta["mask"])
        if meta.get("polys") is not None:
            dfl.set_seg_ie_polys(meta["polys"])
        dfl.save()
    non_dfl = dirpath / "non_dfl.jpg"
    cv2.imwrite(str(non_dfl), synth_face())
    return specs


def reload_dflimg(path):
    dfl = DFLIMG.load(path)
    assert dfl is not None
    return dfl


# --- 1. static import scan -------------------------------------------


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_editor_package_imports_are_tf_and_model_free():
    editor_files = sorted(EDITOR_DIR.glob("*.py"))
    assert editor_files, "XSegEditor package files missing"
    for path in editor_files:
        mods = _module_imports(path)
        tf_like = {m for m in mods
                   if m in FORBIDDEN_MODULES
                   or m.startswith("tensorflow.")
                   or m.startswith("keras.")}
        assert not tf_like, f"{path.name} imports TF: {tf_like}"
        # the editor must not pull model-runtime code directly: it
        # works on DFLIMG metadata only (its samplelib import is the
        # preview-faceset reader, torch-only)
        model_like = {m for m in mods
                      if m == "facelib" or m.startswith("facelib.")
                      or m == "core.leras" or m.startswith("core.leras.")}
        assert not model_like, f"{path.name} imports model code: {model_like}"


def test_qtex_package_imports_are_tf_free():
    qtex_files = sorted(QTEX_DIR.glob("*.py"))
    assert qtex_files, "restored core/qtex package missing"
    for path in qtex_files:
        mods = _module_imports(path)
        tf_like = {m for m in mods
                   if m in FORBIDDEN_MODULES
                   or m.startswith("tensorflow.")
                   or m.startswith("keras.")}
        assert not tf_like, f"{path.name} imports TF: {tf_like}"


def test_editor_numpy2_gui_paths_avoid_removed_int_alias():
    """Normal paint/polygon paths must remain usable with pinned NumPy 2."""
    paths = [EDITOR_DIR / "XSegEditor.py", QTEX_DIR / "qtex.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        removed_aliases = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr == "int"
        ]
        assert not removed_aliases, (
            f"{path.name} uses removed np.int on a classic editor GUI path")


# --- 2. live import (real PyQt5 or stub) -----------------------------


def test_editor_import_without_tensorflow():
    had_pyqt5 = importlib.util.find_spec("PyQt5") is not None
    stubbed = []
    if not had_pyqt5:
        stubbed = _install_pyqt5_stub()
    try:
        module = importlib.import_module("XSegEditor.XSegEditor")
        assert callable(getattr(module, "start", None)), \
            "editor entry point (XSegEditor.start) missing"
        # the restored official core/qtex package feeds the editor's
        # QActionEx / QXMainWindow / QXIconButton / QSubprocessor
        # (catches a future core/qtex removal even under the stub)
        assert module.QActionEx.__module__.startswith("core.qtex")
        assert not _forbidden_loaded(), \
            "editor import chain loaded TensorFlow: %s" % _forbidden_loaded()
    finally:
        for name in list(sys.modules):
            if name == "XSegEditor" or name.startswith("XSegEditor."):
                del sys.modules[name]
            elif (name == "PyQt5" or name.startswith("PyQt5.")) and stubbed:
                del sys.modules[name]
        gc.collect()
    assert not _forbidden_loaded()


# --- 3. DFLIMG XSeg mask metadata round-trip --------------------------


def test_dflimg_xseg_mask_round_trip(tmp_root):
    root = tmp_root
    cv2.imwrite(str(root / "face.jpg"), synth_face())
    dfl = DFLIMG.load(root / "face.jpg")
    assert dfl is not None

    # binary mask: the lossless PNG path must be chosen (buffer < the
    # 50 KB cap) and the round-trip value-exact (the stored buffer is
    # the cv2.imencode output as a uint8 ndarray)
    dfl.set_face_type(FT_WF)
    dfl.set_seg_ie_polys(make_labeled_polys())  # labels coexist with mask
    dfl.set_xseg_mask(binary_mask())
    dfl.save()

    reloaded = reload_dflimg(root / "face.jpg")
    assert reloaded.has_xseg_mask()
    buf = reloaded.get_xseg_mask_compressed()
    assert buf is not None
    assert buf[:4].tobytes() == b"\x89PNG", \
        "binary mask must stay on the lossless PNG path"
    assert len(buf) <= 50000
    mask = reloaded.get_xseg_mask()
    assert mask.shape == (RES, RES, 1) and mask.dtype == np.float32
    np.testing.assert_array_equal(mask, binary_mask())  # exact
    # coexisting metadata survived the same save
    assert reloaded.get_face_type() == FT_WF
    polys = reloaded.get_seg_ie_polys()
    assert polys.has_polys()
    assert polys.identical(make_labeled_polys())

    # fractional mask: within the 8-bit quantization bound (1/255)
    cv2.imwrite(str(root / "frac.jpg"), synth_face())
    dfl2 = DFLIMG.load(root / "frac.jpg")
    dfl2.set_face_type(FT_FULL)
    dfl2.set_xseg_mask(gradient_mask())
    dfl2.save()
    reloaded2 = reload_dflimg(root / "frac.jpg")
    mask2 = reloaded2.get_xseg_mask()
    assert mask2.shape == (RES, RES, 1) and mask2.dtype == np.float32
    assert mask2.min() >= 0.0 and mask2.max() <= 1.0
    err = np.abs(mask2 - gradient_mask()).max()
    assert err <= 1.0 / 255.0 + 1e-6, \
        "fractional mask round-trip error %r exceeds 8-bit bound" % err

    # remove write path: None clears the metadata through save/reload
    reloaded.set_xseg_mask(None)
    reloaded.save()
    reloaded3 = reload_dflimg(root / "face.jpg")
    assert not reloaded3.has_xseg_mask()
    assert reloaded3.get_xseg_mask() is None
    assert reloaded3.get_face_type() == FT_WF
    assert reloaded3.get_seg_ie_polys().identical(make_labeled_polys())


# --- 4. polygon label compatibility -----------------------------------


def test_seg_ie_polys_label_round_trip(tmp_root):
    root = tmp_root
    cv2.imwrite(str(root / "lab.jpg"), synth_face())
    dfl = DFLIMG.load(root / "lab.jpg")
    polys = make_labeled_polys()
    dfl.set_face_type(FT_WF)
    dfl.set_seg_ie_polys(polys)
    dfl.save()

    reloaded = reload_dflimg(root / "lab.jpg")
    got = reloaded.get_seg_ie_polys()
    assert got.has_polys()
    assert len(got.get_polys()) == 2
    assert got.identical(polys), "labeled polygons must round-trip exact"
    # the editor's overlay preview is deterministic (the exact fill the
    # classic editor draws in VIEW_BAKED mode)
    overlay = np.zeros((RES, RES, 3), dtype=np.float32)
    got.overlay_mask(overlay)
    assert overlay[100, 150, 0] == 1.0     # inside INCLUDE, outside EXCLUDE
    assert overlay[127, 123, 0] == 0.0     # inside EXCLUDE (painted over)
    assert overlay[5, 5, 0] == 0.0         # outside all polygons

    # legacy list-form labels (old facesets) still load
    legacy = [
        (int(SegIEPolyType.INCLUDE), polys.get_polys()[0].get_pts()),
        (int(SegIEPolyType.EXCLUDE), polys.get_polys()[1].get_pts()),
    ]
    legacy_polys = SegIEPolys.load(legacy)
    assert legacy_polys.identical(polys)


# --- 5. CLI workflow contracts (remove / remove_labels / fetch) -------


def test_remove_xseg_cli_contract(tmp_root, no_cli_prompts):
    make_faceset(tmp_root)
    root = tmp_root
    non_dfl_before = (root / "non_dfl.jpg").read_bytes()
    labeled_before = make_labeled_polys()

    XSegUtil.remove_xseg(root)

    m = reload_dflimg(root / "mask.jpg")
    assert not m.has_xseg_mask(), "remove_xseg must clear the mask"
    assert m.get_face_type() == FT_WF  # other metadata untouched
    l = reload_dflimg(root / "labels.jpg")
    assert l.get_seg_ie_polys().identical(labeled_before), \
        "remove_xseg must not touch polygon labels"
    p = reload_dflimg(root / "plain.jpg")
    assert not p.has_xseg_mask()
    assert (root / "non_dfl.jpg").read_bytes() == non_dfl_before, \
        "non-DFLIMG files must be skipped untouched"


def test_remove_xseg_labels_cli_contract(tmp_root, no_cli_prompts):
    make_faceset(tmp_root)
    root = tmp_root

    XSegUtil.remove_xseg_labels(root)

    l = reload_dflimg(root / "labels.jpg")
    assert not l.has_seg_ie_polys(), "remove_xseg_labels must clear labels"
    assert l.get_face_type() == FT_WF
    m = reload_dflimg(root / "mask.jpg")
    mask = m.get_xseg_mask()
    assert m.has_xseg_mask()
    np.testing.assert_array_equal(mask, binary_mask()), \
        "remove_xseg_labels must not touch applied masks"


def test_fetch_xseg_cli_contract(tmp_root, no_cli_prompts):
    make_faceset(tmp_root)
    root = tmp_root
    out_dir = root.parent / (root.name + "_xseg")
    assert not out_dir.exists()

    XSegUtil.fetch_xseg(root)

    assert out_dir.is_dir(), "fetch_xseg must create <input>_xseg/"
    copied = sorted(p.name for p in out_dir.iterdir())
    assert copied == ["labels.jpg"], \
        "only faces WITH polygons may be copied: %s" % copied
    l = reload_dflimg(out_dir / "labels.jpg")
    assert l.has_seg_ie_polys(), "copied face must keep its labels"
    # "delete originals?" answered NO: originals survive
    assert (root / "labels.jpg").exists()
    l2 = reload_dflimg(root / "labels.jpg")
    assert l2.get_seg_ie_polys().identical(l.get_seg_ie_polys())


# --- 6. apply_xseg end-to-end (torch) + editor consumption ------------


def _build_model_dir(model_dir, seed=1234):
    """Deterministic XSeg checkpoint: seeded net weights -> XSeg_256.npy
    + the XSeg_data.dat options file apply_xseg reads (face_type)."""
    import pickle

    model_dir.mkdir(parents=True, exist_ok=True)
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NHWC")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = XSegNet("XSeg", RES, load_weights=False, weights_file_root=model_dir)
    net.save_weights()
    del net
    gc.collect()
    (model_dir / "XSeg_data.dat").write_bytes(
        pickle.dumps({"options": {"face_type": "wf"}, "iter": 1}))
    assert (model_dir / "XSeg_256.npy").exists()


def test_apply_xseg_writes_editor_consumable_mask(
        tmp_root, no_cli_prompts, monkeypatch):
    model_dir = tmp_root / "model"
    _build_model_dir(model_dir)
    # pin the device selection apply_xseg asks for (CPU, headless)
    monkeypatch.setattr(
        dfl_nn.DeviceConfig, "ask_choose_device",
        staticmethod(lambda *args, **kwargs: dfl_nn.DeviceConfig([])))

    input_root = tmp_root / "apply_in"
    input_root.mkdir()
    face = input_root / "face.jpg"
    cv2.imwrite(str(face), synth_face())
    dfl = DFLIMG.load(face)
    dfl.set_face_type(FT_WF)
    dfl.save()

    XSegUtil.apply_xseg(input_root, model_dir)

    dfl2 = reload_dflimg(face)
    assert dfl2.has_xseg_mask(), "apply_xseg must write the mask metadata"
    mask = dfl2.get_xseg_mask()  # the editor's exact read path
    assert mask.shape == (RES, RES, 1)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    vals = np.unique(mask)
    assert set(np.round(vals, 6).tolist()) <= {0.0, 1.0}, \
        "applied mask must be binary after the 0.5 gate: %s" % vals
    # non-mask metadata survives the apply+save
    assert dfl2.get_face_type() == FT_WF
    assert not dfl2.get_seg_ie_polys().has_polys()
    # the whole apply path (torch XSeg inference) stayed TF-free
    assert not _forbidden_loaded(), _forbidden_loaded()

    # determinism: a second run with the same weights reproduces the
    # exact same mask
    input2 = tmp_root / "apply_in2"
    input2.mkdir()
    face2 = input2 / "face.jpg"
    cv2.imwrite(str(face2), synth_face())
    dfl_b = DFLIMG.load(face2)
    dfl_b.set_face_type(FT_WF)
    dfl_b.save()
    XSegUtil.apply_xseg(input2, model_dir)
    mask2 = reload_dflimg(face2).get_xseg_mask()
    np.testing.assert_array_equal(mask, mask2)

    # apply -> remove loop closes (the documented editor workflow)
    XSegUtil.remove_xseg(input_root)
    assert not reload_dflimg(face).has_xseg_mask()
