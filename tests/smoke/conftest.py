"""Shared fixtures for the smoke test package.

Deterministic CPU autograd environment:
    The Phase 5 resume-equivalence tests compare A/B training
    histories bit-exactly (parameters, optimizer state, loss
    values). The CPU tests therefore run single-threaded:
    ``OMP_NUM_THREADS=1`` / ``DNNL_MAX_NUM_THREADS=1`` (set before
    OneDNN initializes) plus ``torch.set_num_threads(1)`` — a
    standard reproducibility guard for exact cross-instance
    comparisons, because OneDNN's multi-threaded f32 primitives are
    not guaranteed run-to-run deterministic. This is a
    test-environment measure only: no production code is affected,
    and CUDA tests are unaffected (GPU kernels do not use these
    pools).

``plain_tmp``: pytest 9 uses extended-length (``\\\\?\\``) temporary
paths on Windows, which the local sandbox rejects; directories created
through open-based calls (tempfile.mkdtemp / TemporaryDirectory) also
deny inner file writes under the same sandbox. This fixture therefore
probes the platform temp directory lifecycle and, when usable, creates
plain ``os.makedirs`` directories there; otherwise it falls back to the
git-ignored ``<repo>/.pytest-tmp`` directory (same pattern as the Phase
1 runtime-baseline fixture).
"""

import os
import random

# --- BEFORE any torch import (see the module docstring) -------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DNNL_MAX_NUM_THREADS", "1")
try:  # no-op when torch is unavailable; harmless when it is
    import torch as _torch

    _torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    pass

import pytest


def _probe_dir_lifecycle(root):
    """True if create/write/list/unlink/rmdir all work in ``root``."""
    probe = os.path.join(root, "dsh_probe_dir")
    try:
        os.makedirs(probe, exist_ok=True)
        f = os.path.join(probe, "p")
        with open(f, "wb") as fh:
            fh.write(b"x")
        os.scandir(probe)
        os.unlink(f)
        os.rmdir(probe)
        return True
    except OSError:
        return False


@pytest.fixture
def plain_tmp():
    """Temporary directory (makedirs-based) that works in sandboxed
    environments.

    Prefers the platform temp area; falls back to the git-ignored
    ``<repo>/.pytest-tmp`` directory when the temp area is unusable.
    Cleanup is lenient: leftovers in the platform temp area are removed
    by the OS, and workspace leftovers are git-ignored.
    """
    import tempfile
    from pathlib import Path

    root = tempfile.gettempdir()
    if not _probe_dir_lifecycle(root):
        root = str(Path(__file__).resolve().parents[2] / ".pytest-tmp")
        os.makedirs(root, exist_ok=True)
    name = "dfl_p3a_%s" % "".join(random.choices("0123456789abcdef", k=8))
    d = os.path.join(root, name)
    os.makedirs(d)
    yield d
    for entry in os.listdir(d):
        try:
            os.unlink(os.path.join(d, entry))
        except OSError:
            pass
    try:
        os.rmdir(d)
    except OSError:
        pass


# --- deterministic owned-process audit (Q11 stability audit) ----------------
#
# A model lifecycle owns real OS processes: the PreviewHistoryWriter
# (a daemon process spawned on the first preview post when
# write_preview_history is on) and, for non-debug training models, the
# SampleGeneratorFace subprocess workers (core.joblib.SubprocessGenera-
# tor). Python reaps daemon children only on a CLEAN interpreter exit;
# a crashed / killed / hung session orphans them, and the orphans keep
# spinning (pinning sample files and burning CPU scheduling) until the
# machine is manually cleaned — the measured root cause behind the
# Q11 fresh-subprocess pin's order-dependent instability.
#
# The production code now carries deterministic close() primitives
# (PreviewHistoryWriter.close, SubprocessGenerator.close,
# SampleGeneratorBase.close, ModelBase.finalize). This conftest wires
# them into the test lifecycle:
#   * every preview writer is registered when the model lazily creates
#     it, and every subprocess sample generator at construction;
#   * after EACH test, the writers/generators it created that no longer
#     belong to a live model are closed (zero owned live processes
#     between tests);
#   * at SESSION end, everything still open is closed and the session
#     FAILS if any owned worker process survives (a future leak
#     detector).
#
# No process scanning and no killing by executable name: only the exact
# multiprocessing.Process objects this test session spawned are ever
# touched.

import weakref

_AUDIT_AVAILABLE = True
try:
    # NOTE: `from models import ModelBase` resolves to the CLASS, not
    # the models.ModelBase submodule — models/__init__.py re-exports
    # the class at package level (it shadows the submodule name).
    from models import ModelBase as _ModelBase

    # One entry per model that owns OS processes:
    #   token      -> unique identity, also stored on the model
    #   model_ref  -> weakref to the owning model (dead once GC'd)
    #   generators -> the SampleGeneratorFace objects the model set
    #   writer     -> its PreviewHistoryWriter (lazy; None until used)
    _OWNED = {}

    def _own_entry(self):
        tok = getattr(self, "_dfl_owned_token", None)
        if tok is None:
            tok = object()
            self._dfl_owned_token = tok
        entry = _OWNED.get(tok)
        if entry is None:
            entry = {"token": tok, "model_ref": weakref.ref(self),
                     "generators": set(), "writer": None}
            _OWNED[tok] = entry
        return entry

    _real_set_gen = _ModelBase.set_training_data_generators

    def _tracked_set_gen(self, generator_list):
        _real_set_gen(self, generator_list)
        _own_entry(self)["generators"] = set(generator_list)
    _ModelBase.set_training_data_generators = _tracked_set_gen

    _real_gphw = _ModelBase.get_preview_history_writer

    def _tracked_gphw(self):
        writer = _real_gphw(self)
        if writer is not None:
            _own_entry(self)["writer"] = writer
        return writer
    _ModelBase.get_preview_history_writer = _tracked_gphw

except Exception:  # pragma: no cover
    _AUDIT_AVAILABLE = False
    _OWNED = {}
    # Fail LOUD, never silent: an audit that quietly no-ops once (e.g.
    # an import failure in one venv) would let owned worker processes
    # accumulate with zero visible sign — exactly how the Q11
    # instability got missed. The marker goes to raw stderr at import
    # time (before pytest capture starts), so it lands in every log.
    import sys as _sys
    _sys.stderr.write(
        "OWNED PROCESS AUDIT: UNAVAILABLE (models import or base-class "
        "patching failed at conftest import) — the suite runs WITHOUT "
        "its per-test/session process leak detector. Fix this before "
        "relying on any process-lifecycle result.\n")


def _proc_alive(owner):
    p = getattr(owner, "p", None)
    return bool(p is not None and p.is_alive())


def _owner_procs_alive(owner):
    # Liveness of the OS processes an owner owns. A
    # SampleGeneratorFace owns its workers as a list of
    # SubprocessGenerator/ThisThreadGenerator objects (``owner.generators``);
    # only the SubprocessGenerators carry an OS process (``owner.p``) —
    # the Face object itself has no ``p`` attribute, so checking the
    # Face directly would always report "no live process" and let every
    # worker pool (plus its index-host threads) run away to interpreter
    # shutdown. A bare SubprocessGenerator (direct unit-test usage) is
    # its own process.
    gens = getattr(owner, "generators", None)
    if gens is not None:
        return any(_proc_alive(g) for g in gens)
    return _proc_alive(owner)


def _entry_alive(entry):
    w = entry.get("writer")
    if w is not None and _proc_alive(w):
        return True
    return any(_owner_procs_alive(g) for g in entry.get("generators", ()))


def _close_entry_quiet(entry):
    w = entry.get("writer")
    if w is not None:
        try:
            w.close()
        except Exception:
            pass
    for g in entry.get("generators", ()):
        if g is not None:
            try:
                g.close()
            except Exception:
                pass


@pytest.fixture(autouse=True)
def dfl_owned_process_cleanup():
    """Reap the owned worker processes of models created by THIS test
    once the model itself is gone (garbage-collected). Models that are
    still alive (shared module-scoped models, or ones a later test
    still references) are deliberately left untouched and are reaped at
    session end instead. This is what keeps a later fresh-subprocess
    numerical pin (the Q11 test) from starting on top of the previous
    test's leftover worker processes. Only the exact Process objects
    this session's models spawned are ever touched — no scanning, no
    executable-name matching."""
    if not _AUDIT_AVAILABLE:
        yield
        return
    before = set(_OWNED.keys())
    yield
    import gc as _gc
    _gc.collect()
    for tok, entry in list(_OWNED.items()):
        if tok in before:
            continue  # created by an earlier test / a shared model
        if entry["model_ref"]() is not None:
            continue  # model still alive -> keep for session end
        # Close UNCONDITIONALLY — never gate on "is a worker process
        # still alive": a test can exhaust its samples, which makes the
        # worker process exit on its own (prefetch loop -> StopIteration
        # -> exit), while the generator's index-host helper THREAD is
        # still alive and must be stopped deterministically. Gating on
        # process liveness left exactly those hosts running to
        # interpreter shutdown, where the queue finalizers raised in
        # them (stderr storm) and wedged the exit. close() is
        # idempotent and a clean no-op for entries whose processes are
        # already gone.
        _close_entry_quiet(entry)


@pytest.fixture(scope="session", autouse=True)
def dfl_session_owned_process_audit():
    """Session end: close everything still open and FAIL the run if
    any owned worker process survived — the suite-wide proof that the
    lifecycle returns to zero owned live workers."""
    if not _AUDIT_AVAILABLE:
        yield
        return
    yield
    import gc as _gc
    _gc.collect()
    entries = list(_OWNED.values())
    closed_now = 0
    for entry in entries:
        # Close everything, unconditionally (same rationale as the
        # per-test reaper: entries whose worker already exited on its
        # own still own live index-host threads that must be stopped
        # before the interpreter shuts down). close() is idempotent.
        _close_entry_quiet(entry)
        closed_now += 1
    _gc.collect()
    remaining = [e for e in entries if _entry_alive(e)]
    n_gens = sum(len(e.get("generators", ())) for e in entries)
    n_writers = sum(1 for e in entries if e.get("writer") is not None)
    print("OWNED PROCESS AUDIT: %d model(s) tracked, %d writer(s), "
          "%d sample generator(s); %d reaped at session end; %d "
          "still alive" % (len(entries), n_writers, n_gens,
                           closed_now, len(remaining)))
    assert not remaining, (
        "owned worker processes survived the session end: %r"
        % [[getattr(g, "p", None) and g.p.pid
            for g in e.get("generators", ())]
           for e in remaining])
