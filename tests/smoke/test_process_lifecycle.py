"""Q11 regression-stability audit: deterministic ownership and
shutdown of the OS processes a model lifecycle owns.

Measured root cause: a training model owns real OS processes — the
PreviewHistoryWriter daemon process (spawned on the first preview
post when write_preview_history is on) and, for non-debug training
models, the SampleGeneratorFace subprocess workers
(core.joblib.SubprocessGenerator, one per sample-generator slot).
Python reaps daemon children only on a CLEAN interpreter exit; a
crashed / killed / hung session (and ``os._exit`` in a child, as the
Q11 pin child performs) orphans them. The orphans keep spinning
(polling their queues, pinning sample files, burning CPU
scheduling), and the accumulated pressure is the infrastructure
instability behind the Q11 fresh-subprocess numerical pin.

The production fix under test (NO tolerance or formula change):

- ``core.joblib.SubprocessGenerator.close()`` — terminate+join the
  owned worker (idempotent; a closed generator can never be
  re-started);
- ``core.joblib.ThisThreadGenerator.close()`` — the no-op
  counterpart (the in-process generator owns nothing);
- ``samplelib.SampleGeneratorBase.close()`` — cascades close() to
  every owned worker generator;
- ``models.ModelBase.PreviewHistoryWriter.close()`` — graceful
  ``None`` sentinel to the child, bounded terminate fallback,
  join, queue close (idempotent);
- ``models.ModelBase.finalize()`` — closes the owned preview
  writer and the training sample generators before
  ``nn.close_session()``.

Pinned contracts (CPU tier; the process layer is
backend-neutral, so this file runs in BOTH venvs):

1. a sample-generator lifecycle that spawns workers returns to
   ZERO owned live workers after shutdown;
2. repeated construct/close cycles do NOT accumulate workers;
3. the full model lifecycle (subprocess sample workers + the
   preview writer) finalizes to zero owned live processes.

Run:  .venv-cpu\\Scripts\\python.exe -m pytest tests/smoke/test_process_lifecycle.py -q
"""

import multiprocessing
import time

import pytest

import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.joblib import SubprocessGenerator

from test_model_amp_training import (  # noqa: E402
    TINY as AMP_TINY,
    seed as amp_seed,
)
from Model_AMPTest.Model import (  # noqa: E402
    AMPHeadless,
    make_model as make_amp,
    make_training_dirs as amp_make_training_dirs,
)
from test_model_saehd_training import (  # noqa: E402
    TINY as SAEHD_TINY,
    seed as saehd_seed,
)
from Model_SAEHDTest.Model import (  # noqa: E402
    SAEHDHeadless,
    make_model as make_saehd,
    make_training_dirs as saehd_make_training_dirs,
)


# --- the abstraction-level worker stream -------------------------------------

def _tiny_stream(param):
    """Infinite batch stream for the SubprocessGenerator
    abstraction tests. Module-level so the SPAWNED worker child can
    re-import this module and unpickle the reference."""
    i = 0
    while True:
        i += 1
        yield ("batch", i)


def _settle_to(baseline, timeout=15.0):
    """Wait (bounded) for process-teardown bookkeeping to settle
    after a close() that already terminated+joined the child. This
    is a wait for a GUARANTEED event, not a retry of a failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(multiprocessing.active_children()) <= baseline:
            return True
        time.sleep(0.05)
    return len(multiprocessing.active_children()) <= baseline


# --- 1 + 2: the owning abstraction (SubprocessGenerator) ---------------------

def test_subprocess_generator_close_returns_zero_owned_workers():
    """A SubprocessGenerator that spawns a worker returns to zero
    owned live workers after close() — and a closed generator can
    never be re-started."""
    baseline = len(multiprocessing.active_children())

    gen = SubprocessGenerator(_tiny_stream, start_now=True)
    assert gen.p is not None and gen.p.is_alive()
    assert next(gen) == ("batch", 1)  # the worker really ran

    gen.close()
    assert gen.p is None, "close() must detach the owned process"
    with pytest.raises(StopIteration):
        next(gen)
    assert _settle_to(baseline), "owned worker survived close()"
    assert len(multiprocessing.active_children()) == baseline


def test_repeated_subprocess_cycles_do_not_accumulate_workers():
    """construct -> use -> close, repeated: the owned-worker count
    returns to the baseline after EVERY cycle (no accumulation)."""
    baseline = len(multiprocessing.active_children())
    for cycle in range(4):
        gen = SubprocessGenerator(_tiny_stream, start_now=True)
        assert gen.p is not None and gen.p.is_alive()
        assert next(gen) == ("batch", 1)
        gen.close()
        assert _settle_to(baseline), \
            "cycle %d left a worker behind" % cycle
        assert len(multiprocessing.active_children()) == baseline, \
            "cycle %d accumulated workers" % cycle


# --- 3a: full SAEHD model lifecycle (subprocess sample workers) --------------

def test_model_finalize_reaps_subprocess_sample_workers(plain_tmp):
    """A non-debug (subprocess-worker) SAEHD training model: after
    construction it owns exactly ``2 * (cpu_count // 2)`` worker
    processes; one real training iteration runs through them;
    finalize() closes them all — zero owned live workers remain."""
    from pathlib import Path

    per_side = multiprocessing.cpu_count() // 2
    expected = 2 * per_side  # the src + dst faces, no ct_mode in TINY

    baseline = len(multiprocessing.active_children())
    d = Path(plain_tmp) / "model"
    saehd_make_training_dirs(d)
    model = make_saehd(SAEHDHeadless, d, is_training=True,
                       seed=saehd_seed(**SAEHD_TINY),
                       debug=False, cpu_only=True)
    try:
        assert len(multiprocessing.active_children()) \
            == baseline + expected, \
            "expected %d owned sample workers, saw %d" % (
                expected,
                len(multiprocessing.active_children()) - baseline)
        # one real training iteration through the subprocess workers
        model.train_one_iter()
        assert model.iter == 1
        # finalize(): the owned sample generators (and, here, no
        # preview writer) are closed deterministically
        model.finalize()
        assert _settle_to(baseline), \
            "owned sample workers survived finalize()"
        assert len(multiprocessing.active_children()) == baseline
    finally:
        for g in (model.generator_list or []):
            g.close()


# --- 3b: full AMP model lifecycle (workers + preview writer) -----------------

def test_model_finalize_reaps_preview_writer_and_sample_workers(plain_tmp):
    """The full owned-process set of a non-debug AMP training model
    with write_preview_history on: the subprocess sample workers
    AND the PreviewHistoryWriter daemon process. finalize() closes
    every one of them — zero owned live processes remain. (The
    preview post on iteration 0 is what lazily spawns the writer,
    so the writer child really exists before finalize().)"""
    from pathlib import Path

    baseline = len(multiprocessing.active_children())
    d = Path(plain_tmp) / "model"
    amp_make_training_dirs(d)
    model = make_amp(AMPHeadless, d, is_training=True,
                     seed=amp_seed(**AMP_TINY, write_preview_history=1),
                     debug=False, cpu_only=True)
    try:
        model.train_one_iter()  # spawns the writer via the preview post
        assert model.preview_history_writer is not None
        assert model.preview_history_writer.p is not None \
            and model.preview_history_writer.p.is_alive(), \
            "the preview writer process must exist before finalize()"
        n_workers = len(multiprocessing.active_children())
        assert n_workers > baseline, "no owned workers spawned?"

        model.finalize()

        # the writer child is gone (close() detaches its handle)
        assert model.preview_history_writer.p is None
        assert _settle_to(baseline), \
            "owned writer/worker processes survived finalize()"
        assert len(multiprocessing.active_children()) == baseline
    finally:
        writer = getattr(model, "preview_history_writer", None)
        if writer is not None:
            writer.close()
        for g in (model.generator_list or []):
            g.close()
