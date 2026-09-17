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
