"""Deterministic DFL-style checkpoint payload builder (Phase 1).

Pure-numpy module shared by:

- ``test_runtime_baseline.py`` (compares numpy 2.x round-trips and the
  loaded numpy 1.x fixture against this payload), and
- ``generate_numpy1_fixture.py`` (serializes the payload inside a
  numpy 1.x environment).

The payload mirrors the official DFL checkpoint convention (pickled dict
in a .npy file, TF-style variable names, HWIO conv kernels, ms_/vs_
optimizer state, iters:0 — IMPLEMENTATION_PLAN.md section 17).
Keeping this builder free of any non-numpy imports lets it run in a
minimal numpy 1.x environment.
"""

import numpy as np


def build_np1_fixture_dict():
    rng = np.random.default_rng(42)
    return {
        "encoder/conv2d/kernel:0": rng.standard_normal((3, 3, 3, 8), dtype=np.float32),
        "encoder/conv2d/bias:0": rng.standard_normal((8,), dtype=np.float32),
        "iters:0": np.array(1234, dtype=np.int64),
        "ms_encoder/conv2d/kernel:0": rng.standard_normal((3, 3, 3, 8), dtype=np.float32),
        "vs_encoder/conv2d/kernel:0": rng.standard_normal((3, 3, 3, 8), dtype=np.float32),
    }
