"""Generate the NumPy 1.x-era checkpoint fixture (Phase 1).

Run inside a numpy<2 environment (the baseline uses .venv-np1):

    .venv-np1/Scripts/python tests/smoke/generate_numpy1_fixture.py

Writes tests/smoke/fixtures/numpy1_checkpoint_fixture.npy — a pickled
dict in the official DFL checkpoint convention, serialized by numpy 1.x.
tests/smoke/test_runtime_baseline.py then loads it under the NumPy 2
runtime to prove cross-version checkpoint compatibility (the primary risk
of adopting NumPy 2 for official DFL checkpoints).

The generated file is a test artifact and is git-ignored (*.npy); it is
regenerated on demand and never committed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np1_fixture import build_np1_fixture_dict  # noqa: E402

import numpy as np


def main():
    if not np.__version__.startswith("1."):
        sys.exit(f"requires a numpy 1.x environment, found {np.__version__}")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "numpy1_checkpoint_fixture.npy")
    np.save(out, build_np1_fixture_dict(), allow_pickle=True)
    print(f"numpy {np.__version__} wrote {out}")


if __name__ == "__main__":
    main()
