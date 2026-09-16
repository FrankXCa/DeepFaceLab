# Smoke tests

Runtime and pipeline smoke tests.

Populated in Phase 1:

- `test_runtime_baseline.py` — Phase 1 runtime baseline acceptance
  (`IMPLEMENTATION_PLAN.md` section 12): Python floor, torch
  import/version (L0), CUDA availability + GPU enumeration + device
  name, CPU fallback tensor ops, CUDA tensor ops, NumPy version,
  torch<->numpy interop, NumPy 2 checkpoint pickle round-trip, and a
  cross-version load of a numpy 1.x-era checkpoint fixture.
- `generate_numpy1_fixture.py` — run in the numpy 1.x environment
  (`.venv-np1`) to produce the git-ignored fixture
  `fixtures/numpy1_checkpoint_fixture.npy`.

Environments (git-ignored, created with `uv`):

- `.venv`      — CUDA baseline (`requirements-cuda.txt`)
- `.venv-cpu`  — CPU-only baseline (`requirements-cpu.txt`)
- `.venv-np1`  — numpy 1.x fixture generator (numpy only)

Later phases add:

- L1 initialization, L7 short training (100-iteration) smoke tests.
- L11 end-to-end Extract -> Train -> Merge on a small sample set.
