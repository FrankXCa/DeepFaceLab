# Smoke tests

Runtime and pipeline smoke tests:

- **L0 import:** Torch-only core imports cleanly, CPU path included.
- **L1 initialization:** layers/models initialize deterministically.
- **Runtime baseline acceptance** (Phase 1): torch version check, GPU
  enumeration, CPU fallback, basic tensor allocation.
- **L7 short training:** per-model 100-iteration smoke tests (stable
  loss, no NaN/Inf, checkpoint save/resume).
- **L11 end-to-end:** Extract -> Train -> Merge on a small sample set.

Created as a skeleton in Phase 0; test code is populated from Phase 1.
