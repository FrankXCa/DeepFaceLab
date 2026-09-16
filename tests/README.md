# Tests

Test infrastructure is a first-class project feature, not cleanup after
implementation (`IMPLEMENTATION_PLAN.md` section 9): the largest weakness
shared by all existing Torch ports is the lack of reliable TF-vs-Torch
numeric parity tests.

## Layout

| Directory | Purpose | Populated from |
|---|---|---|
| `parity/` | TF-vs-Torch numerical parity tests (forward, loss, gradient, optimizer step) | Phase 5 |
| `checkpoints/` | Checkpoint round-trip, conversion, and failure-behavior tests | Phase 4 |
| `smoke/` | L0 import, L1 initialization, runtime baseline, short-training smoke tests | Phase 1 |

## Test levels

Per `IMPLEMENTATION_PLAN.md` section 9:

```text
L0  Import
L1  Initialization
L2  Forward parity
L3  Loss parity
L4  Gradient parity
L5  Optimizer-step parity
L6  Checkpoint round-trip
L7  Short training smoke test
L8  Preview parity
L9  Merge parity
L10 DFM/ONNX contract test
L11 End-to-end Extract -> Train -> Merge
```

## Ground rules

- The production application remains Torch-only. An isolated official
  DFL / TensorFlow environment is used only to generate reference
  values for parity tests (`IMPLEMENTATION_PLAN.md` section 21).
- Synthetic deterministic inputs are preferred. Seeds are set for
  Python, NumPy, Torch, and the TensorFlow reference; nondeterministic
  optimizations are disabled where required for comparison.
- Tolerances are defined per operation and dtype, never one global
  tolerance, and the reason is documented wherever exact parity is
  unrealistic.
- Parity may never be claimed without a numerical test
  (`IMPLEMENTATION_PLAN.md` section 8); parity status uses the labels
  of `REVIEW_AND_COMMIT_GUIDE.md` section 28.

Phase 0 creates the directory skeleton only; no test code is added yet.
