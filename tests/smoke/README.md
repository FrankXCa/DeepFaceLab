# Smoke tests

Runtime and pipeline smoke tests.

Populated in Phase 1:

- `test_runtime_baseline.py` — Phase 1 runtime baseline acceptance
  (`IMPLEMENTATION_PLAN_v2.md` section 12): Python floor, torch
  import/version (L0), CUDA availability + GPU enumeration + device
  name, CPU fallback tensor ops, CUDA tensor ops, NumPy version,
  torch<->numpy interop, NumPy 2 checkpoint pickle round-trip, and a
  cross-version load of a numpy 1.x-era checkpoint fixture.
- `generate_numpy1_fixture.py` — run in the numpy 1.x environment
  (`.venv-np1`) to produce the git-ignored fixture
  `fixtures/numpy1_checkpoint_fixture.npy`.

Populated in Phase 2:

- `test_device.py` — Phase 2 device abstraction acceptance
  (`IMPLEMENTATION_PLAN_v2.md` section 13): CPU-only initialization,
  CUDA availability detection, RTX 4090 detection, GPU name, index
  selection (official BestGPU/WorstGPU/GPUIndexes/CPU semantics), VRAM
  reporting, invalid device index handling, multi-GPU enumeration logic
  (labeled MOCK backend — structural only), backend metadata,
  capability metadata, model-facing allocation without direct
  `torch.cuda.*` calls, no TensorFlow import in the device layer,
  `NN_DEVICE_*` environment contract, and `nn.initialize_main_env()` /
  `nn.DeviceConfig` alias compatibility.
- `conftest.py` (repository root) — puts the repo root on `sys.path`
  so `import core...` works from any pytest invocation style.

Populated in Phase 3A:

- `test_leras_foundation.py` — Phase 3A torch-only leras foundation
  acceptance (`IMPLEMENTATION_PLAN_v2.md` section 14, 3A scope):
  foundation imports without TensorFlow, LayerBase/module construction
  (official two-phase `build_weights`/`init_weights` lifecycle on a
  synthetic stand-in layer), deterministic official-name enumeration,
  stable naming across repeated construction, device placement through
  the Phase 2 abstraction (CPU; CUDA-on-4090 where available),
  official-format save/load round-trip on a synthetic module, strict
  failure behavior (`CheckpointLoadError` for missing/extra/shape
  mismatches, all-or-nothing), no direct CUDA API in foundation sources
  (AST-checked).
- `conftest.py` (this directory) — shared `plain_tmp` fixture
  (sandbox-safe makedirs-based temp directories; also used unchanged
  in style by the Phase 1 fixture).

Current counts (2026-09, after Phase 3A): CUDA environment 42 passed;
CPU environment 34 passed + 8 skipped.

Environments (git-ignored, created with `uv`):

- `.venv`      — CUDA baseline (`requirements-cuda.txt` + `requirements-dev.txt`)
- `.venv-cpu`  — CPU-only baseline (`requirements-cpu.txt` + `requirements-dev.txt`)
- `.venv-np1`  — numpy 1.x fixture generator (numpy only)

The Phase 2 device suite must be run in BOTH environments:

```
.venv\Scripts\python    -m pytest tests/smoke -v -p no:cacheprovider   (17/17 + baseline)
.venv-cpu\Scripts\python -m pytest tests/smoke -v -p no:cacheprovider  (GPU tests skip)
```

`requirements-dev.txt` (test-only: opencv/tqdm/IPython/matplotlib/
pillow) is installed in both test environments because
`core.leras.nn` imports `core.interact`; it is NOT part of the minimal
runtime requirements.

Later phases add:

- L1 initialization, L7 short training (100-iteration) smoke tests.
- L11 end-to-end Extract -> Train -> Merge on a small sample set.
