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

Populated in Phase 3B:

- `test_leras_layers.py` — Phase 3B concrete layer acceptance for all
  12 migrated layers (Conv2D, Conv2DTranspose, DepthwiseConv2D,
  Dense, DenseNorm, BatchNorm2D, InstanceNorm2D, FRNorm2D, BlurPool,
  AdaIN, TLU, ScaleAdd): config/type errors, two-phase build
  lifecycle, official parameter names/shapes, dtype/device via the
  Phase 2 abstraction, forward shape + parity vs manual NumPy
  references (deterministic unique-value tensors; WITHIN_TOLERANCE
  1e-4 or EXACT per test), CPU always / RTX 4090 GPU execution
  (skip-if-CPU-only), naming stability across repeated construction,
  Saveable round-trips producing OFFICIAL-layout `.npy` files,
  synthetic official->torch layout conversions with exact index maps,
  strict invalid-shape rejection, import boundary (no TensorFlow),
  and an AST check that no layer source calls `torch.cuda` or
  hardcodes `'cuda:'` device strings.

Populated in Phase 3C:

- `test_depth_to_space.py` — Phase 3C acceptance for the
  official-compatible `depth_to_space` (TF R-R-C semantics):
  deterministic exact index placement (block size 2, every output
  cell checked, plus the classic 2x2 placement table), random
  unique-value tensors against an independent NumPy implementation of
  the official index formula (multiple channel counts 4/8/12/16/9/18,
  non-square spatial sizes, block sizes 1/2/3 — TensorFlow runtime
  parity honestly labeled NOT_VERIFIED in this phase), invalid
  channel-count/size rejection (strict ValueError), CPU execution,
  RTX 4090 GPU execution through the Phase 2 abstraction with
  bit-exact CPU-vs-GPU equality (skip-if-CPU-only), dtype
  preservation (float32/float16/float64), gradient-flow smoke (the op
  is an element bijection, so `sum().backward()` gives exact ones),
  the NHWC boundary contract, the `nn.depth_to_space` alias used by
  official call sites, no TensorFlow import in the production path,
  and an AST check that the ops source contains no `torch.cuda` or
  `cuda:` literals.

Populated in Phase 3D:

- `test_ops_core.py` — Phase 3D acceptance for the core numerical
  ops with official semantics: `dssim` (parity vs an independent
  NumPy reference of the official formula - arange-centered
  softmax-normalized window, **VALID** convolution, the official
  luminance×cs expressions, (N, C) reduction, even filter sizes like
  22, identical/different inputs, dtype round-trip and mismatch
  rejection, gradient smoke), `gaussian_blur` (kernel normalization,
  impulse response at the kernel-center tap, interior constant
  preservation with the official zero-padded edge behavior, the
  sigma values used by the official models 2/4/8/32, non-square
  shapes, gradient smoke), `style_loss` (per-channel TF-moments
  formula - both terms squared, per-sample (N,) results,
  loss_weight scaling, identical-input zero, shifted/scaled inputs,
  channel-mismatch rejection, the blurred-radius path, and an
  explicit test proving the implementation is NOT the External A/B
  gram-matrix variant), `pixel_norm` (official epsilon 1e-6 with a
  test that discriminates it from External B's 1e-8, zero/near-zero
  stability, per-axis behavior, gradient smoke); plus GPU execution
  on the RTX 4090 with CPU-vs-GPU parity (skip-if-CPU-only), the
  NHWC boundary contract, the TensorFlow import boundary, and the
  no-direct-CUDA AST check.

Populated in Phase 3E1:

- `test_ops_lowlevel.py` — Phase 3E1 acceptance for the remaining
  low-level ops with official semantics: `flatten` (exact
  channel-major element placement in NCHW, the NHWC boundary-
  transpose guarantee - channel-major even for NHWC input, dtype
  preservation, gradient flow), `reshape_4D` (exact element
  placement NCHW and NHWC output, invalid flat tail fails
  explicitly - no permissive heuristics, gradient flow),
  `average_tensor_list` (single-element identity, exact mean
  reference, official tf_device_string signature parity, gradient
  flow 0.5 per input), `total_variation_mse` (the official
  VERBATIM formula - axis-1/axis-2 slice differences, squared,
  SUMMED over axes 1..3 -> per-sample (N,) vector; under NCHW the
  axis-1 term is the official channel-difference quirk that the
  baseline SAEHD/AMP GAN loss actually computed; the suite
  explicitly distinguishes the official per-sample sum from
  External A's global scalar mean variant; zero-input exact zero;
  gradient smoke); plus the RTX 4090 GPU execution test with
  bit-exact CPU-vs-GPU parity on the deterministic integer-valued
  inputs (skip-if-CPU-only) and the nn/module alias registration.

Populated in Phase 3E2:

- `test_optimizers.py` — Phase 3E2 acceptance for the torch
  optimizer layer (official semantics; the verbatim TF reference is
  `core/leras/optimizers/optimizers_tf.py`, never imported by torch
  paths): independent-NumPy-reference parity for one-step and
  multi-step `AdaBelief`/`RMSprop` updates (fresh-state and
  evolving-state references; the torch code is never its own
  oracle), zero-gradient exact no-ops, positive/negative gradient
  direction, multi-parameter updates, the official denominator
  epsilon == `np.finfo(dtype).resolution` - the DECIMAL resolution
  (1e-06 for f32; verified under the official pinned NumPy 1.19.3,
  identical under NumPy 2.x; the machine epsilon
  `torch.finfo(...).eps` = 1.19e-07 is NOT the official value - a
  test discriminates the two), NO bias correction / NO momentum / NO
  weight decay (official has none), the lr_cos schedule (official
  literal `2*3.1415926535/lr_cos` and the POST-increment iteration
  count - a step-1 discrimination test pins the official TF
  queue-order semantics), global-norm gradient clipping (float32
  norm over ALL gradients, per-gradient c/n scaling; the below-
  threshold run must equal the no-clip run), lr_dropout (one fresh
  mask per parameter per step - the USER_LEGACY frozen mask is
  rejected: consecutive steps must differ; p=1.0 == disabled,
  p=0.0 freezes weights while states still evolve; statistical
  rate), the official state layout and checkpoint sub-names
  (`iters:0` + all `ms_*`/`vs_*`/`acc_*`, positional `param_i`
  keys for unnamed params, never object ids), the mandatory
  resume test (5 steps -> snapshot weights+state -> fresh
  optimizer -> restore -> step 6 == continuous run), `random_
  binomial` (p=0/p=1 extremes, dtype/shape, seeded determinism,
  per-call independence, Bernoulli rate over 200k draws), the nn
  registry aliases, the TensorFlow-free import boundary, the
  no-direct-CUDA AST check, and RTX 4090 execution through the
  Phase 2 abstraction with CPU-vs-GPU parity (skip-if-CPU-only).

Populated in Phase 3F:

- `test_archis.py` — Phase 3F acceptance for the official archis and
  discriminators (official TF references preserved in
  `core/leras/archis/archis_tf.py` and
  `core/leras/models/discriminators_tf.py`, never imported by torch
  paths): `DeepFakeArchi` factory construction across the official
  option space ('' / 't' / 'd' / 'td' / 'u' / 'ud' / 'c' at
  resolutions 64-256), the official `get_out_ch`/`get_out_res`
  contracts, official encoder/inter/decoder tensor layouts for every
  combo (full-resolution x/m heads), the official `use_fp16` conv
  dtypes (construction + dtype checks; no fp16 numerics), the 'c'
  CosRelu x*cos(x) branch (alpha ignored) vs the default
  leaky_relu(0.1/0.2) branches, the 'u' pixel_norm branch
  (official 1e-6 epsilon, cross-checked against an identical seeded
  non-'u' twin), the `mod='quick'` official dead end (mirrored
  NameError), block-level wiring through the instantiated
  Encoder/Inter/Decoder submodules (the official block classes are
  factory closures, not archi attributes), the depth_to_space flow
  in Upscale/the 'd' decoder head vs an independent oracle mirroring
  the official R-R-C grouping (all official branches - the manual
  NCHW-CPU/NHWC code and the NCHW-GPU tf.depth_to_space built-in -
  use it, per the Phase 3F P0 re-audit; torch's F.pixel_shuffle
  groups channels C-R-R and is NOT the official semantics),
  `CodeDiscriminator` (official
  n_downscales = 1 + code_res//8, kernel 4 then 3),
  `PatchDiscriminator` (the official 46-entry
  patch_discriminator_kernels table, verified byte-for-byte against
  the baseline), `UNetPatchDiscriminator` (official find_archi /
  calc_receptive_field_size layer search, level_chs progression,
  1x1 VALID out_conv/center convs, (center_out, x) outputs, even-
  input requirement of the official skip concat - the official models
  feed power-of-2 crops), official checkpoint key sets
  (convs_<i>/downs_<i>/upconvs_<i> list scopes, singular
  weight/bias, :0 suffix, deterministic), official-format
  save/load round trip (pickled dict protocol 4, official HWIO
  layouts, strict rejection of unexpected keys), flat deterministic
  get_weights, backward reaching every conv/dense parameter, NCHW/
  NHWC data-format equivalence (inputs fed in the active format),
  no TensorFlow import and no direct torch.cuda.* in the migrated
  sources (AST), and RTX 4090 execution with CPU/GPU parity
  (weights generated on the CPU twin and copied to the GPU twin -
  torch CUDA/CPU RNG streams differ per seed; measured device-noise
  band atol 1e-3, skip-if-CPU-only).

Populated in Phase 4:

- `test_checkpoint_conversion.py` — Phase 4 checkpoint
  compatibility / conversion acceptance (`core/leras/convert.py`,
  the centralized conversion engine — independent reimplementation;
  concept sources: official DFL file-format contract (GPL-3.0),
  EXTERNAL_A strict two-pass load (GPL-3.0), EXTERNAL_B
  component/optimizer-state mapping concepts (unlicensed, NOT
  copied)): official file-format validation — the EXACT outer
  container is a RAW pickle protocol-4 stream of a
  `dict[str, np.ndarray]` (the official `save_weights` writes
  `pickle.dumps(d, 4)` + `write_bytes_safe`; the `.npy` extension is
  a misnomer, the file is NOT a NumPy `.npy` container —
  `np.save`/`np.load` play no role; "protocol 4" is the pickle
  protocol of the whole file) — pinned independently of the
  converter's reader (leading pickle bytes identical to the real
  artifacts, `pickle.loads` — the official load primitive — parses
  keys/shapes/dtypes/values exactly, structural comparison against a
  real artifact, cross-checked in the NumPy 1.x environment) and
  against the REAL official artifacts tracked in the baseline
  (`facelib/*.npy` pickled dicts written by official TF-era DFL
  code: S3FD/2DFAN float32 with the NHWC 4-D singleton-padded
  `(1,1,1,C)` bias/BN forms, 3DFAN float32 1-D `(C,)` forms,
  FaceEnhancer float16) + protocol-4 re-pickle round-trip +
  corrupt-file rejection (truncated pickles, non-dict pickles, real
  `np.save` array files, non-string keys, non-ndarray values); name
  mapping (torch dotted <-> official slashed `:0`, `:0`-variant
  tolerance); declared layout rules with UNIQUE-VALUE tensors
  (Conv2D HWIO<->OIHW, Conv2DTranspose, DepthwiseConv2D, Dense
  official-layout identity, and the `channel_broadcast` WHITELIST
  for 1-D bias/BN/eps parameters — exactly the known official
  singleton layouts `(C,)` (identity), `(1,1,1,C)` (NHWC padding)
  and `(1,C,1,1)` (NCHW padding); every other shape, including
  same-element-count placements a naive `np.squeeze()` would fix
  (`(1,C)`, `(1,1,C)`, `(1,C,1)`, `(C,1)`, `(1,1,C,1)`,
  `(C,1,1,1)`, `(2,1,1,C)`), is rejected explicitly — no
  element-count reshape fallback anywhere);
  strict two-pass all-or-nothing conversion (missing required
  weights, unexpected extra keys, same-element-count wrong shapes,
  dtype mismatch, ambiguous mapping -> `CheckpointLoadError` with
  the full structured report, nothing copied on failure); reverse
  export (torch -> official-layout dict via the per-layer
  `convert_weight_to_official` hooks) with explicit
  `UnsupportedExportError` rejection of non-exportable state (no
  silent drop/approximate/reshape/coerce); archi (canonical
  option combos) + discriminator conversion and file-level
  round-trips through write/read_official_checkpoint;
  Saveable.load_weights agreement with the converter; no
  TensorFlow import / no direct `torch.cuda.*` in the conversion
  source (AST); SAEHD/AMP/Quick96/XSeg full-model checkpoint
  compatibility is NOT_YET_IMPLEMENTED (Phases 6-8) and facelib
  extractor-model compatibility is Phase 9 (their real files
  validate the FORMAT contract here). Parity: EXACT (pure index
  rearrangement + value copy — torch.equal/np.array_equal; the GPU
  copy test is bit-exact, RTX 4090, skip-if-CPU-only).

Current counts (2026-09, after Phase 4): CUDA environment 278
passed; CPU environment 263 passed + 15 skipped.

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
