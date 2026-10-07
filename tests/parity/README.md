# Parity tests

TensorFlow-vs-Torch numerical parity coverage required by
`IMPLEMENTATION_PLAN.md` sections 21-22:

- **Core operations:** Conv2D, ConvTranspose, Dense, `depth_to_space`
  (TF R-R-C semantics; placement, not just shape), DSSIM, Gaussian
  blur, style loss (official moment matching), pixel norm
  (epsilon 1e-6), discriminator losses, optimizer updates.
- **Model modules:** encoder, inter, decoder, mask decoder, GAN
  discriminator, CodeDiscriminator.
- **Training:** forward outputs, individual loss terms, total loss,
  selected gradients, updated weights, optimizer iteration/state.
- **Checkpoint:** official save -> torch load -> inference comparison;
  official save -> converter -> torch load -> inference comparison.

Reference environment: an isolated official DFL / TensorFlow
environment, seeded and deterministic; Torch is the system under test.
The production application never imports TensorFlow.

Created as a skeleton in Phase 0; test code is implemented from Phase 5
(`depth_to_space` parity coverage is a P0 requirement of
`IMPLEMENTATION_PLAN.md` section 15).

## FaceEnhancer TensorFlow reference

`fixtures/face_enhancer_tf_reference.npz` is the authenticated, deterministic
CPU reference for the retained official FaceEnhancer at official DeepFaceLab
commit `14cc9d4e5ffc062856a739c8e64707654780774e`.  It contains only the tracked
official `facelib/FaceEnhancer.npy` checkpoint and deterministic synthetic
inputs; it contains no user media.

Generate an initial full-intermediate fixture for candidate measurement, run
the parity-only Torch comparison, then generate the compact reviewed fixture:

```powershell
.venv-tf\Scripts\python.exe tests\parity\generate_face_enhancer_tf_reference.py `
  --official-root .cache\phase14-faceenhancer-official-a `
  --checkpoint facelib\FaceEnhancer.npy `
  --output .cache\face_enhancer_tf_reference_full.npz

.venv\Scripts\python.exe tests\parity\generate_face_enhancer_tf_reference.py `
  --compare-torch `
  --checkpoint facelib\FaceEnhancer.npy `
  --fixture .cache\face_enhancer_tf_reference_full.npz `
  --comparison-output .cache\face_enhancer_torch_comparison.json

.venv-tf\Scripts\python.exe tests\parity\generate_face_enhancer_tf_reference.py `
  --official-root .cache\phase14-faceenhancer-official-a `
  --checkpoint facelib\FaceEnhancer.npy `
  --comparison-json .cache\face_enhancer_torch_comparison.json `
  --output tests\parity\fixtures\face_enhancer_tf_reference.npz
```

The generator refuses a non-root checkout, the wrong commit, tracked source
changes, an unexpected checkpoint inventory/digest, or a graph whose resize
attributes differ from the reviewed contract.  Under TensorFlow 2.21 graph
mode, the retained official `tf.image.resize` call emits `ResizeBilinear` with
`align_corners=false` and `half_pixel_centers=false`.  Consequently neither
stock PyTorch `align_corners` setting is a valid replacement; the fixture also
records measurements for a parity-only legacy-asymmetric candidate.

The executed FaceEnhancer reference environment is Python 3.12.11,
TensorFlow 2.21.0, NumPy 2.5.3, and OpenCV 5.0.0.  It is CPU-only NHWC graph
execution with seed 140014, one intra-op thread, one inter-op thread, and
`OMP_NUM_THREADS=1`.  It also sets `CUDA_VISIBLE_DEVICES=-1`,
`TF_ENABLE_ONEDNN_OPTS=0`, and `TF_DETERMINISTIC_OPS=1`.  The historical
requirements pinned TensorFlow GPU 2.4.0 and NumPy 1.19.3, but that historical
runtime was **not** the executed reference environment and is not claimed to
have been reproduced.

The authenticated official FaceEnhancer graph contains nine model-connected
`ResizeBilinear` sites.  All use `align_corners=false` and
`half_pixel_centers=false`.  Stock
`torch.nn.functional.interpolate(..., align_corners=False)` does not reproduce
these legacy-asymmetric coordinates; a production implementation must use an
equivalent legacy-asymmetric operation.  This evidence does not claim that the
production implementation already exists.

Reviewed CPU tolerances use `rtol=0` throughout:

| Field | Measured maximum absolute difference | `atol` |
| --- | ---: | ---: |
| Resize primitive | 1.1920929e-7 | 2e-7 |
| Raw network | 2.8192997e-5 | 5e-5 |
| Stitched/cropped | 2.8192997e-5 | 5e-5 |
| Normal final | 1.4096498e-5 | 2e-5 |
| Preserve-size | 7.6889992e-6 | 2e-5 |
| Merger-style tanh | 6.3553453e-6 | 2e-5 |

CUDA FaceEnhancer parity tolerance is **not finalized**.  No physical-CUDA
FaceEnhancer production implementation measurement exists yet; production
review must measure it and establish an evidence-based tolerance.  No previous
provisional ceiling is approved as a CUDA tolerance.

## Historical blur-sort sharpness reference

`fixtures/sharpness_skimage0142_reference.npz` freezes the authenticated
historical Canny/CPBD behavior used by blur sorting before its modernization.
It is evidence only: it does not provide or claim a production replacement.
All image content is deterministic and synthetic; no user media or reference
machine paths are stored.

The generator must run under the official packaged reference interpreter and
fails closed unless the environment is exactly Python 3.6.8, NumPy 1.19.3,
SciPy 1.4.1, scikit-image 0.14.2, and OpenCV 4.1.0.  OpenCV's runtime version
is checked fail-closed rather than merely recorded.  Before importing the
historical scorer, the generator reads the authenticated inputs as bytes,
verifies their full SHA-256 and Git blob identities, and verifies
installed-package metadata.  It then compiles the reviewed scorer, Canny, and
edges modules directly from those authenticated bytes through a source-only
loader.  The loader does not consult pre-existing bytecode caches, and module
identity checks prevent an entry already in `sys.modules` from bypassing the
authenticated source.  The reference is bound to authoritative DeepFaceLab commit
`e4b7543ffa1d73b26fce1e31852727f658ba490c` and these packaged files:

| Source | Packaged SHA-256 | Git blob SHA-1 |
| --- | --- | --- |
| `core/imagelib/estimate_sharpness.py` | `08b3eea0a30a9f41df7c0bf63d8f9387483c159c2a53276fb908d51e160fbc60` | `e4b3e2dce92cc55cf7bccea633f548db0557d40d` |
| `skimage/feature/_canny.py` | `3646016e5b56d60a94c3fec1d41625df16c5c6db71373859a88cdf94b5c74d15` | `1d685f202806eb286f6e1ff8166a6d123c887b6b` |
| `skimage/filters/edges.py` | `95adf833c66fe1fadf2efd6fabf0092d408881665e1db1c89c33d138bae5724a` | `471edad657afb75cf9959fdf12b586dd572a313e` |

Pass the package root explicitly; do not put a local reference path in tracked
files:

```powershell
<reference-root>\_internal\python-3.6.8\python.exe `
  tests\parity\generate_sharpness_skimage0142_reference.py `
  --reference-root <reference-root> `
  --output tests\parity\fixtures\sharpness_skimage0142_reference.npz
```

The NPZ schema is `sharpness-skimage0142-reference-v1`, schema version 1, and
the source-authenticated correction generator version is 2.  It stores
the source representation and historical float64 grayscale image, Boolean
Canny map, directional Sobel response, raw and thresholded squared response,
the CPBD local-maximum thinning map, edge-width map, probability map and
histogram, qualified-edge count, and exact float64 score for every case.  The
inventory covers constants, impulse and horizontal/vertical/diagonal edges,
multiple checkerboard sizes, controlled-width bars, gradient, edge-rich blur
sequences, odd/even dimensions, rank-2/one-channel/BGR/BGRA representations,
and face-like blur sequences.  Subgroup hashes bind the inputs, Canny maps,
CPBD intermediates, scores, ordering metadata, complete case/sorter inventories,
historical Canny/CPBD/tie contracts, derived summaries, privacy declarations,
scalar-tolerance status, and generator self-test records.  The validator owns
the authoritative top-level metadata key set; an unexpected, unbound metadata
claim is rejected rather than accepted from a fixture-provided allowlist.

Temporary synthetic DFL-compatible JPEGs with 68 landmarks exercise the two
historical scorer input contracts.  The fixture records descending public
blur-sort order after the BGR landmark mask and the internal
`sort_best(..., faster=False)` blur-preselection order after the grayscale
landmark mask.  Non-tied score ordering is a hard parity target.  For exact
ties, the fixture preserves only `GENERATOR_OBSERVED_TIE_ORDER`, classified as
`OBSERVED_REFERENCE_ORDER`, for its deterministic generated candidate list.
It does not establish filename order, original input order, or a stable
multiprocessing tie contract.  It intentionally does not invoke the accepted
final-by-blur parser alias or claim evidence for the later
yaw/pitch/histogram selection stages.

NPZ keys and ZIP members have a fixed order, timestamps are 1980-01-01, and
permissions are fixed.  Regeneration is accepted only when two independent
runs have identical whole-file SHA-256 values.  Generator self-validation
covers wrong Python, NumPy, SciPy, scikit-image, and OpenCV versions; wrong
source identities; substituted bytecode; altered inputs, Canny maps, every
CPBD intermediate (including thresholds and qualified-edge counts), all score
evidence, public and internal sorter inputs/scores/orders/mappings, source and
environment metadata, schema/version, fixed ZIP metadata, object-array
rejection, and both sort permutations.  Stored self-test results are
integrity-bound historical records, not substitutes for the validator's own
checks.  Validation independently recomputes the stored input, Canny, CPBD,
final-score, public-sort, internal-final-blur, source/environment,
contract-metadata, inventory-metadata, and summary-metadata subgroup hashes.
It also recomputes the score summary from the stored score arrays, checks the
contract structures against trusted constants, and independently checks the
payload's privacy properties instead of trusting stored summary or privacy
claims.

Scalar compatibility tolerances: **UNFINALIZED**.  A later implementation must
first measure exact Boolean edge-map and ordering parity and then justify any
float64 scalar tolerance from evidence; this baseline does not pre-loosen it.

The future Option B compatibility implementation is derived from the narrow
historical behavior of BSD-licensed scikit-image code.  Its implementation
review must retain the applicable scikit-image copyright and BSD license
notice in the derived source or an accompanying attribution notice.  The CPBD
scorer's existing Arizona Board of Regents/IVU license and citation notice must
also remain intact.  Whether the production change uses an in-file notice or a
separate notice is deliberately left for that implementation review.
