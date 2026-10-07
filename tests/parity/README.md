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
