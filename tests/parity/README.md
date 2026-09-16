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
