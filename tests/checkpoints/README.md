# Checkpoint tests

Checkpoint compatibility coverage required by
`IMPLEMENTATION_PLAN.md` sections 17-20:

- Official checkpoint convention: pickle-dict `*.npy` files with
  TF-style variable names, HWIO conv kernels, `ms_*`/`vs_*` optimizer
  state, `iters:0` iteration counter.
- Round trip: official save -> torch load -> inference comparison, and
  official save -> converter -> torch load -> inference comparison.
- Strict load policy: missing required key, shape mismatch, ambiguous
  mapping, and corrupt file must each be a hard error; partial loading
  is only allowed behind an explicit user flag.
- Optimizer state: full state (weights + `ms`/`vs` + iterations,
  including GAN optimizer states) must survive save/load/resume;
  silent reinitialization is prohibited.
- Conversion report: mapped tensors, unmapped tensors, shape
  transforms, optimizer states mapped, source/destination hashes; zero
  unresolved required tensors for a successful conversion.

Created as a skeleton in Phase 0; test code is implemented from Phase 4.
