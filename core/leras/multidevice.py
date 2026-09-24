"""Multi-device (Phase 12) foundation: replica gradient aggregation
(Commit 1) and device-independent replica mirror management (Commit 2).

Commit 1 of the approved Phase 12 plan (docs/PHASE12_STATE.md, §16):
the torch port of the official DeepFaceLab ``nn.average_gv_list`` op.

Provenance
----------
Independent reimplementation. Semantic source: the official TF
implementation of ``average_gv_list`` (official commit ``e4b7543``),
preserved verbatim in this repository under ``core/leras/ops``
(``ops_tf.py``) — the official TF code is the authority on the
semantics; the code is NOT copied, and no external port was used.

Official semantics reproduced (torch-native execution):
- Each replica contributes the batch-SUM gradients over its own
  samples (differentiated through that replica's weights).
- Across replicas, per variable, the gradients are reduced by
  MEAN over the replica axis:

      G_final = (1/N) * sum_r G_r

  computed elementwise per variable: the per-replica gradients of one
  variable are stacked along a new leading replica axis and reduced
  by ``mean`` over that axis — the torch analogue of the official
  ``tf.reduce_mean(tf.concat([expand_dims(g, 0) for g in gs], 0), 0)``.
- N == 1: identity. The single replica list is returned unchanged
  (same list object, same gradient/parameter objects; the official
  op returns ``grad_var_list[0]`` without inspecting it).

Commit-1 contract (deliberately narrow; the approved plan's Commit 2
owns mirror management and cross-device transfer):
- ``replica_gv_lists`` is a list of per-replica ``[(grad, param), ...]``
  lists (one ``grad/param`` pair per trainable variable, in the same
  order on every replica).
- For N > 1 every replica list is validated before any aggregation
  and the call fails loudly (``ReplicaGradientError``):
  * an empty ``replica_gv_lists`` (at least one replica list is
    required);
  * a replica list with a different number of pairs than replica 0;
  * a ``None`` gradient (the official op raises when a variable has no
    flowing gradient; no silent skip, no averaging over the non-None
    replicas only);
  * a non-tensor gradient, or a gradient whose layout is not
    ``torch.strided`` — ONLY dense strided gradients are accepted;
    every sparse/compressed-sparse layout (COO, CSR, CSC, BSR, BSC,
    ...) is rejected (not part of the DFL training semantics; no
    silent densification);
  * a variable entry whose parameter object differs from replica 0's
    parameter at the same position (all replica entries of one logical
    variable must reference the canonical parameter object; if a later
    commit provides mirror parameters, the normalized canonical
    mapping must be applied BEFORE this call);
  * the same parameter object at two positions of one replica list
    (ambiguous variable association);
  * gradient shapes that differ across replicas for one variable
    (no broadcasting);
  * gradient dtypes that differ across replicas for one variable
    (no implicit cast);
  * replica gradients for one variable living on different devices —
    the aggregation device is the device of replica 0's gradient for
    that variable; Commit 1 performs NO cross-device transfer and no
    mirror construction, so all replica gradients must already be
    normalized onto the aggregation/canonical device by the caller.
- The returned list pairs each variable's mean gradient with the
  CANONICAL parameter object (replica 0's parameter).
- N == 1: no validation is performed (official identity behavior).
- The helper does NO optimizer work (no clipping, no unscale, no
  parameter update) and makes no GradScaler assumptions; the caller
  installs the returned gradients (e.g. into ``param.grad``) and
  drives the optimizer.
- Finiteness is deliberately NOT validated here: the FP16 Class B
  flow (approved Phase 12 plan §6.1) deliberately passes SCALED
  gradients that may contain inf/nan so the caller's
  ``torch.amp.GradScaler`` can observe them during unscale. Finiteness
  rejection for fp32/bf16 gradients is the caller's (the model
  closure's) responsibility per the approved plan; a blanket
  ``torch.isfinite`` check inside this helper would make the Class B
  flow impossible and is therefore absent by design.

Commit 2 (this module, below the Commit-1 section): device-independent
replica mirror management, per the approved Phase 12 plan
(docs/PHASE12_STATE.md §10.1, normative replica contracts):

- **Replica plan** (``ReplicaPlan``): the ordered replica device list.
  Replica 0 is the CANONICAL/primary replica (``devices[0]`` == the
  Phase 2 ``nn.device`` primary-device contract); replicas r > 0 are
  secondary replicas. Production plans are derived from a Phase 2
  ``DeviceConfig`` through the backend-neutral device abstraction
  (``from_device_config`` — CPU-only is the single-replica case); no
  direct ``torch.cuda.*`` call appears in this module, and the plan
  NEVER reads or writes the global ``nn.device``.
- **Mirror construction** (§10.1.1): for each non-primary replica
  device, ONE mirror set per trainable component, produced by the
  PERMITTED module-level factory ``copy.deepcopy(canonical_module).to
  (target_device)`` — which yields structurally identical mirror
  modules with DISTINCT new Parameters (verified on real
  ``DeepFakeArchi`` hierarchies), then the mirror's MODULE TREE is
  re-validated against the canonical's (see **Module-tree
  validation** below) so that a factory defect that merely reproduces
  the parameter paths cannot produce a structurally invalid mirror.
  Mirrors are DISPOSABLE RUNTIME
  state: never checkpoint-owned, never saved, never resumed; a
  later run reconstructs them from the canonical modules.
  Construction is a pure per-component operation: it reads/writes NO
  global state (in particular it never temporarily changes
  ``nn.device``) and must run only AFTER the checkpoint load and all
  canonical initialization.
- **Parameter mapping** (§10.1.2): the deterministic alias-aware
  contract. Canonical and mirror modules are enumerated with
  ``named_parameters(remove_duplicate=False)``; the ordered
  module-local logical path lists must be IDENTICAL (missing,
  reordered, or duplicated paths are a construction-time error);
  per path the mirror parameter must have the canonical's shape,
  dtype, ``requires_grad`` and live on the replica's target device.
  The mapping is keyed by MODULE-LOCAL PATHS, never by copied
  ``_dfl_name`` metadata (module-level ``copy.deepcopy`` does not
  reliably preserve custom Parameter attributes; a canonical
  parameter that carries ``_dfl_name`` is checked as an ADDITIONAL
  construction-time invariant only). The optimizer owns CANONICAL
  parameters only.
- **Module-tree validation**: after the ``copy.deepcopy`` factory,
  matching parameter paths are NOT sufficient for structural
  validity — the canonical and mirror module trees are compared via a
  complete traversal (``named_modules(remove_duplicate=False)``):
  identical ORDERED module path lists (missing / extra / reordered
  submodule paths are a construction-time error; identical paths also
  fix the identical module count and the parent/child structure
  implied by them), the identical CONCRETE module type at every path
  (a mirror rebuilt as plain ``torch.nn.Module``s with the same
  parameter paths is refused), and the identical per-submodule
  ``.training`` flag at every path (the top-level training flag alone
  is NOT sufficient — one submodule flipped to the other mode is a
  structural mismatch).
- **Alias rejection — GLOBAL, any-to-any, base-storage aware**
  (fail-fast at construction; Phase 12 defines no safe aliasing
  policy): alias safety is validated GLOBALLY, not only for same-path
  canonical/mirror pairs. Explicit registries are built for: the
  canonical parameter object identities and base-storage identities,
  each mirror's parameter object/storage identities, the canonical
  buffer object/storage identities, and each mirror's buffer
  object/storage identities — and the plan keeps accumulating
  registries of ALL of them across every registered component.
  Rejected in every form: two distinct canonical parameters sharing
  one storage; two distinct mirror parameters/buffers sharing one
  storage; a canonical tensor at path A sharing storage with a mirror
  tensor at path B (any cross-path canonical<->mirror overlap,
  parameter<->parameter, buffer<->buffer, or parameter<->buffer);
  storage shared between different secondary mirrors; overlapping
  storage or shared tensor object identity across two registered
  components; duplicate parameter/buffer object identity across
  logical paths (inside one tree, or a mirror tensor IS-ing a
  canonical tensor object at any path). The storage checks reason on
  BASE-storage identity (``untyped_storage().data_ptr()``): two
  tensors at different view offsets of the same underlying storage
  are aliasing and ARE caught, whereas comparing each view's own
  ``data_ptr()`` would miss them. All traversal is alias-aware:
  ``named_parameters(remove_duplicate=False)`` and
  ``named_buffers(remove_duplicate=False)`` — the default
  deduplicating ``named_buffers()`` behavior is never used (it would
  silently hide the same buffer object registered under several
  logical paths; buffer declaration/classification therefore always
  operates on logical paths).
- **Buffer policy** (§10.1.4): mirrors track registered buffers
  generically; each buffer is classified at replica-plan
  construction — class A (immutable during training; copied
  canonical -> mirror at the initial sync and re-synced by the same
  canonical -> mirror sync; a buffer is class A only when the
  component explicitly declares it, e.g. ``BatchNorm2D``'s
  inference-only running statistics declare it via
  ``_dfl_buffer_classes`` or the caller passes an explicit
  ``{buffer path: class}`` registry) or class B (mutable replica-local
  state — supported ONLY via an explicit declaration plus a
  registered reduction/synchronization rule; no rule is registered in
  Phase 12, so a class-B declaration fails loud). EVERY registered
  buffer of a mirrored component must be classified; the default for
  an UNCLASSIFIED buffer is class C -> FAIL-FAST at construction. No
  name/usage heuristics are ever applied.
- **Sync** (§10.1.5): the initial canonical -> mirror sync (after
  construction + checkpoint load + canonical init, before the first
  replica forward) and the post-successful-step sync use the SAME
  deterministic mechanism (``initial_sync()`` /
  ``sync_from_canonical()``): under ``no_grad`` it copies parameter
  data and class-A buffers from the canonical module into every
  mirror on every non-primary replica device. Sync happens ONLY after
  a successful canonical step; a skipped step must not call it.
  Sync never touches optimizer state, never registers mirrors with
  any optimizer, never creates Saveable/checkpoint entries.
- **Checkpoint isolation** (B7): mirrors are never Saveable children
  of any checkpoint component and are never entries of a model's
  ``model_filename_list``; no mirror parameter/buffer appears in any
  canonical Saveable's weight enumeration or saved file, and mirror
  construction creates no new checkpoint keys or filename/state
  names.
- **Gradient installation foundation** (§10.1.3):
  ``ReplicaPlan.canonicalize_grads(replica_grad_lists)`` validates
  that replica 0's entries reference CANONICAL parameters and that
  each secondary replica's entries reference the MIRROR counterpart
  of the same canonical parameter, transfers the secondary replica
  gradients to the canonical device (replica 0's gradients are
  already canonical), and returns per-replica lists rebinding every
  gradient to the CANONICAL parameter object — the normalized form
  that ``average_gv_list`` (Commit 1) accepts. This performs NO
  averaging, NO optimizer work, and NO GradScaler logic.
- **Level-B simulation**: tests may map several LOGICAL replicas onto
  one PHYSICAL device (``ReplicaPlan.from_torch_devices``); such
  results are labeled ``SIMULATED_MULTI_REPLICA`` and physical
  multi-GPU acceptance stays ``PENDING_ENVIRONMENTALLY /
  NOT_VERIFIED``.

Commit 3 (below the Commit-2 section): mixed-precision validation
ACROSS REPLICA DEVICES (Phase 12 §6/§6.1):

- **All-device precision validation** (§6, Milestone E): the
  requested mode is validated against EVERY selected replica device
  (``mixed_precision.resolve_precision_devices`` — fp16 needs
  >= (5,3) on each CUDA device, bf16 >= (8,0) on each CUDA device
  or a CPU device, ``off`` anywhere); an unsupported mode on ANY
  selected device raises ``PrecisionUnsupportedError`` naming every
  failing device — NO silent per-device fallback. The returned plan
  is the ONE global ``PrecisionPlan`` of the run; each replica's
  autocast region pins its own device type via
  ``autocast_context(device_type=...)``.
- **One global GradScaler** (§6): there is exactly ONE model-level
  ``torch.amp.GradScaler`` (the fp16 plan's native-default scaler;
  off/bf16 plans carry NONE) — never one scaler per replica, per
  mirror, or per device. ``run_replica_precision_step`` refuses any
  plan/scaler combination that violates this (an fp16 plan with
  no scaler, or an off/bf16 plan with a scaler).
- **Canonical-side precision step** (§6.1/§10.1.3,
  ``run_replica_precision_step``): Commit-2
  ``canonicalize_grads`` (association + secondary mirror grads
  transferred to the canonical device, rebound to the canonical
  parameters) → STRUCTURAL validation before any value-level policy
  (Commit-1 ``average_gv_list`` layout/shape/dtype/None contract at
  N>1; entry validation at N==1 — sparse/compressed-sparse
  gradients always fail through ``ReplicaGradientError``) →
  mode-aware PRE-unscale nonfinite policy over the structurally
  valid grads (fp16: nonfinite SCALED grad values are NOT
  rejected — a numeric overflow, not a layout defect: they pass
  through canonical ``.grad`` installation so the GradScaler owns
  overflow detection; off/bf16: hard error, matching the
  single-device behavior) → ``average_gv_list`` (per-variable MEAN
  over replicas; identity at N==1) → the averaged (scaled) grads
  installed on the canonical ``.grad`` → exactly ONE canonical
  ``unscale`` / ``step`` / ``update`` sequence per attempt,
  executed THROUGH THE INJECTED MODEL-LAYER HOOKS (the existing
  ``ModelBase._mp_unscale_opt`` / ``_mp_opt_step`` /
  ``_mp_scaler_update`` bound methods supplied by the model — this
  module never calls native scaler methods itself and never
  imports ``models/``; off/bf16 hooks are the no-op / direct
  ``get_update_op`` semantics). A skipped canonical step (Class B
  overflow) clears ALL canonical + secondary mirror grads before
  returning ``False`` — the caller raises ``SkippedGeneratorStep``
  (model layer) and the existing ``_mp_run_generator`` retry policy
  applies (16 bounded attempts under fp16, 1 under off/bf16).
  Likewise the per-replica SCALED backward is the model layer's
  existing ``ModelBase._mp_backward`` (the one global scaler
  scales every replica's per-sample loss vector; off/bf16 stays
  unscaled) — no second scaler path exists.
- **Class A / Class B separation** (§6.1): a nonfinite forward
  output or loss on ANY replica is Class A — the attempt closure
  clears the canonical + all mirror grads (``clear_replica_grads``)
  and raises the hard ``FloatingPointError`` (never retried, never
  scaler-recorded, never a ``SkippedGeneratorStep``); Class B is
  the GradScaler's overflow bookkeeping only.
- **Replica gradient ownership** (§6.1): replica 0 (primary)
  computes on the CANONICAL parameters (it has no mirror); replica
  r > 0 computes on its MIRROR parameters. ``replica_param_lists``
  returns the per-replica parameter lists that own each replica's
  gradients (path-aligned through the Commit-2 §10.1.2 mapping);
  ``clear_replica_grads`` is the complete attempt-start / Class A /
  Class B grad cleanup over canonical + all mirrors.
"""

import copy

import torch

from .checkpoint import official_name as _official_name
from .mixed_precision import PrecisionPlan as _PrecisionPlan

__all__ = [
    "average_gv_list",
    "ReplicaGradientError",
    "ReplicaPlanError",
    "ReplicaPlan",
    "ComponentMirrorSet",
    "build_replica_mirrors",
    "ReplicaPrecisionError",
    "clear_replica_grads",
    "replica_param_lists",
    "run_replica_precision_step",
]


class ReplicaGradientError(ValueError):
    """The per-replica gradient lists violate the Commit-1 aggregation
    contract (structure, layout, association, shape, dtype, or device)."""


def _validate_entry(replica, position, entry):
    """Validate one ``(grad, param)`` entry; returns ``(grad, param)``.

    Raises ``ReplicaGradientError`` for a malformed entry (wrong arity,
    ``None`` / non-tensor / non-strided-layout gradient,
    non-``Parameter`` variable).
    """
    try:
        grad, param = entry
    except (TypeError, ValueError):
        raise ReplicaGradientError(
            f"average_gv_list: replica {replica} variable {position}: entry "
            f"{entry!r} is not a (gradient, parameter) pair"
        ) from None
    if grad is None:
        raise ReplicaGradientError(
            f"average_gv_list: replica {replica} variable {position}: gradient "
            f"is None — every expected gradient must be present (no silent "
            f"skip, no averaging over the non-None replicas only)"
        )
    if not torch.is_tensor(grad):
        raise ReplicaGradientError(
            f"average_gv_list: replica {replica} variable {position}: gradient "
            f"must be a torch tensor (got {type(grad).__name__})"
        )
    # Layout gate: ONLY torch.strided (dense) gradients are accepted.
    # A blanket ``layout == torch.strided`` check (instead of
    # ``is_sparse``) is deliberate: torch reports ``is_sparse == False``
    # for compressed-sparse layouts such as CSR/CSC, so the layout is
    # the only reliable discriminator. Every sparse/compressed-sparse
    # layout (COO, CSR, CSC, BSR, BSC, ...) is rejected — no sparse or
    # compressed-sparse tensor may reach ``torch.stack``, and no silent
    # densification is ever performed.
    if grad.layout != torch.strided:
        raise ReplicaGradientError(
            f"average_gv_list: replica {replica} variable {position}: gradient "
            f"layout {grad.layout} is not torch.strided — only dense strided "
            f"gradients are supported; sparse and compressed-sparse layouts "
            f"(e.g. COO/CSR/CSC) are rejected, with no silent densification"
        )
    if not isinstance(param, torch.nn.Parameter):
        raise ReplicaGradientError(
            f"average_gv_list: replica {replica} variable {position}: the "
            f"variable must be a torch.nn.Parameter (got {type(param).__name__})"
        )
    return grad, param


def average_gv_list(replica_gv_lists):
    """Official-compatible replica gradient aggregation (Phase 12, Commit 1).

    Torch port of the official DFL ``nn.average_gv_list`` op (see the
    module docstring for the full contract).

    Args:
        replica_gv_lists: list of per-replica gradient/variable lists.
            Each element is a list of ``(grad, param)`` pairs, one per
            trainable variable, in the same order on every replica.
            Each ``grad`` is the batch-SUM gradient over that replica's
            samples for that variable.

    Returns:
        N == 1: the single replica list, returned unchanged (identity).
        N > 1: a new list of ``(mean_grad, canonical_param)`` pairs in
        replica-0 order, where ``mean_grad`` is the per-element mean
        over the replica axis (the torch analogue of the official
        ``tf.reduce_mean(tf.concat([expand_dims(g, 0) for g in gs], 0), 0)``)
        and ``canonical_param`` is replica 0's parameter object.

    Raises:
        ReplicaGradientError: N > 1 and any replica list violates the
            Commit-1 contract (structure, layout, association, shape,
            dtype, or device — see the module docstring).
    """
    if not replica_gv_lists:
        raise ReplicaGradientError(
            "average_gv_list: replica_gv_lists must contain at least one "
            "replica gradient/variable list"
        )
    if len(replica_gv_lists) == 1:
        # Official identity behavior: the single replica list is
        # returned unchanged (same list object, same objects inside).
        return replica_gv_lists[0]

    n_replicas = len(replica_gv_lists)
    n_vars = len(replica_gv_lists[0])

    # ---- structural validation: every entry of every replica list ----
    for r in range(n_replicas):
        gv_list = replica_gv_lists[r]
        if len(gv_list) != n_vars:
            raise ReplicaGradientError(
                f"average_gv_list: replica {r} has {len(gv_list)} "
                f"gradient/parameter pair(s); replica 0 has {n_vars} — all "
                f"replica lists must have the same number of pairs"
            )
        seen_positions = {}
        for i in range(n_vars):
            _, param = _validate_entry(r, i, gv_list[i])
            key = id(param)
            if key in seen_positions:
                raise ReplicaGradientError(
                    f"average_gv_list: replica {r}: the same parameter object "
                    f"appears at both position {seen_positions[key]} and "
                    f"position {i} (ambiguous variable association)"
                )
            seen_positions[key] = i

    # ---- per-variable cross-replica validation + aggregation ----------
    result = []
    for i in range(n_vars):
        ref_grad, canonical = _validate_entry(0, i, replica_gv_lists[0][i])
        grads = [ref_grad]
        for r in range(1, n_replicas):
            grad, param = _validate_entry(r, i, replica_gv_lists[r][i])
            if param is not canonical:
                raise ReplicaGradientError(
                    f"average_gv_list: variable {i}: replica {r} references a "
                    f"different parameter object than replica 0 — all replica "
                    f"entries of one logical variable must reference the "
                    f"canonical parameter object (mirror parameters must be "
                    f"normalized onto the canonical association before "
                    f"aggregation; Phase 12 Commit 2)"
                )
            if grad.shape != ref_grad.shape:
                raise ReplicaGradientError(
                    f"average_gv_list: variable {i}: replica {r} gradient "
                    f"shape {tuple(grad.shape)} differs from replica 0 "
                    f"shape {tuple(ref_grad.shape)}"
                )
            if grad.dtype != ref_grad.dtype:
                raise ReplicaGradientError(
                    f"average_gv_list: variable {i}: replica {r} gradient "
                    f"dtype {grad.dtype} differs from replica 0 dtype "
                    f"{ref_grad.dtype} (no implicit cast)"
                )
            if grad.device != ref_grad.device:
                raise ReplicaGradientError(
                    f"average_gv_list: variable {i}: replica {r} gradient is on "
                    f"device {grad.device} but replica 0's is on "
                    f"{ref_grad.device} — Commit 1 performs no cross-device "
                    f"transfer; the caller must normalize all replica "
                    f"gradients onto the aggregation/canonical device first"
                )
            grads.append(grad)
        # Mean over the replica axis only: the per-replica grads are the
        # batch SUMs, so this is exactly the official
        # reduce_mean(concat, 0) — no second division, dtype/shape/device
        # of the replica gradients preserved.
        result.append((torch.stack(grads, 0).mean(dim=0), canonical))
    return result


# ---------------------------------------------------------------------------
# Commit 2: device-independent replica mirror management
# ---------------------------------------------------------------------------

class ReplicaPlanError(ValueError):
    """The replica plan or mirror construction violates the approved
    Phase 12 mirror contract (structure, device, alias, buffer, or
    lifecycle precondition)."""


class ComponentMirrorSet:
    """One trainable component plus its per-replica mirrors and the
    deterministic canonical<->mirror parameter/buffer mapping (§10.1.2).

    The object is immutable after construction (created by
    :func:`build_replica_mirrors` / :meth:`ReplicaPlan.add_component`);
    mirrors are DISPOSABLE runtime state owned by the :class:`ReplicaPlan`
    — they are never checkpoint-owned, never saved, never resumed.

    ``object_ids`` / ``storage_ptrs`` are the GLOBAL alias registries of
    this component (every canonical + mirror parameter/buffer object
    identity, and every non-empty base-storage identity): the plan uses
    them to refuse overlapping storage / shared objects ACROSS
    registered components.
    """

    def __init__(self, plan, canonical, mirrors, paths,
                 canonical_params, mirror_params,
                 canonical_buffers, mirror_buffers, buffer_classes,
                 object_ids, storage_ptrs):
        self._plan = plan
        self.canonical = canonical
        # index r-1 of ``mirrors`` is the mirror on replica r (r > 0)
        self.mirrors = mirrors
        # ordered module-local logical paths (identical on every replica)
        self.paths = paths
        self.canonical_params = canonical_params
        self.mirror_params = mirror_params  # {replica r: [mirror params]}
        self.canonical_buffers = canonical_buffers
        self.mirror_buffers = mirror_buffers  # {replica r: [mirror buffers]}
        # resolved buffer classes: {buffer path: 'A'} (class B is
        # unsupported in Phase 12 and fails at construction)
        self.buffer_classes = buffer_classes
        self.is_synced = False
        # global alias registries (see the class docstring)
        self.object_ids = object_ids
        self.storage_ptrs = storage_ptrs

    @property
    def plan(self):
        return self._plan

    @property
    def num_mirrors(self):
        return len(self.mirrors)


def _base_storage_ptr(tensor):
    """BASE-storage identity of a tensor, or ``None`` for empty storage.

    ``tensor.untyped_storage()`` returns the storage the tensor is a
    view of; ``.data_ptr()`` on THAT storage is the storage's base
    address — NOT the view's own (possibly offset) data pointer. Two
    tensors at DIFFERENT view offsets of the same underlying storage
    therefore report the SAME base pointer here, while distinct
    storages never collide. This is what makes the alias checks below
    view/offset-aware instead of comparing each view's own
    ``data_ptr()`` (which a non-zero view offset would defeat).
    """
    storage = tensor.untyped_storage()
    if not storage.size():
        return None  # empty storage: no base block that could alias
    return storage.data_ptr()


def _check_module_tree(canonical, mirror, replica):
    """Structural module-tree validation (§10.1.1): after the
    ``copy.deepcopy`` factory, the mirror must be the SAME tree as the
    canonical's, not merely a module that happens to expose identical
    parameter paths. Comparing parameter/buffer path lists alone is
    insufficient — a mirror rebuilt from PLAIN ``torch.nn.Module``s,
    with swapped/omitted/reordered submodules, can carry identical
    parameter paths and still be structurally invalid.

    Validated via a complete alias-aware traversal
    (``named_modules(remove_duplicate=False)``):

    - identical ORDERED module path lists (missing / extra / reordered
      submodule paths are a construction-time error) — which also fixes
      the identical module count and the parent/child structure implied
      by the paths;
    - the identical CONCRETE module type at every path
      (``type(mirror_module) is type(canonical_module)``);
    - the identical per-submodule ``.training`` flag at every path
      (the top-level flag alone is insufficient: one submodule flipped
      to the other mode is a structural mismatch).
    """
    canon_modules = list(canonical.named_modules(remove_duplicate=False))
    mirror_modules = list(mirror.named_modules(remove_duplicate=False))
    canon_paths = [name for name, _ in canon_modules]
    mirror_paths = [name for name, _ in mirror_modules]
    if mirror_paths != canon_paths:
        missing = [p for p in canon_paths if p not in set(mirror_paths)]
        extra = [p for p in mirror_paths if p not in set(canon_paths)]
        if set(mirror_paths) == set(canon_paths):
            problem = ("module paths are REORDERED relative to the "
                       "canonical component")
        else:
            problem = (f"module paths are not identical to the canonical "
                       f"component's (missing: {missing}, extra: {extra})")
        raise ReplicaPlanError(
            f"build_replica_mirrors: replica {replica} {problem} — the "
            f"mirror must be a structurally identical module tree "
            f"(named_modules traversal, remove_duplicate=False): "
            f"identical ordered module paths, identical module count and "
            f"parent/child structure"
        )
    for (canon_path, canon_module), (_, mirror_module) in zip(
            canon_modules, mirror_modules):
        # (the path lists were validated identical, element by element,
        # above; only the module objects are compared here)
        if type(mirror_module) is not type(canon_module):
            raise ReplicaPlanError(
                f"build_replica_mirrors: module '{canon_path}' on replica "
                f"{replica} is a {type(mirror_module).__name__} but the "
                f"canonical submodule is a {type(canon_module).__name__} — "
                f"the mirror must preserve the CONCRETE module type at "
                f"every path (matching parameter paths are not "
                f"sufficient for structural validity)"
            )
        if mirror_module.training != canon_module.training:
            raise ReplicaPlanError(
                f"build_replica_mirrors: module '{canon_path}' on replica "
                f"{replica} has training={mirror_module.training} but the "
                f"canonical submodule has training={canon_module.training} "
                f"— the mirror must preserve the PER-SUBMODULE "
                f"training/eval state (the top-level flag alone is not "
                f"validated as sufficient)"
            )


def _check_canonical_tree(plan, canonical):
    """Canonical-side fail-fast checks (§10.1.2 / §10.1.4).

    Buffer traversal is ALIAS-AWARE: ``named_buffers`` is used with
    ``remove_duplicate=False`` (the default deduplicating behavior would
    silently hide the same buffer object registered under several
    logical paths); buffer declaration/classification therefore always
    operates on LOGICAL PATHS.

    Returns ``(param_pairs, buffer_pairs, storage_owners, object_ids)``:
    ``param_pairs`` / ``buffer_pairs`` are in
    ``named_parameters`` / ``named_buffers``
    (``remove_duplicate=False``) registration order; ``storage_owners``
    is the component's GLOBAL base-storage registry
    (base pointer -> owner description, any-to-any over parameters AND
    buffers); ``object_ids`` the set of every canonical tensor object
    identity. Raises ``ReplicaPlanError`` on:
    non-module canonical; duplicate logical paths; duplicate object
    identity inside the canonical tree (two logical paths resolving to
    the same tensor object — parameter vs parameter, buffer vs buffer,
    or parameter vs buffer; Phase 12 defines no safe mirror policy for
    aliased tensors); two distinct canonical tensors sharing one BASE
    storage (including view/offset aliases of the same underlying
    storage block — the check reasons on base-storage identity, not on
    the view's own data_ptr); or canonical state not co-located on the
    plan's primary device (the Phase 2 co-location model).
    """
    if not isinstance(canonical, torch.nn.Module):
        raise ReplicaPlanError(
            f"build_replica_mirrors: the canonical component must be a "
            f"torch.nn.Module (got {type(canonical).__name__})"
        )
    param_pairs = list(canonical.named_parameters(remove_duplicate=False))
    paths = [name for name, _ in param_pairs]
    if len(set(paths)) != len(paths):
        raise ReplicaPlanError(
            "build_replica_mirrors: duplicate logical parameter paths in the "
            f"canonical component — a path must identify exactly one "
            f"parameter (duplicate detection, §10.1.2)"
        )
    param_ids = [id(p) for _, p in param_pairs]
    if len(set(param_ids)) != len(param_ids):
        raise ReplicaPlanError(
            "build_replica_mirrors: duplicate object identity inside the "
            "canonical component (two logical paths resolve to the same "
            "Parameter object) — Phase 12 defines no safe mirror policy "
            "for aliased parameters; construction is refused"
        )
    buffer_pairs = list(canonical.named_buffers(recurse=True,
                                                remove_duplicate=False))
    buffer_paths = [name for name, _ in buffer_pairs]
    if len(set(buffer_paths)) != len(buffer_paths):
        raise ReplicaPlanError(
            "build_replica_mirrors: duplicate logical buffer paths in the "
            f"canonical component (duplicate detection, §10.1.2)"
        )
    buffer_ids = [id(b) for _, b in buffer_pairs]
    if len(set(buffer_ids)) != len(buffer_ids):
        raise ReplicaPlanError(
            "build_replica_mirrors: duplicate object identity inside the "
            "canonical component (two logical buffer paths resolve to the "
            "same buffer object) — Phase 12 defines no safe mirror policy "
            "for aliased buffers; construction is refused"
        )
    # any-to-any object identity across parameters AND buffers: a tensor
    # object registered as both a parameter and a buffer is aliasing too
    param_id_set = set(param_ids)
    for path, b in buffer_pairs:
        if id(b) in param_id_set:
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{path}' IS a canonical "
                "parameter object (two logical paths resolve to the same "
                "tensor object) — Phase 12 defines no safe mirror policy "
                "for aliased tensors; construction is refused"
            )
    # GLOBAL base-storage registry, any-to-any over parameters AND
    # buffers (distinct tensors must own distinct base storage; views at
    # different offsets of one shared storage are aliasing as well)
    storage_owners = {}
    all_canon_tensors = [(f"parameter '{n}'", t)
                         for n, t in param_pairs] + \
                        [(f"buffer '{n}'", t) for n, t in buffer_pairs]
    for path, t in all_canon_tensors:
        ptr = _base_storage_ptr(t)
        if ptr is None:
            continue  # empty storage: no base block to alias
        if ptr in storage_owners:
            raise ReplicaPlanError(
                f"build_replica_mirrors: canonical {path} shares the base "
                f"storage of {storage_owners[ptr]} inside the canonical "
                f"component — two distinct registered tensors must never "
                f"use one storage (the check covers base-storage "
                f"identity, including view/offset aliases); Phase 12 "
                f"supports no such aliasing; construction is refused"
            )
        storage_owners[ptr] = f"canonical {path}"
    object_ids = set(param_ids) | set(buffer_ids)
    primary = plan.primary_device
    for name, p in param_pairs:
        if p.device != primary:
            raise ReplicaPlanError(
                f"build_replica_mirrors: canonical parameter '{name}' is on "
                f"{p.device} but the plan's primary device is {primary} — "
                f"the Phase 2 co-location model requires ALL canonical "
                f"state on the primary device"
            )
    for name, b in buffer_pairs:
        if b.device != primary:
            raise ReplicaPlanError(
                f"build_replica_mirrors: canonical buffer '{name}' is on "
                f"{b.device} but the plan's primary device is {primary} — "
                f"the Phase 2 co-location model requires ALL canonical "
                f"state on the primary device"
            )
    return param_pairs, buffer_pairs, storage_owners, object_ids


def _check_dfl_name_consistency(obj, path, scope, what):
    """Additional (never keying) construction-time invariant (§10.1.2):
    when a canonical parameter/buffer carries ``_dfl_name`` (the
    optimizer-state official-name binding), the name must be
    consistent with the official name the optimizer binding expects
    at that module-local path. The mapping itself is keyed by the
    module-local path, never by ``_dfl_name``."""
    dfl_name = getattr(obj, "_dfl_name", None)
    if dfl_name is None:
        return
    expected = (scope + "/" + _official_name(path)
                if scope is not None else _official_name(path))
    if dfl_name != expected:
        raise ReplicaPlanError(
            f"build_replica_mirrors: {what} '{path}' carries _dfl_name "
            f"{dfl_name!r}, which is not consistent with the official "
            f"name the optimizer binding expects at that path "
            f"({expected!r}) — _dfl_name is checked as an additional "
            f"invariant; the canonical<->mirror mapping is keyed by the "
            f"module-local path"
        )


def _resolve_buffer_classes(canonical, buffer_pairs, declared):
    """Classify EVERY registered buffer (§10.1.4). No heuristics: a
    buffer is class A only when explicitly declared — either by the
    caller's ``{buffer path: class}`` registry (checked first) or by
    the OWNING module class's ``_dfl_buffer_classes`` attribute (the
    component's own explicit declaration, e.g. ``BatchNorm2D``).
    Anything else is class C and fails fast; class B requires a
    registered reduction/synchronization rule, of which Phase 12 has
    none, so a B declaration also fails loud."""
    owner_by_id = {}
    for module in canonical.modules():
        # alias-aware: the default deduplicating behavior could pick an
        # arbitrary local name for a buffer object registered under
        # several names (duplicate identity is rejected upstream, this
        # is defense-in-depth for the ownership bookkeeping)
        for local_name, buf in module.named_buffers(recurse=False,
                                                    remove_duplicate=False):
            owner_by_id[id(buf)] = (module, local_name)
    resolved = {}
    for buffer_path, buf in buffer_pairs:
        cls = None
        if declared is not None and buffer_path in declared:
            cls = declared[buffer_path]
        if cls is None:
            owner = owner_by_id.get(id(buf))
            if owner is not None:
                decl = getattr(type(owner[0]), "_dfl_buffer_classes", None)
                if isinstance(decl, dict):
                    cls = decl.get(owner[1])
        if cls == "A":
            resolved[buffer_path] = "A"
        elif cls == "B":
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{buffer_path}' is declared "
                "class B (mutable replica-local state), but Phase 12 "
                "registers no reduction/synchronization rule for class-B "
                "buffers — only class A is supported; remove the "
                "declaration (or register the rule in a later commit)"
            )
        elif cls is None:
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{buffer_path}' is "
                "UNCLASSIFIED -> class C: every registered buffer of a "
                "mirrored component must be explicitly declared 'A' "
                "(immutable during training) or 'B' (replica-local with a "
                "registered synchronization rule); the plan refuses to "
                "build so no silent divergence can occur at runtime"
            )
        else:
            raise ReplicaPlanError(
                f"build_replica_mirrors: invalid buffer class {cls!r} for "
                f"buffer '{buffer_path}' (expected 'A' or 'B')"
            )
    return resolved


def _shared_storage_error(path, replica, detail):
    raise ReplicaPlanError(
        f"build_replica_mirrors: {detail} for '{path}' (replica {replica}) "
        "— a mirror must own DISTINCT storage: writing a mirror must never "
        "touch canonical (checkpoint-owned) state"
    )


def _build_one_mirror(plan, canonical, replica, param_pairs, buffer_pairs,
                      storage_owners, canon_obj_ids):
    """Construct + validate the mirror of ``canonical`` on one
    non-primary replica device (the §10.1.1 factory plus the §10.1.2
    alias-aware mapping checks plus the GLOBAL alias-safety and
    module-tree validations). ``storage_owners`` / ``canon_obj_ids``
    are the canonical tree's global base-storage registry and object
    identity set (from :func:`_check_canonical_tree`); the mirror is
    validated against them ANY-TO-ANY (any mirror tensor vs any
    canonical tensor, at any path), and its own tensors any-to-any
    inside the mirror tree. Returns
    ``(mirror, mirror_param_pairs, mirror_buffer_pairs)``."""
    device = plan.replica_devices[replica]
    # The PERMITTED factory: module-level copy.deepcopy + .to(target)
    # (distinct new Parameters/buffers, structure preserved; see the
    # module docstring and docs/PHASE12_STATE.md §10.1.1).
    mirror = copy.deepcopy(canonical).to(device)
    if mirror.training != canonical.training:
        raise ReplicaPlanError(
            f"build_replica_mirrors: replica {replica} mirror does not "
            f"preserve the canonical training/eval mode "
            f"(canonical {canonical.training!r} vs mirror {mirror.training!r})"
        )
    mirror_param_pairs = list(mirror.named_parameters(remove_duplicate=False))
    canonical_paths = [name for name, _ in param_pairs]
    mirror_paths = [name for name, _ in mirror_param_pairs]
    if mirror_paths != canonical_paths:
        missing = [p for p in canonical_paths if p not in set(mirror_paths)]
        extra = [p for p in mirror_paths if p not in set(canonical_paths)]
        if set(mirror_paths) == set(canonical_paths):
            problem = (f"paths are REORDERED relative to the canonical "
                       f"component")
        else:
            problem = (f"paths are not identical to the canonical "
                       f"component's (missing: {missing}, extra: {extra})")
        raise ReplicaPlanError(
            f"build_replica_mirrors: replica {replica} {problem} — the "
            f"module-local logical path contract (named_parameters, "
            f"remove_duplicate=False) requires the identical ordered path "
            f"list on every replica"
        )
    mirror_ids = [id(p) for _, p in mirror_param_pairs]
    if len(set(mirror_ids)) != len(mirror_ids):
        raise ReplicaPlanError(
            f"build_replica_mirrors: duplicate object identity inside the "
            f"replica {replica} mirror tree (two logical paths resolve to "
            f"the same Parameter object)"
        )
    for (path, cparam), (_, mparam) in zip(param_pairs, mirror_param_pairs):
        if mparam is cparam:
            raise ReplicaPlanError(
                f"build_replica_mirrors: parameter '{path}' on replica "
                f"{replica} IS the canonical parameter object — a mirror "
                f"must be a DISTINCT object; construction is refused"
            )
        if tuple(mparam.shape) != tuple(cparam.shape):
            raise ReplicaPlanError(
                f"build_replica_mirrors: parameter '{path}' (replica "
                f"{replica}) has shape {tuple(mparam.shape)} but the "
                f"canonical has {tuple(cparam.shape)}"
            )
        if mparam.dtype != cparam.dtype:
            raise ReplicaPlanError(
                f"build_replica_mirrors: parameter '{path}' (replica "
                f"{replica}) has dtype {mparam.dtype} but the canonical "
                f"has {cparam.dtype} — no implicit dtype cast"
            )
        if mparam.requires_grad != cparam.requires_grad:
            raise ReplicaPlanError(
                f"build_replica_mirrors: parameter '{path}' (replica "
                f"{replica}) has requires_grad={mparam.requires_grad} but "
                f"the canonical has requires_grad={cparam.requires_grad}"
            )
        if mparam.device != device:
            raise ReplicaPlanError(
                f"build_replica_mirrors: parameter '{path}' (replica "
                f"{replica}) is on {mparam.device} but the replica target "
                f"device is {device}"
            )
        cstor = cparam.untyped_storage()
        mstor = mparam.untyped_storage()
        if cstor.size() and mstor.size():
            if cstor.data_ptr() == mstor.data_ptr():
                _shared_storage_error(
                    path, replica,
                    f"canonical and mirror parameter share the same storage")
    # Structural module-tree validation: matching parameter paths alone
    # do NOT make the mirror structurally valid (a plain-Module rebuild
    # or a swapped/omitted/reordered submodule can still expose the
    # same parameter paths) — validate the complete module tree.
    _check_module_tree(canonical, mirror, replica)
    # ALIAS-AWARE buffer traversal: the default deduplicating
    # named_buffers() would silently hide the same buffer object
    # registered under several logical paths.
    mirror_buffer_pairs = list(
        mirror.named_buffers(recurse=True, remove_duplicate=False))
    canonical_buffer_paths = [name for name, _ in buffer_pairs]
    if [name for name, _ in mirror_buffer_pairs] != canonical_buffer_paths:
        raise ReplicaPlanError(
            f"build_replica_mirrors: replica {replica} mirror's registered "
            f"buffer paths are not identical (and ordered) to the "
            f"canonical component's buffer paths"
        )
    for (path, cbuffer), (_, mbuffer) in zip(buffer_pairs, mirror_buffer_pairs):
        if mbuffer is cbuffer:
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{path}' on replica "
                f"{replica} IS the canonical buffer object — a mirror "
                f"must be a DISTINCT object; construction is refused"
            )
        if tuple(mbuffer.shape) != tuple(cbuffer.shape):
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{path}' (replica {replica}) "
                f"has shape {tuple(mbuffer.shape)} but the canonical has "
                f"{tuple(cbuffer.shape)}"
            )
        if mbuffer.dtype != cbuffer.dtype:
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{path}' (replica {replica}) "
                f"has dtype {mbuffer.dtype} but the canonical has "
                f"{cbuffer.dtype}"
            )
        if mbuffer.device != device:
            raise ReplicaPlanError(
                f"build_replica_mirrors: buffer '{path}' (replica {replica}) "
                f"is on {mbuffer.device} but the replica target device is "
                f"{device}"
            )
        cstor = cbuffer.untyped_storage()
        mstor = mbuffer.untyped_storage()
        if cstor.size() and mstor.size():
            if cstor.data_ptr() == mstor.data_ptr():
                _shared_storage_error(
                    path, replica,
                    f"canonical and mirror buffer share the same storage")
    # ------------------------------------------------------------------
    # GLOBAL alias safety (any-to-any, base-storage aware): the per-path
    # checks above catch SAME-path aliasing; these checks catch every
    # other aliasing — a mirror tensor at path B aliasing canonical
    # storage/object at path A, a mirror tensor IS-ing any canonical
    # tensor object, and any storage sharing INSIDE the mirror tree
    # itself (parameter vs buffer, across logical paths). Base-storage
    # identity (see :func:`_base_storage_ptr`) is what makes view/offset
    # aliases of one shared storage block detectable.
    # ------------------------------------------------------------------
    mirror_tensors = ([("parameter '" + n + "'", t)
                       for n, t in mirror_param_pairs]
                      + [("buffer '" + n + "'", t)
                         for n, t in mirror_buffer_pairs])
    for what, t in mirror_tensors:
        if id(t) in canon_obj_ids:
            raise ReplicaPlanError(
                f"build_replica_mirrors: mirror {what} on replica "
                f"{replica} IS a canonical tensor object — a mirror must "
                f"own DISTINCT objects; cross-path object aliasing "
                f"between the mirror and canonical trees is refused"
            )
    mirror_owners = {}
    for what, t in mirror_tensors:
        ptr = _base_storage_ptr(t)
        if ptr is None:
            continue  # empty storage: no base block to alias
        if ptr in storage_owners:
            raise ReplicaPlanError(
                f"build_replica_mirrors: mirror {what} (replica "
                f"{replica}) shares the base storage of "
                f"{storage_owners[ptr]} — a mirror tensor may never "
                f"alias canonical (checkpoint-owned) storage, at ANY "
                f"path (base-storage identity is compared, so "
                f"view/offset aliases are caught as well)"
            )
        if ptr in mirror_owners:
            raise ReplicaPlanError(
                f"build_replica_mirrors: mirror {what} (replica "
                f"{replica}) shares the base storage of another mirror "
                f"tensor {mirror_owners[ptr]} inside the same mirror "
                f"tree — two distinct registered tensors must never "
                f"use one storage"
            )
        mirror_owners[ptr] = f"mirror {what} (replica {replica})"
    return mirror, mirror_param_pairs, mirror_buffer_pairs


def build_replica_mirrors(canonical_module, plan, buffer_classes=None):
    """Construct the per-replica mirrors of ``canonical_module`` under
    ``plan`` and return the :class:`ComponentMirrorSet` (the §10.1.1
    factory + §10.1.2 mapping + §10.1.4 buffer policy, all fail-fast).

    Args:
        canonical_module: the trainable component (a torch module;
            e.g. an archi sub-model). Its parameters/buffers must
            already be fully initialized (checkpoint loaded,
            ``init_weights`` done) and co-located on ``plan.
            primary_device``.
        plan: the :class:`ReplicaPlan` this component joins.
        buffer_classes: optional explicit ``{buffer path: 'A'|'B'}``
            registry (module-local paths relative to the component
            root); overrides the owning module class's
            ``_dfl_buffer_classes`` declaration. Every registered
            buffer must be classified; unclassified buffers are class
            C and refuse construction.

    Note:
        Construction does NOT copy values (that is the separate
        ``initial_sync()`` step, §10.1.1) and reads/writes no global
        state (``nn.device`` is never touched).
    """
    if not isinstance(plan, ReplicaPlan):
        raise TypeError(
            "build_replica_mirrors: plan must be a ReplicaPlan (got "
            f"{type(plan).__name__})"
        )
    if plan.is_disposed:
        raise ReplicaPlanError(
            "build_replica_mirrors: the replica plan is disposed — "
            "components can no longer be added"
        )
    (param_pairs, buffer_pairs, storage_owners,
     canon_obj_ids) = _check_canonical_tree(plan, canonical_module)
    resolved = _resolve_buffer_classes(canonical_module, buffer_pairs,
                                       buffer_classes)
    scope = getattr(canonical_module, "name", None)
    for path, p in param_pairs:
        _check_dfl_name_consistency(p, path, scope, "canonical parameter")
    for path, b in buffer_pairs:
        _check_dfl_name_consistency(b, path, scope, "canonical buffer")

    if not plan.is_multi:
        # N == 1: NO mirrors at all (canonical == the only replica).
        return ComponentMirrorSet(
            plan=plan,
            canonical=canonical_module,
            mirrors=[],
            paths=[name for name, _ in param_pairs],
            canonical_params=[p for _, p in param_pairs],
            mirror_params={},
            canonical_buffers=[b for _, b in buffer_pairs],
            mirror_buffers={},
            buffer_classes=resolved,
            object_ids=canon_obj_ids,
            storage_ptrs=set(storage_owners),
        )

    mirrors = []
    mirror_param_lists = []
    mirror_buffer_lists = []
    seen_ptrs = set()
    for replica in range(1, plan.num_replicas):
        mirror, m_pairs, m_bufs = _build_one_mirror(
            plan, canonical_module, replica, param_pairs, buffer_pairs,
            storage_owners, canon_obj_ids)
        for _, mp in m_pairs:
            stor = mp.untyped_storage()
            if stor.size():
                ptr = stor.data_ptr()
                if ptr in seen_ptrs:
                    raise ReplicaPlanError(
                        f"build_replica_mirrors: mirror parameter storage "
                        f"(replica {replica}) is shared with another "
                        f"mirror replica — every mirror must own DISTINCT "
                        f"storage"
                    )
                seen_ptrs.add(ptr)
        for _, mb in m_bufs:
            stor = mb.untyped_storage()
            if stor.size():
                ptr = stor.data_ptr()
                if ptr in seen_ptrs:
                    raise ReplicaPlanError(
                        f"build_replica_mirrors: mirror buffer storage "
                        f"(replica {replica}) is shared with another "
                        f"mirror replica — every mirror must own DISTINCT "
                        f"storage"
                    )
                seen_ptrs.add(ptr)
        mirrors.append(mirror)
        mirror_param_lists.append([p for _, p in m_pairs])
        mirror_buffer_lists.append([b for _, b in m_bufs])

    # Global alias registries (canonical + every mirror, parameters AND
    # buffers): the plan uses them to refuse overlapping storage /
    # shared object identity ACROSS registered components.
    object_ids = set(canon_obj_ids)
    storage_ptrs = set(storage_owners)
    for r, plist in enumerate(mirror_param_lists, start=1):
        for p in plist:
            object_ids.add(id(p))
            ptr = _base_storage_ptr(p)
            if ptr is not None:
                storage_ptrs.add(ptr)
    for r, blist in enumerate(mirror_buffer_lists, start=1):
        for b in blist:
            object_ids.add(id(b))
            ptr = _base_storage_ptr(b)
            if ptr is not None:
                storage_ptrs.add(ptr)

    return ComponentMirrorSet(
        plan=plan,
        canonical=canonical_module,
        mirrors=mirrors,
        paths=[name for name, _ in param_pairs],
        canonical_params=[p for _, p in param_pairs],
        mirror_params={r: lst
                       for r, lst in enumerate(mirror_param_lists, start=1)},
        canonical_buffers=[b for _, b in buffer_pairs],
        mirror_buffers={r: lst
                        for r, lst in enumerate(mirror_buffer_lists, start=1)},
        buffer_classes=resolved,
        object_ids=object_ids,
        storage_ptrs=storage_ptrs,
    )


class ReplicaPlan:
    """The ordered replica device plan plus the per-component mirror
    registry (Commit 2 entry point; exposed as ``nn.ReplicaPlan``).

    Replica 0 is the canonical/primary replica (the Phase 2
    ``nn.device`` primary-device contract); replicas r > 0 are the
    secondary replicas that own the disposable mirrors. Construction
    never reads or writes the global ``nn.device`` and never calls
    ``torch.cuda.*`` directly (device resolution goes through the
    Phase 2 backend-neutral abstraction).
    """

    def __init__(self, primary_device, replica_devices=None):
        if not isinstance(primary_device, torch.device):
            raise TypeError(
                f"ReplicaPlan: primary_device must be a torch.device (got "
                f"{type(primary_device).__name__})"
            )
        if replica_devices is None:
            replica_devices = [primary_device]
        else:
            replica_devices = list(replica_devices)
            for d in replica_devices:
                if not isinstance(d, torch.device):
                    raise TypeError(
                        f"ReplicaPlan: replica_devices must be torch.device "
                        f"objects (got {type(d).__name__})"
                    )
        if not replica_devices:
            raise ReplicaPlanError(
                "ReplicaPlan: at least one replica device is required "
                "(replica 0 = the canonical/primary replica)"
            )
        if replica_devices[0] != primary_device:
            raise ReplicaPlanError(
                f"ReplicaPlan: replica 0 (the canonical/primary replica) "
                f"must be on the primary device {primary_device}, got "
                f"{replica_devices[0]}"
            )
        self.primary_device = primary_device
        self.replica_devices = tuple(replica_devices)
        self._components = []
        self._component_ids = set()
        self._canon_id = {}    # id(canonical param) -> canonical param
        self._mirror_canon = {}  # replica r -> {id(mirror param) -> canonical param}
        # Plan-level GLOBAL alias registries: every tensor object
        # identity / base-storage identity registered by any previous
        # component (canonical + all mirrors, parameters + buffers).
        # ``add_component`` refuses a component whose tensors overlap
        # these sets (no cross-component storage sharing or object
        # identity is supported in Phase 12).
        self._seen_obj_ids = set()
        self._seen_storage_ptrs = set()
        self._disposed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_device_config(cls, device_config):
        """Production path: the ordered replica device list of a Phase 2
        ``DeviceConfig`` (``devices[0]`` = primary = ``nn.device``; an
        empty device list is the CPU-only single-replica case). Device
        resolution uses the backend-neutral abstraction only."""
        from .device import get_torch_device
        devices = list(device_config.devices)
        if not devices:
            return cls(torch.device("cpu"))
        primary = get_torch_device(devices[0])
        return cls(primary, [get_torch_device(d) for d in devices])

    @classmethod
    def from_torch_devices(cls, primary_device, replica_devices=None):
        """Explicit torch-device plan. Production derives plans from a
        ``DeviceConfig`` (``from_device_config``); this constructor is
        the narrow tested abstraction for Level-B simulations that map
        several LOGICAL replicas onto one PHYSICAL device
        (labeled ``SIMULATED_MULTI_REPLICA``)."""
        return cls(primary_device, replica_devices)

    # -- introspection -----------------------------------------------------

    @property
    def num_replicas(self):
        return len(self.replica_devices)

    @property
    def is_multi(self):
        return self.num_replicas > 1

    @property
    def is_disposed(self):
        return self._disposed

    @property
    def components(self):
        return list(self._components)

    def _check_alive(self, operation):
        if self._disposed:
            raise ReplicaPlanError(
                f"ReplicaPlan.{operation}: the plan is disposed — mirrors "
                f"are disposable runtime state and were released; "
                f"rebuild a fresh plan (mirrors are reconstructed from "
                f"the canonical modules)"
            )

    # -- component registry --------------------------------------------------

    def add_component(self, canonical_module, buffer_classes=None):
        """Build the mirrors of one trainable component (and register
        its mapping in this plan). Returns the
        :class:`ComponentMirrorSet`."""
        self._check_alive("add_component")
        cid = id(canonical_module)
        if cid in self._component_ids:
            raise ReplicaPlanError(
                "ReplicaPlan.add_component: this component is already "
                "registered in the plan — one mirror set per component"
            )
        component = build_replica_mirrors(canonical_module, self,
                                          buffer_classes)
        # CROSS-component global alias safety: every tensor object
        # identity / base-storage identity of this component (canonical
        # + all mirrors, parameters + buffers) must be disjoint from
        # the plan-level registries of every previously registered
        # component — Phase 12 supports no storage sharing or shared
        # tensor objects ACROSS components. Checked before ANY plan
        # state is committed, so a refusal leaves the plan untouched.
        for oid in component.object_ids:
            if oid in self._seen_obj_ids:
                raise ReplicaPlanError(
                    "ReplicaPlan.add_component: this component shares a "
                    "tensor OBJECT identity with another registered "
                    "component (two components reference the same "
                    "parameter/buffer object) — Phase 12 supports no "
                    "cross-component aliasing"
                )
        for ptr in component.storage_ptrs:
            if ptr in self._seen_storage_ptrs:
                raise ReplicaPlanError(
                    "ReplicaPlan.add_component: this component's base "
                    "storage overlaps with another registered component "
                    "(two components use the same underlying storage, "
                    "possibly at different view offsets) — Phase 12 "
                    "supports no cross-component storage sharing"
                )
        self._seen_obj_ids |= component.object_ids
        self._seen_storage_ptrs |= component.storage_ptrs
        self._component_ids.add(cid)
        self._components.append(component)
        for p in component.canonical_params:
            self._canon_id[id(p)] = p
        for replica, mlist in component.mirror_params.items():
            targets = self._mirror_canon.setdefault(replica, {})
            for m, c in zip(mlist, component.canonical_params):
                targets[id(m)] = c
        return component

    # -- synchronization (§10.1.5) -------------------------------------------

    def _sync_all(self):
        if not self.is_multi:
            return  # N == 1: no mirrors exist, nothing to sync
        with torch.no_grad():
            for component in self._components:
                for replica in range(1, self.num_replicas):
                    target = self.replica_devices[replica]
                    for cparam, mparam in zip(component.canonical_params,
                                              component.mirror_params[replica]):
                        if cparam.dtype != mparam.dtype:
                            raise ReplicaPlanError(
                                "ReplicaPlan sync: parameter dtype changed "
                                f"after mirror construction (canonical "
                                f"{cparam.dtype} vs mirror {mparam.dtype})"
                            )
                        mparam.copy_(cparam.to(target))
                    for cbuffer, mbuffer in zip(component.canonical_buffers,
                                                component.mirror_buffers[replica]):
                        if cbuffer.dtype != mbuffer.dtype:
                            raise ReplicaPlanError(
                                "ReplicaPlan sync: buffer dtype changed "
                                f"after mirror construction (canonical "
                                f"{cbuffer.dtype} vs mirror {mbuffer.dtype})"
                            )
                        mbuffer.copy_(cbuffer.to(target))

    def initial_sync(self):
        """The INITIAL canonical -> mirror synchronization (§10.1.5):
        run after construction + checkpoint load + all canonical
        initialization and before the first replica forward. Copies
        parameter data and class-A buffers into every mirror on every
        non-primary replica device (no_grad; never touches optimizer
        state, Saveable/checkpoint state, or ``nn.device``)."""
        self._check_alive("initial_sync")
        self._sync_all()
        for component in self._components:
            component.is_synced = True

    def sync_from_canonical(self):
        """The POST-successful-step canonical -> mirror synchronization
        (the same deterministic mechanism as ``initial_sync``; the
        approved plan calls it after each successful canonical
        optimizer step — a SKIPPED step must not call it)."""
        self._check_alive("sync_from_canonical")
        for component in self._components:
            if not component.is_synced:
                raise ReplicaPlanError(
                    "ReplicaPlan.sync_from_canonical: component "
                    f"'{getattr(component.canonical, 'name', None)}' has not "
                    "been initially synced — call initial_sync() before "
                    "any post-step sync (§10.1.1/§10.1.5)"
                )
        self._sync_all()

    # -- gradient installation foundation (§10.1.3) ---------------------------

    def canonicalize_grads(self, replica_grad_lists):
        """Normalize per-replica gradient lists for ``average_gv_list``
        (the §10.1.3 transfer foundation): validates that replica 0's
        entries reference the CANONICAL parameters of the plan's
        mirrored components and that each secondary replica's entries
        reference the MIRROR counterpart of the same canonical
        parameter, transfers the secondary gradients to the canonical
        device, and returns per-replica lists rebinding every gradient
        to the CANONICAL parameter object.

        Performs NO averaging, NO optimizer work, NO GradScaler logic,
        and no gradient-structure validation (that remains
        ``average_gv_list``'s job).
        """
        self._check_alive("canonicalize_grads")
        if len(replica_grad_lists) != self.num_replicas:
            raise ReplicaPlanError(
                f"ReplicaPlan.canonicalize_grads: expected exactly "
                f"{self.num_replicas} per-replica gradient list(s) (one per "
                f"replica), got {len(replica_grad_lists)}"
            )
        if self.num_replicas == 1:
            # official N == 1 identity behavior
            return replica_grad_lists[0]
        n = len(replica_grad_lists[0])
        for replica in range(1, self.num_replicas):
            if len(replica_grad_lists[replica]) != n:
                raise ReplicaPlanError(
                    f"ReplicaPlan.canonicalize_grads: replica {replica} has "
                    f"{len(replica_grad_lists[replica])} entries but replica "
                    f"0 has {n} — every replica must list the same variables"
                )
        result = [[] for _ in range(self.num_replicas)]
        for i in range(n):
            entry0 = replica_grad_lists[0][i]
            try:
                p0 = entry0[1]
            except (IndexError, TypeError, KeyError):
                raise ReplicaPlanError(
                    f"ReplicaPlan.canonicalize_grads: replica 0 variable "
                    f"{i} entry {entry0!r} is not a (gradient, parameter) "
                    f"pair"
                ) from None
            if self._canon_id.get(id(p0)) is not p0:
                raise ReplicaPlanError(
                    f"ReplicaPlan.canonicalize_grads: replica 0 variable "
                    f"{i} references a parameter that is not a CANONICAL "
                    f"parameter of this plan's mirrored components — "
                    f"replica 0 computes on the canonical parameters"
                )
            # replica 0's gradient is already canonical (no transfer);
            # structure validation (None / non-tensor / layout) remains
            # average_gv_list's job
            result[0].append((entry0[0], p0))
            for replica in range(1, self.num_replicas):
                entry_r = replica_grad_lists[replica][i]
                try:
                    grad_r, p_r = entry_r[0], entry_r[1]
                except (IndexError, TypeError, KeyError):
                    raise ReplicaPlanError(
                        f"ReplicaPlan.canonicalize_grads: replica {replica} "
                        f"variable {i} entry {entry_r!r} is not a "
                        f"(gradient, parameter) pair"
                    ) from None
                if self._mirror_canon.get(replica, {}).get(id(p_r)) is not p0:
                    raise ReplicaPlanError(
                        f"ReplicaPlan.canonicalize_grads: replica {replica} "
                        f"variable {i} references a parameter that is not "
                        f"the mirror counterpart of replica 0's canonical "
                        f"parameter — secondary replicas compute on their "
                        f"mirrors, whose gradients are transferred to the "
                        f"canonical device and rebound to the canonical "
                        f"parameter"
                    )
                if grad_r is None:
                    # pass through; average_gv_list reports the None grad
                    result[replica].append((None, p0))
                elif torch.is_tensor(grad_r):
                    # transfer the secondary replica's gradient to the
                    # canonical device (replica 0's grads never transfer);
                    # structure/layout validation stays with average_gv_list
                    if grad_r.device != self.primary_device:
                        grad_r = grad_r.detach().to(self.primary_device)
                    else:
                        grad_r = grad_r.detach()
                    result[replica].append((grad_r, p0))
                else:
                    # non-tensor gradient passes through unchanged;
                    # average_gv_list reports it
                    result[replica].append((grad_r, p0))
        return result

    # -- lifecycle -------------------------------------------------------------

    def dispose(self):
        """Dispose all mirrors (end of the training run). Releases the
        mirror module references (torch's allocator reclaims/reuses the
        blocks; this layer makes no backend cache-reset call) and the
        plan refuses all further use. The canonical modules/
        parameters/buffers and all checkpoint-owned state are
        untouched."""
        if self._disposed:
            return
        for component in self._components:
            component.mirrors = []
            component.mirror_params = {}
            component.mirror_buffers = {}
            component.is_synced = False
        self._components = []
        self._component_ids = set()
        self._canon_id = {}
        self._mirror_canon = {}
        self._seen_obj_ids = set()
        self._seen_storage_ptrs = set()
        self._disposed = True


# ---------------------------------------------------------------------------
# Commit 3: mixed precision across replica devices (Phase 12 §6/§6.1)
# ---------------------------------------------------------------------------

class ReplicaPrecisionError(ValueError):
    """A canonical-side precision-lifecycle contract violation
    (Phase 12 Commit 3): a plan/precision-plan/scaler/optimizer
    combination that violates the ONE-global-scaler rule, a
    replica-count mismatch, or a nonfinite replica gradient under
    an off/bf16 plan (no GradScaler exists there to own overflow
    detection — hard error, matching the single-device behavior).

    Distinct from ``ReplicaPlanError`` (construction/registry
    state) and ``ReplicaGradientError`` (Commit-1 aggregation
    structure): this is the precision step's own contract layer.
    """


def clear_replica_grads(plan):
    """Clear the accumulated gradients of EVERY trainable tensor the
    plan's components own: the CANONICAL (primary, replica 0)
    parameters AND all secondary MIRROR parameters (the §6.1
    attempt-start, Class A and Class B cleanup).

    Gradients accumulated by an aborted or skipped attempt must
    NEVER survive into a later attempt or step (a later replica's
    overflow discards every earlier replica's grads; a Class A
    abort clears the partial-attempt state before the hard
    ``FloatingPointError`` propagates). Only ``.grad`` references
    drop — parameter/buffer DATA, optimizer state and all
    checkpoint-owned state are untouched, and the mirrors stay in
    place (only their accumulated gradients clear)."""
    if not isinstance(plan, ReplicaPlan):
        raise TypeError(
            f"clear_replica_grads: plan must be a ReplicaPlan (got "
            f"{type(plan).__name__})"
        )
    plan._check_alive("clear_replica_grads")
    for component in plan.components:
        for p in component.canonical_params:
            p.grad = None
        for replica in range(1, plan.num_replicas):
            for p in component.mirror_params[replica]:
                p.grad = None


def replica_param_lists(plan, canonical_params):
    """The per-replica parameter lists that OWN each replica's
    gradients (the §6.1 ownership rule, exposed for the attempt
    closures): replica 0 (primary) computes on the CANONICAL
    parameters — the returned replica-0 list is ``canonical_params``
    itself (same objects, same order); replica r > 0 computes on its
    MIRROR counterpart of each canonical parameter (path-aligned
    through the Commit-2 §10.1.2 mapping).

    ``canonical_params`` must be canonical parameters of the
    plan's registered components — a subset (one optimizer's
    variable group) is fine; the production G/D closures pass each
    optimizer's weight list. Returns a list of ``num_replicas``
    per-replica parameter lists.

    Raises:
        TypeError: ``plan`` is not a :class:`ReplicaPlan`.
        ReplicaPrecisionError: a listed parameter is not a
            canonical parameter of a registered component, or the
            plan is disposed.
    """
    if not isinstance(plan, ReplicaPlan):
        raise TypeError(
            f"replica_param_lists: plan must be a ReplicaPlan (got "
            f"{type(plan).__name__})"
        )
    plan._check_alive("replica_param_lists")
    index_by_id = {}
    for component in plan.components:
        for i, p in enumerate(component.canonical_params):
            index_by_id[id(p)] = (component, i)
    n = plan.num_replicas
    result = [[] for _ in range(n)]
    for p in canonical_params:
        hit = index_by_id.get(id(p))
        if hit is None:
            raise ReplicaPrecisionError(
                "replica_param_lists: a listed parameter is not a "
                "canonical parameter of this plan's registered "
                "components — replica 0 computes on the canonical "
                "parameters; register the owning component first"
            )
        component, i = hit
        result[0].append(p)
        for replica in range(1, n):
            result[replica].append(component.mirror_params[replica][i])
    return result


def run_replica_precision_step(plan, precision_plan, scaler, optimizer,
                               replica_grad_lists, active_weights=None,
                               unscale_hook=None, step_hook=None,
                               update_hook=None):
    """The canonical-side §6.1 precision step of ONE attempt (Phase 12
    Commit 3 foundation; the per-replica compute — per-replica
    forward under the run-wide plan's per-replica autocast, the Class
    A nonfinite forward/loss check, and the per-replica SCALED
    backward through the model layer's ``ModelBase._mp_backward`` —
    is owned by the caller's attempt closure).

    Pipeline (the ordering is the §6.1 contract):

    1. Commit-2 ``canonicalize_grads``: validate replica 0 against
       the CANONICAL parameters and each secondary replica against
       the MIRROR counterpart of the same canonical parameter
       (association), and transfer the secondary MIRROR grads to the
       canonical device (replica 0's grads are already canonical);
    2. STRUCTURAL validation before any value-level policy: N > 1
       through Commit-1 ``average_gv_list`` (pair structure, DENSE
       STRIDED layout — sparse/compressed-sparse rejected, shape,
       dtype, None rejection); N == 1 entry-by-entry (pair
       structure, tensor, strided layout, grad/param shape) — a
       malformed or sparse gradient ALWAYS fails through the
       deliberate Commit-1 ``ReplicaGradientError`` contract, never
       through a backend-specific value error;
    3. mode-aware PRE-unscale nonfinite policy over the
       STRUCTURALLY VALID per-replica gradients: fp16 — nonfinite
       SCALED grad values are NOT rejected (a numeric overflow is
       not a layout / structure defect: they deliberately pass
       through canonical ``.grad`` installation so the GradScaler
       records ``found_inf`` and owns overflow detection; §10.1.3
       PRE/POST split); off/bf16 — a nonfinite dense replica
       gradient is a hard :class:`ReplicaPrecisionError` (no scaler
       exists to classify it; parity with the single-device
       behavior);
    4. Commit-1 ``average_gv_list``: per-variable MEAN over the
       replicas (identity at N==1) — under fp16 this is the MEAN OF
       THE SCALED GRADIENTS;
    5. the averaged grads are installed on the canonical
       ``parameter.grad`` (the canonical parameters are the
       optimizer's variables — mirror params never enter optimizer
       state);
    6. exactly ONE canonical precision sequence per attempt,
       executed through the injected MODEL-LAYER hooks:
       ``unscale_hook(optimizer, active_weights)`` → the existing
       ``ModelBase._mp_unscale_opt`` (exposes the true/unscaled
       grads before the update op; no-op for off/bf16),
       ``step_hook(optimizer, grads_vars) -> bool`` → the existing
       ``ModelBase._mp_opt_step`` (the native GradScaler SKIPS the
       update op when its overflow check fired; off/bf16 run the
       direct ``get_update_op``), ``update_hook()`` → the existing
       ``ModelBase._mp_scaler_update`` (overflow backoff / growth
       bookkeeping; no-op for off/bf16) — exactly ONE call to each
       per attempt (never once per replica).

    The hooks are the bound ``ModelBase`` mixed-precision methods,
    supplied by the model (production) or the test harness; this
    module NEVER calls native scaler methods itself and NEVER
    imports ``models/`` (no dependency, no duplicated ModelBase hook
    logic — the same hooks the single-device production path
    uses), and the ``_mp_run_generator`` retry authority stays in
    ``ModelBase``. ``scaler`` is the one global ``GradScaler`` of
    the run (``None`` for off/bf16); it is inspected here ONLY for
    the one-global-scaler contract check (an fp16 plan without its
    scaler, or an off/bf16 plan carrying one, is refused) — all
    scaler OPERATION happens inside the injected hooks.

    Returns:
        bool: ``True`` — the canonical optimizer stepped (a
        successful canonical step; the caller then syncs the
        mirrors, §10.1.5). ``False`` — the canonical step was
        SKIPPED by the GradScaler overflow check (Class B): ALL
        canonical + secondary mirror grads were already cleared
        here, no mirror sync may follow (no update to sync), no
        optimizer-iteration increment occurred; the caller (the
        model-layer attempt closure) raises ``SkippedGeneratorStep``
        so the EXISTING ``_mp_run_generator`` retry policy applies
        (16 bounded attempts under fp16, 1 under off/bf16).

    Raises:
        ReplicaPrecisionError: the plan is disposed; a
            plan/scaler/precision-plan combination violates the
            ONE-global-scaler rule (an fp16 plan with no scaler, or
            an off/bf16 plan with a scaler); a replica-count
            mismatch; a hook is missing or not callable; a
            nonfinite dense replica gradient under an off/bf16 plan.
        ReplicaPlanError: the Commit-2 ``canonicalize_grads``
            ownership/transfer contract is violated.
        ReplicaGradientError: the Commit-1 structural contract
            (pair structure, dense strided layout, shape, dtype,
            None) is violated.
    """
    if not isinstance(plan, ReplicaPlan):
        raise TypeError(
            f"run_replica_precision_step: plan must be a ReplicaPlan (got "
            f"{type(plan).__name__})"
        )
    plan._check_alive("run_replica_precision_step")
    if not isinstance(precision_plan, _PrecisionPlan):
        raise TypeError(
            "run_replica_precision_step: precision_plan must be a "
            "mixed_precision.PrecisionPlan (the one global plan of the "
            f"run, resolved across ALL selected devices; got "
            f"{type(precision_plan).__name__})"
        )
    # The ONE-global-scaler rule (§6) — CONTRACT CHECK ONLY: exactly
    # one model-level GradScaler per run, never one per replica /
    # mirror / device. fp16 plans REQUIRE it; off/bf16 plans must NOT
    # carry it. The scaler itself is operated only by the injected
    # model-layer hooks (below).
    if precision_plan.scaler_required and scaler is None:
        raise ReplicaPrecisionError(
            "run_replica_precision_step: an fp16 plan REQUIRES the one "
            "global GradScaler (exactly one model-level scaler per run; "
            "never one scaler per replica or per device)"
        )
    if not precision_plan.scaler_required and scaler is not None:
        raise ReplicaPrecisionError(
            "run_replica_precision_step: off/bf16 plans carry NO "
            "GradScaler (bf16 forbids a scaler; passing one makes the "
            "step depend on FP16 machinery that does not apply)"
        )
    if len(replica_grad_lists) != plan.num_replicas:
        raise ReplicaPrecisionError(
            f"run_replica_precision_step: expected exactly "
            f"{plan.num_replicas} per-replica gradient list(s) (one per "
            f"replica), got {len(replica_grad_lists)}"
        )
    # The injected MODEL-LAYER hooks (the dependency-injected
    # interface): the canonical unscale/step/update sequence is owned
    # by the EXISTING ModelBase mixed-precision hooks (the bound
    # methods the model supplies — ModelBase._mp_unscale_opt /
    # ModelBase._mp_opt_step / ModelBase._mp_scaler_update). This
    # module never calls native scaler methods itself and never
    # duplicates ModelBase hook logic.
    for name, hook in (("unscale_hook", unscale_hook),
                       ("step_hook", step_hook),
                       ("update_hook", update_hook)):
        if not callable(hook):
            raise ReplicaPrecisionError(
                f"run_replica_precision_step: {name} must be a callable "
                "model-layer hook — the canonical precision sequence is "
                "owned by the existing ModelBase mixed-precision hooks "
                "(ModelBase._mp_unscale_opt / _mp_opt_step / "
                f"_mp_scaler_update, bound and supplied by the model); "
                f"got {type(hook).__name__}"
            )

    # 1: Commit-2 association + transfer (secondary MIRROR grads to
    # the canonical device, rebound to the CANONICAL parameters);
    # N == 1 is the official identity.
    canonical_lists = plan.canonicalize_grads(replica_grad_lists)
    # 2: STRUCTURAL validation (+ N > 1 aggregation) BEFORE any
    # value-level policy, so a malformed / sparse / compressed-
    # sparse gradient fails through the deliberate Commit-1
    # contract.
    if plan.num_replicas == 1:
        for i, entry in enumerate(canonical_lists):
            g, p = _validate_entry(0, i, entry)
            if g.shape != p.shape:
                raise ReplicaGradientError(
                    f"run_replica_precision_step: variable {i} gradient "
                    f"shape {tuple(g.shape)} does not match the "
                    f"parameter shape {tuple(p.shape)}"
                )
        averaged = canonical_lists
    else:
        averaged = average_gv_list(canonical_lists)
    # 3: mode-aware PRE-unscale nonfinite policy over the
    # structurally valid per-replica gradients (the §10.1.3
    # PRE/POST split): off/bf16 reject nonfinite grads immediately
    # (hard error — no scaler exists to detect them); fp16
    # deliberately lets nonfinite SCALED values through so the
    # GradScaler owns overflow detection (Class B) — a numeric
    # overflow, never a layout defect.
    if not precision_plan.scaler_required:
        lists = (canonical_lists if plan.num_replicas > 1
                 else [canonical_lists])
        for replica in range(plan.num_replicas):
            for i, (g, _p) in enumerate(lists[replica]):
                if not torch.isfinite(g).all():
                    raise ReplicaPrecisionError(
                        f"run_replica_precision_step: replica {replica} "
                        f"variable {i} has a NONFINITE gradient under "
                        f"precision '{precision_plan.mode}' — no "
                        f"GradScaler exists in off/bf16 to classify "
                        f"overflow, so nonfinite optimizer input is a "
                        f"hard error (the single-device behavior)"
                    )
    # 5: install on the canonical parameters (the optimizer's
    # variables).
    for g, p in averaged:
        p.grad = g
    # 6: exactly ONE canonical unscale / step / update per attempt,
    # through the injected model-layer hooks (the one global scaler
    # is operated exclusively by those hooks; the off/bf16 hooks are
    # ModelBase's no-op / direct-get_update_op semantics).
    unscale_hook(optimizer, active_weights)
    stepped = step_hook(optimizer, averaged)
    update_hook()
    if not stepped:
        # Class B: the GradScaler rejected the attempt (found_inf
        # — a secondary replica's overflow reached the canonical
        # grads THROUGH the averaging step). Complete cleanup:
        # ALL canonical + secondary mirror grads (a later
        # replica's overflow discards every earlier replica's
        # grads; none may survive into the retry). No mirror sync
        # (no update to sync); the caller raises
        # SkippedGeneratorStep → the existing bounded retry.
        clear_replica_grads(plan)
    return stepped
