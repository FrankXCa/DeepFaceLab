"""Multi-device (Phase 12) foundation: replica gradient aggregation.

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
"""

import torch

__all__ = ["average_gv_list", "ReplicaGradientError"]


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
