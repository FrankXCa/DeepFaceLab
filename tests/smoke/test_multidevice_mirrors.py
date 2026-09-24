"""Phase 12 Commit 2 acceptance: device-independent replica mirror
management (core.leras.multidevice, exposed as nn.ReplicaPlan /
nn.ReplicaPlanError / nn.build_replica_mirrors).

Covers the approved Phase 12 plan's normative replica contracts
(docs/PHASE12_STATE.md §10.1, §16 commit boundary 2):

- replica plan representation: ordered replica device list, replica 0 ==
  canonical/primary replica (Phase 2 nn.device primary-device
  contract), production derivation from DeviceConfig through the
  backend-neutral abstraction (CPU-only = the single-replica case),
  the narrow from_torch_devices abstraction for Level-B simulation
- mirror construction: the PERMITTED module-level factory
  copy.deepcopy(canonical_module).to(target_device) on REAL leras
  components (a small DeepFakeArchi Encoder through the production
  factory + a real inference-only BatchNorm2D), distinct mirror
  modules, distinct Parameters, distinct storage (canonical vs mirror
  AND between mirror replicas), preserved module structure/types,
  preserved training/eval mode
- module-tree validation (real-tree positive path + synthetic
  factory-defect negative paths): after the factory, the mirror's
  COMPLETE module tree is compared with the canonical's
  (named_modules, remove_duplicate=False) — identical ordered module
  paths (missing/reordered submodule paths rejected), identical
  concrete module type at every path (a plain-Module rebuild with
  matching parameter paths is refused), identical per-submodule
  training/eval flags (a single flipped submodule is refused; the
  top-level flag alone is not sufficient)
- deterministic canonical<->mirror parameter mapping: identical
  ordered module-local logical paths (named_parameters,
  remove_duplicate=False), positional parameter/buffer alignment;
  REORDERED and MISSING path lists are rejected
- GLOBAL alias safety (fail-fast at construction; any-to-any,
  base-storage aware, not same-path pairs): two distinct canonical
  Parameters sharing one storage; view/OFFSET aliases of the same
  base storage (the check reasons on untyped_storage identity — two
  tensors at different view offsets of one storage are aliasing); a
  mirror tensor at path B aliasing canonical storage at path A; a
  mirror tensor IS-ing a canonical tensor object at any path; shared
  storage inside a mirror tree (parameter vs buffer, any paths);
  shared storage between mirror replicas; overlapping storage or
  shared tensor object identity ACROSS registered components
  (plan-level registries); duplicate buffer object identity across
  logical paths (the same buffer under two paths — surfaced only by
  the alias-aware named_buffers(remove_duplicate=False) traversal);
  distinct buffers sharing one storage; a tensor registered as both
  a parameter and a buffer
- shape / dtype / requires_grad mismatch rejection
- _dfl_name: tolerated when the copied mirror attribute is LOST
  (module-level deepcopy does not reliably preserve custom Parameter
  attributes), consistent canonical _dfl_name verified as an
  ADDITIONAL invariant (never the mapping key); an inconsistent
  binding is rejected
- buffer policy (named_buffers): real BatchNorm2D running statistics
  are CLASS A via the component's own explicit declaration
  (_dfl_buffer_classes); the caller registry override works;
  UNCLASSIFIED buffers are CLASS C and refuse construction; a class B
  declaration fails loud (no reduction/synchronization rule is
  registered in Phase 12); class-A buffers are copied at the initial
  sync and re-synced by the post-step sync
- initial canonical -> mirror sync: bit-exact parameter AND class-A
  buffer copy into every mirror on every non-primary replica device
- post-modification (post-successful-step) sync: the same mechanism;
  a sync before the initial sync is refused
- a mirror write can never touch canonical (checkpoint-owned)
  storage: mirror parameter/buffer writes and full replica
  forward/backward leave canonical values and gradients untouched
- checkpoint/Saveable isolation: mirrors are absent from the
  canonical Saveable weight enumeration, from any simulated
  model_filename_list, own no optimizer state and create no save
  keys; a save_weights file carries exactly the canonical keys;
  disposable mirrors are reconstructed from the canonical after a
  simulated checkpoint resume (save -> dispose -> fresh module ->
  load_weights -> rebuild plan -> initial_sync). All checkpoint
  scratch in these tests uses pytest's per-test ``tmp_path`` — a
  unique temporary directory per test; no repo-private scratch path,
  no fixed private filename, no pre-existing file can be overwritten,
  and a public clean checkout works even if ``docs/`` is absent
- no optimizer ownership: a real AdaBelief optimizer initialized on
  the CANONICAL parameters tracks only canonical parameters (state
  buffers, weight list); the averaged-gradient one-step update
  changes canonical parameters only, per the official formula vs an
  independent NumPy reference, and increments iterations once;
  mirrors receive the update only through the explicit sync
- SIMULATED_MULTI_REPLICA integration: two LOGICAL replicas through
  the production framework path (real nn.initialize, real DeepFakeArchi
  factory, real BatchNorm2D, real AdaBelief optimizer, real
  average_gv_list — no test-only fakes in the production flow;
  synthetic modules are used for failure paths only): per-replica
  forward/backward, canonicalize_grads (parameter association +
  cross-device transfer + rebind to canonical parameters),
  average_gv_list, one canonical optimizer step, post-step sync,
  forward parity. The GPU variant maps both logical replicas onto ONE
  physical device (cuda:0) and is labeled SIMULATED_MULTI_REPLICA;
  physical 2-GPU acceptance stays PENDING_ENVIRONMENTALLY /
  NOT_VERIFIED (single-GPU machine). A mixed-device plan
  (primary cuda:0 + CPU secondary replica) exercises the real
  cross-device gradient transfer.
- N == 1 single-replica bypass: the production CPU-only / 1-GPU plan
  creates NO mirrors, initial_sync is a no-op, canonicalize_grads is
  the official identity — the foundation of the unchanged single-GPU
  path
- disposal: disposing the plan releases all mirror references,
  refuses all further use, and leaves the canonical modules/
  parameters/buffers and their Saveable behavior untouched

Parity label: EXACT (CPU values are bit-exact; the GPU simulation
runs on one physical device so both logical replicas share the
backend and stay bit-exact; optimizer one-step values are checked
against an independent NumPy reference of the official formula).
Provenance: INDEPENDENT_REIMPLEMENTATION (official DFL multi-GPU
semantics — one canonical variable set, per-replica gradients,
cross-replica mean — are the semantic authority; no external code
was copied).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn as dfl_nn  # noqa: E402
from core.leras.layers.Saveable import Saveable  # noqa: E402
from core.leras.multidevice import (  # noqa: E402
    ReplicaPlan,
    ReplicaPlanError,
    average_gv_list,
    build_replica_mirrors,
)

CUDA_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Phase 12 GPU test: CUDA device required",
)

# the OFFICIAL optimizer denominator epsilon (finfo RESOLUTION,
# decimal): 1e-06 for f32.
OFFICIAL_EPS_F32 = 1e-6

CPU = torch.device("cpu")

_GPU_MAIN_ENV_DONE = False


def _ensure_gpu_main_env():
    """Publish the Phase 2 device main environment once per process
    (required by DeviceConfig.GPUIndexes); CPU-only tests never
    touch the device list."""
    global _GPU_MAIN_ENV_DONE
    if not _GPU_MAIN_ENV_DONE:
        dfl_nn.initialize_main_env()
        _GPU_MAIN_ENV_DONE = True


@pytest.fixture(autouse=True)
def _dfl_nn_cpu_nchw():
    """Every test runs with a CPU + NCHW foundation unless it
    initializes something else explicitly."""
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")
    yield
    dfl_nn.initialize(dfl_nn.DeviceConfig([]), "float32", "NCHW")


# ---------------------------------------------------------------------------
# component builders
# ---------------------------------------------------------------------------

def _make_encoder_component(scope="encoder", device=None):
    """A REAL DeepFakeArchi component through the production factory
    path: the archi factory class, the two-phase lifecycle
    (build_leaf_weights in __init__ + init_weights) and the
    production per-parameter binding pattern
    (models/Model_SAEHD/_bind_official_names: _dfl_name +
    _dfl_owner_layer)."""
    archi = dfl_nn.DeepFakeArchi(16, use_fp16=False)
    comp = archi.Encoder(3, 4, name=scope)
    comp.init_weights()
    if device is not None:
        comp.to(device)
    owners = {}
    for module in comp.modules():
        for p in module.parameters(recurse=False):
            owners[id(p)] = module
    for sub_name, obj in comp._iter_official_weights():
        obj._dfl_name = f"{scope}/{sub_name}"
        owner = owners.get(id(obj))
        if owner is not None:
            obj._dfl_owner_layer = owner
    return comp


def _make_bn_component(dim=4, name="bn", device=None):
    """A REAL inference-only BatchNorm2D layer as its own component
    (real registered buffers running_mean/running_var, CLASS A via
    the layer's own explicit _dfl_buffer_classes declaration) with
    the production binding pattern."""
    bn = dfl_nn.BatchNorm2D(dim, name=name)
    bn.build_weights()
    bn.init_weights()
    if device is not None:
        bn.to(device)
    owners = {}
    for module in bn.modules():
        for p in module.parameters(recurse=False):
            owners[id(p)] = module
    for sub_name, obj in bn._iter_official_weights():
        obj._dfl_name = f"{name}/{sub_name}"
        owner = owners.get(id(obj))
        if owner is not None:
            obj._dfl_owner_layer = owner
    return bn


def _plan_2cpu():
    """Two logical replicas on the one physical CPU device (the
    Level-B simulation shape, CPU flavor)."""
    return ReplicaPlan.from_torch_devices(CPU, [CPU, CPU])


# ---------------------------------------------------------------------------
# synthetic modules — failure paths ONLY (the production flow above
# uses exclusively real leras modules)
# ---------------------------------------------------------------------------

class _AliasedCanonical(torch.nn.Module):
    """Duplicate object identity INSIDE the canonical tree: two
    logical paths (a, b) resolve to the SAME Parameter object."""

    def __init__(self):
        super().__init__()
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.a = p
        self.b = p


class _DropParamCopy(torch.nn.Module):
    """Factory-level structural defect: the mirror tree is MISSING
    parameter b (named_parameters path list shorter than canonical)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    def __deepcopy__(self, memo):
        import copy as _copy
        m = torch.nn.Module()
        m.a = _copy.deepcopy(self.a, memo)
        # self.b intentionally dropped
        return m


class _ReorderParamCopy(torch.nn.Module):
    """Factory-level structural defect: the mirror tree lists its
    paths in a DIFFERENT order (b before a)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    def __deepcopy__(self, memo):
        import copy as _copy
        m = torch.nn.Module()
        m.b = _copy.deepcopy(self.b, memo)  # b registered first
        m.a = _copy.deepcopy(self.a, memo)
        return m


class _WrongShapeCopy(torch.nn.Module):
    """Mirror parameter with a DIFFERENT shape (no implicit reshape)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = torch.nn.Module()
        m.a = torch.nn.Parameter(torch.ones(3))
        return m


class _WrongDtypeCopy(torch.nn.Module):
    """Mirror parameter with a DIFFERENT dtype (no implicit cast)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = torch.nn.Module()
        m.a = torch.nn.Parameter(torch.ones(2).half())
        return m


class _WrongRequiresGradCopy(torch.nn.Module):
    """Mirror parameter with a DIFFERENT requires_grad flag."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = torch.nn.Module()
        m.a = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        return m


class _SameObjectCopy(torch.nn.Module):
    """Mirror parameter that IS the canonical parameter object
    (cross-replica object identity)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = torch.nn.Module()
        m.a = self.a  # the canonical object itself
        return m


class _SharedStorageCopy(torch.nn.Module):
    """Mirror parameter with a DISTINCT object but the SAME storage
    as the canonical parameter (canonical/mirror storage aliasing)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = torch.nn.Module()
        m.a = torch.nn.Parameter(self.a.data)  # distinct object, same storage
        return m


class _CrossMirrorSharedStorage(torch.nn.Module):
    """Storage shared BETWEEN two mirror replicas: the second mirror
    reuses the first mirror's parameter storage. (The mirror root
    keeps the canonical's concrete type so the module-tree
    validation passes and the storage defect is what is exercised.)"""

    _first = None

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        m = type(self)()  # same root type: the defect is the storage, not the tree
        if _CrossMirrorSharedStorage._first is None:
            _CrossMirrorSharedStorage._first = torch.nn.Parameter(torch.ones(2))
            m.a = _CrossMirrorSharedStorage._first
        else:
            m.a = torch.nn.Parameter(_CrossMirrorSharedStorage._first.data)
        return m


class _BufferModule(torch.nn.Module):
    """Component with an UNDECLARED registered buffer (class C by the
    buffer policy default) and one parameter."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("stat", torch.zeros(2))


class _CanonStorageAlias(torch.nn.Module):
    """GLOBALLY-aliased canonical tree: two DISTINCT canonical
    Parameters (a, b) sharing ONE base storage (b wraps a's data with
    the same, zero, view offset)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.b = torch.nn.Parameter(self.a.data)  # distinct object, same storage


class _CanonStorageOffsetAlias(torch.nn.Module):
    """VIEW/OFFSET alias of the same underlying storage: b is a SLICE
    of a's storage — the two tensors have DIFFERENT view data_ptrs but
    the SAME base storage (a base-storage check must catch this where
    a plain data_ptr comparison on each view would not)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.arange(8.0))
        self.b = torch.nn.Parameter(self.a.data[4:])  # offset 16B into a's storage


class _CrossPathMirrorStorageCopy(torch.nn.Module):
    """Cross-path canonical<->mirror alias: the canonical tree is clean
    (a, b own distinct storages), but the factory puts the mirror's
    path-B parameter on CANONICAL path A's storage. Same-path checks
    alone cannot see this; only a GLOBAL any-to-any storage registry
    can."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    def __deepcopy__(self, memo):
        import copy as _copy
        m = type(self)()
        m.a = _copy.deepcopy(self.a, memo)
        m.b = torch.nn.Parameter(self.a.data)  # mirror b on canonical a's storage
        return m


class _DoubleBufferObject(torch.nn.Module):
    """The SAME buffer object registered under TWO logical paths
    (s1, s2). Only the alias-aware
    ``named_buffers(remove_duplicate=False)`` traversal sees both
    paths; the default deduplicating traversal hides one of them."""

    def __init__(self):
        super().__init__()
        buf = torch.zeros(2)
        self.register_buffer("s1", buf)
        self.register_buffer("s2", buf)  # same object, second logical path


class _SharedBufferStorage(torch.nn.Module):
    """Two DISTINCT buffer objects sharing ONE base storage (b2 is a
    zero-offset view of b1's storage) — unsupported buffer aliasing."""

    def __init__(self):
        super().__init__()
        base = torch.zeros(4)
        self.register_buffer("b1", base)
        self.register_buffer("b2", base.data)  # distinct object, same storage


class _ParamBufferObjectAlias(torch.nn.Module):
    """One tensor object registered as BOTH a parameter (w) and a
    buffer (w_buf): any-to-any object-identity aliasing across the
    parameter/buffer kinds."""

    def __init__(self):
        super().__init__()
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.w = p
        self.register_buffer("w_buf", p.data)  # the same underlying object


class _PlainRootCopy(torch.nn.Module):
    """Module-tree defect: the factory returns a PLAIN
    ``torch.nn.Module`` whose parameter paths happen to MATCH the
    canonical's — parameter-path equality alone would admit this
    structurally invalid mirror (root type differs)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        import copy as _copy
        m = torch.nn.Module()
        m.a = _copy.deepcopy(self.a, memo)
        return m


class _WrapA(torch.nn.Module):
    """Plain wrapper module with one parameter (module-tree fixture)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class _WrapB(torch.nn.Module):
    """A DIFFERENT wrapper class with the same internal parameter
    structure (so parameter paths match while the module type does
    not)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class _SubmoduleTypeCopy(torch.nn.Module):
    """Module-tree defect: the mirror's 'sub' submodule is a
    DIFFERENT concrete type than the canonical's, with identical
    parameter paths (sub.w on both sides)."""

    def __init__(self):
        super().__init__()
        self.sub = _WrapA()

    def __deepcopy__(self, memo):
        m = type(self)()  # same root type
        m.sub = _WrapB()  # different submodule type, same parameter paths
        return m


class _MixedTrainingCopy(torch.nn.Module):
    """Module-tree defect: the top-level training flag is preserved,
    but ONE mirror submodule is flipped to the other mode (s1 eval,
    s2 train) — the per-submodule training state differs even though
    the top-level flag matches."""

    def __init__(self):
        super().__init__()
        self.s1 = _WrapA()
        self.s2 = _WrapA()

    def __deepcopy__(self, memo):
        m = type(self)()  # same root type, both submodules re-created
        m.s1 = _WrapA()
        m.s2 = _WrapA()
        m.s1.eval()  # one submodule flipped relative to the canonical (train)
        return m


class _MissingSubmoduleCopy(torch.nn.Module):
    """Module-tree defect: the canonical has an extra parameterless
    submodule ('act') that the mirror omits — parameter paths still
    match ('a' on both sides), so only the module-tree traversal
    catches the structural mismatch."""

    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()  # parameterless submodule
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def __deepcopy__(self, memo):
        import copy as _copy
        m = torch.nn.Module()
        m.a = _copy.deepcopy(self.a, memo)  # 'act' intentionally missing
        return m


class _ReorderedSubmoduleCopy(torch.nn.Module):
    """Module-tree defect: the mirror's submodules are registered in a
    DIFFERENT order than the canonical's (act2, a, act1 vs act1, a,
    act2) — the ordered MODULE path list differs even though both
    module paths exist and the parameter paths match. (Reordering a
    root-level PARAMETER would not be visible in the module tree at
    all, so two submodules are what gets reordered.)"""

    def __init__(self):
        super().__init__()
        self.act1 = torch.nn.ReLU()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.act2 = torch.nn.ReLU()

    def __deepcopy__(self, memo):
        import copy as _copy
        m = torch.nn.Module()
        m.act2 = torch.nn.ReLU()  # act2 registered first
        m.a = _copy.deepcopy(self.a, memo)
        m.act1 = torch.nn.ReLU()  # act1 registered last
        return m


class _CrossComponentParamView(torch.nn.Module):
    """Cross-component alias fixture: 'v' is a VIEW (offset slice) of
    storage owned by ANOTHER component's canonical parameter — the
    two components overlap in base storage, which Phase 12 refuses."""

    def __init__(self, source_param):
        super().__init__()
        self.v = torch.nn.Parameter(source_param.data[2:6])


class _CrossComponentParamObject(torch.nn.Module):
    """Cross-component alias fixture: 'v' IS the same parameter
    OBJECT as another component's canonical parameter (shared tensor
    object identity across registered components)."""

    def __init__(self, source_param):
        super().__init__()
        self.v = source_param


# ---------------------------------------------------------------------------
# independent NumPy reference of the official AdaBelief one-step formula
# (the torch optimizer is never its own oracle)
# ---------------------------------------------------------------------------

def _ref_adabelief_one_step(g, lr=0.1, b1=0.9, b2=0.999, eps=OFFICIAL_EPS_F32):
    # official AdaBelief, one step from zero state, mask == 1.0:
    #   m = (1-b1)*g
    #   v = (1-b2)*(g - m)^2
    #   x += -lr*m / (sqrt(v) + eps)
    g = np.asarray(g, dtype=np.float64)
    m = (1.0 - b1) * g
    v = (1.0 - b2) * (g - m) ** 2
    return -lr * m / (np.sqrt(v) + eps)


# ---------------------------------------------------------------------------
# public API / aliases (production framework path, not test-only fakes)
# ---------------------------------------------------------------------------

def test_nn_alias_exposes_commit2_api():
    # the call-site aliases are the exact same objects as the module's
    assert dfl_nn.ReplicaPlan is ReplicaPlan
    assert dfl_nn.ReplicaPlanError is ReplicaPlanError
    assert dfl_nn.build_replica_mirrors is build_replica_mirrors
    # the alias constructs a working plan
    plan = dfl_nn.ReplicaPlan(CPU)
    assert plan.num_replicas == 1


# ---------------------------------------------------------------------------
# replica plan representation / canonical primary ownership
# ---------------------------------------------------------------------------

def test_plan_from_device_config_cpu_only_is_single_replica():
    # CPU-only (empty device list) is the single "replica" case:
    # canonical == the only replica, no mirrors
    plan = ReplicaPlan.from_device_config(dfl_nn.DeviceConfig([]))
    assert plan.num_replicas == 1
    assert not plan.is_multi
    assert plan.primary_device.type == "cpu"
    assert plan.replica_devices == (torch.device("cpu"),)


def test_plan_replica0_must_be_primary():
    # replica 0 (the canonical/primary replica) must be on the primary
    # device — a plan cannot start anywhere else
    with pytest.raises(ReplicaPlanError, match="replica 0"):
        ReplicaPlan(CPU, [torch.device("cuda", 0)])


def test_plan_rejects_malformed_devices():
    with pytest.raises(ReplicaPlanError, match="at least one replica"):
        ReplicaPlan(CPU, [])
    with pytest.raises(TypeError, match="torch.device"):
        ReplicaPlan("cpu")
    with pytest.raises(TypeError, match="torch.device"):
        ReplicaPlan(CPU, ["cpu"])


def test_plan_from_torch_devices_simulated_two_replicas():
    # the narrow Level-B abstraction: several LOGICAL replicas on one
    # PHYSICAL device (SIMULATED_MULTI_REPLICA shape)
    plan = ReplicaPlan.from_torch_devices(CPU, [CPU, CPU])
    assert plan.num_replicas == 2
    assert plan.is_multi
    assert plan.replica_devices[0] == plan.primary_device
    # three logical replicas on the same physical device also works
    plan3 = ReplicaPlan.from_torch_devices(CPU, [CPU, CPU, CPU])
    assert plan3.num_replicas == 3


@requires_gpu
def test_plan_from_device_config_production_gpu():
    # production path: DeviceConfig -> backend-neutral device
    # resolution (no direct torch.cuda.* in the plan layer)
    _ensure_gpu_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.GPUIndexes([0]), "float32", "NCHW")
    plan = ReplicaPlan.from_device_config(dfl_nn.getCurrentDeviceConfig())
    assert plan.num_replicas == 1
    assert plan.primary_device.type == "cuda"
    assert plan.primary_device.index == 0


# ---------------------------------------------------------------------------
# mirror construction (real archi components, the permitted factory)
# ---------------------------------------------------------------------------

def test_mirror_construction_real_archi_component():
    # a small DeepFakeArchi Encoder through the production factory,
    # mirrored for one secondary replica (two logical replicas on CPU)
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    ms = plan.add_component(comp)
    assert ms.canonical is comp
    assert ms.num_mirrors == 1
    mirror = ms.mirrors[0]
    # the factory preserves the module class (a structurally identical
    # mirror module — the permitted deepcopy + .to mechanism)
    assert type(mirror) is type(comp)
    assert mirror is not comp
    # a real archi component is a Saveable — the mirror copy is too,
    # which is precisely why mirrors must stay OUT of checkpoint
    # ownership (checked in the isolation tests below)
    assert isinstance(mirror, Saveable)
    # the mirror is registered with the plan
    assert plan.components[0] is ms
    # module-tree validation (positive path, real tree): identical
    # ordered module paths, identical concrete module types, identical
    # per-submodule training state — the complete named_modules
    # traversal of both trees matches
    canon_modules = list(comp.named_modules(remove_duplicate=False))
    mirror_modules = list(mirror.named_modules(remove_duplicate=False))
    assert [n for n, _ in mirror_modules] == [n for n, _ in canon_modules]
    for (n1, m1), (n2, m2) in zip(canon_modules, mirror_modules):
        assert n1 == n2
        assert type(m2) is type(m1)
        assert m2.training == m1.training
    plan.dispose()


def test_mirror_construction_real_bn_component_class_a_buffers():
    # a real inference-only BatchNorm2D component: its registered
    # buffers are CLASS A via the component's own explicit declaration
    bn = _make_bn_component()
    plan = _plan_2cpu()
    ms = plan.add_component(bn)
    mirror = ms.mirrors[0]
    # buffer mapping: identical ordered buffer paths on the mirror
    assert [n for n, _ in mirror.named_buffers()] == \
        [n for n, _ in bn.named_buffers()]
    assert [n for n, _ in mirror.named_buffers()] == \
        ["running_mean", "running_var"]
    # resolved classes come from the component's own declaration (no
    # caller registry, no heuristics)
    assert ms.buffer_classes == {"running_mean": "A",
                                 "running_var": "A"}
    plan.dispose()


def test_mirror_preserves_training_eval_mode():
    comp = _make_encoder_component()
    # eval() BEFORE construction: the mirror follows the canonical mode
    comp.eval()
    plan = _plan_2cpu()
    plan.add_component(comp)
    assert plan.components[0].mirrors[0].training is False
    plan.dispose()

    comp2 = _make_encoder_component()
    comp2.train()
    plan2 = _plan_2cpu()
    plan2.add_component(comp2)
    assert plan2.components[0].mirrors[0].training is True
    plan2.dispose()


@requires_gpu
def test_mirror_construction_on_gpu_simulated_replicas():
    # SIMULATED_MULTI_REPLICA (GPU flavor): one real archi component on
    # cuda:0, mirrors for two logical replicas on the same physical
    # device. Physical 2-GPU acceptance: PENDING_ENVIRONMENTALLY /
    # NOT_VERIFIED.
    _ensure_gpu_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.GPUIndexes([0]), "float32", "NCHW")
    dev = torch.device("cuda", 0)
    plan = ReplicaPlan.from_torch_devices(dev, [dev, dev])
    comp = _make_encoder_component(device=dev)
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    for mp in ms.mirror_params[1]:
        assert mp.device == dev
    for mb in ms.mirror_buffers[1]:
        assert mb.device == dev
    plan.dispose()


# ---------------------------------------------------------------------------
# distinct objects / distinct storage
# ---------------------------------------------------------------------------

def test_mirrors_are_distinct_objects():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    ms = plan.add_component(comp)
    canon_ids = {id(p) for p in comp.parameters(recurse=True)}
    for mparams in ms.mirror_params.values():
        for mp in mparams:
            assert mp is not comp
            assert id(mp) not in canon_ids
    # the mirror's parameter objects are all distinct among themselves
    mparams = ms.mirror_params[1]
    assert len({id(p) for p in mparams}) == len(mparams)
    plan.dispose()


def test_mirrors_own_distinct_storage_across_replicas():
    # three logical replicas on one physical device: canonical, mirror 1
    # and mirror 2 must own three DISTINCT storages per parameter
    comp = _make_encoder_component()
    plan = ReplicaPlan.from_torch_devices(CPU, [CPU, CPU, CPU])
    ms = plan.add_component(comp)
    assert ms.num_mirrors == 2
    for cp, m1, m2 in zip(ms.canonical_params,
                          ms.mirror_params[1], ms.mirror_params[2]):
        cptr = cp.untyped_storage().data_ptr()
        p1 = m1.untyped_storage().data_ptr()
        p2 = m2.untyped_storage().data_ptr()
        assert cptr != p1 and cptr != p2 and p1 != p2
    plan.dispose()


# ---------------------------------------------------------------------------
# alias rejection (fail-fast at construction)
# ---------------------------------------------------------------------------

def test_duplicate_identity_inside_canonical_tree_rejected():
    # two logical paths (a, b) resolving to the SAME Parameter object:
    # Phase 12 defines no safe mirror policy for aliased parameters
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="duplicate object identity"):
        plan.add_component(_AliasedCanonical())
    plan.dispose()


def test_mirror_sharing_canonical_object_rejected():
    # a mirror parameter that IS the canonical parameter object
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="IS the canonical parameter"):
        plan.add_component(_SameObjectCopy())
    plan.dispose()


def test_canonical_mirror_shared_storage_rejected():
    # distinct objects but the SAME storage: writing a mirror would
    # corrupt canonical (checkpoint-owned) state
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="share the same storage"):
        plan.add_component(_SharedStorageCopy())
    plan.dispose()


def test_storage_shared_between_mirror_replicas_rejected():
    _CrossMirrorSharedStorage._first = None
    plan = ReplicaPlan.from_torch_devices(CPU, [CPU, CPU, CPU])
    with pytest.raises(ReplicaPlanError, match="DISTINCT storage"):
        plan.add_component(_CrossMirrorSharedStorage())
    plan.dispose()
    _CrossMirrorSharedStorage._first = None


# ---------------------------------------------------------------------------
# GLOBAL alias safety: any-to-any, base-storage aware (the checks are
# GLOBAL registries, not same-path pairs), including cross-component
# overlap at the plan level
# ---------------------------------------------------------------------------

def test_canonical_params_sharing_storage_rejected():
    # (A) two DISTINCT canonical Parameters sharing one base storage
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="base storage"):
        plan.add_component(_CanonStorageAlias())
    plan.dispose()


def test_canonical_params_sharing_storage_view_offset_rejected():
    # (D) view/offset alias: b is a slice of a's storage — the two
    # tensors have different view data_ptrs but the SAME base storage;
    # the check must reason on base-storage identity, not the view's
    # own data_ptr
    plan = _plan_2cpu()
    m = _CanonStorageOffsetAlias()
    assert m.a.data_ptr() != m.b.data_ptr()  # different view offsets...
    assert m.a.untyped_storage().data_ptr() == \
        m.b.untyped_storage().data_ptr()     # ...same base storage
    with pytest.raises(ReplicaPlanError, match="base storage"):
        plan.add_component(m)
    plan.dispose()


def test_cross_path_mirror_canonical_storage_alias_rejected():
    # (B) the canonical tree is clean (a, b own distinct storages);
    # the mirror's path-B parameter sits on CANONICAL path A's
    # storage — a same-path check cannot see this; the GLOBAL
    # any-to-any registry must
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError,
                       match="base storage of canonical"):
        plan.add_component(_CrossPathMirrorStorageCopy())
    plan.dispose()


def test_same_buffer_object_two_logical_paths_rejected():
    # (E) the SAME buffer object registered under two logical paths:
    # only the alias-aware named_buffers(remove_duplicate=False)
    # traversal surfaces both paths (the default deduplicating
    # traversal hides one of them)
    plan = _plan_2cpu()
    m = _DoubleBufferObject()
    paths = [n for n, _ in m.named_buffers(recurse=True, remove_duplicate=False)]
    assert paths == ["s1", "s2"]  # both logical paths exist
    with pytest.raises(ReplicaPlanError, match="duplicate object identity"):
        plan.add_component(m)
    plan.dispose()


def test_distinct_buffers_sharing_storage_rejected():
    # (F) two DISTINCT buffer objects sharing one base storage — no
    # documented Phase 12 policy allows buffer storage aliasing
    plan = _plan_2cpu()
    m = _SharedBufferStorage()
    b1 = m._buffers["b1"]
    b2 = m._buffers["b2"]
    assert b1 is not b2
    assert b1.untyped_storage().data_ptr() == b2.untyped_storage().data_ptr()
    with pytest.raises(ReplicaPlanError, match="base storage"):
        plan.add_component(m)
    plan.dispose()


def test_param_buffer_object_alias_rejected():
    # (H) one tensor registered as BOTH a parameter and a buffer:
    # the any-to-any object/storage registries span both kinds
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="base storage"):
        plan.add_component(_ParamBufferObjectAlias())
    plan.dispose()


def test_cross_component_storage_overlap_rejected():
    # (G) component 2's parameter is an offset VIEW into component 1's
    # canonical storage: overlapping base storage ACROSS registered
    # components is refused at the plan level
    comp1 = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp1)
    p = next(comp1.parameters())
    comp2 = _CrossComponentParamView(p)
    with pytest.raises(ReplicaPlanError,
                       match="another registered component"):
        plan.add_component(comp2)
    plan.dispose()


def test_cross_component_object_sharing_rejected():
    # (G2) component 2 references the SAME parameter OBJECT as
    # component 1: shared object identity across registered
    # components is refused at the plan level. The parameter's
    # component-1-bound _dfl_name is stripped first so that the
    # cross-component check (not the additional per-component
    # _dfl_name invariant) is what rejects it.
    comp1 = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp1)
    p = next(comp1.parameters())
    if hasattr(p, "_dfl_name"):
        del p._dfl_name
    comp2 = _CrossComponentParamObject(p)
    with pytest.raises(ReplicaPlanError,
                       match="another registered component"):
        plan.add_component(comp2)
    plan.dispose()


# ---------------------------------------------------------------------------
# module-tree validation: matching parameter paths alone do NOT make a
# mirror structurally valid (named_modules traversal, remove_duplicate=False)
# ---------------------------------------------------------------------------

def test_plain_module_mirror_tree_rejected():
    # the factory returns a PLAIN torch.nn.Module with matching
    # parameter paths: the concrete module type at the root differs
    # from the canonical's — structurally invalid
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="CONCRETE module type"):
        plan.add_component(_PlainRootCopy())
    plan.dispose()


def test_submodule_type_change_rejected():
    # parameter paths match (sub.w on both sides), but the mirror's
    # 'sub' submodule is a different concrete type than the
    # canonical's
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="CONCRETE module type"):
        plan.add_component(_SubmoduleTypeCopy())
    plan.dispose()


def test_mixed_submodule_training_flags_rejected():
    # the top-level training flag is preserved on both sides, but ONE
    # mirror submodule is flipped to eval: the PER-SUBMODULE
    # training state differs (the top-level flag alone is not
    # sufficient)
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="PER-SUBMODULE"):
        plan.add_component(_MixedTrainingCopy())
    plan.dispose()


def test_missing_submodule_path_rejected():
    # the canonical has a parameterless submodule ('act') the mirror
    # omits: parameter paths still match, the module-tree traversal
    # catches the missing path
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError,
                       match="module paths are not identical"):
        plan.add_component(_MissingSubmoduleCopy())
    plan.dispose()


def test_reordered_submodule_paths_rejected():
    # the mirror's submodules are ordered differently (act before a):
    # the ORDERED module path list differs
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="REORDERED"):
        plan.add_component(_ReorderedSubmoduleCopy())
    plan.dispose()


def test_real_archi_module_tree_preserved():
    # positive path on a REAL leras/DeepFakeArchi tree: the factory
    # preserves the complete module tree — identical ordered module
    # paths, identical concrete types, identical per-submodule
    # training state (and the implied identical module count /
    # parent-child structure)
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    ms = plan.add_component(comp)
    mirror = ms.mirrors[0]
    canon_modules = list(comp.named_modules(remove_duplicate=False))
    mirror_modules = list(mirror.named_modules(remove_duplicate=False))
    assert [n for n, _ in mirror_modules] == [n for n, _ in canon_modules]
    assert len(mirror_modules) == len(canon_modules)
    for (n1, m1), (n2, m2) in zip(canon_modules, mirror_modules):
        assert n1 == n2
        assert type(m2) is type(m1)
        assert m2.training == m1.training
    plan.dispose()


# ---------------------------------------------------------------------------
# deterministic path mapping: identical ordered paths; reject
# reordered / missing
# ---------------------------------------------------------------------------

def test_path_mapping_is_deterministic_and_ordered():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    ms = plan.add_component(comp)
    # the component's ordered logical paths ARE its
    # named_parameters(remove_duplicate=False) paths
    assert ms.paths == [n for n, _ in comp.named_parameters(remove_duplicate=False)]
    # positional alignment canonical <-> mirror
    for (cn, cp), (mn, mp) in zip(comp.named_parameters(remove_duplicate=False),
                                  ms.mirrors[0].named_parameters(remove_duplicate=False)):
        assert cn == mn
        assert tuple(cp.shape) == tuple(mp.shape)
        assert cp.dtype == mp.dtype
    # a fresh plan + fresh component of the same architecture produces
    # the same deterministic path order
    comp2 = _make_encoder_component()
    plan2 = _plan_2cpu()
    ms2 = plan2.add_component(comp2)
    assert ms2.paths == ms.paths
    plan.dispose()
    plan2.dispose()


def test_reordered_paths_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="REORDERED"):
        plan.add_component(_ReorderParamCopy())
    plan.dispose()


def test_missing_paths_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="missing"):
        plan.add_component(_DropParamCopy())
    plan.dispose()


# ---------------------------------------------------------------------------
# shape / dtype / requires_grad mismatch rejection
# ---------------------------------------------------------------------------

def test_shape_mismatch_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="shape"):
        plan.add_component(_WrongShapeCopy())
    plan.dispose()


def test_dtype_mismatch_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="dtype"):
        plan.add_component(_WrongDtypeCopy())
    plan.dispose()


def test_requires_grad_mismatch_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="requires_grad"):
        plan.add_component(_WrongRequiresGradCopy())
    plan.dispose()


# ---------------------------------------------------------------------------
# _dfl_name: additional invariant, never the mapping key
# ---------------------------------------------------------------------------

def test_copied_dfl_name_absence_tolerated():
    # module-level copy.deepcopy does not reliably preserve custom
    # Parameter attributes (the round-2 reviewer verified the loss on
    # real archi hierarchies); the mapping is keyed by module-local
    # PATHS, so construction must succeed whether or not the mirror
    # retained the copied _dfl_name
    comp = _make_encoder_component()  # binds _dfl_name on every weight
    plan = _plan_2cpu()
    ms = plan.add_component(comp)  # must not raise
    checked = 0
    for cp, mp in zip(ms.canonical_params, ms.mirror_params[1]):
        assert getattr(cp, "_dfl_name", None) is not None
        v = getattr(mp, "_dfl_name", None)
        # lost (None) or faithfully copied (== canonical) — both OK
        assert v is None or v == getattr(cp, "_dfl_name", None)
        checked += 1
    assert checked == len(ms.canonical_params)
    plan.dispose()


def test_dfl_name_inconsistency_rejected():
    # a canonical _dfl_name that is NOT consistent with the official
    # name the optimizer binding expects at that module-local path is
    # a construction-time error (additional invariant, fail-fast)
    comp = _make_encoder_component()
    first_param = next(comp.parameters())
    first_param._dfl_name = "totally/wrong/name:0"
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="_dfl_name"):
        plan.add_component(comp)
    plan.dispose()


# ---------------------------------------------------------------------------
# buffer policy (named_buffers): class A / unclassified->C / class B
# ---------------------------------------------------------------------------

def test_unclassified_buffer_fails_fast_class_c():
    # an UNCLASSIFIED registered buffer is CLASS C by default: the plan
    # refuses to build (no silent divergence at runtime)
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="class C"):
        plan.add_component(_BufferModule(name="uncl"))
    plan.dispose()


def test_buffer_class_declared_by_caller_registry():
    # the caller's explicit {buffer path: class} registry classifies an
    # otherwise-unclassified buffer as A
    plan = _plan_2cpu()
    ms = plan.add_component(_BufferModule(name="decl"),
                            buffer_classes={"stat": "A"})
    assert ms.buffer_classes == {"stat": "A"}
    plan.initial_sync()
    # the class-A buffer is mirrored (distinct object, same path,
    # target device) and carries the canonical value
    mb = dict(ms.mirrors[0].named_buffers())["stat"]
    cb = ms.canonical_buffers[0]
    assert mb is not cb
    assert tuple(mb.shape) == tuple(cb.shape)
    assert mb.device == plan.primary_device
    assert torch.equal(mb.detach(), cb.detach())
    plan.dispose()


def test_buffer_class_b_declaration_fails_loud():
    # class B requires a registered reduction/synchronization rule;
    # Phase 12 registers none, so a B declaration fails loud
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="class B"):
        plan.add_component(_BufferModule(name="clsB"),
                           buffer_classes={"stat": "B"})
    plan.dispose()


def test_class_a_buffer_copied_and_resynced():
    # class-A buffers are copied at the initial sync and re-synced by
    # the same canonical -> mirror mechanism when the canonical value
    # changes
    bn = _make_bn_component()
    plan = _plan_2cpu()
    ms = plan.add_component(bn)
    plan.initial_sync()
    # exact copy at the initial sync (the BN is its own component
    # root: buffer paths 'running_mean' / 'running_var')
    mir_rm = dict(ms.mirrors[0].named_buffers())["running_mean"]
    mir_rv = dict(ms.mirrors[0].named_buffers())["running_var"]
    assert torch.equal(bn.running_mean, mir_rm)
    assert torch.equal(bn.running_var, mir_rv)
    # canonical values change (class-A re-sync case) -> post-step sync
    with torch.no_grad():
        bn.running_mean.add_(1.0)
        bn.running_var.add_(0.5)
    plan.sync_from_canonical()
    assert torch.equal(bn.running_mean, mir_rm)
    assert torch.equal(bn.running_var, mir_rv)
    plan.dispose()


# ---------------------------------------------------------------------------
# initial / post-step sync
# ---------------------------------------------------------------------------

def test_initial_sync_exact():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    for cp, mp in zip(ms.canonical_params, ms.mirror_params[1]):
        assert torch.equal(cp.detach(), mp.detach())
    for cb, mb in zip(ms.canonical_buffers, ms.mirror_buffers[1]):
        assert torch.equal(cb.detach(), mb.detach())
    plan.dispose()


def test_post_step_sync_after_initial():
    # initial sync -> canonical changes (simulated successful step) ->
    # post-step sync carries the update into every mirror, bit-exact
    bn = _make_bn_component()
    plan = _plan_2cpu()
    plan.add_component(bn)
    plan.initial_sync()
    ms = plan.components[0]
    with torch.no_grad():
        for p in bn.parameters():
            p.mul_(1.5)
        bn.running_mean.add_(1.0)
    plan.sync_from_canonical()
    for cp, mp in zip(ms.canonical_params, ms.mirror_params[1]):
        assert torch.equal(cp.detach(), mp.detach())
    for cb, mb in zip(ms.canonical_buffers, ms.mirror_buffers[1]):
        assert torch.equal(cb.detach(), mb.detach())
    plan.dispose()


def test_sync_before_initial_sync_refused():
    # the approved lifecycle: the initial sync must precede any
    # post-step sync
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    with pytest.raises(ReplicaPlanError, match="initial_sync"):
        plan.sync_from_canonical()
    plan.dispose()


# ---------------------------------------------------------------------------
# a mirror write can never touch canonical (checkpoint-owned) state
# ---------------------------------------------------------------------------

def test_mirror_write_cannot_touch_canonical_storage():
    # mirror parameter writes, unauthorized replica-side class-A buffer
    # writes, and a full replica forward/backward can never reach
    # canonical (checkpoint-owned) storage or gradients
    comp = _make_encoder_component()      # parameters only
    bn = _make_bn_component()             # parameters + class-A buffers
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.add_component(bn)
    plan.initial_sync()
    enc = plan.components[0]
    bnc = plan.components[1]
    # (1) a direct mirror parameter write stays on the mirror storage
    enc_snap = [p.detach().clone() for p in enc.canonical_params]
    with torch.no_grad():  # optimizer-style storage write on the mirror
        enc.mirror_params[1][0].add_(1.0)
    for cp, sp in zip(enc.canonical_params, enc_snap):
        assert torch.equal(cp.detach(), sp)
    # (2) an unauthorized replica-side write of a class-A buffer
    bn_snap = [b.detach().clone() for b in bnc.canonical_buffers]
    with torch.no_grad():
        for mb in bnc.mirror_buffers[1]:
            mb.add_(1.0)
    for cb, sb in zip(bnc.canonical_buffers, bn_snap):
        assert torch.equal(cb.detach(), sb)
    # (3) a full replica forward/backward leaves canonical values AND
    #     canonical gradients untouched
    x = torch.randn(2, 3, 16, 16)
    loss = enc.mirrors[0](x).sum()
    loss.backward()
    for mp in enc.mirror_params[1]:
        assert mp.grad is not None  # the mirror absorbed its own grads
    for cp in enc.canonical_params:
        assert cp.grad is None
    for cp, sp in zip(enc.canonical_params, enc_snap):
        assert torch.equal(cp.detach(), sp)
    plan.dispose()


# ---------------------------------------------------------------------------
# checkpoint / Saveable isolation
# ---------------------------------------------------------------------------

def test_mirrors_absent_from_saveable_enumeration_and_filename_list():
    comp = _make_encoder_component()
    bn = _make_bn_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.add_component(bn)
    plan.initial_sync()
    mirror_obj_ids = set()
    for c in plan.components:
        for m in c.mirrors:
            mirror_obj_ids |= {id(p) for p in m.parameters(recurse=True)}
            mirror_obj_ids |= {id(b) for b in m.buffers(recurse=True)}
    assert mirror_obj_ids
    # (a) the canonical saveables enumerate ONLY canonical objects
    for saveable in (comp, bn):
        for _, obj in saveable._iter_official_weights():
            assert id(obj) not in mirror_obj_ids
    # (b) the mirror modules are not children of any canonical tree
    for c in plan.components:
        tree_ids = {id(m) for m in c.canonical.modules()}
        for m in c.mirrors:
            assert id(m) not in tree_ids
    # (c) simulated model_filename_list (the official
    #     get_model_filename_list shape [[saveable, filename], ...]):
    #     only canonical components + the canonical optimizer; no
    #     mirror object may ever appear as an entry
    opt = dfl_nn.AdaBelief(lr=0.1, name="G")
    opt.initialize_variables(plan.components[0].canonical_params)
    filename_list = [[comp, "encoder"], [opt, "G"]]
    for entry, _name in filename_list:
        for _, obj in entry._iter_official_weights():
            assert id(obj) not in mirror_obj_ids
    # (d) the mirrors are self-contained saveables — exactly the reason
    #     they must never be registered in a filename list
    for c in plan.components:
        for m in c.mirrors:
            assert isinstance(m, Saveable)
            assert len(m._iter_official_weights()) == \
                len(c.canonical._iter_official_weights())
    plan.dispose()


def test_save_weights_file_has_canonical_keys_only(tmp_path):
    # a saved checkpoint carries exactly the canonical component's
    # official keys — mirror construction created no new keys, no
    # filename/state-naming change. Checkpoint scratch is pytest's
    # per-test ``tmp_path`` (a unique temporary directory per test;
    # no pre-existing file can be overwritten, no repo-private
    # scratch path is used).
    comp = _make_encoder_component(scope="encoder")
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    f = tmp_path / "encoder.npy"
    comp.save_weights(str(f), force_dtype=np.float32)
    with open(f, "rb") as fh:
        saved = pickle.load(fh)
    expected_names = {name for name, _ in comp._iter_official_weights()}
    assert set(saved) == expected_names
    # every saved value matches the canonical tensor exactly, in the
    # OFFICIAL layout the file always stores (save_weights writes
    # official DFL layouts to disk)
    for name, obj in comp._iter_official_weights():
        key = name if name.endswith(":0") else name + ":0"
        official = comp.convert_weight_to_official(
            obj.detach().cpu().numpy().copy(), obj)
        assert np.array_equal(saved[key], official)
    plan.dispose()


def test_disposable_mirrors_reconstructed_after_resume(tmp_path):
    # mirrors are never checkpoint-owned: after a save -> dispose ->
    # resume (fresh canonical module + load_weights), the mirrors are
    # RECONSTRUCTED from the canonical by the same factory + initial
    # sync, and they carry the resumed values. Checkpoint scratch is
    # pytest's per-test ``tmp_path``.
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    # simulate a successful step: canonical values move on
    with torch.no_grad():
        for p in comp.parameters():
            p.mul_(1.5)
    f = tmp_path / "encoder.npy"
    comp.save_weights(str(f), force_dtype=np.float32)
    plan.dispose()  # mirrors released; they were never saved

    # resume: fresh canonical module loads the checkpoint
    comp2 = _make_encoder_component()
    assert comp2.load_weights(str(f)) is True
    plan2 = _plan_2cpu()
    plan2.add_component(comp2)
    plan2.initial_sync()
    ms2 = plan2.components[0]
    # reconstructed mirrors == resumed canonical values, per path
    for (cn, cp), (mn, mp) in zip(comp2.named_parameters(remove_duplicate=False),
                                  ms2.mirrors[0].named_parameters(remove_duplicate=False)):
        assert cn == mn
        assert torch.equal(cp, mp)
    # and match the saved file values (compared in the OFFICIAL layout
    # the file stores; the resumed module holds torch-layout values)
    with open(f, "rb") as fh:
        saved = pickle.load(fh)
    for name, obj in comp2._iter_official_weights():
        key = name if name.endswith(":0") else name + ":0"
        official = comp2.convert_weight_to_official(
            obj.detach().cpu().numpy().copy(), obj)
        assert np.array_equal(saved[key], official)
    plan2.dispose()


# ---------------------------------------------------------------------------
# no optimizer ownership (the optimizer owns CANONICAL params only)
# ---------------------------------------------------------------------------

def test_optimizer_owns_canonical_params_only():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=0.0, lr_dropout=1.0, name="G")
    opt.initialize_variables(ms.canonical_params)
    # the optimizer weight list is exactly the canonical parameter
    # objects (canonical ownership)
    assert list(opt._weights) == ms.canonical_params
    canon_ids = {id(p) for p in ms.canonical_params}
    # every optimizer state buffer tracks a CANONICAL parameter
    for _buf, tracked in opt._state_owner.items():
        assert id(tracked) in canon_ids
    # no optimizer state for any mirror parameter
    mirror_ids = {id(p) for p in ms.mirror_params[1]}
    assert not (canon_ids & mirror_ids)
    for tracked in opt._state_owner.values():
        assert id(tracked) not in mirror_ids
    # the checkpoint-owned optimizer weights reference the canonical
    # bindings only (official ms_/vs_ naming from _dfl_name)
    names = [n for n, _ in opt._iter_official_weights()]
    assert names[0] == "iters:0"
    # state sub-names are built from _dfl_name with ':' -> '_'
    canon_dfl = {p._dfl_name.replace(":", "_")
                 for p in ms.canonical_params
                 if hasattr(p, "_dfl_name")}
    assert any(any(d in n for d in canon_dfl) for n in names[1:])
    plan.dispose()


def test_averaged_gradient_step_updates_canonical_only():
    # full Commit-2 data path: per-replica forward/backward ->
    # canonicalize_grads (association + rebind) -> average_gv_list ->
    # ONE canonical optimizer step; the step updates CANONICAL
    # parameters only (official AdaBelief formula vs the independent
    # NumPy reference); mirrors stay stale until the explicit sync
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    for m in ms.mirrors:
        m.train()
    x = torch.randn(2, 3, 16, 16)
    comp(x).sum().backward()          # replica 0 on canonical
    ms.mirrors[0](x.clone()).sum().backward()  # replica 1 on its mirror
    list0 = [(p.grad, p) for p in ms.canonical_params]
    list1 = [(mp.grad, mp) for mp in ms.mirror_params[1]]
    for pairs in (list0, list1):
        for g, _ in pairs:
            assert g is not None  # every parameter has a flowing grad
    norm = plan.canonicalize_grads([list0, list1])
    # every replica's entries are rebound to the CANONICAL parameters
    for replica_pairs in norm:
        for _g, p in replica_pairs:
            assert id(p) in {id(c) for c in ms.canonical_params}
    agged = average_gv_list(norm)
    assert [pv for _g, pv in agged] == ms.canonical_params

    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=0.0, lr_dropout=1.0, name="G")
    opt.initialize_variables(ms.canonical_params)
    canon_snap = [p.detach().clone() for p in ms.canonical_params]
    mir_snap = [mp.detach().clone() for mp in ms.mirror_params[1]]
    opt.get_update_op(agged)()
    assert int(opt.iterations.item()) == 1  # ONE canonical step

    # canonical parameters moved exactly per the official formula
    # (both replicas are identical after the sync, so the mean grad
    # equals each per-replica grad)
    for (g, cp), snap in zip(agged, canon_snap):
        g_np = g.detach().cpu().numpy().astype(np.float64)
        ref = _ref_adabelief_one_step(g_np, lr=0.1)
        actual = cp.detach().cpu().numpy().astype(np.float64)
        assert np.abs(actual - (snap.detach().cpu().numpy().astype(np.float64) + ref)).max() < 1e-5
    # mirrors are untouched by the optimizer...
    for mp, snap in zip(ms.mirror_params[1], mir_snap):
        assert torch.equal(mp.detach(), snap)
    # ...until the explicit post-step sync
    plan.sync_from_canonical()
    for cp, mp in zip(ms.canonical_params, ms.mirror_params[1]):
        assert torch.equal(cp.detach(), mp.detach())
    plan.dispose()


# ---------------------------------------------------------------------------
# canonicalize_grads: association validation + device transfer
# ---------------------------------------------------------------------------

def test_canonicalize_grads_rebinds_and_validates():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    p0, p1 = ms.canonical_params[0], ms.canonical_params[1]
    m0, m1 = ms.mirror_params[1][0], ms.mirror_params[1][1]
    z0 = torch.zeros_like(p0)
    z1 = torch.zeros_like(p1)
    zm0 = torch.zeros_like(m0)
    zm1 = torch.zeros_like(m1)
    # replica 0: canonical params; replica 1: mirror counterparts
    norm = plan.canonicalize_grads([[(z0, p0), (z1, p1)],
                                    [(zm0, m0), (zm1, m1)]])
    # replica 0 gradients pass through unchanged (already canonical)
    assert norm[0][0][0] is z0
    # replica 1 entries are rebound to the CANONICAL parameter objects
    assert norm[1][0][1] is p0
    assert norm[1][1][1] is p1
    assert norm[1][0][0].device == plan.primary_device
    # now consumable by the Commit-1 aggregation
    agged = average_gv_list(norm)
    assert [pv for _g, pv in agged] == [p0, p1]
    plan.dispose()


def test_canonicalize_grads_association_errors():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    p0 = ms.canonical_params[0]
    p1 = ms.canonical_params[1]
    m0 = ms.mirror_params[1][0]
    z = lambda p: torch.zeros_like(p)
    # replica 0 must reference CANONICAL parameters
    with pytest.raises(ReplicaPlanError, match="CANONICAL"):
        plan.canonicalize_grads([[(z(m0), m0)], [(z(p0), p0)]])
    # replica 1 must reference the MIRROR counterpart of replica 0's
    # canonical parameter
    with pytest.raises(ReplicaPlanError, match="mirror counterpart"):
        plan.canonicalize_grads([[(z(p0), p0)], [(z(p0), p0)]])
    # a mirror of the WRONG canonical parameter at this position:
    # position 1 is p1 on replica 0 but the mirror of p0 on replica 1
    with pytest.raises(ReplicaPlanError, match="mirror counterpart"):
        plan.canonicalize_grads([[(z(p0), p0), (z(p1), p1)],
                                 [(z(m0), m0), (z(m0), m0)]])
    # a parameter no mirror tracks (unmirrored component / stray param)
    stray = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ReplicaPlanError, match="CANONICAL"):
        plan.canonicalize_grads([[(z(stray), stray)], [(z(p0), p0)]])
    # replica count mismatch
    with pytest.raises(ReplicaPlanError, match="per-replica"):
        plan.canonicalize_grads([[(z(p0), p0)]])
    # per-replica length mismatch
    with pytest.raises(ReplicaPlanError, match="same variables"):
        plan.canonicalize_grads([[(z(p0), p0), (z(p1), p1)],
                                 [(z(m0), m0)]])
    plan.dispose()


@requires_gpu
def test_cross_device_secondary_grad_transfer():
    # mixed-device plan: canonical on cuda:0, the secondary replica on
    # CPU — the secondary replica's gradients are transferred to the
    # canonical device by canonicalize_grads (Commit 1 never transfers)
    _ensure_gpu_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.GPUIndexes([0]), "float32", "NCHW")
    dev = torch.device("cuda", 0)
    plan = ReplicaPlan.from_torch_devices(dev, [dev, CPU])
    comp = _make_encoder_component(device=dev)
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    # the mirror lives on the CPU replica device
    for mp in ms.mirror_params[1]:
        assert mp.device == CPU
    comp.eval()
    ms.mirrors[0].eval()
    x_gpu = torch.randn(1, 3, 16, 16, device=dev)
    x_cpu = torch.randn(1, 3, 16, 16)
    comp(x_gpu).sum().backward()
    ms.mirrors[0](x_cpu).sum().backward()
    list0 = [(p.grad, p) for p in ms.canonical_params]
    list1 = [(mp.grad, mp) for mp in ms.mirror_params[1]]
    for g, _ in list1:
        assert g.device == CPU  # secondary grads are on the replica device
    norm = plan.canonicalize_grads([list0, list1])
    for g, _ in norm[1]:
        assert g.device == dev  # transferred to the canonical device
    agged = average_gv_list(norm)
    for g, _ in agged:
        assert g.device == dev
    plan.dispose()


# ---------------------------------------------------------------------------
# SIMULATED_MULTI_REPLICA integration (production framework path)
# ---------------------------------------------------------------------------

def test_simulated_two_replicas_cpu_full_flow():
    # two LOGICAL replicas through the production framework path
    # (real nn.initialize foundation, real DeepFakeArchi component,
    # real AdaBelief optimizer, real average_gv_list — no test-only
    # fakes in the production flow): forward/backward per replica ->
    # canonicalize_grads -> average_gv_list -> ONE canonical step ->
    # post-step sync -> forward parity
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    for m in ms.mirrors:
        m.train()
    x = torch.randn(2, 3, 16, 16)
    comp(x).sum().backward()
    ms.mirrors[0](x.clone()).sum().backward()
    list0 = [(p.grad, p) for p in ms.canonical_params]
    list1 = [(mp.grad, mp) for mp in ms.mirror_params[1]]
    agged = average_gv_list(plan.canonicalize_grads([list0, list1]))
    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=0.0, lr_dropout=1.0, name="G")
    opt.initialize_variables(ms.canonical_params)
    canon_snap = [p.detach().clone() for p in ms.canonical_params]
    opt.get_update_op(agged)()
    assert int(opt.iterations.item()) == 1
    for (g, cp), snap in zip(agged, canon_snap):
        ref = _ref_adabelief_one_step(
            g.detach().cpu().numpy().astype(np.float64), lr=0.1)
        actual = cp.detach().cpu().numpy().astype(np.float64)
        assert np.abs(actual - (snap.detach().cpu().numpy().astype(np.float64) + ref)).max() < 1e-5
    plan.sync_from_canonical()
    # after the sync both replicas are bit-exact (same device, same
    # weights) — the official single-session parity at the weight level
    comp.eval()
    ms.mirrors[0].eval()
    assert torch.equal(comp(x), ms.mirrors[0](x))
    plan.dispose()


@requires_gpu
def test_simulated_multi_replica_two_logical_on_one_physical_gpu():
    # SIMULATED_MULTI_REPLICA: two LOGICAL replicas mapped onto ONE
    # physical device (cuda:0) through the production factory/mapping/
    # sync/averaging path. Physical 2-GPU acceptance is
    # PENDING_ENVIRONMENTALLY / NOT_VERIFIED (this machine has a
    # single GPU); the label must never be read as physical coverage.
    _ensure_gpu_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.GPUIndexes([0]), "float32", "NCHW")
    dev = torch.device("cuda", 0)
    plan = ReplicaPlan.from_torch_devices(dev, [dev, dev])
    comp = _make_encoder_component(device=dev)
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    for m in ms.mirrors:
        m.train()
    x = torch.randn(1, 3, 16, 16, device=dev)
    comp(x).sum().backward()
    ms.mirrors[0](x.clone()).sum().backward()
    list0 = [(p.grad, p) for p in ms.canonical_params]
    list1 = [(mp.grad, mp) for mp in ms.mirror_params[1]]
    norm = plan.canonicalize_grads([list0, list1])
    for g, _ in norm[1]:
        assert g.device == dev  # already canonical-device
    agged = average_gv_list(norm)
    opt = dfl_nn.AdaBelief(lr=0.1, clipnorm=0.0, lr_dropout=1.0, name="G")
    opt.initialize_variables(ms.canonical_params)
    canon_snap = [p.detach().clone() for p in ms.canonical_params]
    opt.get_update_op(agged)()
    assert int(opt.iterations.item()) == 1
    for (g, cp), snap in zip(agged, canon_snap):
        ref = _ref_adabelief_one_step(
            g.detach().cpu().numpy().astype(np.float64), lr=0.1)
        actual = cp.detach().cpu().numpy().astype(np.float64)
        assert np.abs(actual - (snap.detach().cpu().numpy().astype(np.float64) + ref)).max() < 1e-5
    plan.sync_from_canonical()
    # same physical backend for both logical replicas -> bit-exact
    comp.eval()
    ms.mirrors[0].eval()
    assert torch.equal(comp(x), ms.mirrors[0](x))
    plan.dispose()


@requires_gpu
def test_production_plan_single_replica_gpu_no_mirrors():
    # the 1-GPU production plan (DeviceConfig-derived) is the
    # single-replica case: NO mirrors, initial_sync a no-op,
    # canonicalize_grads the official identity — the foundation of the
    # unchanged single-GPU training path
    _ensure_gpu_main_env()
    dfl_nn.initialize(dfl_nn.DeviceConfig.GPUIndexes([0]), "float32", "NCHW")
    plan = ReplicaPlan.from_device_config(dfl_nn.getCurrentDeviceConfig())
    assert plan.num_replicas == 1 and not plan.is_multi
    comp = _make_encoder_component(device=torch.device("cuda", 0))
    plan.add_component(comp)
    plan.initial_sync()
    assert plan.components[0].mirrors == []
    cp0 = plan.components[0].canonical_params[0]
    inner = [(torch.zeros_like(cp0), cp0)]
    assert plan.canonicalize_grads([inner]) is inner
    plan.dispose()


# ---------------------------------------------------------------------------
# N == 1 single-replica bypass (CPU production shape)
# ---------------------------------------------------------------------------

def test_n1_plan_no_mirrors_and_identity():
    plan = ReplicaPlan.from_device_config(dfl_nn.DeviceConfig([]))
    comp = _make_encoder_component()
    plan.add_component(comp)
    plan.initial_sync()  # no-op: nothing to sync
    ms = plan.components[0]
    assert ms.mirrors == []
    assert ms.mirror_params == {}
    # standalone factory agrees: the N=1 ComponentMirrorSet has no mirrors
    direct = build_replica_mirrors(_make_bn_component(), plan)
    assert direct.num_mirrors == 0
    plan.dispose()


def test_duplicate_component_registration_refused():
    # one mirror set per component: re-adding the same module would
    # silently shadow its mapping, so the plan refuses it
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    with pytest.raises(ReplicaPlanError, match="already"):
        plan.add_component(comp)
    plan.dispose()


def test_non_module_canonical_rejected():
    plan = _plan_2cpu()
    with pytest.raises(ReplicaPlanError, match="torch.nn.Module"):
        plan.add_component(object())
    plan.dispose()


# ---------------------------------------------------------------------------
# disposal: mirrors released, canonical untouched, plan locked
# ---------------------------------------------------------------------------

def test_dispose_leaves_canonical_untouched(tmp_path):
    # checkpoint scratch is pytest's per-test ``tmp_path``
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    plan.add_component(comp)
    plan.initial_sync()
    ms = plan.components[0]
    snap_params = [p.detach().clone() for p in ms.canonical_params]
    snap_bufs = [b.detach().clone() for b in ms.canonical_buffers]
    plan.dispose()
    # (a) the plan is locked: every operation refuses
    assert plan.is_disposed
    with pytest.raises(ReplicaPlanError, match="disposed"):
        plan.add_component(_make_bn_component())
    with pytest.raises(ReplicaPlanError, match="disposed"):
        plan.initial_sync()
    with pytest.raises(ReplicaPlanError, match="disposed"):
        plan.sync_from_canonical()
    with pytest.raises(ReplicaPlanError, match="disposed"):
        plan.canonicalize_grads([[]])
    # (b) the canonical module/parameters/buffers are untouched
    for p, sp in zip(comp.parameters(recurse=True), snap_params):
        assert torch.equal(p.detach(), sp)
        assert p.grad is None
    for b, sb in zip(comp.buffers(recurse=True), snap_bufs):
        assert torch.equal(b.detach(), sb)
    # (c) the canonical Saveable behavior is fully intact
    f = tmp_path / "encoder.npy"
    comp.save_weights(str(f), force_dtype=np.float32)
    with open(f, "rb") as fh:
        saved = pickle.load(fh)
    assert set(saved) == {name for name, _ in comp._iter_official_weights()}
    # (d) disposal is idempotent
    plan.dispose()


def test_disposed_plan_mirrors_released():
    comp = _make_encoder_component()
    plan = _plan_2cpu()
    ms = plan.add_component(comp)
    assert len(ms.mirrors) == 1
    plan.dispose()
    # the component mirror set released its mirror references
    assert ms.mirrors == []
    assert ms.mirror_params == {}
    assert ms.mirror_buffers == {}
    # the canonical side of the mapping is gone with the plan's
    # ownership bookkeeping; the canonical module itself survives
    assert sum(1 for _ in comp.parameters(recurse=True)) == \
        len(ms.canonical_params)
