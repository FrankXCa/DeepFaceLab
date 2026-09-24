"""Test-only lifecycle harness for Phase 12 Commit 3 (mixed-precision
validation ACROSS replica devices).

NOT a production architecture: it exists to validate the replica
precision FOUNDATION lifecycle — all-device precision resolution, the
ONE global GradScaler, the per-replica SCALED backward flow THROUGH
THE EXISTING ``ModelBase._mp_backward`` HOOK (the one global scaler
scales every replica's per-sample loss vector; off/bf16 stays
unscaled), canonicalization / official aggregation / canonical grad
installation, the canonical ``unscale``/``step``/``update`` sequence
executed EXACTLY ONCE per attempt THROUGH THE EXISTING
``ModelBase._mp_unscale_opt`` / ``_mp_opt_step`` /
``_mp_scaler_update`` HOOKS (dependency-injected into the production
driver — no native scaler calls in core/leras), the Class A
(nonfinite forward/loss) vs Class B (scaled-gradient overflow) split,
the complete canonical + mirror grad cleanup, the production
16-attempt retry policy and the post-step mirror sync — through a
SMALL PRODUCTION-PATH framework: a minimal ``ModelBase`` subclass
that drives the production ``_mp_run_generator`` retry wrapper and
the production ``core.leras.multidevice`` foundation
(``resolve_precision_devices`` + ``run_replica_precision_step`` +
``clear_replica_grads`` + ``replica_param_lists`` + ``ReplicaPlan``).

No parallel fake implementation stands in for any production path:
the only test-local constructs are the deterministic sample
generator, the small net, the attempt hooks that INJECT failure
conditions the production paths then must handle, and the per-hook
call counters — wrappers that wrap the ORIGINAL bound ModelBase
hooks and delegate to them immediately (the hook semantics are never
reimplemented; the production paths under test are exactly the
ModelBase ``_mp_*`` methods themselves).

No model-specific training wiring is present: SAEHD multi-replica
wiring (Commit 4) and the AMP model (Commit 5) build on this
foundation; the physical 2-GPU results stay PENDING_ENVIRONMENTALLY /
NOT_VERIFIED — GPU tests map several LOGICAL replicas onto ONE
physical device (SIMULATED_MULTI_REPLICA, Level-B simulation).
"""

from pathlib import Path

import numpy as np
import torch

from core.leras import nn as dfl_nn
import core.leras.models  # noqa: F401  (binds nn.ModelBase)
from core.leras import mixed_precision
from core.leras import multidevice
from core.leras.multidevice import ReplicaPlan
from models import ModelBase
from models.ModelBase import SkippedGeneratorStep
from samplelib import SampleGeneratorBase


class HarnessNet(dfl_nn.ModelBase):
    """Two convs (3->4->2, 3x3 SAME, NHWC), zero-initializer weights —
    the training step overwrites them with a deterministic per-parameter
    constant, so the whole harness is zero-RNG and EXACT-comparable
    (a clean retry must bit-match a fresh attempt at the same scale)."""

    def on_build(self):
        self.conv1 = dfl_nn.Conv2D(
            3, 4, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros, name="conv1")
        self.conv2 = dfl_nn.Conv2D(
            4, 2, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros, name="conv2")

    def forward(self, x):
        return self.conv2(self.conv1(x))


class FakeGenerator(SampleGeneratorBase):
    """Deterministic fixed batch (no RNG, no dataset, no TF)."""

    def __init__(self, batch_size=4):
        super().__init__(debug=False, batch_size=batch_size)

    def is_initialized(self):
        return True

    def generate_next(self):
        b = self.batch_size
        a = (np.arange(b * 8 * 8 * 3) % 5) * 0.1
        return a.reshape(b, 8, 8, 3).astype(np.float32)


def make_harness_class(precision, n_replicas, scaler_factory=None):
    """A test-local ModelBase subclass wired for ``precision`` ('off' |
    'fp16' | 'bf16') on ``n_replicas`` logical replicas (replica 0 =
    the Phase 2 nn.device primary; several entries on one physical
    device = the SIMULATED_MULTI_REPLICA Level-B shape).

    The canonical unscale/step/update sequence of the production
    driver runs through the EXISTING ModelBase mixed-precision hooks
    (``_mp_unscale_opt`` / ``_mp_opt_step`` / ``_mp_scaler_update``,
    bound and injected by the attempt closure); the per-replica
    SCALED backward is the existing ``_mp_backward`` hook. The harness
    wraps those four bound hooks with per-hook call counters (the
    counters delegate immediately to the original bound hooks — no
    hook behavior is reimplemented) so the tests can assert the exact
    per-attempt hook-call bookkeeping.

    ``scaler_factory`` (test-only) replaces the plan's default native
    GradScaler factory with a ``device_type -> GradScaler`` callable
    (e.g. a half-scale native ``torch.amp.GradScaler`` for the
    clean-retry bit-exact comparison); the harness STILL constructs
    exactly ONE scaler for the run (the ONE-global-scaler contract)."""
    class HarnessModel(ModelBase):
        test_precision = precision
        test_num_replicas = n_replicas
        test_scaler_factory = scaler_factory
        # test-only attempt hooks (set on the INSTANCE by the tests;
        # they INJECT failure conditions — the production paths under
        # test handle them):
        overflow_replica = None       # Class B: this replica's post-backward
                                      # SCALED grads are corrupted to inf on
                                      # the FIRST attempt only (one-shot — a
                                      # transient overflow; the clean retry at
                                      # the backed-off scale runs uncorrupted,
                                      # the production contract under test)
        overflow_every_attempt = False  # persistent overflow: the LAST replica
                                        # overflows on EVERY attempt (the 16-
                                        # attempt bound + the no-retry-once pin)
        nonfinite_loss_replica = None   # Class A: this replica's per-sample
                                        # loss vector is nonfinite
        nonfinite_forward_replica = None  # Class A: this replica's forward
                                         # output is nonfinite
        nonfinite_grad_replica = None     # off/bf16 only: this replica's
                                          # post-backward grads corrupted to
                                          # inf (the PRE-unscale hard-error)

        def on_initialize_options(self):
            # direct assignment (deterministic — the official ask_*
            # prompts would be interactive)
            self.batch_size = self.options['batch_size'] = 4

        def on_initialize(self):
            torch = dfl_nn.torch
            self.net = HarnessNet(name="harness_net")
            # official pattern: get_weights() auto-builds the container
            # and yields the saveable weights the optimizer registers
            saveable_weights = self.net.get_weights()
            # deterministic, non-trivial first-run weights (zero RNG
            # anywhere) — exact-comparability of retries
            with torch.no_grad():
                for i, p in enumerate(self.net.parameters(), start=1):
                    p.copy_(torch.full(p.shape, i * 0.01,
                                       dtype=p.dtype, device=p.device))
            self.trainable_weights = list(saveable_weights)
            self.harness_opt = dfl_nn.AdaBelief(lr=0.01, name="harness_opt")
            self.harness_opt.initialize_variables(saveable_weights)

            # Phase 12 Commit 3: the replica plan for the test-specified
            # topology (replica 0 = the primary nn.device contract);
            # ALL selected devices are validated against the requested
            # mode — the returned plan is the ONE global PrecisionPlan
            # of the run, and the ONE global GradScaler is created from
            # it (never one scaler per replica / mirror / device)
            primary = dfl_nn.device
            if self.test_num_replicas == 1:
                self.plan = ReplicaPlan.from_torch_devices(primary)
            else:
                self.plan = ReplicaPlan.from_torch_devices(
                    primary, [primary] * self.test_num_replicas)
            self.component = self.plan.add_component(self.net)
            self._mp_plan = mixed_precision.resolve_precision_devices(
                self.test_precision, list(self.plan.replica_devices))
            # read the class attribute via the CLASS (a plain function
            # stored as a class attribute would otherwise bind as a
            # method on instance access and receive the model as its
            # first argument)
            factory = type(self).test_scaler_factory
            if factory is not None:
                self._mp_scaler = factory(self._mp_plan.device_type)
            else:
                self._mp_scaler = self._mp_plan.make_scaler()
            # hook instrumentation (test-only): per-hook call
            # counters that wrap the ORIGINAL bound ModelBase hooks
            # and delegate to them immediately — the hook semantics
            # are never reimplemented; the production paths under
            # test are exactly the ModelBase ``_mp_*`` methods
            # themselves (``_mp_backward`` per replica, ``_mp_
            # unscale_opt`` / ``_mp_opt_step`` / ``_mp_scaler_update``
            # exactly once per attempt, injected into the production
            # driver)
            self._hook_counts = {"backward": 0, "unscale": 0,
                                 "step": 0, "update": 0}
            _orig_backward = self._mp_backward
            _orig_unscale = self._mp_unscale_opt
            _orig_step = self._mp_opt_step
            _orig_update = self._mp_scaler_update

            def _counted_backward(loss_vec):
                self._hook_counts["backward"] += 1
                return _orig_backward(loss_vec)

            def _counted_unscale(optimizer, active_weights):
                self._hook_counts["unscale"] += 1
                return _orig_unscale(optimizer, active_weights)

            def _counted_step(optimizer, grads_vars):
                self._hook_counts["step"] += 1
                return _orig_step(optimizer, grads_vars)

            def _counted_update():
                self._hook_counts["update"] += 1
                return _orig_update()

            self._hook_backward = _counted_backward
            self._hook_unscale = _counted_unscale
            self._hook_step = _counted_step
            self._hook_update = _counted_update
            self.plan.initial_sync()
            # the per-replica parameter lists that OWN each replica's
            # gradients (replica 0 = canonical, replica r>0 = mirrors)
            self._replica_params = multidevice.replica_param_lists(
                self.plan, self.trainable_weights)
            self.stepped_history = []
            self._attempt_loss_vecs = None
            self.set_training_data_generators(
                [FakeGenerator(batch_size=self.batch_size)])

        def onTrainOneIter(self):
            # the model consumes its own samples (the official pattern);
            # the attempt runs under the PRODUCTION retry wrapper
            # (16 bounded attempts under fp16, 1 under off/bf16)
            sample = self.generate_next_samples()
            torch = dfl_nn.torch
            x = torch.from_numpy(np.ascontiguousarray(sample[0])).to(
                dfl_nn.device, dtype=torch.float32)
            # official batch adjustment: the global batch is reduced to
            # a multiple of the replica count; the sharded tensor is
            # (N, b, H, W, C) in replica order
            n = self.plan.num_replicas
            b = max(1, x.shape[0] // n)
            x = x[:n * b].view(n, b, *x.shape[1:])
            self._mp_run_generator(self._g_attempt, x)
            # the per-sample loss vectors concatenated in shard order
            # (the model-layer loss history entry; Commit 4/5 own the
            # model-specific loss semantics)
            concat = torch.cat(self._attempt_loss_vecs, dim=0)
            return [("loss", float(concat.mean().item()))]

        def _g_attempt(self, x_sharded):
            torch = dfl_nn.torch
            plan = self.plan
            precision = self._mp_plan
            scaler = self._mp_scaler
            # attempt start (the first attempt AND every retry): the
            # complete canonical + mirror grad cleanup
            multidevice.clear_replica_grads(plan)
            lists, loss_vecs = [], []
            for r in range(plan.num_replicas):
                dev = plan.replica_devices[r]
                module = (self.component.canonical if r == 0
                          else self.component.mirrors[r - 1])
                x_r = x_sharded[r].to(dev)
                with precision.autocast_context(dev.type):
                    out = module(x_r)
                    out32 = out.to(torch.float32)
                    # per-sample loss vector (the batch SUM is what
                    # backward(loss_vec, ones) accumulates — the
                    # official nn.gradients(loss, vars) semantics)
                    loss_vec = ((out32 - 0.5) ** 2).mean(dim=(1, 2, 3))
                    # test-only Class A injection (a nonfinite forward
                    # output / loss on this replica)
                    if self.nonfinite_forward_replica == r:
                        out32 = out32 + torch.full_like(out32, float("inf"))
                        loss_vec = torch.full_like(loss_vec, float("inf"))
                    if self.nonfinite_loss_replica == r:
                        loss_vec = loss_vec + torch.full_like(
                            loss_vec, float("inf"))
                # CLASS A: a nonfinite forward output or loss on ANY
                # replica clears the canonical + ALL mirror grads and
                # raises the hard FloatingPointError — never a skipped
                # step, never scaler-recorded, never retried
                if (not torch.isfinite(out32).all().item()
                        or not torch.isfinite(loss_vec).all().item()):
                    multidevice.clear_replica_grads(plan)
                    raise FloatingPointError(
                        f"harness: nonfinite forward/loss on replica {r}")
                # the per-replica SCALED backward THROUGH the model
                # layer's EXISTING ModelBase._mp_backward hook (the
                # ONE global scaler scales every replica's loss
                # vector; off/bf16 stays unscaled; the seed is
                # ones_like — batch-SUM gradients)
                self._hook_backward(loss_vec)
                # test-only injection: off/bf16 PRE-unscale nonfinite
                # replica grad (the driver must hard-error it)
                if self.nonfinite_grad_replica == r:
                    for p in self._replica_params[r]:
                        if p.grad is not None:
                            p.grad = torch.full_like(p.grad, float("inf"))
                # test-only Class B injection: a LATER replica's scaled
                # gradients overflow (inf) — the production GradScaler
                # overflow path must then own the whole outcome
                if (self.overflow_replica == r
                        or (self.overflow_every_attempt
                            and r == plan.num_replicas - 1)):
                    for p in self._replica_params[r]:
                        if p.grad is not None:
                            p.grad = torch.full_like(p.grad, float("inf"))
                    if self.overflow_replica == r:
                        # one-shot: the transient overflow fires on the
                        # FIRST attempt only — the clean retry (same
                        # samples, clean grads, backed-off scale) runs
                        # uncorrupted, exactly the production scenario
                        self.overflow_replica = None
                lists.append([(p.grad, p) for p in self._replica_params[r]])
                loss_vecs.append(loss_vec.detach())
            self._attempt_loss_vecs = loss_vecs
            # the canonical-side precision step (structural
            # validation, pre-unscale policy, canonicalization,
            # official aggregation, canonical install, the canonical
            # unscale/step/update — exactly once per attempt —
            # THROUGH THE INJECTED MODEL-LAYER HOOKS, the Class B
            # cleanup)
            stepped = multidevice.run_replica_precision_step(
                plan, precision, scaler, self.harness_opt, lists,
                active_weights=self.trainable_weights,
                unscale_hook=self._hook_unscale,
                step_hook=self._hook_step,
                update_hook=self._hook_update)
            self.stepped_history.append(stepped)
            if not stepped:
                raise SkippedGeneratorStep(
                    "harness: FP16 gradients overflowed")
            # the successful-step integration point: the mirrors are
            # re-synced from the canonical (the disposable runtime
            # state follows the checkpoint-owned canonical weights)
            plan.sync_from_canonical()

        def onGetPreview(self, sample, for_history=False):
            return []

    return HarnessModel


def make_harness(model_class, tmpdir, **device_kwargs):
    """Construct a harness model instance under ``tmpdir`` (the official
    lifecycle constructor; ``cpu_only=True`` / ``force_gpu_idxs=[0]``
    select the device without any interactive prompt). The harness's
    ``test_precision`` is also published through the constructor's
    ``precision`` channel so the model's own runtime channel states
    the requested mode (the harness resolves it across ALL replica
    devices in on_initialize — the production single-device
    resolution is never run for the harness models)."""
    return model_class(
        is_training=True,
        saved_models_path=Path(tmpdir),
        training_data_src_path=Path(tmpdir) / "src",
        training_data_dst_path=Path(tmpdir) / "dst",
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name="lifecycle_Harness",
        precision=model_class.test_precision,
        **device_kwargs,
    )


def snapshot_harness(model):
    """The full lifecycle state: canonical parameters (in order), every
    mirror replica's parameters, the optimizer state ([iters] + ms + vs),
    the scaler scale, the model iter and the per-hook call counters."""
    torch = dfl_nn.torch
    params = [p.detach().cpu().clone() for p in model.component.canonical_params]
    mirror_params = {
        r: [p.detach().cpu().clone()
            for p in model.component.mirror_params[r]]
        for r in range(1, model.plan.num_replicas)
    }
    states = [t.detach().cpu().clone()
              for t in model.harness_opt.get_weights()]
    snap = {
        "params": params,
        "mirror_params": mirror_params,
        "states": states,
        "iter": model.get_iter(),
        "iterations": int(model.harness_opt.iterations.item()),
        "hook_counts": dict(model._hook_counts),
    }
    scaler = model._mp_scaler
    if scaler is not None:
        snap["scale"] = float(scaler.get_scale())
    return snap
