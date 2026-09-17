"""OptimizerBase — official DFL optimizer foundation (Phase 3E2, torch).

Contract preserved from the official TF implementation (preserved
verbatim in ``optimizers_tf.py``; this package is now the torch
foundation and never imports TensorFlow):

- an optimizer is a ``torch.nn.Module`` AND a ``Saveable`` (official
  ``OptimizerBase(Saveable)`` contract, so models can treat
  optimizers as saveables exactly like official code does);
- construction requires a scope ``name`` (official raises
  ``ValueError`` when it is missing);
- ``initialize_variables(trainable_weights, vars_on_cpu=True,
  lr_dropout_on_cpu=False)`` creates the per-parameter optimizer
  state (zero-initialized, same shape/dtype as each parameter);
- ``get_update_op(grads_vars)`` returns a zero-argument callable
  that performs ONE official update step using the given
  ``(gradient, weight)`` pairs: iteration counter increment, the
  lr_cos schedule, the optional global-norm gradient clip, the
  per-parameter state/weight update, and the optional lr_dropout
  mask. Unlike the TF graph (which re-ran the gradient graph on
  every session run), the caller must call ``get_update_op`` again
  with fresh gradients on every iteration;
- ``get_weights()`` returns ``[iterations] + state tensors`` (the
  official order, subclass-defined: all ms before all vs for
  AdaBelief; all accs for RMSprop) — the layout the Phase 4
  converter maps to the official ``iters`` / ``ms_*`` / ``vs_*`` /
  ``acc_*`` checkpoint names 1:1 by name, shape and dtype.

State naming (checkpoint compatibility, Phase 4): the official TF
variables live under the optimizer scope: ``<name>/iters:0``,
``<name>/ms_<varname>:0`` (AdaBelief), ``<name>/vs_<varname>:0``,
``<name>/acc_<varname>:0`` (RMSprop) where ``<varname>`` is the
trainable variable's official name with ``:`` replaced by ``_``
(e.g. ``ms_encoder/conv1/weights_0:0``). This implementation keeps
the exact official sub-names (slashes included) in
``_iter_official_weights`` so Saveable checkpoints and the Phase 4
converter map official optimizer state directly; the torch buffer
ATTRIBUTE names are sanitized (``/`` and ``:`` removed) — that never
touches the checkpoint keys. State lookup keys come from the
parameter's official name when the model code assigns one — via
``.name`` (official TF variable naming) or the torch-era attribute
``_dfl_name`` (``Tensor.name`` is a reserved, read-only property in
torch, so the torch model code assigns ``param._dfl_name =
'encoder/conv1/weights:0'``) — otherwise a stable positional
``param_<index>`` key. Object ids are never used (state ordering
must not depend on them).

State LAYOUT (checkpoint compatibility, Phase 4): the official
optimizer state variables have the SAME layout as the trained
variables they track (official NHWC conv kernels on disk, whatever
the runtime layout), while the torch state buffers have the
parameter's torch layout. So an optimizer's ``load_weights`` /
``save_weights`` (inherited ``Saveable`` contract) must convert a
state tensor through the layout hook of the OWNING LAYER of the
parameter it tracks — the model code attaches that binding to every
optimized parameter exactly like the name binding:
``param._dfl_owner_layer = <the leaf layer whose registered
parameters include it>``. Without the binding (or when the owning
layer declares no layout difference) the value passes through
unchanged — the rule applies to the ``iters`` counter and to plain
layers (Dense/bias) exactly as to the official identity case. The
Phase 4 converter (``convert_optimizer_state_*``) applies the same
rule. Evidence (real official checkpoint): official-layout (NHWC)
state tensors read without this conversion fail the strict shape
check against the torch-layout state buffers (62 NHWC/NCHW
mismatches on a real 320-liae SAEHD ``src_dst_opt.npy``).

Documented semantics (official behavior preserved):
- **epsilon** = the official ``np.finfo(g.dtype).resolution`` —
  the DECIMAL resolution ``10 ** ceil(log10(machine eps))``, which
  is NOT the machine epsilon itself: f32 -> 1e-06, f16 -> 1e-03,
  f64 -> 1e-15. Verified under the official pinned NumPy 1.19.3
  (and identical under NumPy 1.26.x / 2.5.x — the property did not
  change across these versions). ``torch.finfo(dtype).eps``
  (f32: 1.19e-07) is NOT the official value and is never used here;
  a test pins the official value and discriminates it from the
  machine epsilon.
- **no bias correction** in either optimizer (the official has
  none — a third-party AdaBelief's bias correction is NOT added);
- **lr_cos** uses the official literal ``2*3.1415926535/lr_cos``
  and the POST-increment iteration count: the official graph queued
  ``assign_add(iterations, 1)`` ahead of the per-parameter updates
  and the cosine term read the counter behind that in the TF
  executor queue (EXTERNAL_A and USER_LEGACY make the same choice);
- **gradient clipping** = the official global norm:
  ``norm = sqrt(sum_g sum(g^2))`` computed in float32 over ALL
  gradients, each gradient scaled by ``clipnorm/norm`` when
  ``norm >= clipnorm`` (per-parameter scaling by the GLOBAL norm,
  never per-parameter norms; no epsilon added — the USER_LEGACY
  ``+eps`` variant is rejected);
- **lr_dropout** masks are ONE FRESH mask per parameter per step
  (``nn.random_binomial(v.shape, p, dtype)`` — the official TF
  graph re-evaluated its random op on every run of the update op;
  the USER_LEGACY torch bridge froze the mask at
  ``initialize_variables`` — that bug is NOT reproduced);
- **placement**: the official kept ms/vs/acc on the CPU by default
  (``vars_on_cpu``) and optionally the masks too (``lr_dropout_on_
  cpu``); under the Phase 2 device model every tensor lives on
  ``nn.device`` and state co-locates with its parameter. The
  official kwargs are kept for signature parity and are otherwise
  ignored — numerics are unchanged by this placement difference.
"""

import torch

from core.leras import nn
from core.leras.layers.Saveable import Saveable

# The official denominator epsilon: ``np.finfo(dtype).resolution`` —
# the DECIMAL resolution ``10 ** ceil(log10(machine eps))``, NOT the
# machine epsilon itself. Values verified under the official pinned
# NumPy 1.19.3 (identical under NumPy 1.26.x / 2.5.x — the property
# did not change across these versions): f32 -> 1e-06, f16 -> 1e-03,
# f64 -> 1e-15. ``torch.finfo(dtype).eps`` (e.g. 1.19e-07 for f32)
# is NOT the official value.
_OFFICIAL_FINFO_RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1e-6,
    torch.float64: 1e-15,
}


def official_finfo_resolution(dtype):
    """The official ``np.finfo(dtype).resolution`` value for a torch
    floating dtype (see the module docstring for the verification
    record). Raises ``ValueError`` for dtypes without an official
    value."""
    try:
        return _OFFICIAL_FINFO_RESOLUTION[dtype]
    except KeyError:
        raise ValueError(
            f"no official finfo resolution for dtype {dtype}")


class OptimizerBase(torch.nn.Module, Saveable):
    def __init__(self, name=None):
        super().__init__()
        Saveable.__init__(self, name)  # explicit: Saveable is not in
        # the torch.nn.Module branch of the MRO (LayerBase pattern)
        self.clipnorm = 0.0
        self.lr_cos = 0
        self.lr_dropout = 1.0

        # official: tf.Variable(0, dtype=tf.int64, name='iters') under
        # the optimizer scope -> checkpoint sub-name 'iters:0'
        self.iterations = torch.zeros((), dtype=torch.long)
        self.register_buffer("iters", self.iterations)

        # state bookkeeping (filled by initialize_variables)
        self._weights = []       # initialized parameters, order kept
        self._weight_keys = {}   # id(weight) -> stable state key
        # official sub-name for every state tensor, aligned with
        # get_weights() (after the leading 'iters:0')
        self._state_official_names = []
        # id(state buffer) -> the tracked parameter it was created for
        # (filled by _zero_state) — the state LAYOUT delegation key
        # (module docstring, 'State LAYOUT')
        self._state_owner = {}

    # --- official initialize_variables contract -------------------------

    def _weight_key(self, weight, index):
        """Stable state key for a trainable parameter: its official
        name when assigned by the model code (``.name`` — official
        TF variable names like 'encoder/conv1/weights:0' — or the
        torch-era ``_dfl_name`` attribute), else a positional key.
        Object ids are never used (state ordering must not depend
        on them)."""
        key = getattr(weight, "name", None) or getattr(weight, "_dfl_name", None)
        if key is None:
            key = f"param_{index}"
        return key

    @staticmethod
    def _state_sub_name(prefix, weight_key):
        # official: f'{prefix}_{v.name}'.replace(':','_') + ':0'
        return f"{prefix}_{weight_key}".replace(":", "_") + ":0"

    def _zero_state(self, official_sub_name, param):
        """Create a zero state buffer co-located with its parameter
        (official: same shape/dtype, constant 0.0, non-trainable)."""
        attr = official_sub_name.replace(":", "").replace("/", "_")
        buf = torch.zeros(param.shape, dtype=param.dtype,
                          device=param.device)
        self.register_buffer(attr, buf)
        self._state_owner[id(buf)] = param  # state LAYOUT delegation
        return buf

    def initialize_variables(self, trainable_weights, vars_on_cpu=True,
                             lr_dropout_on_cpu=False):
        """Official contract: register this optimizer's per-parameter
        state (zero-initialized) for the trainable weights. The
        official ``vars_on_cpu`` / ``lr_dropout_on_cpu`` placement
        kwargs are kept for signature parity; under the Phase 2
        device model the state co-locates with its parameter on
        ``nn.device`` (documented placement deviation, numerics
        unchanged). lr_dropout masks need no pre-creation: they are
        resampled per step (official TF graph behavior)."""
        trainable_weights = list(trainable_weights)
        self._weights = trainable_weights
        self._weight_keys = {
            id(v): self._weight_key(v, i) for i, v in enumerate(trainable_weights)
        }
        self._state_owner = {}
        if nn.device is not None and self.iterations.device != nn.device:
            self.iterations.data = self.iterations.data.to(nn.device)
        self._build_state(trainable_weights)

    #override — subclasses create their state buffers (official order)
    def _build_state(self, weights):
        pass

    def _key_of(self, weight):
        key = self._weight_keys.get(id(weight))
        if key is None:
            raise ValueError(
                f"optimizer state for {getattr(weight, 'name', weight)} "
                "was not registered by initialize_variables"
            )
        return key

    # --- official state LAYOUT delegation (module docstring) --------------

    def convert_weight_layout(self, value, param):
        """Official -> torch layout for a checkpoint value of THIS
        saveable. The optimizer's own tensors are the zero-state
        buffers; a state buffer is converted through the layout hook
        of the OWNING LAYER of the parameter it tracks (the official
        state tensor has the tracked variable's layout). Untracked
        values (the ``iters`` counter) and tracked parameters without
        a ``_dfl_owner_layer`` binding pass through unchanged —
        identical to the base identity behavior."""
        tracked = self._state_owner.get(id(param))
        if tracked is None:
            return value
        layer = getattr(tracked, "_dfl_owner_layer", None)
        if layer is None or not isinstance(layer, Saveable):
            return value
        return layer.convert_weight_layout(value, tracked)

    def convert_weight_to_official(self, value, param):
        """Torch -> official layout, inverse of ``convert_weight_layout``
        (applied by ``save_weights`` so the file on disk is
        official-layout, exactly like the component weights)."""
        tracked = self._state_owner.get(id(param))
        if tracked is None:
            return value
        layer = getattr(tracked, "_dfl_owner_layer", None)
        if layer is None or not isinstance(layer, Saveable):
            return value
        return layer.convert_weight_to_official(value, tracked)

    # --- official get_update_op contract ---------------------------------

    def get_update_op(self, grads_vars):
        """Returns a zero-argument callable performing ONE official
        update step with the given ``(gradient, weight)`` pairs
        (official ``grads_vars`` contract). Each call: increments the
        iteration counter (the lr_cos schedule reads the
        POST-increment value — see the module docstring), resamples
        the lr_dropout masks when enabled, applies the global-norm
        clip, then the per-parameter state/weight updates. The
        caller must call ``get_update_op`` again with fresh
        gradients on every iteration (torch does not re-run a
        gradient graph inside the step, unlike the TF session)."""
        def run_update():
            self.step(grads_vars)
        return run_update

    def step(self, grads_vars):
        if not grads_vars:
            return
        with torch.no_grad():
            # official: norm = sqrt(sum_g reduce_sum(square(cast(g, f32))))
            # global norm in float32 over ALL gradients
            if self.clipnorm > 0.0:
                g0 = grads_vars[0][0]
                norm_sq = torch.zeros((), dtype=torch.float32, device=g0.device)
                for g, v in grads_vars:
                    gf = g.to(torch.float32)
                    norm_sq = norm_sq + torch.sum(gf * gf)
                norm = torch.sqrt(norm_sq)

            # official: assign_add(iterations, 1) queued AHEAD of the
            # per-parameter updates -> post-increment value below
            self.iterations.add_(1)
            iters = self.iterations.clone()

            for g, v in grads_vars:
                if g is None:
                    raise ValueError(
                        f"optimizer got a None gradient for "
                        f"{getattr(v, 'name', v)}: the official update op "
                        "expects a gradient for every trainable weight"
                    )
                if self.clipnorm > 0.0:
                    # official tf_clip_norm: g * (c/n) when n >= c
                    n = norm.to(g.dtype)
                    if n >= self.clipnorm:
                        g = g * (self.clipnorm / n)
                # official: lr = constant(self.lr, g.dtype), then the
                # lr_cos factor, computed in the gradient's dtype
                lr = torch.full((), float(self.lr), dtype=g.dtype,
                                device=v.device)
                if self.lr_cos != 0:
                    lr = lr * (torch.cos(
                        iters.to(dtype=g.dtype)
                        * (2 * 3.1415926535 / float(self.lr_cos))) + 1.0) / 2.0
                self._update(g, v, lr)

    #override — subclasses: per-parameter state + weight update
    def _update(self, g, v, lr):
        raise NotImplementedError

    def _apply_lr_dropout_mask(self, v_diff, v):
        """Official: v_diff *= lr_rnd, with lr_rnd a FRESH
        random_binomial mask per parameter per step (the official TF
        graph re-evaluated its random op on every run of the update
        op)."""
        if self.lr_dropout == 1.0:
            return v_diff
        mask = nn.random_binomial(v.shape, p=self.lr_dropout,
                                  dtype=v.dtype, device=v.device)
        return v_diff * mask

    # --- official get_weights / checkpoint contract ----------------------

    def get_weights(self):
        # official: [iterations] + state tensors (subclass order)
        return [self.iterations] + self._states()

    #override — subclass state tensors in official get_weights() order
    def _states(self):
        return []

    def _iter_official_weights(self):
        # official checkpoint sub-names relative to this optimizer's
        # scope (Saveable saves them under the optimizer scope name)
        names = ["iters:0"] + self._state_official_names
        if len(names) != len(self.get_weights()):
            raise RuntimeError(
                "optimizer state name/tensor mismatch — this is a bug"
            )
        return list(zip(names, self.get_weights()))

    def __str__(self):
        r = f"{self.__class__.__name__}"
        if self.name is not None:
            r += f" : {self.name}"
        return r


nn.OptimizerBase = OptimizerBase
