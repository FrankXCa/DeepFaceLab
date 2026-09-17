"""ModelBase — official leras model container (Phase 5, torch).

Replaces the official TensorFlow source in place (the official code is
preserved verbatim in ``ModelBase_tf.py`` — same pattern as
``discriminators_tf.py``). ``nn.ModelBase`` is now importable under the
torch foundation (previously the whole class was gated behind
``hasattr(nn, 'tf')`` in ``core/leras/models/__init__.py``).

Official semantics preserved (identical public API):

- a named container that registers sub-components (``nn.LayerBase``
  layers and nested ``ModelBase`` models) assigned as attributes in
  ``on_build``; the official attribute-discovery build loop is kept
  (``xor_list`` over the container attributes; ``on_build`` may be a
  generator so dependent models initialize between build passes). The
  discovery is deterministic (insertion order — no object ids, no hash
  iteration).
- ``build()`` — the official two-phase build; the TF
  ``tf.variable_scope`` contexts are dropped: torch layers register
  their parameters directly in ``build_weights()``, and the official
  checkpoint scope names come from the layer ``.name`` (Phase 3A naming
  contract, ``core.leras.checkpoint.official_name``). The official
  one-shot variable creation (initializer applied at creation time) is
  reproduced by running each registered layer's ``init_weights()``
  inside the container build, so a built container is fully
  initialized (the official end state).
- ``get_weights()`` — concatenates the registered sub-components'
  weights in registration order (the official order — the checkpoint
  key order); auto-builds on first use (official).
- ``get_layer_by_name`` / ``get_layers`` (flattens nested containers
  down to ``nn.LayerBase``) / ``summary`` (the official text table) —
  unchanged.
- ``__call__`` — the official auto-build on first use (exactly once,
  the ``built`` flag), then DELEGATES to ``nn.Module.__call__``
  (``super().__call__``) instead of calling ``forward`` directly: the
  official TF ``__call__`` was a plain dispatch (TF has no hook
  machinery), but this container is a torch module and must behave
  like one — forward pre-hooks, forward hooks and full backward hooks
  registered on the container work on the first call (the build
  happens before the delegation) and on every later call through the
  same PyTorch Module call path.

Torch adaptations (documented, minimal):

- the container is a ``torch.nn.Module`` AND a ``Saveable`` — the same
  dual contract as ``LayerBase`` / ``OptimizerBase`` (the official
  ``ModelBase(nn.Saveable)`` is preserved: it still IS a Saveable, and
  like every torch leras component it is also a torch module). This is
  what makes whole sub-models first-class Phase 4 checkpoint
  components: ``save_weights`` / ``load_weights`` / the
  ``core.leras.convert`` engine enumerate the container tree through
  the registered sub-modules (official checkpoint keys
  ``<layer>/<param>:0`` under the container scope) and the per-layer
  layout hooks cascade through the tree exactly like on the archis —
  the official calling pattern ``sub_model.save_weights(filename)`` /
  ``sub_model.load_weights(filename)`` works unchanged (official SAEHD
  saves its encoder/inter/decoder containers this way).
- child modules live in torch's ``_modules`` registry rather than in
  ``vars(self)`` (torch 2.14 stores submodule assignments ONLY in
  ``_modules``; attribute access resolves through the module
  ``__getattr__``): the discovery set is therefore extended with the
  registered module names (insertion order — still deterministic), and
  torch-internal attributes (``_parameters`` / ``_buffers`` /
  ``_modules``) are excluded from the attribute scan so the official
  ``dict`` branch never re-registers the children. A registered
  submodule whose attribute name starts with an underscore is REJECTED
  with an explicit ``ValueError`` at build time (Phase 5 publish
  audit): torch treats underscore names specially (underscore buffers
  become non-persistent and drop out of the state-dict enumeration),
  which would make the container's persistent state inconsistent with
  the official all-state naming contract and the Phase 4 engine's
  enumeration; underscore names are reserved for torch module
  internals and the official DFL model code never names components
  that way — register components under plain names.
- ``build_for_run(shapes_list)`` no longer creates ``tf.placeholder``s
  and no longer pre-evaluates the graph: it records the input contract
  (the official ``run_placeholders`` attribute now holds the
  ``(dtype, shape)`` input specs) and forces the build, which is all
  the official pre-evaluation achieved (build + early error
  detection).
- ``run(inputs)`` no longer runs a ``tf_sess.run(feed_dict=...)``: it
  executes the built model's forward pass with ``torch.no_grad()`` —
  the inference/no-grad boundary (the official session run created no
  gradient bookkeeping either). ``run()`` is exclusively an
  inference/evaluation convenience (the official callers: preview,
  merge, AE inference); NO training code may use it — the official
  training step runs the gradient graph, and the torch equivalent is
  the normal model call (``model(x)`` -> ``loss.backward()`` -> the
  optimizer's ``get_update_op``), because ``run()`` returns detached
  CPU NumPy and creates no gradient path (the ``no_grad`` wrap mirrors
  the official session run exactly; it cannot suppress gradients of
  any training path, since no training path goes through ``run()``).
  Inputs may be torch tensors or NumPy
  arrays (converted to ``nn.device``/the declared spec dtype — the
  official ``feed_dict`` placement semantics); outputs are returned as
  NumPy (the official caller contract, e.g. preview code). Calling
  ``run`` before ``build_for_run`` raises the official
  ``"Model didn't build for run."`` exception; a wrong input count
  raises the official ``ValueError``.
- ``nn.close_session`` (no-op) stands in for the TF session teardown in
  the top-level ``models.ModelBase.finalize()`` — see Phase 5.
- ``summary`` uses the builtin ``sum`` over the per-layer parameter
  counts: the official ``np.sum(<generator>)`` line raises
  ``TypeError`` under NumPy >= 2 (the official environment pinned
  NumPy 1.19.3); the builtin is numerically identical for these int
  counts.

Official provenance: ``core/leras/models/ModelBase.py`` at upstream
baseline ``e4b7543ffa1d73b26fce1e31852727f658ba490c`` (see
``ModelBase_tf.py``).
"""

import types

import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn


def _run_input_to_torch(value, spec_dtype):
    """Convert one ``run()`` input to a torch tensor on ``nn.device``.

    Official ``feed_dict`` semantics: the fed value is placed on the
    graph device and cast to the placeholder dtype. Torch form: tensors
    are moved to ``nn.device`` (never silently left on another device),
    arrays are converted, and the declared spec dtype is applied when
    given.
    """
    torch = nn.torch
    device = nn.device if nn.device is not None else torch.device('cpu')

    if isinstance(value, torch.Tensor):
        if value.device != device:
            value = value.to(device)
    elif isinstance(value, np.ndarray):
        value = torch.from_numpy(np.ascontiguousarray(value)).to(device)
    else:
        raise TypeError(
            f"run() input must be a torch.Tensor or a numpy array, got {type(value)}"
        )

    if spec_dtype is not None and value.dtype != spec_dtype:
        value = value.to(spec_dtype)
    return value


def _run_output_to_numpy(result):
    """Official ``run`` returns NumPy (TF session outputs); keep the
    caller contract: tensors are detached to CPU NumPy, lists elementwise."""
    torch = nn.torch
    if isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy()
    if isinstance(result, (list, tuple)):
        return [_run_output_to_numpy(r) for r in result]
    return result


class ModelBase(torch.nn.Module, nn.Saveable):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__()  # torch.nn.Module init (MRO)
        nn.Saveable.__init__(self, name)  # explicit: Saveable is not in
        # the torch.nn.Module branch of the MRO (LayerBase pattern)
        self.layers = []
        self.layers_by_name = {}
        self.built = False
        self.args = args
        self.kwargs = kwargs
        # torch: holds the recorded (dtype, shape) input contract of
        # build_for_run (official: the list of tf.placeholders)
        self.run_placeholders = None

    def _build_sub(self, layer, name):
        if isinstance (layer, list):
            for i,sublayer in enumerate(layer):
                self._build_sub(sublayer, f"{name}_{i}")
        elif isinstance (layer, dict):
            for subname in layer.keys():
                sublayer = layer[subname]
                self._build_sub(sublayer, f"{name}_{subname}")
        elif isinstance (layer, nn.LayerBase) or \
                isinstance (layer, ModelBase):

            if layer.name is None:
                layer.name = name

            if isinstance (layer, nn.LayerBase):
                # torch: no tf.variable_scope context — the layer's `.name`
                # IS its official checkpoint scope (Phase 3A naming). The
                # official TF variable creation applied the initializer in
                # one shot; the torch two-phase lifecycle is completed
                # here so a built container is fully initialized (the
                # official end state), exactly like the archis:
                # build_weights() then init_weights().
                layer.build_weights()
                layer.init_weights()
            elif isinstance (layer, ModelBase):
                layer.build()

            self.layers.append (layer)
            self.layers_by_name[layer.name] = layer

    def xor_list(self, lst1, lst2):
        return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]

    def build(self):
        # torch: the official `with tf.variable_scope(self.name)` context is
        # dropped (see module docstring) — sub-components are scoped by name
        # alone, exactly like the torch archis (Phase 3F).

        current_vars = []
        generator = None
        while True:

            if generator is None:
                generator = self.on_build(*self.args, **self.kwargs)
                if not isinstance(generator, types.GeneratorType):
                    generator = None

            if generator is not None:
                try:
                    next(generator)
                except StopIteration:
                    generator = None

            # Phase 5 publish audit contract: a registered submodule
            # whose attribute name starts with an underscore is
            # rejected explicitly, on every discovery pass. Such an
            # assignment is a legitimate torch mechanism (torch 2.14
            # stores submodules only in ``_modules``, so it would be
            # picked up by the discovery below) but it is NOT part of
            # the component contract: torch treats underscore names
            # specially (underscore buffers become non-persistent and
            # drop out of the state-dict enumeration), which would make
            # the container's persistent state inconsistent with both
            # the official all-state naming contract and the Phase 4
            # engine's enumeration. The official DFL model code never
            # names components with an underscore prefix — fail loudly
            # instead of registering or omitting such state.
            for k in self._modules:
                if k.startswith("_"):
                    raise ValueError(
                        f"submodule attribute '{k}' must not start "
                        "with '_': underscore-prefixed names are "
                        "reserved for torch module internals and are "
                        "not part of the official component naming "
                        "contract — register components under plain "
                        "names (see the module docstring)"
                    )

            # torch adaptation (documented in the module docstring): child
            # modules live in the ``_modules`` registry, not in
            # ``vars(self)``, so the discovery set is the plain attributes
            # (torch-internal underscore names excluded, so the official
            # ``dict`` branch never re-registers the module registries)
            # plus the registered module names, in insertion order
            v = vars(self)
            items = [(k, val) for k, val in v.items() if not k.startswith("_")]
            plain = set(v.keys())
            for k in self._modules.keys():
                if k not in plain:
                    items.append((k, self._modules[k]))

            current_set = set(current_vars)
            new_vars = [name for name, _ in items if name not in current_set]

            for name in new_vars:
                value = v[name] if name in v else self._modules[name]
                self._build_sub(value, name)

            current_vars += new_vars

            if generator is None:
                break

        self.built = True

    #override
    def get_weights(self):
        if not self.built:
            self.build()

        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        return weights

    def get_layer_by_name(self, name):
        return self.layers_by_name.get(name, None)

    def get_layers(self):
        if not self.built:
            self.build()
        layers = []
        for layer in self.layers:
            if isinstance (layer, nn.LayerBase):
                layers.append(layer)
            else:
                layers += layer.get_layers()
        return layers

    #override
    def on_build(self, *args, **kwargs):
        """
        init model layers here

        return 'yield' if build is not finished
                    therefore dependency models will be initialized
        """
        pass

    #override
    def forward(self, *args, **kwargs):
        #flow layers/models/tensors here
        pass

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build()

        # torch (Phase 5 publish audit): the official TF ``__call__`` was
        # a plain dispatch (TF has no module hook machinery), but this
        # container IS a torch.nn.Module — it must route through the real
        # Module call path (``nn.Module.__call__`` -> ``_call_impl``) so
        # that forward pre-hooks, forward hooks and full backward hooks
        # registered on the container work on the FIRST call (lazy build
        # happens before the delegation, exactly once) and on every
        # later call. Calling ``self.forward`` directly here would
        # silently bypass all of that machinery.
        return super().__call__(*args, **kwargs)

    def build_for_run(self, shapes_list):
        if not isinstance(shapes_list, list):
            raise ValueError("shapes_list must be a list.")

        # torch: record the input contract instead of creating TF
        # placeholders; force the build (the official pre-evaluation of the
        # placeholder graph only served to build it and surface errors early)
        self.run_placeholders = list(shapes_list)

        if not self.built:
            self.build()

    def run (self, inputs):
        if self.run_placeholders is None:
            raise Exception ("Model didn't build for run.")

        if len(inputs) != len(self.run_placeholders):
            raise ValueError("len(inputs) != self.run_placeholders")

        torch_inputs = [
            _run_input_to_torch(inp, spec[0])
            for spec, inp in zip(self.run_placeholders, inputs)
        ]

        # inference boundary: the official tf_sess.run created no gradient
        # bookkeeping; torch.no_grad() is the native equivalent
        torch = nn.torch
        with torch.no_grad():
            result = self.forward(*torch_inputs)

        return _run_output_to_numpy(result)

    def summary(self):
        layers = self.get_layers()
        layers_names = []
        layers_params = []

        max_len_str = 0
        max_len_param_str = 0
        delim_str = "-"

        total_params = 0

        #Get layers names and str lenght for delim
        for l in layers:
            if len(str(l))>max_len_str:
                max_len_str = len(str(l))
            layers_names+=[str(l).capitalize()]

        #Get params for each layer
        # NumPy 2.x adaptation (documented in the module docstring): the
        # official `np.sum(<generator>)` line raises TypeError under
        # NumPy >= 2; the builtin sum is numerically identical here.
        layers_params = [ int(sum(np.prod(w.shape) for w in l.get_weights())) for l in layers ]
        total_params = sum(layers_params)

        #Get str lenght for delim
        for p in layers_params:
            if len(str(p))>max_len_param_str:
                max_len_param_str=len(str(p))

        #Set delim
        for i in range(max_len_str+max_len_param_str+3):
            delim_str += "-"

        output = "\n"+delim_str+"\n"

        #Format model name str
        model_name_str = "| "+self.name.capitalize()
        len_model_name_str = len(model_name_str)
        for i in range(len(delim_str)-len_model_name_str):
            model_name_str+= " " if i!=(len(delim_str)-len_model_name_str-2) else " |"

        output += model_name_str +"\n"
        output += delim_str +"\n"


        #Format layers table
        for i in range(len(layers_names)):
            output += delim_str +"\n"

            l_name = layers_names[i]
            l_param = str(layers_params[i])
            l_param_str = ""
            if len(l_name)<=max_len_str:
                for i in range(max_len_str - len(l_name)):
                    l_name+= " "

            if len(l_param)<=max_len_param_str:
                for i in range(max_len_param_str - len(l_param)):
                    l_param_str+= " "

            l_param_str += l_param


            output +="| "+l_name+"|"+l_param_str+"| \n"

        output += delim_str +"\n"

        #Format sum of params
        total_params_str = "| Total params count: "+str(total_params)
        len_total_params_str = len(total_params_str)
        for i in range(len(delim_str)-len_total_params_str):
            total_params_str+= " " if i!=(len(delim_str)-len_total_params_str-2) else " |"

        output += total_params_str +"\n"
        output += delim_str +"\n"

        io.log_info(output)

nn.ModelBase = ModelBase
