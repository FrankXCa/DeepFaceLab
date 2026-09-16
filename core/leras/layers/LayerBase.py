"""LayerBase — torch module base with the official leras contract (Phase 3A).

Contract (official DFL behavior preserved):
- a layer is a ``torch.nn.Module`` AND a ``Saveable`` (the official
  ``LayerBase(Saveable)`` contract, so archis/models can treat layers
  as saveables exactly like official code does);
- construction is two-phase like the official graph mode:
    __init__(config...)        -> store config (no weights yet)
    archi registers the layer  -> module tree placement (naming)
    layer.build_weights()      -> create the parameters (register them
      with the official parameter names weight/bias/running_mean/
      running_var, device=nn.device, dtype=nn.floatx) and register the
      per-parameter initializers (``register_param_initializer``)
    archi/layer.init_weights() -> initialization lifecycle
  Parameters may be registered after the module is attached to the
  parent tree (torch allows late parameter registration), so the
  official archi calling pattern is preserved unchanged.
- ``forward()`` is the torch module forward (official ``forward`` /
  ``__call__`` contract).
- weight enumeration/serialization: inherited from ``Saveable``
  (deterministic ``named_parameters``/``named_buffers`` order, official
  checkpoint keys, strict load).

Device/dtype ownership: layers must create parameters through
``nn.device`` / ``nn.floatx`` (set by ``nn.initialize`` via the Phase 2
device abstraction); no ``torch.cuda.*`` call belongs in layer code.
"""

import torch

from core.leras import nn
from .Saveable import Saveable


class LayerBase(torch.nn.Module, Saveable):
    def __init__(self, name=None, **kwargs):
        super().__init__()  # torch.nn.Module init (MRO)
        Saveable.__init__(self, name)  # explicit: Saveable is not in
        # the torch.nn.Module branch of the MRO
        # (extra kwargs are accepted and ignored by the foundation;
        # concrete layers forward what they need)
        self._param_initializers = {}

    #override — create parameters (called by archis after registration)
    def build_weights(self):
        pass

    #override — torch module forward (official 'forward' contract)
    def forward(self, *args, **kwargs):
        pass

    def get_weights(self):
        # Official contract: the weights this layer owns, in creation
        # order. In torch: registered parameters (recursively), which
        # preserves registration order deterministically.
        return list(self.parameters(recurse=True))

    # --- initialization lifecycle (Phase 3A contract) ---

    def register_param_initializer(self, param_name, initializer):
        """Register the initializer for a parameter created in
        ``build_weights`` (keyed by the torch registered name, e.g.
        'weight'). Applied by ``init_weights``; parameters without a
        registered initializer keep their construction-time value."""
        self._param_initializers[param_name] = initializer

    def get_param_initializers(self):
        return self._param_initializers

    # get_weights_np / set_weights / save_weights / load_weights /
    # init_weights / convert_weight_layout are inherited from Saveable.

    def __str__(self):
        r = f"{self.__class__.__name__}"
        if self.name is not None:
            r += f" : {self.name}"
        return r


nn.LayerBase = LayerBase
