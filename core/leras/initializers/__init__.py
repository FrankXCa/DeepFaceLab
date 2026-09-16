"""Initializers — torch initialization lifecycle (Phase 3A foundation).

``nn.initializers`` mirrors the official surface used by layers:
official layers call TF built-ins (``tf.initializers.zeros/ones/
random_normal/glorot_uniform/glorot_normal``) plus the DFL-specific
``nn.initializers.ca`` (cross-GPU deterministic batch initialization,
generated through a subprocess).

Lifecycle (official two-phase contract, torch form):

    layer.build_weights()     -> create the parameters
    layer.init_weights()      -> nn.init_weights(layer): for every
      DIRECT parameter/buffer of the layer that has an initializer
      registered on the layer (``register_param_initializer(name, fn)``
      in build_weights), the initializer replaces its value
    layers WITHOUT a registered initializer for a parameter keep the
      value given at construction — the foundation never fills weights
      silently
    modules composing sub-layers (archis, Phase 3B) override
      ``init_weights()`` to cascade to their children, exactly like the
      official archis

``ca`` is a placeholder in Phase 3A: its subprocess batch generation is
rebuilt with the ops/initialization lifecycle work (Phase 3B) before
any layer defaulting to CA (official Conv2D) is restored. Using it
before then fails loudly, never silently.
"""

import torch

from core.leras import nn


def _resolve(dtype, device):
    dtype = dtype if dtype is not None else (nn.floatx if nn.floatx is not None else torch.float32)
    device = device if device is not None else (nn.device if nn.device is not None else torch.device('cpu'))
    return dtype, device


class initializers():
    @staticmethod
    def zeros(shape, dtype=None, device=None):
        dtype, device = _resolve(dtype, device)
        return torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=None, device=None):
        dtype, device = _resolve(dtype, device)
        return torch.ones(shape, device=device, dtype=dtype)

    @staticmethod
    def random_normal(shape, mean=0.0, std=1.0, dtype=None, device=None):
        dtype, device = _resolve(dtype, device)
        return torch.nn.init.normal_(
            torch.empty(shape, device=device, dtype=dtype), mean, std
        )

    @staticmethod
    def glorot_uniform(shape, dtype=None, device=None):
        # Xavier uniform, matching TF's glorot_uniform (fan_avg gain 1)
        dtype, device = _resolve(dtype, device)
        return torch.nn.init.xavier_uniform_(
            torch.empty(shape, device=device, dtype=dtype), gain=1.0
        )

    @staticmethod
    def glorot_normal(shape, dtype=None, device=None):
        # Xavier normal, matching TF's glorot_normal (fan_avg gain 1)
        dtype, device = _resolve(dtype, device)
        return torch.nn.init.xavier_normal_(
            torch.empty(shape, device=device, dtype=dtype), gain=1.0
        )

    @staticmethod
    def ca(shape, dtype=None, device=None):
        # Official DFL CA (cross-GPU deterministic) initializer: in the
        # official flow the variables hold a zero placeholder and the
        # real values are filled by a subprocess batch generation
        # (CAInitializerSubprocessor) during init_weights.
        raise NotImplementedError(
            "nn.initializers.ca batch generation is rebuilt in Phase 3B "
            "(ops/initialization lifecycle); a layer must not fall back "
            "silently — remove the CA default or wait for Phase 3B"
        )


nn.initializers = initializers
