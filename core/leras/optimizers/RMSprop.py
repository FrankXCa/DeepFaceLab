"""RMSprop — official DFL optimizer, torch (Phase 3E2).

The official update (preserved verbatim; the TF reference is in
``optimizers_tf.py``) is, per parameter per step:

    new_a = rho * acc + (1 - rho) * g^2
    v_diff = - lr * g / (sqrt(new_a) + finfo(g.dtype).resolution)
    v += v_diff                                          # (+ lr_dropout mask)

with ``lr`` optionally scaled by the official lr_cos factor
``(cos(iters * 2*3.1415926535/lr_cos) + 1)/2`` (post-increment
iteration count) and the gradient pre-clipped by the optimizer's
global norm when ``clipnorm > 0``.

Official semantics preserved (see the OptimizerBase docstring for
the shared mechanics): the official denominator epsilon IS the
dtype's finfo RESOLUTION - the decimal resolution (f32: 1e-06,
NOT the machine epsilon ``torch.finfo(...).eps`` = 1.19e-07),
there is NO momentum and NO centering
(standard ``torch.optim.RMSprop`` has neither the official state
naming — ``acc_*``, not ``vs_*`` — nor the official lr_cos /
lr_dropout / global-norm-clip machinery, so it is NOT substituted),
the state layout is ``iters`` + all ``acc_*``, and the lr_dropout
mask is resampled every step (the USER_LEGACY frozen mask is
rejected).
"""

import torch

from core.leras import nn
from .OptimizerBase import OptimizerBase, official_finfo_resolution


class RMSprop(OptimizerBase):
    def __init__(self, lr=0.001, rho=0.9, lr_dropout=1.0, lr_cos=0,
                 clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.lr = lr
        self.rho = rho
        self.clipnorm = clipnorm

        # official: tf variable 'acc_<varname>' under the optimizer
        # scope (zero-init, same shape/dtype as the parameter)
        self.accumulators_dict = {}

    def _build_state(self, weights):
        for i, v in enumerate(weights):
            key = self._weight_key(v, i)
            self.accumulators_dict[key] = self._zero_state(
                self._state_sub_name('acc', key), v)
        self._state_official_names = [
            self._state_sub_name('acc', k) for k in self.accumulators_dict
        ]

    def _states(self):
        return list(self.accumulators_dict.values())

    def _update(self, g, v, lr):
        key = self._key_of(v)
        acc = self.accumulators_dict[key]

        new_a = self.rho * acc + (1. - self.rho) * g.pow(2)

        # official: np.finfo(g.dtype).resolution - the DECIMAL
        # resolution (f32: 1e-06), NOT the machine epsilon
        v_diff = -lr * g / (torch.sqrt(new_a)
                            + official_finfo_resolution(g.dtype))
        v_diff = self._apply_lr_dropout_mask(v_diff, v)
        v.add_(v_diff)

        acc.copy_(new_a)


nn.RMSprop = RMSprop
