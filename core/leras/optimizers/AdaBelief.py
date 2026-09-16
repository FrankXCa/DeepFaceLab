"""AdaBelief — official DFL optimizer, torch (Phase 3E2).

The official update (preserved verbatim; the TF reference is in
``optimizers_tf.py``) is, per parameter per step:

    m_t = beta_1 * ms + (1 - beta_1) * g
    v_t = beta_2 * vs + (1 - beta_2) * (g - m_t)^2     # belief
    v_diff = - lr * m_t / (sqrt(v_t) + finfo(g.dtype).resolution)
    v += v_diff                                          # (+ lr_dropout mask)

with ``lr`` optionally scaled by the official lr_cos factor
``(cos(iters * 2*3.1415926535/lr_cos) + 1)/2`` (post-increment
iteration count) and the gradient pre-clipped by the optimizer's
global norm when ``clipnorm > 0``.

Official semantics preserved (see the OptimizerBase docstring for
the shared mechanics): the official denominator epsilon IS the
dtype's finfo RESOLUTION - the decimal resolution (f32: 1e-06,
NOT the machine epsilon ``torch.finfo(...).eps`` = 1.19e-07),
there is NO bias correction and NO weight decay, the state layout
is ``iters`` + all ``ms_*`` + all
``vs_*`` (the EXTERNAL_A plateau-scheduler/``lr_cur`` extension is
NOT part of the official optimizer and is not adopted), and the
lr_dropout mask is resampled every step (the USER_LEGACY frozen
mask is rejected).
"""

import torch

from core.leras import nn
from .OptimizerBase import OptimizerBase, official_finfo_resolution


class AdaBelief(OptimizerBase):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, lr_dropout=1.0,
                 lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm

        # official: tf variables 'ms_<varname>' / 'vs_<varname>' under
        # the optimizer scope (zero-init, same shape/dtype as the
        # parameter); keys keep initialize_variables order
        self.ms_dict = {}
        self.vs_dict = {}

    def _build_state(self, weights):
        # official order: ALL ms states first, then ALL vs states
        for i, v in enumerate(weights):
            key = self._weight_key(v, i)
            self.ms_dict[key] = self._zero_state(
                self._state_sub_name('ms', key), v)
        for i, v in enumerate(weights):
            key = self._weight_key(v, i)
            self.vs_dict[key] = self._zero_state(
                self._state_sub_name('vs', key), v)
        self._state_official_names = (
            [self._state_sub_name('ms', k) for k in self.ms_dict]
            + [self._state_sub_name('vs', k) for k in self.vs_dict]
        )

    def _states(self):
        return list(self.ms_dict.values()) + list(self.vs_dict.values())

    def _update(self, g, v, lr):
        key = self._key_of(v)
        ms = self.ms_dict[key]
        vs = self.vs_dict[key]

        m_t = self.beta_1 * ms + (1.0 - self.beta_1) * g
        # official: the belief residual uses the NEW first moment m_t
        v_t = self.beta_2 * vs + (1.0 - self.beta_2) * (g - m_t).pow(2)

        # official: np.finfo(g.dtype).resolution - the DECIMAL
        # resolution (f32: 1e-06), NOT the machine epsilon
        v_diff = -lr * m_t / (torch.sqrt(v_t)
                              + official_finfo_resolution(g.dtype))
        v_diff = self._apply_lr_dropout_mask(v_diff, v)
        v.add_(v_diff)

        ms.copy_(m_t)
        vs.copy_(v_t)


nn.AdaBelief = AdaBelief
