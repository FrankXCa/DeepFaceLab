# DEAD TensorFlow code - preserved for reference only (Phase 3E2).
#
# This file is the official DeepFaceLab TF optimizer reference: the
# three official modules core/leras/optimizers/{OptimizerBase,
# RMSprop, AdaBelief}.py moved here verbatim (one section per
# original file) so the torch foundation can own the package
# namespace. It is NOT imported by any torch code path. It is also
# not importable under the Phase 3A torch nn (it requires nn.tf
# / nn.tf_sess, which no longer exist), so it only ever runs in
# the official TF runtime.
#
# The torch versions (Phase 3E2) of OptimizerBase / AdaBelief /
# RMSprop are the real modules now: OptimizerBase.py, RMSprop.py,
# AdaBelief.py in this package. The official semantics they preserve
# (state naming ms_*/vs_*/acc_*/iters, lr_cos post-increment
# scheduling, global-norm clipping, per-step lr_dropout mask
# resampling, finfo-resolution epsilon) are documented in the torch
# module docstrings and in the local docs/COMPATIBILITY.md record.

# --- official OptimizerBase.py (verbatim) ---
import copy
from core.leras import nn
tf = nn.tf

class OptimizerBase(nn.Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)

    def tf_clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
        # Arguments
            g: Tensor, the gradient tensor
            c: float >= 0. Gradients will be clipped
                when their L2 norm exceeds this value.
            n: Tensor, actual norm of `g`.
        # Returns
            Tensor, the gradient clipped if required.
        """
        if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
            return g

        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

        return g
nn.OptimizerBase = OptimizerBase



# --- official RMSprop.py (verbatim) ---
import numpy as np
from tensorflow.python.ops import control_flow_ops, state_ops
from core.leras import nn
tf = nn.tf

class RMSprop(nn.OptimizerBase):
    def __init__(self, lr=0.001, rho=0.9, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.lr = lr
        self.rho = rho
        self.clipnorm = clipnorm

        with tf.device('/CPU:0') :
            with tf.variable_scope(self.name):
                
                self.iterations = tf.Variable(0, dtype=tf.int64, name='iters')

        self.accumulators_dict = {}
        self.lr_rnds_dict = {}

    def get_weights(self):
        return [self.iterations] + list(self.accumulators_dict.values())

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        # Initialize here all trainable variables used in training
        e = tf.device('/CPU:0') if vars_on_cpu else None
        if e: e.__enter__()
        with tf.variable_scope(self.name):
            accumulators = { v.name : tf.get_variable ( f'acc_{v.name}'.replace(':','_'), v.shape, dtype=v.dtype, initializer=tf.initializers.constant(0.0), trainable=False) for v in trainable_weights }
            self.accumulators_dict.update ( accumulators)

            if self.lr_dropout != 1.0:
                e = tf.device('/CPU:0') if lr_dropout_on_cpu else None
                if e: e.__enter__()                    
                lr_rnds = [ nn.random_binomial( v.shape, p=self.lr_dropout, dtype=v.dtype) for v in trainable_weights ]
                if e: e.__exit__(None, None, None)                
                self.lr_rnds_dict.update ( { v.name : rnd for v,rnd in zip(trainable_weights,lr_rnds) } )
        if e: e.__exit__(None, None, None)

    def get_update_op(self, grads_vars):
        updates = []

        if self.clipnorm > 0.0:
            norm = tf.sqrt( sum([tf.reduce_sum(tf.square(tf.cast(g, tf.float32))) for g,v in grads_vars]))
        updates += [ state_ops.assign_add( self.iterations, 1) ]
        for i, (g,v) in enumerate(grads_vars):
            if self.clipnorm > 0.0:
                g = self.tf_clip_norm(g, self.clipnorm, tf.cast(norm, g.dtype) )

            a = self.accumulators_dict[ v.name ]

            new_a = self.rho * a + (1. - self.rho) * tf.square(g)

            lr = tf.constant(self.lr, g.dtype)
            if self.lr_cos != 0:
                lr *= (tf.cos(  tf.cast(self.iterations, g.dtype) * (2*3.1415926535/ float(self.lr_cos) )  ) + 1.0) / 2.0

            v_diff = - lr * g / (tf.sqrt(new_a) + np.finfo( g.dtype.as_numpy_dtype ).resolution  )
            if self.lr_dropout != 1.0:
                lr_rnd = self.lr_rnds_dict[v.name]
                v_diff *= lr_rnd
            new_v = v + v_diff

            updates.append (state_ops.assign(a, new_a))
            updates.append (state_ops.assign(v, new_v))

        return control_flow_ops.group ( *updates, name=self.name+'_updates')
nn.RMSprop = RMSprop


# --- official AdaBelief.py (verbatim) ---
import numpy as np
from core.leras import nn
from tensorflow.python.ops import control_flow_ops, state_ops

tf = nn.tf

class AdaBelief(nn.OptimizerBase):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm

        with tf.device('/CPU:0') :
            with tf.variable_scope(self.name):
                self.iterations = tf.Variable(0, dtype=tf.int64, name='iters')

        self.ms_dict = {}
        self.vs_dict = {}
        self.lr_rnds_dict = {}

    def get_weights(self):
        return [self.iterations] + list(self.ms_dict.values()) + list(self.vs_dict.values())

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        # Initialize here all trainable variables used in training
        e = tf.device('/CPU:0') if vars_on_cpu else None
        if e: e.__enter__()
        with tf.variable_scope(self.name):
            ms = { v.name : tf.get_variable ( f'ms_{v.name}'.replace(':','_'), v.shape, dtype=v.dtype, initializer=tf.initializers.constant(0.0), trainable=False) for v in trainable_weights }
            vs = { v.name : tf.get_variable ( f'vs_{v.name}'.replace(':','_'), v.shape, dtype=v.dtype, initializer=tf.initializers.constant(0.0), trainable=False) for v in trainable_weights }
            self.ms_dict.update (ms)
            self.vs_dict.update (vs)
            
            if self.lr_dropout != 1.0:
                e = tf.device('/CPU:0') if lr_dropout_on_cpu else None
                if e: e.__enter__()                    
                lr_rnds = [ nn.random_binomial( v.shape, p=self.lr_dropout, dtype=v.dtype) for v in trainable_weights ]
                if e: e.__exit__(None, None, None)                
                self.lr_rnds_dict.update ( { v.name : rnd for v,rnd in zip(trainable_weights,lr_rnds) } )
        if e: e.__exit__(None, None, None)

    def get_update_op(self, grads_vars):
        updates = []

        if self.clipnorm > 0.0:
            norm = tf.sqrt( sum([tf.reduce_sum(tf.square(tf.cast(g, tf.float32))) for g,v in grads_vars]))
        updates += [ state_ops.assign_add( self.iterations, 1) ]
        for i, (g,v) in enumerate(grads_vars):
            if self.clipnorm > 0.0:
                g = self.tf_clip_norm(g, self.clipnorm, tf.cast(norm, g.dtype) )

            ms = self.ms_dict[ v.name ]
            vs = self.vs_dict[ v.name ]
            
            m_t = self.beta_1*ms + (1.0-self.beta_1) * g
            v_t = self.beta_2*vs + (1.0-self.beta_2) * tf.square(g-m_t)

            lr = tf.constant(self.lr, g.dtype)
            if self.lr_cos != 0:
                lr *= (tf.cos(  tf.cast(self.iterations, g.dtype) * (2*3.1415926535/ float(self.lr_cos) )  ) + 1.0) / 2.0

            v_diff = - lr * m_t / (tf.sqrt(v_t) + np.finfo( g.dtype.as_numpy_dtype ).resolution )
            if self.lr_dropout != 1.0:
                lr_rnd = self.lr_rnds_dict[v.name]
                v_diff *= lr_rnd
            new_v = v + v_diff

            updates.append (state_ops.assign(ms, m_t))
            updates.append (state_ops.assign(vs, v_t))
            updates.append (state_ops.assign(v, new_v))

        return control_flow_ops.group ( *updates, name=self.name+'_updates')
nn.AdaBelief = AdaBelief


