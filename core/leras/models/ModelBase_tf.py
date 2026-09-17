"""DEAD REFERENCE - official DeepFaceLab (TF) source, preserved verbatim for the
Phase 5 review record. Do not import, do not execute, do not maintain.

This file contains the official TensorFlow leras model container
(``core/leras/models/ModelBase.py``) that Phase 5 replaced in place with the
torch version (``ModelBase.py`` — ``nn.ModelBase`` is now importable under the
torch foundation; the TF session concepts it used — variable_scope,
placeholder/feed_dict, tf_sess.run — are replaced by native torch execution,
see the torch module docstring).

The official code is kept exactly as shipped in the upstream baseline
(commit e4b7543ffa1d73b26fce1e31852727f658ba490c). The torch replacement
preserves the constructor signature, the attribute-discovery build loop
(xor_list over vars(self), generator on_build support), the layers/
layers_by_name registration, get_weights ordering, get_layers flattening,
__call__ auto-build, summary table, and the nn.ModelBase binding; only
build_for_run/run drop the TF placeholder/tf_sess.run mechanism (recorded
input contract + no_grad forward returning NumPy).

Note: this file is NOT imported by ``core/leras/models/__init__.py`` under
the torch foundation (it binds ``tf = nn.tf`` == None at import and its
build/run paths require TensorFlow); it exists as the review reference next
to the torch replacement, exactly like ``discriminators_tf.py``.
"""
import types
import numpy as np
from core.interact import interact as io
from core.leras import nn
tf = nn.tf

class ModelBase(nn.Saveable):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.layers = []
        self.layers_by_name = {}
        self.built = False
        self.args = args
        self.kwargs = kwargs
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
                with tf.variable_scope(layer.name):
                    layer.build_weights()
            elif isinstance (layer, ModelBase):
                layer.build()

            self.layers.append (layer)
            self.layers_by_name[layer.name] = layer

    def xor_list(self, lst1, lst2):
        return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]

    def build(self):
        with tf.variable_scope(self.name):

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

                v = vars(self)
                new_vars = self.xor_list (current_vars, list(v.keys()) )

                for name in new_vars:
                    self._build_sub(v[name],name)

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

        return self.forward(*args, **kwargs)

    # def compute_output_shape(self, shapes):
    #     if not self.built:
    #         self.build()

    #     not_list = False
    #     if not isinstance(shapes, list):
    #         not_list = True
    #         shapes = [shapes]

    #     with tf.device('/CPU:0'):
    #         # CPU tensors will not impact any performance, only slightly RAM "leakage"
    #         phs = []
    #         for dtype,sh in shapes:
    #             phs += [ tf.placeholder(dtype, sh) ]

    #         result = self.__call__(phs[0] if not_list else phs)

    #         if not isinstance(result, list):
    #             result = [result]

    #         result_shapes = []

    #         for t in result:
    #             result_shapes += [ t.shape.as_list() ]

    #         return result_shapes[0] if not_list else result_shapes

    def build_for_run(self, shapes_list):
        if not isinstance(shapes_list, list):
            raise ValueError("shapes_list must be a list.")

        self.run_placeholders = []
        for dtype,sh in shapes_list:
            self.run_placeholders.append ( tf.placeholder(dtype, sh) )

        self.run_output = self.__call__(self.run_placeholders)

    def run (self, inputs):
        if self.run_placeholders is None:
            raise Exception ("Model didn't build for run.")

        if len(inputs) != len(self.run_placeholders):
            raise ValueError("len(inputs) != self.run_placeholders")

        feed_dict = {}
        for ph, inp in zip(self.run_placeholders, inputs):
            feed_dict[ph] = inp

        return nn.tf_sess.run ( self.run_output, feed_dict=feed_dict)

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
        layers_params = [ int(np.sum(np.prod(w.shape) for w in l.get_weights())) for l in layers ]
        total_params = np.sum(layers_params)

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

