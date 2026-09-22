"""
Leras.

like lighter keras.
This is my lightweight neural network library written from scratch.

Phase 3A: torch-only foundation.
+ the TensorFlow graph/session layer is replaced by a torch foundation:
  nn.torch (the torch module), nn.device (a torch.device placed through
  the Phase 2 backend-neutral device abstraction), nn.floatx (torch
  dtype), and the data-format helpers preserved from the official API
+ model/layer code creates parameters with device=nn.device and
  dtype=nn.floatx and must not call torch.cuda.* directly
+ checkpoint/initialization contracts: see core.leras.checkpoint
  (official DFL variable naming), core.leras.layers.Saveable
  (official-format serialization, strict load), and
  core.leras.initializers (initialization lifecycle)
+ Phase 4: core.leras.convert — the centralized checkpoint
  compatibility/conversion engine: official pickled-dict format <->
  torch, strict two-pass (all-or-nothing) conversion, declared
  layout rules, optimizer-state mapping (iters/ms_/vs_/acc_),
  structured deterministic reports, explicit reverse-export
  rejection; no TensorFlow import (pickle + NumPy parsing of the
  official format)
+ Phase 5: core.leras.models.ModelBase (nn.ModelBase) is the torch model
  container (official TF session concepts replaced by native torch
  execution: no variable_scope, no placeholder/feed_dict/tf_sess.run;
  run() = no_grad forward returning NumPy) and models/ModelBase.py (the
  top-level training lifecycle) runs on the torch foundation unchanged
  except documented hardenings; the official TF leras ModelBase source is
  preserved in core.leras.models.ModelBase_tf.py; model phases use native
  torch autograd in onTrainOneIter (nn.tf.gradients is not stubbed);
  nn.average_gv_list stays TF until the multi-GPU phase

Remaining TensorFlow-dependent leras areas (later Phase 3 subphases /
model phases): ops/* (Phase 3C migrated depth_to_space; Phase 3D
migrated dssim, gaussian_blur, style_loss, pixel_norm; Phase 3E1
migrated flatten, reshape_4D, average_tensor_list,
total_variation_mse; Phase 3E2 migrated random_binomial; the rest of
the official ops are preserved in ops/ops_tf.py and rebuilt in later
subphases), optimizers/* (Phase 3E2 migrated the optimizer
foundation, AdaBelief and RMSprop in torch; the TF reference is
preserved in optimizers/optimizers_tf.py; the TF gradient
machinery - nn.gradients/average_gv_list - is NOT stubbed under
torch: Phase 5 single-device model code uses native torch autograd;
average_gv_list (multi-GPU gradient averaging) stays TF until the
multi-GPU phase); archis/* (Phase 3F migrated ArchiBase and the
DeepFakeArchi factory to torch - the official TF source is preserved in
archis/archis_tf.py; Phase 3F also migrated the official discriminator
classes CodeDiscriminator/PatchDiscriminator/UNetPatchDiscriminator in
core.leras.models to torch - the official TF source is preserved in
models/discriminators_tf.py); models/* (Phase 5 migrated the leras
ModelBase container to torch - the official TF source is preserved in
models/ModelBase_tf.py; Phase 10B migrated the official XSeg model to
torch - the official TF source is preserved in models/XSeg_tf.py).

NCHW speed up training for 10-20%.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from .device import Devices, DeviceConfig, ask_choose_device_idxs  # noqa: F401  (device layer owns selection semantics; nn.DeviceConfig / nn.ask_choose_device_idxs stay aliases for existing call sites)


class nn():
    current_DeviceConfig = None

    # Phase 2: device-selection semantics live in core.leras.device and are
    # exposed here as class attributes, because every call site uses
    # `from core.leras import nn` (the class), not the nn.py module.
    DeviceConfig = DeviceConfig
    ask_choose_device_idxs = staticmethod(ask_choose_device_idxs)

    torch = None
    device = None

    data_format = None
    conv2d_ch_axis = None
    conv2d_spatial_axes = None

    floatx = None  # torch dtype

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NHWC"):
        if device_config is None:
            device_config = nn.getCurrentDeviceConfig()
        nn.setCurrentDeviceConfig(device_config)

        if nn.torch is None:
            import torch
            nn.torch = torch

            # Torch foundation registries (Phase 3A: layers foundation and
            # initializers; Phase 3C/3D/3E1: torch ops - depth_to_space,
            # dssim, gaussian_blur, style_loss, pixel_norm, flatten,
            # reshape_4D, average_tensor_list, total_variation_mse;
            # Phase 3E2: torch optimizers - OptimizerBase, AdaBelief,
            # RMSprop + the random_binomial op; Phase 3F: torch archis -
            # ArchiBase, the DeepFakeArchi factory (Encoder/Inter/Decoder
            # block classes) and the official discriminator classes
            # (CodeDiscriminator, PatchDiscriminator,
            # UNetPatchDiscriminator; Phase 5: the torch ModelBase
            # container; Phase 10B: the torch XSeg model). The remaining
            # leras subpackages (remaining ops) are rebuilt in later Phase
            # 3 subphases / model phases and are imported by their own
            # subphase entry points.
            import core.leras.layers  # noqa: F401
            import core.leras.initializers  # noqa: F401
            import core.leras.checkpoint  # noqa: F401
            import core.leras.ops  # noqa: F401
            import core.leras.optimizers  # noqa: F401  (Phase 3E2: torch optimizers)
            import core.leras.archis  # noqa: F401  (Phase 3F: torch archis foundation)
            import core.leras.models  # noqa: F401  (Phase 3F: torch discriminator classes;
                                          #  Phase 5: torch nn.ModelBase container — the
                                          #  official TF XSeg foundation stays behind the
                                          #  hasattr(nn, 'tf') guard in that __init__)

        torch = nn.torch

        # Device placement goes through the Phase 2 backend-neutral
        # device abstraction: CPU when no device is selected, otherwise
        # the backend (CUDA today; AMD/Intel future) resolves the
        # torch.device. No CUDA-specific call appears here.
        if len(device_config.devices) == 0:
            nn.device = torch.device('cpu')
        else:
            from .device import get_torch_device
            nn.device = get_torch_device(device_config.devices[0])

        if floatx == "float32":
            nn.set_floatx(torch.float32)
        elif floatx == "float16":
            nn.set_floatx(torch.float16)
        else:
            raise ValueError(f"unsupported floatx {floatx}")
        nn.set_data_format(data_format)

    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()

    @staticmethod
    def init_weights(target):
        """Torch form of the official ``nn.init_weights``: apply the
        per-parameter initializers that ``target`` (a torch module,
        e.g. a LayerBase) registered in ``build_weights`` (via
        ``register_param_initializers``/``get_param_initializers``,
        keyed by the torch registered parameter name) to its DIRECT
        parameters and buffers.

        - a parameter without a registered initializer keeps the value
          given at construction (explicit, never filled silently);
        - the DFL ``ca`` initializer placeholder fails loudly until its
          subprocess batch generation lands in Phase 3B;
        - modules composing sub-layers (archis) override
          ``init_weights()`` to cascade to their children, like the
          official archis do.
        """
        if not isinstance(target, nn.torch.nn.Module):
            return

        torch = nn.torch
        registered = target.get_param_initializers()
        direct = dict(target.named_parameters(recurse=False))
        direct.update(dict(target.named_buffers(recurse=False)))

        for name, param in direct.items():
            initializer = registered.get(name)
            if initializer is None:
                continue
            if initializer is nn.initializers.ca:
                raise NotImplementedError(
                    f"parameter '{name}' uses nn.initializers.ca whose "
                    "batch generation is not implemented yet (Phase 3B)"
                )
            with torch.no_grad():
                shape = tuple(param.shape)
                param.copy_(
                    initializer(shape, dtype=param.dtype).to(device=param.device)
                )

    @staticmethod
    def set_floatx(torch_dtype):
        """
        set default float type for all layers when dtype is None for them
        """
        nn.floatx = torch_dtype

    @staticmethod
    def set_data_format(data_format):
        if data_format != "NHWC" and data_format != "NCHW":
            raise ValueError(f"unsupported data_format {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1,2]
        elif data_format == "NCHW":
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2,3]

    @staticmethod
    def get4Dshape ( w, h, c ):
        """
        returns 4D shape based on current data_format
        """
        if nn.data_format == "NHWC":
            return (None,h,w,c)
        else:
            return (None,c,h,w)

    @staticmethod
    def to_data_format( x, to_data_format, from_data_format):
        if to_data_format == from_data_format:
            return x

        # torch tensors use the native (autograd-safe) permute: np.transpose
        # would route through Tensor.__array__ (Tensor.numpy()), which
        # raises for tensors that require grad - breaking every NHWC
        # forward pass under autograd (the official CPU training path)
        if nn.torch is not None and isinstance(x, nn.torch.Tensor):
            if to_data_format == "NHWC":
                return x.permute(0, 2, 3, 1)
            elif to_data_format == "NCHW":
                return x.permute(0, 3, 1, 2)

        if to_data_format == "NHWC":
            return np.transpose(x, (0,2,3,1) )
        elif to_data_format == "NCHW":
            return np.transpose(x, (0,3,1,2) )
        else:
            raise ValueError(f"unsupported to_data_format {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        # Torch foundation: there is no TF graph/session to reset; memory is
        # managed by torch's caching allocator. Kept as an API no-op so
        # existing call sites (reworked in later phases) keep working.
        pass

    @staticmethod
    def close_session():
        pass

    # Phase 2: device-selection semantics (DeviceConfig, ask_choose_device_idxs)
    # live in core.leras.device and are exposed on this class, so
    # `nn.DeviceConfig.*` and `nn.ask_choose_device_idxs(...)` keep working
    # for all existing call sites (`from core.leras import nn`, see the
    # import at the top of this file).
