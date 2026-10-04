"""Torch-native Quick96 training, persistence, and merge inference.

The architecture, loss stack, batching, optimizer, sample-reuse behavior, and
merge output ordering are the official Quick96 contract.

Low-VRAM training keeps the checkpoint-owning modules and RMSprop state on
CPU and builds disposable compute mirrors on every selected GPU. Sufficient-
VRAM training keeps the canonical state on the primary GPU and mirrors only
secondary GPUs. In both cases every tower produces batch-sum gradients,
those gradients are averaged once, and one canonical optimizer step is made.
"""

import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


class QModel(ModelBase):
    _SUFFICIENT_VRAM_GB = 4

    @staticmethod
    def _validate_merge_model_data(path):
        """Require the minimum metadata written by ``ModelBase.save``.

        ``loss_history``, ``sample_for_preview``, and
        ``choosed_gpu_indexes`` are deliberately optional because the shared
        loader already treats them as historical fields with defaults.  The
        stable saved-model identity fields are ``iter`` and ``options``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                "Quick96 merge requires an existing saved model data.dat")
        try:
            model_data = pickle.loads(path.read_bytes())
        except Exception as exc:
            raise ValueError(
                f"Quick96 merge requires readable saved-model metadata: "
                f"{path}") from exc

        if not isinstance(model_data, dict):
            raise ValueError(
                "Quick96 merge data.dat must contain a metadata dictionary")

        missing = [name for name in ("iter", "options")
                   if name not in model_data]
        if missing:
            raise ValueError(
                "Quick96 merge data.dat is missing required field(s): "
                + ", ".join(missing))

        iteration = model_data["iter"]
        if type(iteration) is not int or iteration < 0:
            raise ValueError(
                "Quick96 merge data.dat field 'iter' must be a "
                "non-negative integer")

        options = model_data["options"]
        if (not isinstance(options, dict)
                or any(not isinstance(name, str) for name in options)):
            raise ValueError(
                "Quick96 merge data.dat field 'options' must be a "
                "dictionary with string keys")

    def get_strpath_storage_for_file(self, filename):
        path = super().get_strpath_storage_for_file(filename)
        if filename == "data.dat" and not self.is_training:
            # ModelBase resolves this virtual path immediately before it
            # unpickles data.dat.  Validate here so even a non-mapping pickle
            # fails with the Quick96-local merge error rather than reaching
            # the generic loader's mapping operations.
            self._validate_merge_model_data(path)
        return path

    @staticmethod
    def _official_batch_layout(selected_device_count):
        tower_count = max(1, selected_device_count)
        per_tower = max(1, 4 // tower_count)
        return tower_count, per_tower, tower_count * per_tower

    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = (
            "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC")
        nn.initialize(data_format=self.model_data_format)

        resolution = self.resolution = 96
        self.face_type = FaceType.FULL
        ae_dims = 128
        e_dims = 64
        d_dims = 64
        d_mask_dims = 16
        self.pretrain = False
        self.pretrain_just_disabled = False
        masked_training = True

        # Official Quick96 placement decision: canonical state is on GPU only
        # for active training when every selected device has >= 4 GiB VRAM.
        self.models_opt_on_gpu = bool(
            devices and self.is_training
            and all(dev.total_mem_gb >= self._SUFFICIENT_VRAM_GB
                    for dev in devices))
        self.low_vram_split = bool(
            self.is_training and devices and not self.models_opt_on_gpu)
        canonical_device = (nn.device if (self.models_opt_on_gpu
                                          or not self.is_training)
                            else torch.device("cpu"))

        input_ch = 3
        self.bgr_shape = nn.get4Dshape(resolution, resolution, input_ch)
        self.mask_shape = nn.get4Dshape(resolution, resolution, 1)
        self.model_filename_list = []

        model_archi = nn.DeepFakeArchi(resolution, opts="ud")
        self.encoder = model_archi.Encoder(
            in_ch=input_ch, e_ch=e_dims, name="encoder")
        encoder_out_ch = (
            self.encoder.get_out_ch()
            * self.encoder.get_out_res(resolution) ** 2)
        self.inter = model_archi.Inter(
            in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims,
            name="inter")
        inter_out_ch = self.inter.get_out_ch()
        self.decoder_src = model_archi.Decoder(
            in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims,
            name="decoder_src")
        self.decoder_dst = model_archi.Decoder(
            in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims,
            name="decoder_dst")

        self._quick96_components = (
            self.encoder, self.inter, self.decoder_src, self.decoder_dst)
        for component in self._quick96_components:
            component.to(canonical_device)

        self.model_filename_list += [
            [self.encoder, "encoder.npy"],
            [self.inter, "inter.npy"],
            [self.decoder_src, "decoder_src.npy"],
            [self.decoder_dst, "decoder_dst.npy"],
        ]

        if self.is_training:
            self.src_dst_trainable_weights = (
                self.encoder.get_weights() + self.inter.get_weights()
                + self.decoder_src.get_weights()
                + self.decoder_dst.get_weights())

            self._bind_official_optimizer_names(self._quick96_components)
            self.src_dst_opt = nn.RMSprop(
                lr=2e-4, rho=0.9, lr_dropout=0.3, name="src_dst_opt")
            self.src_dst_opt.initialize_variables(
                self.src_dst_trainable_weights,
                vars_on_cpu=not self.models_opt_on_gpu)
            if not self.models_opt_on_gpu:
                # OptimizerBase normally follows the global primary device for
                # the scalar counter. Quick96's low-VRAM contract instead
                # keeps ALL canonical optimizer state, including iters, on CPU.
                self.src_dst_opt.iterations.data = (
                    self.src_dst_opt.iterations.data.to(canonical_device))
            self.model_filename_list += [
                (self.src_dst_opt, "src_dst_opt.npy")]

            gpu_count, self._quick96_bs_per_tower, official_batch_size = (
                self._official_batch_layout(len(devices)))
            self.set_batch_size(official_batch_size)

            # Feature-A persistence: data.dat means a strict full resume;
            # otherwise initialize, optionally from the historical package.
            self._quick96_saved_model_resume = self.model_data_path.exists()
            self.pretrained_components_loaded = []
            self._load_or_initialize_state()

            # Build tower topology only after canonical load/init. In low-VRAM
            # mode replica 0 is CPU state, not a compute tower.
            selected_plan = nn.ReplicaPlan.from_device_config(device_config)
            if self.low_vram_split:
                self.replica_plan = nn.ReplicaPlan.from_torch_devices(
                    torch.device("cpu"),
                    [torch.device("cpu"), *selected_plan.replica_devices])
                self._quick96_tower_offset = 1
            else:
                self.replica_plan = selected_plan
                self._quick96_tower_offset = 0

            self.tower_devices = tuple(
                self.replica_plan.replica_devices[
                    self._quick96_tower_offset:
                    self._quick96_tower_offset + gpu_count])

            if self.replica_plan.num_replicas > 1:
                for component in self._quick96_components:
                    self.replica_plan.add_component(component)
                self.replica_plan.initial_sync()

            self._install_eager_paths(masked_training)
            self._initialize_sample_generators()
            self.last_samples = None
        else:
            # Merge is only valid for a genuine saved model. The four model
            # components are required strictly; optimizer/training state is
            # deliberately neither constructed nor loaded.
            if not self.model_data_path.exists():
                raise FileNotFoundError(
                    "Quick96 merge requires an existing saved model data.dat")
            self._quick96_saved_model_resume = True
            self.pretrained_components_loaded = []
            self._load_or_initialize_state()
            for component in self._quick96_components:
                component.eval()
            self._install_merge_path()

    @staticmethod
    def _bind_official_optimizer_names(components):
        """Bind each parameter to its official variable name and owner."""
        for component in components:
            owners = {}
            for module in component.modules():
                for parameter in module.parameters(recurse=False):
                    owners[id(parameter)] = module
            for sub_name, parameter in component._iter_official_weights():
                parameter._dfl_name = f"{component.name}/{sub_name}"
                owner = owners.get(id(parameter))
                if owner is not None:
                    parameter._dfl_owner_layer = owner

    def _load_or_initialize_state(self):
        pretrained_root = (None if self.pretrained_model_path is None else
                           Path(self.pretrained_model_path))
        for model, filename in io.progress_bar_generator(
                self.model_filename_list, "Initializing models"):
            if self._quick96_saved_model_resume:
                path = self.get_strpath_storage_for_file(filename)
                if not model.load_weights(path):
                    raise FileNotFoundError(
                        f"required component file missing on resume: {path}")
                continue

            loaded = False
            if pretrained_root is not None:
                pretrained_path = pretrained_root / filename
                if pretrained_path.exists():
                    # Strict loader failures propagate: a supplied corrupt or
                    # incompatible pretrained component is never ignored.
                    model.load_weights(pretrained_path)
                    self.pretrained_components_loaded.append(filename)
                    loaded = True
            if not loaded:
                model.init_weights()

    def _component_for_tower(self, component, tower):
        replica = tower + self._quick96_tower_offset
        if replica == 0:
            return component
        for component_set in self.replica_plan.components:
            if component_set.canonical is component:
                return component_set.mirrors[replica - 1]
        raise RuntimeError(
            f"Quick96 tower {tower} has no mirror for {component.name}")

    def _tower_parameter_lists(self):
        if self.replica_plan.num_replicas == 1:
            return [self.src_dst_trainable_weights]
        all_replicas = nn.replica_param_lists(
            self.replica_plan, self.src_dst_trainable_weights)
        start = self._quick96_tower_offset
        return all_replicas[start:start + len(self.tower_devices)]

    @staticmethod
    def _to_tensor_device(value, device):
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(np.ascontiguousarray(value))
        return value.to(device=device, dtype=nn.floatx)

    def _forward_tower(self, warped_src, warped_dst, tower,
                       include_swap=True):
        device = self.tower_devices[tower]
        warped_src = self._to_tensor_device(warped_src, device)
        warped_dst = self._to_tensor_device(warped_dst, device)
        encoder = self._component_for_tower(self.encoder, tower)
        inter = self._component_for_tower(self.inter, tower)
        decoder_src = self._component_for_tower(self.decoder_src, tower)
        decoder_dst = self._component_for_tower(self.decoder_dst, tower)

        src_code = inter(encoder(warped_src))
        dst_code = inter(encoder(warped_dst))
        pred_src_src, pred_src_srcm = decoder_src(src_code)
        pred_dst_dst, pred_dst_dstm = decoder_dst(dst_code)
        result = {
            "pred_src_src": pred_src_src,
            "pred_src_srcm": pred_src_srcm,
            "pred_dst_dst": pred_dst_dst,
            "pred_dst_dstm": pred_dst_dstm,
        }
        if include_swap:
            pred_src_dst, pred_src_dstm = decoder_src(dst_code)
            result.update(pred_src_dst=pred_src_dst,
                          pred_src_dstm=pred_src_dstm)
        return result

    def _prepare_targets(self, target_src, target_srcm,
                         target_dst, target_dstm, device):
        target_src = self._to_tensor_device(target_src, device)
        target_srcm = self._to_tensor_device(target_srcm, device)
        target_dst = self._to_tensor_device(target_dst, device)
        target_dstm = self._to_tensor_device(target_dstm, device)
        radius = max(1, self.resolution // 32)
        target_srcm_blur = nn.gaussian_blur(target_srcm, radius)
        target_dstm_blur = nn.gaussian_blur(target_dstm, radius)
        return {
            "target_src": target_src,
            "target_srcm": target_srcm,
            "target_dst": target_dst,
            "target_dstm": target_dstm,
            "target_srcm_blur": target_srcm_blur,
            "target_dstm_blur": target_dstm_blur,
        }

    def _loss_vectors(self, targets, predictions, masked_training=True):
        target_src = targets["target_src"]
        target_dst = targets["target_dst"]
        target_srcm = targets["target_srcm"]
        target_dstm = targets["target_dstm"]
        target_srcm_blur = targets["target_srcm_blur"]
        target_dstm_blur = targets["target_dstm_blur"]
        pred_src_src = predictions["pred_src_src"]
        pred_dst_dst = predictions["pred_dst_dst"]

        target_src_opt = (target_src * target_srcm_blur
                          if masked_training else target_src)
        target_dst_opt = (target_dst * target_dstm_blur
                          if masked_training else target_dst)
        pred_src_opt = (pred_src_src * target_srcm_blur
                        if masked_training else pred_src_src)
        pred_dst_opt = (pred_dst_dst * target_dstm_blur
                        if masked_training else pred_dst_dst)
        filter_size = int(self.resolution / 11.6)

        src_loss = torch.mean(
            10 * nn.dssim(target_src_opt, pred_src_opt,
                          max_val=1.0, filter_size=filter_size), dim=1)
        src_loss = src_loss + torch.mean(
            10 * torch.square(target_src_opt - pred_src_opt),
            dim=(1, 2, 3))
        # Official mask loss uses the RAW target mask, never the blurred mask.
        src_loss = src_loss + torch.mean(
            10 * torch.square(target_srcm
                              - predictions["pred_src_srcm"]),
            dim=(1, 2, 3))

        dst_loss = torch.mean(
            10 * nn.dssim(target_dst_opt, pred_dst_opt,
                          max_val=1.0, filter_size=filter_size), dim=1)
        dst_loss = dst_loss + torch.mean(
            10 * torch.square(target_dst_opt - pred_dst_opt),
            dim=(1, 2, 3))
        dst_loss = dst_loss + torch.mean(
            10 * torch.square(target_dstm
                              - predictions["pred_dst_dstm"]),
            dim=(1, 2, 3))
        return src_loss, dst_loss

    def _clear_training_grads(self):
        if self.replica_plan.num_replicas > 1:
            nn.clear_replica_grads(self.replica_plan)
        else:
            for parameter in self.src_dst_trainable_weights:
                parameter.grad = None

    def _install_eager_paths(self, masked_training):
        def train_vectors(warped_src, target_src, target_srcm,
                          warped_dst, target_dst, target_dstm):
            expected = self.get_batch_size()
            if (len(warped_src) != len(warped_dst)
                    or not 0 < len(warped_src) <= expected):
                raise ValueError(
                    f"Quick96 expected matching non-empty batches no larger "
                    f"than {expected}, got "
                    f"source={len(warped_src)}, destination={len(warped_dst)}")

            self._clear_training_grads()
            params_by_tower = self._tower_parameter_lists()
            src_vectors = []
            dst_vectors = []
            tower_gradients = []
            shard_size = self._quick96_bs_per_tower

            for tower, device in enumerate(self.tower_devices):
                sl = slice(tower * shard_size, (tower + 1) * shard_size)
                predictions = self._forward_tower(
                    warped_src[sl], warped_dst[sl], tower,
                    include_swap=False)
                targets = self._prepare_targets(
                    target_src[sl], target_srcm[sl],
                    target_dst[sl], target_dstm[sl], device)
                src_loss, dst_loss = self._loss_vectors(
                    targets, predictions, masked_training)
                torch.autograd.backward(
                    src_loss + dst_loss,
                    torch.ones_like(src_loss + dst_loss))

                params = params_by_tower[tower]
                gradients = []
                for parameter in params:
                    if parameter.grad is None:
                        raise RuntimeError(
                            "Quick96 trainable parameter has no gradient: "
                            f"{getattr(parameter, '_dfl_name', parameter)}")
                    gradients.append(parameter.grad)
                tower_gradients.append(gradients)
                src_vectors.append(src_loss.detach())
                dst_vectors.append(dst_loss.detach())

            canonical_device = self.src_dst_trainable_weights[0].device
            canonical_gv = []
            for gradients in tower_gradients:
                canonical_gv.append([
                    (gradient.detach().to(canonical_device), canonical)
                    for gradient, canonical in zip(
                        gradients, self.src_dst_trainable_weights)
                ])
            averaged_gv = nn.average_gv_list(canonical_gv)
            self.src_dst_opt.get_update_op(averaged_gv)()
            if self.replica_plan.num_replicas > 1:
                self.replica_plan.sync_from_canonical()

            # Official average_tensor_list semantics: tower vectors are
            # averaged elementwise, not concatenated, before display mean.
            src_display = nn.average_tensor_list([
                value.to(canonical_device) for value in src_vectors])
            dst_display = nn.average_tensor_list([
                value.to(canonical_device) for value in dst_vectors])
            return src_display, dst_display

        self._src_dst_train_vectors = train_vectors

        def src_dst_train(warped_src, target_src, target_srcm,
                          warped_dst, target_dst, target_dstm):
            src_loss, dst_loss = train_vectors(
                warped_src, target_src, target_srcm,
                warped_dst, target_dst, target_dstm)
            return (float(src_loss.mean().cpu()),
                    float(dst_loss.mean().cpu()))

        self.src_dst_train = src_dst_train

        def AE_view(warped_src, warped_dst):
            expected = self.get_batch_size()
            if (len(warped_src) != len(warped_dst)
                    or not 0 < len(warped_src) <= expected):
                raise ValueError(
                    f"Quick96 AE_view expected matching non-empty batches no "
                    f"larger than {expected}, got "
                    f"source={len(warped_src)}, destination={len(warped_dst)}")
            outputs = [[] for _ in range(5)]
            shard_size = self._quick96_bs_per_tower
            with torch.no_grad():
                for tower in range(len(self.tower_devices)):
                    sl = slice(tower * shard_size,
                               (tower + 1) * shard_size)
                    f = self._forward_tower(
                        warped_src[sl], warped_dst[sl], tower,
                        include_swap=True)
                    values = (
                        f["pred_src_src"], f["pred_dst_dst"],
                        f["pred_dst_dstm"], f["pred_src_dst"],
                        f["pred_src_dstm"])
                    for bucket, value in zip(outputs, values):
                        bucket.append(value.detach().cpu().numpy())
            return [np.concatenate(bucket, axis=0) for bucket in outputs]

        self.AE_view = AE_view

    def _install_merge_path(self):
        """Install the official non-training eager inference boundary."""
        device = nn.device

        def AE_merge(warped_dst):
            warped_dst = self._to_tensor_device(warped_dst, device)
            with torch.inference_mode():
                dst_code = self.inter(self.encoder(warped_dst))
                pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
                _, pred_dst_dstm = self.decoder_dst(dst_code)
                # Official internal order. predictor_func deliberately
                # reorders the two masks for the external merger consumer.
                return [
                    value.detach().cpu().numpy()
                    for value in (
                        pred_src_dst, pred_dst_dstm, pred_src_dstm)
                ]

        self.AE_merge = AE_merge

    def _initialize_sample_generators(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        src_generators_count = cpu_count // 2
        dst_generators_count = cpu_count // 2
        common_outputs = [
            {"sample_type": SampleProcessor.SampleType.FACE_IMAGE,
             "warp": True, "transform": True,
             "channel_type": SampleProcessor.ChannelType.BGR,
             "face_type": self.face_type, "data_format": nn.data_format,
             "resolution": self.resolution},
            {"sample_type": SampleProcessor.SampleType.FACE_IMAGE,
             "warp": False, "transform": True,
             "channel_type": SampleProcessor.ChannelType.BGR,
             "face_type": self.face_type, "data_format": nn.data_format,
             "resolution": self.resolution},
            {"sample_type": SampleProcessor.SampleType.FACE_MASK,
             "warp": False, "transform": True,
             "channel_type": SampleProcessor.ChannelType.G,
             "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
             "face_type": self.face_type, "data_format": nn.data_format,
             "resolution": self.resolution},
        ]
        self.set_training_data_generators([
            SampleGeneratorFace(
                self.training_data_src_path, debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=SampleProcessor.Options(
                    random_flip=False),
                output_sample_types=common_outputs,
                generators_count=src_generators_count),
            SampleGeneratorFace(
                self.training_data_dst_path, debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=SampleProcessor.Options(
                    random_flip=False),
                output_sample_types=common_outputs,
                generators_count=dst_generators_count),
        ])

    def finalize(self):
        plan = getattr(self, "replica_plan", None)
        if plan is not None:
            plan.dispose()
            self.replica_plan = None
        super().finalize()

    def get_model_filename_list(self):
        return self.model_filename_list

    def onSave(self):
        for model, filename in io.progress_bar_generator(
                self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights(self.get_strpath_storage_for_file(filename))

    def onTrainOneIter(self):
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = self.last_samples
            warped_src = target_src
            warped_dst = target_dst
        else:
            self.last_samples = self.generate_next_samples()
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = self.last_samples

        src_loss, dst_loss = self.src_dst_train(
            warped_src, target_src, target_srcm,
            warped_dst, target_dst, target_dstm)
        return (("src_loss", src_loss), ("dst_loss", dst_loss))

    def onGetPreview(self, samples, for_history=False):
        ((warped_src, target_src, target_srcm),
         (warped_dst, target_dst, target_dstm)) = samples

        values = [target_src, target_dst] + self.AE_view(
            target_src, target_dst)
        S, D, SS, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(
                value, "NHWC", self.model_data_format), 0.0, 1.0)
            for value in values]
        DDM, SDM = [np.repeat(value, (3,), -1)
                    for value in (DDM, SDM)]
        target_srcm, target_dstm = [
            nn.to_data_format(value, "NHWC", self.model_data_format)
            for value in (target_srcm, target_dstm)]

        # Debug generators intentionally force batch size 1; production keeps
        # the official min(4, configured batch) behavior.
        n_samples = min(4, self.get_batch_size(), len(S), len(D))
        regular = []
        masked = []
        for i in range(n_samples):
            regular.append(np.concatenate(
                (S[i], SS[i], D[i], DD[i], SD[i]), axis=1))
            masked.append(np.concatenate(
                (S[i] * target_srcm[i], SS[i],
                 D[i] * target_dstm[i], DD[i] * DDM[i],
                 SD[i] * (DDM[i] * SDM[i])), axis=1))
        return [
            ("Quick96", np.concatenate(regular, axis=0)),
            ("Quick96 masked", np.concatenate(masked, axis=0)),
        ]

    def predictor_func(self, face=None):
        face = nn.to_data_format(
            np.asarray(face, dtype=np.float32)[None, ...],
            self.model_data_format, "NHWC")
        bgr, mask_dst_dstm, mask_src_dstm = [
            nn.to_data_format(
                value, "NHWC", self.model_data_format).astype(
                    np.float32, copy=False)
            for value in self.AE_merge(face)
        ]
        return (bgr[0], mask_src_dstm[0][..., 0],
                mask_dst_dstm[0][..., 0])

    def get_MergerConfig(self):
        import merger
        return (self.predictor_func,
                (self.resolution, self.resolution, 3),
                merger.MergerConfigMasked(
                    face_type=self.face_type, default_mode="overlay"))


Model = QModel
