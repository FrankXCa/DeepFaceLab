"""XSeg — official DeepFaceLab XSeg training model on the torch
foundation (Phase 10C: XSeg training port).

The official TF source is preserved verbatim in ``Model_tf.py`` (dead
reference, never imported — project convention, like
``Model_SAEHD/Model_tf.py``).

Official behavior preserved:
- ``on_initialize_options``: the official first-run / override prompt
  flow verbatim — the official "Restart training?" prompt on resume
  override, the official ``face_type`` prompt on first run, the
  official ``ask_batch_size(4, range=[2,16])`` + "Enable pretraining
  mode" prompts, the official pretraining_data_path guard (skipped for
  export), and the official ``pretrain_just_disabled`` detection
  (stored pretrain True -> now False);
- ``on_initialize``: the official structural portion — the official
  data-format rule (NCHW when exporting or on GPU, NHWC otherwise;
  ``nn.initialize(data_format=...)``), the fixed ``resolution = 256``,
  the official face_type mapping, the official ``XSegNet`` construction
  (``name='XSeg'``, ``load_weights=not is_first_run``, the official
  ``nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt')`` passed in,
  ``place_model_on_cpu = len(devices) == 0``), the official
  ``pretrain_just_disabled -> set_iter(0)`` transition rule (the
  optimizer state is PRESERVED — only the iteration counter resets),
  the official multi-GPU batch split (the identity for the single-
  device torch port), the official train / view closures and the
  official sample-generator wiring (pretrain: the official
  ``SampleGeneratorFace`` on the pretraining data — warped BGR face +
  warped single-channel (G) "target"; normal: the official
  ``SampleGeneratorFaceXSeg`` src/dst generator + the official
  non-warped src / dst preview generators);
- the official loss stack (per-sample (N,) vectors): pretrain =
  ``5·DSSIM(fs=int(256/11.6)=22) + 5·DSSIM(fs=int(256/23.2)=11) +
  mean(10·square(target − pred))`` over the ZEROED-skip (pretrain)
  forward, normal = the official per-sample
  ``sigmoid_cross_entropy_with_logits(labels=target, logits=logits)``
  (mean over all axes except the batch — the migrated
  ``nn.sigmoid_cross_entropy`` is the verbatim formula);
- the official gradient semantics: ``nn.gradients(per-sample loss
  vector, model.get_weights())`` = the BATCH SUM over the per-sample
  vector (torch ``backward(loss_vec, ones_like(loss_vec))`` — NOT
  ``loss.mean().backward()``; the Phase 6B SAEHD pattern, pinned by
  the Phase 10B frozen-reference evidence), then the official
  ``opt.get_update_op(...)`` RMSprop step (Phase 3E2 torch RMSprop —
  official ``acc_*`` state naming, the official denominator epsilon,
  the per-step lr_dropout mask);
- ``onTrainOneIter`` (the official sample fetch + train + the
  ``('loss', np.mean(loss))`` return), ``onGetPreview`` (the official
  n_samples = min(4, batch_size, 800//256) preview rows — pretrain:
  the (I, IM) two-panel rows; normal: the official composite
  ``I·IM + 0.5·I·(1−IM) + 0.5·green_bg·(1−IM)`` / IM /
  ``I·IM + 0.5·I·(1−IM) + 0.5·green_bg·(1−IM)`` three-panel rows,
  the official 1-channel-to-3 mask repeats, the official
  ``'XSeg training faces'`` / ``'XSeg src faces'`` / ``'XSeg dst
  faces'`` layout), ``get_model_filename_list`` (the official XSegNet
  list: ``[opt, 'XSeg_256_opt.npy']`` + ``[model,
  'XSeg_256.npy']``) and ``onSave`` (the official
  ``model.save_weights()`` — Phase 4 Saveable engine, official raw
  pickle protocol-4 ``.npy`` streams);
- the official top-level lifecycle (Phase 5 ``models/ModelBase.py``)
  is used unchanged: data.dat-gated resume, the two distinct
  iteration counters, the 24-slot autobackup ring.

Documented torch adaptations (Phase 10C — the official TF
graph/session concepts removed, exactly like the Phase 6 SAEHD port):
- the official CPU placeholders (``input_t`` / ``target_t``) and the
  ``feed_dict`` session runs are removed — the torch foundation feeds
  arrays directly and executes eagerly on ``nn.device`` (the Phase 2
  device model); the official train / view closures become eager
  closures over the same ``XSegNet.flow`` net (the ``view`` closure
  runs the same mode — pretrain or normal — as the training graph,
  the official graph shared the same ``pred`` node; no-grad, NumPy in
  the model data format, CPU NumPy out);
- the official ``nn.gradients`` + ``average_gv_list`` +
  ``get_update_op`` session-op trio becomes native torch autograd:
  ``backward(loss_vec, ones_like(loss_vec))`` (the official batch SUM
  — a bare ``.backward()`` is only legal for numel()==1 and crashes
  for batch > 1, which the official model uses) + the Phase 3E2
  ``opt.get_update_op([(p.grad, p) ...])``; the multi-GPU
  ``average_gv_list`` gradient averaging is the identity for the
  single-device port (official ops_tf L76-77);
- the official ``tf.device`` placement contexts (model on GPU /
  optimizer update on ``models_opt_device``) are replaced by the
  Phase 2 co-location semantics: the net and its optimizer state
  live on ``nn.device`` (identical placement for every official XSeg
  call path: ``Model.py`` passes ``place_model_on_cpu = len(devices)
  == 0``);
- the official ``RMSprop`` lr_dropout is the Phase 3E2 production
  behavior: ONE FRESH per-parameter mask per step (the official TF
  graph re-evaluated its seeded random op on every update run; the
  frozen-mask streams of the Phase 10B reference are test-harness
  inputs ONLY and are never hardcoded into the production runtime);
- a missing REQUIRED file on a resume fails explicitly with
  ``FileNotFoundError`` instead of the official silent
  re-initialization (the Phase 10B ``XSegNet`` strict policy — the
  Phase 6 SAEHD precedent; the official intentional
  ``pretrain_just_disabled`` transition keeps its official
  semantics: iteration reset, optimizer state preserved through the
  persistent ``.npy`` files);
- ``export_dfm`` (Phase 10E, DFM/ONNX): the official tf2onnx export
  becomes the legacy torch TorchScript ONNX exporter
  (``dynamo=False`` — torch 2.14's dynamo path requires onnxscript,
  which the project deliberately does not install) driven by a
  plain-``torch.nn.Module`` wrapper that implements the official
  NHWC -> NCHW -> NHWC boundary around the NCHW net and the official
  ``_, pred`` flow-tuple selection (the sigmoid is ``out_mask``);
  the official ``in_face:0 -> out_mask:0`` opset-13 contract, the
  name-based dynamic batch axis and the onnxproto annotation of
  ``out_mask`` are preserved (see the method).

No TensorFlow import appears on this path and device placement remains
through the Phase 2 abstraction. CUDA training temporarily disables
cuDNN TF32 only across XSeg forward/loss/backward, then restores the
caller's flag exactly; matmul policy and unrelated models are untouched.
"""

import multiprocessing
from contextlib import contextmanager

import numpy as np
import onnx
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType, XSegNet
from models import ModelBase
from models.ModelBase import SkippedGeneratorStep
from samplelib import *


@contextmanager
def _xseg_training_precision(input_t):
    """Scope XSeg CUDA convolution forward/backward to full FP32."""
    if not input_t.is_cuda:
        yield
        return

    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


class XSegModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_model_class_name='XSeg', **kwargs)

    #override
    def on_initialize_options(self):
        # official Model_tf.py L20-40, verbatim semantics (this logic
        # is backend-agnostic — no torch adaptation)
        ask_override = self.ask_override()

        if not self.is_first_run() and ask_override:
            if io.input_bool(f"Restart training?", False, help_message="Reset model weights and start training from scratch."):
                self.set_iter(0)

        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        if self.is_first_run():
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Choose the same as your deepfake model.").lower()

        if self.is_first_run() or ask_override:
            self.ask_batch_size(4, range=[2,16])
            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain)

        if not self.is_exporting and (self.options['pretrain'] and self.get_pretraining_data_path() is None):
            raise Exception("pretraining_data_path is not defined")

        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        self.model_data_format = "NCHW" if self.is_exporting or (len(device_config.devices) != 0 and not self.is_debug()) else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        # torch (Phase 10C): the official `tf = nn.tf` import, the TF
        # placeholders and the `tf.device` placement contexts are
        # removed — the torch foundation executes eagerly on nn.device
        # (Phase 2 device model; the official model/optimizer
        # placement branches are documented co-location semantics).

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.resolution = resolution = 256

        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]

        place_model_on_cpu = len(devices) == 0
        # torch: the official models_opt_device selection (CPU:0 when
        # place_model_on_cpu, else the default device) is the Phase 2
        # co-location — the net and its optimizer state live on
        # nn.device (identical placement for every official XSeg call
        # path: Model.py passes place_model_on_cpu = len(devices) == 0).

        bgr_shape = nn.get4Dshape(resolution,resolution,3)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        # Initializing model classes (Phase 10B torch XSegNet — the
        # official net wrapper; the official
        # nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt') is
        # constructed by the caller and passed in, exactly like the
        # official code)
        self.model = XSegNet(name='XSeg',
                               resolution=resolution,
                               load_weights=not self.is_first_run(),
                               weights_file_root=self.get_model_root_path(),
                               training=True,
                               place_model_on_cpu=place_model_on_cpu,
                               optimizer=nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt'),
                               data_format=nn.data_format)

        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            # official transition semantics: the iteration counter
            # resets, the optimizer state is PRESERVED (it persists
            # through the .npy files and is reloaded by the XSegNet
            # load/init loop above)
            self.set_iter(0)

        if self.is_training:
            # Adjust batch size for multiple GPU (the official per-GPU
            # loop collapses to the single device — the official
            # average_gv_list over one tower is the identity)
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # --- tensor plumbing (torch replacement of the official
            #     CPU placeholders + feed_dict) ---------------------
            def _to_tensor(x):
                # official feed_dict placement semantics: to the
                # current nn.device in the declared floatx (NumPy or
                # tensor in, the model data format — the sample
                # generators emit in nn.data_format)
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(np.ascontiguousarray(x))
                x = x.to(device=nn.device, dtype=nn.floatx)
                if x.dim() == 3:
                    x = x[None, ...]
                return x

            def _to_numpy(x):
                # official caller contract: TF session outputs were
                # NumPy (in the graph's data format)
                return x.detach().cpu().numpy()

            # --- training / view closures (the official L130-142
            #     session closures) ---------------------------------
            # torch (Phase 10C): the official per-tower TF graph —
            # the mode-dependent forward (self.model.flow with the
            # official pretrain flag), the official loss stack, the
            # nn.gradients (batch SUM over the per-sample vector) and
            # the opt.get_update_op update op — is reproduced as
            # native eager torch with identical semantics.

            def train(input_np, target_np):
                input_t = _to_tensor(input_np)
                target_t = _to_tensor(target_np)

                # CUDA: the same cuDNN policy must cover BOTH forward
                # and backward; inference's forward-only scope is not
                # sufficient for training-gradient parity.
                with _xseg_training_precision(input_t):
                    # official L103: the mode-dependent forward (the
                    # pretrain forward zeroes the skip connections);
                    # the Phase 8 autocast region wraps the forward.
                    with self._mp_autocast():
                        gpu_pred_logits_t, gpu_pred_t = self.model.flow(input_t, pretrain=self.pretrain)

                    # official L107-114: per-sample (N,) loss vector
                    if self.pretrain:
                        # Structural loss: official DSSIM sizes 22/11.
                        gpu_loss = torch.mean( 5*nn.dssim(target_t, gpu_pred_t, max_val=1.0, filter_size=int(resolution/11.6)), dim=1)
                        gpu_loss += torch.mean( 5*nn.dssim(target_t, gpu_pred_t, max_val=1.0, filter_size=int(resolution/23.2)), dim=1)
                        # Pixel loss
                        gpu_loss += torch.mean( 10*torch.square(target_t - gpu_pred_t), dim=(1,2,3))
                    else:
                        # official per-sample mean BCE with logits
                        gpu_loss = nn.sigmoid_cross_entropy(target_t, gpu_pred_logits_t)

                    # official nn.gradients(per-sample vector, weights)
                    # = batch-SUM backward. Keep backward inside the
                    # full-f32 cuDNN scope on CUDA.
                    self.model.model.zero_grad()
                    self._mp_backward(gpu_loss)

                # The optimizer is elementwise and does not use cuDNN.
                self._mp_unscale_opt(self.model.opt, self.model.get_weights())
                stepped = self._mp_opt_step(
                    self.model.opt,
                    [(p.grad, p) for p in self.model.get_weights()])
                self._mp_scaler_update()
                if not stepped:
                    raise SkippedGeneratorStep('XSeg FP16 gradients overflowed')

                # official: the closure returns the (numpy) loss
                # vector l from the session run
                return _to_numpy(gpu_loss)

            def view(input_np):
                # official: tf_sess.run([pred], feed_dict=...) — the
                # SAME pred node as the training graph, so the SAME
                # mode (pretrain/normal) as self.pretrain
                input_t = _to_tensor(input_np)
                with torch.no_grad():
                    _, gpu_pred_t = self.model.flow(input_t, pretrain=self.pretrain)
                return [ _to_numpy(gpu_pred_t) ]
                # torch: the official nn.tf_sess.run inference
                # boundary — no gradient bookkeeping

            self.train = train
            self.view = view

            # initializing sample generators (official L144-181,
            # verbatim)
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_dst_generators_count = cpu_count // 2
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            if self.pretrain:
                pretrain_gen = SampleGeneratorFace(self.get_pretraining_data_path(), debug=self.is_debug(), batch_size=self.get_batch_size(),
                                    sample_process_options=SampleProcessor.Options(random_flip=True),
                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                            {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                          ],
                                    uniform_yaw_distribution=False,
                                    generators_count=cpu_count )
                self.set_training_data_generators ([pretrain_gen])
            else:
                srcdst_generator = SampleGeneratorFaceXSeg([self.training_data_src_path, self.training_data_dst_path],
                                                            debug=self.is_debug(),
                                                            batch_size=self.get_batch_size(),
                                                            resolution=resolution,
                                                            face_type=self.face_type,
                                                            generators_count=src_dst_generators_count,
                                                            data_format=nn.data_format)

                src_generator = SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=src_generators_count,
                                                    raise_on_no_data=False )
                dst_generator = SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=dst_generators_count,
                                                    raise_on_no_data=False )

                self.set_training_data_generators ([srcdst_generator, src_generator, dst_generator])

    #override
    def get_model_filename_list(self):
        return self.model.model_filename_list

    #override
    def onSave(self):
        self.model.save_weights()

    #override
    def onTrainOneIter(self):
        # official Model_tf.py L192-196: the sample fetch, the train
        # closure (forward + official loss + batch-SUM gradients +
        # the RMSprop update), the official per-sample-mean loss
        # return. torch (Phase 10C): through the ModelBase Phase 8
        # helper — in 'off' precision mode (the XSeg 10C target) this
        # is the exact direct call.
        image_np, target_np = self.generate_next_samples()[0]
        loss = self._mp_run_generator(self.train, image_np, target_np)

        return ( ('loss', np.mean(loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        # official Model_tf.py L199-252, verbatim layout (the view()
        # closure is now an eager no-grad forward over the same mode
        # as the training graph)
        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        if self.pretrain:
            srcdst_samples, = samples
            image_np, mask_np = srcdst_samples
        else:
            srcdst_samples, src_samples, dst_samples = samples
            image_np, mask_np = srcdst_samples

        I, M, IM, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([image_np,mask_np] + self.view (image_np) ) ]
        M, IM, = [ np.repeat (x, (3,), -1) for x in [M, IM] ]

        green_bg = np.tile( np.array([0,1,0], dtype=np.float32)[None,None,...], (self.resolution,self.resolution,1) )

        result = []
        st = []
        for i in range(n_samples):
            if self.pretrain:
                ar = I[i], IM[i]
            else:
                ar = I[i]*M[i]+0.5*I[i]*(1-M[i])+0.5*green_bg*(1-M[i]), IM[i], I[i]*IM[i]+0.5*I[i]*(1-IM[i]) + 0.5*green_bg*(1-IM[i])
            st.append ( np.concatenate ( ar, axis=1) )
        result += [ ('XSeg training faces', np.concatenate (st, axis=0 )), ]

        if not self.pretrain and len(src_samples) != 0:
            src_np, = src_samples

            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([src_np] + self.view (src_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i] + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg src faces', np.concatenate (st, axis=0)), ]

        if not self.pretrain and len(dst_samples) != 0:
            dst_np, = dst_samples

            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([dst_np] + self.view (dst_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i]  + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg dst faces', np.concatenate (st, axis=0)), ]

        return result

    # Phase 10E (DFM/ONNX): the official tf2onnx export
    # (Model_tf.py L254-281 — the in_face:0 (None,256,256,3) NHWC ->
    # out_mask:0 (None,256,256,1) NHWC opset-13 contract) on the
    # torch foundation.
    def export_dfm (self):
        output_path = self.get_strpath_storage_for_file('model.onnx')
        io.log_info(f'Dumping .onnx to {output_path}')

        # torch replacement for the official placeholder graph: the
        # official 'in_face' NHWC placeholder is transposed to NCHW
        # (the NCHW net built by the is_exporting data-format rule
        # above), the official `_, pred_t` selection of the flow
        # tuple (the sigmoid — the contract's out_mask) is executed
        # and transposed back to NHWC under the name 'out_mask'.
        # The torch exporter is the legacy TorchScript ONNX exporter
        # (dynamo=False): torch 2.14's dynamo path requires
        # onnxscript (deliberately not installed — the Phase 10E
        # dependency decision in PHASE10_STATE.md), and the legacy
        # path reproduces the official tf2onnx name-based contract
        # (dynamic_axes + opset_version + input/output names) exactly.
        class _OnnxWrapper (torch.nn.Module):
            def __init__ (self, model):
                super().__init__()
                self.model = model

            def forward (self, x_nhwc):
                x = x_nhwc.permute(0, 3, 1, 2)
                _, pred = self.model(x)
                return pred.permute(0, 2, 3, 1)

        wrapper = _OnnxWrapper(self.model.model)
        example = torch.zeros(1, self.resolution, self.resolution, 3,
                              dtype=nn.floatx)

        torch.onnx.export(
            wrapper, (example,),
            f=output_path,
            input_names=['in_face'],
            output_names=['out_mask'],
            opset_version=13,
            dynamo=False,
            dynamic_axes={'in_face': {0: 'batch'},
                          'out_mask': {0: 'batch'}},
            do_constant_folding=True,
        )

        # The TorchScript exporter leaves the out_mask H/W annotation
        # symbolic although the executed output is always
        # (N,res,res,1) — align the annotation with the contract.
        model = onnx.load(output_path)
        for out in model.graph.output:
            if out.name == 'out_mask':
                dims = out.type.tensor_type.shape.dim
                dims[1].dim_value = self.resolution
                dims[2].dim_value = self.resolution
        onnx.save(model, output_path)

Model = XSegModel
