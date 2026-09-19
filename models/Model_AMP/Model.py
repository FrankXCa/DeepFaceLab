"""AMP — official DeepFaceLab AMP model on the torch foundation
(Phase 7: single-device training semantics — the official loss
stack, train closures, update routing, preview rendering and
inference paths).

The official TF source is preserved verbatim in ``Model_tf.py`` (dead
reference, never imported — project convention, like
``ModelBase_tf.py`` / ``discriminators_tf.py``).

Official behavior preserved:
- ``on_initialize_options``: the complete official option set
  (resolution/face_type/models_opt_on_gpu, ae/inter/e/d/d_mask dims
  with the official first-run clipping — resolution a multiple of 32
  in 64-640, ae 32-1024, inter 32-2048, e/d/d_mask 16-256 with the
  even rounding, the d_mask_dims default derived from d_dims (d//3
  rounded even), morph_factor 0.1-0.5 — and the official
  first-run/override prompt order: the batch-size/flip/backup
  prompts, then resolution/face_type/dims, then uniform_yaw /
  blur_out_mask / lr_dropout, then gan_power (with the conditional
  gan_patch_size / gan_dims prompts), models_opt_on_gpu,
  random_warp, ct_mode, clipgrad, and the official
  ``gan_model_changed`` detection (stored vs prompted
  gan_patch_size / gan_dims);
- ``on_initialize``: the official structural portion — the
  official hard-wired ``model_data_format = "NCHW"`` (Model_tf.py
  L107), the official archi construction through the Phase 7
  ``nn.AMPArchi`` factory (core/leras/archis/AMP.py — the official
  no-argument Encoder / Inter / Decoder constructors and names:
  the flat pixel-normalized encoder codes with ``dense1`` inside
  the encoder, the single-dense inter reshaped to
  ``(N, inter_ch, inter_res, inter_res)``, the decoder image head
  (``out_conv`` / ``out_conv1..3`` concat -> depth_to_space RRC ->
  sigmoid) and mask head (``upscalem0..4`` -> ``out_convm`` ->
  sigmoid); the official ``use_fp16`` export-only conv-dtype knob),
  the official two-optimizer construction (Phase 3E2
  ``nn.AdaBelief`` x2, ``lr=5e-5``, the official
  ``lr_dropout in ['y','cpu']`` -> ``lr_cos=500`` /
  ``lr_dropout=0.3`` coupling, ``clipgrad`` -> clipnorm=1.0; the
  official ``src_dst_opt`` over ``G_weights = encoder + decoder``
  (L301) and, iff ``gan_power != 0``, the official
  ``nn.UNetPatchDiscriminator`` (Phase 3F) + ``GAN_opt``), the
  official ``model_filename_list`` (official ``.npy`` filenames,
  ``[model, file]`` pairs for components and ``(opt, file)`` tuples
  for optimizers), and the official sample-generator wiring
  (migrated samplelib; the official output-sample spec incl. the
  warped/unwarped FACE_IMAGE pair and the FULL_FACE / EYES_MOUTH
  FACE_MASK pair on both sides);
- the official training semantics (Phase 7, single-device — the
  Phase 2 device model): ``onTrainOneIter`` (the official
  L646-657 driver: sample fetch, the always-run generator step,
  the GAN step iff ``gan_power != 0``, the official two-value loss
  return = the per-sample vector means), the train closures
  (``train`` / ``GAN_train`` = the official L484-510 closures)
  with the complete official loss stack: the target preparation
  (L382-414: the ``blur_out_mask`` target rewrite with
  ``sigma = resolution/128`` and the element-wise div-zero guard,
  the softened loss masks ``clip(gblur, 0, 0.5) * 2`` and their
  anti-complements, the masked / anti-masked tensor set), the
  official five-term src/dst stacks (5x dssim @ int(R/11.6) +
  5x dssim @ int(R/23.2) on the blurred masks, 10x MSE masked,
  300x |target*em - pred*em| on the RAW targets, 10x (raw mask -
  pred mask)^2), the official background terms (0.1x MSE
  anti-masked + 1e-6 total_variation_mse on the anti-masked dst
  pred) and the official GAN terms iff ``gan_power != 0``: the
  discriminator forwards on the MASKED tensors, the 4-term
  ``DLossOnes`` generator loss x ``gan_power`` UNNORMALIZED,
  1e-6 total_variation_mse on the RAW full src pred and 0.02x MSE
  anti-masked src, and the official 8-term D loss x (1/8) — the
  1/8 factor is the OFFICIAL upstream normalization (L448-452) and
  is preserved, as is the official batch-SUM gradient over the
  per-sample loss vectors;
- the official morph semantics (Model_tf.py L360-367): the
  training morph mask is the per-sample EXACT-k uniform k-subset
  (k = int(inter_dims * morph_factor)) shuffle of the k-ones /
  (D-k)-zeros base pattern, stop-gradient (the Phase 7
  ``core.leras.archis.AMP.exact_k_morph_mask`` op — the
  torch.randperm implementation; NOT an i.i.d. Bernoulli mask),
  applied as ``src*m + dst*(1-m)`` on the inter codes; the
  inference morph (AE_view / AE_merge / the merger prompt) is the
  official DETERMINISTIC floor-channel-slice
  (first k = int(inter_dims * morph_value) channels from the
  inter_src head, the remainder from the inter_dst head);
- ``AE_view`` / ``AE_merge``: the official inference functions
  (the official TF session-run graph is replaced by an eager
  ``torch.no_grad`` forward pass over the same component chain —
  the official inference boundary; NumPy in ``model_data_format``,
  NumPy out, exactly like the official ``tf_sess.run`` caller
  contract used by ``predictor_func`` / the merger);
- ``onGetPreview`` (the official L660-705 preview layout — the
  three named strips 'AMP morph 1.0' / 'AMP morph list' /
  'AMP morph list masked', the official 2 rows x 3 tiles incl.
  the 3-channel repeats of the DDM masks, the official
  one-sample rendering — ``n_samples = min(4, batch_size,
  800 // resolution)`` bounds only the RANDOM index, and
  ``for_history`` fixes it to 0),
- ``predictor_func(face, morph_value)`` / ``get_MergerConfig``:
  official — the merger prompts the morph factor
  (``io.input_number("Morph factor", 1.0)`` clipped 0-1) and the
  ``predictor_morph`` closure feeds it to the official
  ``AE_merge``; the official ``merger.MergerConfigMasked``
  (face_type, default_mode='overlay');
- ``get_model_filename_list`` / ``onSave`` /
  ``should_save_preview_history``: official, unchanged (component
  saves go through the Phase 4 Saveable engine — official raw
  pickle protocol-4 ``.npy`` streams);
- the official top-level lifecycle (Phase 5 ``models/ModelBase.py``)
  is used unchanged: data.dat-gated resume, the two distinct
  iteration counters, the 2-stage save and the official first-run
  ``default_options.dat`` snapshot STAYS ENABLED (AMP never calls
  ``disable_default_options_autosave()`` — the official behavior;
  the stale ModelBase docstring claim is corrected comment-only in
  Phase 7).

Torch deviations (documented, numerics unchanged):
- the official multi-GPU graph (the per-GPU batch slices and the
  ``average_gv_list`` averaging, Model_tf.py L314-331 /
  L466-477) collapses to the single-device torch foundation
  (Phase 2): one forward, one loss stack, one optimizer step per
  closure; on one device the official
  ``gpu_count * bs_per_gpu`` batch-size reconstruction is the
  identity (omitted, like the Phase 6A/6B ports);
- the official ``tf.device`` placement (the CPU placeholders, the
  /CPU:0 mask-shuffle draw, the ``models_opt_on_gpu`` /
  ``'/CPU:0'`` optimizer-vars branch) is expressed through the
  Phase 2 device config: everything executes on ``nn.device``
  (single GPU or CPU); ``initialize_variables(vars_on_cpu=...)``
  keeps the official signature — torch co-locates optimizer state
  with its parameters (documented placement deviation, numerics
  unchanged); the official mask draw on ``/CPU:0`` becomes a draw
  on ``nn.device`` — the exact-k uniform k-subset distribution is
  device-independent;
- the official per-sample loss vectors are per-sample (N,)
  tensors; the official TF ``nn.gradients(loss_vec, vars)``
  (gradient seeds of ones) is the batch SUM over samples —
  reproduced as ``torch.autograd.backward(loss_vec,
  torch.ones_like(loss_vec))`` for BOTH the generator loss and
  the D loss (probe-verified: a bare ``.backward()`` raises for
  N > 1; the externals' batch-MEAN substitution is NOT
  inherited);
- the official slice-code decoder call (L371-376) is pruned from
  the train / GAN_train subgraphs by TF's lazy execution (the
  fetched tensors never depend on it — the dst stack uses
  ``pred_dst_dst``, not ``pred_src_dst``); the torch closures
  simply never invoke it; only ``AE_view`` / ``AE_merge`` feed it;
- the official ``gan_model_changed`` re-init set names only the
  GAN component (L539-541); the GAN_opt file then fails the
  official ``do_init = not load_weights(...)`` fallback and the
  official silently re-initializes it — under the Phase 4/5
  strict load policy (no silent fallback) the same official
  outcome (a changed discriminator gets a FRESH optimizer state)
  is preserved by re-initializing GAN and GAN_opt explicitly
  (the Phase 6B SAEHD precedent);
- a missing required component file on resume fails explicitly
  (``FileNotFoundError``) instead of the official silent
  re-initialization; a corrupt file fails the strict Phase 4
  load (``CheckpointLoadError``);
- the ``export_dfm`` (ONNX/DFM) export is out of scope for
  Phase 7 (the ONNX/DFM exclusion) — a ``NotImplementedError``
  stub, like the Phase 6B SAEHD port;
- the frozen inter heads are torch parameters with
  requires_grad=False: autograd constants the encoder
  gradients flow THROUGH (the TF back-prop-through-untrained-
  variables behavior — the official gradient requests at
  L454/L464 cover only the GAN weights and the G_weights set,
  so the inter variables never receive any gradient), keeping
  their .grad None (finding F4);
- the ``ask_batch_size`` prompt passes the official AMP
  constant 8 (the official SAEHD's VRAM-based suggestion is
  SAEHD-specific and NOT inherited — the plan option table
  records 'suggest 8').
"""

import multiprocessing

import numpy as np
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from core.leras.archis.AMP import exact_k_morph_mask
from facelib import FaceType
from models import ModelBase
from samplelib import *


class AMPModel(ModelBase):

    #override
    def on_initialize_options(self):
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 224)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_inter_dims         = self.options['inter_dims']         = self.load_or_def_option('inter_dims', 1024)

        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_morph_factor       = self.options['morph_factor']       = self.options.get('morph_factor', 0.5)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)
        default_lr_dropout         = self.options['lr_dropout']         = self.load_or_def_option('lr_dropout', 'n')
        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(8)

        if self.is_first_run():
            resolution = io.input_int ("Resolution", default_resolution, add_info="64-640",
                                       help_message="Training and processing data resolution in pixels. Too low values result in blurry faces, too high values require more VRAM, time and iterations to train. Typical fine value is 128.")
            resolution = np.clip( (resolution // 32) * 32, 64, 640 )
            self.options['resolution'] = resolution

            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['f','wf','head'], help_message="Whole face has better quality but covers less area of face. Head covers full head, but requires xseg for src and dst facesets.").lower()

        default_d_dims = self.options['d_dims'] = self.load_or_def_option('d_dims', 64)
        default_d_mask_dims = default_d_dims // 3
        default_d_mask_dims += default_d_mask_dims % 2
        default_d_mask_dims = self.options['d_mask_dims'] = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip (io.input_int ("AutoEncoder dimensions", default_ae_dims, add_info="32-1024",
                                   help_message="The higher the value, the higher the quality of the model, but the more iterations and VRAM are required. If there are enough iterations, 256 is fine for most use cases."), 32, 1024)

            self.options['inter_dims'] = np.clip (io.input_int ("Inter dimensions", default_inter_dims, add_info="32-2048",
                                   help_message="The higher the value, the higher the quality of the model, but the more iterations and VRAM are required. If there are enough iterations, 1024 is fine for most use cases."), 32, 2048)

            e_dims = np.clip (io.input_int ("Encoder dimensions", default_e_dims, add_info="16-256",
                                   help_message="The higher the value, the higher the quality of the model, but the more iterations and VRAM are required. If there are enough iterations, 64 is fine for most use cases."), 16, 256)
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip (io.input_int ("Decoder dimensions", default_d_dims, add_info="16-256",
                                   help_message="The higher the value, the higher the quality of the model, but the more iterations and VRAM are required. If there are enough iterations, 64 is fine for most use cases."), 16, 256)
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip (io.input_int ("Decoder mask dimensions", default_d_mask_dims, add_info="16-256",
                                   help_message="The higher the value, the higher the quality of the model, but the more iterations and VRAM are required. If there are enough iterations, 64 is fine for most use cases."), 16, 256)
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

            morph_factor = np.clip (io.input_number ("Morph factor.", default_morph_factor, add_info="0.1-0.5",
                                   help_message="How strongly the src and dst inter codes are mixed. The lower the value, the less the face is morphed towards the src face."), 0.1, 0.5)
            self.options['morph_factor'] = morph_factor

        if self.is_first_run() or ask_override:
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')
            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 5.0", help_message="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1"), 0.0, 5.0 )

            if self.options['gan_power'] != 0.0:
                gan_patch_size = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8." ), 3, 640 )
                self.options['gan_patch_size'] = gan_patch_size

                gan_dims = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-512", help_message="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16." ), 4, 512 )
                self.options['gan_dims'] = gan_dims

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. If src faceset is deverse enough, then lct mode is fine in most cases.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)
        # torch (Phase 7): the official `tf = nn.tf` import, the TF
        # placeholders and the `tf.device` placement contexts are
        # removed — the torch foundation executes eagerly on
        # nn.device (Phase 2 device model; the official
        # models_opt_on_gpu / CPU optimizer-vars placement branches
        # are documented co-location semantics, kept below as
        # initialize_variables kwargs).

        input_ch=3
        resolution  = self.resolution = self.options['resolution']
        e_dims      = self.options['e_dims']
        ae_dims     = self.options['ae_dims']
        inter_dims  = self.inter_dims = self.options['inter_dims']
        inter_res   = self.inter_res = resolution // 32
        d_dims      = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        face_type   = self.face_type = {'f'    : FaceType.FULL,
                                        'wf'   : FaceType.WHOLE_FACE,
                                        'head' : FaceType.HEAD}[ self.options['face_type'] ]
        morph_factor = self.options['morph_factor']
        gan_power    = self.gan_power = self.options['gan_power']
        random_warp  = self.options['random_warp']

        blur_out_mask = self.options['blur_out_mask']

        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        use_fp16 = False
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        # torch (Phase 7, backend-neutral device model): the official
        # selection (Model_tf.py L258-259: models_opt_device =
        # tf_default_device_name if models_opt_on_gpu and is_training
        # else '/CPU:0') is expressed through the Phase 2 device
        # config — optimizer vars live on the selected (GPU) device
        # only when a device is selected, models_opt_on_gpu is set
        # and this is a training run; otherwise on CPU. The boolean
        # is kept as the Phase 3E2 initialize_variables
        # signature-parity kwarg (torch co-locates optimizer state
        # with its parameters on nn.device — documented placement
        # deviation, numerics unchanged).
        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        optimizer_vars_on_cpu = not (len(devices) != 0 and models_opt_on_gpu and self.is_training)

        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        # torch (Phase 7): the official nine CPU placeholders
        # (warped_src/dst, target_src/dst, target_srcm/dstm, _em,
        # morph_value_t — Model_tf.py L265-278) are removed — the
        # torch foundation passes tensors directly to the eager
        # forward paths (training: the closures below; inference:
        # the no-grad AE_view/AE_merge); the official
        # gpu_count * bs_per_gpu batch-size reconstruction
        # (L314-318) is the identity on the single-device
        # foundation (omitted, like the Phase 6A/6B ports).

        self.model_filename_list = []

        # Initializing model classes (Phase 7 torch archi factory —
        # the official AMP archi contract: same no-argument
        # constructors, names, the flat pixel-normalized encoder
        # codes, the single-dense inter, the RRC decoder heads)
        model_archi = nn.AMPArchi(resolution, use_fp16=use_fp16, e_ch=e_dims, ae_ch=ae_dims,
                                  inter_ch=inter_dims, inter_res=inter_res,
                                  d_ch=d_dims, d_mask_ch=d_mask_dims)

        self.encoder = model_archi.Encoder(name='encoder')
        self.inter_src = model_archi.Inter(name='inter_src')
        self.inter_dst = model_archi.Inter(name='inter_dst')
        self.decoder = model_archi.Decoder(name='decoder')

        self.model_filename_list += [   [self.encoder,  'encoder.npy'],
                                        [self.inter_src, 'inter_src.npy'],
                                        [self.inter_dst , 'inter_dst.npy'],
                                        [self.decoder , 'decoder.npy'] ]

        # torch (Phase 7, official optimizer-state naming): the
        # official DFL optimizer names its per-parameter state after
        # the trained variables ('ms_<full_varname>_0:0' /
        # 'vs_<full_varname>_0:0', where <full_varname> =
        # '<component>/<sub_name>:0', e.g.
        # 'ms_encoder/down1/conv1/weight_0:0'). The Phase 3E2
        # OptimizerBase emits exactly that naming from a per-
        # parameter binding (param._dfl_name; Tensor.name is
        # reserved/read-only in torch), so BEFORE
        # initialize_variables — which registers the state keys —
        # every component is bound to its full official DFL
        # variable name: the component's checkpoint scope
        # (component.name) + the official sub-name from its weight
        # enumeration, e.g. 'encoder/down1/conv1/weight:0'. The inter
        # components are bound (official sub-names) but NEVER
        # optimized — the official src_dst_opt trains encoder+decoder
        # only (Model_tf.py L301), and the official src_dst_opt.npy
        # state file accordingly contains no inter keys.
        def _bind_official_names(component):
            # the owning leaf layer per parameter (the module whose
            # DIRECT registrations include it) — the optimizer-state
            # layout hooks delegate a state tensor to this layer's
            # official layout rule (a state tensor has the layout of
            # the variable it tracks; OptimizerBase module
            # docstring, 'State LAYOUT')
            owners = {}
            for module in component.modules():
                for p in module.parameters(recurse=False):
                    owners[id(p)] = module
            for sub_name, param in component._iter_official_weights():
                param._dfl_name = f"{component.name}/{sub_name}"
                owner = owners.get(id(param))
                if owner is not None:
                    param._dfl_owner_layer = owner

        for _component in (self.encoder, self.inter_src, self.inter_dst, self.decoder):
            _bind_official_names(_component)

        # Official: the inter_src / inter_dst heads are FROZEN fixed
        # random projections — the official gradient requests cover
        # only the GAN weights (L454) and the G_weights set (L464:
        # encoder + decoder), so the inter variables never receive
        # any gradient. torch: requires_grad=False makes the inter
        # weights autograd constants — the encoder still receives
        # gradients THROUGH them (the TF back-prop-through-untrained-
        # variables behavior: the backprop path from the inter codes
        # to the encoder passes through the inter weight values as
        # constants) and their .grad stays None (finding F4). The
        # freeze does not affect save/load (data-only checkpoints).
        for _p in self.inter_src.get_weights() + self.inter_dst.get_weights():
            _p.requires_grad_(False)

        if self.is_training:
            # Initialize optimizers
            # (the official lr_dropout 'y' / 'cpu' values are
            # equivalent on the single-device torch foundation —
            # the official AMP model never passes the SAEHD
            # lr_dropout_on_cpu knob: its optimizer state is
            # co-located with its parameters on nn.device; Q3)
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0
            if self.options['lr_dropout'] in ['y','cpu']:
                lr_cos = 500
                lr_dropout = 0.3
            else:
                lr_cos = 0
                lr_dropout = 1.0
            self.G_weights = self.encoder.get_weights() + self.decoder.get_weights()

            self.src_dst_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables (self.G_weights, vars_on_cpu=optimizer_vars_on_cpu)
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

            if gan_power != 0:
                self.GAN = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], name="GAN")
                _bind_official_names(self.GAN)
                self.GAN_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                self.GAN_opt.initialize_variables ( self.GAN.get_weights(), vars_on_cpu=optimizer_vars_on_cpu)
                self.model_filename_list += [ [self.GAN, 'GAN.npy'],
                                              [self.GAN_opt, 'GAN_opt.npy'] ]

        def _to_tensor(x):
            # official feed_dict placement semantics: to the current
            # nn.device in the declared floatx (NumPy or tensor in)
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(np.ascontiguousarray(x))
            x = x.to(device=nn.device, dtype=nn.floatx)
            if x.dim() == 3:
                x = x[None, ...]
            return x

        def _to_numpy(x):
            # official caller contract: TF session outputs were NumPy
            # (in the graph's data format — model_data_format here)
            return x.detach().cpu().numpy()

        if self.is_training:
            def DLossOnes(logits):
                # official L333-334: the per-sample sigmoid BCE on
                # the ones labels (mean over the (1,2,3) axes -> the
                # per-sample (N,) vector) — the migrated
                # nn.sigmoid_cross_entropy is the verbatim formula
                return nn.sigmoid_cross_entropy(torch.ones_like(logits), logits)

            def DLossZeros(logits):
                # official L336-337: the per-sample sigmoid BCE on
                # the zeros labels
                return nn.sigmoid_cross_entropy(torch.zeros_like(logits), logits)

            def AE_forward(warped_src, warped_dst):
                # the official forward graph (Model_tf.py L354-376),
                # grad-capable: both inter heads on the src code
                # (L357), the dst code = inter_dst of the dst code
                # (L358/L368), the EXACT-k random morph mask
                # (L360-367: the per-sample uniform k-subset shuffle
                # of the k-ones/(D-k)-zeros base pattern,
                # stop_gradient — the Phase 7
                # core.leras.archis.AMP.exact_k_morph_mask op,
                # re-drawn on every call = the official fresh
                # tf.random.shuffle per session.run) and the two
                # training decoder calls (L374-375). The official
                # slice-code decoder call (L371-376) is pruned from
                # every training fetch by TF's lazy execution (the
                # fetched tensors never depend on it — the dst stack
                # consumes pred_dst_dst, not pred_src_dst); it lives
                # in AE_view / AE_merge below.
                warped_src = _to_tensor(warped_src)
                warped_dst = _to_tensor(warped_dst)
                src_code = self.encoder(warped_src)
                dst_code = self.encoder(warped_dst)

                src_inter_src_code = self.inter_src(src_code)
                src_inter_dst_code = self.inter_dst(src_code)
                dst_inter_dst_code = self.inter_dst(dst_code)

                inter_rnd_binomial = exact_k_morph_mask(src_code.shape[0], inter_dims,
                                                        morph_factor, device=nn.device, dtype=nn.floatx)
                morph_src_code = src_inter_src_code * inter_rnd_binomial + src_inter_dst_code * (1-inter_rnd_binomial)

                pred_src_src, pred_src_srcm = self.decoder(morph_src_code)
                pred_dst_dst, pred_dst_dstm = self.decoder(dst_inter_dst_code)

                return {
                    'dst_code': dst_code,
                    'pred_src_src': pred_src_src,
                    'pred_src_srcm': pred_src_srcm,
                    'pred_dst_dst': pred_dst_dst,
                    'pred_dst_dstm': pred_dst_dstm,
                }

            def AE_view(warped_src, warped_dst, morph_value):
                # official L512-514: the inference boundary — a
                # fresh session.run per call (the stochastic exact-k
                # mask is re-drawn per call), plus the official
                # DETERMINISTIC morph-value floor-slice
                # (L370-372: k = int(inter_dims * morph_value)
                # leading inter channels from the inter_src head,
                # the remainder from the inter_dst head) for the
                # SD/SDM tiles
                with torch.no_grad():
                    f = AE_forward(warped_src, warped_dst)

                    inter_dims_slice = int(inter_dims*morph_value)
                    dst_inter_src_code = self.inter_src(f['dst_code'])
                    dst_inter_dst_code = self.inter_dst(f['dst_code'])
                    src_dst_code = torch.cat( ( dst_inter_src_code[:, :inter_dims_slice],
                                                dst_inter_dst_code[:, inter_dims_slice:] ), dim=nn.conv2d_ch_axis )

                    pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)

                    return [ _to_numpy(x) for x in
                             (f['pred_src_src'], f['pred_dst_dst'], f['pred_dst_dstm'], pred_src_dst, pred_src_dstm) ]
                # torch: the official nn.tf_sess.run([...], feed_
                # dict=...) inference boundary — no gradient
                # bookkeeping

            self.AE_view = AE_view
        else:
            #Initializing merge function (the official non-training
            # branch, L519-534 — the dst code chain + the official
            # DETERMINISTIC morph-value floor slice; the morph value
            # comes from the merger prompt below)

            def AE_merge(warped_dst, morph_value):
                warped_dst = _to_tensor(warped_dst)
                with torch.no_grad():
                    dst_code = self.encoder(warped_dst)
                    dst_inter_src_code = self.inter_src(dst_code)
                    dst_inter_dst_code = self.inter_dst(dst_code)

                    inter_dims_slice = int(inter_dims*morph_value)
                    src_dst_code = torch.cat( ( dst_inter_src_code[:, :inter_dims_slice],
                                                dst_inter_dst_code[:, inter_dims_slice:] ), dim=nn.conv2d_ch_axis )

                    pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)
                    _, pred_dst_dstm = self.decoder(dst_inter_dst_code)

                    return [ _to_numpy(x) for x in
                             (pred_src_dst, pred_dst_dstm, pred_src_dstm) ]
                # torch: the official nn.tf_sess.run([...], feed_
                # dict=...) inference boundary — no gradient
                # bookkeeping

            self.AE_merge = AE_merge

        if self.is_training:
            # --- training closures (official loss stack + update
            # ops) ---
            # torch (Phase 7): the official per-GPU TF graph — the
            # loss stack (Model_tf.py L382-464), the per-closure
            # nn.gradients calls and the get_update_op update ops —
            # is reproduced as native eager torch with identical
            # semantics: the same tensor ops (the migrated
            # nn.dssim / nn.gaussian_blur / nn.total_variation_mse /
            # nn.sigmoid_cross_entropy), the same per-sample (N,)
            # loss vectors, the same per-optimizer gradient sets
            # (official nn.gradients(loss, vars) -> the backward
            # with the ones seed + the group's (grad, param) pairs;
            # the official TF nn.gradients on a vector loss seeds
            # the backward with ones = the batch SUM over samples,
            # probe-verified — a bare .backward() raises for N > 1;
            # the externals' batch-mean substitution is NOT
            # inherited), the same update order (the G step first,
            # then the D step with the post-update weights and a
            # fresh mask draw) and the single-device collapse of
            # the multi-GPU graph (one device; average_gv_list over
            # one tower is the identity).

            def _zero_grads(param_groups):
                # torch has no tf session boundary: the official
                # fresh nn.gradients per session.run means no .grad
                # may accumulate across update ops / iterations —
                # make that explicit per group.
                for group in param_groups:
                    for p in group:
                        p.grad = None

            def _prepare_targets(warped_src, target_src, target_srcm, target_srcm_em,
                                 warped_dst, target_dst, target_dstm, target_dstm_em):
                # the official per-tower input preparation (L382-
                # 414): the anti masks, the blur_out_mask target
                # rewrite (sigma = resolution/128, the element-wise
                # div-zero guard) and the softened loss-mask
                # products (L385-391, L406-409)
                warped_src = _to_tensor(warped_src)
                target_src = _to_tensor(target_src)
                target_srcm = _to_tensor(target_srcm)
                target_srcm_em = _to_tensor(target_srcm_em)
                warped_dst = _to_tensor(warped_dst)
                target_dst = _to_tensor(target_dst)
                target_dstm = _to_tensor(target_dstm)
                target_dstm_em = _to_tensor(target_dstm_em)

                target_srcm_anti = 1-target_srcm
                target_dstm_anti = 1-target_dstm

                if blur_out_mask:
                    # official L393-404: the div-zero guard
                    # tf.where(tf.equal(y, 0), ones, y) is ELEMENT-
                    # WISE (y == 0, not torch.equal's whole-tensor
                    # comparison)
                    sigma = resolution / 128

                    x = nn.gaussian_blur(target_src*target_srcm_anti, sigma)
                    y = 1-nn.gaussian_blur(target_srcm, sigma)
                    y = torch.where(y == 0, torch.ones_like(y), y)
                    target_src = target_src*target_srcm + (x/y)*target_srcm_anti

                    x = nn.gaussian_blur(target_dst*target_dstm_anti, sigma)
                    y = 1-nn.gaussian_blur(target_dstm, sigma)
                    y = torch.where(y == 0, torch.ones_like(y), y)
                    target_dst = target_dst*target_dstm + (x/y)*target_dstm_anti

                target_srcm_blur = torch.clip(nn.gaussian_blur(target_srcm,  max(1, resolution // 32) ), 0, 0.5) * 2
                target_dstm_blur = torch.clip(nn.gaussian_blur(target_dstm,  max(1, resolution // 32) ), 0, 0.5) * 2
                target_srcm_anti_blur = 1.0-target_srcm_blur
                target_dstm_anti_blur = 1.0-target_dstm_blur

                target_src_masked = target_src*target_srcm_blur
                target_dst_masked = target_dst*target_dstm_blur
                target_src_anti_masked = target_src*target_srcm_anti_blur
                target_dst_anti_masked = target_dst*target_dstm_anti_blur

                return {
                    'warped_src': warped_src, 'target_src': target_src,
                    'target_srcm': target_srcm, 'target_srcm_em': target_srcm_em,
                    'warped_dst': warped_dst, 'target_dst': target_dst,
                    'target_dstm': target_dstm, 'target_dstm_em': target_dstm_em,
                    'target_srcm_blur': target_srcm_blur,
                    'target_srcm_anti_blur': target_srcm_anti_blur,
                    'target_dstm_blur': target_dstm_blur,
                    'target_dstm_anti_blur': target_dstm_anti_blur,
                    'target_src_masked': target_src_masked,
                    'target_dst_masked': target_dst_masked,
                    'target_src_anti_masked': target_src_anti_masked,
                    'target_dst_anti_masked': target_dst_anti_masked,
                }

            def train(warped_src, target_src, target_srcm, target_srcm_em,
                      warped_dst, target_dst, target_dstm, target_dstm_em):
                # official train (L484-496): the full generator loss
                # stack (L417-462) -> ONE src_dst_opt step over the
                # official G_weights set (L301: encoder+decoder —
                # the inter-head and GAN grads computed by the
                # backward are dropped, exactly as
                # nn.gradients(gpu_G_loss, G_weights) requests only
                # the listed variables). Returns the official src /
                # dst loss vectors — the 5-term stacks snapshotted
                # BEFORE the background / GAN terms are added to
                # G_loss (L434-435), as the official
                # gpu_src_losses / gpu_dst_losses lists hold.
                t = _prepare_targets(warped_src, target_src, target_srcm, target_srcm_em,
                                     warped_dst, target_dst, target_dstm, target_dstm_em)

                # Phase 8: the AE forward runs under this plan's
                # autocast (in 'off' mode: nullcontext -> the exact
                # Phase 7 fp32 path); the boundary cast starts the
                # FP32 loss island — the loss stack, the GAN
                # forwards and the backward all see fp32 (in 'off'
                # mode .to is a no-op)
                with self._mp_autocast():
                    f = AE_forward(t['warped_src'], t['warped_dst'])
                f = { k: v.to(nn.floatx) for k, v in f.items() }
                pred_src_src = f['pred_src_src']
                pred_src_srcm = f['pred_src_srcm']
                pred_dst_dst = f['pred_dst_dst']
                pred_dst_dstm = f['pred_dst_dstm']

                pred_src_src_masked = pred_src_src*t['target_srcm_blur']
                pred_dst_dst_masked = pred_dst_dst*t['target_dstm_blur']
                pred_src_src_anti_masked = pred_src_src*t['target_srcm_anti_blur']
                pred_dst_dst_anti_masked = pred_dst_dst*t['target_dstm_anti_blur']

                target_src = t['target_src']
                target_dst = t['target_dst']
                target_srcm = t['target_srcm']
                target_srcm_em = t['target_srcm_em']
                target_dstm = t['target_dstm']
                target_dstm_em = t['target_dstm_em']

                # --- src loss (official L417-431) ---
                src_loss = torch.mean( 5*nn.dssim(t['target_src_masked'], pred_src_src_masked, max_val=1.0, filter_size=int(resolution/11.6)), dim=1)
                src_loss = src_loss + torch.mean( 5*nn.dssim(t['target_src_masked'], pred_src_src_masked, max_val=1.0, filter_size=int(resolution/23.2)), dim=1)
                src_loss = src_loss + torch.mean( 10*torch.square( t['target_src_masked'] - pred_src_src_masked ), dim=(1,2,3))
                src_loss = src_loss + torch.mean( 300*torch.abs( target_src*target_srcm_em - pred_src_src*target_srcm_em ), dim=(1,2,3))
                src_loss = src_loss + torch.mean( 10*torch.square( target_srcm - pred_src_srcm ), dim=(1,2,3) )

                # --- dst loss (official L418-432) ---
                dst_loss = torch.mean( 5*nn.dssim(t['target_dst_masked'], pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/11.6) ), dim=1)
                dst_loss = dst_loss + torch.mean( 5*nn.dssim(t['target_dst_masked'], pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/23.2) ), dim=1)
                dst_loss = dst_loss + torch.mean( 10*torch.square( t['target_dst_masked'] - pred_dst_dst_masked ), dim=(1,2,3))
                dst_loss = dst_loss + torch.mean( 300*torch.abs( target_dst*target_dstm_em - pred_dst_dst*target_dstm_em ), dim=(1,2,3))
                dst_loss = dst_loss + torch.mean( 10*torch.square( target_dstm - pred_dst_dstm ), dim=(1,2,3) )

                # the official gpu_src_losses / gpu_dst_losses
                # snapshot (L434-435) — train() returns these
                # vectors (the 5-term stacks, pre-GAN terms)
                src_loss_snap = src_loss.detach()
                dst_loss_snap = dst_loss.detach()

                G_loss = src_loss + dst_loss

                # dst-dst background weak loss (official L438-439)
                G_loss = G_loss + torch.mean(0.1*torch.square(pred_dst_dst_anti_masked-t['target_dst_anti_masked']), dim=(1,2,3))
                G_loss = G_loss + 0.000001*nn.total_variation_mse(pred_dst_dst_anti_masked)

                if gan_power != 0:
                    # official L443-446: the GAN forwards on the
                    # MASKED tensors; the 4-term GAN generator loss
                    # x gan_power UNNORMALIZED (official L456-458);
                    # the target-D forwards belong to the D
                    # closure's own graph branch — not part of G
                    # loss
                    pred_src_src_d, pred_src_src_d2 = self.GAN(pred_src_src_masked)
                    pred_dst_dst_d, pred_dst_dst_d2 = self.GAN(pred_dst_dst_masked)

                    G_loss = G_loss + (DLossOnes(pred_src_src_d) + DLossOnes(pred_src_src_d2) + \
                                       DLossOnes(pred_dst_dst_d) + DLossOnes(pred_dst_dst_d2)) * gan_power

                    # Minimal src-src-bg rec with total_variation_
                    # mse to suppress random bright dots from gan
                    # (official L460-462)
                    G_loss = G_loss + 0.000001*nn.total_variation_mse(pred_src_src)
                    G_loss = G_loss + 0.02*torch.mean(torch.square(pred_src_src_anti_masked-t['target_src_anti_masked']), dim=(1,2,3))

                _zero_grads([self.G_weights])
                # Q11 (probe-verified): the official TF
                # nn.gradients(gpu_G_loss, G_weights) on the
                # per-sample (N,) loss vector seeds the backward
                # with ones — the batch SUM over samples
                # Phase 8: the scaled/FP32 explicit-grad backward
                # (official nn.gradients = the batch SUM over the
                # per-sample (N,) loss vector), then the native
                # fp16 unscale + the overflow-aware update of the
                # official update op (in 'off' mode: the exact
                # Phase 7 code path — plain backward + direct
                # update op)
                self._mp_backward(G_loss)
                self._mp_unscale_opt(self.src_dst_opt)
                self._mp_opt_step(
                    self.src_dst_opt,
                    [ (p.grad, p) for p in self.G_weights ])
                self._mp_scaler_update()

                return ( src_loss_snap, dst_loss_snap )

            self.train = train

            if gan_power != 0:
                def GAN_train(warped_src, target_src, target_srcm, target_srcm_em,
                              warped_dst, target_dst, target_dstm, target_dstm_em):
                    # official GAN_train (L499-510): the D step on
                    # the POST-src_dst-step weights — a fresh graph
                    # evaluation (a fresh exact-k mask draw) whose
                    # generator recompute is gradient-free (the
                    # official TF gradients are requested only wrt
                    # the GAN weights, L454), the 8-term D loss x
                    # (1/8) (official L448-452 — the 1/8 factor is
                    # the OFFICIAL upstream normalization,
                    # preserved; the gradient over the batch is the
                    # SUM via the ones seed, Q11) -> ONE GAN_opt
                    # step over the GAN weights (the official
                    # GAN_opt state is the FULL per-parameter state
                    # — iters / ms_ / vs_ — never filtered by a
                    # parameter-name prefix)
                    t = _prepare_targets(warped_src, target_src, target_srcm, target_srcm_em,
                                         warped_dst, target_dst, target_dstm, target_dstm_em)

                    # Phase 8: the GAN step is ENTIRELY fp32 in
                    # every precision mode (the official GAN never
                    # receives an fp16 treatment) — no autocast
                    # region here; its backward/update also never
                    # touch the shared fp16 scaler
                    with torch.no_grad():
                        f = AE_forward(t['warped_src'], t['warped_dst'])

                    pred_src_src_masked = f['pred_src_src']*t['target_srcm_blur']
                    pred_dst_dst_masked = f['pred_dst_dst']*t['target_dstm_blur']

                    # the GAN forwards on the MASKED tensors
                    # (official L443-446), now gradient-capable wrt
                    # the GAN weights only (the preds / targets are
                    # constants from the no_grad recompute)
                    pred_src_src_d, pred_src_src_d2 = self.GAN(pred_src_src_masked)
                    pred_dst_dst_d, pred_dst_dst_d2 = self.GAN(pred_dst_dst_masked)
                    target_src_d, target_src_d2 = self.GAN(t['target_src_masked'])
                    target_dst_d, target_dst_d2 = self.GAN(t['target_dst_masked'])

                    GAN_loss = (DLossOnes (target_src_d)   + DLossOnes (target_src_d2) + \
                                DLossZeros(pred_src_src_d) + DLossZeros(pred_src_src_d2) + \
                                DLossOnes (target_dst_d)   + DLossOnes (target_dst_d2) + \
                                DLossZeros(pred_dst_dst_d) + DLossZeros(pred_dst_dst_d2)
                                ) * (1.0 / 8)

                    # the G step's backward left grads on the GAN
                    # weights (they feed the G loss via the
                    # generator GAN terms) — the official fresh
                    # nn.gradients per session.run means they must
                    # not leak into the D step
                    _zero_grads([self.GAN.get_weights()])
                    torch.autograd.backward(GAN_loss, torch.ones_like(GAN_loss))
                    self.GAN_opt.get_update_op(
                        [ (p.grad, p) for p in self.GAN.get_weights() ])()

                self.GAN_train = GAN_train

        # Loading/initializing all models/optimizers weights (the
        # official 536-546 loop; Phase 4/5 strict policy: a missing
        # required component file on resume fails explicitly instead
        # of the official silent re-initialization — the OFFICIAL
        # intentional re-init rule below is preserved. GAN_opt joins
        # the GAN in the official gan_model_changed re-init set: the
        # official loop only names the GAN, but the stale GAN_opt
        # file then fails the official `do_init = not load_weights(
        # ...)` fallback and the official silently re-initializes
        # it — under the strict load policy (no fallback) the same
        # official outcome (a changed discriminator gets a FRESH
        # optimizer state) is preserved by re-initializing GAN_opt
        # explicitly (the Phase 6B SAEHD precedent))
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()
            if self.is_training and gan_power != 0 and model in (self.GAN, self.GAN_opt):
                if self.gan_model_changed:
                    do_init = True
            if not do_init:
                if not self.is_first_run():
                    file_path = self.get_strpath_storage_for_file(filename)
                    import os
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(
                            f"required component file missing on resume: {file_path}")
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()

        ###############

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path
            training_data_dst_path = self.training_data_dst_path

            random_ct_samples_path=training_data_dst_path if ct_mode is not None else None

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,   'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],
                        generators_count=dst_generators_count )
                         ])

    def export_dfm (self):
        raise NotImplementedError(
            "AMP DFM/ONNX export is out of scope for Phase 7 (the "
            "official export_dfm is a TF/tf2onnx graph export — the "
            "ONNX/DFM exclusion); the torch AE_merge path serves the "
            "merger and the predictor")

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
               (io.is_colab() and self.iter % 100 == 0)

    #override
    def onTrainOneIter(self):
        # torch (Phase 7): the official onTrainOneIter (Model_tf.py
        # L646-657) — the sample fetch, the always-run generator
        # step (the official train closure) and the GAN step (the
        # official two-phase post-step D on the post-update weights),
        # and the official two-value loss return (the per-sample
        # vector means). The train closures are built in
        # on_initialize (above), mirroring the official
        # graph-construction location.
        bs = self.get_batch_size()
        # official L647 (dead in the official code too: the batch
        # size drives the generators, not the closures) — kept for
        # fidelity (F9)

        ( (warped_src, target_src, target_srcm, target_srcm_em), \
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = self.generate_next_samples()

        # gradient hygiene: the official fresh nn.gradients per
        # session.run means no .grad may survive into this
        # iteration — zero every graph group before the first
        # backward (the GAN closure additionally re-zeros its own
        # group before its backward, discarding the stale GAN grads
        # accumulated by the G-loss backward)
        grad_groups = [self.G_weights, self.inter_src.get_weights(), self.inter_dst.get_weights()]
        if self.gan_power != 0:
            grad_groups.append(self.GAN.get_weights())
        for group in grad_groups:
            for p in group:
                p.grad = None

        src_loss, dst_loss = self.train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.gan_power != 0:
            self.GAN_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        # the official returns the per-sample means as plain floats
        # (np.mean of the fetched numpy vectors) — same values;
        # .detach() first, no torch grad-conversion warning
        return ( ('src_loss', float(src_loss.mean().detach()) ),
                 ('dst_loss', float(dst_loss.mean().detach()) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        # torch (Phase 7): the official onGetPreview (Model_tf.py
        # L660-705) — AE_view fed with the UNWARPED targets, the
        # official NHWC preview strips, the official 3 named strips
        # x 2 rows x 3 tiles (the DDM masks 3-channel repeated), the
        # official one-sample rendering (n_samples bounds only the
        # RANDOM index; for_history fixes it to 0) and the
        # resolution-invariant layout (no SAEHD-style branch)
        ( (warped_src, target_src, target_srcm, target_srcm_em),
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = samples

        S, D, SS, DD, DDM_000, _, _ = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst, 0.0)  ) ]

        _, _, DDM_025, SD_025, SDM_025 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.25) ]
        _, _, DDM_050, SD_050, SDM_050 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.50) ]
        _, _, DDM_065, SD_065, SDM_065 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.65) ]
        _, _, DDM_075, SD_075, SDM_075 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.75) ]
        _, _, DDM_100, SD_100, SDM_100 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 1.00) ]

        (DDM_000,
         DDM_025, SDM_025,
         DDM_050, SDM_050,
         DDM_065, SDM_065,
         DDM_075, SDM_075,
         DDM_100, SDM_100) = [ np.repeat (x, (3,), -1) for x in (DDM_000,
                                                                 DDM_025, SDM_025,
                                                                 DDM_050, SDM_050,
                                                                 DDM_065, SDM_065,
                                                                 DDM_075, SDM_075,
                                                                 DDM_100, SDM_100) ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]
        # official L684 (dead in the official code too — the masks
        # are never used in the strips) — kept for fidelity

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        result = []

        i = np.random.randint(n_samples) if not for_history else 0

        st =  [ np.concatenate ((S[i],  D[i],  DD[i]*DDM_000[i]), axis=1) ]
        st += [ np.concatenate ((SS[i], DD[i], SD_100[i] ), axis=1) ]

        result += [ ('AMP morph 1.0', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i],  SD_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i], SD_075[i], SD_100[i]), axis=1) ]
        result += [ ('AMP morph list', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i]*DDM_025[i]*SDM_025[i],  SD_050[i]*DDM_050[i]*SDM_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i]*DDM_065[i]*SDM_065[i], SD_075[i]*DDM_075[i]*SDM_075[i], SD_100[i]*DDM_100[i]*SDM_100[i]), axis=1) ]
        result += [ ('AMP morph list masked', np.concatenate (st, axis=0 )), ]

        return result

    def predictor_func (self, face, morph_value):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face, morph_value) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        morph_factor = np.clip ( io.input_number ("Morph factor", 1.0, add_info="0.0 .. 1.0"), 0.0, 1.0 )

        def predictor_morph(face):
            return self.predictor_func(face, morph_factor)

        import merger
        return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = AMPModel
