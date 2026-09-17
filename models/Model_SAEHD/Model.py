"""SAEHD — official DeepFaceLab SAEHD model on the torch foundation
(Phase 6A: structural skeleton, wiring, compatibility foundation).

The official TF source is preserved verbatim in ``Model_tf.py`` (dead
reference, never imported — project convention, like
``ModelBase_tf.py`` / ``discriminators_tf.py``).

Official behavior preserved (structure/wiring layer):
- ``on_initialize_options``: the complete official option set and the
  official first-run/override prompts (resolution/face_type/archi
  validation loop with the official ``u``/``d``/``t``/``c`` modifier
  rules, the official dimension clipping + even rounding, pretrain
  coupling, ``gan_model_changed`` / ``pretrain_just_disabled``
  detection) — unchanged;
- ``on_initialize``: the official structural portion — device/data
  format, archi parsing, component construction through the Phase 3F
  ``nn.DeepFakeArchi`` factory (official Encoder/Inter/Decoder
  constructors and names), the official ``nn.CodeDiscriminator`` /
  ``nn.UNetPatchDiscriminator`` (Phase 3F) creation rules, the
  official three-optimizer construction (AdaBelief/RMSprop, Phase
  3E2) with the official ``lr=5e-5`` / ``lr_dropout``-``lr_cos=500``
  coupling / ``clipgrad`` -> clipnorm rules and the official
  ``src_dst_trainable_weights`` architecture/random-warp rules, the
  official ``model_filename_list`` (official ``.npy`` filenames,
  ``[model, file]`` pairs for components and ``(opt, file)`` tuples
  for optimizers), the official 637-657 component load/init loop
  (official ``pretrain_just_disabled`` inter re-init rule and
  ``gan_model_changed`` D_src re-init rule kept), and the official
  sample-generator wiring (migrated samplelib);
- ``AE_merge`` / ``AE_view``: the official inference functions. The
  official TF session-run graph is replaced by an eager
  ``torch.no_grad`` forward pass over the same component chain (the
  official inference boundary — no gradient bookkeeping; NumPy in
  ``model_data_format``, NumPy out, exactly like the official
  ``tf_sess.run`` caller contract used by ``predictor_func`` / the
  merger);
- ``predictor_func`` / ``get_MergerConfig``: official, unchanged
  (the ``merger`` package is importable under the torch foundation);
- ``get_model_filename_list`` / ``onSave`` / ``should_save_preview_
  history``: official, unchanged (component saves go through the
  Phase 4 Saveable engine — official raw pickle protocol-4
  ``.npy`` streams);
- the official top-level lifecycle (Phase 5 ``models/ModelBase.py``)
  is used unchanged: data.dat-gated resume, the two distinct
  iteration counters, the 24-slot autobackup ring.

Documented torch adaptations (Phase 6A):
- no TF placeholders / session graph: the eight official
  placeholders (warped_src/dst, target_src/dst, target_srcm/dstm,
  _em) are removed — the torch foundation feeds tensors directly and
  executes eagerly on ``nn.device`` (the Phase 2 device model);
- the official ``tf.device`` placement contexts (models on GPU /
  optimizer vars on CPU) are replaced by the Phase 2 co-location
  semantics: everything lives on ``nn.device``. The official
  ``models_opt_on_gpu`` option is kept and the official
  ``optimizer_vars_on_cpu`` computation is preserved as a
  signature-parity kwarg to the Phase 3E2
  ``initialize_variables`` (documented placement deviation,
  numerics unchanged — Phase 3E2/4/5 pattern);
- the official archi validation is an interactive re-prompt loop in
  ``on_initialize_options``; a headless torch run has no re-prompt,
  so ``on_initialize`` additionally VALIDATES the stored archi
  string explicitly (invalid base / modifier / empty opts / >2
  parts -> ``ValueError``). This hardening cannot change the
  behavior of any valid official configuration;
- the official silent re-initialization on a MISSING component file
  (``do_init = not model.load_weights(...)``) is replaced by the
  Phase 4/5 strict policy: a missing required file on resume fails
  with an explicit ``FileNotFoundError`` (all-or-nothing; a corrupt
  file already fails strictly in the Phase 4 engine). The OFFICIAL
  intentional re-init rules (``pretrain_just_disabled`` -> inter(s),
  ``gan_model_changed`` -> D_src) are preserved exactly;
- ``export_dfm`` (DFM/ONNX via tf2onnx) is Phase 11 — the method
  exists and fails explicitly;
- ``onTrainOneIter`` and the official train closures
  (``src_dst_train`` / ``D_train`` / ``D_src_dst_train``) plus the
  per-GPU loss stack (official L354-530) are Phase 6B (native torch
  autograd with the official reduction semantics) — 6A provides
  explicit deferral stubs that fail loudly instead of silently
  mis-training; ``onGetPreview`` (preview rendering) is likewise
  deferred (the base-class default is kept).

No TensorFlow import appears on this path; no direct CUDA-backend
calls or CUDA device-string literals (device handling through the
Phase 2 abstraction); components/discriminators/optimizers come from
the migrated Phases 3F/3E2 foundations (nothing duplicated here).
"""

import multiprocessing

import numpy as np
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


class SAEHDModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()

        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb

        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        yn_str = {True:'y',False:'n'}
        min_res = 64
        max_res = 640

        #default_usefp16            = self.options['use_fp16']           = self.load_or_def_option('use_fp16', False)
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 128)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_archi              = self.options['archi']              = self.load_or_def_option('archi', 'liae-ud')

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)
        default_eyes_mouth_prio    = self.options['eyes_mouth_prio']    = self.load_or_def_option('eyes_mouth_prio', False)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)

        default_adabelief          = self.options['adabelief']          = self.load_or_def_option('adabelief', True)

        lr_dropout = self.load_or_def_option('lr_dropout', 'n')
        lr_dropout = {True:'y', False:'n'}.get(lr_dropout, lr_dropout) #backward comp
        default_lr_dropout         = self.options['lr_dropout'] = lr_dropout

        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_random_hsv_power   = self.options['random_hsv_power']   = self.load_or_def_option('random_hsv_power', 0.0)
        default_true_face_power    = self.options['true_face_power']    = self.load_or_def_option('true_face_power', 0.0)
        default_face_style_power   = self.options['face_style_power']   = self.load_or_def_option('face_style_power', 0.0)
        default_bg_style_power     = self.options['bg_style_power']     = self.load_or_def_option('bg_style_power', 0.0)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(suggest_batch_size)
            #self.options['use_fp16'] = io.input_bool ("Use fp16", default_usefp16, help_message='Increases training/inference speed, reduces model size. Model may crash. Enable it after 1-5k iters.')

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi.")
            resolution = np.clip ( (resolution // 16) * 16, min_res, max_res)
            self.options['resolution'] = resolution



            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()

            while True:
                archi = io.input_str ("AE architecture", default_archi, help_message=\
"""
'df' keeps more identity-preserved face.
'liae' can fix overly different face shapes.
'-u' increased likeness of the face.
'-d' (experimental) doubling the resolution using the same computation cost.
Examples: df, liae, df-d, df-ud, liae-ud, ...
""").lower()

                archi_split = archi.split('-')

                if len(archi_split) == 2:
                    archi_type, archi_opts = archi_split
                elif len(archi_split) == 1:
                    archi_type, archi_opts = archi_split[0], None
                else:
                    continue

                if archi_type not in ['df', 'liae']:
                    continue

                if archi_opts is not None:
                    if len(archi_opts) == 0:
                        continue
                    if len([ 1 for opt in archi_opts if opt not in ['u','d','t','c'] ]) != 0:
                        continue

                    if 'd' in archi_opts:
                        self.options['resolution'] = np.clip ( (self.options['resolution'] // 32) * 32, min_res, max_res)

                break
            self.options['archi'] = archi

        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head':
                self.options['masked_training']  = io.input_bool ("Masked training", default_masked_training, help_message="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.")

            self.options['eyes_mouth_prio'] = io.input_bool ("Eyes and mouth priority", default_eyes_mouth_prio, help_message='Helps to fix eye problems during training like "alien eyes" and wrong eyes direction. Also makes the detail of the teeth higher.')
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['adabelief'] = io.input_bool ("Use AdaBelief optimizer?", default_adabelief, help_message="Use AdaBelief optimizer. It requires more VRAM, but the accuracy and the generalization of the model is higher.")

            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['random_hsv_power'] = np.clip ( io.input_number ("Random hue/saturation/light intensity", default_random_hsv_power, add_info="0.0-0.3", help_message="Adds random hue/saturation/light value intensity. Good to increase dataset diversity. If too high, it can make skin look weird or even change skin tone." ), 0.0, 0.3 )

            # Official Model_tf.py L158-178 prompt order (part of the
            # official interactive override flow): gan_power (with the
            # conditional gan_patch_size / gan_dims prompts) first, then
            # the true-face / style prompts, ct_mode, clipgrad, and
            # finally the pretrain prompt.
            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0-5.0", help_message="If enabled, can improve skin and eyes quality a lot. It requires more VRAM, time and iterations to train. Enable it after at least 20k iters of training without GAN. If it makes your model worse, disable it back." ), 0.0, 5.0 )
            if self.options['gan_power'] != 0.0:
                self.options['gan_patch_size'] = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="Patch size for GAN discriminator." ), 3, 640 )
                self.options['gan_dims'] = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-512", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 4, 512 )

            self.options['true_face_power'] = np.clip ( io.input_number ("Train true face recognition", default_true_face_power, add_info="0.0-1.0", help_message="Trains face recognition network to preserve identity better. If it doesn't help your model, then turn it off." ), 0.0, 1.0 )
            if self.options['true_face_power'] != 0.0 and 'df' not in self.options['archi']:
                io.log_info("Warning: true face recognition is available only for 'df' architecture. Setting it to 0.0.")
                self.options['true_face_power'] = 0.0

            self.options['face_style_power'] = np.clip ( io.input_number ("Face style loss power", default_face_style_power, add_info="0-100", help_message="Trains the model to keep the dst face style (skin texture, face color) in the swapped face. If it doesn't help your model, then turn it off."), 0.0, 100.0 )

            self.options['bg_style_power'] = np.clip ( io.input_number ("Background style loss power", default_bg_style_power, add_info="0-100", help_message="Trains the model to keep the dst background style in the swapped face. If it doesn't help your model, then turn it off."), 0.0, 100.0 )

            # official Model_tf.py L175 prompt identity: label, the valid
            # list ['none','rct','lct','mkl','idt','sot'] (input_str falls
            # back to the default for a value outside the list) and help
            # text — the migrated color_transfer implements all five
            # non-'none' modes
            self.options['ct_mode'] = io.input_str ("Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. Try all modes to find the best.")

            self.options['clipgrad'] = io.input_bool ("Clip gradient", default_clipgrad, help_message="This option clips the gradient to prevent the optimizer from taking too big steps. It can help to stabilize training, but it can also slow it down.")

            # official Model_tf.py L178 — missing in the original port:
            # without this prompt the pretrain_just_disabled detection
            # (official L185) can never trigger on the override path
            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain, help_message="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly. Forces random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y")

        # official Model_tf.py L180-185, verbatim semantics: the pretrain
        # data-path guard and the two detection flags, both assigned as
        # UNCONDITIONAL booleans (the official code never leaves
        # pretrain_just_disabled unset — on_initialize references both
        # flags on every resume path):
        #   gan_model_changed — D_src archi (patch size / dims) differs
        #     from the stored defaults -> the official 637-657 loop
        #     re-initializes D_src;
        #   pretrain_just_disabled — pretrain was on in the stored
        #     defaults and is now off -> the official re-init rule for
        #     the inter components + set_iter(0).
        if self.options['pretrain'] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)

        self.pretrain = self.options['pretrain']

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        # torch (Phase 6A): the official `tf = nn.tf` import, the TF
        # placeholders and the `tf.device` placement contexts are
        # removed — the torch foundation executes eagerly on nn.device
        # (Phase 2 device model; the official models_on_gpu / CPU
        # optimizer-vars placement branches are documented co-location
        # semantics, kept below as initialize_variables kwargs).

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]

        if 'eyes_prio' in self.options:
            self.options.pop('eyes_prio')

        eyes_mouth_prio = self.options['eyes_mouth_prio']

        # torch (Phase 6A, documented hardening): the official code
        # validates the archi string only through the interactive
        # re-prompt loop above; a headless torch run has no re-prompt,
        # so a malformed stored archi must fail explicitly here.
        archi = self.options['archi']
        archi_split = archi.split('-')
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        else:
            raise ValueError(
                f"invalid SAEHD archi {archi!r}: expected 'df' or 'liae' "
                "with an optional '-<opts>' suffix (subset of u/d/t/c)")
        if archi_type not in ('df', 'liae'):
            raise ValueError(
                f"invalid SAEHD archi {archi!r}: base must be 'df' or 'liae'")
        if archi_opts is not None:
            if len(archi_opts) == 0:
                raise ValueError(
                    f"invalid SAEHD archi {archi!r}: empty modifier list")
            if len([1 for opt in archi_opts if opt not in ('u','d','t','c')]) != 0:
                raise ValueError(
                    f"invalid SAEHD archi {archi!r}: modifiers must be a "
                    "subset of u/d/t/c")

        self.archi_type = archi_type

        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)

        adabelief = self.options['adabelief']

        use_fp16 = False
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        self.gan_power = gan_power = 0.0 if self.pretrain else self.options['gan_power']
        random_warp = False if self.pretrain else self.options['random_warp']
        random_src_flip = self.random_src_flip if not self.pretrain else True
        random_dst_flip = self.random_dst_flip if not self.pretrain else True
        random_hsv_power = self.options['random_hsv_power'] if not self.pretrain else 0.0
        blur_out_mask = self.options['blur_out_mask']

        if self.pretrain:
            self.options_show_override['lr_dropout'] = 'n'
            self.options_show_override['random_warp'] = False
            self.options_show_override['gan_power'] = 0.0
            self.options_show_override['random_hsv_power'] = 0.0
            self.options_show_override['face_style_power'] = 0.0
            self.options_show_override['bg_style_power'] = 0.0
            self.options_show_override['uniform_yaw'] = True

        masked_training = self.options['masked_training']
        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None


        # torch (Phase 6A, backend-neutral device model): the official
        # selection (Model_tf.py L252-254: models_opt_device =
        # tf_default_device_name if models_opt_on_gpu and is_training
        # else '/CPU:0') is expressed through the Phase 2 device config —
        # optimizer vars live on the selected (GPU) device only when a
        # device is selected, models_opt_on_gpu is set and this is a
        # training run; otherwise on CPU. The boolean is kept as the
        # Phase 3E2 initialize_variables signature-parity kwarg (torch
        # co-locates optimizer state with its parameters on nn.device —
        # documented placement deviation, numerics unchanged).
        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        optimizer_vars_on_cpu = not (len(devices) != 0 and models_opt_on_gpu and self.is_training)

        input_ch=3
        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        # torch (Phase 6A): the official eight CPU placeholders
        # (warped_src/dst, target_src/dst, target_srcm/dstm, _em) are
        # removed — the torch foundation passes tensors directly to the
        # eager forward paths (training: Phase 6B; inference: the
        # no-grad AE_merge/AE_view below).

        self.model_filename_list = []

        # Initializing model classes (Phase 3F torch archi factory — the
        # official DeepFakeArchi contract: same constructors, names,
        # u/d/t/c modifier semantics, R-R-C depth_to_space)
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=use_fp16, opts=archi_opts)

        if 'df' in archi_type:
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

            self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.get_out_ch()

            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
            self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                if self.options['true_face_power'] != 0:
                    self.code_discriminator = nn.CodeDiscriminator(ae_dims, code_res=self.inter.get_out_res(), name='dis' )
                    self.model_filename_list += [ [self.code_discriminator, 'code_discriminator.npy'] ]

        elif 'liae' in archi_type:
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

            self.inter_AB = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_AB')
            self.inter_B  = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_B')

            inter_out_ch = self.inter_AB.get_out_ch()
            inters_out_ch = inter_out_ch*2
            self.decoder = model_archi.Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder')

            self.model_filename_list += [ [self.encoder,  'encoder.npy'],
                                          [self.inter_AB, 'inter_AB.npy'],
                                          [self.inter_B , 'inter_B.npy'],
                                          [self.decoder , 'decoder.npy'] ]

        if self.is_training:
            if gan_power != 0:
                self.D_src = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], name="D_src")
                self.model_filename_list += [ [self.D_src, 'GAN.npy'] ]

            # Initialize optimizers (Phase 3E2 torch optimizers — the
            # official AdaBelief/RMSprop semantics: lr_cos/lr_dropout
            # coupling, clipnorm, iters state)
            lr=5e-5
            if self.options['lr_dropout'] in ['y','cpu'] and not self.pretrain:
                lr_cos = 500
                lr_dropout = 0.3
            else:
                lr_cos = 0
                lr_dropout = 1.0
            OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0

            if 'df' in archi_type:
                self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                self.src_dst_trainable_weights = self.src_dst_saveable_weights
            elif 'liae' in archi_type:
                self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter_AB.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()
                if random_warp:
                    self.src_dst_trainable_weights = self.src_dst_saveable_weights
                else:
                    self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()

            # torch (Phase 6A, official optimizer-state naming): the
            # official DFL optimizer names its per-parameter state after
            # the trained variables ('ms_<full_varname>_0:0' /
            # 'vs_<full_varname>_0:0', where <full_varname> =
            # '<component>/<sub_name>:0', e.g.
            # 'ms_encoder/down1/conv1/weight_0:0'). The Phase 3E2
            # OptimizerBase emits exactly that naming from a per-
            # parameter binding (param._dfl_name; Tensor.name is
            # reserved/read-only in torch), so BEFORE
            # initialize_variables — which registers the state keys —
            # every optimized parameter is bound to its full official
            # DFL variable name: the component's checkpoint scope
            # (component.name) + the official sub-name from its weight
            # enumeration, e.g. 'encoder/down1/conv1/weight:0'.
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

            if 'df' in archi_type:
                for _component in (self.encoder, self.inter,
                                   self.decoder_src, self.decoder_dst):
                    _bind_official_names(_component)
            else:
                for _component in (self.encoder, self.inter_AB,
                                   self.inter_B, self.decoder):
                    _bind_official_names(_component)
            if self.options['true_face_power'] != 0:
                _bind_official_names(self.code_discriminator)
            if gan_power != 0:
                _bind_official_names(self.D_src)

            self.src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables (self.src_dst_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

            if self.options['true_face_power'] != 0:
                self.D_code_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='D_code_opt')
                self.D_code_opt.initialize_variables ( self.code_discriminator.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                self.model_filename_list += [ (self.D_code_opt, 'D_code_opt.npy') ]

            if gan_power != 0:
                self.D_src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                self.D_src_dst_opt.initialize_variables ( self.D_src.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                self.model_filename_list += [ (self.D_src_dst_opt, 'GAN_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

        # torch (Phase 6A): the official per-GPU TF graph (loss stack,
        # nn.gradients, nn.average_gv_list, the src_dst_train /
        # D_train / D_src_dst_train session closures) is Phase 6B —
        # native torch autograd with the official reduction semantics.
        # The inference closures (AE_view / AE_merge) are implemented
        # below as no-grad eager forward passes over the same component
        # chain (the official inference boundary).

        # --- inference closures (official AE_view / AE_merge) ---------

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
            def AE_view(warped_src, warped_dst):
                warped_src = _to_tensor(warped_src)
                warped_dst = _to_tensor(warped_dst)
                with torch.no_grad():
                    if 'df' in archi_type:
                        src_code = self.inter(self.encoder(warped_src))
                        dst_code = self.inter(self.encoder(warped_dst))
                        pred_src_src, pred_src_srcm = self.decoder_src(src_code)
                        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
                        pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
                    elif 'liae' in archi_type:
                        src_code = self.encoder(warped_src)
                        src_inter_AB_code = self.inter_AB(src_code)
                        src_code = torch.concat([src_inter_AB_code, src_inter_AB_code], dim=nn.conv2d_ch_axis)
                        dst_code = self.encoder(warped_dst)
                        dst_inter_B_code = self.inter_B(dst_code)
                        dst_inter_AB_code = self.inter_AB(dst_code)
                        dst_code = torch.concat([dst_inter_B_code, dst_inter_AB_code], dim=nn.conv2d_ch_axis)
                        src_dst_code = torch.concat([dst_inter_AB_code, dst_inter_AB_code], dim=nn.conv2d_ch_axis)

                        pred_src_src, pred_src_srcm = self.decoder(src_code)
                        pred_dst_dst, pred_dst_dstm = self.decoder(dst_code)
                        pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)

                    return [ _to_numpy(x) for x in
                             (pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm) ]
                # torch: the official nn.tf_sess.run([...], feed_dict=...)
                # inference boundary — no gradient bookkeeping
            self.AE_view = AE_view
        else:
            # Initializing merge function (official non-training branch)
            def AE_merge(warped_dst):
                warped_dst = _to_tensor(warped_dst)
                with torch.no_grad():
                    if 'df' in archi_type:
                        dst_code = self.inter(self.encoder(warped_dst))
                        pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
                        _, pred_dst_dstm = self.decoder_dst(dst_code)

                    elif 'liae' in archi_type:
                        dst_code = self.encoder(warped_dst)
                        dst_inter_B_code = self.inter_B(dst_code)
                        dst_inter_AB_code = self.inter_AB(dst_code)
                        dst_code = torch.concat([dst_inter_B_code, dst_inter_AB_code], dim=nn.conv2d_ch_axis)
                        src_dst_code = torch.concat([dst_inter_AB_code, dst_inter_AB_code], dim=nn.conv2d_ch_axis)

                        pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)
                        _, pred_dst_dstm = self.decoder(dst_code)

                    return [ _to_numpy(x) for x in
                             (pred_src_dst, pred_dst_dstm, pred_src_dstm) ]
                # torch: the official nn.tf_sess.run([...], feed_dict=...)
                # inference boundary — no gradient bookkeeping

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights (the
        # official 637-657 loop; Phase 4/5 strict policy: a missing
        # required component file on resume fails explicitly instead of
        # the official silent re-initialization — the OFFICIAL
        # intentional re-init rules below are preserved. GAN_opt joins
        # D_src in the official gan_model_changed re-init set: the
        # official loop only names D_src, but the stale GAN_opt file
        # then fails the official `do_init = not load_weights(...)`
        # fallback and the official silently re-initializes it — under
        # the strict load policy (no fallback) the same official
        # outcome (a changed discriminator gets a FRESH optimizer
        # state) is preserved by re-initializing GAN_opt explicitly)
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if 'df' in archi_type:
                    if model == self.inter:
                        do_init = True
                elif 'liae' in archi_type:
                    if model == self.inter_AB or model == self.inter_B:
                        do_init = True
            else:
                do_init = self.is_first_run()
                if self.is_training and gan_power != 0 and model in (self.D_src, self.D_src_dst_opt):
                    # the optimizer's official display name is 'GAN_opt'
                    # (its file is GAN_opt.npy); its Python attribute is
                    # D_src_dst_opt (official naming)
                    if self.gan_model_changed:
                        do_init = True

            if not do_init:
                if not model.load_weights( self.get_strpath_storage_for_file(filename) ):
                    if not self.is_first_run():
                        # Phase 4/5 strict policy (documented): the
                        # official `do_init = not load_weights(...)`
                        # silent re-initialization on a missing file is
                        # NOT reproduced on resume
                        raise FileNotFoundError(
                            f"required component file missing on resume: "
                            f"{self.get_strpath_storage_for_file(filename)}")
                    do_init = True

            if do_init:
                model.init_weights()

        ###############

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None and not self.pretrain else None

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,   'random_hsv_shift_amount' : random_hsv_power,                                        'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=dst_generators_count ),
                ])

        ###############

    #override
    def onTrainOneIter(self):
        # Phase 6A boundary: the official training step (the loss stack,
        # nn.gradients, the src_dst_train/D_train/D_src_dst_train
        # update closures — official L354-606) is Phase 6B (native torch
        # autograd with the official reduction semantics). Failing
        # explicitly instead of silently skipping/mis-training.
        raise NotImplementedError(
            "SAEHD training step (loss composition, GAN/true-face "
            "discriminator training, gradient orchestration) is Phase "
            "6B — Phase 6A implements the structural foundation only")

    # Phase 11 (DFM/ONNX): the official tf2onnx export is out of the
    # torch-6A scope; the official I/O contract (in_face:0 ->
    # out_face_mask:0 / out_celeb_face:0 / out_celeb_face_mask:0,
    # opset 12, dynamic batch) is documented in
    # docs/IMPLEMENTATION_PLAN_v2.md section 33.
    def export_dfm (self):
        raise NotImplementedError(
            "SAEHD DFM/ONNX export is Phase 11 — Phase 6A implements "
            "the structural foundation only")

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

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = SAEHDModel
