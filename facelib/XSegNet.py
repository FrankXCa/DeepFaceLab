"""XSegNet — torch port of the official XSeg net wrapper (Phase 10B).

The official TF source is preserved verbatim in ``XSegNet_tf.py`` (dead
reference, never imported — project convention, like
``core/leras/models/XSeg_tf.py``).

Official behavior preserved:
- constructor signature (``name``, ``resolution=256``, ``load_weights``,
  ``weights_file_root``, ``training``, ``place_model_on_cpu``,
  ``run_on_cpu``, ``optimizer``, ``data_format``,
  ``raise_on_no_model_files``) and the official
  ``nn.initialize(data_format=data_format)`` at construction start;
- the net is the Phase 10B torch ``nn.XSeg(3, 32, 1, name=name)``
  (the official architecture — the torch replacement of
  ``core/leras/models/XSeg.py``);
- the official ``model_filename_list`` lifecycle:
  ``[[opt, '{name}_{resolution}_opt.npy']]`` (training only, the
  official ``nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt')`` is
  constructed by the caller — ``models/Model_XSeg/Model.py`` — and
  passed in) followed by ``[[model, '{name}_{resolution}.npy']]``;
  the official load/init loop semantics (``do_init = not
  load_weights``; ``do_init = not model.load_weights(path)``; on
  failure the official ``raise_on_no_model_files`` exception; in
  inference mode a missing file sets ``self.initialized = False`` and
  breaks — ``extract`` then returns the official 0.5-ones fallback);
- ``opt.initialize_variables(model_weights, vars_on_cpu=
  place_model_on_cpu)`` (Phase 3E2 — the official optimizer-state
  creation, official ``iters``/``acc_*`` variable names);
- ``get_resolution`` / ``flow(x, pretrain)`` (the official
  ``model(x, pretrain=pretrain)`` dispatch — the pretrain flag routes
  the official zeroed-skip-tensor forward branch) /
  ``get_weights`` / ``save_weights`` (the official progress-bar save
  of every saveable in the filename list — Phase 4 Saveable engine,
  official raw pickle protocol-4 ``.npy`` streams in the official
  layouts) / ``extract`` (the official contract: uninitialized net ->
  ``0.5 * ones((res, res, 1))``; the official single-image batch-dim
  add/strip; ``np.clip(result, 0, 1.0)``; the official noise gate
  ``result[result < 0.1] = 0``; NumPy in, NumPy out);
- ``VERSION = 1``.

Documented torch adaptations (Phase 10B — the official TF graph/session
concepts removed, exactly like the Phase 5 ``ModelBase`` / Phase 6
SAEHD ports):
- the official CPU placeholders (``input_t`` (res,res,3) /
  ``target_t`` (res,res,1) in the active data format) are removed —
  the torch foundation feeds tensors/arrays directly and executes
  eagerly on ``nn.device``; the inference ``net_run`` closure (the
  official ``tf_sess.run([pred], feed_dict=...)``) becomes a
  ``torch.no_grad`` forward: NumPy in (the net's data format — the
  caller contract of ``extract`` / the sample generators) -> tensor on
  ``nn.device`` -> the official sigmoid output -> CPU NumPy out (the
  official session-run caller contract);
- the official ``tf.device`` placement contexts (model on CPU when
  ``place_model_on_cpu`` / ``run_on_cpu``, otherwise the default
  device) are replaced by the Phase 2 co-location semantics: the net
  and its optimizer state live on ``nn.device`` (CPU when no device
  is selected, the backend device otherwise) — for every official XSeg
  call path (``Model_XSeg/Model.py`` passes
  ``place_model_on_cpu = len(devices) == 0``; ``mainscripts/XSegUtil.
  py`` initializes the caller-chosen device) the resulting placement
  is identical, so the flags are kept signature-only;
- the official ``nn.floatx.as_numpy_dtype`` (a TF dtype) in the
  ``extract`` fallback is ``np.float32`` in the torch foundation
  (``nn.floatx`` is a torch dtype; XSeg is always float32);
- a missing REQUIRED file on a resume load (``load_weights=True``)
  fails with an explicit ``FileNotFoundError`` instead of the
  official silent re-initialization (the Phase 6 SAEHD strict-policy
  precedent; IMPLEMENTATION_PLAN_v2.md section 19 — a corrupt
  existing file already fails strictly with ``CheckpointLoadError``
  in the Phase 4 engine, and first-run initialization — the official
  ``do_init = not load_weights`` branch — is unchanged);
- no TensorFlow import on this path; no direct CUDA-backend calls or
  CUDA device-string literals (device handling through the Phase 2
  abstraction).
- CUDA inference scopes cuDNN TF32 off for the forward, restoring the
  caller's setting afterward. Phase 10B measured up to 8.1e-3 mask
  difference against CPU under the default TF32 policy, falling to
  1.2e-5 with full f32 cuDNN. CPU and matmul policy are unchanged.

Official provenance: ``facelib/XSegNet.py`` at upstream baseline
``e4b7543ffa1d73b26fce1e31852727f658ba490c`` (see ``XSegNet_tf.py``).
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from core.interact import interact as io
from core.leras import nn


class XSegNet(object):
    VERSION = 1

    def __init__(self, name,
                    resolution=256,
                    load_weights=True,
                    weights_file_root=None,
                    training=False,
                    place_model_on_cpu=False,
                    run_on_cpu=False,
                    optimizer=None,
                    data_format="NHWC",
                    raise_on_no_model_files=False):

        self.resolution = resolution
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent

        nn.initialize(data_format=data_format)

        model_name = f'{name}_{resolution}'
        self.model_filename_list = []

        # torch: the official CPU placeholders (input_t / target_t) are
        # removed — inputs are fed as tensors directly (see module
        # docstring). The net is built on nn.device (Phase 2
        # co-location replaces the official tf.device placement
        # contexts).
        self.model = nn.XSeg(3, 32, 1, name=name)
        self.model.build()
        self.model_weights = self.model.get_weights()

        if training:
            if optimizer is None:
                raise ValueError("Optimizer should be provided for training mode.")
            self.opt = optimizer
            self.opt.initialize_variables(self.model_weights, vars_on_cpu=place_model_on_cpu)
            self.model_filename_list += [ [self.opt, f'{model_name}_opt.npy' ] ]

        self.model_filename_list += [ [self.model, f'{model_name}.npy'] ]

        if not training:
            # torch: the official `_, pred = self.model(self.input_t)`
            # graph fragment + `tf_sess.run(feed_dict=...)` becomes an
            # eager no-grad forward (NumPy in the net's data format ->
            # the official sigmoid output -> CPU NumPy out).
            def net_run(input_np):
                torch = nn.torch
                x = torch.from_numpy(np.ascontiguousarray(input_np))
                x = x.to(device=nn.device, dtype=nn.floatx)
                # Measured on the Phase 10B XSeg checkpoint: default
                # cuDNN TF32 changes the CUDA mask by up to 8.1e-3
                # against CPU; full f32 cuDNN brings it to 1.2e-5.
                # Scope the flag to this inference call and restore the
                # caller's setting. The 10C training step needs its own
                # forward-and-backward precision scope.
                precision = (torch.backends.cudnn.flags(allow_tf32=False)
                             if x.is_cuda else nullcontext())
                with torch.no_grad(), precision:
                    _, pred = self.model(x)
                return pred.detach().cpu().numpy()
            self.net_run = net_run

        self.initialized = True
        # Loading/initializing all models/optimizers weights
        for model, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / filename
                do_init = not model.load_weights( model_file_path )
                if do_init:
                    # a missing file (an existing corrupt file already
                    # failed strictly with CheckpointLoadError)
                    if raise_on_no_model_files:
                        raise Exception(f'{model_file_path} does not exists.')
                    if load_weights:
                        # Phase 10B strict policy (Phase 6 SAEHD
                        # precedent): a missing required file on a
                        # resume fails explicitly instead of the
                        # official silent re-initialization
                        raise FileNotFoundError(
                            f'required model file missing on resume: {model_file_path}')
                    if not training:
                        self.initialized = False
                        break

            if do_init:
                model.init_weights()

    def get_resolution(self):
        return self.resolution

    def flow(self, x, pretrain=False):
        return self.model(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            model.save_weights( self.weights_file_root / filename )

    def extract(self, input_image):
        if not self.initialized:
            # torch: the official `nn.floatx.as_numpy_dtype` is
            # np.float32 (XSeg is always float32 — the torch foundation
            # floatx is a torch dtype)
            return 0.5*np.ones ( (self.resolution, self.resolution, 1), np.float32 )

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None,...]

        result = np.clip ( self.net_run(input_image), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result
