"""S3FD face detector — torch implementation of the official extractor
(Phase 9B).

Official provenance: ``facelib/S3FDExtractor.py`` at upstream baseline
``e4b7543ffa1d73b26fce1e31852727f658ba490c`` (the file is unchanged in
the modernization branch; the official TF source is preserved in the
archived reference tree ``docs/_p9_tfref/official``).

Migration notes (torch foundation, Phase 3A/3B/4 contracts):
- the network is the official S3FD architecture UNCHANGED: VGG-style
  backbone (conv1_1..conv5_3, fc6 with int padding 3, fc7,
  conv6_1/6_2, conv7_1/7_2), L2Norm on the conv3_3/4_3/5_3 heads, six
  confidence/loc heads, single-softmax ``cls1`` convention (max over
  the RAW conv outputs, then ONE softmax — the USER_LEGACY torch port's
  double softmax is a deviation and is NOT replicated);
- ``tf.nn.*`` call sites become the native torch equivalents with
  identical semantics: ``torch.relu``, layout-aware VALID
  ``F.max_pool2d(x, 2, 2)`` via the ``_max_pool2`` helper (the
  official ``tf.nn.max_pool … 'VALID'`` is ``data_format``-aware;
  torch's kernel is NCHW-only, so the helper permutes the NHWC
  boundary), ``torch.softmax(…, dim=-1)``, ``torch.maximum``,
  ``torch.cat(…, dim=-1)``;
- data format NHWC (the official extractor initializes with
  ``data_format="NHWC"``); ``nn.Conv2D`` handles the NHWC boundary
  permute itself;
- ``minus [104,117,123]`` was a ``tf.constant`` (NOT a checkpoint
  variable) -> a plain unregistered torch tensor attribute, so it is
  excluded from the checkpoint enumeration (it would fail the strict
  load as a missing key otherwise);
- L2Norm keeps the official 4-D gain layout ``(1,1,1,C)`` as a 4-D
  torch Parameter: the official checkpoint stores it in exactly this
  form (it is NOT a 1-D channel state), so it maps by IDENTITY through
  the Phase 4 converter — the ``channel_broadcast`` whitelist
  deliberately does not apply (``param.ndim != 1``);
- weight loading: the official strict file format (pickle protocol-4
  ``dict[str, np.ndarray]``) is read with
  ``core.leras.convert.read_official_checkpoint`` and applied with
  ``convert_official_to_torch`` — strict two-pass, all-or-nothing:
  missing key / extra key / shape mismatch / dtype mismatch all fail
  loudly with the full report. The official greedy TF loader and the
  non-strict ``load_state_dict(strict=False)`` / ``hasattr``-guarded
  loaders of the legacy/external ports are NOT reintroduced;
- device: ``nn.initialize`` (the worker's device config, or forced CPU
  when ``place_model_on_cpu`` — the torch form of the official
  ``tf.device('/CPU:0')`` variable-placement context) places every
  parameter and every ``run()`` input;
- preprocessing / decode / NMS / post-processing are the official
  algorithms verbatim: BGR->RGB flip, ``minus [104,117,123]`` after
  the flip (in-graph), ``scale_to = 640 if d>=1280 else d/2`` (min
  64), ``cv2.INTER_LINEAR`` resize, strides ``2**(i+2)``, candidacy
  ``ocls[...,1] > 0.05``, priors ``[w*s+s/2, h*s+s/2, 4s, 4s]``,
  ``box = p[:2] + loc[:2]*0.1*p2`` / ``p2*exp(loc[2:]*0.2)``,
  ``refine_nms(0.3)`` with the official ``+1``-IoU convention and
  ``order[inds+1]`` pattern, final ``score >= 0.5``, ``min(r-l, b-t) <
  40`` drop, ``+10 %`` chin enlarge, int truncation, area-desc sort,
  optional intersects removal;
- NUMPY 2: the official ``astype(np.int)`` sites use the builtin
  ``int`` (the NumPy-1.x ``np.int`` alias was exactly the builtin).

Behavioral pinning: the frozen official (unmodified TF baseline, CPU-
only) reference outputs in ``docs/_p9_tfref/frozen`` (private; see
``docs/PHASE9_STATE.md``) — 12 olist maps, post-NMS boxes with scores,
final rects — on the fixed sample set.
"""

import operator
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
# the modernized core.leras attaches its foundation registries (the
# ``nn.LayerBase`` / ``nn.ModelBase`` container bases used by the
# module-level classes below) lazily — on subpackage import or on the
# first ``nn.initialize`` call (core/leras/nn.py). The official tree's
# eager ``core.leras`` import made this module importable standalone;
# import the registry-establishing subpackages explicitly to keep that
# contract (``facelib/__init__`` and the test suite import this module
# before any ``nn.initialize`` call). ``core.leras.models`` skips the
# TF XSeg foundation under the torch foundation (the ``nn.tf`` guard).
import core.leras.layers  # noqa: F401
import core.leras.models  # noqa: F401
from core.leras import convert
from ._extractor_precision import cudnn_fp32_for_extractor


def _max_pool2(x, kernel_size=2, strides=2):
    """Official ``tf.nn.max_pool(x, [1,k,k,1], [1,s,s,1], 'VALID')`` (the
    S3FD forward calls the raw TF op, not the ``nn.max_pool`` helper):
    layout-aware — ``nn.Conv2D`` returns tensors in ``nn.data_format``
    (NHWC here) while ``F.max_pool2d`` consumes NCHW, so the boundary
    permutes mirror the official ``data_format``-aware TF kernel."""
    nhwc = (nn.data_format == "NHWC")
    if nhwc:
        x = x.permute(0, 3, 1, 2)
    x = F.max_pool2d(x, kernel_size, strides)
    if nhwc:
        x = x.permute(0, 2, 3, 1)
    return x


class L2Norm(nn.LayerBase):
    """Official S3FD L2Norm (upstream L18-32):

    ``x / (sqrt(sum(x^2, axis=-1, keepdims=True)) + 1e-10) * weight``

    with ``weight`` in the official 4-D ``(1,1,1,C)`` gain layout
    (initialized to ones; a genuine checkpoint variable in the official
    file — loaded by identity, see the module docstring).
    """

    def __init__(self, n_channels, **kwargs):
        self.n_channels = n_channels
        super().__init__(**kwargs)

    def build_weights(self):
        # 4-D gain parameter mirroring the official checkpoint layout
        self.weight = torch.nn.Parameter(
            torch.empty(1, 1, 1, self.n_channels,
                        device=nn.device, dtype=nn.floatx),
            requires_grad=False,
        )
        self.register_param_initializer("weight", nn.initializers.ones)

    def get_weights(self):
        return [self.weight]

    def forward(self, x):
        x = x / (torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)) + 1e-10) * self.weight
        return x

    def __str__(self):
        r = f"{self.__class__.__name__}"
        if self.name is not None:
            r += f" : {self.name}"
        return r


class S3FD(nn.ModelBase):
    """Official S3FD network (upstream ``on_build`` L38-88, ``forward``
    L90-159) on the torch foundation. ``minus`` is a plain tensor
    attribute (official ``tf.constant`` — NOT checkpoint state). The
    ``cls1`` head uses the official single-softmax-after-max-out on the
    raw conv outputs (upstream L135, L153-157)."""

    def __init__(self):
        super().__init__(name='S3FD')

    def on_build(self):
        # official tf.constant([104,117,123], dtype=nn.floatx) (L39) —
        # a constant, not a variable: kept as an unregistered attribute
        # so it never enters the checkpoint enumeration
        self.minus = torch.tensor([104, 117, 123], device=nn.device, dtype=nn.floatx)

        self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding='SAME')
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding='SAME')

        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, strides=1, padding='SAME')
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, strides=1, padding='SAME')

        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, strides=1, padding='SAME')
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')

        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, strides=1, padding='SAME')
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

        self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, strides=1, padding=3)
        self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, strides=1, padding='SAME')

        self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, strides=1, padding='SAME')
        self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, strides=2, padding='SAME')

        self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, strides=1, padding='SAME')
        self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, strides=2, padding='SAME')

        self.conv3_3_norm = L2Norm(256)
        self.conv4_3_norm = L2Norm(512)
        self.conv5_3_norm = L2Norm(512)

        self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
        self.conv3_3_norm_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

        self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
        self.conv4_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

        self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
        self.conv5_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

        self.fc7_mbox_conf = nn.Conv2D(1024, 2, kernel_size=3, strides=1, padding='SAME')
        self.fc7_mbox_loc = nn.Conv2D(1024, 4, kernel_size=3, strides=1, padding='SAME')

        self.conv6_2_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
        self.conv6_2_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

        self.conv7_2_mbox_conf = nn.Conv2D(256, 2, kernel_size=3, strides=1, padding='SAME')
        self.conv7_2_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

    def forward(self, x):
        with cudnn_fp32_for_extractor(x):
            return self._forward_impl(x)

    def _forward_impl(self, x):
        # torch container contract: ModelBase.run feeds each run input as
        # a positional argument (the official TF ``inp`` list of
        # placeholders is already unpacked — the official ``x, = inp``
        # idiom is not repeated here)
        x = x - self.minus
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = _max_pool2(x)

        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = _max_pool2(x)

        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        f3_3 = x
        x = _max_pool2(x)

        x = torch.relu(self.conv4_1(x))
        x = torch.relu(self.conv4_2(x))
        x = torch.relu(self.conv4_3(x))
        f4_3 = x
        x = _max_pool2(x)

        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.conv5_2(x))
        x = torch.relu(self.conv5_3(x))
        f5_3 = x
        x = _max_pool2(x)

        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        ffc7 = x

        x = torch.relu(self.conv6_1(x))
        x = torch.relu(self.conv6_2(x))
        f6_2 = x

        x = torch.relu(self.conv7_1(x))
        x = torch.relu(self.conv7_2(x))
        f7_2 = x

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        cls2 = torch.softmax(self.conv4_3_norm_mbox_conf(f4_3), dim=-1)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)

        cls3 = torch.softmax(self.conv5_3_norm_mbox_conf(f5_3), dim=-1)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)

        cls4 = torch.softmax(self.fc7_mbox_conf(ffc7), dim=-1)
        reg4 = self.fc7_mbox_loc(ffc7)

        cls5 = torch.softmax(self.conv6_2_mbox_conf(f6_2), dim=-1)
        reg5 = self.conv6_2_mbox_loc(f6_2)

        cls6 = torch.softmax(self.conv7_2_mbox_conf(f7_2), dim=-1)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label: the official single softmax applies
        # AFTER the max-out on the raw conv outputs (upstream L153-157)
        bmax = torch.maximum(torch.maximum(cls1[..., 0:1], cls1[..., 1:2]), cls1[..., 2:3])

        cls1 = torch.cat([bmax, cls1[..., 3:4]], dim=-1)
        cls1 = torch.softmax(cls1, dim=-1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("Unable to load S3FD.npy")

        if place_model_on_cpu:
            # official: the variables were created inside a
            # tf.device('/CPU:0') context -> torch form: force the
            # extractor (model + run() inputs) onto the CPU device
            nn.initialize(nn.DeviceConfig.CPU(), data_format="NHWC")
        else:
            nn.initialize(data_format="NHWC")

        self.model = S3FD()
        self.model.build()

        # strict official-format load (Phase 4 converter engine:
        # all-or-nothing, missing/extra key, shape and dtype mismatches
        # fail loudly with the full report)
        d = convert.read_official_checkpoint(str(model_path))
        convert.convert_official_to_torch(self.model, d, component=self.model.name)

        self.model.build_for_run([ (nn.floatx, nn.get4Dshape(None, None, 3)) ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True, is_remove_intersects=False):

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize (input_image, ( int(w/input_scale), int(h/input_scale) ), interpolation=cv2.INTER_LINEAR)

        olist = self.model.run ([ input_image[None,...] ] )

        detected_faces = []
        for ltrb in self.refine (olist):
            l,t,r,b = [ x*input_scale for x in ltrb]
            bt = b-t
            if min(r-l,bt) < 40: #filtering faces < 40pix by any side
                continue
            b += bt*0.1 #enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append ( [int(x) for x in (l,t,r,b)] )

        #sort by largest area first
        detected_faces = [ [(l,t,r,b), (r-l)*(b-t) ]  for (l,t,r,b) in detected_faces ]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
        detected_faces = [ x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        # the official ((ocls,), (oreg,)) idiom binds ocls = map[0]:
        # the run() outputs keep the batch dim (1,H,W,C) and the idiom
        # slices batch 0 — kept verbatim from the official (see the
        # module docstring; docs/PHASE9_STATE.md §12.1)
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls[...,1] > 0.05)):
                score = ocls[hindex, windex, 1]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        # official astype(np.int) — the NumPy-1.x alias of the builtin
        bboxlist = [ x[:-1].astype(int) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
