"""FAN (2D/3D) facial landmark extractor — torch implementation of the
official extractor (Phase 9C).

Official provenance: ``facelib/FANExtractor.py`` at upstream baseline
``e4b7543ffa1d73b26fce1e31852727f658ba490c`` ("ported from
https://github.com/1adrianb/face-alignment"; the file is unchanged in
the modernization branch; the official TF source is preserved in the
archived reference tree ``docs/_p9_tfref/official``).

Migration notes (torch foundation, Phase 3A/3B/4/9C contracts):
- the network is the official FAN-4 architecture UNCHANGED: trunk
  (conv1 7x7 s2 3->64 + bn1 + relu; conv2 = ConvBlock(64,128);
  ``avg_pool`` 2x2 VALID; conv3 = ConvBlock(128,128); conv4 =
  ConvBlock(128,256)); four refinement levels of HourGlass(256, 4)
  (recursive depth-4 hourglass: b1, b2, b2_plus, b3 of ConvBlock /
  nested HourGlass, nearest-2x upsample + add), each level followed
  by top_m ConvBlock(256,256), conv_last 1x1 256->256, bn_end, relu,
  l 1x1 256->68; residual links bl 1x1 256->256 / al 1x1 68->256
  for levels 0..2 (``previous = previous + bl(ll) + al(tmp_out)``);
  output = the LAST level's l head, transposed to NCHW (1,68,64,64)
  (the official ``tf.transpose(x, (0,3,1,2))``);
- ``tf.nn.*`` call sites become native torch equivalents with
  identical semantics: ``torch.relu``, ``torch.cat(..., dim=-1)``,
  layout-aware VALID ``F.avg_pool2d(x, 2, 2)`` via the ``_avg_pool2``
  helper (the official ``tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1],
  'VALID')`` is ``data_format``-aware; torch's kernel is NCHW-only, so
  the helper permutes the NHWC boundary — the same contract as the
  Phase 9B ``_max_pool2``), and the official ``nn.upsample2d`` (migrated
  into ``core/leras/ops`` in this phase: nearest-neighbor 2x = the
  official ``tf.image.resize_nearest_neighbor``);
- ConvBlock / HourGlass are nested ``nn.ModelBase`` containers exactly
  as official (defined in ``__init__``); the official LIST-valued
  attributes (``m``, ``top_m``, ``conv_last``, ``bn_end``, ``l``,
  ``bl``, ``al``) are built by the torch ``ModelBase.build`` discovery
  loop, which names the elements ``m_0..m_3``, ``top_m_0..3``,
  ``conv_last_0..3``, ``bn_end_0..3``, ``l_0..3``, ``bl_0..2``,
  ``al_0..2`` — but torch's module registry cannot hold lists, so the
  ported ``FAN.build`` additionally registers each element under the
  name the build loop assigned (``register_module``); that is what
  makes the checkpoint engine's ``named_parameters`` tree reproduce
  the official dotted scope keys 1:1 (``m_0/b2_plus/...``);
- weight loading: the official strict file format (pickle protocol-4
  ``dict[str, np.ndarray]``, 945 keys) is read with
  ``core.leras.convert.read_official_checkpoint`` and applied with
  ``convert_official_to_torch`` — strict two-pass, all-or-nothing:
  missing key / extra key / shape mismatch / dtype mismatch all fail
  loudly with the full report. The two official artifacts use two
  different 1-D state layouts — ``2DFAN.npy`` stores every 1-D state
  (conv biases, BN weight/bias/running_mean/running_var) as the NHWC
  4-D singleton form ``(1,1,1,C)``; ``3DFAN.npy`` stores them as
  plain ``(C,)`` (the official TF variable shape) — the Phase 4
  ``channel_broadcast`` whitelist covers BOTH forms without any
  model-specific table; conv kernels go through the ``Conv2D``
  HWIO->OIHW layout hook;
- device: same contract as Phase 9B (S3FD) — ``nn.initialize`` with
  the worker's device config, or forced ``DeviceConfig.CPU()`` when
  ``place_model_on_cpu`` (the torch form of the official
  ``tf.device('/CPU:0')`` variable-placement context; faithful here
  because the pipeline applies the SAME ``place_model_on_cpu`` to
  S3FD and FAN on a worker);
- the official ``run()`` returns a BARE 4-D array for this
  single-output model (the official ``tf_sess.run`` of one tensor);
  the torch ``ModelBase.run`` keeps that contract (single tensor out
  -> bare NumPy array, list of tensors -> list), so the official
  ``self.model.run([img[None, ...]])[0]`` batch-0 slice idiom and the
  3-D ``get_pts_from_predict`` work UNCHANGED;
- preprocessing / decode / second pass are the official algorithms
  verbatim: per-rect ``scale = (r-l+b-t)/195.0`` / ``center =
  [(l+r)/2, (t+b)/2]``; ``crop`` = the official 3x3 affine
  (``m[0,0] = m[1,1] = 256/(200*scale)``,
  ``m[0,2] = 256*(-cx/(200*s)+0.5)`` etc., inverted) + zero-paste +
  ``cv2.resize(256, 256, INTER_LINEAR)``; ``/255.0`` float32;
  ``multi_sample`` adds the four corner +-1 centers and averages the
  per-center landmarks; heatmap decode = argmax per channel over
  64x64 (``c[:,0] %= a_w``, ``c[:,1] = floor(c/a_w)``) with the
  official subpixel correction (``sign(diff)*0.25`` inside the
  ``0 < p < 63`` window) + ``c += 0.5`` + the official ``transform``
  back to image coords; the second pass (S3FD on the warpAffine'd 256
  crop, INTER_CUBIC, re-extract with multi_sample when exactly 1
  face) is kept verbatim, including its bare ``except: pass``;
- NUMPY 2: the official ``astype(np.int)`` / ``astype(np.float)`` /
  ``dtype=np.int`` sites use the builtins (the NumPy-1.x aliases
  were exactly those builtins).

Behavioral pinning: the frozen official (unmodified TF baseline, CPU-
only) reference outputs in ``docs/_p9_tfref/frozen`` (private; see
``docs/PHASE9_STATE.md``) — per-rect 68 landmarks of both the 2D and
3D FAN on the fixed sample set (the FAN rects are the frozen S3FD
final rects; the synthetic sample forces rect [128,128,512,512]),
plus the full 68x64x64 prediction map for the synthetic sample.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from facelib import FaceType, LandmarksProcessor
from core.leras import nn
# the modernized core.leras attaches its foundation registries
# (nn.ModelBase / nn.LayerBase / the layer and ops registries) LAZILY
# on the subpackage imports that run inside the first nn.initialize();
# import them explicitly so this module keeps the official tree's
# standalone-import contract (facelib/__init__ imports FANExtractor
# before any nn.initialize runs), and so the migrated nn.upsample2d op
# is attached for the official call sites below.
import core.leras.layers  # noqa: F401
import core.leras.models  # noqa: F401
import core.leras.ops  # noqa: F401
from core.leras import convert


def _avg_pool2(x):
    """Official ``tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')``
    (the FAN forward calls the raw TF op, not the ``nn.max_pool``
    helper): layout-aware — the official op honors ``data_format``;
    torch ``F.avg_pool2d`` is NCHW-only, so the NHWC boundary is
    permuted. 2x2 stride-2 with no padding is exact (VALID)."""
    nhwc = (nn.data_format == "NHWC")
    if nhwc:
        x = x.permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, 2, 2)
    if nhwc:
        x = x.permute(0, 2, 3, 1)
    return x


class FANExtractor(object):
    def __init__(self, landmarks_3D=False, place_model_on_cpu=False):

        model_path = Path(__file__).parent / (
            "2DFAN.npy" if not landmarks_3D else "3DFAN.npy")
        if not model_path.exists():
            raise Exception("Unable to load FANExtractor model")

        if place_model_on_cpu:
            # official: the variables were created inside a
            # tf.device('/CPU:0') context -> torch form: force the
            # extractor (model + run() inputs) onto the CPU device
            nn.initialize(nn.DeviceConfig.CPU(), data_format="NHWC")
        else:
            nn.initialize(data_format="NHWC")

        class ConvBlock(nn.ModelBase):
            def on_build(self, in_planes, out_planes):
                self.in_planes = in_planes
                self.out_planes = out_planes

                self.bn1 = nn.BatchNorm2D(in_planes)
                self.conv1 = nn.Conv2D(in_planes, out_planes//2, kernel_size=3, strides=1, padding='SAME', use_bias=False)

                self.bn2 = nn.BatchNorm2D(out_planes//2)
                self.conv2 = nn.Conv2D(out_planes//2, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False)

                self.bn3 = nn.BatchNorm2D(out_planes//4)
                self.conv3 = nn.Conv2D(out_planes//4, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False)

                if self.in_planes != self.out_planes:
                    self.down_bn1 = nn.BatchNorm2D(in_planes)
                    self.down_conv1 = nn.Conv2D(in_planes, out_planes, kernel_size=1, strides=1, padding='VALID', use_bias=False)
                else:
                    self.down_bn1 = None
                    self.down_conv1 = None

            def forward(self, input):
                x = input
                x = self.bn1(x)
                x = torch.relu(x)
                x = out1 = self.conv1(x)

                x = self.bn2(x)
                x = torch.relu(x)
                x = out2 = self.conv2(x)

                x = self.bn3(x)
                x = torch.relu(x)
                x = out3 = self.conv3(x)

                x = torch.cat([out1, out2, out3], dim=-1)

                if self.in_planes != self.out_planes:
                    downsample = self.down_bn1(input)
                    downsample = torch.relu(downsample)
                    downsample = self.down_conv1(downsample)
                    x = x + downsample
                else:
                    x = x + input

                return x

        class HourGlass(nn.ModelBase):
            def on_build(self, in_planes, depth):
                self.b1 = ConvBlock(in_planes, 256)
                self.b2 = ConvBlock(in_planes, 256)

                if depth > 1:
                    self.b2_plus = HourGlass(256, depth-1)
                else:
                    self.b2_plus = ConvBlock(256, 256)

                self.b3 = ConvBlock(256, 256)

            def forward(self, input):
                up1 = self.b1(input)

                low1 = _avg_pool2(input)
                low1 = self.b2(low1)

                low2 = self.b2_plus(low1)
                low3 = self.b3(low2)

                up2 = nn.upsample2d(low3)

                return up1+up2

        class FAN(nn.ModelBase):
            def __init__(self):
                super().__init__(name='FAN')

            def on_build(self):
                self.conv1 = nn.Conv2D(3, 64, kernel_size=7, strides=2, padding='SAME')
                self.bn1 = nn.BatchNorm2D(64)

                self.conv2 = ConvBlock(64, 128)
                self.conv3 = ConvBlock(128, 128)
                self.conv4 = ConvBlock(128, 256)

                self.m = []
                self.top_m = []
                self.conv_last = []
                self.bn_end = []
                self.l = []
                self.bl = []
                self.al = []
                for i in range(4):
                    self.m += [HourGlass(256, 4)]
                    self.top_m += [ConvBlock(256, 256)]

                    self.conv_last += [nn.Conv2D(256, 256, kernel_size=1, strides=1, padding='VALID')]
                    self.bn_end += [nn.BatchNorm2D(256)]

                    self.l += [nn.Conv2D(256, 68, kernel_size=1, strides=1, padding='VALID')]

                    if i < 4-1:
                        self.bl += [nn.Conv2D(256, 256, kernel_size=1, strides=1, padding='VALID')]
                        self.al += [nn.Conv2D(68, 256, kernel_size=1, strides=1, padding='VALID')]

            def build(self):
                # torch adaptation (the ONLY structural deviation from
                # the official class): the official list-valued
                # attributes (m, top_m, conv_last, bn_end, l, bl, al)
                # are discovered and built by the ModelBase loop above
                # (which names the elements m_0..m_3, ...), but torch's
                # module registry cannot hold lists — register each
                # element under the name the build loop assigned so
                # named_parameters (the checkpoint engine's tree walk)
                # reproduces the official dotted scope keys 1:1
                super().build()
                for attr in ("m", "top_m", "conv_last", "bn_end", "l", "bl", "al"):
                    for i, sub in enumerate(getattr(self, attr)):
                        self.register_module(f"{attr}_{i}", sub)

            def forward(self, x):
                # torch container contract: ModelBase.run feeds each run
                # input as a positional argument (the official TF
                # placeholder list is already unpacked — the official
                # ``x, = inp`` idiom is not repeated here)
                x = self.conv1(x)
                x = self.bn1(x)
                x = torch.relu(x)

                x = self.conv2(x)
                x = _avg_pool2(x)
                x = self.conv3(x)
                x = self.conv4(x)

                outputs = []
                previous = x
                for i in range(4):
                    ll = self.m[i](previous)
                    ll = self.top_m[i](ll)
                    ll = self.conv_last[i](ll)
                    ll = self.bn_end[i](ll)
                    ll = torch.relu(ll)
                    tmp_out = self.l[i](ll)
                    outputs.append(tmp_out)
                    if i < 4-1:
                        ll = self.bl[i](ll)
                        previous = previous + ll + self.al[i](tmp_out)
                x = outputs[-1]
                x = x.permute(0, 3, 1, 2)
                return x

        self.model = FAN()
        self.model.build()

        # strict official-format load (Phase 4 converter engine:
        # all-or-nothing, missing/extra key, shape and dtype
        # mismatches fail loudly with the full report; the two
        # official 1-D state forms — 2DFAN (1,1,1,C) and 3DFAN (C,) —
        # are both covered by the channel_broadcast whitelist)
        d = convert.read_official_checkpoint(str(model_path))
        convert.convert_official_to_torch(self.model, d, component=self.model.name)

        self.model.build_for_run([(nn.floatx, (None, 256, 256, 3))])

    def extract(self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0

            center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
            centers = [center]

            if multi_sample:
                centers += [center + [-1, -1],
                            center + [1, -1],
                            center + [1, 1],
                            center + [-1, 1],
                            ]

            images = []
            ptss = []

            try:
                for c in centers:
                    images += [self.crop(input_image, c, scale)]

                images = np.stack(images)
                images = images.astype(np.float32) / 255.0

                predicted = []
                for i in range(len(images)):
                    predicted += [self.model.run([images[i][None, ...]])[0]]

                predicted = np.stack(predicted)

                for i, pred in enumerate(predicted):
                    ptss += [self.get_pts_from_predict(pred, centers[i], scale)]
                pts_img = np.mean(np.array(ptss), 0)

                landmarks.append(pts_img)
            except:
                landmarks.append(None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC)

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if len(rects2) == 1:  # dont do second pass if faces != 1 detected in cropped image
                            lmrks2 = self.extract(face_image, [rects2[0]], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points(lmrks2, image_to_face_mat, True)
                except:
                    pass

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array([point[0], point[1], 1.0])
        h = 200.0 * scale
        m = np.eye(3)
        m[0, 0] = resolution / h
        m[1, 1] = resolution / h
        m[0, 2] = resolution * (-center[0] / h + 0.5)
        m[1, 2] = resolution * (-center[1] / h + 0.5)
        m = np.linalg.inv(m)
        return np.matmul(m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype(int)
        br = self.transform([resolution, resolution], center, scale, resolution).astype(int)

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0]-1:newY[1], newX[0]-1:newX[1]] = image[oldY[0]-1:oldY[1], oldX[0]-1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape((a_ch, a_h*a_w))
        c = b.argmax(1).reshape((a_ch, 1)).repeat(2, axis=1).astype(float)
        c[:, 0] %= a_w
        c[:, 1] = np.apply_along_axis(lambda x: np.floor(x / a_w), 0, c[:, 1])

        for i in range(a_ch):
            pX, pY = int(c[i, 0]), int(c[i, 1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array([a[i, pY, pX+1]-a[i, pY, pX-1], a[i, pY+1, pX]-a[i, pY-1, pX]])
                c[i] += np.sign(diff)*0.25

        c += 0.5

        return np.array([self.transform(c[i], center, scale, a_w) for i in range(a_ch)])
