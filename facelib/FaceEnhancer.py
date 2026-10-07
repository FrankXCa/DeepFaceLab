"""Torch-native implementation of the official DeepFaceLab FaceEnhancer.

This module is an independent reimplementation constrained by the authenticated
TensorFlow parity fixture in ``tests/parity``. It deliberately does not use or
mutate the process-global Leras runtime: tensor layout and both model/compute
devices are local, immutable construction choices.
"""

from collections import OrderedDict
import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F


CHECKPOINT_SHA256 = (
    "254958f67c9adfe97a0c9fc7b3c343ba490a1519c01862a50945fa228875476a"
)
PATCH_SIZE = 192
UPSCALE = 4
OVERLAP = 96

_CONV_SPECS = OrderedDict([
    ("conv1", (3, 64)), ("e0_conv0", (64, 64)),
    ("e0_conv1", (64, 64)), ("e1_conv0", (64, 112)),
    ("e1_conv1", (112, 112)), ("e2_conv0", (112, 192)),
    ("e2_conv1", (192, 192)), ("e3_conv0", (192, 336)),
    ("e3_conv1", (336, 336)), ("e4_conv0", (336, 512)),
    ("e4_conv1", (512, 512)), ("center_conv0", (512, 512)),
    ("center_conv1", (512, 512)), ("center_conv2", (512, 512)),
    ("center_conv3", (512, 512)), ("d4_conv0", (1024, 512)),
    ("d4_conv1", (512, 512)), ("d3_conv0", (848, 512)),
    ("d3_conv1", (512, 512)), ("d2_conv0", (704, 288)),
    ("d2_conv1", (288, 288)), ("d1_conv0", (400, 160)),
    ("d1_conv1", (160, 160)), ("d0_conv0", (224, 96)),
    ("d0_conv1", (96, 96)), ("out1x_conv0", (96, 48)),
    ("out1x_conv1", (48, 3)), ("dec2x_conv0", (96, 96)),
    ("dec2x_conv1", (96, 96)), ("out2x_conv0", (96, 48)),
    ("out2x_conv1", (48, 3)), ("dec4x_conv0", (96, 72)),
    ("dec4x_conv1", (72, 72)), ("out4x_conv0", (72, 36)),
    ("out4x_conv1", (36, 3)),
])


def legacy_resize2d(value, size=None, scale_factor=None):
    """TensorFlow ``ResizeBilinear(false, false)`` for NCHW float32 tensors."""
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise TypeError("legacy_resize2d expects a four-dimensional tensor")
    if value.dtype != torch.float32:
        raise TypeError("legacy_resize2d expects float32 input")
    if (size is None) == (scale_factor is None):
        raise ValueError("specify exactly one of size or scale_factor")

    input_height, input_width = value.shape[-2:]
    if size is None:
        output_height = int(input_height * scale_factor)
        output_width = int(input_width * scale_factor)
    else:
        output_height, output_width = (int(size[0]), int(size[1]))
    if output_height <= 0 or output_width <= 0:
        raise ValueError("resize output dimensions must be positive")

    # Official coordinates are output_index * input_size / output_size. Keep
    # all coordinate arithmetic in float32 so CPU/CUDA follow the oracle.
    y = (torch.arange(output_height, device=value.device, dtype=value.dtype)
         * (float(input_height) / float(output_height)))
    x = (torch.arange(output_width, device=value.device, dtype=value.dtype)
         * (float(input_width) / float(output_width)))
    y0 = torch.floor(y).to(torch.long)
    x0 = torch.floor(x).to(torch.long)
    y1 = torch.clamp(y0 + 1, max=input_height - 1)
    x1 = torch.clamp(x0 + 1, max=input_width - 1)
    wy = (y - y0.to(value.dtype))[None, None, :, None]
    wx = (x - x0.to(value.dtype))[None, None, None, :]
    rows = value[:, :, y0, :] * (1.0 - wy) + value[:, :, y1, :] * wy
    return rows[:, :, :, x0] * (1.0 - wx) + rows[:, :, :, x1] * wx


class _FaceEnhancerNetwork(tnn.Module):
    def __init__(self, tensors):
        super().__init__()
        for name, (inputs, outputs) in _CONV_SPECS.items():
            setattr(self, name, tnn.Conv2d(inputs, outputs, 3, padding=1))
        self.dense1 = tnn.Linear(1, 64, bias=False)
        self.dense2 = tnn.Linear(1, 64, bias=False)

        with torch.no_grad():
            for name in _CONV_SPECS:
                layer = getattr(self, name)
                layer.weight.copy_(tensors[f"{name}/weight:0"])
                layer.bias.copy_(tensors[f"{name}/bias:0"])
            self.dense1.weight.copy_(tensors["dense1/weight:0"])
            self.dense2.weight.copy_(tensors["dense2/weight:0"])
        self.eval()


class FaceEnhancer:
    """Official x4 FaceEnhancer behavior backed by local Torch inference."""

    def __init__(self, place_model_on_cpu=False, run_on_cpu=False,
                 device_config=None, checkpoint_path=None):
        compute_device = self._resolve_compute_device(
            run_on_cpu=run_on_cpu, device_config=device_config)
        model_device = (torch.device("cpu") if place_model_on_cpu
                        else compute_device)

        # Both choices are captured once. Inference never consults mutable
        # Leras layout/device globals.
        self.compute_device = compute_device
        self.model_device = model_device
        self.place_model_on_cpu = bool(place_model_on_cpu)
        self.run_on_cpu = bool(run_on_cpu)
        self.checkpoint_path = Path(
            checkpoint_path or Path(__file__).with_name("FaceEnhancer.npy"))

        network = self._network_from_checkpoint(self.checkpoint_path)
        network.to(self.model_device)
        self.model = network

    @staticmethod
    def _resolve_compute_device(run_on_cpu, device_config):
        if run_on_cpu or device_config is None:
            return torch.device("cpu")
        devices = tuple(getattr(device_config, "devices", ()) or ())
        if not devices or not torch.cuda.is_available():
            return torch.device("cpu")
        index = int(devices[0].index)
        if index < 0 or index >= torch.cuda.device_count():
            return torch.device("cpu")
        return torch.device("cuda", index)

    @classmethod
    def _decode_checkpoint(cls, raw, expected_sha256=CHECKPOINT_SHA256):
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "FaceEnhancer checkpoint authentication failed before "
                f"deserialization: expected {expected_sha256}, "
                f"got {actual_sha256}")
        return cls._validate_checkpoint_values(pickle.loads(raw))

    @staticmethod
    def _validate_checkpoint_values(values):
        if not isinstance(values, dict):
            raise RuntimeError("FaceEnhancer checkpoint must be a dictionary")
        expected_keys = {
            f"{name}/{field}:0" for name in _CONV_SPECS
            for field in ("weight", "bias")
        } | {"dense1/weight:0", "dense2/weight:0"}
        actual_keys = set(values)
        if actual_keys != expected_keys or len(values) != 72:
            raise RuntimeError(
                "FaceEnhancer checkpoint key inventory mismatch: missing="
                f"{sorted(expected_keys - actual_keys)}, extra="
                f"{sorted(actual_keys - expected_keys)}")

        def require_finite(key, value):
            if not np.isfinite(value).all():
                raise RuntimeError(
                    f"checkpoint tensor {key} contains non-finite values")

        tensors = OrderedDict()
        for name, (inputs, outputs) in _CONV_SPECS.items():
            weight_key = f"{name}/weight:0"
            bias_key = f"{name}/bias:0"
            weight = values[weight_key]
            bias = values[bias_key]
            if not isinstance(weight, np.ndarray) or not isinstance(
                    bias, np.ndarray):
                raise RuntimeError(f"non-array checkpoint value for {name}")
            expected_weight_shape = (3, 3, inputs, outputs)
            expected_bias_shape = (1, 1, 1, outputs)
            if (weight.dtype != np.float16
                    or weight.shape != expected_weight_shape):
                raise RuntimeError(
                    f"invalid checkpoint weight {weight_key}: expected "
                    f"float16 {expected_weight_shape}, got "
                    f"{weight.dtype} {weight.shape}")
            if bias.dtype != np.float16 or bias.shape != expected_bias_shape:
                raise RuntimeError(
                    f"invalid checkpoint bias {bias_key}: expected "
                    f"float16 {expected_bias_shape}, got "
                    f"{bias.dtype} {bias.shape}")
            require_finite(weight_key, weight)
            require_finite(bias_key, bias)
            tensors[weight_key] = torch.from_numpy(
                weight.astype(np.float32).transpose(3, 2, 0, 1).copy())
            tensors[bias_key] = torch.from_numpy(
                bias.astype(np.float32).reshape(outputs).copy())

        for name in ("dense1", "dense2"):
            key = f"{name}/weight:0"
            value = values[key]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(f"non-array checkpoint value for {key}")
            if value.dtype != np.float16 or value.shape != (1, 64):
                raise RuntimeError(
                    f"invalid checkpoint dense weight {key}: expected "
                    f"float16 (1, 64), got {value.dtype} {value.shape}")
            require_finite(key, value)
            tensors[key] = torch.from_numpy(
                value.astype(np.float32).T.copy())
        return tensors

    @classmethod
    def _network_from_checkpoint(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Unable to load {path}")
        tensors = cls._decode_checkpoint(path.read_bytes())
        return _FaceEnhancerNetwork(tensors)

    def load_checkpoint(self, path=None):
        """Atomically replace model state after complete strict validation."""
        candidate = self._network_from_checkpoint(path or self.checkpoint_path)
        candidate.to(self.model_device)
        self.model = candidate

    def _conv(self, name, value):
        layer = getattr(self.model, name)
        if self.model_device == self.compute_device:
            return layer(value)
        # Low-VRAM split placement: these are operation-local copies. The
        # registered parameters remain owned by the persistent CPU module.
        weight = layer.weight.detach().to(self.compute_device)
        bias = layer.bias.detach().to(self.compute_device)
        return F.conv2d(value, weight, bias, stride=1, padding=1)

    def _linear(self, name, value):
        layer = getattr(self.model, name)
        if self.model_device == self.compute_device:
            return layer(value)
        return F.linear(value, layer.weight.detach().to(self.compute_device))

    @staticmethod
    def _act(value):
        return F.leaky_relu(value, negative_slope=0.1)

    def _forward(self, image, param, param1):
        x = self._conv("conv1", image)
        x = self._act(
            x + self._linear("dense1", param)[:, :, None, None]
            + self._linear("dense2", param1)[:, :, None, None])

        x = self._act(self._conv("e0_conv0", x))
        e0 = self._act(self._conv("e0_conv1", x))
        x = F.avg_pool2d(e0, kernel_size=2, stride=2, padding=0)
        x = self._act(self._conv("e1_conv0", x))
        e1 = self._act(self._conv("e1_conv1", x))
        x = F.avg_pool2d(e1, kernel_size=2, stride=2, padding=0)
        x = self._act(self._conv("e2_conv0", x))
        e2 = self._act(self._conv("e2_conv1", x))
        x = F.avg_pool2d(e2, kernel_size=2, stride=2, padding=0)
        x = self._act(self._conv("e3_conv0", x))
        e3 = self._act(self._conv("e3_conv1", x))
        x = F.avg_pool2d(e3, kernel_size=2, stride=2, padding=0)
        x = self._act(self._conv("e4_conv0", x))
        e4 = self._act(self._conv("e4_conv1", x))
        x = F.avg_pool2d(e4, kernel_size=2, stride=2, padding=0)
        x = self._act(self._conv("center_conv0", x))
        x = self._act(self._conv("center_conv1", x))
        x = self._act(self._conv("center_conv2", x))
        x = self._act(self._conv("center_conv3", x))

        x = torch.cat((legacy_resize2d(x, scale_factor=2), e4), dim=1)
        x = self._act(self._conv("d4_conv0", x))
        x = self._act(self._conv("d4_conv1", x))
        x = torch.cat((legacy_resize2d(x, scale_factor=2), e3), dim=1)
        x = self._act(self._conv("d3_conv0", x))
        x = self._act(self._conv("d3_conv1", x))
        x = torch.cat((legacy_resize2d(x, scale_factor=2), e2), dim=1)
        x = self._act(self._conv("d2_conv0", x))
        x = self._act(self._conv("d2_conv1", x))
        x = torch.cat((legacy_resize2d(x, scale_factor=2), e1), dim=1)
        x = self._act(self._conv("d1_conv0", x))
        x = self._act(self._conv("d1_conv1", x))
        x = torch.cat((legacy_resize2d(x, scale_factor=2), e0), dim=1)
        x = self._act(self._conv("d0_conv0", x))
        d0 = self._act(self._conv("d0_conv1", x))

        x = self._act(self._conv("out1x_conv0", d0))
        out1x = image + torch.tanh(self._conv("out1x_conv1", x))
        x = self._act(self._conv("dec2x_conv0", d0))
        d2x = legacy_resize2d(
            self._act(self._conv("dec2x_conv1", x)), scale_factor=2)
        x = self._act(self._conv("out2x_conv0", d2x))
        out2x = (legacy_resize2d(out1x, scale_factor=2)
                 + torch.tanh(self._conv("out2x_conv1", x)))
        x = self._act(self._conv("dec4x_conv0", d2x))
        d4x = legacy_resize2d(
            self._act(self._conv("dec4x_conv1", x)), scale_factor=2)
        x = self._act(self._conv("out4x_conv0", d4x))
        return (legacy_resize2d(out2x, scale_factor=2)
                + torch.tanh(self._conv("out4x_conv1", x)))

    def infer_raw(self, batch_nhwc):
        """Run the raw network: float32 NHWC 192x192 -> NHWC 768x768."""
        value = np.asarray(batch_nhwc)
        if value.dtype != np.float32:
            raise TypeError("FaceEnhancer raw input must be float32")
        if value.ndim != 4 or value.shape[1:] != (192, 192, 3):
            raise ValueError(
                "FaceEnhancer raw input must have shape (N,192,192,3)")
        tensor = torch.from_numpy(
            np.ascontiguousarray(value.transpose(0, 3, 1, 2)))
        tensor = tensor.to(self.compute_device)
        param = torch.full((len(value), 1), 0.2, dtype=torch.float32,
                           device=self.compute_device)
        param1 = torch.ones((len(value), 1), dtype=torch.float32,
                            device=self.compute_device)
        with torch.inference_mode():
            if self.compute_device.type == "cuda":
                # Ampere+ cuDNN enables TF32 by default; its reduced mantissa
                # is visibly outside the authenticated FP32 oracle. The
                # context restores the caller's backend flags on exit.
                with torch.backends.cudnn.flags(
                        enabled=True, benchmark=False, deterministic=True,
                        allow_tf32=False):
                    result = self._forward(tensor, param, param1)
            else:
                result = self._forward(tensor, param, param1)
        return np.ascontiguousarray(
            result.cpu().numpy().transpose(0, 2, 3, 1), dtype=np.float32)

    @staticmethod
    def _patch_coordinates(length):
        limit = length - PATCH_SIZE + 1
        result = []
        position = 0
        while position < limit:
            result.append(position)
            if position == limit - 1:
                break
            position = min(position + OVERLAP, limit - 1)
        return result

    @staticmethod
    def _patch_mask():
        line = np.concatenate([
            np.linspace(0, 1, PATCH_SIZE // 2 * UPSCALE),
            np.linspace(1, 0, PATCH_SIZE // 2 * UPSCALE),
        ])
        xx, yy = np.meshgrid(line, line)
        return (xx * yy)[..., None]

    def _run_pipeline(self, inp_img, is_tanh=False, preserve_size=True):
        image = np.asarray(inp_img)
        if image.dtype != np.float32:
            raise TypeError("FaceEnhancer input must be float32")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("FaceEnhancer input must have shape (H,W,3)")
        original_height, original_width, channels = image.shape
        if original_height <= 0 or original_width <= 0:
            raise ValueError("FaceEnhancer input dimensions must be positive")

        processed = (image if is_tanh else
                     np.clip(image * 2 - 1, -1, 1).astype(np.float32))
        top = bottom = left = right = 0
        if original_height < PATCH_SIZE:
            top = (PATCH_SIZE - original_height) // 2
            bottom = PATCH_SIZE - original_height - top
        if original_width < PATCH_SIZE:
            left = (PATCH_SIZE - original_width) // 2
            right = PATCH_SIZE - original_width - left
        padded = np.pad(
            processed, ((top, bottom), (left, right), (0, 0)),
            mode="constant").astype(np.float32)
        height, width, _ = padded.shape

        accumulated = np.zeros(
            (height * UPSCALE, width * UPSCALE, channels), np.float32)
        divisor = np.zeros(
            (height * UPSCALE, width * UPSCALE, 1), np.float32)
        mask = self._patch_mask()
        for y in self._patch_coordinates(height):
            for x in self._patch_coordinates(width):
                patch = padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                output = self.infer_raw(patch[None, ...])[0]
                oy, ox = y * UPSCALE, x * UPSCALE
                accumulated[oy:oy + PATCH_SIZE * UPSCALE,
                            ox:ox + PATCH_SIZE * UPSCALE] += output * mask
                divisor[oy:oy + PATCH_SIZE * UPSCALE,
                        ox:ox + PATCH_SIZE * UPSCALE] += mask
        divisor_safe = divisor.copy()
        divisor_safe[divisor_safe == 0] = 1.0
        stitched = accumulated / divisor_safe
        cropped = stitched[
            top * UPSCALE:(height - bottom) * UPSCALE,
            left * UPSCALE:(width - right) * UPSCALE]
        preserved = (cv2.resize(
            cropped, (original_width, original_height),
            interpolation=cv2.INTER_LANCZOS4)
            if preserve_size else cropped)
        final = (preserved if is_tanh else
                 np.clip(preserved / 2 + 0.5, 0, 1))
        return {
            "preprocessed": processed,
            "padded": padded,
            "accumulated": accumulated,
            "divisor": divisor,
            "stitched": stitched,
            "cropped": cropped,
            "preserved": preserved,
            "final": np.asarray(final, dtype=np.float32),
            "padding": (top, bottom, left, right),
            "x_coordinates": self._patch_coordinates(width),
            "y_coordinates": self._patch_coordinates(height),
        }

    def enhance(self, inp_img, is_tanh=False, preserve_size=True):
        return self._run_pipeline(
            inp_img, is_tanh=is_tanh,
            preserve_size=preserve_size)["final"]
