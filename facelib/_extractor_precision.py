"""Precision scope for the CUDA-only S3FD and FAN inference forwards."""

from contextlib import contextmanager

import torch


@contextmanager
def cudnn_fp32_for_extractor(input_tensor):
    """Preserve the caller's cuDNN TF32 policy outside extractor inference.

    The cuDNN flag is process-wide while the forward runs. Extractor workers
    run inference serially in their own processes; this scope restores the
    caller's setting even if the model raises. No CPU or matmul flag changes.
    """
    if not input_tensor.is_cuda:
        yield
        return

    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous
