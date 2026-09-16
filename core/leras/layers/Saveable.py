"""Saveable — official-format (re)serialization foundation (Phase 3A, torch).

Contract (official DFL behavior preserved):
- a Saveable owns a scope name (``self.name``, e.g. 'encoder', 'GAN');
- ``get_weights()`` returns the weights to initialize/load/save
  (torch parameters/buffers in the torch leras);
- ``save_weights(filename, force_dtype)`` writes the official DFL
  checkpoint format: a pickled ``dict[str, np.ndarray]`` (protocol 4)
  with keys ``{sub_name}:0`` (scope prefix stripped), written atomically
  via ``core.pathex.write_bytes_safe``;
- ``load_weights(filename)`` returns True when the file exists and is
  loaded, False when the file is absent;
- ``init_weights()`` runs the initialization lifecycle
  (``nn.init_weights``).

Torch enumeration: weights are enumerated deterministically from the
module's registered ``named_parameters`` / ``named_buffers`` (registration
order — stable across repeated construction of the same architecture),
so the torch checkpoint key of a weight is its official DFL name
(``core.leras.checkpoint.official_name``: dotted name -> slashed name
with the ':0' suffix). Official DFL parameter names
(``weight``/``bias``/``running_mean``/``running_var``) are identical in
the torch leras, which is what makes future bidirectional Compatibility
Mode (official .npy round-trip, Phase 4) possible.

Strict load policy (IMPLEMENTATION_PLAN_v2.md sections 19 and 42 — the
official defects must not be reintroduced):
- a missing weight key is an error (official silently re-initialized);
- an unexpected extra key is an error;
- a shape mismatch is an error (official reshaped greedily);
- loading is all-or-nothing: everything is validated before anything
  is copied. Layout differences between the official checkpoint layout
  and the torch layout are resolved through the per-layer
  ``convert_weight_layout`` hook (identity in Phase 3A; real conversions
  land with the layers / Phase 4 converter).
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from core import pathex
from core.leras import nn
from core.leras import checkpoint as ckpt


class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return torch parameters/buffers that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        return [
            # numpy has no bfloat16; store it as float32 (External A concept)
            w.detach().cpu().to(torch.float32).numpy() if w.dtype == torch.bfloat16
            else w.detach().cpu().numpy()
            for w in weights
        ]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError('len of lists mismatch')

        for w, new_w in zip(weights, new_weights):
            if isinstance(new_w, torch.Tensor):
                src = new_w.detach()
            elif isinstance(new_w, np.ndarray):
                src = torch.from_numpy(np.ascontiguousarray(new_w))
            else:
                src = torch.as_tensor(new_w)

            # Strict: the official implementation reshaped greedily on any
            # shape difference; that is removed (v2 section 42).
            if tuple(src.shape) != tuple(w.shape):
                raise ValueError(
                    f"set_weights shape mismatch: parameter {w} has shape "
                    f"{tuple(w.shape)}, value has shape {tuple(src.shape)}; "
                    f"refusing to reshape silently"
                )
            with torch.no_grad():
                w.copy_(src.to(device=w.device, dtype=w.dtype))

    # --- official-format serialization --------------------------------

    def _iter_official_weights(self):
        """Yield (official_sub_name, tensor) in deterministic order:
        registered parameters first, then buffers — both in torch
        registration order, which is stable across repeated construction
        of the same architecture.

        The sub-name is the checkpoint key relative to this Saveable's
        scope (``self.name``): for a layer it is the plain parameter
        name with the ':0' suffix ('weight:0'); for a module tree it is
        the dotted path converted to slashes ('conv1_1/weight:0').
        """
        if not isinstance(self, torch.nn.Module):
            raise TypeError(
                f"{type(self).__name__} is not a torch module: torch "
                f"Saveables enumerate registered parameters/buffers"
            )
        items = []
        for name, param in self.named_parameters(recurse=True):
            items.append((ckpt.official_name(name), param))
        for name, buf in self.named_buffers(recurse=True):
            items.append((ckpt.official_name(name), buf))
        return items

    def save_weights(self, filename, force_dtype=None):
        if self.name is None:
            raise Exception("name must be defined.")

        d = {}
        for sub_name, w in self._iter_official_weights():
            arr = w.detach().cpu()
            if arr.dtype == torch.bfloat16:
                arr = arr.to(torch.float32)
            arr = arr.numpy().copy()
            if force_dtype is not None:
                arr = arr.astype(force_dtype)
            d[sub_name] = arr

        d_dumped = pickle.dumps(d, 4)
        pathex.write_bytes_safe(Path(filename), d_dumped)

    def load_weights(self, filename):
        """
        returns True if the file exists (and loads strictly), False if not.
        A file that exists but cannot be loaded STRICTLY raises
        CheckpointLoadError (no silent recovery — v2 section 19).
        """
        filepath = Path(filename)
        if not filepath.exists():
            return False

        d = pickle.loads(filepath.read_bytes())

        if self.name is None:
            raise Exception("name must be defined.")

        items = self._iter_official_weights()

        # --- Pass 1: validate everything (missing / extra / shape) ---
        problems = []
        planned = []
        for sub_name, param in items:
            value, matched_key = ckpt._lookup_key(d, sub_name)
            if value is None:
                problems.append(
                    f"missing weight '{sub_name}' in {filename} "
                    f"(official DFL re-initialized it silently; this project "
                    f"must not)"
                )
                continue

            value = self.convert_weight_layout(value, param)
            if tuple(np.asarray(value).shape) != tuple(param.shape):
                problems.append(
                    f"shape mismatch for '{sub_name}': file "
                    f"{np.asarray(value).shape} != parameter {tuple(param.shape)}"
                )
                continue

            planned.append((matched_key, value, param))

        expected_stripped = {
            ckpt.strip_zero_suffix(ckpt.official_name(sub)) for sub, _ in items
        }
        for key in d:
            if ckpt.strip_zero_suffix(key) not in expected_stripped:
                problems.append(f"unexpected extra key '{key}' in {filename}")

        if problems:
            raise ckpt.CheckpointLoadError(
                f"strict load of {filename} failed:\n  " + "\n  ".join(problems)
            )

        # --- Pass 2: apply (all-or-nothing) ---
        for matched_key, value, param in planned:
            value = np.ascontiguousarray(value)
            with torch.no_grad():
                param.copy_(
                    torch.from_numpy(value).to(device=param.device, dtype=param.dtype)
                )

        return True

    def convert_weight_layout(self, value, param):
        """Convert a loaded checkpoint array to the torch layout of
        ``param``. Identity in Phase 3A: the official conv kernel layout
        (H,W,in,out) / dense (in,out) conversions are implemented by the
        layer classes (Phase 3B) and the Phase 4 converter. Overriding
        this hook is how a layer declares its layout difference — the
        hook is the clean path for Phase 4, no ad-hoc reshaping elsewhere.
        """
        return value

    def get_param_initializers(self):
        """Initializer registered per parameter (torch leras lifecycle).
        Plain (non-module) Saveables register none; LayerBase
        implements it via ``register_param_initializer``."""
        return {}

    def init_weights(self):
        # Official contract: run this saveable's weight initialization.
        # Torch form: apply the per-parameter initializers registered by
        # build_weights (nn.init_weights); modules composing sub-layers
        # (archis, Phase 3B) override this to cascade to their children,
        # exactly like the official archis.
        nn.init_weights(self)


nn.Saveable = Saveable
