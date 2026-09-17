"""Test-only dummy model for the Phase 5 lifecycle tests.

NOT a production architecture: it exists to exercise the model
foundation (component registration, the model-owned training step, the
Phase 4 checkpoint integration, save/resume) with the smallest
deterministic vehicle — two convs, one AdaBelief optimizer, a fixed
sample batch, zero RNG. It follows the official calling pattern exactly
(``on_initialize`` builds the components; ``onTrainOneIter`` consumes
its own samples and runs the native-torch step; ``onSave`` saves the
official ``[[model, filename], ...]`` pairs through the Phase 4
Saveable). See ``tests/smoke/test_model_lifecycle.py``.
"""

from pathlib import Path

import numpy as np
import torch

from core.leras import nn as dfl_nn
from models import ModelBase
from samplelib import SampleGeneratorBase


class DummyNet(dfl_nn.ModelBase):
    """Two convs (3->4->2, 3x3 SAME, NHWC), zero-initializer weights —
    the training step overwrites them with a deterministic per-parameter
    constant, so the whole test is zero-RNG and EXACT-comparable."""

    def on_build(self):
        self.conv1 = dfl_nn.Conv2D(
            3, 4, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros, name="conv1")
        self.conv2 = dfl_nn.Conv2D(
            4, 2, kernel_size=3, padding="SAME", use_bias=True,
            kernel_initializer=dfl_nn.initializers.zeros,
            bias_initializer=dfl_nn.initializers.zeros, name="conv2")

    def forward(self, x):
        return self.conv2(self.conv1(x))


class FakeGenerator(SampleGeneratorBase):
    """Deterministic fixed batch (no RNG, no dataset, no TF)."""

    def __init__(self):
        super().__init__(debug=False, batch_size=2)

    def is_initialized(self):
        return True

    def generate_next(self):
        a = (np.arange(2 * 8 * 8 * 3) % 5) * 0.1
        return a.reshape(2, 8, 8, 3).astype(np.float32)


class DummyModel(ModelBase):
    """Test-only model (official calling pattern): owns its leras
    container, optimizer and sample generator; the training step is
    model-owned (native torch autograd + the official get_update_op
    contract); components save through the Phase 4 Saveable."""

    def on_initialize_options(self):
        # direct assignment (deterministic — the official ask_* prompts
        # would be interactive); on resume the same constants are
        # re-applied to the restored options (no drift)
        self.options['dummy_alpha'] = 0.25
        self.batch_size = self.options['batch_size'] = 2

    def on_initialize(self):
        self.net = DummyNet(name="dummy_net")
        # official pattern (SAEHD on_initialize): get_weights() auto-builds
        # the container and yields the saveable weights the optimizer
        # registers (the official calls this BEFORE initialize_variables)
        saveable_weights = self.net.get_weights()
        # deterministic, non-trivial first-run weights (zero RNG
        # anywhere) — the values the official `model.init_weights()`
        # first-run branch would have produced
        torch = dfl_nn.torch
        with torch.no_grad():
            for i, p in enumerate(self.net.parameters(), start=1):
                p.copy_(torch.full(p.shape, i * 0.01,
                                   dtype=p.dtype, device=p.device))
        self.dummy_opt = dfl_nn.AdaBelief(lr=0.01, name="dummy_opt")
        self.dummy_opt.initialize_variables(saveable_weights)

        # official load/init loop (SAEHD lines 637-657): on a resume,
        # load each component from disk. The official code re-initialized
        # a component silently when its file was missing (`do_init = not
        # model.load_weights(...)`); the Phase 4/5 strict policy rejects
        # that for required state: a missing component file on resume
        # fails explicitly (a corrupted file already fails strictly in
        # the Phase 4 load engine)
        for model, filename in self.get_model_filename_list():
            if self.is_first_run():
                continue  # first-run values already in place
            if not model.load_weights(self.get_strpath_storage_for_file(filename)):
                raise FileNotFoundError(
                    f"required component file missing on resume: "
                    f"{self.get_strpath_storage_for_file(filename)}")

        self.set_training_data_generators([FakeGenerator()])

    def onTrainOneIter(self):
        # official pattern: the model consumes its own samples (the base
        # train_one_iter does not generate samples)
        sample = self.generate_next_samples()
        torch = dfl_nn.torch
        x = torch.from_numpy(np.ascontiguousarray(sample[0])).to(
            dfl_nn.device, dtype=torch.float32)
        out = self.net(x)
        loss = torch.nn.functional.mse_loss(out, torch.full_like(out, 0.5))
        # torch adaptation: the official TF graph re-computed FRESH
        # gradients on every session run; torch's ``backward()``
        # ACCUMULATES into ``param.grad`` — the model code must clear
        # the gradients of the previous iteration first, or the step
        # would train on the sum of all past gradients (this exact
        # hazard breaks resume equivalence: a reloaded model has clean
        # gradients while a continuing model accumulates)
        for p in self.net.parameters():
            p.grad = None
        loss.backward()
        params = list(self.net.parameters())
        self.dummy_opt.get_update_op([(p.grad, p) for p in params])()
        return [("loss", loss.item())]

    def get_model_filename_list(self):
        # official contract: [ [model, filename], ... ] pairs
        return [(self.net, "dummy_net.npy"), (self.dummy_opt, "dummy_opt.npy")]

    def onSave(self):
        for model, filename in self.get_model_filename_list():
            model.save_weights(self.get_strpath_storage_for_file(filename))

    def onGetPreview(self, sample, for_history=False):
        return []


class DummyModelNoGen(DummyModel):
    """Forwards everything but never sets training-data generators —
    the official lifecycle must reject a training model without
    generators (documented official quirk: an UNSET generator_list
    raises AttributeError; a SET non-conforming list raises ValueError)."""

    def on_initialize(self):
        self.net = DummyNet(name="dummy_net")
        self.dummy_opt = dfl_nn.AdaBelief(lr=0.01, name="dummy_opt")
        self.dummy_opt.initialize_variables(list(self.net.parameters()))
        # no set_training_data_generators call


class DummyModelNoSnapshot(DummyModel):
    """Uses the adopted ``disable_default_options_autosave()`` hook (the
    USER_LEGACY escape hatch that AMP's torch model uses) to opt out of
    the first-run ``default_options.dat`` snapshot."""

    def on_initialize_options(self):
        super().on_initialize_options()
        self.disable_default_options_autosave()


class DummyModelBadGen(DummyModel):
    """Sets a generator that is NOT a SampleGeneratorBase — the
    official isinstance validation must reject it."""

    def on_initialize(self):
        super().on_initialize()
        self.set_training_data_generators([object()])


def make_model(model_class, tmpdir, **device_kwargs):
    """Construct a dummy model instance under ``tmpdir`` (the official
    lifecycle constructor; ``cpu_only=True`` / ``force_gpu_idxs=[0]``
    select the device without any interactive prompt)."""
    return model_class(
        is_training=True,
        saved_models_path=Path(tmpdir),
        training_data_src_path=Path(tmpdir) / "src",
        training_data_dst_path=Path(tmpdir) / "dst",
        pretraining_data_path=None,
        pretrained_model_path=None,
        force_model_class_name="dummy_Dummy",
        **device_kwargs,
    )


def snapshot(model):
    """Parameters (in order), optimizer state ([iters] + ms + vs), the
    model-level iter and the loss history."""
    params = [p.detach().cpu().clone() for p in model.net.parameters()]
    states = [t.detach().cpu().clone() for t in model.dummy_opt.get_weights()]
    return {
        "params": params,
        "states": states,
        "iter": model.get_iter(),
        "losses": [list(x) for x in model.loss_history],
    }
