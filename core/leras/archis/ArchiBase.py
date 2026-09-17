"""ArchiBase — official leras contract (Phase 3F, torch).

Official behavior preserved (verbatim API, see the dead official reference
in ``archis_tf.py``):
- a plain class (NOT a torch module, NOT a Saveable — the official
  ``ArchiBase`` is a plain class too): a ``name`` attribute set by the
  constructor, ``flow()`` that raises, ``get_weights()`` that returns
  None (overridable);
- archis are factory/namespace objects that expose sub-architecture
  classes (e.g. ``DeepFakeArchi.Encoder/Inter/Decoder``); the official
  models instantiate those classes and register the instances as their
  own attributes (``self.encoder = model_archi.Encoder(..., name=...)``);
- in the torch leras those sub-classes are ``torch.nn.Module``-based
  saveables (Phase 3A ``LayerBase``), so the official calling pattern
  and the checkpoint naming (attribute path -> official DFL variable
  scope, ``core.leras.checkpoint.official_name``) are unchanged.
"""

from core.leras import nn


class ArchiBase():

    def __init__(self, *args, name=None, **kwargs):
        self.name=name


    #overridable
    def flow(self, *args, **kwargs):
        raise Exception("this archi does not support flow. Use model classes directly.")

    #overridable
    def get_weights(self):
        pass


nn.ArchiBase = ArchiBase
