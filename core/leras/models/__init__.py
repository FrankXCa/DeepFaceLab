from core.leras import nn

# Phase 3F: the official discriminator classes (CodeDiscriminator,
# PatchDiscriminator, UNetPatchDiscriminator) are torch and importable
# under both foundations.
from .PatchDiscriminator import *
from .CodeDiscriminator import *

# The official TF model foundation (ModelBase, XSeg) executes ``tf = nn.tf``
# at import time, so it is importable only when the TF foundation is active.
# Under the torch foundation (Phase 3A+) it is skipped here and is rebuilt
# in Phases 6-8 (see core/leras/nn.py).
if hasattr(nn, 'tf'):
    from .ModelBase import *
    from .XSeg import *
