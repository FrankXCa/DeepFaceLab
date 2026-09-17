from core.leras import nn

# Phase 3F: the official discriminator classes (CodeDiscriminator,
# PatchDiscriminator) are torch and importable under both foundations.
from .PatchDiscriminator import *
from .CodeDiscriminator import *

# Phase 5: the official leras model container (nn.ModelBase) is torch and
# importable under both foundations (see ModelBase.py; the official TF
# source is preserved verbatim in ModelBase_tf.py as a dead reference,
# like discriminators_tf.py).
from .ModelBase import *

# The official TF XSeg model foundation executes ``tf = nn.tf`` at import
# time, so it is importable only when the TF foundation is active. Under
# the torch foundation (Phase 3A+) it is skipped here and is rebuilt in the
# XSeg model phase (see core/leras/nn.py).
if hasattr(nn, 'tf'):
    from .XSeg import *
