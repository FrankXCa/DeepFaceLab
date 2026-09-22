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

# Phase 10B: the official XSeg model foundation is torch (see XSeg.py;
# the official TF source is preserved verbatim in XSeg_tf.py, dead
# reference like ModelBase_tf.py) and is importable under both
# foundations, unguarded.
from .XSeg import *
