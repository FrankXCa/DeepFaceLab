# Phase 3A: torch leras foundation only.
#
# Saveable and LayerBase are the torch foundation (official contract,
# torch implementation). The concrete layer modules (Conv2D,
# Conv2DTranspose, DepthwiseConv2D, Dense, BatchNorm2D, InstanceNorm2D,
# FRNorm2D, BlurPool, TLU, ScaleAdd, DenseNorm, AdaIN, TanhPolar) are
# still the TensorFlow versions on disk and are recorded as dead code
# until they are rebuilt in Phase 3B; they are intentionally NOT
# imported here, so importing core.leras.layers never touches
# TensorFlow.
from .Saveable import *
from .LayerBase import *
