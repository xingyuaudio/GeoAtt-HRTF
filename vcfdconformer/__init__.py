"""GeoAtt: geometry-aware variable-context HRTF magnitude reconstruction."""

from .model import VariableContextFDConformer
GeoAtt = VariableContextFDConformer
from .losses import log_spectral_distortion, spectral_gradient_loss, training_objective
from .metrics import broadband_ild, ild_error

__all__ = [
    "GeoAtt",
    "VariableContextFDConformer",
    "log_spectral_distortion",
    "spectral_gradient_loss",
    "training_objective",
    "broadband_ild",
    "ild_error",
]
