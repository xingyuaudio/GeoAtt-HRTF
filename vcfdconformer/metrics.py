"""Unseen-direction evaluation metrics."""
from __future__ import annotations

import torch
from torch import Tensor

from .losses import log_spectral_distortion


def broadband_ild(log_magnitude_db: Tensor, eps: float = 1e-12) -> Tensor:
    """Broadband left-minus-right ILD for (..., 2, F) spectra."""
    if log_magnitude_db.size(-2) != 2:
        raise ValueError("ear dimension must have size 2")
    magnitude = torch.pow(10.0, log_magnitude_db / 20.0)
    level = 10.0 * torch.log10(magnitude.square().sum(dim=-1).clamp_min(eps))
    return level[..., 0] - level[..., 1]


def ild_error(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}")
    return torch.mean(torch.abs(broadband_ild(prediction) - broadband_ild(target)))


def evaluation_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    return {
        "LSD": float(log_spectral_distortion(prediction, target)),
        "ILD": float(ild_error(prediction, target)),
    }
