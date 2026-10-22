"""Training objective from the paper."""
from __future__ import annotations

import torch
from torch import Tensor


def log_spectral_distortion(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean per-query, per-ear RMS error over frequency."""
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}")
    return torch.sqrt(torch.mean((prediction - target).square(), dim=-1)).mean()


def spectral_gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean L1 error between adjacent-frequency differences."""
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}")
    return torch.mean(torch.abs(torch.diff(prediction, dim=-1) - torch.diff(target, dim=-1)))


def training_objective(prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    lsd = log_spectral_distortion(prediction, target)
    sgl = spectral_gradient_loss(prediction, target)
    return lsd + sgl, {"lsd": lsd.detach(), "sgl": sgl.detach()}
