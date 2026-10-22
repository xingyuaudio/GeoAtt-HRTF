"""Determinism, model construction, and checkpoint helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .model import VariableContextFDConformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(config: dict) -> VariableContextFDConformer:
    model = config["model"].copy()
    direction_bands = model.pop("direction_bands")
    return VariableContextFDConformer(
        n_freq=config["data"]["n_freq"], direction_bands=direction_bands, **model
    )


def save_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def save_checkpoint(path: str | Path, model, config: dict, stage: str, epoch: int, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": "VCFDConformer-Direct",
        "stage": stage,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "config": config,
        **metadata,
    }, path)


def load_model_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "VCFDConformer-Direct":
        raise ValueError("checkpoint is not in the clean VCFDConformer-Direct format")
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
