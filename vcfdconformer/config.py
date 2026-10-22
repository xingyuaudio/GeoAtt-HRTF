"""Configuration loading and protocol validation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def apply_data_overrides(
    config: dict[str, Any], hrtf_path: str | None, directions_path: str | None
) -> dict[str, Any]:
    result = deepcopy(config)
    if hrtf_path:
        result["data"]["hrtf_path"] = hrtf_path
    if directions_path:
        result["data"]["directions_path"] = directions_path
    validate_config(result)
    return result


def _subject_ids(bounds: list[int]) -> set[int]:
    if len(bounds) != 2 or bounds[0] < 0 or bounds[0] >= bounds[1]:
        raise ValueError(f"invalid half-open subject range: {bounds}")
    return set(range(*bounds))


def validate_config(config: dict[str, Any]) -> None:
    data, sampling, model, training = (
        config["data"], config["sampling"], config["model"], config["training"]
    )
    selection_train = _subject_ids(data["selection_train_subjects"])
    selection_valid = _subject_ids(data["selection_valid_subjects"])
    final_train = _subject_ids(data["final_train_subjects"])
    test = _subject_ids(data["test_subjects"])
    if selection_train & selection_valid:
        raise ValueError("selection training and validation listeners overlap")
    if selection_train | selection_valid != final_train:
        raise ValueError("final training listeners must equal selection train + validation")
    if final_train & test:
        raise ValueError("final training and test listeners overlap")
    if max(final_train | test) >= data["n_subjects"]:
        raise ValueError("subject split exceeds n_subjects")
    if sampling["canonical_probability"] != 0.5:
        raise ValueError("experiment protocol requires canonical_probability=0.5")
    if sampling["canonical_sizes"] != [3, 5, 19, 100]:
        raise ValueError("experiment protocol requires canonical sizes [3, 5, 19, 100]")
    if (sampling["random_context_min"], sampling["random_context_max"]) != (3, 100):
        raise ValueError("experiment protocol requires random M in [3,100]")
    if sampling["n_query"] not in (32, 64):
        raise ValueError("supported experiment query counts are Q=32 and Q=64")
    expected_model = {"d_model": 128, "d_ff": 256, "n_heads": 8, "n_layers": 4,
                      "conv_kernel": 7, "dropout": 0.1, "direction_bands": 4}
    if model != expected_model:
        raise ValueError(f"model config differs from the final architecture: {model}")
    if training["batch_size"] <= 0 or training["learning_rate"] <= 0:
        raise ValueError("batch size and learning rate must be positive")
    if training.get("optimizer", "adam").lower() not in ("adam", "adamw"):
        raise ValueError("optimizer must be adam or adamw")
    if int(training.get("validation_every_epochs", 1)) <= 0:
        raise ValueError("validation_every_epochs must be positive")
    start_epoch = int(training.get("early_stopping_start_epoch", 0))
    if start_epoch < 0 or start_epoch >= int(training["max_selection_epochs"]):
        raise ValueError("early_stopping_start_epoch must be in [0, max_selection_epochs)")
    if int(training.get("early_stopping_patience_checks",
                        training.get("early_stopping_patience", 0))) < 0:
        raise ValueError("early-stopping patience must be non-negative")
