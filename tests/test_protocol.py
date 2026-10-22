from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from vcfdconformer.config import load_config, validate_config
from vcfdconformer.data import VariableContextDataset, collate_variable_context
from vcfdconformer.layouts import CANONICAL_LAYOUTS
from vcfdconformer.evaluate import TABLE2_LAYOUTS
from vcfdconformer.losses import (
    log_spectral_distortion,
    spectral_gradient_loss,
    training_objective,
)
from vcfdconformer.metrics import broadband_ild, ild_error


ROOT = Path(__file__).parents[1]


def test_paper_config_and_disjoint_splits():
    config = load_config(ROOT / "configs" / "paper.yaml")
    assert config["data"]["selection_train_subjects"] == [0, 160]
    assert config["data"]["selection_valid_subjects"] == [160, 180]
    assert config["data"]["final_train_subjects"] == [0, 180]
    assert config["data"]["test_subjects"] == [180, 200]
    invalid = deepcopy(config)
    invalid["data"]["test_subjects"] = [179, 200]
    with pytest.raises(ValueError, match="overlap"):
        validate_config(invalid)


def test_q64_adamw_paper_config():
    config = load_config(ROOT / "configs" / "paper.yaml")
    assert config["sampling"]["n_query"] == 64
    assert config["training"]["batch_size"] == 8
    assert config["training"]["optimizer"] == "adamw"
    assert config["training"]["learning_rate"] == 1e-3
    assert config["training"]["weight_decay"] == 1e-4
    assert config["training"]["validation_every_epochs"] == 20
    assert config["training"]["early_stopping_start_epoch"] == 400
    assert config["training"]["early_stopping_patience_checks"] == 2
    assert config["training"]["early_stopping_min_delta"] == 0.01


def test_canonical_layout_sizes_and_unique_indices():
    for size, indices in CANONICAL_LAYOUTS.items():
        assert len(indices) == size
        assert len(np.unique(indices)) == size
        assert indices.min() >= 0 and indices.max() < 793


def test_table2_fixed_layouts():
    import json

    manifest = json.loads((ROOT / "protocols" / "table2_m19_layouts.json").read_text())
    assert manifest["M"] == 19
    for name in TABLE2_LAYOUTS:
        indices = np.asarray(manifest[name]["sonicom_indices"])
        assert indices.shape == (19,)
        assert len(np.unique(indices)) == 19
        assert indices.min() >= 0 and indices.max() < 793


def test_loss_definitions_match_manual_values():
    target = torch.zeros(1, 1, 2, 3)
    prediction = torch.tensor([[[[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]]])
    expected_lsd = (torch.tensor(1.0) + torch.sqrt(torch.tensor(5.0 / 3.0))) / 2.0
    assert torch.allclose(log_spectral_distortion(prediction, target), expected_lsd)
    assert torch.allclose(spectral_gradient_loss(prediction, target), torch.tensor(0.5))
    total, components = training_objective(prediction, target)
    assert torch.allclose(total, expected_lsd + 0.5)
    assert set(components) == {"lsd", "sgl"}


def test_broadband_ild_and_error():
    flat = torch.zeros(2, 4, 2, 5)
    assert torch.allclose(broadband_ild(flat), torch.zeros(2, 4))
    louder_left = flat.clone()
    louder_left[..., 0, :] = 6.0
    assert torch.allclose(broadband_ild(louder_left), torch.full((2, 4), 6.0), atol=1e-5)
    assert torch.allclose(ild_error(louder_left, flat), torch.tensor(6.0), atol=1e-5)


def test_sampling_excludes_context_and_collates_variable_sizes(monkeypatch):
    hrtfs = np.arange(4 * 793 * 2 * 6, dtype=np.float32).reshape(4, 793, 2, 6)
    directions = np.zeros((793, 3), dtype=np.float32)
    directions[:, 0] = 1.0
    sampling = {
        "canonical_sizes": [3, 5, 19, 100],
        "canonical_probability": 0.0,
        "random_context_min": 3,
        "random_context_max": 100,
        "n_query": 32,
    }
    dataset = VariableContextDataset(hrtfs, directions, [0, 1], sampling, 42, True)
    first = dataset[0]
    second = dataset[1]
    batch = collate_variable_context([first, second])
    assert batch[0].shape[0] == 2 and batch[2].dtype == torch.bool
    assert batch[3].shape == (2, 32, 2, 6)
    dataset.set_epoch(1)
    changed = dataset[0]
    assert not torch.equal(first[0], changed[0]) or not torch.equal(first[2], changed[2])
