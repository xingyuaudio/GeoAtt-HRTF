"""External SONICOM arrays and variable-context task sampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .layouts import canonical_layout


def azimuth_elevation_to_unit(angles_degrees: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(angles_degrees[..., 0])
    elevation = np.deg2rad(angles_degrees[..., 1])
    return np.stack((
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ), axis=-1).astype(np.float32)


def load_external_arrays(data_config: dict, require_hrtf: bool = True) -> tuple[np.ndarray | None, np.ndarray]:
    hrtf_path = Path(data_config["hrtf_path"])
    directions_path = Path(data_config["directions_path"])
    if require_hrtf and not hrtf_path.is_file():
        raise FileNotFoundError(f"external HRTF array not found: {hrtf_path}")
    if not directions_path.is_file():
        raise FileNotFoundError(f"external direction array not found: {directions_path}")
    hrtfs = np.load(hrtf_path, mmap_mode="r") if require_hrtf else None
    raw_directions = np.load(directions_path, mmap_mode="r")
    angles = raw_directions[0] if raw_directions.ndim == 3 else raw_directions
    directions = azimuth_elevation_to_unit(angles)
    if directions.shape != (data_config["n_directions"], 3):
        raise ValueError(f"expected direction shape {(data_config['n_directions'], 3)}, got {directions.shape}")
    if hrtfs is not None:
        expected = (data_config["n_subjects"], data_config["n_directions"], 2, data_config["n_freq"])
        if hrtfs.shape != expected:
            raise ValueError(f"expected HRTF shape {expected}, got {hrtfs.shape}")
    return hrtfs, directions


class VariableContextDataset(Dataset):
    """One deterministic variable-context/query task per listener and epoch."""

    def __init__(
        self,
        hrtfs: np.ndarray,
        directions: np.ndarray,
        subject_ids: range | list[int] | np.ndarray,
        sampling: dict,
        seed: int,
        vary_by_epoch: bool,
    ) -> None:
        self.hrtfs = hrtfs
        self.directions = directions
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.sampling = sampling
        self.seed = int(seed)
        self.vary_by_epoch = vary_by_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _sample_context(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.sampling["canonical_probability"]:
            size = int(rng.choice(self.sampling["canonical_sizes"]))
            return canonical_layout(size)
        size = int(rng.integers(
            self.sampling["random_context_min"], self.sampling["random_context_max"] + 1
        ))
        return np.sort(rng.choice(len(self.directions), size, replace=False))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        epoch = self.epoch if self.vary_by_epoch else 0
        rng = np.random.default_rng(self.seed + 1_000_003 * epoch + index)
        context = self._sample_context(rng)
        remaining = np.setdiff1d(np.arange(len(self.directions)), context, assume_unique=True)
        query = np.sort(rng.choice(remaining, self.sampling["n_query"], replace=False))
        listener = self.hrtfs[self.subject_ids[index]]
        return (
            torch.from_numpy(np.array(listener[context], dtype=np.float32, copy=True)),
            torch.from_numpy(self.directions[context]),
            torch.from_numpy(np.array(listener[query], dtype=np.float32, copy=True)),
            torch.from_numpy(self.directions[query]),
        )


def collate_variable_context(batch: list[tuple[Tensor, Tensor, Tensor, Tensor]]):
    context_hrtfs, context_dirs, targets, query_dirs = zip(*batch)
    batch_size, max_context = len(batch), max(item.size(0) for item in context_hrtfs)
    n_freq = context_hrtfs[0].size(-1)
    padded_hrtfs = torch.zeros(batch_size, max_context, 2, n_freq)
    padded_dirs = torch.zeros(batch_size, max_context, 3)
    valid_mask = torch.zeros(batch_size, max_context, dtype=torch.bool)
    for row, (hrtf, direction) in enumerate(zip(context_hrtfs, context_dirs)):
        count = hrtf.size(0)
        padded_hrtfs[row, :count] = hrtf
        padded_dirs[row, :count] = direction
        valid_mask[row, :count] = True
    return padded_hrtfs, padded_dirs, valid_mask, torch.stack(targets), torch.stack(query_dirs)
