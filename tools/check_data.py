"""Validate the local SONICOM arrays before training or evaluation."""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrtf-path", required=True)
    parser.add_argument("--directions-path", required=True)
    args = parser.parse_args()
    hrtf = np.load(args.hrtf_path, mmap_mode="r")
    directions = np.load(args.directions_path, mmap_mode="r")
    if hrtf.shape != (200, 793, 2, 106):
        raise ValueError(f"unexpected HRTF shape: {hrtf.shape}")
    if directions.shape not in ((200, 793, 3), (1, 793, 3), (793, 3),
                                (1, 793, 2), (793, 2)):
        raise ValueError(f"unexpected direction shape: {directions.shape}")
    if directions.ndim == 3 and not np.all(
            directions[:, :, :2] == directions[0:1, :, :2]):
        raise ValueError("measurement-direction order differs across listeners")
    if not np.isfinite(hrtf).all() or not np.isfinite(directions).all():
        raise ValueError("data contain non-finite values")
    print(f"data_ok H_log={hrtf.shape} locs={directions.shape}")


if __name__ == "__main__":
    main()
