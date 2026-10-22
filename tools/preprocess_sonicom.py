"""Convert 200 SONICOM SOFA files to the exact GeoAtt NumPy data contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sofar
from scipy.fft import fft
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sonicom"))
    args = parser.parse_args()
    hrtfs, positions = [], []
    for number in tqdm(range(1, 201), desc="Preprocessing SONICOM"):
        pid = f"P{number:04d}"
        path = args.raw_dir / f"{pid}.sofa"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; download all 200 subjects first")
        sofa = sofar.read_sofa(str(path))
        hrir = np.asarray(sofa.Data_IR)
        source_positions = np.asarray(sofa.SourcePosition)
        if hrir.shape[:2] != (793, 2) or source_positions.shape != (793, 3):
            raise ValueError(f"unexpected SOFA shape for {pid}: {hrir.shape}, {source_positions.shape}")
        if hrir.shape[-1] != 256:
            raise ValueError(f"expected 256-sample HRIR for {pid}, got {hrir.shape[-1]}")
        spectrum = np.abs(fft(hrir, axis=-1))[..., 1:107]
        hrtfs.append((20.0 * np.log10(np.maximum(spectrum, 1e-8))).astype(np.float32))
        positions.append(source_positions.astype(np.float32))
    hrtf_array = np.stack(hrtfs)
    position_array = np.stack(positions)
    if not np.all(position_array[:, :, :2] == position_array[0:1, :, :2]):
        raise ValueError("measurement-direction order differs across listeners")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "H_log.npy", hrtf_array)
    np.save(args.output_dir / "locs.npy", position_array)
    print(f"H_log.npy {hrtf_array.shape}; locs.npy {position_array.shape}")


if __name__ == "__main__":
    main()
