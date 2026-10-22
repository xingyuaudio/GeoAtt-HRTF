"""Download SONICOM FreeFieldCompMinPhase 48 kHz SOFA files P0001–P0200.

SONICOM may require access permission. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


URL = (
    "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/"
    "{pid}/HRTF/HRTF/48kHz/{pid}_FreeFieldCompMinPhase_48kHz.sofa"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for number in range(1, 201):
            pid = f"P{number:04d}"
            destination = args.output_dir / f"{pid}.sofa"
            if destination.is_file():
                continue
            partial = destination.with_suffix(".sofa.part")
            try:
                with session.get(URL.format(pid=pid), stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in tqdm(response.iter_content(chunk_size=1 << 20),
                                          desc=pid, unit="MiB", unit_scale=True):
                            if chunk:
                                handle.write(chunk)
                partial.replace(destination)
            except Exception:
                print(f"Download interrupted at {pid}; incomplete file: {partial}")
                raise


if __name__ == "__main__":
    main()
