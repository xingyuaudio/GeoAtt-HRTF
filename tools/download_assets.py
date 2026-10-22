"""Download the preprocessed SONICOM arrays and GeoAtt checkpoint."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ASSETS = {
    "dataset/H_log.npy": Path("data/sonicom/H_log.npy"),
    "dataset/locs.npy": Path("data/sonicom/locs.npy"),
    "model/geoatt_final_ep440.pt": Path("weights/geoatt_final_ep440.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="xingyuchen-8/GeoAtt-HRTF")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    for source, destination in ASSETS.items():
        cached = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=source,
            repo_type="model",
            cache_dir=cache_dir,
        ))
        target = output_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        print(f"{source} -> {target}")


if __name__ == "__main__":
    main()
