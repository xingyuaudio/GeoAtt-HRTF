"""Evaluate GeoAtt on the fixed protocols from Tables 1 and 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import apply_data_overrides, validate_config
from .data import load_external_arrays
from .layouts import canonical_layout
from .metrics import evaluation_metrics
from .reproducibility import build_model, save_json


@torch.no_grad()
def predict_chunks(model, context_hrtf, context_dirs, query_dirs, chunk_size: int):
    mask = torch.ones(context_hrtf.size(0), context_hrtf.size(1), dtype=torch.bool,
                      device=context_hrtf.device)
    return torch.cat([
        model(context_hrtf, context_dirs, mask, query_dirs[:, start:start + chunk_size])
        for start in range(0, query_dirs.size(1), chunk_size)
    ], dim=1)


def load_checkpoint(path: str, device: torch.device, hrtf_path: str | None,
                    directions_path: str | None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "VCFDConformer-Direct" or checkpoint.get("stage") != "final":
        raise ValueError("evaluation requires a clean final-stage VCFDConformer-Direct checkpoint")
    config = apply_data_overrides(checkpoint["config"], hrtf_path, directions_path)
    validate_config(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), config


def evaluate_one(model, hrtfs, directions, subject: int, context: np.ndarray,
                 device: torch.device, chunk_size: int) -> dict[str, float]:
    unseen = np.setdiff1d(np.arange(len(directions)), context, assume_unique=True)
    context_hrtf = torch.from_numpy(np.array(hrtfs[subject, context], dtype=np.float32, copy=True))[None].to(device)
    context_dirs = torch.from_numpy(directions[context])[None].to(device)
    query_dirs = torch.from_numpy(directions[unseen])[None].to(device)
    target = torch.from_numpy(np.array(hrtfs[subject, unseen], dtype=np.float32, copy=True))[None].to(device)
    prediction = predict_chunks(model, context_hrtf, context_dirs, query_dirs, chunk_size)
    return evaluation_metrics(prediction, target)


def summarize(records: list[dict]) -> dict:
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in records])),
            "sample_std": float(np.std([row[metric] for row in records], ddof=1)),
        }
        for metric in ("LSD", "ILD")
    }


def evaluate_canonical(model, config, hrtfs, directions, device) -> dict:
    records = {}
    test_subjects = range(*config["data"]["test_subjects"])
    chunk = config["evaluation"]["query_chunk_size"]
    for size in config["sampling"]["canonical_sizes"]:
        rows = []
        context = canonical_layout(size)
        for subject in test_subjects:
            rows.append({"subject": subject, **evaluate_one(
                model, hrtfs, directions, subject, context, device, chunk
            )})
        records[str(size)] = {"per_subject": rows, "summary": summarize(rows)}
    return {
        "protocol": "held-out listeners 180:200; canonical layouts; context directions excluded from metrics",
        "results": records,
    }


TABLE2_LAYOUTS = ("lebedev_based", "t_design_based", "hemisphere", "horizontal_ring")


def evaluate_table2(model, config, hrtfs, directions, device, layouts_path: str | Path) -> dict:
    layouts_path = Path(layouts_path)
    with layouts_path.open("r", encoding="utf-8") as handle:
        layouts = json.load(handle)
    if layouts.get("M") != 19:
        raise ValueError("Table 2 layout manifest must have M=19")
    records = {}
    for name in TABLE2_LAYOUTS:
        context = np.asarray(layouts[name]["sonicom_indices"], dtype=np.int64)
        if (context.shape != (19,) or len(np.unique(context)) != 19
                or np.any(context < 0) or np.any(context >= len(directions))):
            raise ValueError(f"invalid Table 2 layout: {name}")
        rows = []
        for subject in range(*config["data"]["test_subjects"]):
            rows.append({"subject": subject, **evaluate_one(
                model, hrtfs, directions, subject, context, device,
                config["evaluation"]["query_chunk_size"],
            )})
        records[name] = {"context_indices": context.tolist(),
                         "per_subject": rows, "summary": summarize(rows)}
    return {
        "protocol": "held-out listeners 180:200; fixed Table 2 M=19 layouts; context directions excluded",
        "layouts_manifest": str(layouts_path.resolve()),
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", choices=("table1", "table2", "all"), default="all")
    parser.add_argument("--layouts", default="protocols/table2_m19_layouts.json",
                        help="fixed M=19 layout manifest for Table 2")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--hrtf-path")
    parser.add_argument("--directions-path")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device, args.hrtf_path, args.directions_path)
    hrtfs, directions = load_external_arrays(config["data"])
    output_dir = Path(args.output_dir)
    if args.protocol in ("table1", "all"):
        result = evaluate_canonical(model, config, hrtfs, directions, device)
        result["checkpoint"] = str(Path(args.checkpoint).resolve())
        save_json(result, output_dir / "table1_canonical.json")
        print({size: value["summary"] for size, value in result["results"].items()}, flush=True)
    if args.protocol in ("table2", "all"):
        result = evaluate_table2(model, config, hrtfs, directions, device, args.layouts)
        result["checkpoint"] = str(Path(args.checkpoint).resolve())
        save_json(result, output_dir / "table2_m19.json")
        print({key: value["summary"] for key, value in result["results"].items()}, flush=True)


if __name__ == "__main__":
    main()
