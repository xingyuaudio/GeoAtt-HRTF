"""Two-stage paper training: 160/20 epoch selection, then fresh 180-listener retraining."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import apply_data_overrides, load_config
from .data import VariableContextDataset, collate_variable_context, load_external_arrays
from .losses import log_spectral_distortion, training_objective
from .reproducibility import build_model, save_checkpoint, save_json, seed_everything


def subject_range(bounds: list[int]) -> range:
    return range(bounds[0], bounds[1])


def make_loader(dataset, training: dict, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["num_workers"],
        collate_fn=collate_variable_context,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def make_optimizer(model, training: dict):
    name = training.get("optimizer", "adam").lower()
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[name]
    return optimizer_class(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )


def train_epoch(model, loader, optimizer, device, clip_norm: float) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "lsd": 0.0, "sgl": 0.0}
    examples = 0
    for context_hrtf, context_dirs, context_mask, target, query_dirs in loader:
        batch_size = context_hrtf.size(0)
        context_hrtf, context_dirs, context_mask = (
            context_hrtf.to(device), context_dirs.to(device), context_mask.to(device)
        )
        target, query_dirs = target.to(device), query_dirs.to(device)
        prediction = model(context_hrtf, context_dirs, context_mask, query_dirs)
        loss, components = training_objective(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        totals["loss"] += float(loss.detach()) * batch_size
        totals["lsd"] += float(components["lsd"]) * batch_size
        totals["sgl"] += float(components["sgl"]) * batch_size
        examples += batch_size
    return {name: value / examples for name, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total, examples = 0.0, 0
    for context_hrtf, context_dirs, context_mask, target, query_dirs in loader:
        batch_size = context_hrtf.size(0)
        prediction = model(
            context_hrtf.to(device), context_dirs.to(device), context_mask.to(device), query_dirs.to(device)
        )
        total += float(log_spectral_distortion(prediction, target.to(device))) * batch_size
        examples += batch_size
    return total / examples


def provenance(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_selection(config: dict, hrtfs, directions, device: torch.device, output_dir: Path,
                  resume_checkpoint: str | None = None,
                  resume_summary: str | None = None) -> int:
    seed = config["seed"]
    seed_everything(seed)
    train_data = VariableContextDataset(
        hrtfs, directions, subject_range(config["data"]["selection_train_subjects"]),
        config["sampling"], seed, vary_by_epoch=True,
    )
    valid_data = VariableContextDataset(
        hrtfs, directions, subject_range(config["data"]["selection_valid_subjects"]),
        config["sampling"], seed + 10_000_000, vary_by_epoch=False,
    )
    train_loader = make_loader(train_data, config["training"], True, seed)
    valid_loader = make_loader(valid_data, config["training"], False, seed)
    model = build_model(config).to(device)
    start_epoch, best_lsd, selected_epochs, history = 0, float("inf"), 0, []
    resume_metadata = None
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "VCFDConformer-Direct":
            raise ValueError("resume checkpoint is not a clean VCFDConformer-Direct checkpoint")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        start_epoch = int(checkpoint["epoch"])
        best_lsd = float(checkpoint.get("validation_lsd", float("inf")))
        selected_epochs = start_epoch
        resume_metadata = {
            "checkpoint": str(Path(resume_checkpoint).resolve()),
            "checkpoint_epoch": start_epoch,
            "optimizer_restored": False,
            "note": "Model-only warm start; Adam state was unavailable and was reset.",
        }
        if resume_summary:
            with Path(resume_summary).open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            history = [row for row in previous.get("history", [])
                       if int(row["epoch"]) <= start_epoch]
            best_lsd = float(previous["best_validation_LSD"])
            selected_epochs = int(previous["selected_epochs"])
        print({"resume": resume_metadata, "best_validation_LSD": best_lsd}, flush=True)
        save_checkpoint(output_dir / "selection_best.pt", model, config, "selection",
                        selected_epochs, validation_lsd=best_lsd,
                        warm_start=resume_metadata)
    optimizer = make_optimizer(model, config["training"])
    validation_every = int(config["training"].get("validation_every_epochs", 1))
    early_stopping_start = int(config["training"].get("early_stopping_start_epoch", 0))
    patience = int(config["training"].get(
        "early_stopping_patience_checks",
        config["training"].get("early_stopping_patience", 0),
    ))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    stalled_checks = 0
    last_epoch = start_epoch
    last_valid_lsd = best_lsd
    max_epochs = int(config["training"]["max_selection_epochs"])
    for epoch in range(start_epoch, max_epochs):
        train_data.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    config["training"]["gradient_clip_norm"])
        row = {"epoch": epoch + 1,
               **{f"train_{k}": v for k, v in train_metrics.items()}}
        should_validate = ((epoch + 1) % validation_every == 0 or epoch + 1 == max_epochs)
        if should_validate:
            last_valid_lsd = validate(model, valid_loader, device)
            row["validation_LSD"] = last_valid_lsd
        history.append(row)
        print(row, flush=True)
        last_epoch = epoch + 1
        if not should_validate:
            continue
        if last_valid_lsd < best_lsd - min_delta:
            best_lsd, selected_epochs = last_valid_lsd, epoch + 1
            stalled_checks = 0
            save_checkpoint(output_dir / "selection_best.pt", model, config, "selection",
                            epoch + 1, validation_lsd=last_valid_lsd,
                            warm_start=resume_metadata)
        else:
            if epoch + 1 <= early_stopping_start:
                stalled_checks = 0
                continue
            stalled_checks += 1
            if patience and stalled_checks >= patience:
                print({"early_stopping": True, "epoch": epoch + 1,
                       "patience_checks": patience,
                       "validation_every_epochs": validation_every,
                       "early_stopping_start_epoch": early_stopping_start,
                       "selected_epochs": selected_epochs}, flush=True)
                break
    save_checkpoint(output_dir / "selection_last.pt", model, config, "selection-last",
                    last_epoch, validation_lsd=last_valid_lsd,
                    optimizer_state=optimizer.state_dict(), warm_start=resume_metadata)
    summary = {
        "protocol": "listeners 0:160 train; 160:180 validation; fixed validation tasks",
        "selected_epochs": selected_epochs,
        "best_validation_LSD": best_lsd,
        "history": history,
        "provenance": provenance(device),
        "last_epoch": last_epoch,
        "validation_every_epochs": validation_every,
        "early_stopping_start_epoch": early_stopping_start,
        "early_stopping_patience_checks": patience,
        "early_stopping_min_delta": min_delta,
        "warm_start": resume_metadata,
    }
    save_json(summary, output_dir / "selection_summary.json")
    return selected_epochs


def run_final(config: dict, hrtfs, directions, device: torch.device, output_dir: Path,
              selected_epochs: int, checkpoint_name: str = "final_model.pt") -> Path:
    if selected_epochs <= 0:
        raise ValueError("selected_epochs must be positive")
    seed = config["seed"]
    seed_everything(seed)
    final_data = VariableContextDataset(
        hrtfs, directions, subject_range(config["data"]["final_train_subjects"]),
        config["sampling"], seed, vary_by_epoch=True,
    )
    loader = make_loader(final_data, config["training"], True, seed)
    # Intentionally construct a fresh model: no selection checkpoint is loaded.
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config["training"])
    history = []
    for epoch in range(selected_epochs):
        final_data.set_epoch(epoch)
        metrics = train_epoch(model, loader, optimizer, device,
                              config["training"]["gradient_clip_norm"])
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in metrics.items()}}
        history.append(row)
        print(row, flush=True)
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise ValueError("checkpoint_name must be a .pt filename without directory components")
    checkpoint_path = output_dir / checkpoint_name
    save_checkpoint(checkpoint_path, model, config, "final", selected_epochs,
                    selected_epochs=selected_epochs,
                    protocol="fresh initialization; listeners 0:180; no validation/test access")
    save_json({"selected_epochs": selected_epochs, "checkpoint": str(checkpoint_path.resolve()),
               "history": history,
               "provenance": provenance(device)}, output_dir / "final_summary.json")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=("selection", "final", "all"), default="all")
    parser.add_argument("--selection-summary", help="selection_summary.json for --stage final")
    parser.add_argument("--resume-selection-checkpoint")
    parser.add_argument("--resume-selection-summary")
    parser.add_argument("--selected-epochs", type=int, help="explicit override for --stage final")
    parser.add_argument("--final-checkpoint-name", default="final_model.pt")
    parser.add_argument("--hrtf-path")
    parser.add_argument("--directions-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = apply_data_overrides(load_config(args.config), args.hrtf_path, args.directions_path)
    output_dir = Path(args.output_dir or config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    hrtfs, directions = load_external_arrays(config["data"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected_epochs = None
    if args.stage in ("selection", "all"):
        selected_epochs = run_selection(
            config, hrtfs, directions, device, output_dir,
            args.resume_selection_checkpoint, args.resume_selection_summary,
        )
    if args.stage in ("final", "all"):
        if args.selected_epochs is not None:
            selected_epochs = args.selected_epochs
        elif selected_epochs is None:
            if not args.selection_summary:
                parser.error("--stage final requires --selection-summary or --selected-epochs")
            with Path(args.selection_summary).open("r", encoding="utf-8") as handle:
                selected_epochs = int(json.load(handle)["selected_epochs"])
        run_final(config, hrtfs, directions, device, output_dir, selected_epochs,
                  args.final_checkpoint_name)


if __name__ == "__main__":
    main()
