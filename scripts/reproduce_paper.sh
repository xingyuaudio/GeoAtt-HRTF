#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HRTF_PATH="${HRTF_PATH:?Set HRTF_PATH to H_log.npy}"
DIRECTIONS_PATH="${DIRECTIONS_PATH:?Set DIRECTIONS_PATH to locs.npy}"
run_tag="$(date +%Y%m%d_%H%M%S)"
output_dir="runs/geoatt_${run_tag}"
checkpoint_name="geoatt_final_${run_tag}.pt"

python -m vcfdconformer.train \
  --config configs/paper.yaml \
  --stage all \
  --output-dir "$output_dir" \
  --final-checkpoint-name "$checkpoint_name" \
  --hrtf-path "$HRTF_PATH" \
  --directions-path "$DIRECTIONS_PATH"

python -m vcfdconformer.evaluate \
  --checkpoint "$output_dir/$checkpoint_name" \
  --protocol all \
  --layouts protocols/table2_m19_layouts.json \
  --hrtf-path "$HRTF_PATH" \
  --directions-path "$DIRECTIONS_PATH" \
  --output-dir "results/geoatt_${run_tag}"

printf 'checkpoint=%s\n' "$output_dir/$checkpoint_name"
