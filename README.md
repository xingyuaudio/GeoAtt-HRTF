# GeoAtt: variable-context HRTF upsampling

Official implementation of [*HRTF Upsampling Across Varying Measurement Configurations with Geometry-Aware Query-Conditioned Aggregation*](https://arxiv.org/html/2609.25995v1). One GeoAtt model handles different numbers and layouts of context measurements. This repository contains the final model, SONICOM data preparation, the paper training protocol, and inference/evaluation for the GeoAtt rows of Tables 1 and 2. Ablation variants and baseline implementations are not included.

## Installation

Use Python 3.10 or newer. Install a platform-compatible PyTorch 2.6.0 build first; the project pins `torch==2.6.0` so a compatible existing installation will not be upgraded to a different version.

```bash
python -m pip install -e ".[test]"
```

For raw SOFA download and preprocessing, install the optional data dependencies as well:

```bash
python -m pip install -e ".[data,test]"
```

## Data

The [SONICOM dataset](https://www.sonicom.eu/) is external. The model expects `H_log.npy` with shape `(200, 793, 2, 106)` and `locs.npy` with shape `(200, 793, 3)`. `H_log` contains left/right log-magnitude spectra in dB. `locs` contains source azimuth, elevation, and radius; the first two coordinates are in degrees. Listener rows are P0001–P0200, and the 793 directions must retain their original order.

Choose one route:

1. Download the preprocessed arrays from Hugging Face: **dataset link to be added after upload**. Place both files in `data/sonicom/`.
2. Generate them from SONICOM SOFA files:

   ```bash
   python tools/download_sonicom.py --output-dir data/raw
   python tools/preprocess_sonicom.py --raw-dir data/raw --output-dir data/sonicom
   ```

   SONICOM download access may be required. Do not reorder the SOFA measurement directions.

Validate either route:

```bash
python tools/check_data.py \
  --hrtf-path data/sonicom/H_log.npy \
  --directions-path data/sonicom/locs.npy
```

## Reproduce Tables 1 and 2 with the released model

Download the paper checkpoint from Hugging Face: **model link to be added after upload**. Save it as `weights/geoatt_final_ep440.pt`; the `weights/` directory and `.pt` file are intentionally not tracked by Git.

```bash
python -m vcfdconformer.evaluate \
  --checkpoint weights/geoatt_final_ep440.pt \
  --protocol all \
  --hrtf-path data/sonicom/H_log.npy \
  --directions-path data/sonicom/locs.npy \
  --output-dir results/paper
```

`results/paper/table1_canonical.json` reports the four canonical LAP configurations (`M=3,5,19,100`). `results/paper/table2_m19.json` reports the fixed Lebedev-based, spherical *t*-design-based, hemisphere, and horizontal-ring configurations from `protocols/table2_m19_layouts.json`. Both protocols use listeners P0181–P0200, evaluate only directions outside the context, and report per-listener LSD/ILD plus mean and sample standard deviation across listeners. Use `--protocol table1` or `--protocol table2` to run only one table. The repository does not reproduce other methods' rows.

## Reproduce training

`configs/paper.yaml` records the paper configuration. Selection uses listeners P0001–P0160 for training and P0161–P0180 for validation; validation LSD is checked every 20 epochs, with early-stopping counting beginning after epoch 400. The selected duration was 440 epochs. Final training initializes a new model and trains on P0001–P0180 for 440 epochs. P0181–P0200 are held out for evaluation.

Each training example chooses a canonical context with probability 0.5 or a random context size in `[3,100]` otherwise, then samples `Q=64` non-context query directions. The model has four 128-dimensional FD-Conformer blocks, 256-dimensional feed-forward layers, eight heads, kernel size seven, four direction-encoding bands, and dropout 0.1. Training uses LSD + SGL, AdamW, learning rate `1e-3`, weight decay `1e-4`, and batch size eight.

```bash
export HRTF_PATH="$PWD/data/sonicom/H_log.npy"
export DIRECTIONS_PATH="$PWD/data/sonicom/locs.npy"
bash scripts/reproduce_paper.sh
```

To repeat only the final 440-epoch fit after accepting the selected duration, run `python -m vcfdconformer.train --config configs/paper.yaml --stage final --selected-epochs 440` with `--hrtf-path`, `--directions-path`, and `--output-dir` set to your locations. The original final fit used an NVIDIA L4. Small numerical differences are possible across hardware and dependency builds.

The import package remains `vcfdconformer` to preserve strict compatibility with the paper checkpoint. Its public `GeoAtt` alias refers to the final model class; checkpoint metadata still uses the original architecture identifier `VCFDConformer-Direct` for the same reason. No ablation code is needed to load or run this checkpoint.

## Citation and license

If you use GeoAtt, please cite [the paper](https://arxiv.org/html/2609.25995v1). Code is released under the [MIT License](LICENSE). SONICOM data retain their own terms; this code license does not grant data redistribution rights.
