# HRTF Upsampling Across Varying Measurement Configurations with Geometry-Aware Query-Conditioned Aggregation

Official PyTorch implementation of [*HRTF Upsampling Across Varying Measurement Configurations with Geometry-Aware Query-Conditioned Aggregation*](https://arxiv.org/abs/2609.25995). GeoAtt estimates binaural HRTFs at query directions from a variable-size set of measured directions. The model combines geometry-aware spatial cross-attention with frequency-domain Conformer blocks.

This repository provides model code, SONICOM preparation tools, training, and evaluation for the GeoAtt results in [Tables 1 and 2](https://arxiv.org/html/2609.25995v1). Preprocessed data and the paper checkpoint are hosted in the [GeoAtt-HRTF Hugging Face repository](https://huggingface.co/xingyuchen-8/GeoAtt-HRTF).

## Installation

Clone the repository and create a Python 3.10 environment:

```bash
git clone https://github.com/xingyuaudio/GeoAtt-HRTF.git
cd GeoAtt-HRTF
conda create --prefix ./.conda/geoatt python=3.10 -y
conda activate ./.conda/geoatt
```

Install a PyTorch 2.6.0 build suitable for your system. For a CUDA 12.4 environment:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[data,hub,test]"
```

For another platform, use the appropriate PyTorch 2.6.0 command from the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/). The project pins `torch==2.6.0`.

## Dataset preparation

The [SONICOM dataset](https://www.sonicom.eu/) is external. The model expects `H_log.npy` with shape `(200, 793, 2, 106)` and `locs.npy` with shape `(200, 793, 3)`. `H_log` contains left/right log-magnitude spectra in dB. `locs` contains source azimuth, elevation, and radius; the first two coordinates are in degrees. Listener rows are P0001–P0200, and the 793 directions must retain their original order.

Option A — download the preprocessed arrays and checkpoint from [Hugging Face](https://huggingface.co/xingyuchen-8/GeoAtt-HRTF):

```bash
python tools/download_assets.py
```

This places the arrays in `data/sonicom/` and the checkpoint in `weights/`. The arrays are derived from the [SONICOM HRTF dataset](https://www.sonicom.eu/); please observe the original dataset terms and attribution requirements.

Option B — generate the arrays from the original SONICOM SOFA files:

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

## Evaluation

After downloading the checkpoint with Option A, evaluate the held-out listeners:

```bash
python -m vcfdconformer.evaluate \
  --checkpoint weights/geoatt_final_ep440.pt \
  --protocol all \
  --hrtf-path data/sonicom/H_log.npy \
  --directions-path data/sonicom/locs.npy \
  --output-dir results/paper
```

`results/paper/table1_canonical.json` reports the four canonical LAP configurations (`M=3, 5, 19, 100`). `results/paper/table2_m19.json` reports the fixed Lebedev-based, spherical *t*-design-based, hemisphere, and horizontal-ring configurations in [`protocols/table2_m19_layouts.json`](protocols/table2_m19_layouts.json). Both protocols use listeners P0181–P0200, evaluate only directions outside the context, and report per-listener LSD/ILD plus mean and sample standard deviation across listeners. Use `--protocol table1` or `--protocol table2` to run one evaluation.

## Training

`configs/paper.yaml` records the paper configuration. Selection uses listeners P0001–P0160 for training and P0161–P0180 for validation; validation LSD is checked every 20 epochs, with early-stopping counting beginning after epoch 400. The selected duration was 440 epochs. Final training initializes a new model and trains on P0001–P0180 for 440 epochs. P0181–P0200 are held out for evaluation.

Each training example chooses a canonical context with probability 0.5 or a random context size in `[3,100]` otherwise, then samples `Q=64` non-context query directions. The model has four 128-dimensional FD-Conformer blocks, 256-dimensional feed-forward layers, eight heads, kernel size seven, four direction-encoding bands, and dropout 0.1. Training uses LSD + SGL, AdamW, learning rate `1e-3`, weight decay `1e-4`, and batch size eight.

```bash
python -m vcfdconformer.train \
  --config configs/paper.yaml \
  --stage all \
  --hrtf-path data/sonicom/H_log.npy \
  --directions-path data/sonicom/locs.npy \
  --output-dir runs/geoatt
```

To train only the final model with the selected duration:

```bash
python -m vcfdconformer.train \
  --config configs/paper.yaml \
  --stage final \
  --selected-epochs 440 \
  --hrtf-path data/sonicom/H_log.npy \
  --directions-path data/sonicom/locs.npy \
  --output-dir runs/geoatt_final
```

## Citation

```bibtex
@article{chen2026geoatt,
  title={HRTF Upsampling Across Varying Measurement Configurations with Geometry-Aware Query-Conditioned Aggregation},
  author={Chen, Xingyu and Bi, Hanwen and Zhao, Sipei and Ma, Fei and Cheng, Eva and Burnett, Ian S.},
  journal={arXiv preprint arXiv:2609.25995},
  year={2026}
}
```

## License

Code is released under the [MIT License](LICENSE). SONICOM data retain their own terms.
