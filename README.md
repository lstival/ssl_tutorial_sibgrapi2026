# SIBGRAPI 2026 Tutorial — Self-Supervised Learning: Contrastive, Masking, and Distillation Methods

Materials for the SIBGRAPI 2026 tutorial on self-supervised learning (SSL), built around one
running idea: SSL methods differ in *mechanism* (contrastive, generative/masking,
self-distillation) but share one *principle* — learn structure from the data itself by
defining a pretext task, instead of relying on manual labels.

This repository covers two hands-on domains: **Remote Sensing (EuroSAT)** and **Time Series
(UCR)**. Hour 1 (theory) lives in the accompanying slides/paper. The time-series half mirrors
the remote-sensing half one-to-one — the same three mechanisms, the same fill-in-the-blank
teaching pattern, and the **same downstream task and evaluation protocol** (frozen encoder +
linear probe + accuracy), on a 1D sequence instead of a 2D image grid.

Holding the task fixed across both hours is deliberate: it is what lets the two result tables be
read side by side, so the differences between them are attributable to the *modality* rather
than to the yardstick.

**This repository is self-contained.** Code and notebooks live here; the pretrained encoders
are published as assets on a [GitHub Release](https://github.com/lstival/ssl_tutorial_sibgrapi2026/releases/tag/weights-v1)
(`weights-v1`) and downloaded on demand, so a clone or a Colab run can execute every notebook
without training from scratch.

## Quick start

### Colab

Open any notebook with its "Open in Colab" badge and run all cells. The first code cell
downloads the one shared module it needs (`tutorial_rs.py` / `tutorial_ts.py`); the pretrained
encoders are downloaded on demand from the [weights release](https://github.com/lstival/ssl_tutorial_sibgrapi2026/releases/tag/weights-v1).
Nothing to install.

### Local

```bash
git clone https://github.com/lstival/ssl_tutorial_sibgrapi2026.git
cd ssl_tutorial_sibgrapi2026
pip install -r requirements.txt

cd notebooks/time_series          # or notebooks/remote_sensing
jupyter notebook 00_setup_and_data.ipynb
```

The first notebook in each track downloads the pretrained encoders it needs from the
[weights release](https://github.com/lstival/ssl_tutorial_sibgrapi2026/releases/tag/weights-v1)
the first time it runs, and reuses them afterwards. To fetch all ten encoders ahead of time
(e.g. for an offline tutorial session), run:

```bash
python tools/verify_assets.py --remote   # confirm the release assets are reachable
```

Datasets are *not* committed (they are multi-GB) and download themselves on first use: EuroSAT
via torchvision with mirror fallback, and the UCR archive (~316 MB) from its official host.

### Verifying the weights

`tools/verify_assets.py` checks every encoder the notebooks load, from both directions:

```bash
python tools/verify_assets.py            # local copy + release download URLs
python tools/verify_assets.py --local    # local copy only, no network
python tools/verify_assets.py --remote   # release download URLs only
```

The remote check confirms the GitHub Release serves the real tensors, not a 404. Each notebook
also carries a small self-test cell that reports which encoders are on disk or downloadable
before any training starts.

### Continuous validation

Two layers, because Colab has no free API to trigger automated runs on its own GPUs:

- **CI, every push/PR + weekly** ([`.github/workflows/notebooks-ci.yml`](.github/workflows/notebooks-ci.yml)):
  runs every teaching notebook headless on CPU via `tools/run_notebooks_ci.py`, both tracks in
  parallel. Catches import/path/logic breakage and confirms the release assets are reachable.
  Does not exercise a GPU.
- **Colab GPU smoke test, run by hand before a session** ([`notebooks/colab_smoke_test.ipynb`](notebooks/colab_smoke_test.ipynb)):
  open it in Colab, Runtime > GPU, Runtime > Run all. It clones the repo fresh, confirms a GPU
  is attached, and runs every notebook end-to-end on that GPU with the real weights downloaded
  from the release -- i.e. exactly what a participant's browser will do. The last cell reports
  which notebook (if any) broke and why.

```bash
python tools/run_notebooks_ci.py                  # both tracks, CPU, local or CI
python tools/run_notebooks_ci.py --track time_series --only 00,01
```

## Repository layout

```
notebooks/
├── remote_sensing/             Hour-2 teaching notebooks
│   ├── 00_setup_and_data.ipynb        Dataset, RS-specific augmentations, shared backbone
│   ├── 01_contrastive_simclr.ipynb    Contrastive learning (InfoNCE)
│   ├── 02_masking_mae.ipynb           Masked autoencoding (MAE)
│   ├── 03_distillation_dino.ipynb     Self-distillation (DINO)
│   ├── 04_comparative_evaluation.ipynb  Linear-probe comparison across all three encoders
│   └── figures/                       Result JSON + figures behind the paper and the website
├── time_series/                Hour-3 teaching notebooks (same style, UCR classification)
│   ├── 00 .. 04                       Same five-notebook structure as remote_sensing
│   └── figures/
└── colab_smoke_test.ipynb      Run by hand on a Colab GPU before a session, see below

src/
├── remote_sensing/
│   ├── tutorial_rs.py          Shared EuroSAT data, ViT-S/8 backbone, plotting, checkpoint loader
│   └── pretraining/            Large-scale offline SSL pretraining on Sentinel-2 (SeCo)
└── time_series/
    ├── tutorial_ts.py          Shared UCR data, patch-Transformer, linear probe, checkpoint loader
    └── pretraining/            Offline SSL pretraining on the pooled 128-dataset UCR corpus

artifacts/<domain>/checkpoints/ Pretrained encoders -- downloaded from the weights release, see below
site/                           The tutorial website (GitHub Pages)
tools/
├── release_weights.sh          Uploads the encoders to the GitHub Release (maintainers only)
├── verify_assets.py            Checks every encoder is present locally and downloadable
└── run_notebooks_ci.py         Executes every notebook headless and reports what broke (CI + Colab smoke test)
```

This repository ships the tutorial itself: the notebooks, the modules they import, the
pretrained encoders and the website. The machinery used to *produce* those artifacts -- the
SLURM/PowerShell launchers for the offline pretraining runs, the LaTeX paper source and the
internal planning notes -- is kept out of it.

## Pretrained encoders

Ten encoders are committed, all of them loaded by the notebooks. Remote sensing uses a
ViT-Small/8 (64×64 input, 8×8 token grid, embed dim 384); time series uses a patch Transformer
(128-step series, patch 16, embed dim 128).

| Checkpoint | Mechanism | Corpus | Size |
|---|---|---|---|
| `contrastive_vit_s8.pt` | Contrastive | SeCo (Sentinel-2) | 41 MB |
| `mae_vit_s8.pt` | Masking | SeCo | 41 MB |
| `dino_vit_s8.pt` | Distillation | SeCo | 41 MB |
| `contrastive_vit_s8_ben.pt` | Contrastive | BigEarthNet-S2 | 41 MB |
| `mae_vit_s8_ben.pt` | Masking | BigEarthNet-S2 | 41 MB |
| `dino_vit_s8_ben.pt` | Distillation | BigEarthNet-S2 | 41 MB |
| `random_init_vit_s8.pt` | — (baseline) | — | 41 MB |
| `contrastive_ts_encoder.pt` | Contrastive | UCR (128 datasets) | 2.3 MB |
| `mae_ts_encoder.pt` | Masking | UCR | 2.3 MB |
| `dino_ts_encoder.pt` | Distillation | UCR | 2.3 MB |

Each `.pt` has a sibling `.json` recording the run that produced it (step count, final loss,
mechanism hyper-parameters). The optimizer/resume state (`*.train.pt`, 140–196 MB per run) is
**not** committed — no notebook reads it, and it is only needed to resume offline pretraining.

Notebooks 1–3 each work in two modes:

- **With the pretrained encoder** — loads it and jumps straight to evaluation/visualization.
- **Without it** — falls back to a short live-training loop on the downstream data, so every
  notebook still runs end-to-end. Slower convergence, since it is a demo of the *mechanism*,
  not a full pretraining run.

## Results

Both halves use the same protocol — frozen encoder, linear probe, accuracy — which is what
makes them comparable. The headline finding is that **the ranking flips between modalities**:

| | Contrastive | Masking (MAE) | Distillation (DINO) |
|---|---|---|---|
| **Remote sensing** (EuroSAT) | **93.8%** | 91.3% | 84.6% |
| **Time series** (UCR, SwedishLeaf) | 91.0% | 90.2% | **93.9%** |

Contrastive learning wins on imagery; self-distillation wins on time series. The paradigm is
chosen by the structure of the data, not by which method is most recent.

The website under `site/` renders these numbers directly from the result files in
`notebooks/*/figures/`.

## Citation

```bibtex
@inproceedings{stival2026ssl,
  title     = {Self-Supervised Learning: Contrastive, Masking, and Distillation Methods},
  author    = {Stival, Leandro and Zhang, Chao and da Silva Torres, Ricardo},
  booktitle = {Proceedings of the 39th SIBGRAPI Conference on Graphics, Patterns and Images},
  year      = {2026},
  publisher = {IEEE}
}
```

Artificial Intelligence Group · Wageningen University & Research
EU Horizon "AI Foundation Models in Agricultural Sciences" · grant 101293777
