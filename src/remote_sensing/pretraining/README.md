# Large-scale pretraining (SeCo -> ViT-S/8 checkpoints)

Produces the three encoder checkpoints that `notebooks/remote_sensing/01_contrastive_simclr.ipynb`,
`02_masking_mae.ipynb`, `03_distillation_dino.ipynb`, and `04_comparative_evaluation.ipynb`
try to download at the top of each notebook (`artifacts/remote_sensing/checkpoints/{contrastive,mae,dino}_vit_s8.pt`).
This is the offline infrastructure step described in `docs/remote_sensing_implementation_plan.md`,
section 13.

The model architecture (`build_vit_s8`, `TransformerBlock`) is imported directly from
`../src/remote_sensing/tutorial_rs.py` so the pretrained weights are always structurally compatible with
what the notebooks load -- there is exactly one definition of the encoder, in `tutorial_rs.py`.
Everything else in this folder (data loading, training loops, losses) is independent,
complete reference code with no fill-in-the-blank sentinels, since it's not teaching material.

## 1. Download 10% of SeCo

[SeCo](https://arxiv.org/abs/2103.16607) (Mañas et al., 2021) is distributed as a fixed
archive on Zenodo -- no Google Earth Engine account needed, unlike the original repo's raw
collection pipeline. `download_seco.py` downloads the "100k" version (~100,000 Sentinel-2
locations, up to 5 seasonal revisits each), extracts it, then **keeps only a random 10% of
locations** (seeded, reproducible) and deletes the rest plus the archive to save disk space.

```bash
pip install -r requirements.txt
python download_seco.py --root ../data/seco --fraction 0.10 --seed 42
```

This writes:

```
../data/seco/
    seasonal_contrast_100k/
        <location_id>/<season>/B1.tif ... B12.tif   (~10,000 locations kept)
    manifest.txt                                     (the kept location IDs, one per line)
```

`--fraction` and `--seed` control how much data is kept and which subset -- pass
`--keep-archive` to retain `seco_100k.zip` if you want to resample a different fraction later
without re-downloading.

## 2. Pretrain each encoder

Three independent scripts, one per SSL mechanism, mirroring Notebooks 1-3:

```bash
python train_contrastive.py --seco-root ../data/seco/seasonal_contrast_100k --manifest ../data/seco/manifest.txt
python train_mae.py         --seco-root ../data/seco/seasonal_contrast_100k --manifest ../data/seco/manifest.txt
python train_dino.py        --seco-root ../data/seco/seasonal_contrast_100k --manifest ../data/seco/manifest.txt
```

By default each writes its encoder-only checkpoint to
`../artifacts/remote_sensing/checkpoints/{contrastive,mae,dino}_vit_s8.pt` (override with `--out`), plus a
`*_train_log.jsonl` step-by-step loss log and a `*.json` metadata sidecar next to the
checkpoint. Only the encoder is saved (projection head / decoder / DINO head are
pretraining-only and discarded) -- this is exactly the state dict shape
`src/remote_sensing/tutorial_rs.py`'s `try_load_checkpoint` loads into a bare `build_vit_s8()`.

Key flags (see `--help` on each script for the full list):

| Flag | Meaning | Default |
|---|---|---|
| `--steps` | optimizer steps | 20,000 |
| `--batch-size` | per-step batch size | 256 (64 for DINO, multi-crop is memory-heavier) |
| `--checkpoint-every` | save a checkpoint every N steps (also always saves at the last step) | 1,000 |
| `--num-workers` | DataLoader worker processes | 4 |

`train_contrastive.py` also accepts `--seasonal-positives`, which swaps the SimCLR-style
"two augmentations of the same season" positive pair for SeCo's own "two different seasonal
revisits of the same location" positive pair (see the discussion in Notebook 1's aside on
SeCo) -- a genuinely remote-sensing-specific invariance signal that doesn't exist for natural
photos.

### The MAE fix: per-patch target normalization (`norm_pix`)

An earlier version of the MAE encoder scored ~0.82 on the EuroSAT linear probe (mean-pooled
patch tokens), only ~3 points above a **completely untrained** ViT (~0.78), while contrastive
reached ~0.88 and DINO ~0.86 on the identical encoder, data, and protocol. Fine-tuning did not
separate it from a random encoder either -- so the encoder had genuinely learned almost nothing.

The root cause was a missing detail in the reconstruction loss: **per-patch target
normalization**. Canonical MAE (He et al., 2022, Sec. 3) standardizes each *target* patch by
its own mean and variance before the MSE. Without it, the loss is dominated by each patch's
mean brightness -- a trivial statistic the shallow decoder predicts from the visible tokens
without the encoder ever having to encode content, so the encoder learns nothing transferable.
`mae_reconstruction_loss` now does this by default (`norm_pix=True`). A focused 2.5k-step SeCo
sweep, linear probe only, isolated it: adding this single line moved the probe from 0.766 to
0.805 on the same subset (random floor 0.723). The reconstruction *loss value* rises when
`norm_pix` is on (the target now has unit variance) -- that is expected, not a regression.

Two further changes were made on general grounds (correct hygiene, but neither closed the gap
on its own -- `norm_pix` did):

- **Augmentation.** MAE now trains on augmented views via `build_seco_mae_augmentations`
  (random resized crop + flips + 90-degree rotations, *no* photometric jitter -- the original
  MAE recipe is geometry-only). With only ~10,000 SeCo patches, deterministic views invite
  memorization. `--no-augment` restores the old behavior for ablation.
- **Defaults = Optuna recipe.** The script defaults are now the values in
  `optuna_best_mae.txt`, so a bare `python train_mae.py --seco-root ...` reproduces the tuned
  run instead of a generic one that has to be overridden on the command line.

## GPU augmentation (all three trainers)

The CPU pipeline (PIL -> torchvision -> ToTensor) costs ~2 ms per view and, because the in-RAM
preload forces `num_workers=0`, runs fully serial with the GPU. In Task Manager this looks like
a GPU utilization trace that spikes to 100% and drops back to near zero between batches: the
GPU is not the bottleneck, the input pipeline is. Bigger batches do not fix this -- they make
each spike wider and the gap between spikes longer, because the per-view Python cost scales
with the batch too.

`gpu_aug.py` removes the stall by keeping the decoded corpus as one uint8 tensor and fusing
RandomResizedCrop + flips + rotation into a single batched `grid_sample` on the GPU. This was
originally wired into `train_contrastive.py` only; `train_mae.py` and `train_dino.py` now take
the same path, and it is the default on CUDA (`--no-gpu-aug` restores the CPU path).

DINO gains the most: it augments `n_global + n_local` crops per sample per step, so it was
paying the per-view Python cost six times over.

Reuse the decoded cache across runs -- it costs ~15 min of .tif reads to build and seconds to
reload:

```bash
python train_mae.py --seco-root ../../../data/seco/seasonal_contrast_100k   --manifest ../../../data/seco/manifest.txt --preload-res 128   --patch-cache ../../../data/seco/patch_cache_r128   --out ../../../artifacts/remote_sensing/checkpoints/mae_vit_s8.pt
```

Note the cache is placed on the GPU when it fits beside the model (`--cache-device auto`), so
a second training job on the same GPU will push it to host RAM and run slower. Run these one
at a time.

The two paths are not bit-identical (see the fidelity note at the top of `gpu_aug.py`), so a
single comparison must not mix them: checkpoints trained before this change used the CPU path.

## 3. Publish the checkpoints for the notebooks

Copy the trained `*_vit_s8.pt` files into `artifacts/remote_sensing/checkpoints/` and commit
them. They are tracked by Git LFS (see `.gitattributes`), so they travel with the repository
and Notebooks 1-4 pick them up with the "Found pretrained model, loading..." pattern instead
of falling back to their built-in short live-training demo.

    cp <trained>.pt ../../../artifacts/remote_sensing/checkpoints/
    git add artifacts/remote_sensing/checkpoints/<trained>.pt
    python ../../../tools/verify_assets.py --local    # confirm it is real data, not a pointer

There is no release to publish and no URL to update: `CHECKPOINT_BASE_URL` in
`src/remote_sensing/tutorial_rs.py` already points at this repository's own LFS storage, which
is what Colab downloads from. After pushing, `python tools/verify_assets.py --remote` confirms
the download URLs resolve.

## Design notes

- **RGB only.** SeCo's 12 Sentinel-2 bands are read down to B4/B3/B2 (true color), matching
  EuroSAT RGB -- pretraining and downstream probing need the same channel count and a
  compatible spectral normalization (plan section 3.1).
- **64x64 everywhere.** `RandomResizedCrop(size=64)` at pretraining time keeps the ViT-S/8
  token grid (8x8, patch size 8) identical to what EuroSAT produces natively, so no
  position-embedding interpolation is ever needed between pretraining and the tutorial notebooks.
- **Why a separate folder from `notebooks/remote_sensing/`?** These scripts are not teaching material -- no
  fill-in-the-blank cells, no markdown narrative, no plotting. They exist purely to produce
  checkpoint files. `src/remote_sensing/tutorial_rs.py` remains the single source of truth for the
  architecture; everything data/training-related here is intentionally independent so changes
  to the (large, slow, offline) pretraining pipeline can't accidentally break the notebooks'
  teaching code path, and vice versa.
