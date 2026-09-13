# Pretraining on BigEarthNet-S2 (an alternative corpus to SeCo)

This branch adds **BigEarthNet-S2** as a second pretraining corpus for the same three
mechanisms and the same `build_vit_s8()` encoder. It is opt-in: everything defaults to SeCo
exactly as before, and `--dataset bigearthnet` switches the corpus.

## Why BigEarthNet for the EuroSAT downstream task

| | SeCo-100k (Zenodo) | BigEarthNet-S2 |
|---|---|---|
| Sensor / bands | Sentinel-2, 8-bit contrast-stretched RGB previews | Sentinel-2, uint16 L2A reflectance (B04/B03/B02) |
| Scene selection | uncurated locations, skews to coast / ocean / ice | built to span the CORINE land-cover classes |
| Overlap with EuroSAT's 10 classes | partial | near-complete (forest, crops, pasture, water, urban, ...) |
| Size | ~100k locations (~500k patches, 5 seasons each) | ~590k patches (one image each) |
| Native tile size | 264x264 preview | 120x120 |
| Positive-pair options | augmentation **or** seasonal revisit | augmentation only (no revisits) |

EuroSAT is itself a Sentinel-2 land-cover benchmark, so BigEarthNet's scene distribution is
the closer match. Published SSL-in-RS work (SeCo, SatMAE, Scale-MAE) consistently shows
BigEarthNet pretraining beating SeCo on EuroSAT linear-probe with a fixed backbone.

## 1. Get the data

The BigEarthNet-S2 hosting URL/checksum has changed several times, so it is **not**
hard-coded. Provide a working archive URL, or point at an already-extracted copy:

```bash
# download from an explicit mirror:
python download_bigearthnet.py --root ../../../data/bigearthnet \
    --url https://<mirror>/BigEarthNet-S2.tar.zst --fraction 1.0 --keep-archive

# or, if the data is already on disk somewhere:
python download_bigearthnet.py --root /path/to/BigEarthNet-S2 --skip-download
```

Sources: TU Berlin / RSiM at <https://bigearth.net>, or the Zenodo mirror
(<https://zenodo.org/records/10891137>). This writes `manifest.txt` (one patch dir per line)
and `s2_root.txt` (where the patch dirs live) into `--root`.

`--fraction < 1.0` keeps a seeded random patch subset and deletes the rest, same as
`download_seco.py`.

## 2. Pretrain

Identical scripts to the SeCo path, with two extra flags:

```bash
python train_contrastive.py --dataset bigearthnet --ben-root ../../../data/bigearthnet \
    --out ../../../artifacts/remote_sensing/checkpoints/contrastive_vit_s8_ben.pt \
    --preload-res 120 --patch-cache ../../../data/bigearthnet/patch_cache_r120
python train_mae.py  --dataset bigearthnet --ben-root ../../../data/bigearthnet --out .../mae_vit_s8_ben.pt  ...
python train_dino.py --dataset bigearthnet --ben-root ../../../data/bigearthnet --out .../dino_vit_s8_ben.pt ...
```

`--seasonal-positives` is SeCo-only and errors under `--dataset bigearthnet`. The
GPU-augmentation path (`gpu_aug.py`) is used exactly as for SeCo — `BigEarthNetPatchCache`
mirrors `SeCoPatchCache`'s interface, one patch per "location", so the InfoNCE / masking /
multi-crop code is unchanged. The `--no-gpu-aug` CPU path is not yet wired for BigEarthNet.

Normalization uses BigEarthNet RGB stats (`bigearthnet_data.BEN_MEAN/STD`) for the
pretraining forward pass; downstream EuroSAT probing still uses EuroSAT's own stats.

## 3. SLURM

`scripts/slurm/bigearthnet/` has the cluster jobs and `submit_all_ben.sh` (a
download -> {contrastive, mae, dino} -> eval dependency chain). Set `BEN_URL` first:

```bash
export BEN_URL=https://<mirror>/BigEarthNet-S2.tar.zst
bash scripts/slurm/bigearthnet/submit_all_ben.sh
```

## 4. Compare against SeCo

The eval job writes `notebooks/remote_sensing/figures/rs_eval_results_ben.json`
(vs `rs_eval_results.json` for SeCo), same linear-probe protocol and baselines, so the two
corpora sit side by side.
