# Time-series pretraining (pooled UCR archive -> patch-Transformer checkpoints)

The time series counterpart of `src/remote_sensing/pretraining/`. Produces the three encoder checkpoints
that `notebooks/time_series/01_contrastive_simclr.ipynb`, `02_masking_mae.ipynb`,
`03_distillation_dino.ipynb`, and `04_comparative_evaluation.ipynb` try to download at the top
of each notebook (`artifacts/time_series/checkpoints/{contrastive,mae,dino}_ts_encoder.pt`).

The model architecture (`build_ts_encoder`, `TransformerBlock`) is imported directly from
`../tutorial_ts.py` so the pretrained weights are always structurally compatible with what the
notebooks load -- there is exactly one definition of the encoder, in `tutorial_ts.py`.
Everything else here (data loading, training loops, losses) is independent, complete reference
code with no fill-in-the-blank sentinels.

## 1. Data

**The UCR Time Series Classification Archive (2018).** 128 univariate classification datasets,
distributed as a single ~316 MB password-protected zip that is downloaded and extracted on first
use into `--data-path` (default `../../../data`, gitignored). See `ensure_ucr_archive` in
`../tutorial_ts.py`. The password is published on the archive's own page and is not a secret; it
exists so downloaders acknowledge the accompanying documentation.

Every series in the archive is put on a common footing by two transformations:

- **resampled to `SERIES_LEN` (128)** by linear interpolation, so one backbone with fixed
  position embeddings reads all 128 datasets (native lengths run from 15 to 2,844);
- **per-series z-normalized**, the UCR convention and the right invariance for classification.

Variable-length datasets (padded with trailing NaN) and the DodgerLoop family (genuine interior
missing values) are repaired by `_clean_series` -- padding is trimmed, interior gaps are linearly
interpolated.

### The pretraining corpus

The corpus is the **TRAIN split of all 128 datasets with labels discarded**. Each dataset is
capped at `--per-dataset-cap` (default 2,000) series so the pool is not dominated by its largest
members (Crop alone has 7,200 training series; Chinatown has 20). That yields **~45k series**,
about 23 MB as float32 -- the whole corpus sits in RAM, so there is no per-item decode and no
DataLoader worker fan-out is needed.

### The downstream probe

Notebook 4 probes on **SwedishLeaf alone** (500 train / 625 test, 15 classes, native length 128).
This is the "pretrain on a broad corpus, transfer to one target dataset" setup -- the direct
analogue of the remote sensing part's "pretrain on SeCo, probe on EuroSAT". SwedishLeaf is only ~1% of the
pretraining corpus, so the encoder has seen the target domain but is nowhere near dominated by
it.

The target's **test** split is never touched during pretraining. Its train split *is* in the pool
with labels discarded -- standard self-supervised practice, and exactly what the remote sensing part does.
`--exclude-target SwedishLeaf` drops it entirely for the strict cross-dataset ablation.

## 2. Pretrain each encoder

Three independent scripts, one per SSL mechanism, mirroring Notebooks 1-3:

```bash
python train_contrastive_ts.py --data-path ../../../data
python train_mae_ts.py         --data-path ../../../data
python train_dino_ts.py        --data-path ../../../data
```

Each writes its encoder-only checkpoint to
`../../../artifacts/time_series/checkpoints/{contrastive,mae,dino}_ts_encoder.pt` (override with
`--out`), plus a `*_train_log.jsonl` step log and a `*.json` metadata sidecar. Only the encoder
is saved (projection head / decoder / DINO head are pretraining-only) -- exactly the state-dict
shape `tutorial_ts.py`'s `try_load_checkpoint` loads into a bare `build_ts_encoder()`.

Common flags (see `--help` on each):

| Flag | Meaning | Default |
|---|---|---|
| `--steps` | optimizer steps | 15,000 |
| `--batch-size` | per-step batch (series) | 256 (128 for DINO -- multi-crop is heavier) |
| `--per-dataset-cap` | max series contributed by any one UCR dataset | 2,000 |
| `--exclude-target` | drop one dataset from the corpus entirely | none |
| `--checkpoint-every` | save every N steps (also at the last step) | 2,500 |

On a single RTX 3060 each run takes roughly 20-30 minutes at these defaults.

### The MAE mask ratio

`train_mae_ts.py` defaults to `--mask-ratio 0.5`, not the 0.75 of image MAE. A 128-step series
gives only **8** patch tokens, so 0.75 would leave 2 visible -- too little context for the task
to be learnable. 0.5 keeps 4 visible: enough to anchor the reconstruction, few enough that
filling the gaps requires real structure. This is a worked example of a general point: SSL
hyperparameters are functions of the token budget the modality provides, not universal
constants.

### The MAE norm_pix ablation

`train_mae_ts.py --no-norm-pix` drops per-patch target normalization, reproducing the remote sensing MAE
failure mode: the reconstruction loss still descends smoothly, but the frozen encoder barely
beats random-init on the probe. `norm_pix=True` (the default) is what Notebook 2 builds in from
the start. Notebook 4's discussion asks you to compare the two.

## 3. Evaluate

`eval_ts_encoders.py` reproduces Notebook 4 headlessly: it loads the three checkpoints, runs the
frozen linear probe on the target dataset (full-label and few-label), sweeps cross-dataset
transfer, and adds the random-init floor, the 1-NN Euclidean baseline and a supervised
from-scratch ceiling. Results are written to
`../../../notebooks/time_series/figures/ts_eval_results.json`.

```bash
python eval_ts_encoders.py            # full run
python eval_ts_encoders.py --quick    # fewer transfer datasets / epochs, smoke test
```

Each mechanism is probed with the readout its objective actually trains: `cls` for contrastive
and DINO, `mean` for MAE (whose loss never touches the `[CLS]` token).

## 4. Publish the checkpoints for the notebooks

Copy the trained `*_ts_encoder.pt` files into `artifacts/time_series/checkpoints/` and commit
them. They are tracked by Git LFS (see `.gitattributes`), so they travel with the repository
and Notebooks 1-4 pick them up with the "Found pretrained model, loading..." pattern instead
of falling back to their built-in short live-training demo.

    cp <trained>.pt ../../../artifacts/time_series/checkpoints/
    git add artifacts/time_series/checkpoints/<trained>.pt
    python ../../../tools/verify_assets.py --local    # confirm it is real data, not a pointer

There is no release to publish and no tag to bump: `CHECKPOINT_BASE_URL` in `../tutorial_ts.py`
already points at this repository's own LFS storage, which is what Colab downloads from.

## Design notes

- **One fixed geometry everywhere.** Every series is 128 steps; patch length 16, stride 16 -> 8
  patch tokens + 1 `[CLS]`. Pretraining and the downstream probe use identical lengths, so no
  position-embedding interpolation is ever needed.
- **Channel-independent by contract.** The backbone treats a `(B, L, C)` input as `B x C`
  univariate series. UCR is univariate (`C = 1`) so this reduces to the obvious thing, but the
  contract is kept because it is what would let the same encoder read multivariate series
  unchanged.
- **Why a separate folder from the notebooks?** Same reason as the remote sensing part: these scripts are not
  teaching material. `tutorial_ts.py` stays the single source of truth for the architecture;
  everything data/training-related here is intentionally independent so changes to the (larger,
  slower) pretraining pipeline can't break the notebooks' teaching code path.
