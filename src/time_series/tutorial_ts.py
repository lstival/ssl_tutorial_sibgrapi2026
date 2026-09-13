"""
Shared utilities for the SIBGRAPI 2026 SSL Tutorial: time series part (UCR).

This is the time-series counterpart of `src/remote_sensing/tutorial_rs.py`. It centralizes the
plumbing that every notebook in `notebooks/time_series/` needs, so each notebook can stay
focused on the SSL mechanism it teaches. The parallel to the remote sensing part is deliberate
and now complete: same three mechanisms (contrastive / masking / distillation), same
fill-in-the-blank teaching pattern, same "pretrain offline -> freeze -> linear probe"
evaluation, and, the point of this module, the same *downstream task*. The remote sensing part
classifies satellite tiles; the time series part classifies time series. Only the data geometry
changes, from a 2D image grid to a 1D sequence.

Why classification and not forecasting: the tutorial's whole argument is that one SSL recipe
transfers across modalities, and that argument is only legible if the evaluation is held fixed.
A frozen encoder + linear probe + accuracy reads identically in both parts, so the comparison
tables line up row for row. It also matches the literature the paper cites for time series,
which is classification-oriented throughout.

What lives here:

  - the UCR Time Series Archive (2018): 128 univariate classification datasets, loaded from
    the official `.tsv` files, with a downloader that fetches and unzips the archive on first use
  - the archive's own canonical TRAIN/TEST split (unlike EuroSAT, UCR ships one - so we use it
    rather than inventing a split)
  - fixed-length resampling to `SERIES_LEN` (128) so one backbone serves every dataset
  - per-series z-normalization (the UCR convention, and the 1D analogue of image normalization)
  - time-series-specific augmentations (see the note in `build_ts_augmentations`)
  - a shared patch Transformer backbone (`TSEncoder`, PatchTST-style): patch the series,
    linearly embed, prepend a [CLS] token, add learnable position embeddings, run Transformer
    blocks (embed dim 128, depth 3)
  - plotting helpers with a consistent visual style
  - a checkpoint / precomputed-output downloader ("Found pretrained model, loading...")

Every notebook downloads this file at the top (Colab) or imports it directly (local run).
"""

import math
import os
import urllib.request
import zipfile
from urllib.error import HTTPError

# Some conda environments ship two OpenMP runtimes (PyTorch's and MKL's, via NumPy/SciPy).
# When a notebook calls into SciPy/NumPy LAPACK (e.g. scikit-learn's logistic regression in
# Notebook 4) after torch is already loaded, that clash can abort the kernel with
# "OMP: Error #15". Setting this before NumPy is imported is the standard, documented
# workaround and does not affect results at the scale used here. Set only if the user has not
# already chosen a value.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# =====================================================================================
# Reproducibility & device
# =====================================================================================

SEED = 42


def seed_everything(seed=SEED):
    """Seed python, numpy and torch (CPU+CUDA) for reproducible splits/augmentations/training."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# =====================================================================================
# The UCR Time Series Archive (2018)
# =====================================================================================
#
# The UCR archive (Dau et al., 2019) is the standard benchmark for time-series
# classification - the "ImageNet of time series" in the sense that essentially every method in
# the literature reports on it. The 2018 release ships 128 univariate datasets spanning sensor
# traces, motion capture, spectrograms, ECG, image-derived outlines and simulated signals.
#
# Each dataset is a directory holding `<Name>_TRAIN.tsv` and `<Name>_TEST.tsv`. Every row is
# one series: the first column is the class label, the remaining columns are the values. All
# series within a dataset share one length; lengths differ *between* datasets (15 to 2844
# steps), which is exactly why this module resamples everything to a common `SERIES_LEN`.
#
# The archive is distributed as a single password-protected zip (the password is published on
# the archive's own page and is not a secret - it exists so that downloaders acknowledge the
# accompanying documentation). We fetch and unzip it on first use, the same way
# `tutorial_rs.py` lets torchvision download EuroSAT.

UCR_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
UCR_ZIP_PASSWORD = b"someone"
UCR_DIRNAME = "UCRArchive_2018"

# The downstream target for the time series part, chosen as the flagship the way EuroSAT is
# the flagship of the remote sensing part.
#
# SwedishLeaf: 500 train / 625 test series, 15 classes, length 128. Each series is the radial
# distance from a leaf's centroid to its outline, traced once around the boundary - so a
# "time" series that is really a shape descriptor, which makes the classes visually separable
# in a plot and gives Notebook 0 something honest to show. Three properties earned it the slot:
#   - 15 classes and a 625-series test set: enough resolution that linear-probe accuracy moves
#     meaningfully between encoders, unlike the many 2-class UCR datasets where everything
#     saturates near the ceiling
#   - native length 128 == SERIES_LEN, so the flagship needs no resampling at all and what the
#     notebooks plot is exactly what the model sees
#   - a genuinely hard-but-not-hopeless task: a 1-NN Euclidean baseline sits around 0.79, so
#     there is real headroom for representation learning to show up in the numbers
TARGET_DATASET = "SwedishLeaf"

# Every series is resampled to this length, so a single backbone with fixed position embeddings
# serves all 128 datasets in the pretraining corpus and the downstream probe alike. 128 is the
# median-ish scale of the archive, is SwedishLeaf's native length, and divides evenly by the
# patch length below.
SERIES_LEN = 128

# Patch settings for the backbone. 128 / 16 = 8 patches per series (+1 [CLS] token).
PATCH_LEN = 16
PATCH_STRIDE = 16


def ucr_root(data_path):
    """The extracted archive directory inside `data_path`."""
    return os.path.join(data_path, UCR_DIRNAME)


def ensure_ucr_archive(data_path):
    """
    Download and extract the UCR 2018 archive into `data_path` if it is not already there.
    Returns the archive root directory. The zip is ~316 MB, so this runs once.
    """
    root = ucr_root(data_path)
    if os.path.isdir(root) and len(_list_ucr_datasets(root)) > 100:
        return root

    os.makedirs(data_path, exist_ok=True)
    zip_path = os.path.join(data_path, "UCRArchive_2018.zip")
    if not os.path.isfile(zip_path):
        print(f"Downloading the UCR archive (~316 MB) from {UCR_URL} ...")
        urllib.request.urlretrieve(UCR_URL, zip_path)
    print("Extracting the UCR archive ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_path, pwd=UCR_ZIP_PASSWORD)
    print(f"UCR archive ready at {root}")
    return root


def _list_ucr_datasets(root):
    """Names of every directory under `root` that actually holds a `<Name>_TRAIN.tsv`."""
    if not os.path.isdir(root):
        return []
    names = []
    for d in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, d, f"{d}_TRAIN.tsv")):
            names.append(d)
    return names


def list_ucr_datasets(data_path):
    """
    All usable dataset names in the archive (128 of them). The archive also ships a
    `Missing_value_and_variable_length_datasets_adjusted` helper folder, which holds
    pre-imputed copies rather than a dataset of its own; it has no top-level `_TRAIN.tsv`
    and is filtered out here.
    """
    return _list_ucr_datasets(ensure_ucr_archive(data_path))


def _read_ucr_tsv(path):
    """
    Read one UCR `.tsv` into (labels, series). Values may contain NaN: 15 of the 128 datasets
    hold variable-length series that the archive pads with NaN to a common width, and a few
    (the DodgerLoop family) have genuinely missing interior samples.
    """
    raw = np.genfromtxt(path, delimiter="\t", dtype=np.float64)
    if raw.ndim == 1:  # a single-row file
        raw = raw[None, :]
    return raw[:, 0], raw[:, 1:].astype(np.float32)


def _clean_series(x):
    """
    Repair one raw series: fill NaNs, then guarantee a finite, non-degenerate result.

    Two distinct NaN patterns exist in the archive and both land here:
      - *trailing* NaN padding, where a short series was widened to the dataset's max length
        (PLAID, the Wiimote/gesture families). Dropping the pad is the right move - the series
        genuinely ends there - and the resampler below stretches what remains back to
        SERIES_LEN.
      - *interior* NaN (the DodgerLoop family), which is real sensor dropout. Those are filled
        by linear interpolation between the surrounding observed samples.
    """
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(max(len(x), 2), dtype=np.float32)
    # Trim padding at either end, then interpolate whatever holes remain inside.
    idx = np.where(finite)[0]
    x = x[idx[0]: idx[-1] + 1]
    finite = np.isfinite(x)
    if not finite.all():
        pos = np.arange(len(x))
        x = np.interp(pos, pos[finite], x[finite]).astype(np.float32)
    if len(x) < 2:
        x = np.repeat(x, 2)
    return x.astype(np.float32)


def resample_series(x, length=SERIES_LEN):
    """
    Resample a 1-D series to exactly `length` points by linear interpolation over a normalized
    time axis.

    This is what lets one fixed-size backbone read all 128 datasets. It is the sequence
    analogue of resizing every image to 64x64 in the remote sensing part: it changes the
    sampling rate, not the
    shape of the signal, so trend and morphology - the things the classes actually differ in -
    survive. Datasets already at `length` (SwedishLeaf among them) pass through untouched.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == length:
        return x
    src = np.linspace(0.0, 1.0, num=len(x), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    return np.interp(dst, src, x).astype(np.float32)


def znormalize_series(x, eps=1e-8):
    """
    Per-series z-normalization: subtract this series' own mean, divide by its own std.

    This is the UCR convention (most datasets ship already normalized this way) and it is the
    right invariance for classification: a leaf outline or an ECG beat means the same thing
    whether the sensor was offset or scaled. It is also the counterpart of the per-window
    instance normalization the forecasting setup needed - here it is applied once, up front,
    instead of being a learnable module, because the label is invariant to level and scale by
    construction.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean()
    std = x.std()
    if std < eps:
        std = 1.0
    return ((x - mean) / std).astype(np.float32)


def load_ucr_dataset(data_path, name=TARGET_DATASET, split="train", length=SERIES_LEN,
                     znorm=True):
    """
    Load one UCR dataset split as (X, y).

    X: (n_series, length) float32, cleaned, resampled to `length`, per-series z-normalized.
    y: (n_series,) int64 class indices remapped to a contiguous 0..n_classes-1 range.

    The remap matters: raw UCR labels are inconsistent across the archive - most datasets use
    1..k, several use -1/+1 (ECG200, FordA/B, Wafer, Lightning2), and the phalanx-outline
    "TW" datasets start at 3. Class indices are always derived from the TRAIN split so that
    train and test agree even when a rare class is absent from one of them.
    """
    assert split in ("train", "test")
    root = ensure_ucr_archive(data_path)
    ds_dir = os.path.join(root, name)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"UCR dataset {name!r} not found under {root}")

    # Always read TRAIN to fix the label vocabulary, even when returning TEST.
    train_labels, _ = _read_ucr_tsv(os.path.join(ds_dir, f"{name}_TRAIN.tsv"))
    classes = np.unique(train_labels)
    label_to_index = {float(c): i for i, c in enumerate(classes)}

    split_file = f"{name}_{'TRAIN' if split == 'train' else 'TEST'}.tsv"
    labels, series = _read_ucr_tsv(os.path.join(ds_dir, split_file))

    X = np.empty((len(series), length), dtype=np.float32)
    for i, row in enumerate(series):
        s = resample_series(_clean_series(row), length)
        X[i] = znormalize_series(s) if znorm else s

    # A label unseen in TRAIN (does not occur in the archive, but be explicit) maps to -1.
    y = np.array([label_to_index.get(float(v), -1) for v in labels], dtype=np.int64)
    return X, y


def ucr_class_names(data_path, name=TARGET_DATASET):
    """
    Display names for the classes, in the same order as the remapped indices.

    UCR ships no semantic class names in the data files, so these are the raw archive labels
    rendered as strings ("class 1", "class 2", ...) - enough to key a legend or a confusion
    matrix. SwedishLeaf's 15 classes are 15 tree species; the archive does not name them.
    """
    root = ensure_ucr_archive(data_path)
    train_labels, _ = _read_ucr_tsv(os.path.join(root, name, f"{name}_TRAIN.tsv"))
    classes = np.unique(train_labels)
    return [f"class {int(c) if float(c).is_integer() else c}" for c in classes]


def ucr_dataset_summary(data_path, name=TARGET_DATASET):
    """Shape/label summary of one dataset, for the tables in Notebook 0."""
    Xtr, ytr = load_ucr_dataset(data_path, name, "train")
    Xte, yte = load_ucr_dataset(data_path, name, "test")
    return {
        "name": name,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "n_classes": int(len(np.unique(ytr))),
        "length": Xtr.shape[1],
    }


class UCRDataset(torch.utils.data.Dataset):
    """
    One UCR dataset split as a torch Dataset.

    Each item is `(series, label)` where `series` is a (SERIES_LEN, 1) float32 tensor - the
    trailing 1 is the channel axis, kept so the backbone's input contract is (B, L, C) and
    matches the multivariate case even though UCR is univariate.

    `transform`, if given, is applied to that tensor and may return one augmented view or a
    list of views (mirroring `ContrastiveTransformations` / `MultiCropTransform` in
    `tutorial_rs.py`).
    """

    def __init__(self, data_path, name=TARGET_DATASET, split="train", transform=None,
                 length=SERIES_LEN):
        self.X, self.y = load_ucr_dataset(data_path, name, split, length=length)
        self.transform = transform
        self.name = name
        self.split = split
        self.targets = self.y  # named like torchvision datasets, for stratified subsampling

    @property
    def n_classes(self):
        return int(len(np.unique(self.y)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        series = torch.from_numpy(self.X[idx]).float().unsqueeze(-1)  # (L, 1)
        label = int(self.y[idx])
        if self.transform is not None:
            return self.transform(series), label
        return series, label


# =====================================================================================
# Multi-dataset pretraining corpus
# =====================================================================================
#
# The encoder reads a single univariate series of fixed length, so the pretraining corpus does
# not have to come from one dataset - it can be every series in the archive, pooled and
# unlabeled. That is exactly the remote sensing setup: pretrain on a large unlabeled corpus
# (SeCo there,
# the whole UCR archive here), then probe a frozen encoder on one target dataset (EuroSAT /
# SwedishLeaf).
#
# The corpus is the TRAIN split of all 128 datasets with labels discarded - ~60k series, which
# after resampling to length 128 is ~30 MB and lives entirely in RAM, so there is no per-item
# decode and no DataLoader worker fan-out needed.
#
# The target dataset's TEST split is never touched during pretraining. Its TRAIN split *is*
# included (labels discarded), which is standard self-supervised practice and matches the
# remote sensing part,
# where the EuroSAT probe's training images are also part of the unlabeled pool.


class UCRCorpusDataset(torch.utils.data.Dataset):
    """
    Unlabeled series pooled across many UCR datasets - the SSL pretraining view of the archive.

    Every series is cleaned, resampled to `length` and per-series z-normalized, so a 15-step
    Chinatown trace and a 2844-step Rock spectrum land on the same shape and the same numeric
    scale. Each item is a (length, 1) float32 tensor; `transform` follows the same contract as
    in the notebooks.

    `per_dataset_cap` bounds how many series any single dataset may contribute, so the corpus
    is not dominated by the handful of very large datasets (Crop alone has 7,200 training
    series). The cap is applied with a fixed seed, so the corpus is reproducible.
    """

    def __init__(self, data_path, names=None, length=SERIES_LEN, transform=None,
                 per_dataset_cap=2000, seed=SEED, exclude=(), verbose=True):
        self.transform = transform
        self.length = length
        root = ensure_ucr_archive(data_path)
        if names is None:
            names = _list_ucr_datasets(root)
        names = [n for n in names if n not in set(exclude)]

        rng = np.random.RandomState(seed)
        chunks = []
        self.corpus_sizes = {}
        for name in names:
            try:
                X, _ = load_ucr_dataset(data_path, name, "train", length=length)
            except Exception as e:  # a malformed dataset must not kill a long pretraining run
                if verbose:
                    print(f"  skipping {name}: {e}")
                continue
            if per_dataset_cap is not None and len(X) > per_dataset_cap:
                keep = rng.choice(len(X), size=per_dataset_cap, replace=False)
                X = X[keep]
            chunks.append(X)
            self.corpus_sizes[name] = len(X)

        self.X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, length), np.float32)
        if verbose:
            print(f"UCR pretraining corpus: {len(self.X)} series from "
                  f"{len(self.corpus_sizes)} datasets")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        series = torch.from_numpy(self.X[idx]).float().unsqueeze(-1)  # (L, 1)
        if self.transform is not None:
            return self.transform(series)
        return series


# =====================================================================================
# Time-series-specific augmentations
# =====================================================================================


class ContrastiveTransformations:
    """Applies `base_transforms` independently `n_views` times to the same series (SimCLR-style)."""

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class MultiCropTransform:
    """
    DINO-style multi-crop for sequences: `n_global` long crops + `n_local` short crops of the
    same series, each independently augmented. Short crops are stretched back to `SERIES_LEN`
    by linear interpolation so every view has the same length and the shared backbone can
    encode all of them.
    """

    def __init__(self, global_transform, local_transform, n_global=2, n_local=4):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(self.n_global)]
        crops += [self.local_transform(x) for _ in range(self.n_local)]
        return crops


def _rand(*shape):
    return torch.rand(*shape)


class TSAugment:
    """
    A composable random augmentation for a single (L, C) series. Each call draws fresh
    parameters, so two calls on the same series give two independent views.

    The recipe below is deliberately different from the natural-image / remote-sensing ones,
    and that difference is a teaching point (see Notebook 0):

      - **Jitter (additive Gaussian noise)** models sensor noise - label-preserving for
        essentially every UCR task.
      - **Scaling** (multiply by a scalar near 1) and **level shift** (add a constant) model
        gain and offset drift. The class label is invariant to both - indeed the per-series
        z-normalization already removes them, which is precisely why they are safe to apply.
      - **Time masking** zeros a short contiguous span - the sequence analogue of a cutout, and
        a natural bridge to the masked-prediction mechanism in Notebook 2.
      - **Crop-and-resize** takes a random sub-interval and stretches it back to length L,
        a mild time-warp that keeps the class-defining morphology intact.
      - There is **no vertical flip and no time reversal**: unlike a nadir satellite tile, a
        time series has a canonical direction and a canonical sign. Reversing a leaf outline
        or flipping an ECG changes the very thing the label describes - for several UCR
        datasets a reversed series belongs to a *different* class.
    """

    def __init__(self, jitter_sigma=0.15, scale_sigma=0.10, shift_sigma=0.15,
                 mask_frac=0.10, crop_min=0.5, p_mask=0.5, p_crop=0.5):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.shift_sigma = shift_sigma
        self.mask_frac = mask_frac
        self.crop_min = crop_min
        self.p_mask = p_mask
        self.p_crop = p_crop

    def __call__(self, x):
        # x: (L, C)
        L, C = x.shape
        x = x.clone()

        # crop-and-resize (mild time warp)
        if float(_rand(1)) < self.p_crop:
            frac = self.crop_min + (1.0 - self.crop_min) * float(_rand(1))
            crop_len = max(8, int(round(L * frac)))
            s = int(torch.randint(0, L - crop_len + 1, (1,)))
            sub = x[s:s + crop_len].transpose(0, 1).unsqueeze(0)  # (1, C, crop_len)
            x = torch.nn.functional.interpolate(
                sub, size=L, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)

        # per-channel scaling and level shift (gain/offset nuisance)
        scale = 1.0 + self.scale_sigma * torch.randn(1, C)
        shift = self.shift_sigma * torch.randn(1, C)
        x = x * scale + shift

        # additive jitter (sensor noise)
        x = x + self.jitter_sigma * torch.randn(L, C)

        # contiguous time masking (cutout)
        if float(_rand(1)) < self.p_mask:
            mask_len = max(1, int(round(L * self.mask_frac)))
            s = int(torch.randint(0, L - mask_len + 1, (1,)))
            x[s:s + mask_len] = 0.0

        return x


def build_ts_augmentations():
    """Standard augmentation for the contrastive / DINO-global views."""
    return TSAugment(jitter_sigma=0.15, scale_sigma=0.10, shift_sigma=0.15,
                     mask_frac=0.10, crop_min=0.5, p_mask=0.5, p_crop=0.5)


def build_ts_local_augmentations():
    """Heavier crop for DINO local views: take a short sub-series and stretch it to full length."""
    return TSAugment(jitter_sigma=0.15, scale_sigma=0.10, shift_sigma=0.15,
                     mask_frac=0.15, crop_min=0.15, p_mask=0.5, p_crop=1.0)


def build_eval_transform():
    """Deterministic 'transform' for probing/eval/viz: identity (no augmentation)."""
    return lambda x: x


# =====================================================================================
# Patch Transformer backbone (shared by all three SSL mechanisms)
# =====================================================================================
#
# Design choice mirroring the remote sensing part's single shared ViT-S/8: one backbone
# across the contrastive,
# masking and distillation notebooks so Notebook 4's linear probe is a fair comparison
# (identical architecture and token budget across all three encoders).
#
# The architecture is PatchTST-style (Nie et al., 2023) and CHANNEL-INDEPENDENT: a (B, L, C)
# input is treated as B*C univariate series of length L, each cut into patches, embedded, and
# run through the same Transformer. UCR is univariate (C=1), so this reduces to the obvious
# thing here - but the contract is kept because it is what makes the same encoder reusable for
# multivariate series, and because it keeps the code honest about which axis is which.


class PatchEmbed(nn.Module):
    """Cut a univariate series (N, L) into patches and linearly project each to `embed_dim`."""

    def __init__(self, seq_len=SERIES_LEN, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 embed_dim=128):
        super().__init__()
        assert (seq_len - patch_len) % stride == 0, "seq_len, patch_len, stride are inconsistent"
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, embed_dim)

    def forward(self, x):
        # x: (N, L) univariate -> (N, num_patches, embed_dim)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block: MHSA + MLP, both with residual connections."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=return_attn,
                                           average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


class TSEncoder(nn.Module):
    """
    The shared patch Transformer encoder, used by all SSL mechanisms and by the supervised
    baseline. It plays exactly the role the shared ViT-S/8 plays in the remote sensing part:
    one backbone for every training regime, which is what makes the linear probe a fair
    comparison between mechanisms rather than between architectures.

    Defaults: embed_dim=128, depth=3, num_heads=8, patch_len=stride=16 on a 128-step series
    -> 8 patch tokens (+1 [CLS]). Depth 3 keeps live training fast on a single Colab GPU while
    still showing clear SSL signal at UCR scale (~0.6M parameters).

    A learnable [CLS] token is prepended (the global representation for the contrastive /
    DINO objectives and for the linear probe, and the query for DINO's attention maps).
    Learnable absolute position embeddings are added; the sequence length is fixed everywhere
    so no interpolation is ever needed.

    Forward contract: input is (B, L, C); internally reshaped to (B*C, L), so the token
    sequence returned is (B*C, 1+num_patches, embed_dim). `forward_features` folds the channel
    axis back and returns one (B, embed_dim) representation.
    """

    def __init__(self, seq_len=SERIES_LEN, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 num_channels=1, embed_dim=128, depth=3, num_heads=8,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(seq_len, patch_len, stride, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.grid_size = num_patches  # name kept parallel to the image encoder's grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _to_channel_independent(self, x):
        """(B, L, C) -> (B*C, L), remembering B and C."""
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)
        return x, B, C

    def forward(self, x, return_attn=False):
        """
        x: (B, L, C). Returns token sequence (B*C, 1+num_patches, embed_dim); if `return_attn`,
        also returns the last block's attention weights (B*C, num_heads, 1+P, 1+P).
        """
        x, B, C = self._to_channel_independent(x)
        tokens = self.patch_embed(x)                       # (B*C, P, D)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)    # (B*C, 1+P, D)
        tokens = tokens + self.pos_embed

        attn_weights = None
        for i, block in enumerate(self.blocks):
            is_last = i == len(self.blocks) - 1
            if return_attn and is_last:
                tokens, attn_weights = block(tokens, return_attn=True)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)

        if return_attn:
            return tokens, attn_weights
        return tokens

    def forward_features(self, x, pool="cls", per_channel=False):
        """
        Return a global embedding for linear probing.

        pool="cls"  -> the [CLS] token (contrastive / DINO train this directly).
        pool="mean" -> mean over patch tokens (excludes [CLS]). MAE never supervises [CLS], so
                       a [CLS] probe understates an MAE encoder; mean-pooling patch tokens is
                       the standard masked-model readout.

        per_channel=False (default): average the per-channel embeddings -> (B, embed_dim).
        For univariate UCR data C=1, so this is just a squeeze.
        """
        B, L, C = x.shape
        tokens = self.forward(x)  # (B*C, 1+P, D)
        if pool == "mean":
            feat = tokens[:, 1:].mean(dim=1)
        else:
            feat = tokens[:, 0]
        feat = feat.reshape(B, C, self.embed_dim)
        if per_channel:
            return feat
        return feat.mean(dim=1)

    def forward_tokens(self, x):
        """
        Full patch-token sequence per channel, [CLS] dropped: (B, C, num_patches, embed_dim).
        Used by anything that needs the per-patch representation rather than a pooled vector.
        """
        B, L, C = x.shape
        tokens = self.forward(x)[:, 1:, :]  # (B*C, P, D)
        return tokens.reshape(B, C, self.num_patches, self.embed_dim)


def build_ts_encoder(**kwargs):
    """Factory for the shared patch Transformer with the tutorial defaults."""
    defaults = dict(seq_len=SERIES_LEN, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                    num_channels=1, embed_dim=128, depth=3, num_heads=8)
    defaults.update(kwargs)
    return TSEncoder(**defaults)


# =====================================================================================
# Linear-probe helpers
# =====================================================================================
#
# Identical in protocol to the remote sensing
# `notebooks/remote_sensing/04_comparative_evaluation.ipynb`:
# extract frozen features once, then fit a multinomial logistic regression on top. Keeping the
# probe identical across the two parts is the whole point - it is what makes the remote-sensing
# and time-series result tables directly comparable.


@torch.no_grad()
def extract_features(encoder, dataset, device, pool="cls", batch_size=256):
    """
    Run a frozen encoder over a dataset and return (features, labels) as numpy arrays.
    `dataset` yields (series, label) pairs; no augmentation should be applied.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0)
    encoder.eval()
    feats, labels = [], []
    for series, target in loader:
        series = series.to(device)
        feats.append(encoder.forward_features(series, pool=pool).cpu())
        labels.append(target)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def linear_probe_accuracy(train_feats, train_labels, test_feats, test_labels, max_iter=2000,
                          return_preds=False):
    """
    Fit a standardized multinomial logistic regression on frozen features and report test
    accuracy. Exactly the probe used in the remote sensing part.

    Standardizing first matters: Transformer embeddings have very different per-dimension
    scales, and unscaled features make lbfgs converge slowly or not at all.

    With `return_preds=True` returns (accuracy, preds) instead of just the accuracy, which is
    what the confusion matrices in the comparative-evaluation notebook need.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    accuracy = accuracy_score(test_labels, preds)
    return (accuracy, preds) if return_preds else accuracy


def subsample_per_class(labels, k, seed=SEED):
    """
    Indices of at most `k` examples per class, drawn with a fixed seed - the few-label regime
    used in both parts. Classes with fewer than `k` examples contribute all they have.
    """
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:k])
    return np.array(sorted(indices))


# =====================================================================================
# Checkpoint / precomputed-output downloader
# =====================================================================================

# The pretrained encoders are *committed to this repository* via Git LFS, under
# `artifacts/time_series/checkpoints/`. There is no external release to publish and no tag to
# bump: a clone (with git-lfs installed) already has every weight these notebooks load.
#
# These are the UCR-classification encoders: a 128-step series, patch 16, [CLS] readout. The
# earlier ETT-forecasting checkpoints (512-step lookback, forecasting head) are incompatible
# with this architecture and are deliberately not carried over.
#
# Two ways a notebook finds them, in this order:
#   1. On disk, relative to the notebook -- the local-clone path, and the only path used offline.
#   2. Downloaded from this repository over HTTPS -- the Colab path. LFS-backed files are served
#      by media.githubusercontent.com; raw.githubusercontent.com returns the ~130-byte *pointer
#      file* instead of the tensor, which `_looks_like_lfs_pointer` rejects loudly.
GITHUB_REPO = "lstival/ssl_tutorial_sibgrapi2026"
GITHUB_BRANCH = "main"
CHECKPOINT_REPO_PATH = "artifacts/time_series/checkpoints"
CHECKPOINT_BASE_URL = (
    f"https://media.githubusercontent.com/media/{GITHUB_REPO}/{GITHUB_BRANCH}/{CHECKPOINT_REPO_PATH}/"
)
PRECOMPUTED_BASE_URL = CHECKPOINT_BASE_URL

# Where a local clone keeps the weights, relative to a notebook in notebooks/time_series/.
LOCAL_CHECKPOINT_DIRS = (
    os.path.join("..", "..", CHECKPOINT_REPO_PATH),
    CHECKPOINT_REPO_PATH,
)


def _looks_like_lfs_pointer(file_path):
    """
    True if `file_path` is a Git LFS pointer file rather than real tensor data.

    Happens when a clone is made without git-lfs installed, or when an LFS-backed file is
    fetched from raw.githubusercontent.com instead of the media host. Detecting it here turns
    an opaque torch.load unpickling error into a clear message.
    """
    try:
        if os.path.getsize(file_path) > 1024:
            return False
        with open(file_path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def resolve_local_checkpoint(file_name, checkpoint_path=None):
    """
    Return the path to an already-present copy of `file_name`, or None.

    Checks the caller's own checkpoint directory first, then the in-repo location, so a local
    clone never re-downloads a weight it already has committed.
    """
    candidates = []
    if checkpoint_path:
        candidates.append(os.path.join(checkpoint_path, file_name))
    candidates.extend(os.path.join(d, file_name) for d in LOCAL_CHECKPOINT_DIRS)
    for path in candidates:
        if os.path.isfile(path) and not _looks_like_lfs_pointer(path):
            return path
    return None


def download_files(file_names, checkpoint_path, base_url=CHECKPOINT_BASE_URL):
    """
    Make each file in `file_names` available inside `checkpoint_path`.

    A file already present in the clone is reused as-is. Otherwise it is downloaded from this
    repository (the Colab path). Missing files are reported but never raise: every notebook
    falls back to a short live-training run when a checkpoint cannot be obtained.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    for file_name in file_names:
        file_path = os.path.join(checkpoint_path, file_name)
        if "/" in file_name:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.isfile(file_path):
            if not _looks_like_lfs_pointer(file_path):
                continue
            print(f"{file_path} is a Git LFS pointer, not the weights. Re-downloading...")
            os.remove(file_path)

        local = resolve_local_checkpoint(file_name)
        if local:
            print(f"Using checkpoint from the local clone: {local}")
            continue

        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            if _looks_like_lfs_pointer(file_path):
                os.remove(file_path)
                print(
                    f"  Got a Git LFS pointer instead of {file_name}. The download URL must "
                    f"point at media.githubusercontent.com, not raw.githubusercontent.com."
                )
        except HTTPError as e:
            print(
                f"Could not download {file_name} ({e}). "
                f"This notebook will fall back to live training instead."
            )
        except Exception as e:
            print(f"Could not download {file_name}: {e}")


def try_load_checkpoint(model, checkpoint_path, file_name, device, base_url=CHECKPOINT_BASE_URL):
    """
    Attempt to download + load a state dict into `model`. Returns True if a pretrained
    checkpoint was found and loaded ("Found pretrained model, loading..."), False
    otherwise (caller should train live instead).
    """
    download_files([file_name], checkpoint_path, base_url=base_url)
    file_path = resolve_local_checkpoint(file_name, checkpoint_path)
    if file_path:
        print(f"Found pretrained model at {file_path}, loading...")
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        return True
    print(f"No pretrained checkpoint found for {file_name}. Train live below instead.")
    return False


# =====================================================================================
# Plotting helpers
# =====================================================================================


def setup_plotting():
    """Consistent plotting defaults across the tutorial notebooks.

    Light background with a soft grid, so figures stay readable both in a
    notebook and when projected on a screen during the tutorial.
    """
    import matplotlib

    plt.set_cmap("cividis")
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except ImportError:
        matplotlib.rcParams["axes.grid"] = True

    matplotlib.rcParams.update({
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.6",
        "axes.linewidth": 0.8,
        "grid.color": "0.88",
        "grid.linewidth": 0.8,
        "image.cmap": "cividis",
    })


def plot_curve(values, xlabel, ylabel, title=None, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_series_grid(series, labels=None, class_names=None, n_show=6, title=None,
                     figsize=(12, 6)):
    """
    Plot a small grid of series. `series` is a tensor/array of shape (N, L, C) or (N, L).
    If `labels` is given each panel is titled with its class.
    """
    series = np.asarray(series)
    if series.ndim == 3:
        series = series[..., 0]
    n_show = min(n_show, len(series))
    ncol = 3
    nrow = int(math.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n_show:
            ax.axis("off")
            continue
        ax.plot(series[k], linewidth=1.2)
        if labels is not None:
            lbl = int(labels[k])
            name = class_names[lbl] if class_names is not None else f"class {lbl}"
            ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_class_overlay(X, y, class_names=None, max_classes=6, per_class=8, title=None,
                       figsize=(12, 6)):
    """
    One panel per class, overlaying several series from that class - the quickest read on
    whether a dataset's classes are visually separable at all.
    """
    X = np.asarray(X)
    if X.ndim == 3:
        X = X[..., 0]
    classes = np.unique(y)[:max_classes]
    ncol = 3
    nrow = int(math.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= len(classes):
            ax.axis("off")
            continue
        cls = classes[k]
        idx = np.where(y == cls)[0][:per_class]
        for i in idx:
            ax.plot(X[i], linewidth=0.9, alpha=0.75)
        name = class_names[int(cls)] if class_names is not None else f"class {int(cls)}"
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_embedding_scatter(embedding_2d, labels, class_names=None, title=None, figsize=(7, 6),
                           ax=None, legend=True):
    """
    2D scatter (t-SNE/UMAP output) colored by class, with a legend.

    Pass `ax` to draw into an existing subplot instead of a fresh figure; the caller then owns
    showing and closing it. This is what lets the comparative-evaluation notebook put one panel
    per family side by side with a single shared legend.
    """
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=figsize)
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab20" if len(classes) > 10 else "tab10")
    for i, cls in enumerate(classes):
        mask = labels == cls
        name = class_names[int(cls)] if class_names is not None else f"class {int(cls)}"
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=8, color=cmap(i % cmap.N), label=name, alpha=0.7)
    if legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    if title:
        ax.set_title(title)
    if own_figure:
        plt.tight_layout()
        plt.show()
        plt.close()
    return ax


def plot_reconstruction(original, reconstruction, mask=None, patch_len=PATCH_LEN,
                        stride=PATCH_STRIDE, title=None, figsize=(11, 3)):
    """
    Show an original series against a model reconstruction, shading the patches that were
    masked out - the 1D analogue of MAE's masked-image figure in the remote sensing part.
    """
    original = np.asarray(original).reshape(-1)
    reconstruction = np.asarray(reconstruction).reshape(-1)
    plt.figure(figsize=figsize)
    if mask is not None:
        for p, m in enumerate(np.asarray(mask).reshape(-1)):
            if m:
                s = p * stride
                plt.axvspan(s, min(s + patch_len, len(original)),
                            color="0.75", alpha=0.5, lw=0)
    plt.plot(original, color="0.15", linewidth=1.4, label="original")
    plt.plot(reconstruction, color="tab:red", linestyle="--", linewidth=1.4,
             label="reconstruction")
    plt.xlabel("time step")
    plt.yticks([])
    plt.legend(fontsize=8)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def overlay_attention_1d(series_1d, attn_patches, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                         title=None, figsize=(11, 3)):
    """
    Show a univariate series with a per-patch attention weight (already normalized to [0,1])
    drawn as a shaded band behind it - the 1D analogue of `show_attention_grid` in the image
    `tutorial_rs.py`.
    """
    series_1d = np.asarray(series_1d).reshape(-1)
    L = len(series_1d)
    plt.figure(figsize=figsize)
    for p, w in enumerate(attn_patches):
        s = p * stride
        plt.axvspan(s, min(s + patch_len, L), color="tab:orange", alpha=float(w) * 0.6, lw=0)
    plt.plot(series_1d, color="0.15", linewidth=1.2)
    plt.xlabel("time step")
    plt.yticks([])
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()
