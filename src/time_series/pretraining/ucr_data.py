"""
Data loading for the time-series pretraining scripts. Counterpart of
`src/remote_sensing/pretraining/seco_data.py` for the time series part.

The pretraining corpus is the TRAIN split of all 128 UCR datasets with labels discarded,
every series resampled to `SERIES_LEN` and per-series z-normalized. That is ~45k series once
the per-dataset cap is applied - about 23 MB as float32, so the whole corpus is held in RAM and
there is no per-item decode and no DataLoader worker fan-out needed.

`src/time_series/tutorial_ts.py` already implements `UCRCorpusDataset` and the archive
download/parse helpers; the scripts here reuse them directly rather than re-implementing the
loading.
"""

import os
import sys

import torch

TS_DIR = os.path.join(os.path.dirname(__file__), "..")
if TS_DIR not in sys.path:
    sys.path.insert(0, TS_DIR)

from tutorial_ts import (  # noqa: E402
    SERIES_LEN,
    TARGET_DATASET,
    UCRCorpusDataset,
    list_ucr_datasets,
)

# Default cap on how many series any single UCR dataset may contribute. The archive is very
# unbalanced (Crop has 7,200 training series, Chinatown has 20), so an uncapped pool would be
# dominated by a handful of members. 2,000 keeps every dataset whole except the largest few.
DEFAULT_PER_DATASET_CAP = 2000


def build_pretraining_dataset(data_path, transform, names=None,
                              per_dataset_cap=DEFAULT_PER_DATASET_CAP, exclude=(),
                              length=SERIES_LEN, seed=42):
    """
    Return a `UCRCorpusDataset` over the whole archive (or `names`), plus the per-dataset series
    counts for reporting. `transform` follows the same contract as in the notebooks: it maps one
    (SERIES_LEN, 1) series to one view or a list of views.

    `exclude` drops datasets by name. Passing `exclude=(TARGET_DATASET,)` gives the strict
    "never saw the target domain at all" corpus used for the transfer ablation; the default
    keeps the target's *training* series in the pool (labels discarded), which is standard
    self-supervised practice and matches the remote sensing part.
    """
    dataset = UCRCorpusDataset(
        data_path, names=names, length=length, transform=transform,
        per_dataset_cap=per_dataset_cap, seed=seed, exclude=exclude,
    )
    return dataset, dataset.corpus_sizes


def multicrop_collate(batch):
    """
    Collate for the DINO script: each sample is a list of crops (from MultiCropTransform).
    Returns a list of length n_crops, each a (B, SERIES_LEN, 1) batch tensor.
    """
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(n_crops)]


__all__ = [
    "DEFAULT_PER_DATASET_CAP",
    "SERIES_LEN",
    "TARGET_DATASET",
    "build_pretraining_dataset",
    "list_ucr_datasets",
    "multicrop_collate",
]
