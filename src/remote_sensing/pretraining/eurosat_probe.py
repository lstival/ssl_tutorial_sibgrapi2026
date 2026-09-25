"""
EuroSAT linear-probe helpers shared by the offline evaluation and hyperparameter-search scripts
(eval_encoder.py, eval_finetune.py, optuna_search.py, optuna_dino.py).

Protocol (identical to Notebook 4): the fixed stratified 80/20 split from
`tutorial_rs.stratified_split` (seed=42), frozen-encoder features, StandardScaler + multinomial
LogisticRegression, and per-class subsampling for the few-label regime.

Model selection must never look at the test split, so `hpo_probe_accuracy` scores on a
validation split carved out of the *train* side; the test split is reserved for the final
numbers reported by eval_encoder.py / Notebook 4.
"""

import numpy as np
import torch
import torch.utils.data as data
from pretrain_utils import DATA_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tutorial_rs import (
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMG_SIZE,
    build_eval_transform,
    load_eurosat,
    stratified_split,
)

_EVAL_SPLITS = {}


def eurosat_eval_splits(data_dir=DATA_DIR):
    """(dataset with the deterministic eval transform, train_idx, test_idx), built once per process."""
    if data_dir not in _EVAL_SPLITS:
        transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
        dataset = load_eurosat(data_dir, transform=transform)
        train_idx, test_idx = stratified_split(dataset, test_size=0.2, seed=42)
        _EVAL_SPLITS[data_dir] = (dataset, train_idx, test_idx)
    return _EVAL_SPLITS[data_dir]


@torch.no_grad()
def extract_features(encoder, dataset, device, pool="cls", batch_size=256):
    """Frozen-encoder features for every item of `dataset`, as (features, labels) numpy arrays."""
    encoder.eval()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, targets in loader:
        feats.append(encoder.forward_features(imgs.to(device), pool=pool).cpu())
        labels.append(targets)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def linear_probe_accuracy(train_feats, train_labels, test_feats, test_labels, max_iter=2000):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    clf.fit(train_feats, train_labels)
    return accuracy_score(test_labels, clf.predict(test_feats))


def subsample_per_class(labels, k, seed=42):
    """Up to `k` indices per class (unsorted -- order matches Notebook 4's subsampler)."""
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:k])
    return np.array(indices)


def hpo_probe_accuracy(encoder, device, pool="cls", per_class=200, val_fraction=0.2, seed=0):
    """
    Cheap probe score for hyperparameter search: fit on `per_class` labels/class from the train
    split, score on a stratified validation split that is also drawn from the train split.
    """
    dataset, train_idx, _ = eurosat_eval_splits()
    train_labels = np.asarray(dataset.targets)[train_idx]

    rng = np.random.RandomState(seed)
    fit_pool, val_idx = [], []
    for cls in np.unique(train_labels):
        cls_idx = train_idx[train_labels == cls]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_val])
        fit_pool.extend(cls_idx[n_val:])
    fit_pool = np.asarray(fit_pool)
    fit_idx = fit_pool[subsample_per_class(np.asarray(dataset.targets)[fit_pool], per_class, seed)]

    fit_x, fit_y = extract_features(encoder, data.Subset(dataset, fit_idx), device, pool)
    val_x, val_y = extract_features(encoder, data.Subset(dataset, val_idx), device, pool)
    return linear_probe_accuracy(fit_x, fit_y, val_x, val_y)
