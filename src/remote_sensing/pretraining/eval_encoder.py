"""
Linear-probe evaluation of one encoder-only checkpoint, in BOTH label regimes, using exactly
the Notebook 4 protocol (stratified 80/20 EuroSAT split at seed=42, StandardScaler +
multinomial LogisticRegression, per-class subsampling at seed=42).

Written for the SeCo corpus-scaling study: each pretraining run produces one checkpoint, and
this script appends that checkpoint's full-label and few-label accuracies to a shared results
JSON so the scaling curve can be plotted from a single file.

Unlike `eval_mae_checkpoint.py` (full-label only, MAE-specific defaults) this reports the
few-label curve too -- which is where a larger pretraining corpus is expected to show up most
clearly, since the full-label probe has only ~3 points of headroom below the supervised
from-scratch ceiling (0.944).

Usage:
    python eval_encoder.py --checkpoint ../../../artifacts/remote_sensing/checkpoints/contrastive_seco_f100.pt \\
        --name f100 --pool cls --out-json ../../../notebooks/remote_sensing/figures/seco_scaling_results.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOTEBOOKS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, NOTEBOOKS)

from tutorial_rs import (  # noqa: E402
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMG_SIZE,
    build_eval_transform,
    build_vit_s8,
    build_vit_t8,
    get_device,
    load_eurosat,
    seed_everything,
    stratified_split,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
LABEL_BUDGETS = [10, 20, 50, 100, 200]

# Encoder variants a checkpoint may have been trained with (see train_contrastive.py --arch).
# Both are patch-8 at 64x64, so the probe protocol below is identical for either.
ARCHS = {"vit_s8": build_vit_s8, "vit_t8": build_vit_t8}


@torch.no_grad()
def prepare_data_features(encoder, dataset, device, pool="cls", batch_size=256):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        feats.append(encoder.forward_features(imgs, pool=pool).cpu())
        labels.append(targets)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def linear_probe_accuracy(train_feats, train_labels, test_feats, test_labels, max_iter=2000):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    clf.fit(train_feats, train_labels)
    return accuracy_score(test_labels, clf.predict(test_feats))


def subsample_per_class(labels, k, seed=42):
    """Same per-class subsampler as Notebook 4, so few-label numbers are directly comparable."""
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:k])
    return np.array(indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Encoder-only checkpoint (.pt).")
    ap.add_argument("--name", required=True, help="Key this run is stored under in --out-json.")
    ap.add_argument("--arch", default="vit_s8", choices=sorted(ARCHS),
                    help="Encoder variant the checkpoint was trained with.")
    ap.add_argument("--pool", default="cls", choices=["mean", "cls"])
    ap.add_argument("--out-json", default=None, help="JSON file to append results to.")
    ap.add_argument("--meta", default=None, help="Optional JSON string of extra metadata to record.")
    args = ap.parse_args()

    seed_everything(42)
    device = get_device()
    print(f"Device: {device} | checkpoint={args.checkpoint} | arch={args.arch} | pool={args.pool}")

    encoder = ARCHS[args.arch]().to(device)
    encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    eval_transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    full_dataset = load_eurosat(DATA_DIR, transform=eval_transform)
    train_idx, test_idx = stratified_split(full_dataset, test_size=0.2, seed=42)

    tr_f, tr_y = prepare_data_features(encoder, data.Subset(full_dataset, train_idx), device, pool=args.pool)
    te_f, te_y = prepare_data_features(encoder, data.Subset(full_dataset, test_idx), device, pool=args.pool)
    print(f"train {tr_f.shape}, test {te_f.shape}")

    full_acc = linear_probe_accuracy(tr_f, tr_y, te_f, te_y)
    print(f"\nfull-label linear probe: {full_acc:.4f}")

    few = {}
    for k in LABEL_BUDGETS:
        sub = subsample_per_class(tr_y, k)
        few[k] = linear_probe_accuracy(tr_f[sub], tr_y[sub], te_f, te_y)
        print(f"  k={k:4d} labels/class: {few[k]:.4f}")

    if args.out_json:
        results = {}
        if os.path.isfile(args.out_json):
            with open(args.out_json, encoding="utf-8") as f:
                results = json.load(f)
        entry = {
            "checkpoint": os.path.basename(args.checkpoint),
            "arch": args.arch,
            "pool": args.pool,
            "full_label": full_acc,
            "few_label": {str(k): v for k, v in few.items()},
        }
        if args.meta:
            entry["meta"] = json.loads(args.meta)
        results[args.name] = entry
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results for '{args.name}' to {args.out_json}")


if __name__ == "__main__":
    main()
