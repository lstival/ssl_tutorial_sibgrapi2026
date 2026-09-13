"""
Standalone linear-probe evaluation for a MAE encoder checkpoint, mirroring the exact protocol
used in Notebook 4 (04_comparative_evaluation.ipynb): stratified 80/20 EuroSAT split (seed=42),
mean-pool readout, StandardScaler + multinomial LogisticRegression on the FULL train split
(not the Optuna proxy's 200/class subsample).

Usage:
    python eval_mae_checkpoint.py --checkpoint ../../../artifacts/remote_sensing/checkpoints/mae_vit_s8_optuna40.pt
    python eval_mae_checkpoint.py --checkpoint ../../../artifacts/remote_sensing/checkpoints/mae_vit_s8.pt
"""
import argparse
import os
import sys

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
    get_device,
    load_eurosat,
    seed_everything,
    stratified_split,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


@torch.no_grad()
def prepare_data_features(encoder, dataset, device, pool="mean", batch_size=256):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        h = encoder.forward_features(imgs, pool=pool)
        feats.append(h.cpu())
        labels.append(targets)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to an encoder-only MAE checkpoint (.pt).")
    ap.add_argument("--pool", default="mean", choices=["mean", "cls"])
    args = ap.parse_args()

    seed_everything(42)
    device = get_device()
    print(f"Device: {device} | checkpoint={args.checkpoint} | pool={args.pool}")

    encoder = build_vit_s8().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    eval_transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    full_dataset = load_eurosat(DATA_DIR, transform=eval_transform)
    train_idx, test_idx = stratified_split(full_dataset, test_size=0.2, seed=42)
    train_subset = data.Subset(full_dataset, train_idx)
    test_subset = data.Subset(full_dataset, test_idx)

    train_feats, train_labels = prepare_data_features(encoder, train_subset, device, pool=args.pool)
    test_feats, test_labels = prepare_data_features(encoder, test_subset, device, pool=args.pool)
    print(f"train {train_feats.shape}, test {test_feats.shape}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    acc = accuracy_score(test_labels, preds)
    print(f"\nlinear-probe test accuracy (pool={args.pool}): {acc:.4f}")


if __name__ == "__main__":
    main()
