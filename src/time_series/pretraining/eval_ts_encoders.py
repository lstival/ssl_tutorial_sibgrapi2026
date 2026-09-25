"""
Headless re-run of notebooks/time_series/04_comparative_evaluation.ipynb's numbers.

Frozen-encoder linear-probe protocol, identical to the remote sensing part's
`src/remote_sensing/pretraining/eval_encoder.py`: extract features once from a frozen encoder,
fit a standardized multinomial logistic regression, report test accuracy on the UCR archive's
own TRAIN/TEST split. Keeping the two parts' protocols identical is what makes the
remote-sensing and time-series result tables directly comparable.

Reports, for each of the three SSL encoders plus a random-init floor and a supervised
end-to-end ceiling:
  - full-label linear-probe accuracy on the target dataset (default SwedishLeaf)
  - a few-label sweep (k labeled examples per class), where SSL is expected to help most
  - a cross-dataset generalization sweep: the same frozen encoders probed on several other
    UCR datasets they were never tuned for

Also reports a 1-NN Euclidean baseline, the classical reference point every UCR paper quotes.

Results are written to notebooks/time_series/figures/ts_eval_results.json.

The probe / baseline code here is a faithful copy of the notebook cells, kept in sync by hand
(as src/remote_sensing/pretraining/eval_encoder.py is for the remote sensing part).

Usage:
    python eval_ts_encoders.py
    python eval_ts_encoders.py --quick     # fewer transfer datasets, for a smoke test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_ts import (  # noqa: E402
    SERIES_LEN,
    TARGET_DATASET,
    UCRDataset,
    build_ts_encoder,
    extract_features,
    get_device,
    linear_probe_accuracy,
    seed_everything,
    subsample_per_class,
)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "..", "..", "data")
CHECKPOINT_PATH = os.path.join(HERE, "..", "..", "..", "artifacts", "time_series", "checkpoints")
OUT_JSON = os.path.join(HERE, "..", "..", "..", "notebooks", "time_series", "figures",
                        "ts_eval_results.json")

# Each mechanism's checkpoint, and the readout it should be probed with. MAE never supervises
# the [CLS] token -- its loss lives entirely on the patch tokens -- so a [CLS] probe badly
# understates it; mean-pooling the patch tokens is the standard masked-model readout. This is
# exactly the same asymmetry the remote sensing part documents for image MAE.
CHECKPOINT_FILES = {
    "contrastive": ("contrastive_ts_encoder.pt", "cls"),
    "mae": ("mae_ts_encoder.pt", "mean"),
    "dino": ("dino_ts_encoder.pt", "cls"),
}

# Labeled examples per class for the few-label sweep. SwedishLeaf has ~33 train series per
# class, so 20 is close to the full-label regime and 1 is the extreme.
LABEL_BUDGETS = [1, 2, 5, 10, 20]

# Other UCR datasets used for the cross-dataset transfer check. These are never used to tune
# anything; they are probed with the same frozen encoders to show the representation is
# general rather than SwedishLeaf-specific. Chosen to span domains (image outline, sensor,
# spectro, motion, ECG) and to be small enough that the sweep is quick.
TRANSFER_DATASETS = [
    "FaceAll", "ECG5000", "Trace", "Plane", "SyntheticControl",
    "GunPoint", "OSULeaf", "Fish",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer transfer datasets / epochs, for a smoke test")
    ap.add_argument("--target", default=TARGET_DATASET)
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--out-json", default=OUT_JSON)
    return ap.parse_args()


def load_splits(data_path, name):
    return UCRDataset(data_path, name, "train"), UCRDataset(data_path, name, "test")


def frozen(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def probe(encoder, splits, device, pool="cls"):
    """Full-label linear probe on one (train, test) pair; also returns the features."""
    train_ds, test_ds = splits
    trf, trl = extract_features(encoder, train_ds, device, pool=pool)
    tef, tel = extract_features(encoder, test_ds, device, pool=pool)
    return linear_probe_accuracy(trf, trl, tef, tel), (trf, trl, tef, tel)


def few_label_sweep(feats):
    """Few-label probe accuracies from already-extracted features."""
    trf, trl, tef, tel = feats
    out = {}
    for k in LABEL_BUDGETS:
        sub = subsample_per_class(trl, k)
        out[str(k)] = linear_probe_accuracy(trf[sub], trl[sub], tef, tel)
    return out


def nn_euclidean_baseline(train_ds, test_ds, device):
    """
    1-NN with Euclidean distance on the raw z-normalized series -- the classical UCR baseline
    that every paper on the archive reports, and a genuinely strong one at this scale.
    """
    Xtr = torch.from_numpy(train_ds.X).to(device)
    Xte = torch.from_numpy(test_ds.X).to(device)
    preds = []
    for i in range(0, len(Xte), 256):
        preds.append(torch.cdist(Xte[i:i + 256], Xtr).argmin(dim=1).cpu().numpy())
    return float((train_ds.y[np.concatenate(preds)] == test_ds.y).mean())


def supervised_end_to_end(train_ds, test_ds, device, epochs, lr=1e-3, weight_decay=1e-4,
                          batch_size=64):
    """
    Train the same encoder architecture from random init, end to end with labels -- the
    from-scratch ceiling for this dataset. With only 500 labeled series a Transformer overfits
    quickly, which is precisely the point the SSL comparison is making.
    """
    seed_everything(42)
    enc = build_ts_encoder().to(device)
    head = nn.Linear(enc.embed_dim, train_ds.n_classes).to(device)
    loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(loader)))
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        enc.train(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(head(enc.forward_features(xb, pool="cls")), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    enc.eval(); head.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in data.DataLoader(test_ds, batch_size=256, shuffle=False):
            pred = head(enc.forward_features(xb.to(device), pool="cls")).argmax(1).cpu()
            correct += (pred == yb).sum().item()
    return correct / len(test_ds)


def main():
    args = parse_args()
    sup_epochs = 15 if args.quick else 60
    transfer_names = TRANSFER_DATASETS[:2] if args.quick else TRANSFER_DATASETS

    device = get_device()
    seed_everything(42)

    target = load_splits(args.data_path, args.target)
    train_ds, test_ds = target
    print(f"{args.target}: train {len(train_ds)}, test {len(test_ds)}, "
          f"{train_ds.n_classes} classes, length {SERIES_LEN} | device {device}")
    # Built once and shared by every encoder (each build re-resamples / re-normalizes).
    transfer = {name: load_splits(args.data_path, name) for name in transfer_names}

    results = {
        "target": args.target,
        "series_len": SERIES_LEN,
        "n_classes": train_ds.n_classes,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "label_budgets": LABEL_BUDGETS,
        "full_label": {},
        "few_label": {},
        "transfer": {},
    }

    # --- the three SSL encoders ---
    for family, (fname, pool) in CHECKPOINT_FILES.items():
        path = os.path.join(CHECKPOINT_PATH, fname)
        if not os.path.isfile(path):
            print(f"skip {family}: {path} missing")
            continue
        enc = build_ts_encoder().to(device)
        enc.load_state_dict(torch.load(path, map_location=device))
        frozen(enc)

        meta_path = os.path.splitext(path)[0] + ".json"
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

        # Measure BOTH readouts for every family. The expectation (the remote sensing part,
        # He et al. 2022) is that MAE reads out better from mean-pooled patch tokens and
        # cls-trained families from [CLS] -- a claim to verify per run, not to assume. The
        # family's own readout is then reused below instead of being probed a third time.
        by_readout = {pl: probe(enc, target, device, pool=pl) for pl in ("cls", "mean")}
        both = {pl: res[0] for pl, res in by_readout.items()}
        results.setdefault("readout", {})[family] = both

        acc, feats = by_readout[pool]
        results["full_label"][family] = {"accuracy": acc, "pool": pool,
                                         "by_readout": both, "checkpoint_meta": meta}
        results["few_label"][family] = few_label_sweep(feats)
        print(f"{family:>12s}: full-label probe {acc:.4f} (pool={pool}) "
              f"[cls {both['cls']:.4f} | mean {both['mean']:.4f}]")
        for k, v in results["few_label"][family].items():
            print(f"{'':>12s}  k={k:>3s}/class: {v:.4f}")

        results["transfer"][family] = {}
        for ds_name, splits in transfer.items():
            t_acc, _ = probe(enc, splits, device, pool=pool)
            results["transfer"][family][ds_name] = t_acc
            print(f"{'':>12s}  transfer {ds_name:<18s} {t_acc:.4f}")

    # --- random-init floor: the same architecture, untrained ---
    seed_everything(123)
    rand_enc = frozen(build_ts_encoder().to(device))
    r_acc, r_feats = probe(rand_enc, target, device, pool="cls")
    results["random_init"] = {"accuracy": r_acc, "few_label": few_label_sweep(r_feats)}
    results["transfer"]["random_init"] = {
        ds_name: probe(rand_enc, splits, device, pool="cls")[0]
        for ds_name, splits in transfer.items()
    }
    print(f"{'random-init':>12s}: full-label probe {r_acc:.4f}")

    # --- classical + supervised reference points ---
    results["nn_euclidean"] = nn_euclidean_baseline(train_ds, test_ds, device)
    print(f"{'1-NN euclid':>12s}: {results['nn_euclidean']:.4f}")

    results["supervised"] = supervised_end_to_end(train_ds, test_ds, device, sup_epochs)
    print(f"{'supervised':>12s}: {results['supervised']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.abspath(args.out_json)}")


if __name__ == "__main__":
    main()
