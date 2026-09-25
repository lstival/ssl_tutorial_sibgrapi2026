"""
Linear-probe evaluation of one encoder-only checkpoint, in BOTH label regimes, using exactly
the Notebook 4 protocol (stratified 80/20 EuroSAT split at seed=42, StandardScaler +
multinomial LogisticRegression, per-class subsampling at seed=42).

Written for the SeCo corpus-scaling study: each pretraining run produces one checkpoint, and
this script appends that checkpoint's full-label and few-label accuracies to a shared results
JSON so the scaling curve can be plotted from a single file.

It reports the few-label curve too -- which is where a larger pretraining corpus is expected
to show up most clearly, since the full-label probe has only ~3 points of headroom below the
supervised from-scratch ceiling (0.944). For a quick full-label-only check of an MAE
checkpoint, run it with `--pool mean` and no `--out-json`.

Usage:
    python eval_encoder.py --checkpoint ../../../artifacts/remote_sensing/checkpoints/contrastive_seco_f100.pt \\
        --name f100 --pool cls --out-json ../../../notebooks/remote_sensing/figures/seco_scaling_results.json
"""
import argparse
import json
import os
import sys

import torch
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eurosat_probe import (  # noqa: E402
    eurosat_eval_splits,
    extract_features,
    linear_probe_accuracy,
    subsample_per_class,
)
from pretrain_utils import ARCHS  # noqa: E402

from tutorial_rs import get_device, seed_everything  # noqa: E402

LABEL_BUDGETS = [10, 20, 50, 100, 200]


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

    full_dataset, train_idx, test_idx = eurosat_eval_splits()
    tr_f, tr_y = extract_features(encoder, data.Subset(full_dataset, train_idx), device, args.pool)
    te_f, te_y = extract_features(encoder, data.Subset(full_dataset, test_idx), device, args.pool)
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
