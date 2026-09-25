"""
Frozen probe vs BitFit vs full fine-tuning, on one encoder checkpoint.

The question this answers for the tutorial: once you have a pretrained encoder, how much of
the achievable accuracy does a *frozen* readout leave on the table, and how cheaply can you
recover it?

Three adaptation regimes, all sharing the Notebook 4 protocol (stratified 80/20 EuroSAT split
at seed=42, per-class subsampling at seed=42) so every number is directly comparable to
`eval_encoder.py` and to `probe_results.json`:

  frozen  -- encoder fully frozen; train a linear head only. This is the linear probe, but
             trained with SGD here rather than scikit-learn's LogisticRegression so that all
             three arms share one optimizer, schedule and augmentation pipeline. (The
             scikit-learn probe in eval_encoder.py is the reference number; this arm should
             land close to it, and a large gap means the head recipe, not the features, is
             the limiting factor.)
  bitfit  -- train the bias terms of the encoder plus the head, freezing every weight matrix.
             On ViT-S/8 that is 26,112 of 10,747,008 encoder parameters (0.24%). From
             Ben-Zaken et al. (2022), "BitFit: Simple Parameter-efficient Fine-tuning for
             Transformer-based Masked Language-models" -- the finding being that biases alone
             recover most of the gap to full fine-tuning at a tiny fraction of the trainable
             parameters and optimizer state.
  full    -- every encoder parameter trainable, plus the head. The upper bound on what
             adaptation can buy, and the most expensive.

Augmentation is on for every arm (a frozen encoder still benefits: the head sees more varied
features), and each arm gets its own learning rate, because the right LR differs by roughly an
order of magnitude between them -- a head on frozen features wants a large LR, a full
fine-tune of a pretrained encoder wants a small one or it washes out the very features it was
given. Using one LR for all three is the classic way to make fine-tuning look bad.

Usage:
    python eval_finetune.py --checkpoint ../../../artifacts/remote_sensing/checkpoints/dino_init_vit_s8.pt \\
        --name imagenet_dino_init --out-json ../../../notebooks/remote_sensing/figures/finetune_results.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eurosat_probe import subsample_per_class  # noqa: E402
from pretrain_utils import ARCHS, DATA_DIR, warmup_cosine_lambda  # noqa: E402

from tutorial_rs import (  # noqa: E402
    EUROSAT_CLASSES,
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMG_SIZE,
    build_eval_transform,
    build_rs_augmentations,
    get_device,
    load_eurosat,
    seed_everything,
    stratified_split,
)

# Per-regime learning rates. See the module docstring: these differ by design, not by accident.
DEFAULT_LRS = {"frozen": 1e-3, "bitfit": 5e-4, "full": 1e-4}


class ClassifierWrapper(nn.Module):
    def __init__(self, encoder, n_classes, pool="cls"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = nn.Linear(encoder.embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder.forward_features(x, pool=self.pool))


def configure_regime(model, regime):
    """
    Set requires_grad according to the adaptation regime and return the trainable parameters.

    The head is always trainable -- it is randomly initialized in every regime, so freezing it
    would measure nothing.
    """
    for p in model.encoder.parameters():
        p.requires_grad = False

    if regime == "bitfit":
        # Every bias in the encoder: patch-embed projection, both LayerNorms per block, the
        # fused attention in_proj_bias, attention out_proj, both MLP layers, and the final
        # norm. LayerNorm *weights* stay frozen -- BitFit is biases only, and letting the norm
        # scales train as well is a different (larger) method.
        for name, p in model.encoder.named_parameters():
            if name.endswith(".bias") or name.endswith("_bias"):
                p.requires_grad = True
    elif regime == "full":
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif regime != "frozen":
        raise ValueError(f"unknown regime {regime!r}")

    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model):
    enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    enc_total = sum(p.numel() for p in model.encoder.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return enc_train, enc_total, head


def evaluate(model, loader, device, use_amp):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = model(imgs).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def run_regime(regime, ckpt_state, arch, train_subset, test_loader, device, args, lr):
    seed_everything(args.seed)
    encoder = ARCHS[arch]()
    encoder.load_state_dict(ckpt_state)
    model = ClassifierWrapper(encoder, len(EUROSAT_CLASSES), pool=args.pool).to(device)

    params = configure_regime(model, regime)
    enc_train, enc_total, head = count_trainable(model)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                             drop_last=len(train_subset) > args.batch_size,
                             num_workers=0, pin_memory=use_amp)
    steps_per_epoch = max(1, len(loader))
    # Scale epochs up on small subsets so every budget trains for a comparable number of steps
    # (see --min-steps). Without this the comparison across label budgets is not a comparison of
    # adaptation regimes but of how far each one gets in ~15 steps.
    epochs = max(args.epochs, math.ceil(args.min_steps / steps_per_epoch))
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(steps_per_epoch * args.warmup_epochs, max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_steps, warmup_steps))

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        # The encoder stays in train() only where it is being adapted. For the frozen arm this
        # keeps its (nonexistent) dropout and, more importantly, makes the arm a true frozen
        # feature extractor rather than one whose forward pass changes between epochs.
        if regime == "frozen":
            model.encoder.eval()
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(imgs), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    train_sec = time.perf_counter() - t0

    acc = evaluate(model, test_loader, device, use_amp)
    return {
        "accuracy": acc,
        "lr": lr,
        "trainable_encoder_params": enc_train,
        "encoder_params": enc_total,
        "head_params": head,
        "trainable_fraction": enc_train / enc_total,
        "train_seconds": round(train_sec, 1),
        "epochs": epochs,
        "total_steps": total_steps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Encoder-only checkpoint (.pt).")
    ap.add_argument("--name", required=True, help="Key this run is stored under in --out-json.")
    ap.add_argument("--arch", default="vit_s8", choices=sorted(ARCHS))
    ap.add_argument("--pool", default="cls", choices=["mean", "cls"],
                    help="Feature pooling. Use 'mean' for MAE checkpoints (no trained CLS token).")
    ap.add_argument("--regimes", default="frozen,bitfit,full",
                    help="Comma-separated subset of frozen,bitfit,full.")
    ap.add_argument("--label-budgets", default="",
                    help="Comma-separated k values (labels/class) to run in addition to the full "
                         "train split. Empty = full split only.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--min-steps", type=int, default=300,
                    help="Floor on total gradient steps per run. Small label budgets otherwise get "
                         "a handful of steps (k=10 at batch 256 is 1 step/epoch, so 15 epochs is 15 "
                         "steps) and the randomly-initialized head never converges -- which reads as "
                         "a near-chance accuracy for the frozen and BitFit arms and silently "
                         "flatters full fine-tuning, whose 10.7M parameters move further per step.")
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-frozen", type=float, default=DEFAULT_LRS["frozen"])
    ap.add_argument("--lr-bitfit", type=float, default=DEFAULT_LRS["bitfit"])
    ap.add_argument("--lr-full", type=float, default=DEFAULT_LRS["full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=None, help="JSON file to append results to.")
    args = ap.parse_args()

    device = get_device()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    lrs = {"frozen": args.lr_frozen, "bitfit": args.lr_bitfit, "full": args.lr_full}
    budgets = [int(k) for k in args.label_budgets.split(",") if k.strip()]

    print(f"Device: {device} | checkpoint={args.checkpoint} | arch={args.arch} | pool={args.pool}")
    ckpt_state = torch.load(args.checkpoint, map_location="cpu")

    seed_everything(args.seed)
    aug_transform = build_rs_augmentations(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    eval_transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    train_full = load_eurosat(DATA_DIR, transform=aug_transform)
    test_full = load_eurosat(DATA_DIR, transform=eval_transform)
    train_idx, test_idx = stratified_split(train_full)

    test_loader = data.DataLoader(data.Subset(test_full, test_idx), batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  pin_memory=device.type == "cuda")

    train_labels = np.array(train_full.targets)[train_idx]
    results = {"checkpoint": os.path.basename(args.checkpoint), "arch": args.arch,
               "pool": args.pool, "epochs": args.epochs, "full_label": {}, "few_label": {}}

    for regime in regimes:
        subset = data.Subset(train_full, train_idx)
        res = run_regime(regime, ckpt_state, args.arch, subset, test_loader, device, args,
                         lrs[regime])
        results["full_label"][regime] = res
        print(f"[full-label] {regime:6s} acc={res['accuracy']:.4f}  "
              f"trainable={res['trainable_encoder_params']:,}/{res['encoder_params']:,} "
              f"({100 * res['trainable_fraction']:.2f}%)  {res['train_seconds']:.0f}s")

    for k in budgets:
        sub = subsample_per_class(train_labels, k, seed=args.seed)
        subset = data.Subset(train_full, train_idx[sub])
        results["few_label"][str(k)] = {}
        for regime in regimes:
            res = run_regime(regime, ckpt_state, args.arch, subset, test_loader, device, args,
                             lrs[regime])
            results["few_label"][str(k)][regime] = res
            print(f"[k={k:4d}]     {regime:6s} acc={res['accuracy']:.4f}  {res['train_seconds']:.0f}s")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        store = {}
        if os.path.exists(args.out_json):
            with open(args.out_json, encoding="utf-8") as f:
                store = json.load(f)
        store[args.name] = results
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        print(f"\nWrote results for '{args.name}' to {args.out_json}")


if __name__ == "__main__":
    main()
