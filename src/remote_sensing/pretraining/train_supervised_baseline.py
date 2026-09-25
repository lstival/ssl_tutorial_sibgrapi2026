"""
Offline run of the supervised from-scratch baseline from Notebook 4 (remote sensing).

Same recipe as `train_supervised_baseline()` in `04_comparative_evaluation.ipynb`: ViT-S/8 +
linear head trained end to end on the EuroSAT train split (stratified 80/20, seed=42), AdamW,
warmup -> cosine decay, batch 512, mixed precision, 15 epochs. Running it here and publishing
the weights next to the SSL checkpoints lets the notebook load the baseline instead of spending
~15-45 min training it live.

Writes, into --out-dir:
    supervised_vit_s8.pt       encoder state dict (same format as the SSL checkpoints, so
                               `try_load_checkpoint` loads it unchanged)
    supervised_vit_s8_head.pt  linear classification head, needed to reproduce the end-to-end
                               test accuracy without retraining
    supervised_vit_s8.json     recipe + test accuracy

Usage:
    python train_supervised_baseline.py
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_utils import REPO_ROOT, warmup_cosine_lambda  # noqa: E402

from tutorial_rs import (  # noqa: E402
    EUROSAT_CLASSES,
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMG_SIZE,
    build_eval_transform,
    build_rs_augmentations,
    build_vit_s8,
    get_device,
    load_eurosat,
    seed_everything,
    stratified_split,
)

ROOT = REPO_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "remote_sensing", "checkpoints"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = get_device()
    print("Device:", device)

    # Same data objects as Notebook 4: the split comes from the eval-transform dataset, the
    # training set is a second copy of EuroSAT with the RS augmentations, indexed by train_idx.
    eval_transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    full_dataset = load_eurosat(args.data_dir, transform=eval_transform)
    train_idx, test_idx = stratified_split(full_dataset, test_size=0.2, seed=42)
    test_subset = data.Subset(full_dataset, test_idx)

    aug_transform = build_rs_augmentations(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
    train_subset_aug = data.Subset(load_eurosat(args.data_dir, transform=aug_transform), train_idx)

    seed_everything(42)
    model = build_vit_s8().to(device)
    head = nn.Linear(model.embed_dim, len(EUROSAT_CLASSES)).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    nw = args.num_workers
    loader = data.DataLoader(train_subset_aug, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=nw,
                             pin_memory=use_amp, persistent_workers=nw > 0,
                             prefetch_factor=4 if nw > 0 else None)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_steps, warmup_steps))

    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss, n = 0.0, 0
        for imgs, labels in tqdm(loader, leave=False, desc=f"epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = head(model.forward_features(imgs))
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1
        print(f"epoch {epoch+1}/{args.epochs}  loss={epoch_loss / n:.4f}  "
              f"elapsed={(time.time() - start) / 60:.1f} min", flush=True)
    train_minutes = (time.time() - start) / 60

    model.eval()
    head.eval()
    test_loader = data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=nw, pin_memory=use_amp)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = head(model.forward_features(imgs)).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    acc = correct / total
    print(f"supervised: end-to-end test accuracy = {acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "supervised_vit_s8.pt"))
    torch.save(head.state_dict(), os.path.join(args.out_dir, "supervised_vit_s8_head.pt"))
    with open(os.path.join(args.out_dir, "supervised_vit_s8.json"), "w") as f:
        json.dump({
            "mechanism": "supervised",
            "arch": "vit_s8",
            "img_size": IMG_SIZE,
            "epochs": args.epochs,
            "steps": total_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs,
            "split": "EuroSAT stratified 80/20, seed=42",
            "test_accuracy": acc,
            "train_minutes": round(train_minutes, 1),
            "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
        }, f, indent=2)
    print(f"Saved supervised_vit_s8.pt / _head.pt / .json to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
