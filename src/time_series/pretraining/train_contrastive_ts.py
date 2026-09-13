"""
Contrastive (SimCLR / TS2Vec-style) pretraining of the tutorial's patch Transformer on the
pooled UCR archive.

Offline counterpart to notebooks/time_series/01_contrastive_simclr.ipynb: same InfoNCE mechanism and
the same `build_ts_encoder()` architecture from src/time_series/tutorial_ts.py, run for many more
steps on all 128 UCR datasets instead of a short live demo on SwedishLeaf alone. Only the
encoder weights are exported (the projection head is pretraining-only and is discarded),
matching what `try_load_checkpoint` in Notebooks 1-4 expects.

This script does NOT import the notebooks' teaching code path (no fill-in-the-blank
sentinels) -- the loss below is a fixed, complete reference implementation.

Usage:
    python train_contrastive_ts.py --data-path ../../../data \\
        --out ../../../artifacts/time_series/checkpoints/contrastive_ts_encoder.pt
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from pretrain_utils_ts import RunningLogger, save_encoder_checkpoint  # noqa: E402
from ucr_data import DEFAULT_PER_DATASET_CAP, build_pretraining_dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_ts import (  # noqa: E402
    ContrastiveTransformations,
    build_ts_augmentations,
    build_ts_encoder,
    get_device,
    seed_everything,
)


class SimCLRTSModel(nn.Module):
    """Encoder + small projection head, identical in spirit to Notebook 1's SimCLRTSModel."""

    def __init__(self, encoder, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder.forward_features(x, pool="cls")
        return self.projection_head(h)


def info_nce_loss(feats, temperature=0.1):
    """Reference InfoNCE -- the completed version of Notebook 1's fill-in-the-blank cell."""
    feats = F.normalize(feats.float(), dim=-1)
    cos_sim = feats @ feats.t()
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    loss = nll.mean()

    comb = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
    acc_top1 = (comb.argsort(dim=-1, descending=True).argmin(dim=-1) == 0).float().mean().item()
    return loss, acc_top1


def train(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    transform = ContrastiveTransformations(build_ts_augmentations(), n_views=2)
    exclude = (args.exclude_target,) if args.exclude_target else ()
    dataset, sizes = build_pretraining_dataset(
        args.data_path, transform, per_dataset_cap=args.per_dataset_cap, exclude=exclude)
    print(f"Contrastive pretraining pool: {len(dataset)} series from {len(sizes)} UCR datasets")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    model = SimCLRTSModel(build_ts_encoder()).to(device)
    n_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Encoder: {n_params / 1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, min(args.warmup_steps, args.steps - 1))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, args.steps - warmup)
        return (1.0 / 50) + (1.0 - 1.0 / 50) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = os.path.splitext(args.out)[0] + "_train_log.jsonl"
    logger = RunningLogger(log_path)
    print(f"Mixed precision: {'on' if use_amp else 'off'} | logging to {log_path}")

    model.train()
    data_iter = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            views = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            views = next(data_iter)
        imgs = torch.cat([v.to(device, non_blocking=True) for v in views], dim=0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = model(imgs)
        loss, acc_top1 = info_nce_loss(feats, temperature=args.temperature)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.log(step, loss=loss.item(), acc_top1=acc_top1, lr=scheduler.get_last_lr()[0])
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                model.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "contrastive",
                       "temperature": args.temperature, "batch_size": args.batch_size,
                       "corpus": "ucr", "n_datasets": len(sizes), "n_series": len(dataset),
                       "steps_total": args.steps},
            )
    logger.close()
    print("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=str, default="../../../data",
                   help="Directory holding (or to receive) the UCRArchive_2018 folder.")
    p.add_argument("--out", type=str, default="../../../artifacts/time_series/checkpoints/contrastive_ts_encoder.pt")
    p.add_argument("--per-dataset-cap", type=int, default=DEFAULT_PER_DATASET_CAP,
                   help="Max series contributed by any single UCR dataset.")
    p.add_argument("--exclude-target", type=str, default=None,
                   help="Drop this dataset from the corpus (e.g. SwedishLeaf) for the strict "
                        "cross-dataset transfer ablation.")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
