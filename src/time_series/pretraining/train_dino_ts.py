"""
Self-distillation (DINO-style) pretraining of the tutorial's patch Transformer on the pooled
UCR archive.

Offline counterpart to notebooks/time_series/03_distillation_dino.ipynb: same multi-crop student/teacher
EMA mechanism and the same `build_ts_encoder()` architecture. Only the *teacher* encoder
weights are exported (by convention the EMA teacher is the more stable representation; the
DINO head is pretraining-only).

The EMA update and DINO loss below are fixed, complete reference implementations, structurally
identical to the ones filled in inside Notebook 3.

Usage:
    python train_dino_ts.py --data-path ../../../data \\
        --out ../../../artifacts/time_series/checkpoints/dino_ts_encoder.pt
"""

import argparse
import math
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from pretrain_utils_ts import (  # noqa: E402
    LOG_PRINT_EVERY,
    RunningLogger,
    enable_fast_cuda,
    save_encoder_checkpoint,
    train_log_path,
)
from ucr_data import (  # noqa: E402
    DEFAULT_PER_DATASET_CAP,
    build_pretraining_dataset,
    multicrop_collate,
)

from tutorial_ts import (  # noqa: E402
    MultiCropTransform,
    build_ts_augmentations,
    build_ts_encoder,
    build_ts_local_augmentations,
    get_device,
    seed_everything,
)


class DINOHead(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class DINOTSModel(nn.Module):
    """Encoder + DINO head, identical in spirit to Notebook 3's DINOTSModel."""

    def __init__(self, encoder, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = DINOHead(encoder.embed_dim, hidden_dim, out_dim)

    def forward(self, x):
        return self.head(self.encoder.forward_features(x, pool="cls"))


@torch.no_grad()
def ema_update(student, teacher, m):
    """Reference implementation -- the completed version of Notebook 3's fill-in-the-blank cell."""
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=1 - m)


def dino_loss(student_logits_list, teacher_logits_list, center, tau_student=0.1, tau_teacher=0.04):
    """Reference implementation -- the completed version of Notebook 3's fill-in-the-blank cell."""
    student_log_probs = [F.log_softmax(s / tau_student, dim=-1) for s in student_logits_list]
    teacher_probs = [F.softmax((t - center) / tau_teacher, dim=-1).detach() for t in teacher_logits_list]
    total_loss, n_terms = 0.0, 0
    for t_idx, t_probs in enumerate(teacher_probs):
        for s_idx, s_log_probs in enumerate(student_log_probs):
            if s_idx == t_idx:
                continue
            total_loss += -(t_probs * s_log_probs).sum(dim=-1).mean()
            n_terms += 1
    return total_loss / n_terms


@torch.no_grad()
def update_center(center, teacher_logits_list, momentum=0.9):
    batch_center = torch.cat(teacher_logits_list, dim=0).mean(dim=0, keepdim=True)
    return center * momentum + batch_center * (1 - momentum)


def train(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    enable_fast_cuda(device)

    transform = MultiCropTransform(build_ts_augmentations(), build_ts_local_augmentations(),
                                   n_global=args.n_global, n_local=args.n_local)
    exclude = (args.exclude_target,) if args.exclude_target else ()
    dataset, sizes = build_pretraining_dataset(
        args.data_path, transform, per_dataset_cap=args.per_dataset_cap, exclude=exclude)
    print(f"DINO pretraining pool: {len(dataset)} series from {len(sizes)} UCR datasets "
          f"({args.n_global} global + {args.n_local} local crops)")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=multicrop_collate)

    student = DINOTSModel(build_ts_encoder(), out_dim=args.out_dim).to(device)
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr / 50)
    center = torch.zeros(1, args.out_dim, device=device)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = train_log_path(args.out)
    logger = RunningLogger(log_path, print_every=LOG_PRINT_EVERY)
    print(f"Mixed precision: {'on' if use_amp else 'off'} | logging to {log_path}")

    student.train()
    teacher.eval()
    data_iter = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            crops = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            crops = next(data_iter)
        crops = [c.to(device, non_blocking=True) for c in crops]

        with torch.cuda.amp.autocast(enabled=use_amp):
            student_logits = [student(c) for c in crops]
            with torch.no_grad():
                teacher_logits = [teacher(c) for c in crops[: args.n_global]]
        student_logits = [s.float() for s in student_logits]
        teacher_logits = [t.float() for t in teacher_logits]

        loss = dino_loss(student_logits, teacher_logits, center, args.tau_student, args.tau_teacher)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        m = args.ema_momentum_end - (args.ema_momentum_end - args.ema_momentum_start) * (
            1 + math.cos(math.pi * step / args.steps)
        ) / 2
        ema_update(student, teacher, m=m)
        center = update_center(center, teacher_logits, momentum=args.center_momentum)

        with torch.no_grad():
            tp = F.softmax((teacher_logits[0] - center) / args.tau_teacher, dim=-1)
            entropy = -(tp * tp.clamp_min(1e-8).log()).sum(dim=-1).mean().item()

        logger.log(step, loss=loss.item(), entropy=entropy, ema_m=m, lr=scheduler.get_last_lr()[0])
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                teacher.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "dino", "entropy": entropy,
                       "corpus": "ucr", "n_datasets": len(sizes), "n_series": len(dataset),
                       "steps_total": args.steps},
            )
    logger.close()
    print("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=str, default="../../../data",
                   help="Directory holding (or to receive) the UCRArchive_2018 folder.")
    p.add_argument("--out", type=str, default="../../../artifacts/time_series/checkpoints/dino_ts_encoder.pt")
    p.add_argument("--per-dataset-cap", type=int, default=DEFAULT_PER_DATASET_CAP,
                   help="Max series contributed by any single UCR dataset.")
    p.add_argument("--exclude-target", type=str, default=None,
                   help="Drop this dataset from the corpus (e.g. SwedishLeaf) for the strict "
                        "cross-dataset transfer ablation.")
    p.add_argument("--n-global", type=int, default=2)
    p.add_argument("--n-local", type=int, default=4)
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tau-student", type=float, default=0.1)
    p.add_argument("--tau-teacher", type=float, default=0.04)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--ema-momentum-start", type=float, default=0.996)
    p.add_argument("--ema-momentum-end", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
