"""
Large-scale self-distillation (DINO-style) pretraining of the tutorial's ViT-S/8 encoder on SeCo.

Offline counterpart to notebooks/remote_sensing/03_distillation_dino.ipynb: same multi-crop student/teacher
EMA mechanism and the same `build_vit_s8()` architecture from src/remote_sensing/tutorial_rs.py, run for
many more steps on the SeCo corpus instead of a ~100-step live demo on EuroSAT. Only the
*teacher* encoder weights are exported at the end (by convention -- the EMA teacher is the
higher-quality, more stable representation; the DINO head is pretraining-only and discarded).

This script does NOT import the notebooks' teaching code path (no fill-in-the-blank
sentinels) -- the EMA update and DINO loss below are fixed, complete reference
implementations, structurally identical to the ones filled in inside Notebook 3.

Usage:
    python train_dino.py --seco-root ../../../data/seco/seasonal_contrast_100k \\
        --manifest ../../../data/seco/manifest.txt --out ../../../artifacts/remote_sensing/checkpoints/dino_vit_s8.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from gpu_aug import (  # noqa: E402
    LocationShuffler,
    gpu_augment,
)
from pretrain_utils import (  # noqa: E402
    RunningLogger,
    _validate_corpus_args,
    add_dataset_args,
    build_patch_cache,
    dataset_norm,
    load_train_state,
    resolve_locations,
    save_encoder_checkpoint,
    save_train_state,
)
from seco_data import (  # noqa: E402
    SECO_MEAN,
    SECO_STD,
    SeCoAugmentedDataset,
    build_seco_augmentations,
)

from tutorial_rs import IMG_SIZE, build_vit_s8, get_device, seed_everything  # noqa: E402


class DINOHead(nn.Module):
    """
    3-layer MLP + a weight-normalized last layer, matching Caron et al. (2021). The last layer
    is a weight_norm Linear with its magnitude (`weight_g`) frozen at 1: this bounds the logit
    scale so the sharpened teacher softmax (tau ~= 0.04) cannot run away to a one-hot
    distribution early in training -- the mechanism that, together with `freeze_last_layer`,
    prevents the well-known DINO collapse. An earlier version of this file used a bare
    `nn.Linear` here and the SeCo run collapsed (teacher entropy fell to ~0.18 over 2048
    dims and stayed there; EuroSAT linear probe only +6 pts over a random encoder).
    """

    def __init__(self, embed_dim=384, hidden_dim=512, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)

    def cancel_last_layer_grad(self):
        """Zero the last-layer gradient -- called during the freeze_last_layer warmup epochs."""
        for p in self.last_layer.parameters():
            if p.grad is not None:
                p.grad = None


class DINOModel(nn.Module):
    """Encoder + DINO head, identical in spirit to Notebook 3's DINOModel."""

    def __init__(self, encoder, embed_dim=384, hidden_dim=512, out_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = DINOHead(embed_dim, hidden_dim, out_dim)

    def forward(self, x):
        h = self.encoder.forward_features(x)
        return self.head(h)


@torch.no_grad()
def ema_update(student, teacher, m):
    """Reference implementation -- the completed version of Notebook 3's fill-in-the-blank cell."""
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(m).add_(student_param.data, alpha=1 - m)


def dino_loss(student_logits_list, teacher_logits_list, center, tau_student=0.1, tau_teacher=0.04):
    """Reference implementation -- the completed version of Notebook 3's fill-in-the-blank cell."""
    student_log_probs = [F.log_softmax(s / tau_student, dim=-1) for s in student_logits_list]
    teacher_probs = [F.softmax((t - center) / tau_teacher, dim=-1).detach() for t in teacher_logits_list]

    total_loss = 0.0
    n_terms = 0
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


class MultiCropTransform:
    def __init__(self, global_transform, local_transform, n_global=2, n_local=4):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(self.n_global)]
        crops += [self.local_transform(x) for _ in range(self.n_local)]
        return crops


def build_gpu_source(args, device):
    """
    GPU multi-crop source: the decoded corpus as one uint8 tensor plus a shuffled location
    stream, replacing the DataLoader. DINO is the pipeline's worst case on the CPU path --
    n_global + n_local crops are augmented per sample per step, all single-threaded Python --
    so it gains the most here. See gpu_aug.py.
    """
    cache, cache_device = build_patch_cache(args, device)

    sample_gen = torch.Generator(device=cache_device)
    sample_gen.manual_seed(args.seed)
    aug_gen = torch.Generator(device=device)
    aug_gen.manual_seed(args.seed + 1)

    shuffler = LocationShuffler(len(cache), args.batch_size, cache_device, generator=sample_gen)
    mean, std = dataset_norm(args, device)
    return cache, shuffler, sample_gen, aug_gen, mean, std


def gpu_multicrop(raw_batch, args, mean, std, generator):
    """
    GPU equivalent of MultiCropTransform: `n_global` global views (scale 0.5-1.0, with jitter,
    matching build_seco_augmentations) followed by `n_local` local views (scale 0.2-0.5, no
    jitter and no rotation, matching the local_transform in build_dataloader).
    """
    crops = [gpu_augment(raw_batch, IMG_SIZE, mean, std, generator=generator,
                         scale=(0.5, 1.0))
             for _ in range(args.n_global)]
    crops += [gpu_augment(raw_batch, IMG_SIZE, mean, std, generator=generator,
                          scale=(0.2, 0.5), rot_p=0.0, jitter_p=0.0)
              for _ in range(args.n_local)]
    return crops


def build_dataloader(args):
    global_transform = build_seco_augmentations(img_size=IMG_SIZE)
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.2, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(SECO_MEAN, SECO_STD),
    ])
    multicrop = MultiCropTransform(global_transform, local_transform, n_global=args.n_global, n_local=args.n_local)

    locations = resolve_locations(args.seco_root, args.manifest)
    preload = not args.no_preload
    if preload:
        print("Preloading all SeCo patches into RAM (one-time decode)...")
    dataset = SeCoAugmentedDataset(args.seco_root, multicrop, n_views=1, locations=locations,
                                   preload=preload)
    # SeCoAugmentedDataset with n_views=1 calls `multicrop(img)` once per sample, which itself
    # returns a list of n_global+n_local crops -- unwrap that single-element outer list below.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=_multicrop_collate,
    )
    return dataset, loader


def _multicrop_collate(batch):
    """
    batch: list of (views, label) where views == [crops] (a single-element outer list from
    SeCoAugmentedDataset(n_views=1), and crops is itself a list of n_global+n_local tensors).
    Returns a list of length n_global+n_local, each a (B, C, H, W) batch tensor.
    """
    crops_per_sample = [sample_views[0] for sample_views, _ in batch]  # list of B lists of crops
    n_crops = len(crops_per_sample[0])
    return [torch.stack([crops[i] for crops in crops_per_sample], dim=0) for i in range(n_crops)]


def train(args):
    seed_everything(args.seed)
    _validate_corpus_args(args)
    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_gpu_aug = not args.no_gpu_aug and device.type == "cuda"
    cache = loader = None
    if use_gpu_aug:
        cache, shuffler, sample_gen, aug_gen, mean, std = build_gpu_source(args, device)
        n_locations = len(cache)
    else:
        dataset, loader = build_dataloader(args)
        n_locations = len(dataset)
    print(f"DINO pretraining pool: {n_locations} locations "
          f"({args.n_global} global + {args.n_local} local crops, "
          f"{'GPU' if use_gpu_aug else 'CPU'} augmentation)")

    encoder = build_vit_s8()
    if args.init_encoder:
        encoder.load_state_dict(torch.load(args.init_encoder, map_location="cpu"))
        print(f"Initialized student/teacher encoder weights from {args.init_encoder}")
    student = DINOModel(encoder, out_dim=args.out_dim).to(device)
    # Build the teacher fresh and copy the student's weights rather than deepcopy(student):
    # the weight-normalized last layer of DINOHead is not deepcopy-safe under this torch
    # (pytorch#103001). load_state_dict handles the weight_norm buffers correctly.
    teacher = DINOModel(build_vit_s8(), out_dim=args.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr / 50)
    center = torch.zeros(1, args.out_dim, device=device)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger = RunningLogger(os.path.join(os.path.dirname(args.out), "dino_train_log.jsonl"))

    # Full-state resume checkpoint. DINO has extra state beyond the student (the EMA teacher and
    # the center vector), which we stash in `extra` and restore by hand below.
    state_path = os.path.splitext(args.out)[0] + ".train.pt"
    start_step = 0
    if args.resume:
        import os.path as _osp
        if _osp.isfile(state_path):
            _state = torch.load(state_path, map_location=device)
            teacher.load_state_dict(_state["extra"]["teacher"])
            center = _state["extra"]["center"].to(device)
        start_step = load_train_state(state_path, student, optimizer, scheduler, device)
        if start_step >= args.steps:
            print(f"Already at step {start_step} >= target {args.steps}; nothing to do.")
            logger.close()
            return

    student.train()
    teacher.eval()
    data_iter = iter(loader) if loader is not None else None
    for step in range(start_step + 1, args.steps + 1):
        if use_gpu_aug:
            loc_idx = shuffler.next_batch()
            raw = cache.sample_views(loc_idx, n_views=1, generator=sample_gen)
            crops = gpu_multicrop(raw[0].to(device, non_blocking=True), args, mean, std, aug_gen)
        else:
            try:
                crops = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                crops = next(data_iter)
            crops = [c.to(device, non_blocking=True) for c in crops]

        # Teacher-temperature warmup: DINO linearly ramps tau_teacher from a mild value up to
        # its target over the first `tau_teacher_warmup_steps`. Starting at the sharp target
        # value from step 1 (as an earlier version did) sharpens the teacher before the center
        # has stabilized, which is what drove the SeCo collapse.
        if step <= args.tau_teacher_warmup_steps:
            frac = step / max(1, args.tau_teacher_warmup_steps)
            tau_teacher = args.tau_teacher_warmup + frac * (args.tau_teacher - args.tau_teacher_warmup)
        else:
            tau_teacher = args.tau_teacher

        with torch.cuda.amp.autocast(enabled=use_amp):
            student_logits = [student(c) for c in crops]
            with torch.no_grad():
                teacher_logits = [teacher(c) for c in crops[: args.n_global]]
        # Loss in fp32: the sharpened softmaxes (tau ~0.04) are numerically sensitive, so we
        # upcast the logits before the DINO cross-entropy and the center/entropy statistics.
        student_logits = [s.float() for s in student_logits]
        teacher_logits = [t.float() for t in teacher_logits]

        loss = dino_loss(student_logits, teacher_logits, center, args.tau_student, tau_teacher)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Gradient clipping (DINO uses clip_grad ~= 3.0): a ViT student is unstable in the first
        # few hundred steps without it, and an early spike can tip the head into collapse.
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        # freeze_last_layer: hold the head's last (weight-normalized) layer fixed for the first
        # `freeze_last_layer_steps` so the rest of the network organizes before the prototype
        # directions move. Standard DINO anti-collapse measure.
        if step <= args.freeze_last_layer_steps:
            student.head.cancel_last_layer_grad()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA momentum ramps from ema_momentum_start toward ema_momentum_end over training
        # (slower-moving teacher later). Capped strictly below 1.0: at exactly m=1.0 the teacher
        # stops updating entirely, so the last thousands of steps train the student against a
        # frozen target and the exported (teacher) checkpoint is stale.
        m = args.ema_momentum_end - (args.ema_momentum_end - args.ema_momentum_start) * (
            1 + torch.cos(torch.tensor(step / args.steps * 3.14159265))
        ).item() / 2
        m = min(m, args.ema_momentum_max)
        ema_update(student, teacher, m=m)
        center = update_center(center, teacher_logits, momentum=args.center_momentum)

        with torch.no_grad():
            teacher_probs = F.softmax((teacher_logits[0] - center) / tau_teacher, dim=-1)
            entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
            # KL of the teacher's mean prediction from uniform -- a direct collapse gauge,
            # independent of tau. Near 0 = healthy spread; near log(out_dim) = collapsed.
            mean_prob = teacher_probs.mean(dim=0)
            kl_unif = (mean_prob * (mean_prob.clamp_min(1e-8) * args.out_dim).log()).sum().item()

        logger.log(step, loss=loss.item(), entropy=entropy, kl_unif=kl_unif, ema_m=m,
                   tau_t=tau_teacher, lr=scheduler.get_last_lr()[0])

        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                teacher.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "dino", "entropy": entropy},
            )
            save_train_state(
                state_path, step=step, model=student, optimizer=optimizer, scheduler=scheduler,
                extra={"mechanism": "dino", "teacher": teacher.state_dict(), "center": center.cpu()},
            )

    logger.close()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seco-root", type=str, default=None,
                         help="Path to the extracted seasonal_contrast_100k directory.")
    parser.add_argument("--manifest", type=str, default=None,
                         help="Optional manifest.txt (from download_seco.py) restricting which locations to use.")
    parser.add_argument("--out", type=str, default="../../../artifacts/remote_sensing/checkpoints/dino_vit_s8.pt",
                         help="Output path for the (teacher) encoder-only checkpoint.")
    # Defaults below are the Optuna-selected recipe from optuna_dino.py (see optuna_best_dino.txt),
    # searched under THIS training procedure (weight-normed frozen head, teacher-temp warmup,
    # freeze_last_layer, grad clip, EMA capped < 1.0). The previous defaults (lr=1.02e-4,
    # bs=64, n_local=3, tau_teacher=0.0408 with no warmup) came from a 1000-step proxy of a
    # different procedure and collapsed the full SeCo run: teacher entropy -> ~0.18 over 2048
    # dims, EuroSAT linear probe 0.846 (below the 0.785 random-init floor before the fixes).
    # The new region is bs=256, many local crops, a slow lr, and tau_teacher ~= 0.074.
    parser.add_argument("--n-global", type=int, default=2)
    parser.add_argument("--n-local", type=int, default=6)
    parser.add_argument("--out-dim", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4.715e-5)
    parser.add_argument("--weight-decay", type=float, default=1.291e-3)
    parser.add_argument("--tau-student", type=float, default=0.1258)
    parser.add_argument("--tau-teacher", type=float, default=0.0741,
                        help="Target teacher temperature, reached via a linear warmup from "
                             "--tau-teacher-warmup.")
    parser.add_argument("--tau-teacher-warmup", type=float, default=0.04,
                        help="Teacher temperature at step 0 (warmed up to --tau-teacher).")
    parser.add_argument("--tau-teacher-warmup-steps", type=int, default=500,
                        help="Linear warmup length for the teacher temperature.")
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--clip-grad", type=float, default=3.0,
                        help="Max gradient norm for the student (0 disables). DINO default ~3.0.")
    parser.add_argument("--freeze-last-layer-steps", type=int, default=500,
                        help="Hold the DINO head's last layer fixed for this many steps "
                             "(anti-collapse; DINO freezes it for ~1 epoch).")
    parser.add_argument("--ema-momentum-start", type=float, default=0.9949)
    parser.add_argument("--ema-momentum-end", type=float, default=0.9995)
    parser.add_argument("--ema-momentum-max", type=float, default=0.9998,
                        help="Hard cap on the EMA momentum so the teacher never fully freezes.")
    parser.add_argument("--no-gpu-aug", action="store_true",
                         help="Fall back to the CPU (PIL/torchvision) augmentation pipeline behind a "
                              "DataLoader. Much slower and kept only as a reference path; the two "
                              "paths are not bit-identical, so do not mix them in one comparison.")
    parser.add_argument("--preload-res", type=int, default=None,
                         help="Downscale each patch to this resolution in the cache (default: native "
                              "264x264). Match train_mae.py / train_contrastive.py when comparing runs.")
    parser.add_argument("--cache-device", type=str, default="auto", choices=("auto", "cuda", "cpu"),
                         help="Where to hold the decoded uint8 patch cache. 'auto' puts it on the GPU "
                              "when it fits beside the model, otherwise in host RAM.")
    parser.add_argument("--preload-workers", type=int, default=8,
                         help="Worker processes for the one-time .tif decode into the patch cache.")
    parser.add_argument("--patch-cache", type=str, default=None,
                         help="Reuse a decoded patch cache on disk (a .u8 blob + .json index at this "
                              "path), building it on the first run. Shared across the three trainers.")
    parser.add_argument("--num-workers", type=int, default=2,
                         help="DataLoader workers. DINO's multi-crop augmentation is heavier than "
                              "MAE's, so a couple of workers overlap it with the GPU even after preload.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                         help="Batches each worker prefetches ahead (only used when num_workers>0).")
    parser.add_argument("--no-preload", action="store_true",
                         help="Disable the one-time in-RAM patch preload (use if RAM-constrained; "
                              "falls back to per-item disk decode).")
    parser.add_argument("--no-amp", action="store_true",
                         help="Disable mixed-precision autocast (use fp32).")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the .train.pt sidecar next to --out if it exists.")
    parser.add_argument("--init-encoder", type=str, default=None,
                         help="Path to an encoder-only checkpoint (e.g. a DINO-init export) to "
                              "initialize both student and teacher encoders from before training "
                              "starts. Distinct from --resume: this seeds weights only, training "
                              "still starts at step 0 with a fresh optimizer/scheduler/EMA/center.")
    add_dataset_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
