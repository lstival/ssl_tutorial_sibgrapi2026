"""
Large-scale contrastive (SimCLR-style) pretraining of the tutorial's ViT-S/8 encoder on SeCo.

This is the offline counterpart to notebooks/remote_sensing/01_contrastive_simclr.ipynb: same InfoNCE
mechanism and the same `build_vit_s8()` architecture from src/remote_sensing/tutorial_rs.py, but run for
many more steps on the SeCo remote-sensing corpus instead of a ~100-step live demo on
EuroSAT. Only the encoder weights are exported at the end (the projection head is
pretraining-only and is discarded), matching what src/remote_sensing/tutorial_rs.py's
`try_load_checkpoint` expects for Notebooks 1-4 to load.

This script intentionally does NOT import anything from the notebooks' teaching code path
(no fill-in-the-blank sentinels, no plotting) -- only the shared model architecture. The loss
function below is a fixed, already-complete reference implementation.

Usage:
    python train_contrastive.py --seco-root ../../../data/seco/seasonal_contrast_100k \\
        --manifest ../../../data/seco/manifest.txt --out ../../../artifacts/remote_sensing/checkpoints/contrastive_vit_s8.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gpu_aug import gpu_augment  # noqa: E402
from pretrain_utils import (  # noqa: E402
    ARCHS,
    RunningLogger,
    _validate_corpus_args,
    add_dataset_args,
    build_gpu_source,
    enable_fast_cuda,
    resolve_locations,
    save_encoder_checkpoint,
    train_log_path,
)
from seco_data import (  # noqa: E402
    SeCoAugmentedDataset,
    SeCoSeasonalDataset,
    build_seco_augmentations,
)

from tutorial_rs import IMG_SIZE, get_device, seed_everything  # noqa: E402


class SimCLRModel(nn.Module):
    """Encoder + small projection head, identical in spirit to Notebook 1's SimCLRModel."""

    def __init__(self, encoder, embed_dim=None, hidden_dim=512, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        # Read the width off the encoder rather than hardcoding ViT-S's 384 (vit_t8 is 192-wide).
        if embed_dim is None:
            embed_dim = encoder.embed_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder.forward_features(x)
        z = self.projection_head(h)
        return z


def info_nce_loss(feats, temperature=0.07):
    """
    Reference InfoNCE implementation -- the completed version of Notebook 1's fill-in-the-blank
    cell. feats: (2N, proj_dim) embeddings, with the two views of image k at positions k and
    k+N in the batch.
    """
    # Cosine similarity of every pair = Gram matrix of the L2-normalized embeddings. The
    # broadcast form (feats[:, None] vs feats[None, :]) is the same arithmetic but materializes
    # a (2N, 2N, D) intermediate -- 134 MB per step at 2N=512, D=128 -- where the matmul touches
    # only (2N, 2N). Kept in fp32 so the loss is unaffected by the autocast region around it.
    feats = F.normalize(feats.float(), dim=-1)
    cos_sim = feats @ feats.t()
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    loss = nll.mean()

    comb_sim = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    acc_top1 = (sim_argsort == 0).float().mean().item()
    return loss, acc_top1


def check_pool_size(n_locations, batch_size):
    """
    A pool smaller than one batch yields ZERO batches per epoch (drop_last=True), and the
    training loop's StopIteration retry then re-raises on the fresh iterator -- surfacing as a
    bare StopIteration traceback far from the cause. Fail fast with the actual problem instead.
    Raising (rather than lowering the batch size) is deliberate: batch size sets the number of
    InfoNCE negatives and is therefore part of the objective, so silently shrinking it would
    make runs incomparable -- which matters directly for the corpus-scaling study in
    run_seco_scaling.py, where every point must share one batch size.
    """
    if n_locations < batch_size:
        raise ValueError(
            f"Pretraining pool has {n_locations} locations but --batch-size is {batch_size}; "
            f"with drop_last=True that yields no batches. Use a larger corpus (or manifest) or a "
            f"smaller --batch-size."
        )


def build_dataloader(args):
    transform = build_seco_augmentations(img_size=IMG_SIZE)
    locations = resolve_locations(args.seco_root, args.manifest)

    if args.seasonal_positives:
        # SeCoSeasonalDataset has no in-RAM preload path (unlike SeCoAugmentedDataset below), so
        # this mode still decodes each .tif from disk on every access.
        dataset = SeCoSeasonalDataset(args.seco_root, transform, locations=locations)
    else:
        # Unlike train_mae.py / train_dino.py, this previously never passed preload=True, so every
        # __getitem__ re-decoded both views' .tif bands from disk (~12ms/patch, see seco_data.py) --
        # the dominant per-step cost, which starves the GPU and shows up as high sustained CPU with
        # bursty (not saturated) GPU utilization. Preloading decodes the whole pool into RAM once,
        # up front, so num_workers=0 (workers would just duplicate that RAM copy on Windows spawn).
        preload = not args.no_preload
        if preload:
            res_note = f"at {args.preload_res}x{args.preload_res}" if args.preload_res else "at native 264x264"
            print(f"Preloading all SeCo patches into RAM {res_note} (one-time decode)...")
        dataset = SeCoAugmentedDataset(args.seco_root, transform, n_views=2, locations=locations,
                                       preload=preload, preload_res=args.preload_res)

    check_pool_size(len(dataset), args.batch_size)

    num_workers = 0 if (not args.seasonal_positives and not args.no_preload) else args.num_workers
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return dataset, loader


def train(args):
    seed_everything(args.seed)
    _validate_corpus_args(args)
    device = get_device()
    print(f"Device: {device}")
    enable_fast_cuda(device)

    use_gpu_aug = not args.no_gpu_aug and device.type == "cuda"
    cache = loader = None
    if use_gpu_aug:
        cache, shuffler, sample_gen, aug_gen, mean, std = build_gpu_source(args, device)
        check_pool_size(len(cache), args.batch_size)
        n_locations = len(cache)
    else:
        dataset, loader = build_dataloader(args)
        n_locations = len(dataset)
    print(f"Contrastive pretraining pool: {n_locations} locations "
          f"({'seasonal' if args.seasonal_positives else 'augmentation'} positives, "
          f"{'GPU' if use_gpu_aug else 'CPU'} augmentation)")

    encoder = ARCHS[args.arch]()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder: {args.arch} ({n_params / 1e6:.2f}M parameters)")
    model = SimCLRModel(encoder).to(device)
    start_step = 0
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.encoder.load_state_dict(state_dict)
        start_step = args.resume_step
        print(f"Resumed encoder weights from {args.resume} (treated as step {start_step})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule still spans the full [0, args.steps] range even when resuming, so the LR
    # picks up where the original run's schedule would have been at `start_step` rather than
    # restarting the warm cosine decay from lr_max.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr / 50)
    for _ in range(start_step):
        scheduler.step()

    log_path = train_log_path(args.out)
    logger = RunningLogger(log_path)
    print(f"Logging to {log_path}")

    # Mixed precision, as train_mae.py and train_dino.py already do. It was missing here, so the
    # contrastive runs were the only ones paying full fp32 for a transformer forward/backward.
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision: {'on' if use_amp else 'off'}")

    model.train()
    data_iter = iter(loader) if loader is not None else None
    step_ema = None
    for step in range(start_step + 1, args.steps + 1):
        t_step = time.perf_counter()

        if use_gpu_aug:
            loc_idx = shuffler.next_batch()
            raw = cache.sample_views(loc_idx, n_views=2,
                                     distinct_seasons=args.seasonal_positives,
                                     generator=sample_gen)
            # Each view is augmented independently (out-of-place), so the two entries may be the
            # same source tensor -- that is exactly the SimCLR positive pair.
            views = [gpu_augment(v.to(device, non_blocking=True), IMG_SIZE, mean, std,
                                 generator=aug_gen) for v in raw]
        else:
            try:
                views, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                views, _ = next(data_iter)
            views = [v.to(device, non_blocking=True) for v in views]

        imgs = torch.cat(views, dim=0)
        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = model(imgs)
        # The loss runs outside autocast and recasts to fp32: InfoNCE's logsumexp over 2N logits
        # divided by a small temperature is exactly the kind of large-dynamic-range reduction
        # fp16 handles badly.
        loss, acc_top1 = info_nce_loss(feats, temperature=args.temperature)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        dt = time.perf_counter() - t_step
        step_ema = dt if step_ema is None else 0.98 * step_ema + 0.02 * dt
        logger.log(step, loss=loss.item(), acc_top1=acc_top1, lr=scheduler.get_last_lr()[0],
                   img_per_sec=2 * args.batch_size / step_ema)

        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                model.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "contrastive",
                       "temperature": args.temperature, "arch": args.arch,
                       "batch_size": args.batch_size, "img_size": IMG_SIZE},
            )

    logger.close()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seco-root", type=str, default=None,
                         help="Path to the extracted seasonal_contrast_100k directory.")
    parser.add_argument("--manifest", type=str, default=None,
                         help="Optional manifest.txt (from download_seco.py) restricting which locations to use.")
    parser.add_argument("--out", type=str, default="../../../artifacts/remote_sensing/checkpoints/contrastive_vit_s8.pt",
                         help="Output path for the encoder-only checkpoint.")
    parser.add_argument("--seasonal-positives", action="store_true",
                         help="Use two different seasonal revisits as the positive pair instead of two augmentations of the same season.")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to an encoder-only checkpoint (e.g. a previous --out) to resume training from.")
    parser.add_argument("--resume-step", type=int, default=0,
                         help="Step count to resume from (matches the checkpoint's saved 'step', e.g. from its .json sidecar); controls where the cosine LR schedule and the remaining step range pick up.")
    parser.add_argument("--arch", type=str, default="vit_s8", choices=sorted(ARCHS),
                        help="Encoder variant: vit_s8 (~10.7M params) or vit_t8 (~1.8M params).")
    parser.add_argument("--steps", type=int, default=20000)
    # Defaults below are the Optuna-selected recipe (see optuna_best_contrastive.txt), so that a
    # plain `python train_contrastive.py --seco-root ...` reproduces the tuned run, matching the
    # pattern already used by train_mae.py / train_dino.py. Notably the tuned temperature (0.129)
    # is well above the SimCLR-paper default (0.07) that was hardcoded here before -- a run that
    # forgot --temperature was silently training with a much sharper, unsearched softmax.
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.435e-4)
    parser.add_argument("--weight-decay", type=float, default=3.596e-5)
    parser.add_argument("--temperature", type=float, default=0.1289)
    parser.add_argument("--no-gpu-aug", action="store_true",
                         help="Fall back to the CPU (PIL/torchvision) augmentation pipeline behind a "
                              "DataLoader. Much slower (~1.2 s/step vs ~0.1 s/step at batch 256) and "
                              "kept only as a reference path; note the two paths are not "
                              "bit-identical, so do not mix them within one comparison.")
    parser.add_argument("--cache-device", type=str, default="auto", choices=("auto", "cuda", "cpu"),
                         help="Where to hold the decoded uint8 patch cache. 'auto' puts it on the GPU "
                              "when it fits beside the model (100k patches at --preload-res 128 is "
                              "4.9 GB), otherwise in host RAM. GPU-only affects speed, not results.")
    parser.add_argument("--preload-workers", type=int, default=8,
                         help="Worker processes for the one-time .tif decode into the patch cache.")
    parser.add_argument("--patch-cache", type=str, default=None,
                         help="Reuse a decoded patch cache on disk (a .u8 blob + .json index at this "
                              "path), building it on the first run. Decoding the 100k corpus costs "
                              "~15 min of .tif reads that every process would otherwise repeat; "
                              "reloading costs seconds. The store grows to a superset, so one file "
                              "serves the whole scaling curve.")
    parser.add_argument("--no-amp", action="store_true",
                         help="Disable mixed-precision autocast (use fp32).")
    parser.add_argument("--num-workers", type=int, default=4,
                         help="DataLoader workers, --no-gpu-aug only. Ignored (forced to 0) when "
                              "preloading is active (the default with --seasonal-positives off); "
                              "only used with "
                              "--no-preload or --seasonal-positives.")
    parser.add_argument("--no-preload", action="store_true",
                         help="--no-gpu-aug only (the GPU path always caches). Disable the one-time "
                              "in-RAM patch preload (use if RAM-constrained; "
                              "falls back to per-item disk decode). Has no effect with "
                              "--seasonal-positives, which never preloads.")
    parser.add_argument("--preload-res", type=int, default=None,
                         help="Downscale each patch to this resolution while preloading (default: "
                              "keep native 264x264). Native costs ~204 KB/patch, so the full 100k "
                              "corpus needs ~21 GB RAM; --preload-res 128 brings that to ~4.9 GB. "
                              "All runs being compared must use the SAME value.")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    add_dataset_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
