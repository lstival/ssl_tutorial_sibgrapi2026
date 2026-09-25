"""
Large-scale masked-autoencoder (MAE) pretraining of the tutorial's ViT-S/8 encoder on SeCo.

Offline counterpart to notebooks/remote_sensing/02_masking_mae.ipynb: same masking + asymmetric
encoder-decoder mechanism and the same `build_vit_s8()` architecture from
src/remote_sensing/tutorial_rs.py, run for many more steps on the SeCo corpus instead of a ~100-step live
demo on EuroSAT. Only the encoder weights are exported at the end (the decoder is
pretraining-only and is discarded).

This script does NOT import the notebooks' teaching code path (no fill-in-the-blank
sentinels) -- the masking and loss functions below are fixed, complete reference
implementations, structurally identical to the ones filled in inside Notebook 2.

Usage:
    python train_mae.py --seco-root ../../../data/seco/seasonal_contrast_100k \\
        --manifest ../../../data/seco/manifest.txt --out ../../../artifacts/remote_sensing/checkpoints/mae_vit_s8.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gpu_aug import gpu_augment  # noqa: E402
from pretrain_utils import (  # noqa: E402
    RunningLogger,
    _validate_corpus_args,
    add_dataset_args,
    build_gpu_source,
    enable_fast_cuda,
    load_train_state,
    resolve_locations,
    save_encoder_checkpoint,
    save_train_state,
    train_log_path,
    warmup_cosine_lambda,
)
from seco_data import (  # noqa: E402
    SeCoAugmentedDataset,
    build_seco_eval_transform,
    build_seco_mae_augmentations,
)

from tutorial_rs import (  # noqa: E402
    IMG_SIZE,
    TransformerBlock,
    build_vit_s8,
    get_device,
    seed_everything,
)

PATCH_SIZE = 8
GRID_SIZE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE


def patchify(imgs, patch_size=PATCH_SIZE):
    B, C, H, W = imgs.shape
    grid = H // patch_size
    x = imgs.reshape(B, C, grid, patch_size, grid, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(B, grid * grid, patch_size * patch_size * C)
    return x


def random_masking(x, mask_ratio):
    """Reference implementation -- the completed version of Notebook 2's fill-in-the-blank cell."""
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # ids_keep gives the original patch index of each visible token, in the exact order the
    # tokens appear in x_visible -- the encoder needs this to add the correct positional
    # embedding to each visible token (see forward_encoder).
    return x_visible, mask, ids_restore, ids_keep


class MAEModel(nn.Module):
    """Asymmetric encoder-decoder, identical in spirit to Notebook 2's MAEModel."""

    def __init__(self, encoder, decoder_dim=192, decoder_depth=2, decoder_heads=6,
                 patch_size=PATCH_SIZE, num_patches=NUM_PATCHES, in_chans=3):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.num_patches = num_patches
        embed_dim = encoder.embed_dim
        patch_dim = patch_size * patch_size * in_chans

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward_encoder(self, imgs, mask_ratio):
        x = self.encoder.patch_embed(imgs)
        x_visible, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

        cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        # Add the positional embedding of each visible token's *own* original patch. x_visible is
        # in ids_keep order, so we gather pos_embed at ids_keep -- NOT at argsort(argsort(mask)),
        # which returns the visible patch ids in ascending order and thus mis-pairs pos-embeds
        # with the (shuffle-ordered) visible tokens.
        pos_visible = torch.gather(
            self.encoder.pos_embed[:, 1:, :].expand(x.shape[0], -1, -1), dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        tokens = torch.cat([cls_tokens + self.encoder.pos_embed[:, :1, :], x_visible + pos_visible], dim=1)

        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(self, tokens, ids_restore):
        x = self.decoder_embed(tokens)
        B, N_plus1 = x.shape[0], ids_restore.shape[1] + 1

        mask_tokens = self.mask_token.expand(B, N_plus1 - x.shape[1], -1)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = torch.gather(x_no_cls, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_no_cls.shape[-1]))
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x[:, 1:, :])
        return pred

    def forward(self, imgs, mask_ratio=0.75):
        tokens, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(tokens, ids_restore)
        return pred, mask


def mae_reconstruction_loss(pred, imgs, mask, patch_size=PATCH_SIZE, norm_pix=True):
    """Masked-only pixel MSE -- the completed version of Notebook 2's fill-in-the-blank cell.

    norm_pix (per-patch target normalization) is the single most important MAE detail for
    representation quality, and its absence is why an earlier version of this encoder scored
    barely above random on the EuroSAT linear probe. Each target patch is standardized by its
    OWN mean and variance before the MSE (He et al., 2022, sec. 3). Without it the loss is
    dominated by each patch's mean luminance -- a trivial statistic the decoder can predict
    from the visible tokens without the encoder ever having to encode anything semantic, so
    the encoder learns almost nothing transferable. A 2.5k-step ablation on SeCo moved the
    linear probe from 0.766 to 0.805 by adding this one line.
    """
    target = patchify(imgs, patch_size=patch_size)
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6).sqrt()
    loss_per_patch = (pred - target) ** 2
    loss_per_patch = loss_per_patch.mean(dim=-1)
    loss = (loss_per_patch * mask).sum() / mask.sum()
    return loss


def build_dataloader(args):
    # One view per step: unlike contrastive/DINO, MAE's pretext task comes from masking, not from
    # a positive pair. But that view must still be augmented -- see build_seco_mae_augmentations:
    # the SeCo subset is small enough (10k patches) that a deterministic view lets the model
    # memorize pixels over ~500 epochs instead of learning transferable structure.
    if args.no_augment:
        transform = build_seco_eval_transform(img_size=IMG_SIZE)
    else:
        transform = build_seco_mae_augmentations(img_size=IMG_SIZE, min_scale=args.crop_min_scale)
    locations = resolve_locations(args.seco_root, args.manifest)
    preload = not args.no_preload
    if preload:
        res_note = f"at {args.preload_res}x{args.preload_res}" if args.preload_res else "at native 264x264"
        print(f"Preloading all SeCo patches into RAM {res_note} (one-time decode)...")
    dataset = SeCoAugmentedDataset(args.seco_root, transform, n_views=1, locations=locations,
                                   preload=preload, preload_res=args.preload_res)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    return dataset, loader


def train(args):
    seed_everything(args.seed)
    _validate_corpus_args(args)
    device = get_device()
    print(f"Device: {device}")
    enable_fast_cuda(device)

    # GPU augmentation is the default; --no-augment forces the CPU path because the
    # deterministic-view ablation has no GPU equivalent (gpu_augment always augments).
    use_gpu_aug = not args.no_gpu_aug and not args.no_augment and device.type == "cuda"
    cache = loader = None
    if use_gpu_aug:
        cache, shuffler, sample_gen, aug_gen, mean, std = build_gpu_source(args, device)
        n_locations = len(cache)
    else:
        dataset, loader = build_dataloader(args)
        n_locations = len(dataset)
    print(f"MAE pretraining pool: {n_locations} locations, mask_ratio={args.mask_ratio} "
          f"({'GPU' if use_gpu_aug else 'CPU'} augmentation)")

    decoder_heads = args.decoder_dim // 32  # 128->4, 192->6, 256->8 (all divisible)
    encoder = build_vit_s8()
    if args.init_encoder:
        encoder.load_state_dict(torch.load(args.init_encoder, map_location="cpu"))
        print(f"Initialized encoder weights from {args.init_encoder}")
    model = MAEModel(encoder, decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
                     decoder_heads=decoder_heads).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Linear warmup -> cosine decay. Without warmup the very first steps apply the full LR while
    # the decoder is still random, which is what made the previous run's loss spike (1.48 -> 2.05
    # between steps 1 and 2) before it recovered.
    warmup = max(0, min(args.warmup_steps, args.steps - 1))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda(args.steps, warmup))
    # Mixed precision: ~2x throughput and lower VRAM on Ampere, letting us use a larger batch.
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger = RunningLogger(train_log_path(args.out))

    # Full-state resume checkpoint (model + optimizer + scheduler + step), distinct from the
    # encoder-only checkpoint the notebooks load. Lives as a sidecar next to --out.
    state_path = os.path.splitext(args.out)[0] + ".train.pt"
    start_step = 0
    if args.resume:
        start_step, _ = load_train_state(state_path, model, optimizer, scheduler, device)
        if start_step >= args.steps:
            print(f"Already at step {start_step} >= target {args.steps}; nothing to do.")
            logger.close()
            return

    model.train()
    data_iter = iter(loader) if loader is not None else None
    for step in range(start_step + 1, args.steps + 1):
        if use_gpu_aug:
            loc_idx = shuffler.next_batch()
            raw = cache.sample_views(loc_idx, n_views=1, generator=sample_gen)
            # jitter_p=0.0: MAE regresses raw pixels, so the recipe is geometry-only -- the same
            # choice build_seco_mae_augmentations documents for the CPU path.
            imgs = gpu_augment(raw[0].to(device, non_blocking=True), IMG_SIZE, mean, std,
                               generator=aug_gen, scale=(args.crop_min_scale, 1.0),
                               jitter_p=0.0)
        else:
            try:
                views, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                views, _ = next(data_iter)
            imgs = views[0].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, mask = model(imgs, mask_ratio=args.mask_ratio)
            loss = mae_reconstruction_loss(pred, imgs, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.log(step, loss=loss.item(), lr=scheduler.get_last_lr()[0])

        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                model.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "mae", "mask_ratio": args.mask_ratio},
            )
            save_train_state(
                state_path, step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                extra={"mechanism": "mae", "mask_ratio": args.mask_ratio},
            )

    logger.close()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seco-root", type=str, default=None,
                         help="Path to the extracted seasonal_contrast_100k directory.")
    parser.add_argument("--manifest", type=str, default=None,
                         help="Optional manifest.txt (from download_seco.py) restricting which locations to use.")
    parser.add_argument("--out", type=str, default="../../../artifacts/remote_sensing/checkpoints/mae_vit_s8.pt",
                         help="Output path for the encoder-only checkpoint.")
    # Defaults below are the Optuna-selected recipe (see optuna_best_mae.txt), so that a plain
    # `python train_mae.py --seco-root ...` reproduces the tuned run. Previously the defaults were
    # generic (lr=1.5e-3, decoder 192x2) and had to be overridden on the command line -- a run
    # that forgot --lr silently trained at 5x the tuned rate.
    parser.add_argument("--mask-ratio", type=float, default=0.7678)
    parser.add_argument("--decoder-dim", type=int, default=128,
                         help="MAE decoder width (decoder_heads is derived as decoder_dim//32).")
    parser.add_argument("--decoder-depth", type=int, default=4, help="MAE decoder block count.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--no-amp", action="store_true",
                         help="Disable mixed-precision autocast (use fp32).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.056e-4)
    parser.add_argument("--weight-decay", type=float, default=4.673e-3)
    parser.add_argument("--warmup-steps", type=int, default=500,
                         help="Linear LR warmup before the cosine decay. MAE is unstable at step 0 "
                              "with a cold decoder; warmup is standard in the original recipe.")
    parser.add_argument("--no-augment", action="store_true",
                         help="Train on deterministic (un-augmented) views. Kept for the tutorial's "
                              "ablation -- it reproduces the memorization failure described in "
                              "build_seco_mae_augmentations, and is NOT the recommended setting.")
    parser.add_argument("--crop-min-scale", type=float, default=0.5,
                         help="Lower bound of RandomResizedCrop's area scale for MAE views.")
    parser.add_argument("--no-gpu-aug", action="store_true",
                         help="Fall back to the CPU (PIL/torchvision) augmentation pipeline behind a "
                              "DataLoader. Much slower (the GPU sits idle between batches) and kept "
                              "only as a reference path; the two paths are not bit-identical, so do "
                              "not mix them within one comparison.")
    parser.add_argument("--cache-device", type=str, default="auto", choices=("auto", "cuda", "cpu"),
                         help="Where to hold the decoded uint8 patch cache. 'auto' puts it on the GPU "
                              "when it fits beside the model, otherwise in host RAM. GPU-only affects "
                              "speed, not results.")
    parser.add_argument("--preload-workers", type=int, default=8,
                         help="Worker processes for the one-time .tif decode into the patch cache.")
    parser.add_argument("--patch-cache", type=str, default=None,
                         help="Reuse a decoded patch cache on disk (a .u8 blob + .json index at this "
                              "path), building it on the first run. Shared with train_contrastive.py.")
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader workers. Default 0: with --preload the patches are already "
                              "in RAM, so workers add only spawn/IPC overhead on Windows.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                         help="Batches each worker prefetches ahead (only used when num_workers>0).")
    parser.add_argument("--no-preload", action="store_true",
                         help="Disable the one-time in-RAM patch preload (use if RAM-constrained; "
                              "falls back to per-item disk decode).")
    parser.add_argument("--preload-res", type=int, default=None,
                         help="Downscale each patch to this resolution while preloading (default: "
                              "keep native 264x264, which needs ~21 GB RAM for the full 100k corpus). "
                              "--preload-res 128 brings that to ~4.9 GB. All runs being compared must "
                              "use the SAME value (see train_contrastive.py's flag of the same name).")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the .train.pt sidecar next to --out if it exists.")
    parser.add_argument("--init-encoder", type=str, default=None,
                         help="Path to an encoder-only checkpoint (e.g. a DINO-init export) to "
                              "initialize the encoder from before training starts. Distinct from "
                              "--resume: this seeds weights only, training still starts at step 0 "
                              "with a fresh optimizer/scheduler/decoder.")
    add_dataset_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
