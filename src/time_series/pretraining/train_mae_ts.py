"""
Masked-autoencoder (MAE / PatchTST-self-supervised style) pretraining of the tutorial's patch
Transformer on the pooled UCR archive.

Offline counterpart to notebooks/time_series/02_masking_mae.ipynb: same masking + asymmetric
encoder-decoder mechanism and the same `build_ts_encoder()` architecture. Only the encoder
weights are exported (the decoder is pretraining-only).

The masking and loss functions below are fixed, complete reference implementations,
structurally identical to the ones filled in inside Notebook 2. `--no-norm-pix` reproduces the
The remote sensing failure mode (pretext loss falls smoothly while the encoder learns little) for the
ablation discussed in Notebook 4.

Usage:
    python train_mae_ts.py --data-path ../../../data \\
        --out ../../../artifacts/time_series/checkpoints/mae_ts_encoder.pt
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from pretrain_utils_ts import RunningLogger, save_encoder_checkpoint  # noqa: E402
from ucr_data import DEFAULT_PER_DATASET_CAP, build_pretraining_dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_ts import (  # noqa: E402
    PATCH_LEN,
    PATCH_STRIDE,
    SERIES_LEN,
    TransformerBlock,
    build_ts_encoder,
    get_device,
    seed_everything,
)

NUM_PATCHES = (SERIES_LEN - PATCH_LEN) // PATCH_STRIDE + 1


def patchify(x, patch_len=PATCH_LEN, stride=PATCH_STRIDE):
    x = x.permute(0, 2, 1)
    return x.unfold(dimension=-1, size=patch_len, step=stride).contiguous()  # (B,C,N,plen)


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
    return x_visible, mask, ids_restore, ids_keep


class MAETSModel(nn.Module):
    """Asymmetric encoder-decoder, identical in spirit to Notebook 2's MAETSModel."""

    def __init__(self, encoder, decoder_dim=64, decoder_depth=2, decoder_heads=4,
                 patch_len=PATCH_LEN, num_patches=NUM_PATCHES):
        super().__init__()
        self.encoder = encoder
        self.patch_len = patch_len
        self.num_patches = num_patches
        embed_dim = encoder.embed_dim

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_len)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward_encoder(self, imgs, mask_ratio):
        B, L, C = imgs.shape
        x = imgs.permute(0, 2, 1).reshape(B * C, L)
        x = self.encoder.patch_embed(x)
        x_visible, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)
        cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        pos_visible = torch.gather(
            self.encoder.pos_embed[:, 1:, :].expand(x.shape[0], -1, -1), dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        tokens = torch.cat(
            [cls_tokens + self.encoder.pos_embed[:, :1, :], x_visible + pos_visible], dim=1
        )
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(self, tokens, ids_restore):
        x = self.decoder_embed(tokens)
        B, N_plus1 = x.shape[0], ids_restore.shape[1] + 1
        mask_tokens = self.mask_token.expand(B, N_plus1 - x.shape[1], -1)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = torch.gather(x_no_cls, dim=1,
                                index=ids_restore.unsqueeze(-1).expand(-1, -1, x_no_cls.shape[-1]))
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x[:, 1:, :])

    def forward(self, imgs, mask_ratio=0.5):
        B, L, C = imgs.shape
        tokens, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(tokens, ids_restore).reshape(B, C, self.num_patches, self.patch_len)
        mask = mask.reshape(B, C, self.num_patches)
        return pred, mask


def mae_reconstruction_loss(pred, imgs, mask, norm_pix=True):
    """Masked-only patch MSE -- the completed version of Notebook 2's fill-in-the-blank cell.

    norm_pix (per-patch target normalization) is the single most important MAE detail for
    representation quality (He et al., 2022, sec. 3). `--no-norm-pix` drops it to reproduce
    the remote sensing failure mode for Notebook 4's discussion.
    """
    target = patchify(imgs)
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6).sqrt()
    loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
    return (loss_per_patch * mask).sum() / mask.sum()


def train(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    exclude = (args.exclude_target,) if args.exclude_target else ()
    dataset, sizes = build_pretraining_dataset(
        args.data_path, transform=None, per_dataset_cap=args.per_dataset_cap, exclude=exclude)
    print(f"MAE pretraining pool: {len(dataset)} series from {len(sizes)} UCR datasets, "
          f"mask_ratio={args.mask_ratio}, norm_pix={not args.no_norm_pix}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True)

    decoder_heads = max(1, args.decoder_dim // 16)
    model = MAETSModel(build_ts_encoder(), decoder_dim=args.decoder_dim,
                       decoder_depth=args.decoder_depth, decoder_heads=decoder_heads).to(device)
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
            windows = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            windows = next(data_iter)
        windows = windows.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, mask = model(windows, mask_ratio=args.mask_ratio)
        loss = mae_reconstruction_loss(pred, windows, mask, norm_pix=not args.no_norm_pix)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.log(step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_encoder_checkpoint(
                model.encoder, args.out,
                extra={"step": step, "loss": loss.item(), "mechanism": "mae",
                       "mask_ratio": args.mask_ratio, "norm_pix": not args.no_norm_pix,
                       "corpus": "ucr", "n_datasets": len(sizes), "n_series": len(dataset),
                       "steps_total": args.steps},
            )
    logger.close()
    print("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=str, default="../../../data",
                   help="Directory holding (or to receive) the UCRArchive_2018 folder.")
    p.add_argument("--out", type=str, default="../../../artifacts/time_series/checkpoints/mae_ts_encoder.pt")
    p.add_argument("--per-dataset-cap", type=int, default=DEFAULT_PER_DATASET_CAP,
                   help="Max series contributed by any single UCR dataset.")
    p.add_argument("--exclude-target", type=str, default=None,
                   help="Drop this dataset from the corpus (e.g. SwedishLeaf) for the strict "
                        "cross-dataset transfer ablation.")
    # A 128-step series gives only 8 patch tokens, so a 0.75 image-MAE ratio would leave just 2
    # visible tokens. 0.5 keeps 4 visible -- enough context to make the task solvable while
    # still forcing genuine inference rather than local interpolation.
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--decoder-dim", type=int, default=64)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--no-norm-pix", action="store_true",
                   help="Drop per-patch target normalization -- reproduces the remote sensing MAE "
                        "failure mode (pretext loss falls, representation barely improves).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
