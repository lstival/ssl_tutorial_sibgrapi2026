"""
Optuna hyperparameter search for the MAE and contrastive pretraining recipes (DINO has its
own search, optuna_dino.py, matched to train_dino.py's anti-collapse procedure).

Rather than run a full 10k-step pretrain per trial (hours each), we use a cheap *proxy*: a
short pretrain (PROXY_STEPS) on the SeCo pool, then a linear probe on EuroSAT features, and
maximize probe accuracy on a validation split carved from the EuroSAT *train* side (see
eurosat_probe.hpo_probe_accuracy -- the test split is never used for model selection). The SeCo
dataset is preloaded once and shared across trials. Each trial takes ~1-2 min on an RTX 3060.

The winning hyperparameters are then used for a full-length retrain (train_mae.py /
train_contrastive.py).

Usage:
    python optuna_search.py --mechanism mae         --trials 20
    python optuna_search.py --mechanism contrastive --trials 20
"""
import argparse
import os
import sys

import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_contrastive as contrastive_mod  # noqa: E402
import train_mae as mae_mod  # noqa: E402
from eurosat_probe import hpo_probe_accuracy  # noqa: E402
from pretrain_utils import DATA_DIR, warmup_cosine_lambda  # noqa: E402
from seco_data import (  # noqa: E402
    SeCoAugmentedDataset,
    build_seco_augmentations,
    build_seco_mae_augmentations,
    read_manifest,
)

from tutorial_rs import IMG_SIZE, build_vit_s8, get_device, seed_everything  # noqa: E402

SECO_ROOT = os.path.join(DATA_DIR, "seco", "seasonal_contrast_100k")
MANIFEST = os.path.join(DATA_DIR, "seco", "manifest.txt")

PROXY_STEPS = int(os.environ.get("PROXY_STEPS", 1000))  # short pretrain per trial
PROBE_SUBSET_PER_CLASS = 200  # cap EuroSAT probe train size for speed
device = get_device()


# ----------------------------------------------------------------------------------------
# Shared, load-once resources
# ----------------------------------------------------------------------------------------
_seco_cache = {}
def get_seco_dataset(transform):
    """Preload SeCo once (uint8 patches in RAM) and reuse across trials."""
    if "single" not in _seco_cache:
        locations = read_manifest(MANIFEST)
        _seco_cache["single"] = SeCoAugmentedDataset(SECO_ROOT, transform, n_views=1,
                                                     locations=locations, preload=True)
    ds = _seco_cache["single"]
    ds.transform = transform  # swap the (cheap) transform, keep the preloaded patches
    return ds


# ----------------------------------------------------------------------------------------
# MAE proxy
# ----------------------------------------------------------------------------------------
def train_mae_proxy(trial):
    # LR lower bound goes below the previous 3e-4 floor: with augmented views (and warmup) the
    # useful range shifts down, and the old floor was itself the value the search picked -- a sign
    # the optimum was pinned at the edge of the space.
    lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.5, 0.85)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    decoder_depth = trial.suggest_int("decoder_depth", 1, 4)
    decoder_dim = trial.suggest_categorical("decoder_dim", [128, 192, 256])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])

    seed_everything(42)
    # Must match train_mae.py's dataloader: augmented (geometry-only) views, not deterministic
    # ones. Searching under a deterministic pipeline tunes for the memorization regime and the
    # winning hyperparameters do not transfer to the real run.
    tf = build_seco_mae_augmentations(img_size=IMG_SIZE)
    ds = get_seco_dataset(tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    encoder = build_vit_s8().to(device)
    decoder_heads = decoder_dim // 32  # 128->4, 192->6, 256->8 (all divisible)
    model = mae_mod.MAEModel(encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
                             decoder_heads=decoder_heads).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Same warmup->cosine schedule as train_mae.py (scaled to the proxy's shorter budget), so the
    # proxy ranks hyperparameters under the schedule they will actually be trained with.
    warmup = max(1, PROXY_STEPS // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, warmup_cosine_lambda(PROXY_STEPS, warmup))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    it = iter(loader)
    for _step in range(PROXY_STEPS):
        try:
            views, _ = next(it)
        except StopIteration:
            it = iter(loader); views, _ = next(it)
        imgs = views[0].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred, mask = model(imgs, mask_ratio=mask_ratio)
            loss = mae_mod.mae_reconstruction_loss(pred, imgs, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    return hpo_probe_accuracy(model.encoder, device, pool="mean", per_class=PROBE_SUBSET_PER_CLASS)


# ----------------------------------------------------------------------------------------
# Contrastive (SimCLR / InfoNCE) proxy
# ----------------------------------------------------------------------------------------
def train_contrastive_proxy(trial):
    # Search the knobs that actually move contrastive quality: LR, the InfoNCE temperature (the
    # single most sensitive contrastive hyperparameter), weight decay, and batch size (the number
    # of in-batch negatives). The projection head width is fixed to match train_contrastive.py.
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    temperature = trial.suggest_float("temperature", 0.05, 0.5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    seed_everything(42)
    # Same augmented positive-pair pipeline as train_contrastive.py (two independent augmentations
    # of the same season). n_views=2 makes the dataset return a [view_a, view_b] pair per sample.
    tf = build_seco_augmentations(img_size=IMG_SIZE)
    ds = get_seco_dataset(tf)
    ds.n_views = 2  # the shared single-view cache is fine; only the per-item view count changes
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = contrastive_mod.SimCLRModel(build_vit_s8()).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Same warmup->cosine schedule shape as the MAE proxy so hyperparameters are ranked under a
    # schedule close to the real run (train_contrastive.py uses cosine to lr/50).
    warmup = max(1, PROXY_STEPS // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, warmup_cosine_lambda(PROXY_STEPS, warmup))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    it = iter(loader)
    for _step in range(PROXY_STEPS):
        try:
            views, _ = next(it)
        except StopIteration:
            it = iter(loader); views, _ = next(it)
        imgs = torch.cat(views, dim=0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            feats = model(imgs)
            loss, _ = contrastive_mod.info_nce_loss(feats.float(), temperature=temperature)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    # Contrastive trains the [CLS] token directly, so probe it via CLS pooling (matches Notebook 4).
    return hpo_probe_accuracy(model.encoder, device, pool="cls", per_class=PROBE_SUBSET_PER_CLASS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=["mae", "contrastive"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    print(f"Device: {device} | mechanism={args.mechanism} | trials={args.trials} | proxy_steps={PROXY_STEPS}")
    objective = {
        "mae": train_mae_proxy,
        "contrastive": train_contrastive_proxy,
    }[args.mechanism]
    study = optuna.create_study(direction="maximize",
                                study_name=args.study_name or f"{args.mechanism}_proxy",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print("\n=== BEST ===")
    print("value (probe acc):", study.best_value)
    print("params:", study.best_params)
    out = os.path.join(os.path.dirname(__file__), f"optuna_best_{args.mechanism}.txt")
    with open(out, "w") as f:
        f.write(f"best_value {study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k} {v}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
