"""
Optuna hyperparameter search for the MAE and DINO pretraining recipes.

Rather than run a full 10k-step pretrain per trial (hours each), we use a cheap *proxy*: a
short pretrain (PROXY_STEPS) on the SeCo pool, then a linear probe on EuroSAT features, and
maximize probe accuracy. The SeCo dataset is preloaded once and shared across trials, and the
EuroSAT probe features are cached per-encoder. This keeps each trial to ~1-2 min on the RTX 3060.

The winning hyperparameters are then used for a full-length retrain (train_mae.py / train_dino.py).

Usage:
    python optuna_search.py --mechanism mae         --trials 20
    python optuna_search.py --mechanism dino        --trials 20
    python optuna_search.py --mechanism contrastive --trials 20
"""
import argparse
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
NOTEBOOKS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, NOTEBOOKS)

import optuna
import train_contrastive as contrastive_mod  # noqa: E402
import train_dino as dino_mod  # noqa: E402
import train_mae as mae_mod  # noqa: E402
from seco_data import (  # noqa: E402
    SECO_MEAN,
    SECO_STD,
    SeCoAugmentedDataset,
    build_seco_augmentations,
    build_seco_mae_augmentations,
    read_manifest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tutorial_rs import (  # noqa: E402
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMG_SIZE,
    build_eval_transform,
    build_vit_s8,
    get_device,
    load_eurosat,
    seed_everything,
    stratified_split,
)

SECO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "seco", "seasonal_contrast_100k")
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "seco", "manifest.txt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

PROXY_STEPS = int(os.environ.get("PROXY_STEPS", 1000))  # short pretrain per trial
PROBE_SUBSET_PER_CLASS = 200  # cap EuroSAT probe train size for speed
device = get_device()


# ----------------------------------------------------------------------------------------
# Shared, load-once resources
# ----------------------------------------------------------------------------------------
_seco_cache = {}
def get_seco_dataset(transform, multicrop=False):
    """Preload SeCo once (uint8 patches in RAM) and reuse across trials."""
    key = "multicrop" if multicrop else "single"
    if key not in _seco_cache:
        locations = read_manifest(MANIFEST)
        ds = SeCoAugmentedDataset(SECO_ROOT, transform, n_views=1, locations=locations, preload=True)
        _seco_cache[key] = ds
    ds = _seco_cache[key]
    ds.transform = transform  # swap the (cheap) transform, keep the preloaded patches
    return ds


_eurosat_cache = {}
def get_eurosat_splits():
    if "splits" not in _eurosat_cache:
        tf = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
        full = load_eurosat(DATA_DIR, transform=tf)
        tr, te = stratified_split(full, test_size=0.2, seed=42)
        _eurosat_cache["splits"] = (full, tr, te)
    return _eurosat_cache["splits"]


@torch.no_grad()
def extract_features(encoder, subset_idx, full, pool, batch_size=256):
    import torch.utils.data as data
    loader = data.DataLoader(data.Subset(full, subset_idx), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    encoder.eval()
    for imgs, y in loader:
        imgs = imgs.to(device)
        h = encoder.forward_features(imgs, pool=pool)
        feats.append(h.cpu()); labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def probe_accuracy(encoder, pool):
    full, tr, te = get_eurosat_splits()
    # cap probe-train size per class for speed
    labels_all = np.array([full.targets[i] for i in tr])
    rng = np.random.RandomState(0)
    keep = []
    for c in np.unique(labels_all):
        idx = np.where(labels_all == c)[0]
        rng.shuffle(idx)
        keep.extend(idx[:PROBE_SUBSET_PER_CLASS])
    tr_sub = [tr[i] for i in keep]
    Xtr, ytr = extract_features(encoder, tr_sub, full, pool)
    Xte, yte = extract_features(encoder, te, full, pool)
    # Standardize features -> logistic regression converges much faster/stabler on ViT embeddings.
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


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
    ds = get_seco_dataset(tf, multicrop=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    encoder = build_vit_s8().to(device)
    decoder_heads = decoder_dim // 32  # 128->4, 192->6, 256->8 (all divisible)
    model = mae_mod.MAEModel(encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
                             decoder_heads=decoder_heads).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Same warmup->cosine schedule as train_mae.py (scaled to the proxy's shorter budget), so the
    # proxy ranks hyperparameters under the schedule they will actually be trained with.
    warmup = max(1, PROXY_STEPS // 20)

    def _lr_lambda(step):
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, PROXY_STEPS - warmup)
        return (1 / 50) + (1 - 1 / 50) * 0.5 * (1 + np.cos(np.pi * progress))

    sched = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
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

    return probe_accuracy(model.encoder, pool="mean")


# ----------------------------------------------------------------------------------------
# DINO proxy
# ----------------------------------------------------------------------------------------
def train_dino_proxy(trial):
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    out_dim = trial.suggest_categorical("out_dim", [1024, 2048, 4096])
    tau_teacher = trial.suggest_float("tau_teacher", 0.02, 0.07)
    tau_student = trial.suggest_float("tau_student", 0.1, 0.2)
    ema_start = trial.suggest_float("ema_momentum_start", 0.99, 0.9975)
    n_local = trial.suggest_int("n_local", 2, 6)
    batch_size = trial.suggest_categorical("batch_size", [48, 64])

    seed_everything(42)
    global_tf = build_seco_augmentations(img_size=IMG_SIZE)
    local_tf = transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.2, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(), transforms.Normalize(SECO_MEAN, SECO_STD),
    ])
    n_global = 2
    multicrop = dino_mod.MultiCropTransform(global_tf, local_tf, n_global=n_global, n_local=n_local)
    ds = get_seco_dataset(multicrop, multicrop=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                        collate_fn=dino_mod._multicrop_collate)

    student = dino_mod.DINOModel(build_vit_s8(), out_dim=out_dim).to(device)
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_STEPS, eta_min=lr / 50)
    center = torch.zeros(1, out_dim, device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    student.train(); teacher.eval()
    it = iter(loader)
    for step in range(PROXY_STEPS):
        try:
            crops = next(it)
        except StopIteration:
            it = iter(loader); crops = next(it)
        crops = [c.to(device, non_blocking=True) for c in crops]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            s_logits = [student(c) for c in crops]
            with torch.no_grad():
                t_logits = [teacher(c) for c in crops[:n_global]]
        s_logits = [s.float() for s in s_logits]
        t_logits = [t.float() for t in t_logits]
        loss = dino_mod.dino_loss(s_logits, t_logits, center, tau_student, tau_teacher)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        m = 1.0 - (1.0 - ema_start) * (1 + np.cos(step / PROXY_STEPS * np.pi)) / 2
        dino_mod.ema_update(student, teacher, m=m)
        center = dino_mod.update_center(center, t_logits, momentum=0.9)

    return probe_accuracy(teacher.encoder, pool="cls")


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
    ds = get_seco_dataset(tf, multicrop=False)
    ds.n_views = 2  # the shared single-view cache is fine; only the per-item view count changes
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = contrastive_mod.SimCLRModel(build_vit_s8()).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Same warmup->cosine schedule shape as the MAE proxy so hyperparameters are ranked under a
    # schedule close to the real run (train_contrastive.py uses cosine to lr/50).
    warmup = max(1, PROXY_STEPS // 20)

    def _lr_lambda(step):
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, PROXY_STEPS - warmup)
        return (1 / 50) + (1 - 1 / 50) * 0.5 * (1 + np.cos(np.pi * progress))

    sched = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
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
    return probe_accuracy(model.encoder, pool="cls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=["mae", "dino", "contrastive"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    print(f"Device: {device} | mechanism={args.mechanism} | trials={args.trials} | proxy_steps={PROXY_STEPS}")
    objective = {
        "mae": train_mae_proxy,
        "dino": train_dino_proxy,
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
