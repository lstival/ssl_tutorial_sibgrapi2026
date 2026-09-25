"""
Optuna search for the DINO recipe, matched to the CURRENT train_dino.py training procedure.

Why a separate script from optuna_search.py:

  * The old DINO proxy (optuna_search.train_dino_proxy) tuned a *different* procedure than
    the one train_dino.py now runs. It had no teacher-temp warmup, no freeze_last_layer, no
    grad clip, a plain nn.Linear head, and an EMA schedule that reached m=1.0 -- exactly the
    settings that collapsed the first full SeCo run. A recipe searched under those dynamics
    does not transfer.
  * It also used only 1000 proxy steps on a CPU DataLoader. The DINO collapse-then-recover
    dynamics play out over several thousand steps, so 1000 is too short to rank recipes, and
    the CPU pipeline made longer proxies impractical.

This script:
  * Uses the SAME code paths as train_dino.py -- DINOModel (weight-normed frozen head),
    dino_loss, ema_update, gpu_multicrop -- plus the same warmup / freeze / clip / EMA-cap
    logic, so a winning trial's hyperparameters drop straight into train_dino.py.
  * Runs each trial as a PROXY_STEPS (default 4000) pretrain on the GPU patch cache
    (mmap'd once, shared across trials), then a capped EuroSAT linear probe scored on a
    validation split of the train side (never the test split), maximizing probe accuracy.
  * Searches the knobs that actually move DINO quality on a small backbone + small corpus:
    lr, weight_decay, out_dim, tau_teacher (target), tau_student, tau_teacher_warmup_steps,
    freeze_last_layer_steps, ema_momentum_start, n_local, batch_size.

Reports a plateau/collapse penalty: a trial whose teacher entropy ends below a floor is
scored down even if its probe number looks acceptable, since that recipe will collapse when
run longer.

Usage (on a GPU node):
    PROXY_STEPS=4000 python optuna_dino.py --trials 40 \
        --seco-root ../../../data/seco/seasonal_contrast_100k \
        --manifest  ../../../data/seco/manifest.txt \
        --patch-cache ../../../data/seco/patch_cache_r128 --preload-res 128
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_dino as dino_mod  # noqa: E402
from eurosat_probe import hpo_probe_accuracy  # noqa: E402
from gpu_aug import LocationShuffler  # noqa: E402
from pretrain_utils import (  # noqa: E402
    _validate_corpus_args,
    add_dataset_args,
    build_patch_cache,
    dataset_norm,
)

from tutorial_rs import build_vit_s8, get_device, seed_everything  # noqa: E402

PROXY_STEPS = int(os.environ.get("PROXY_STEPS", 4000))
PROBE_SUBSET_PER_CLASS = 200
ENTROPY_FLOOR = float(os.environ.get("ENTROPY_FLOOR", 0.75))  # below this -> collapse penalty
device = get_device()

_cache = {}


def get_cache(args):
    """Decoded patch cache for the selected corpus, built once and shared across trials."""
    if "cache" not in _cache:
        _cache["cache"], _ = build_patch_cache(args, device)
    return _cache["cache"]


def dino_proxy(trial, args):
    lr = trial.suggest_float("lr", 3e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    out_dim = trial.suggest_categorical("out_dim", [1024, 2048, 4096])
    tau_teacher = trial.suggest_float("tau_teacher", 0.04, 0.09)
    tau_student = trial.suggest_float("tau_student", 0.1, 0.2)
    warmup_steps = trial.suggest_int("tau_teacher_warmup_steps", 500, 4000, step=500)
    freeze_steps = trial.suggest_int("freeze_last_layer_steps", 500, 4000, step=500)
    ema_start = trial.suggest_float("ema_momentum_start", 0.99, 0.999)
    n_local = trial.suggest_int("n_local", 2, 6)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_global = 2

    seed_everything(42)
    cache = get_cache(args)
    sample_gen = torch.Generator(device=cache.device); sample_gen.manual_seed(42)
    aug_gen = torch.Generator(device=device); aug_gen.manual_seed(43)
    shuffler = LocationShuffler(len(cache), batch_size, cache.device, generator=sample_gen)
    mean, std = dataset_norm(args, device)

    student = dino_mod.DINOModel(build_vit_s8(), out_dim=out_dim).to(device)
    teacher = dino_mod.DINOModel(build_vit_s8(), out_dim=out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_STEPS, eta_min=lr / 50)
    center = torch.zeros(1, out_dim, device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ema_end, ema_max = 0.9995, 0.9998
    tau_warm0 = 0.04
    crop_counts = SimpleNamespace(n_global=n_global, n_local=n_local)

    student.train(); teacher.eval()
    last_entropy = 0.0
    for step in range(1, PROXY_STEPS + 1):
        loc_idx = shuffler.next_batch()
        raw = cache.sample_views(loc_idx, n_views=1, generator=sample_gen)
        crops = dino_mod.gpu_multicrop(raw[0].to(device, non_blocking=True), crop_counts, mean, std,
                                        aug_gen)

        if step <= warmup_steps:
            tau_t = tau_warm0 + (step / max(1, warmup_steps)) * (tau_teacher - tau_warm0)
        else:
            tau_t = tau_teacher

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            s_logits = [student(c) for c in crops]
            with torch.no_grad():
                t_logits = [teacher(c) for c in crops[:n_global]]
        s_logits = [s.float() for s in s_logits]
        t_logits = [t.float() for t in t_logits]
        loss = dino_mod.dino_loss(s_logits, t_logits, center, tau_student, tau_t)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        if step <= freeze_steps:
            student.head.cancel_last_layer_grad()
        scaler.step(opt); scaler.update(); sched.step()

        m = ema_end - (ema_end - ema_start) * (1 + math.cos(step / PROXY_STEPS * math.pi)) / 2
        dino_mod.ema_update(student, teacher, m=min(m, ema_max))
        center = dino_mod.update_center(center, t_logits, momentum=0.9)

        if step % 200 == 0 or step == PROXY_STEPS:
            with torch.no_grad():
                tp = torch.softmax((t_logits[0] - center) / tau_t, dim=-1)
                last_entropy = -(tp * tp.clamp_min(1e-8).log()).sum(-1).mean().item()
            trial.report(last_entropy, step)

    acc = hpo_probe_accuracy(teacher.encoder, device, pool="cls", per_class=PROBE_SUBSET_PER_CLASS)
    trial.set_user_attr("final_entropy", last_entropy)
    trial.set_user_attr("raw_probe_acc", acc)
    # Collapse penalty: an acceptable probe number with a near-degenerate teacher is a recipe
    # that will fail when run to 30k steps. Penalize proportionally to how far below the floor.
    if last_entropy < ENTROPY_FLOOR:
        acc -= 0.5 * (ENTROPY_FLOOR - last_entropy) / ENTROPY_FLOOR
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seco-root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--patch-cache", default=None)
    ap.add_argument("--preload-res", type=int, default=128)
    ap.add_argument("--preload-workers", type=int, default=16)
    ap.add_argument("--cache-device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--study-name", default="dino_proxy_v2")
    ap.add_argument("--storage", default=None,
                    help="Optional optuna storage URL (e.g. sqlite:///dino_optuna.db) for resume.")
    add_dataset_args(ap)
    args = ap.parse_args()
    _validate_corpus_args(args)

    print(f"Device: {device} | trials={args.trials} | proxy_steps={PROXY_STEPS} | "
          f"entropy_floor={ENTROPY_FLOOR}")
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=args.storage, load_if_exists=args.storage is not None,
    )
    study.optimize(lambda t: dino_proxy(t, args), n_trials=args.trials, show_progress_bar=False)

    print("\n=== BEST ===")
    print("value (penalized probe acc):", study.best_value)
    b = study.best_trial
    print("raw probe acc:", b.user_attrs.get("raw_probe_acc"))
    print("final entropy:", b.user_attrs.get("final_entropy"))
    print("params:", study.best_params)
    out = os.path.join(os.path.dirname(__file__), "optuna_best_dino.txt")
    with open(out, "w") as f:
        f.write(f"best_value {study.best_value}\n")
        f.write(f"raw_probe_acc {b.user_attrs.get('raw_probe_acc')}\n")
        f.write(f"final_entropy {b.user_attrs.get('final_entropy')}\n")
        for k, v in study.best_params.items():
            f.write(f"{k} {v}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
