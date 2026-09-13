import matplotlib
matplotlib.use('Agg')

## Standard libraries
import math
import os
import urllib.request

## Imports for plotting
import matplotlib.pyplot as plt
import numpy as np

## tqdm for loading bars
from tqdm.auto import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data

## scikit-learn for the linear probe + confusion matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# %% cell boundary

# Make the shared tutorial module importable.
# Local clone: it lives in src/remote_sensing/. Colab: download it next to this notebook.
import os, sys, urllib.request

_LOCAL_SRC = os.path.join("..", "..", "src", "remote_sensing")
if os.path.isfile(os.path.join(_LOCAL_SRC, "tutorial_rs.py")):
    sys.path.insert(0, _LOCAL_SRC)
elif not os.path.isfile("tutorial_rs.py"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/lstival/ssl_tutorial_sibgrapi2026/main/src/remote_sensing/tutorial_rs.py", "tutorial_rs.py")

from tutorial_rs import (
    seed_everything, get_device, setup_plotting,
    EUROSAT_CLASSES, EUROSAT_MEAN, EUROSAT_STD, IMG_SIZE,
    load_eurosat, stratified_split, build_eval_transform, build_rs_augmentations,
    build_vit_s8, try_load_checkpoint,
)

setup_plotting()
seed_everything(42)
device = get_device()
print("Device:", device)

DATASET_PATH = "../../data"
CHECKPOINT_PATH = "../../artifacts/remote_sensing/checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# %% cell boundary

CHECKPOINT_FILES = {
    "contrastive": "contrastive_vit_s8.pt",
    "mae": "mae_vit_s8.pt",
    "dino": "dino_vit_s8.pt",
    # Written by Notebook 1's live-training cell. Same mechanism and same random start as
    # "contrastive" above, but 400 steps instead of 20,000 -- the compute-budget comparison.
    "contrastive (notebook)": "contrastive_vit_s8_notebook.pt",
}

# Checkpoints produced by running the notebooks locally, rather than downloaded. Their absence
# is expected and not an error: it just means that notebook has not been run in this workspace.
OPTIONAL = {"contrastive (notebook)"}

encoders = {}
missing = []
for family, file_name in CHECKPOINT_FILES.items():
    encoder = build_vit_s8().to(device)
    if family in OPTIONAL:
        # Produced locally by a notebook, never downloaded: load straight from disk.
        local_path = os.path.join(CHECKPOINT_PATH, file_name)
        found = os.path.isfile(local_path)
        if found:
            encoder.load_state_dict(torch.load(local_path, map_location=device))
            print(f"Found local notebook checkpoint at {local_path}, loading...")
        else:
            print(f"Optional checkpoint not found, skipping: {file_name} "
                  f"(run Notebook 1 to create it)")
    else:
        found = try_load_checkpoint(encoder, CHECKPOINT_PATH, file_name, device)
    if found:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        encoders[family] = encoder
    elif family not in OPTIONAL:
        missing.append(family)

print(f"Loaded encoders: {list(encoders.keys())}")
if missing:
    print(f"Missing checkpoints (skipping in comparisons below): {missing}")
    print("Run Notebooks 1-3 to produce local checkpoints, or host the pretrained weights "
          "referenced in notebooks_implementation_plan.md Section 13.")


# %% cell boundary

eval_transform = build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
full_dataset = load_eurosat(DATASET_PATH, transform=eval_transform)
train_idx, test_idx = stratified_split(full_dataset, test_size=0.2, seed=42)

train_subset = data.Subset(full_dataset, train_idx)
test_subset = data.Subset(full_dataset, test_idx)

# How to read out each frozen encoder as a single global vector for probing. MAE never
# supervises its [CLS] token (its objective lives on the patch tokens), so a [CLS] probe
# badly understates it; mean-pooling the patch tokens is the standard MAE readout.
# Contrastive and DINO both train the [CLS] token directly, so they use it.
POOL_BY_FAMILY = {"contrastive": "cls", "mae": "mean", "dino": "cls",
                  "contrastive (notebook)": "cls"}


@torch.no_grad()
def prepare_data_features(encoder, dataset, pool="cls", batch_size=256):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, targets in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        h = encoder.forward_features(imgs, pool=pool)
        feats.append(h.cpu())
        labels.append(targets)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


features = {}
for family, encoder in encoders.items():
    pool = POOL_BY_FAMILY.get(family, "cls")
    train_feats, train_labels = prepare_data_features(encoder, train_subset, pool=pool)
    test_feats, test_labels = prepare_data_features(encoder, test_subset, pool=pool)
    features[family] = {
        "train_feats": train_feats, "train_labels": train_labels,
        "test_feats": test_feats, "test_labels": test_labels,
    }
    print(f"{family}: train {train_feats.shape}, test {test_feats.shape} (pool={pool})")

# %% cell boundary

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def linear_probe_accuracy(train_feats, train_labels, test_feats, test_labels, max_iter=2000):
    # Standardize features before logistic regression: ViT embeddings have very different
    # per-dim scales, and unscaled features make lbfgs converge slowly (or not at all).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter),
    )
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    return accuracy_score(test_labels, preds), clf, preds


linear_probe_results = {}
for family, feat_dict in features.items():
    acc, clf, preds = linear_probe_accuracy(
        feat_dict["train_feats"], feat_dict["train_labels"],
        feat_dict["test_feats"], feat_dict["test_labels"],
    )
    linear_probe_results[family] = {"accuracy": acc, "clf": clf, "preds": preds}
    print(f"{family:>12s}: linear-probe test accuracy = {acc:.4f}")

# %% cell boundary

# Random-init encoder baseline (no training at all, pure random projection + linear probe).
#
# We probe the random encoder under BOTH readouts. A random ViT is already a surprisingly
# strong random feature extractor on EuroSAT (~0.79), so this baseline is what every SSL
# number below must be judged against, and the comparison is only meaningful if the random
# encoder is read out the same way as the encoder it is being compared to. MAE is probed with
# pool="mean" and the other two with pool="cls" (see POOL_BY_FAMILY), so a single cls-only
# random baseline would silently compare MAE against the wrong reference.
seed_everything(123)
random_encoder = build_vit_s8().to(device)
random_encoder.eval()
for p in random_encoder.parameters():
    p.requires_grad = False

acc_random_by_pool = {}
for pool in ("cls", "mean"):
    tr_f, tr_y = prepare_data_features(random_encoder, train_subset, pool=pool)
    te_f, te_y = prepare_data_features(random_encoder, test_subset, pool=pool)
    acc_random_by_pool[pool], _, _ = linear_probe_accuracy(tr_f, tr_y, te_f, te_y)
    print(f"{'random-init':>12s} (pool={pool:>4s}): linear-probe test accuracy = {acc_random_by_pool[pool]:.4f}")

# Headline random baseline for the bar chart: the cls readout, matching contrastive/DINO.
acc_random = acc_random_by_pool["cls"]

# The number that actually matters per family: how much did pretraining add over a random
# encoder read out the SAME way? This is the honest measure of what the objective bought.
print("\nGain over random-init (same readout):")
for family, res in linear_probe_results.items():
    pool = POOL_BY_FAMILY.get(family, "cls")
    gain = res["accuracy"] - acc_random_by_pool[pool]
    print(f"  {family:>12s} (pool={pool:>4s}): {res['accuracy']:.4f} - {acc_random_by_pool[pool]:.4f} "
          f"= {gain:+.4f}")

# %% cell boundary

# Supervised from-scratch baseline: end-to-end training with labels, no SSL pretraining.
#
# Tutorial budget: 15 epochs, batch 512, mixed precision. The recipe still matters more than the
# epoch count. An earlier attempt (30 epochs, constant lr=1e-3, no schedule) only reached 0.869:
# a constant learning rate for a from-scratch ViT-S/8 oscillates near a plateau instead of
# settling. With warmup -> cosine decay (the same fix the SSL pretraining scripts needed), the
# short schedule lands within ~0.01 of the 50-epoch run at roughly a fifth of the wall time.
# The point of this baseline is a fair reference for the frozen probes, so it gets the same
# optimization care as the SSL encoders it is compared against.
aug_transform = build_rs_augmentations(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD)
full_dataset_aug = load_eurosat(DATASET_PATH, transform=aug_transform)
train_subset_aug = data.Subset(full_dataset_aug, train_idx)

# Data loading, not the GPU, is the bottleneck at this model size: EuroSAT decodes and augments
# on the CPU, so a few workers keep the GPU fed.
NUM_WORKERS = 4 if os.name != "nt" else 0


def train_supervised_baseline(num_epochs=15, lr=2e-3, weight_decay=1e-4, batch_size=512,
                              warmup_epochs=2):
    seed_everything(42)
    model = build_vit_s8().to(device)
    head = nn.Linear(model.pos_embed.shape[-1], len(EUROSAT_CLASSES)).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = data.DataLoader(train_subset_aug, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=NUM_WORKERS,
                             pin_memory=use_amp, persistent_workers=NUM_WORKERS > 0)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1.0 / 50) + (1.0 - 1.0 / 50) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss, epoch_n = 0.0, 0
        for imgs, labels in tqdm(loader, leave=False, desc=f"epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = head(model.forward_features(imgs))
                loss = criterion(logits, labels)
            epoch_loss += loss.item() * imgs.size(0)
            epoch_n += imgs.size(0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        print(f"  epoch {epoch+1}/{num_epochs}: loss={epoch_loss/epoch_n:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=use_amp)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = head(model.forward_features(imgs)).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


acc_supervised = train_supervised_baseline(num_epochs=50, lr=1e-3, warmup_epochs=5)
print(f"{'supervised':>12s}: end-to-end test accuracy = {acc_supervised:.4f}")


# %% cell boundary

def subsample_per_class(labels, k, seed=42):
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:k])
    return np.array(indices)


LABEL_BUDGETS = [10, 20, 50, 100, 200]

few_label_results = {family: [] for family in features}
for family, feat_dict in features.items():
    for k in LABEL_BUDGETS:
        sub_idx = subsample_per_class(feat_dict["train_labels"], k)
        acc, _, _ = linear_probe_accuracy(
            feat_dict["train_feats"][sub_idx], feat_dict["train_labels"][sub_idx],
            feat_dict["test_feats"], feat_dict["test_labels"],
        )
        few_label_results[family].append(acc)
        print(f"{family:>12s} | k={k:4d} labels/class: acc={acc:.4f}")

# %% cell boundary

if few_label_results:
    plt.figure(figsize=(6, 4))
    for family, accs in few_label_results.items():
        plt.plot(LABEL_BUDGETS, accs, marker="o", label=family)
    plt.xlabel("Labeled examples per class")
    plt.ylabel("Test accuracy")
    plt.title("Few-label linear-probe accuracy by SSL family")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
else:
    print("No encoders available. See the checkpoint-loading cell above.")

# %% cell boundary

all_results = {family: res["accuracy"] for family, res in linear_probe_results.items()}
all_results["random-init"] = acc_random
all_results["supervised"] = acc_supervised

if all_results:
    families_sorted = sorted(all_results, key=lambda k: all_results[k])
    plt.figure(figsize=(7, 4))
    plt.barh(families_sorted, [all_results[f] for f in families_sorted])
    plt.xlabel("Test accuracy (full train split)")
    plt.title("Linear-probe accuracy by pretraining family")
    plt.tight_layout()
    plt.show()
    plt.close()
else:
    print("No results available. See the checkpoint-loading cell above.")

# %% cell boundary

for family, res in linear_probe_results.items():
    cm = confusion_matrix(features[family]["test_labels"], res["preds"], normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=0, vmax=1)
    ax.set_xticks(range(len(EUROSAT_CLASSES)))
    ax.set_yticks(range(len(EUROSAT_CLASSES)))
    ax.set_xticklabels(EUROSAT_CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(EUROSAT_CLASSES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {family}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    plt.close()