"""
Shared utilities for the SIBGRAPI 2026 SSL Tutorial — Hour 2 (Remote Sensing / EuroSAT).

This module centralizes the plumbing that every notebook in `notebooks/remote_sensing/` needs, so that
each notebook can stay focused on the SSL mechanism it teaches:

  - deterministic EuroSAT RGB loading + a fixed stratified 80/20 split (no official split exists)
  - remote-sensing-specific data augmentations (see note in `build_rs_augmentations`)
  - a shared ViT-Small/8 backbone (patch 8, 64x64 input -> 8x8 token grid, embed dim 384)
  - plotting helpers with a consistent visual style
  - a checkpoint / precomputed-output downloader ("Found pretrained model, loading...")

Every notebook downloads this file at the top (Colab) or imports it directly (local run).
"""

import os
import random
import shutil
import sys
import urllib.request
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import EuroSAT

# Failures that mean "this host is unreachable / refused", as opposed to a local problem
# (a bad permission, a half-extracted directory) that a different mirror would not fix.
# (HTTPError subclasses URLError; socket.timeout is TimeoutError since Python 3.10.)
DOWNLOAD_ERRORS = (URLError, TimeoutError, ConnectionError)

# =====================================================================================
# Reproducibility & device
# =====================================================================================

SEED = 42


def seed_everything(seed=SEED):
    """Seed python, numpy and torch (CPU+CUDA) for reproducible splits/augmentations/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# =====================================================================================
# EuroSAT dataset, statistics and fixed split
# =====================================================================================

# Precomputed per-channel statistics of EuroSAT RGB (train split), used for normalization.
# Registering them here (rather than recomputing per-notebook) keeps every notebook consistent
# with the normalization used during offline pretraining on the large Sentinel-2 corpus (see
# docs/remote_sensing_implementation_plan.md, section 3.1) — encoder-frozen evaluation requires the same
# normalization at pretraining and probing time.
EUROSAT_MEAN = (0.3444, 0.3803, 0.4078)
EUROSAT_STD = (0.2037, 0.1366, 0.1148)

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

IMG_SIZE = 64  # native EuroSAT resolution; kept fixed everywhere to avoid pos-embed interpolation


# torchvision hard-codes a single EuroSAT host (madm.dfki.de) that is frequently unreachable.
# These mirrors serve the same RGB archive and are tried in order when it is down.
EUROSAT_MIRRORS = [
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
]


def load_eurosat(dataset_path, transform=None):
    """
    Download (if needed) and return the full EuroSAT RGB dataset (27,000 images, 10 classes).

    Already-downloaded data is reused untouched. If the dataset is missing *and* every host is
    unreachable, this raises with instructions for populating `dataset_path` by hand, rather
    than surfacing a bare connection timeout, because a dead mirror is the single most common
    reason this tutorial fails to start on a fresh machine.
    """
    try:
        return EuroSAT(root=dataset_path, download=True, transform=transform)
    except DOWNLOAD_ERRORS as first_error:
        # Only reachability failures fall through to the mirrors. A PermissionError or a
        # half-extracted directory is a local problem and must surface as itself, not be
        # relabeled "mirror unavailable".
        print(f"Default EuroSAT host failed ({type(first_error).__name__}: {first_error}).")
        for mirror in EUROSAT_MIRRORS:
            try:
                print(f"  trying mirror {mirror} ...")
                torchvision.datasets.utils.download_and_extract_archive(
                    mirror, download_root=os.path.join(dataset_path, "eurosat")
                )
                return EuroSAT(root=dataset_path, download=False, transform=transform)
            except DOWNLOAD_ERRORS as mirror_error:  # noqa: PERF203 - reported per mirror
                print(f"  mirror unavailable: {type(mirror_error).__name__}: {mirror_error}")

        raise RuntimeError(
            "Could not download EuroSAT from any known host.\n"
            f"Copy an existing copy of the dataset to: {os.path.abspath(dataset_path)}\n"
            "so that this path exists and holds the 10 class folders:\n"
            f"    {os.path.abspath(os.path.join(dataset_path, 'eurosat', '2750'))}\n"
            "The archive is also mirrored at https://github.com/phelber/EuroSAT ."
        ) from first_error


def stage_eurosat_locally(local_path, drive_path):
    """
    Colab only: put EuroSAT on the VM's local disk and return `local_path`.

    Reading 27,000 small JPEGs one by one through the Google Drive mount is slow enough to
    starve the GPU (and makes a CPU-only run look like it takes a day). So the images live on
    local disk, and Drive only keeps the single ~90 MB zip, so a new session copies one file
    instead of downloading again. The zip is saved to Drive the first time it is downloaded.
    """
    local_zip = os.path.join(local_path, "eurosat", "EuroSAT.zip")
    drive_zip = os.path.join(drive_path, "eurosat", "EuroSAT.zip")
    if not os.path.isdir(os.path.join(local_path, "eurosat", "2750")):
        if os.path.isfile(drive_zip):
            print(f"Copying the cached EuroSAT zip from Drive to {local_path} ...")
            os.makedirs(os.path.dirname(local_zip), exist_ok=True)
            shutil.copyfile(drive_zip, local_zip)
            torchvision.datasets.utils.extract_archive(local_zip, os.path.dirname(local_zip))
        else:
            load_eurosat(local_path)
    if os.path.isfile(local_zip) and not os.path.isfile(drive_zip):
        os.makedirs(os.path.dirname(drive_zip), exist_ok=True)
        shutil.copyfile(local_zip, drive_zip)
        print(f"Cached the EuroSAT zip on Drive at {drive_zip}")
    return local_path


def check_colab_gpu(device):
    """Stop early on a Colab runtime without a GPU, where training runs ~100x slower."""
    if "google.colab" in sys.modules and device.type != "cuda":
        raise RuntimeError(
            "This Colab runtime has no GPU, so training would take many hours.\n"
            "Go to Runtime > Change runtime type, pick a GPU (e.g. T4), then run all cells again."
        )


def stratified_split(dataset, test_size=0.2, seed=SEED):
    """
    Fixed stratified 80/20 split by class index. EuroSAT (torchvision) ships no official
    train/test split, so we build one deterministically (seed=42) and reuse it everywhere:
    the SSL pretraining pool (labels discarded), the linear-probe train set (labels used),
    and the held-out test set all derive from this same split.

    Returns (train_indices, test_indices) as sorted numpy arrays.
    """
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    train_idx = np.sort(np.array(train_idx))
    test_idx = np.sort(np.array(test_idx))
    return train_idx, test_idx


# =====================================================================================
# Remote-sensing-specific augmentations
# =====================================================================================


class ContrastiveTransformations:
    """Applies `base_transforms` independently `n_views` times to the same image (SimCLR-style)."""

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


def build_rs_augmentations(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD):
    """
    Data augmentation pipeline for satellite imagery.

    This deliberately differs from the standard natural-image recipe used for datasets like
    STL10, and that difference is a teaching point (see Notebook 0):

      - Nadir (top-down) satellite views have **no canonical orientation** — north-up is a
        cartographic convention, not a visual one. So arbitrary rotations and BOTH horizontal
        and vertical flips are valid, label-preserving transformations here (a vertical flip
        of a natural photo of a dog is usually a bad augmentation; for a nadir crop of a field
        it is perfectly fine).
      - Bands carry physical, radiometric meaning. Aggressive color jitter (as used for
        natural images) distorts that meaning, so we use only *mild* brightness/contrast
        jitter and drop hue/saturation distortion and grayscale.
      - No perspective/shear/large-angle distortions: satellite patches are already
        near-orthorectified, so these would introduce geometry the sensor never produces.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_eval_transform(img_size=IMG_SIZE, mean=EUROSAT_MEAN, std=EUROSAT_STD):
    """Deterministic transform (no augmentation) used for linear probing / evaluation / viz."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def denormalize(img_tensor, mean=EUROSAT_MEAN, std=EUROSAT_STD):
    """Invert `Normalize` for visualization. Accepts a (C,H,W) tensor, returns (C,H,W) in [0,1]."""
    mean_t = torch.tensor(mean, device=img_tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=img_tensor.device).view(-1, 1, 1)
    return (img_tensor * std_t + mean_t).clamp(0, 1)


# =====================================================================================
# ViT-Small/8 backbone (shared by all three SSL mechanisms)
# =====================================================================================
#
# Design choice (see docs/remote_sensing_implementation_plan.md, section 2): a single ViT-S/8 backbone
# is shared across the contrastive, masking and distillation notebooks. Patch size 8 on a
# 64x64 input gives an 8x8 = 64-token grid, which keeps the linear-probe comparison in
# Notebook 4 fair (identical architecture and token budget across all three encoders) and
# gives DINO's attention maps a readable spatial grid.


class PatchEmbed(nn.Module):
    """Splits a (B,C,H,W) image into non-overlapping patches and linearly projects each to `embed_dim`."""

    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block: MHSA + MLP, both with residual connections."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=return_attn,
                                            average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


class ViTEncoder(nn.Module):
    """
    ViT-Small/8 encoder shared by all SSL mechanisms in this tutorial.

    Defaults (ViT-S): embed_dim=384, depth=6, num_heads=6. We use depth=6 rather than the
    standard 12 to keep live training in the tutorial fast on a single Colab GPU; the encoder
    is still expressive enough to show clear qualitative SSL signal at EuroSAT scale.

    A learnable [CLS] token is prepended (used as the global representation for linear
    probing and contrastive projection, and as the query for DINO's attention maps). Learnable
    absolute position embeddings are added since input resolution is fixed at 64x64 everywhere.
    """

    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_chans=3, embed_dim=384,
                 depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_attn=False):
        """
        Full forward pass (no masking): patchify -> prepend [CLS] -> add pos-embed -> blocks.
        Returns token sequence (B, 1+num_patches, embed_dim); if `return_attn`, also returns
        the last block's attention weights (B, num_heads, 1+num_patches, 1+num_patches).
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed

        attn_weights = None
        for i, block in enumerate(self.blocks):
            is_last = i == len(self.blocks) - 1
            if return_attn and is_last:
                tokens, attn_weights = block(tokens, return_attn=True)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)

        if return_attn:
            return tokens, attn_weights
        return tokens

    def forward_features(self, x, pool="cls"):
        """
        Return a single global embedding (B, embed_dim) for linear probing.

        pool="cls"  -> the [CLS] token (default; what contrastive/DINO train directly).
        pool="mean" -> mean over the patch tokens (excludes [CLS]). MAE never supervises the
                       [CLS] token -- its objective lives on the patch tokens -- so a [CLS] probe
                       badly understates an MAE encoder; mean-pooling the patch tokens is the
                       standard way to read out an MAE representation.
        """
        tokens = self.forward(x)
        if pool == "mean":
            return tokens[:, 1:].mean(dim=1)
        return tokens[:, 0]


def build_vit_s8(**kwargs):
    """Factory for the shared ViT-S/8 backbone with the tutorial's default hyperparameters."""
    defaults = dict(img_size=IMG_SIZE, patch_size=8, in_chans=3, embed_dim=384, depth=6, num_heads=6)
    defaults.update(kwargs)
    return ViTEncoder(**defaults)


def build_vit_t8(**kwargs):
    """
    Factory for a ViT-Tiny/8 backbone: embed_dim=192, depth=4, num_heads=3 (~1.8M parameters
    against ViT-S/8's ~10.7M).

    Same patch size and the same fixed 64x64 input as `build_vit_s8`, so a Tiny encoder drops
    into the downstream EuroSAT probes and fine-tunes with no position-embedding interpolation
    -- only the embedding width changes. Used when pretraining on the full 100k SeCo corpus,
    where the extra capacity should go into the corpus rather than the encoder.
    """
    defaults = dict(img_size=IMG_SIZE, patch_size=8, in_chans=3, embed_dim=192, depth=4, num_heads=3)
    defaults.update(kwargs)
    return ViTEncoder(**defaults)


# =====================================================================================
# Checkpoint / precomputed-output downloader
# =====================================================================================

# The pretrained encoders are published as assets on a GitHub Release (tag
# `WEIGHTS_RELEASE_TAG` below), not committed to the repository. Release assets have no
# bandwidth quota for public repos -- unlike Git LFS's 1 GB/month free tier, which is what
# previously made these downloads start 404ing after only a few clones or Colab runs.
#
# Two ways a notebook finds them, in this order:
#   1. On disk, relative to the notebook -- the local-clone path, used if the weights were
#      fetched there already (e.g. by tools/release_weights.sh or a manual download).
#   2. Downloaded from the release over HTTPS -- the Colab path, where only the .ipynb is
#      present.
GITHUB_REPO = "lstival/ssl_tutorial_sibgrapi2026"
WEIGHTS_RELEASE_TAG = "weights-v1"
CHECKPOINT_REPO_PATH = "artifacts/remote_sensing/checkpoints"
CHECKPOINT_BASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{WEIGHTS_RELEASE_TAG}/"
)

# Where a local clone keeps the weights, relative to a notebook in notebooks/remote_sensing/.
LOCAL_CHECKPOINT_DIRS = (
    os.path.join("..", "..", CHECKPOINT_REPO_PATH),
    CHECKPOINT_REPO_PATH,
)


def _looks_like_lfs_pointer(file_path):
    """
    True if `file_path` is a Git LFS pointer file rather than real tensor data.

    A leftover from before the weights moved to a GitHub Release: a clone from before that
    change may still have LFS pointer files on disk (if git-lfs wasn't installed) rather than
    the tensors themselves. A pointer is a small text file that starts with a known version
    line; catching it here turns a confusing torch.load unpickling error into a clear message.
    """
    try:
        if os.path.getsize(file_path) > 1024:
            return False
        with open(file_path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def resolve_local_checkpoint(file_name, checkpoint_path=None):
    """
    Return the path to an already-present copy of `file_name`, or None.

    Checks the caller's own checkpoint directory first, then the in-repo location, so a local
    clone never re-downloads a weight it already has committed.
    """
    candidates = []
    if checkpoint_path:
        candidates.append(os.path.join(checkpoint_path, file_name))
    candidates.extend(os.path.join(d, file_name) for d in LOCAL_CHECKPOINT_DIRS)
    for path in candidates:
        if os.path.isfile(path) and not _looks_like_lfs_pointer(path):
            return path
    return None


def download_files(file_names, checkpoint_path, base_url=CHECKPOINT_BASE_URL):
    """
    Make each file in `file_names` available inside `checkpoint_path`.

    A file already present in the clone is reused as-is. Otherwise it is downloaded from this
    repository (the Colab path). Missing files are reported but never raise: every notebook
    falls back to a short live-training run when a checkpoint cannot be obtained.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    for file_name in file_names:
        file_path = os.path.join(checkpoint_path, file_name)
        if "/" in file_name:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.isfile(file_path):
            if not _looks_like_lfs_pointer(file_path):
                continue
            # A pointer file is worse than nothing: remove it and fetch the real bytes.
            print(f"{file_path} is a Git LFS pointer, not the weights. Re-downloading...")
            os.remove(file_path)

        local = resolve_local_checkpoint(file_name)
        if local:
            print(f"Using checkpoint from the local clone: {local}")
            continue

        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            if _looks_like_lfs_pointer(file_path):
                os.remove(file_path)
                print(f"  Got a Git LFS pointer instead of {file_name} from {file_url}.")
        except HTTPError as e:
            print(
                f"Could not download {file_name} ({e}). "
                f"This notebook will fall back to live training instead."
            )
        except Exception as e:
            print(f"Could not download {file_name}: {e}")


def try_load_checkpoint(model, checkpoint_path, file_name, device, base_url=CHECKPOINT_BASE_URL):
    """
    Attempt to download + load a state dict into `model`. Returns True if a pretrained
    checkpoint was found and loaded ("Found pretrained model, loading..."), False
    if none is available (caller should train live instead).
    """
    download_files([file_name], checkpoint_path, base_url=base_url)
    file_path = resolve_local_checkpoint(file_name, checkpoint_path)
    if file_path:
        print(f"Found pretrained model at {file_path}, loading...")
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        return True
    print(f"No pretrained checkpoint found for {file_name}. Train live below instead.")
    return False


# =====================================================================================
# Plotting helpers
# =====================================================================================


def setup_plotting():
    """Consistent plotting defaults across the tutorial notebooks.

    Light background with a soft grid, so figures stay readable both in a
    notebook and when projected on a screen during the tutorial.
    """
    import matplotlib

    plt.set_cmap("cividis")
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except ImportError:
        matplotlib.rcParams["axes.grid"] = True

    matplotlib.rcParams.update({
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.6",
        "axes.linewidth": 0.8,
        "grid.color": "0.88",
        "grid.linewidth": 0.8,
        "image.cmap": "cividis",
    })


def show_image_grid(images, nrow=8, title=None, figsize=(10, 5)):
    """Display a batch of (already normalized-for-display, i.e. in [0,1]) image tensors as a grid."""
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, pad_value=0.9)
    grid = grid.permute(1, 2, 0).detach().cpu().numpy()
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
    plt.close()


def plot_curve(values, xlabel, ylabel, title=None, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_embedding_scatter(embedding_2d, labels, class_names=EUROSAT_CLASSES, title=None, figsize=(7, 6)):
    """2D scatter (t-SNE/UMAP output) colored by class, with a legend."""
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        plt.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            s=8, color=cmap(cls_idx % 10), label=cls_name, alpha=0.7,
        )
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def cls_attention_map(encoder, image_chw, head="mean", device=None):
    """
    Extract the [CLS] -> patch attention of `encoder`'s last block for a single image.

    Returns a (grid_size, grid_size) numpy array of raw attention weights (they sum to <= 1
    over the patches; the remainder is CLS attending to itself).

    `head` selects which of the attention heads to read:
      - "mean"    average over heads. Safe default, but heads specialize a lot: on our DINO
                  checkpoint the per-head concentration ranges from ~0.8 to ~5.3 (coefficient
                  of variation), so averaging blends sharp heads into flat ones and washes
                  most of the structure out.
      - "sharpest" the single head with the highest coefficient of variation for this image,
                  i.e. the most spatially selective one. This is what DINO's own figures show.
      - int       that specific head index.

    The (grid_size, grid_size) reshape is row-major, matching `PatchEmbed.forward`'s
    `flatten(2).transpose(1, 2)`; this orientation is verified by a planted-patch test (put a
    bright square in patch (r, c), and the attention argmax lands on (r, c)).
    """
    device = device or next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        _, attn = encoder(image_chw.unsqueeze(0).to(device), return_attn=True)
    if was_training:
        encoder.train()

    cls_attn = attn[0, :, 0, 1:]  # (num_heads, num_patches): CLS query -> patch keys
    if head == "mean":
        selected = cls_attn.mean(dim=0)
    elif head == "sharpest":
        concentration = cls_attn.std(dim=-1) / cls_attn.mean(dim=-1).clamp_min(1e-12)
        selected = cls_attn[int(concentration.argmax())]
    else:
        selected = cls_attn[int(head)]

    g = encoder.grid_size
    return selected.reshape(g, g).float().cpu().numpy()


def _stretch_attention(attn_map, low_pct=5.0, high_pct=85.0):
    """
    Map raw attention to [0,1] for display using percentile clipping instead of min-max.

    CLS attention is heavy-tailed: one or two spike patches take most of the mass, so plain
    min-max normalization pins the whole rest of the grid near 0 and the overlay reads as a
    flat dark wash (empirically only ~2-11% of patches land above 0.5). Clipping at
    percentiles spends the color range on the part of the distribution that actually varies.

    The p5-p85 default was chosen against this checkpoint: it puts ~17-47% of patches above
    mid-scale on structured scenes. Note the top percentile has to be well below 99 -- on a
    64-patch grid the 99th percentile is essentially the max, so a p20-p99 clip is barely
    different from min-max and leaves the figure just as flat.

    This is a contrast stretch, not a rank normalization: a genuinely uniform map (e.g. open
    water) still renders as uniform rather than being forced to look structured.
    """
    lo, hi = np.percentile(attn_map, low_pct), np.percentile(attn_map, high_pct)
    if hi <= lo:
        return np.zeros_like(attn_map)
    return np.clip((attn_map - lo) / (hi - lo), 0.0, 1.0)


def show_attention_grid(images_chw, attn_maps, titles=None, upsample_to=None,
                        cmap="inferno", alpha=0.6, ncols=6, low_pct=5.0, high_pct=85.0):
    """
    Show a row of images above their [CLS] attention overlays, as one compact figure.

    `images_chw` are (C,H,W) tensors already de-normalized to [0,1]; `attn_maps` are the raw
    (grid, grid) maps from `cls_attention_map`. Each map is percentile-stretched (see
    `_stretch_attention`) and upsampled bilinearly, which reads better than `nearest` on an
    8x8 grid without inventing detail beyond the patch resolution.

    Two rows -- raw image on top, overlay below -- so the attention can be compared against
    what is actually in the scene, with a shared colorbar labeling low/high attention.
    """
    n = min(len(images_chw), ncols)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2), squeeze=False)

    for j in range(n):
        img_np = images_chw[j].permute(1, 2, 0).detach().cpu().numpy()
        size = upsample_to or img_np.shape[0]

        stretched = _stretch_attention(attn_maps[j], low_pct, high_pct)
        attn_t = torch.tensor(stretched, dtype=torch.float32)[None, None]
        attn_up = F.interpolate(attn_t, size=(size, size), mode="bilinear",
                                align_corners=False)[0, 0].numpy()

        axes[0][j].imshow(img_np)
        axes[0][j].axis("off")
        if titles is not None:
            axes[0][j].set_title(titles[j], fontsize=8)

        axes[1][j].imshow(img_np)
        im = axes[1][j].imshow(attn_up, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
        axes[1][j].axis("off")

    axes[0][0].set_ylabel("image", fontsize=8)
    axes[1][0].set_ylabel("attention", fontsize=8)

    fig.subplots_adjust(right=0.9, hspace=0.05, wspace=0.05)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["low", "high"])
    cbar.set_label("[CLS] attention", fontsize=8)
    plt.show()
    plt.close()
