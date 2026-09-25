"""
Shared helpers for the remote-sensing pretraining / evaluation scripts in this folder: corpus
selection (SeCo vs BigEarthNet), the GPU augmentation source, the encoder registry, and the
modality-agnostic training plumbing re-exported from src/common/train_utils.py. Kept separate
from src/remote_sensing/tutorial_rs.py, which the notebooks download as a single file.
"""

import os
import sys

import torch

# Two import roots, both inserted here so every script in this folder gets them by importing
# this module: src/remote_sensing (tutorial_rs.py -- the single source of truth for the ViT
# architecture) and src/common (modality-agnostic training plumbing shared with time series).
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _path in (SRC_DIR, os.path.join(SRC_DIR, "..", "common")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train_utils import (  # noqa: E402,F401 -- re-exported for the scripts in this folder
    RunningLogger,
    enable_fast_cuda,
    load_train_state,
    save_encoder_checkpoint,
    save_train_state,
    train_log_path,
    warmup_cosine_lambda,
)

from tutorial_rs import build_vit_s8, build_vit_t8  # noqa: E402

# Encoder variants selectable with --arch. Both are patch-8 at the fixed 64x64 input, so a
# checkpoint from either loads into the downstream notebooks without pos-embed interpolation.
ARCHS = {"vit_s8": build_vit_s8, "vit_t8": build_vit_t8}
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def resolve_locations(seco_root, manifest_path=None):
    """
    Return the list of SeCo location IDs to use: from a manifest file written by
    download_seco.py if provided/found, otherwise every location folder under `seco_root`.
    """
    from seco_data import list_locations, read_manifest

    if manifest_path and os.path.isfile(manifest_path):
        locations = read_manifest(manifest_path)
        print(f"Using {len(locations)} locations from manifest {manifest_path}")
    else:
        locations = list_locations(seco_root)
        print(f"No manifest found at {manifest_path}; using all {len(locations)} locations under {seco_root}")
    return locations


# --- pretraining corpus selection (SeCo vs BigEarthNet) --------------------------------
# The three trainers share one GPU-augmentation path (gpu_aug.SeCoPatchCache + LocationShuffler).
# add_dataset_args / build_patch_cache / dataset_norm let that path run on either corpus with a
# single --dataset switch, without each trainer growing its own copy of the dispatch.

def add_dataset_args(parser):
    """Register the corpus-selection flags shared by all three trainers."""
    parser.add_argument("--dataset", choices=("seco", "bigearthnet"), default="seco",
                        help="Pretraining corpus. 'seco' uses --seco-root/--manifest; "
                             "'bigearthnet' uses --ben-root/--ben-manifest.")
    parser.add_argument("--ben-root", type=str, default=None,
                        help="BigEarthNet download root (the dir download_bigearthnet.py wrote "
                             "s2_root.txt / manifest.txt into). Required with --dataset bigearthnet.")
    parser.add_argument("--ben-manifest", type=str, default=None,
                        help="Optional explicit BigEarthNet manifest.txt (defaults to "
                             "<ben-root>/manifest.txt).")


def _validate_corpus_args(args):
    """Fail fast with a clear message if the selected corpus is missing its required paths."""
    ds = getattr(args, "dataset", "seco")
    if ds == "bigearthnet":
        if not args.ben_root:
            raise SystemExit("--dataset bigearthnet requires --ben-root.")
        if getattr(args, "no_gpu_aug", False):
            raise SystemExit("--dataset bigearthnet currently supports only the GPU-augmentation "
                             "path; drop --no-gpu-aug (or run on a CUDA node).")
    else:
        if not args.seco_root:
            raise SystemExit("--dataset seco requires --seco-root.")


def dataset_norm(args, device):
    """Return (mean, std) normalization tensors for the selected corpus."""
    from gpu_aug import norm_constants

    if getattr(args, "dataset", "seco") == "bigearthnet":
        from bigearthnet_data import BEN_MEAN, BEN_STD
        return norm_constants(device, mean=BEN_MEAN, std=BEN_STD)
    return norm_constants(device)


def build_patch_cache(args, device):
    """
    Build the decoded uint8 patch cache for the selected corpus and place it on the right
    device. Returns (cache, cache_device). The cache object exposes the SeCoPatchCache
    interface either way (see bigearthnet_data.BigEarthNetPatchCache).
    """
    from gpu_aug import choose_cache_device

    if getattr(args, "dataset", "seco") == "bigearthnet":
        from bigearthnet_data import BigEarthNetPatchCache, read_manifest
        if not args.ben_root:
            raise SystemExit("--dataset bigearthnet requires --ben-root.")
        manifest = args.ben_manifest or os.path.join(args.ben_root, "manifest.txt")
        if not os.path.isfile(manifest):
            raise SystemExit(f"BigEarthNet manifest not found: {manifest} "
                             f"(run download_bigearthnet.py first).")
        patch_names = read_manifest(manifest)
        print(f"Using {len(patch_names)} BigEarthNet patches from {manifest}")
        cache = BigEarthNetPatchCache(args.ben_root, patch_names, res=args.preload_res,
                                      workers=args.preload_workers, cache_file=args.patch_cache)
    else:
        from gpu_aug import SeCoPatchCache
        locations = resolve_locations(args.seco_root, args.manifest)
        cache = SeCoPatchCache(args.seco_root, locations, res=args.preload_res,
                               workers=args.preload_workers, cache_file=args.patch_cache)

    if args.cache_device == "auto":
        cache_device = choose_cache_device(cache.nbytes, device)
    else:
        cache_device = torch.device(args.cache_device)
    cache.to(cache_device)
    return cache, cache_device


def build_gpu_source(args, device):
    """
    GPU augmentation source shared by the three trainers: the decoded corpus as one uint8 tensor
    plus a shuffled location stream, replacing the DataLoader entirely (see gpu_aug.py).

    Returns (cache, shuffler, sample_gen, aug_gen, mean, std). Two generators because the two
    stages live on different devices when the cache is in host RAM: index sampling happens
    wherever the cache is, augmentation always on the GPU.
    """
    from gpu_aug import LocationShuffler

    cache, cache_device = build_patch_cache(args, device)
    sample_gen = torch.Generator(device=cache_device)
    sample_gen.manual_seed(args.seed)
    aug_gen = torch.Generator(device=device)
    aug_gen.manual_seed(args.seed + 1)

    shuffler = LocationShuffler(len(cache), args.batch_size, cache_device, generator=sample_gen)
    mean, std = dataset_norm(args, device)
    return cache, shuffler, sample_gen, aug_gen, mean, std
