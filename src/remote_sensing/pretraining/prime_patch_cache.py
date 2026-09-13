"""
Decode a pretraining corpus into its on-disk patch cache once, then exit.

The three trainers (train_contrastive/mae/dino) each build the same `--patch-cache` on
startup if it does not exist. Running them concurrently against a not-yet-built cache makes
them all decode in parallel (wasteful) and race on the cache file write. Run this first as a
single job, then launch the three with the cache already present -- each then just mmaps it in
seconds.

Usage (SeCo):
    python prime_patch_cache.py --dataset seco \
        --seco-root ../../../data/seco/seasonal_contrast_100k \
        --manifest  ../../../data/seco/manifest.txt \
        --preload-res 128 --patch-cache ../../../data/seco/patch_cache_r128

Usage (BigEarthNet):
    python prime_patch_cache.py --dataset bigearthnet \
        --ben-root ../../../data/bigearthnet \
        --preload-res 120 --patch-cache ../../../data/bigearthnet/patch_cache_r120
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pretrain_utils import add_dataset_args, build_patch_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seco-root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--preload-res", type=int, default=None)
    ap.add_argument("--preload-workers", type=int, default=16)
    ap.add_argument("--patch-cache", required=True,
                    help="Path (a .u8 blob + .json index) to build. Same value the trainers pass.")
    ap.add_argument("--cache-device", default="cpu", choices=("auto", "cuda", "cpu"),
                    help="Where to place the cache after decoding. 'cpu' here (no GPU needed to "
                         "just build the file).")
    add_dataset_args(ap)
    args = ap.parse_args()

    # build_patch_cache decodes and, because --patch-cache is set, writes the .u8/.json store.
    import torch

    cache, _ = build_patch_cache(args, torch.device("cpu"))
    print(f"Patch cache ready: {args.patch_cache}.u8 ({cache.nbytes / 1e9:.2f} GB, "
          f"{len(cache)} entries)")


if __name__ == "__main__":
    main()
