"""
Small shared helpers for the pretraining scripts in this folder (train_contrastive.py,
train_mae.py, train_dino.py). Kept separate from src/remote_sensing/tutorial_rs.py: this module is
training-loop plumbing specific to large-scale offline pretraining (checkpoint saving,
logging, resuming), not something the notebooks themselves need.
"""

import json
import os
import sys
import time

import torch

# src/remote_sensing/tutorial_rs.py is the single source of truth for the model architecture (ViT-S/8,
# TransformerBlock, ...). We import it directly rather than duplicating the architecture here,
# so a change to the backbone only ever needs to happen in one place.
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)


def save_encoder_checkpoint(encoder, out_path, extra=None):
    """
    Save only the encoder's state dict (not any pretraining-only head/decoder), matching what
    src/remote_sensing/tutorial_rs.py's `try_load_checkpoint` expects to load into a bare `build_vit_s8()`
    in the tutorial notebooks and in Notebook 4's comparative evaluation.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(encoder.state_dict(), out_path)
    if extra is not None:
        meta_path = os.path.splitext(out_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2)
    print(f"Saved encoder checkpoint to {out_path}")


def save_train_state(path, *, step, model, optimizer, scheduler, extra=None):
    """
    Save a *complete* training checkpoint (model + optimizer + scheduler + step), separate from
    the encoder-only checkpoint that the notebooks consume. This is what makes `--resume` faithful:
    unlike save_encoder_checkpoint (encoder weights only), it captures everything needed to
    continue the run exactly where it left off. Written to a `.train.pt` sidecar next to `out`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if extra is not None:
        state["extra"] = extra
    # Write to a temp file then atomically replace, so an interrupt mid-write can't corrupt the
    # resume checkpoint (the old one stays intact until the new one is fully flushed).
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_train_state(path, model, optimizer, scheduler, device):
    """
    Restore a checkpoint written by save_train_state. Returns the step to resume *after*
    (i.e. the loop should continue from step+1). Returns 0 if no checkpoint exists.
    """
    if not os.path.isfile(path):
        return 0
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"Resumed training state from {path} at step {state['step']}")
    return state["step"]


class RunningLogger:
    """Minimal step/loss logger: prints periodically and writes a JSONL history file."""

    def __init__(self, log_path, print_every=50):
        self.log_path = log_path
        self.print_every = print_every
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")

    def log(self, step, **metrics):
        record = {"step": step, "elapsed_sec": round(time.time() - self.start_time, 1), **metrics}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        if step % self.print_every == 0:
            metric_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items())
            print(f"[step {step:6d} | {record['elapsed_sec']:8.1f}s] {metric_str}")

    def close(self):
        self._f.close()


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
    import torch as _torch
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
        cache_device = _torch.device(args.cache_device)
    cache.to(cache_device)
    return cache, cache_device
