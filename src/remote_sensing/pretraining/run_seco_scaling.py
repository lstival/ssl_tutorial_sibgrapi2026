"""
SeCo corpus-scaling study: does more UNLABELED pretraining data produce a better encoder?

Trains the contrastive (SimCLR-style) recipe at several pretraining-corpus sizes with the
compute budget held FIXED (same steps, same batch size, same Optuna-tuned hyperparameters),
then linear-probes each resulting encoder on EuroSAT in both label regimes.

Why fixed compute. The tutorial's existing contrastive checkpoint saw 20k steps x 256 batch =
5.12M views drawn from only 10k patches -- roughly 512 epochs over a tiny set. Holding steps
fixed while growing the corpus turns "more data" into the single variable: at the full 100k
corpus the same budget is ~51 epochs, so any accuracy change is attributable to seeing more
distinct scenes rather than to more gradient updates.

The corpora are NESTED (each fraction's locations are a prefix of a single fixed shuffle), so
the curve measures adding data to a set rather than resampling a different one each time.

All runs share --preload-res, which must not vary across the curve: the preload resolution
changes the source resolution that RandomResizedCrop samples from, and letting it differ per
point would confound corpus size with resampling quality.

Usage:
    python run_seco_scaling.py --seco-root "G:/seco_full/seasonal_contrast_100k"
    python run_seco_scaling.py --seco-root ... --mechanism mae     # same curve for MAE (norm-pix)
    python run_seco_scaling.py --seco-root ... --mechanism dino
    python run_seco_scaling.py --seco-root ... --fractions 0.01 0.10 --steps 5000   # smoke test
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from seco_data import list_locations  # noqa: E402

PYTHON = sys.executable
# Roughly log-spaced over 1.7 decades: 2k / 5k / 10k / 30k / 100k patches. The floor is set by
# the batch size, not by taste -- a manifest with fewer locations than one batch yields no
# batches at all (train_contrastive.py rejects it), and shrinking the batch instead would change
# the number of InfoNCE negatives and break comparability across the curve. At 20k total
# locations and batch 256, 0.02 (400 locations) is the smallest safe point.
DEFAULT_FRACTIONS = [0.02, 0.05, 0.10, 0.30, 1.00]

# mechanism -> (training script, linear-probe readout). The readouts match rs_eval_results.json.
MECHANISMS = {
    "contrastive": ("train_contrastive.py", "cls"),
    "mae": ("train_mae.py", "mean"),
    "dino": ("train_dino.py", "cls"),
}
FIGURES_DIR = os.path.join(HERE, "..", "..", "..", "notebooks", "remote_sensing", "figures")


def default_out_json(mechanism):
    # The contrastive curve predates the other two and keeps its original file name.
    name = "seco_scaling_results.json" if mechanism == "contrastive" else f"seco_scaling_{mechanism}_results.json"
    return os.path.join(FIGURES_DIR, name)


def tag_for(fraction):
    return f"f{int(round(fraction * 100)):03d}"


def build_nested_manifests(seco_root, fractions, out_dir, seed=42):
    """
    Write one manifest per fraction, each a prefix of the same fixed shuffle of all locations,
    so that smaller corpora are strict subsets of larger ones.
    """
    all_locations = list_locations(seco_root)
    n_total = len(all_locations)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_total)
    shuffled = [all_locations[i] for i in order]

    os.makedirs(out_dir, exist_ok=True)
    manifests = {}
    for fraction in fractions:
        n_keep = max(1, int(round(n_total * fraction)))
        kept = sorted(shuffled[:n_keep])
        path = os.path.join(out_dir, f"manifest_scaling_{tag_for(fraction)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for loc in kept:
                f.write(loc + "\n")
        manifests[fraction] = (path, n_keep)
        print(f"  {tag_for(fraction)}: {n_keep:>6d}/{n_total} locations (~{n_keep * 5:>6d} patches) -> {path}")
    return manifests


def run(cmd, log_path):
    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(cmd)} =====\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}); see {log_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seco-root", required=True, help="Extracted seasonal_contrast_100k directory (full corpus).")
    ap.add_argument("--mechanism", default="contrastive", choices=sorted(MECHANISMS),
                    help="SSL recipe to scale. Each trainer keeps its Optuna-tuned hyperparameters; "
                         "only steps, batch size and preload resolution are pinned by this script.")
    ap.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    ap.add_argument("--steps", type=int, default=20000, help="Fixed step budget for every corpus size.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--preload-res", type=int, default=128,
                    help="Preload resolution shared by every run (see module docstring).")
    ap.add_argument("--patch-cache", default=None,
                    help="Path for the shared decoded-patch cache (default: next to --seco-root). "
                         "Every fraction is a subset of the same corpus, so the first run decodes "
                         "the .tif files once (~5 min) and the rest load the blob in seconds. "
                         "Pass 'none' to decode separately in each run.")
    ap.add_argument("--ckpt-root", default=os.path.join(HERE, "..", "..", "..", "artifacts", "remote_sensing", "checkpoints", "seco_scaling"))
    ap.add_argument("--out-json", default=None,
                    help="Results file (default: notebooks/remote_sensing/figures/seco_scaling[_<mechanism>]_results.json).")
    ap.add_argument("--log", default=None, help="Training log (default: data/seco_scaling[_<mechanism>].log).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip a fraction whose checkpoint already exists.")
    args = ap.parse_args()

    train_script, pool = MECHANISMS[args.mechanism]
    if args.out_json is None:
        args.out_json = default_out_json(args.mechanism)
    if args.log is None:
        suffix = "" if args.mechanism == "contrastive" else f"_{args.mechanism}"
        args.log = os.path.join(HERE, "..", "..", "..", "data", f"seco_scaling{suffix}.log")

    fractions = sorted(args.fractions)
    print(f"SeCo scaling study | mechanism={args.mechanism} | fractions={fractions} | steps={args.steps} (fixed) | "
          f"batch={args.batch_size} | preload_res={args.preload_res}")

    manifest_dir = os.path.join(os.path.dirname(os.path.abspath(args.seco_root)), "scaling_manifests")
    print("\nBuilding nested manifests:")
    manifests = build_nested_manifests(args.seco_root, fractions, manifest_dir)

    # Preflight the whole curve before spending hours on the first point: a fraction too small
    # for one batch would otherwise fail only when its turn came, after the earlier runs finished.
    too_small = {tag_for(f): n for f, (_, n) in manifests.items() if n < args.batch_size}
    if too_small:
        raise SystemExit(
            f"These fractions yield fewer locations than --batch-size ({args.batch_size}), which "
            f"produces no batches: {too_small}. Raise the fraction or lower the batch size (but "
            f"lower it for EVERY point, since batch size sets the InfoNCE negative count)."
        )

    os.makedirs(os.path.abspath(args.ckpt_root), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)

    # Build the shared patch cache here, over the LARGEST fraction, rather than letting the
    # training runs build it lazily. The store is only reusable by a run whose locations it
    # already covers, so a lazy build starting from the smallest fraction would re-decode (and
    # rewrite) a growing cache at every point -- about 1.5 corpora of decoding instead of one.
    patch_cache = args.patch_cache
    if patch_cache is None:
        patch_cache = os.path.join(os.path.dirname(os.path.abspath(args.seco_root)),
                                   f"patch_cache_r{args.preload_res}")
    if patch_cache != "none":
        largest_manifest, n_largest = manifests[fractions[-1]]
        print(f"\nPreparing shared patch cache at {patch_cache}.u8 "
              f"({n_largest} locations)...")
        from gpu_aug import SeCoPatchCache  # noqa: E402  (imported here: torch is a heavy import)
        with open(largest_manifest, encoding="utf-8") as f:
            largest_locations = [line.strip() for line in f if line.strip()]
        SeCoPatchCache(args.seco_root, largest_locations, res=args.preload_res,
                       cache_file=patch_cache, workers=os.cpu_count() or 8)
        del largest_locations

    for fraction in fractions:
        tag = tag_for(fraction)
        manifest, n_loc = manifests[fraction]
        # Each run gets its own directory: train_contrastive.py writes its JSONL training log
        # next to --out in append mode, so a shared directory would concatenate all five runs'
        # logs into one file (exactly the confusion documented for the MAE logs).
        run_dir = os.path.join(os.path.abspath(args.ckpt_root), tag)
        os.makedirs(run_dir, exist_ok=True)
        ckpt = os.path.join(run_dir, f"{args.mechanism}.pt")

        if args.skip_existing and os.path.isfile(ckpt):
            print(f"\n[{tag}] checkpoint exists, skipping training.")
        else:
            print(f"\n[{tag}] training on {n_loc} locations (~{n_loc * 5} patches), {args.steps} steps...")
            t0 = time.time()
            cmd = [
                PYTHON, os.path.join(HERE, train_script),
                "--seco-root", args.seco_root,
                "--manifest", manifest,
                "--out", ckpt,
                "--steps", str(args.steps),
                "--batch-size", str(args.batch_size),
                "--preload-res", str(args.preload_res),
            ]
            if patch_cache != "none":
                cmd += ["--patch-cache", patch_cache]
            run(cmd, args.log)
            print(f"[{tag}] training done in {(time.time() - t0) / 60:.1f} min")

        print(f"[{tag}] evaluating...")
        meta = json.dumps({
            "fraction": fraction, "locations": n_loc, "patches": n_loc * 5,
            "steps": args.steps, "batch_size": args.batch_size,
            "preload_res": args.preload_res,
            "epochs_equivalent": round(args.steps * args.batch_size / (n_loc * 5), 1),
        })
        run([
            PYTHON, os.path.join(HERE, "eval_encoder.py"),
            "--checkpoint", ckpt,
            "--name", tag,
            "--pool", pool,
            "--out-json", os.path.abspath(args.out_json),
            "--meta", meta,
        ], args.log)

    print(f"\nAll done. Results: {os.path.abspath(args.out_json)}")
    if os.path.isfile(args.out_json):
        with open(args.out_json, encoding="utf-8") as f:
            results = json.load(f)
        print(f"\n{'run':>6s} {'patches':>9s} {'epochs':>7s} {'full':>7s} " +
              " ".join(f"k={k:<4d}" for k in (10, 20, 50, 100, 200)))
        for name in sorted(results):
            e = results[name]
            m = e.get("meta", {})
            few = e["few_label"]
            print(f"{name:>6s} {m.get('patches', '?'):>9} {m.get('epochs_equivalent', '?'):>7} "
                  f"{e['full_label']:>7.4f} " + " ".join(f"{few[str(k)]:.4f}" for k in (10, 20, 50, 100, 200)))


if __name__ == "__main__":
    main()
