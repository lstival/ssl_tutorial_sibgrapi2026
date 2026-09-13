"""
Download the BigEarthNet-S2 archive and build a flat RGB patch manifest for SSL pretraining.

BigEarthNet (Sumbul et al., 2019; reBEN / BigEarthNet v2, Clasen et al., 2024) is a
Sentinel-2 land-cover benchmark: ~590k non-overlapping 120x120 patches over 10 European
countries, each labelled with a multi-label subset of the 19-class CORINE nomenclature.

Why BigEarthNet as a pretraining corpus for the EuroSAT downstream task:

  * Same sensor and bands (Sentinel-2, B02/B03/B04 -> RGB) as EuroSAT -- no spectral or
    resolution domain gap, unlike SeCo's contrast-stretched previews.
  * The scene distribution *is* land cover: BigEarthNet was built to span the CORINE classes,
    which overlap EuroSAT's 10 classes almost entirely (forest, crops, pasture, water,
    urban, ...). SeCo's locations are uncurated and skew to coastline / ocean / ice.
  * ~590k patches vs SeCo-100k's ~100k locations -- ~6x more views, all at 120x120 native
    (bigger than the 64x64 the ViT-S/8 trains at, so RandomResizedCrop has real headroom).

This script downloads the tarball, extracts it, enumerates every patch directory, and writes
`bigearthnet/manifest.txt` (one patch-directory path per line, relative to the extracted
root). Pretraining is label-free, so the CSV split / label parquet are not needed here --
only the S2 rasters. Pass `--fraction` to keep a seeded random subset of patches on disk.

The BigEarthNet-S2 v2 archive is distributed by TU Berlin / RSiM at
https://bigearth.net and mirrored on Zenodo (record 10891137). Because hosting URLs and
checksums for this dataset have changed several times, they are NOT hard-coded here: pass
`--url` (and optionally `--md5`) explicitly, or point `--root` at an already-downloaded and
extracted copy and pass `--skip-download`.

Usage:
    # already have BigEarthNet-S2 extracted somewhere:
    python download_bigearthnet.py --root /path/to/bigearthnet --skip-download

    # download from an explicit mirror URL:
    python download_bigearthnet.py --root ../../../data/bigearthnet \\
        --url https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst \\
        --fraction 1.0
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import time

import numpy as np
import requests
from tqdm import tqdm

# The BigEarthNet-S2 v2 / reBEN archive on Zenodo record 10891137 (~63 GB compressed,
# ~120 GB extracted, ~590k patch directories). Used as the default when neither --url nor
# --skip-download is given. If this stops resolving, pass a working --url or --skip-download.
DEFAULT_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"

# Directory names inside common BigEarthNet-S2 archive layouts. The v2/reBEN archive extracts
# to "BigEarthNet-S2/"; the original v1 to "BigEarthNet-v1.0/". Either is accepted.
KNOWN_S2_DIRNAMES = ("BigEarthNet-S2", "BigEarthNet-v1.0", "BigEarthNet")
# A BigEarthNet-S2 patch directory holds one GeoTIFF per band, named "<patch>_B02.tif" etc.
RGB_BAND_SUFFIXES = ("B04", "B03", "B02")  # R, G, B (Sentinel-2 10 m true-colour)


def md5sum(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, md5=None, max_retries=10):
    """Resumable download of a large archive (HTTP Range), mirroring download_seco.py."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.isfile(dest) and md5 and md5sum(dest) == md5:
        print(f"Found valid archive at {dest}, skipping download.")
        return dest

    print(f"Downloading {url} -> {dest}")
    for attempt in range(1, max_retries + 1):
        have = os.path.getsize(dest) if os.path.isfile(dest) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if have and r.status_code == 200:
                    have, mode = 0, "wb"  # server ignored Range
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + have
                with open(dest, mode) as f, tqdm(
                    total=total or None, initial=have, unit="B", unit_scale=True
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            break
        except (requests.RequestException, OSError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Download failed after {max_retries} attempts.") from exc
            wait = min(60, 2 ** attempt)
            print(f"\nAttempt {attempt} failed ({exc}); retrying in {wait}s...")
            time.sleep(wait)

    if md5:
        print("Verifying checksum...")
        if md5sum(dest) != md5:
            raise RuntimeError(f"Checksum mismatch for {dest}; delete it and re-run.")
        print("Checksum OK.")
    return dest


def extract(archive, root):
    """Extract a .tar, .tar.gz or .tar.zst archive into `root`."""
    os.makedirs(root, exist_ok=True)
    existing = _find_s2_root(root)
    if existing:
        print(f"Found existing extracted BigEarthNet-S2 at {existing}, skipping extraction.")
        return existing

    print(f"Extracting {archive} -> {root}")
    if archive.endswith(".zst"):
        # tarfile has no native zstd; stream through the zstd CLI (present on the cluster).
        try:
            proc = subprocess.Popen(["zstd", "-dc", archive], stdout=subprocess.PIPE)
        except FileNotFoundError:
            sys.exit("Archive is .zst but the `zstd` CLI is not on PATH. "
                     "module load zstd, or extract manually and pass --skip-download.")
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            tf.extractall(root)
        proc.wait()
    else:
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            for m in tqdm(members, unit="file"):
                tf.extract(m, root)

    s2_root = _find_s2_root(root)
    if not s2_root:
        sys.exit(f"Extraction finished but no BigEarthNet-S2 directory found under {root}.")
    return s2_root


def _find_s2_root(root):
    """
    Locate the BigEarthNet-S2 collection directory under `root` (or `root` itself). This is the
    directory named "BigEarthNet-S2" (or the archive's equivalent), not an individual patch dir.
    Patch directories may be one OR two levels below it -- the reBEN v2 archive nests them as
    BigEarthNet-S2/<tile-scene>/<patch_id>/, the v1 archive as BigEarthNet-v1.0/<patch_id>/.
    """
    if _has_patches_within(root):
        return root
    for name in KNOWN_S2_DIRNAMES:
        cand = os.path.join(root, name)
        if os.path.isdir(cand) and _has_patches_within(cand):
            return cand
    if os.path.isdir(root):
        for child in sorted(os.listdir(root)):
            cand = os.path.join(root, child)
            if not os.path.isdir(cand):
                continue
            if _has_patches_within(cand):
                return cand
            for name in KNOWN_S2_DIRNAMES:
                deep = os.path.join(cand, name)
                if os.path.isdir(deep) and _has_patches_within(deep):
                    return deep
    return None


def _has_patches_within(path, max_depth=2):
    """True if a BigEarthNet patch directory exists within `max_depth` levels of `path`."""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    for entry in entries:
        sub = os.path.join(path, entry)
        if not os.path.isdir(sub):
            continue
        if _is_patch_dir(sub):
            return True
        if max_depth > 1 and _has_patches_within(sub, max_depth - 1):
            return True
    return False


def _is_patch_dir(path):
    """True if `path` holds the RGB band GeoTIFFs of a BigEarthNet-S2 patch."""
    try:
        files = os.listdir(path)
    except OSError:
        return False
    return all(any(f.endswith(f"_{b}.tif") or f.endswith(f"_{b}.tiff") for f in files)
               for b in RGB_BAND_SUFFIXES)


def enumerate_patches(s2_root):
    """
    Return sorted patch paths (relative to s2_root, POSIX-style) for every patch directory
    with all three RGB bands. Walks up to two levels deep so both the reBEN v2 (tile/patch)
    and v1 (patch) layouts work.
    """
    patches = []
    top = sorted(os.listdir(s2_root))
    for entry in tqdm(top, desc="Scanning patches", unit="dir"):
        sub = os.path.join(s2_root, entry)
        if not os.path.isdir(sub):
            continue
        if _is_patch_dir(sub):
            patches.append(entry)
            continue
        for child in sorted(os.listdir(sub)):
            deep = os.path.join(sub, child)
            if os.path.isdir(deep) and _is_patch_dir(deep):
                patches.append(f"{entry}/{child}")
    if not patches:
        sys.exit(f"No valid BigEarthNet-S2 patch directories under {s2_root}.")
    return patches


def subsample(s2_root, patches, fraction, seed):
    """Keep a seeded random `fraction` of patches, deleting the rest from disk."""
    if fraction >= 1.0:
        return patches
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(round(len(patches) * fraction)))
    keep_idx = set(rng.choice(len(patches), size=n_keep, replace=False).tolist())
    keep = [p for i, p in enumerate(patches) if i in keep_idx]
    keep_set = set(keep)
    print(f"Keeping {len(keep)}/{len(patches)} patches ({fraction:.0%}, seed={seed}).")
    import shutil
    for p in tqdm(patches, desc="Pruning", unit="dir"):
        if p not in keep_set:
            shutil.rmtree(os.path.join(s2_root, p), ignore_errors=True)
    return keep


def write_manifest(patches, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in patches:
            f.write(p + "\n")
    print(f"Wrote manifest of {len(patches)} patches to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Directory to download/extract into (or an existing extracted copy).")
    ap.add_argument("--url", default=None,
                    help=f"Archive URL. Defaults to the Zenodo reBEN mirror ({DEFAULT_URL}) "
                         f"unless --skip-download is set.")
    ap.add_argument("--md5", default=None, help="Optional MD5 of the archive.")
    ap.add_argument("--skip-download", action="store_true", help="Use an already-extracted copy under --root.")
    ap.add_argument("--archive-name", default="BigEarthNet-S2.tar.zst", help="Local filename for the downloaded archive.")
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of patches to keep on disk (default: 1.0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-archive", action="store_true", help="Do not delete the archive after extraction.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)

    if args.skip_download:
        s2_root = _find_s2_root(root)
        if not s2_root:
            sys.exit(f"--skip-download given but no BigEarthNet-S2 directory found under {root}.")
        archive = None
    else:
        url = args.url or DEFAULT_URL
        archive_path = (args.archive_name if os.path.isabs(args.archive_name)
                        else os.path.join(root, args.archive_name))
        archive = download(url, archive_path, md5=args.md5)
        s2_root = extract(archive, root)

    patches = enumerate_patches(s2_root)
    patches = subsample(s2_root, patches, args.fraction, args.seed)
    write_manifest(patches, os.path.join(root, "manifest.txt"))

    # Record where the patch directories actually live, so the trainer needs only --root.
    with open(os.path.join(root, "s2_root.txt"), "w", encoding="utf-8") as f:
        f.write(s2_root + "\n")

    if archive and not args.keep_archive and os.path.isfile(archive):
        print(f"Removing archive {archive} (pass --keep-archive to retain).")
        os.remove(archive)

    print(f"\nDone. {len(patches)} BigEarthNet-S2 patches ready at {s2_root}")


if __name__ == "__main__":
    main()
