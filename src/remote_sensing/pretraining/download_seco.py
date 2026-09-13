"""
Downloads the SeCo (Seasonal Contrast) pretraining corpus and keeps only a random subset of
its locations on disk.

SeCo (Mañas et al., 2021, "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote
Sensing Data") is distributed as a fixed archive on Zenodo -- no Google Earth Engine account
needed, unlike the raw collection pipeline in the original repo
(https://github.com/ServiceNow/seasonal-contrast). We use the "100k" version: ~100,000
Sentinel-2 locations around the world, each with up to 5 seasonal revisits, hosted at
https://zenodo.org/records/4728033.

On-disk layout after extraction (matches torchgeo's SeasonalContrastS2 dataset):

    seasonal_contrast_100k/
        000000/
            <season_dir>/
                B1.tif ... B12.tif
        000001/
            ...

This script downloads the archive once, extracts it, then deletes every location folder that
falls outside a random `--fraction` subset (default 10%), so a full local mirror of the 100k
corpus is never kept around -- only the subset actually used for pretraining.

Usage:
    python download_seco.py --root ../../../data/seco --fraction 0.10 --seed 42
"""

import argparse
import hashlib
import os
import shutil
import time
import zipfile

import numpy as np
import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/4728033/files/seco_100k.zip?download=1"
ZENODO_MD5 = "ebf2d5e03adc6e657f9a69a20ad863e0"
ARCHIVE_NAME = "seco_100k.zip"
# Exact byte length of the Zenodo archive, used to tell a resumable truncation apart from a
# complete-but-corrupt file (only the latter needs a restart from zero).
ZENODO_SIZE = 7302001636
# Location folders in the full seco_100k release (5 seasonal revisits each -> 100k patches).
# Used to tell a complete extraction apart from the remains of a --fraction < 1.0 run.
FULL_CORPUS_LOCATIONS = 20000
EXTRACTED_DIRNAME = "seasonal_contrast_100k"


def md5sum(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(root, checksum=True, max_retries=10):
    """
    Download seco_100k.zip into `root` with a progress bar, unless already present and valid.

    The transfer is resumable. A 7.3 GB single-shot download over a flaky link frequently dies
    part-way; without resume, every retry restarts from zero and may never finish. Zenodo honors
    HTTP Range on GET (it answers 206 even though a HEAD advertises no `accept-ranges`), so an
    interrupted run continues from the bytes already on disk instead of re-fetching them.
    """
    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, ARCHIVE_NAME)

    if os.path.isfile(archive_path):
        have = os.path.getsize(archive_path)
        if have == ZENODO_SIZE:
            if not checksum or md5sum(archive_path) == ZENODO_MD5:
                print(f"Found existing archive at {archive_path}, skipping download.")
                return archive_path
            # Full length but wrong content: genuinely corrupt, so resuming cannot repair it.
            print("Existing archive is complete but failed checksum, re-downloading from scratch.")
            os.remove(archive_path)
        elif have > ZENODO_SIZE:
            print(f"Existing archive is larger than expected ({have} > {ZENODO_SIZE}), restarting.")
            os.remove(archive_path)
        else:
            print(f"Found partial archive ({have / 1e9:.2f} / {ZENODO_SIZE / 1e9:.2f} GB), resuming.")

    print(f"Downloading {ZENODO_URL} -> {archive_path}")
    print("This is a multi-GB download from Zenodo; it only needs to happen once.")

    for attempt in range(1, max_retries + 1):
        have = os.path.getsize(archive_path) if os.path.isfile(archive_path) else 0
        if have >= ZENODO_SIZE:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        try:
            with requests.get(ZENODO_URL, stream=True, timeout=60, headers=headers) as r:
                if have and r.status_code == 200:
                    # Server ignored the Range header and is sending the whole file: start over
                    # rather than appending a second copy onto the bytes we already have.
                    print("Server ignored Range request; restarting from byte 0.")
                    have, mode = 0, "wb"
                elif have and r.status_code != 206:
                    r.raise_for_status()
                r.raise_for_status()

                with open(archive_path, mode) as f, tqdm(
                    total=ZENODO_SIZE, initial=have, unit="B", unit_scale=True
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except (requests.RequestException, OSError) as exc:
            got = os.path.getsize(archive_path) if os.path.isfile(archive_path) else 0
            if attempt == max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts ({got / 1e9:.2f} / "
                    f"{ZENODO_SIZE / 1e9:.2f} GB fetched). Re-run to resume from here."
                ) from exc
            wait = min(60, 2 ** attempt)
            print(f"\nAttempt {attempt} failed ({exc}); {got / 1e9:.2f} GB on disk. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)
            continue

        got = os.path.getsize(archive_path)
        if got >= ZENODO_SIZE:
            break
        # A clean EOF that is still short of the full length is a silent truncation, which is
        # exactly how the first run of this download failed. Treat it as retryable.
        print(f"\nStream ended early at {got / 1e9:.2f} / {ZENODO_SIZE / 1e9:.2f} GB; resuming...")
        time.sleep(2)

    got = os.path.getsize(archive_path)
    if got != ZENODO_SIZE:
        raise RuntimeError(
            f"Incomplete download: {got} of {ZENODO_SIZE} bytes. Re-run to resume."
        )

    if checksum:
        print("Verifying checksum...")
        if md5sum(archive_path) != ZENODO_MD5:
            raise RuntimeError(
                f"Checksum mismatch for {archive_path}. The download may be corrupted; "
                "delete the file and re-run this script."
            )
        print("Checksum OK.")

    return archive_path


def extract_archive(archive_path, root):
    """Extract seco_100k.zip into `root` if not already extracted."""
    extracted_path = os.path.join(root, EXTRACTED_DIRNAME)
    if os.path.isdir(extracted_path) and os.listdir(extracted_path):
        # The skip is only safe if the directory holds the *full* corpus. A previous run with
        # --fraction < 1.0 prunes locations in place, so a later, larger --fraction would skip
        # extraction, then "keep" 100% of the already-pruned subset and report a corpus size
        # that is silently 10x too small. Refuse instead of producing a mislabeled corpus.
        n_present = sum(
            1 for d in os.listdir(extracted_path) if os.path.isdir(os.path.join(extracted_path, d))
        )
        if n_present < FULL_CORPUS_LOCATIONS:
            raise SystemExit(
                f"{extracted_path} holds {n_present} locations, fewer than the full corpus "
                f"({FULL_CORPUS_LOCATIONS}) -- it is the leftover of an earlier --fraction run. "
                f"Extraction would be skipped and the subset silently reported as complete. "
                f"Move or delete that directory first, then re-run."
            )
        print(f"Found existing extracted directory at {extracted_path}, skipping extraction.")
        return extracted_path

    print(f"Extracting {archive_path} -> {root}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, unit="file"):
            zf.extract(member, root)

    return extracted_path


def subsample_locations(extracted_path, fraction, seed):
    """
    Keep only a random `fraction` of location subfolders under `extracted_path`, deleting the
    rest. Returns the sorted list of kept location folder names.

    This is applied *after* full extraction (simplest to implement correctly) -- for the
    100k archive this trades some disk I/O for certainty that the retained subset is an
    unbiased random sample. If disk space during extraction is a concern, extract to a scratch
    location and move only the sampled folders instead.
    """
    all_locations = sorted(
        d for d in os.listdir(extracted_path) if os.path.isdir(os.path.join(extracted_path, d))
    )
    n_total = len(all_locations)
    n_keep = max(1, int(round(n_total * fraction)))

    rng = np.random.RandomState(seed)
    keep_idx = rng.choice(n_total, size=n_keep, replace=False)
    keep_set = set(all_locations[i] for i in keep_idx)

    print(f"Keeping {n_keep}/{n_total} locations ({fraction:.0%}, seed={seed}).")
    removed = 0
    for loc in tqdm(all_locations, desc="Pruning", unit="location"):
        if loc not in keep_set:
            shutil.rmtree(os.path.join(extracted_path, loc), ignore_errors=True)
            removed += 1
    print(f"Removed {removed} location folders not in the sampled subset.")

    kept = sorted(
        d for d in os.listdir(extracted_path) if os.path.isdir(os.path.join(extracted_path, d))
    )
    return kept


def write_manifest(extracted_path, locations, out_path):
    """Write the kept location IDs to a text file, one per line, for reproducible reuse."""
    with open(out_path, "w", encoding="utf-8") as f:
        for loc in locations:
            f.write(loc + "\n")
    print(f"Wrote manifest of {len(locations)} locations to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="../../../data/seco", help="Directory to download/extract into.")
    parser.add_argument("--fraction", type=float, default=0.10, help="Fraction of locations to keep (default: 0.10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the location subsample.")
    parser.add_argument("--no-checksum", action="store_true", help="Skip MD5 verification of the downloaded archive.")
    parser.add_argument("--keep-archive", action="store_true", help="Do not delete seco_100k.zip after extraction.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    archive_path = download_archive(root, checksum=not args.no_checksum)
    extracted_path = extract_archive(archive_path, root)
    locations = subsample_locations(extracted_path, args.fraction, args.seed)
    write_manifest(extracted_path, locations, os.path.join(root, "manifest.txt"))

    if not args.keep_archive:
        print(f"Removing archive {archive_path} to save disk space (pass --keep-archive to retain it).")
        os.remove(archive_path)

    print(f"\nDone. {len(locations)} SeCo locations ready at {extracted_path}")


if __name__ == "__main__":
    main()
