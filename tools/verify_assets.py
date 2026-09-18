"""
Verify that every pretrained encoder this tutorial loads is actually obtainable.

The notebooks get their weights one of two ways: from an on-disk copy (if one was fetched
there already, e.g. by tools/release_weights.sh), or, on Colab, over HTTPS from the GitHub
Release named by WEIGHTS_RELEASE_TAG. This script checks both, so a broken link is found here
rather than in front of a room.

    python tools/verify_assets.py              # check the local copy, then the release URLs
    python tools/verify_assets.py --local      # local copy only (no network)
    python tools/verify_assets.py --remote     # URLs only
    python tools/verify_assets.py --tag TAG    # check a release other than the default

Exit status is 0 only if every requested check passed, so this is usable in CI.

Note: the weights used to be committed via Git LFS, which is capped at 1 GB/month of free
download bandwidth per repo -- a handful of clones or Colab runs exhausted it, after which
media.githubusercontent.com started 404ing. GitHub Release assets have no such quota, which is
why they replaced LFS as the distribution mechanism.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

REPO = "lstival/ssl_tutorial_sibgrapi2026"
DEFAULT_TAG = "weights-v1"

# (repo-relative directory, file name, approximate expected size in bytes)
# Sizes are the committed ones; the check is a loose sanity bound, not an equality test.
ASSETS = [
    ("artifacts/remote_sensing/checkpoints", "contrastive_vit_s8.pt", 43_016_469),
    ("artifacts/remote_sensing/checkpoints", "contrastive_vit_s8_ben.pt", 43_016_869),
    ("artifacts/remote_sensing/checkpoints", "mae_vit_s8.pt", 43_015_797),
    ("artifacts/remote_sensing/checkpoints", "mae_vit_s8_ben.pt", 43_016_133),
    ("artifacts/remote_sensing/checkpoints", "dino_vit_s8.pt", 43_015_881),
    ("artifacts/remote_sensing/checkpoints", "dino_vit_s8_ben.pt", 43_016_217),
    ("artifacts/remote_sensing/checkpoints", "random_init_vit_s8.pt", 43_015_381),
    ("artifacts/time_series/checkpoints", "contrastive_ts_encoder.pt", 2_409_845),
    ("artifacts/time_series/checkpoints", "mae_ts_encoder.pt", 2_409_397),
    ("artifacts/time_series/checkpoints", "dino_ts_encoder.pt", 2_409_445),
]

LFS_POINTER_PREFIX = b"version https://git-lfs"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mb(n):
    return f"{n / 1_048_576:.1f} MB"


def is_pointer(head_bytes, size):
    """A Git LFS pointer is a small text file beginning with a known version line."""
    return size <= 1024 and head_bytes.startswith(LFS_POINTER_PREFIX)


def check_local(directory, name, expected):
    """Check the file on disk: present, real data (not a stale LFS pointer), right size."""
    path = os.path.join(REPO_ROOT, directory, name)
    if not os.path.isfile(path):
        return False, "not present locally (fetched on demand, or run tools/release_weights.sh)"
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(64)
    if is_pointer(head, size):
        return False, "stale Git LFS pointer -- delete it, it will be re-downloaded"
    if abs(size - expected) > max(4096, expected * 0.02):
        return False, f"unexpected size {mb(size)} (expected ~{mb(expected)})"
    return True, mb(size)


def check_remote(directory, name, expected, tag):
    """Check that the release download URL serves the real tensor, not a pointer or a 404."""
    url = f"https://github.com/{REPO}/releases/download/{tag}/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": "ssl-tutorial-verify"})
    try:
        # Read only the first bytes: enough to tell a pointer from a torch archive, and it
        # avoids pulling 43 MB per file just to prove the URL resolves.
        with urllib.request.urlopen(req, timeout=30) as r:
            head = r.read(64)
            size = int(r.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - any transport failure is a failed check
        return False, f"{type(e).__name__}: {e}"

    if is_pointer(head, size):
        return False, "served an LFS pointer, not the weights"
    if size and abs(size - expected) > max(4096, expected * 0.02):
        return False, f"unexpected size {mb(size)} (expected ~{mb(expected)})"
    return True, mb(size) if size else "ok"


def run(title, check):
    print(f"\n{title}")
    print("-" * len(title))
    failures = 0
    for directory, name, expected in ASSETS:
        ok, detail = check(directory, name, expected)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
        failures += not ok
    n = len(ASSETS)
    print(f"  -> {n - failures}/{n} ok")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true", help="check the on-disk copy only (no network)")
    p.add_argument("--remote", action="store_true", help="check the download URLs only")
    p.add_argument("--tag", default=DEFAULT_TAG, help=f"release tag to check (default: {DEFAULT_TAG})")
    args = p.parse_args()

    # Neither flag given means both checks.
    do_local = args.local or not args.remote
    do_remote = args.remote or not args.local

    failures = 0
    if do_local:
        failures += run("Local copy", check_local)
    if do_remote:
        failures += run(
            f"Download URLs (release: {args.tag})",
            lambda d, n, e: check_remote(d, n, e, args.tag),
        )

    if failures:
        print(f"\n{failures} check(s) failed.")
        print("If the remote checks failed, confirm the release/tag exists and the assets were")
        print("uploaded (tools/release_weights.sh). Local failures just mean the file isn't")
        print("on disk yet -- the notebooks download it on demand.")
    else:
        print("\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
