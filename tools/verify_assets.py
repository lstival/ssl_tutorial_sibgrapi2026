"""
Verify that every pretrained encoder this tutorial loads is actually obtainable.

The notebooks get their weights one of two ways: from the local clone (Git LFS), or, on
Colab, over HTTPS from this repository. This script checks both, so a broken link is found
here rather than in front of a room.

    python tools/verify_assets.py              # check the clone, then the remote URLs
    python tools/verify_assets.py --local      # clone only (no network)
    python tools/verify_assets.py --remote     # URLs only
    python tools/verify_assets.py --branch dev # check a branch other than main

Exit status is 0 only if every requested check passed, so this is usable in CI.

Note on the remote check: the weights are stored in Git LFS, which GitHub serves from
media.githubusercontent.com. A plain raw.githubusercontent.com URL returns the ~130-byte LFS
*pointer file* rather than the tensor -- the check below treats that as a failure, because a
notebook that downloads a pointer fails later with an opaque unpickling error.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

REPO = "lstival/ssl_tutorial_sibgrapi2026"

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
    """Check the file in the clone: present, real data (not an LFS pointer), right size."""
    path = os.path.join(REPO_ROOT, directory, name)
    if not os.path.isfile(path):
        return False, "missing from the clone"
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(64)
    if is_pointer(head, size):
        return False, "Git LFS pointer -- run `git lfs pull`"
    if abs(size - expected) > max(4096, expected * 0.02):
        return False, f"unexpected size {mb(size)} (expected ~{mb(expected)})"
    return True, mb(size)


def check_remote(directory, name, expected, branch):
    """Check that the Colab download URL serves the real tensor, not a pointer or a 404."""
    url = f"https://media.githubusercontent.com/media/{REPO}/{branch}/{directory}/{name}"
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
    p.add_argument("--local", action="store_true", help="check the clone only (no network)")
    p.add_argument("--remote", action="store_true", help="check the download URLs only")
    p.add_argument("--branch", default="main", help="branch to check remotely (default: main)")
    args = p.parse_args()

    # Neither flag given means both checks.
    do_local = args.local or not args.remote
    do_remote = args.remote or not args.local

    failures = 0
    if do_local:
        failures += run("Local clone (Git LFS)", check_local)
    if do_remote:
        failures += run(
            f"Download URLs (branch: {args.branch})",
            lambda d, n, e: check_remote(d, n, e, args.branch),
        )

    if failures:
        print(f"\n{failures} check(s) failed.")
        print("If the remote checks failed, confirm the branch is pushed and `git lfs push`")
        print("has uploaded the objects. Local failures usually mean `git lfs pull` is needed.")
    else:
        print("\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
