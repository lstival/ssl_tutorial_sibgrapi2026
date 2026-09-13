"""
BigEarthNet-S2 pretraining corpus: a flat set of Sentinel-2 RGB patches.

This is the BigEarthNet counterpart to `seco_data.py`. The key structural difference from
SeCo is that BigEarthNet has no "location / seasonal revisit" hierarchy -- every patch is an
independent 120x120 Sentinel-2 tile. So there is exactly one image per "location", and the
only positive-pair definition available is SimCLR-style (two augmentations of the same
patch); SeCo's "distinct seasons of the same place" mode does not apply.

`BigEarthNetPatchCache` deliberately exposes the *same interface* as `gpu_aug.SeCoPatchCache`
(`.patches`, `.loc_start`, `.loc_count`, `.to()`, `__len__`, `.nbytes`, `.n_patches`,
`.sample_views()`), so the GPU-augmentation training path in train_contrastive.py /
train_mae.py / train_dino.py works unchanged once it is handed one of these instead.

BigEarthNet-S2 band files are uint16 reflectance (Sentinel-2 L2A, scale 10000), unlike the
seco_100k Zenodo release's 8-bit contrast-stretched previews -- so RGB is read via rasterio
and divided by SENTINEL2_SCALE, then clipped to [0, 1].
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

# Sentinel-2 true-colour band order (10 m): R, G, B.
RGB_BAND_SUFFIXES = ("B04", "B03", "B02")
SENTINEL2_SCALE = 10000.0

# Per-channel RGB statistics for BigEarthNet-S2, in the [0, 1] scaled space this module
# produces. These are the widely-used reBEN/BigEarthNet v2 means/stds (B04, B03, B02) divided
# by SENTINEL2_SCALE. Only the pretraining forward pass uses them; downstream EuroSAT probing
# uses EuroSAT's own normalization (same convention as seco_data.SECO_MEAN/STD).
BEN_MEAN = (0.05504, 0.06592, 0.06263)
BEN_STD = (0.02777, 0.02383, 0.02465)


def read_manifest(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_s2_root(root):
    """Return the directory that directly contains the patch subdirectories."""
    marker = os.path.join(root, "s2_root.txt")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            p = f.read().strip()
        if os.path.isdir(p):
            return p
    for name in ("BigEarthNet-S2", "BigEarthNet-v1.0", "BigEarthNet"):
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            return cand
    return root


def _band_path(patch_dir, suffix):
    for ext in (".tif", ".tiff"):
        # BigEarthNet names bands "<patch_id>_<SUFFIX>.tif"; the patch dir basename is the id.
        cand = os.path.join(patch_dir, f"{os.path.basename(patch_dir)}_{suffix}{ext}")
        if os.path.isfile(cand):
            return cand
    # Fallback: any file ending in the band suffix.
    for f in os.listdir(patch_dir):
        if f.endswith(f"_{suffix}.tif") or f.endswith(f"_{suffix}.tiff"):
            return os.path.join(patch_dir, f)
    raise FileNotFoundError(f"No {suffix} band in {patch_dir}")


def read_rgb_patch(patch_dir):
    """Read the RGB bands of one BigEarthNet-S2 patch to (H, W, 3) float32 in [0, 1]."""
    import rasterio

    channels = []
    for suffix in RGB_BAND_SUFFIXES:
        with rasterio.open(_band_path(patch_dir, suffix)) as src:
            channels.append(src.read(1).astype(np.float32))
    stacked = np.stack(channels, axis=-1)
    return np.clip(stacked / SENTINEL2_SCALE, 0.0, 1.0)


def _decode_patch(task):
    """Decode one patch to a (3, res, res) uint8 array. Module-level for ProcessPoolExecutor."""
    from PIL import Image

    patch_dir, res = task
    patch = (read_rgb_patch(patch_dir) * 255).astype(np.uint8)  # (H, W, 3)
    if res is not None and patch.shape[0] != res:
        patch = np.asarray(Image.fromarray(patch).resize((res, res), Image.BILINEAR))
    return np.ascontiguousarray(patch.transpose(2, 0, 1))


class BigEarthNetPatchCache:
    """
    The BigEarthNet-S2 pretraining corpus as one uint8 tensor, one patch per row.

    Interface-compatible with gpu_aug.SeCoPatchCache: `loc_start[i] = i`, `loc_count[i] = 1`
    for every patch, so `sample_views` always returns augmentation-only positive pairs.
    """

    NATIVE_RES = 120

    def __init__(self, root, patch_names, res=None, workers=8, verbose=True, cache_file=None):
        self.s2_root = resolve_s2_root(root)
        self.locations = list(patch_names)  # kept named `locations` for interface parity
        self.res = res if res is not None else self.NATIVE_RES
        self.device = torch.device("cpu")

        patch_dirs = [os.path.join(self.s2_root, p) for p in self.locations]
        if not patch_dirs:
            raise RuntimeError(f"No BigEarthNet patches given for {self.s2_root}.")

        if cache_file is not None and self._load_from_disk(cache_file, self.res, verbose):
            self._set_flat_index()
            return

        n_bytes = len(patch_dirs) * 3 * self.res * self.res
        if verbose:
            print(f"Decoding {len(patch_dirs)} BigEarthNet patches at {self.res}x{self.res} "
                  f"into a {n_bytes / 1e9:.2f} GB uint8 cache using {workers} processes...",
                  flush=True)

        self.patches = torch.empty((len(patch_dirs), 3, self.res, self.res), dtype=torch.uint8)
        t0 = time.time()
        tasks = [(d, self.res) for d in patch_dirs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, arr in enumerate(pool.map(_decode_patch, tasks, chunksize=64)):
                self.patches[i] = torch.from_numpy(arr)
                if verbose and (i + 1) % 20000 == 0:
                    print(f"  {i + 1}/{len(patch_dirs)} patches ({time.time() - t0:.0f}s)", flush=True)
        if verbose:
            print(f"Cache ready in {time.time() - t0:.0f}s.", flush=True)

        self._set_flat_index()
        if cache_file is not None:
            self._save_to_disk(cache_file, verbose)

    def _set_flat_index(self):
        n = self.patches.shape[0]
        self.loc_start = torch.arange(n, dtype=torch.long)
        self.loc_count = torch.ones(n, dtype=torch.long)

    # --- on-disk cache (flat blob + json index; no per-location structure needed) -------
    @staticmethod
    def _cache_paths(cache_file):
        base = os.path.splitext(cache_file)[0]
        return base + ".u8", base + ".json"

    def _load_from_disk(self, cache_file, res, verbose):
        blob_path, index_path = self._cache_paths(cache_file)
        if not (os.path.isfile(blob_path) and os.path.isfile(index_path)):
            return False
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        if index["res"] != res:
            if verbose:
                print(f"Patch cache {blob_path} is at res {index['res']}, need {res}; re-decoding.")
            return False
        stored = {name: i for i, name in enumerate(index["patches"])}
        missing = [p for p in self.locations if p not in stored]
        if missing:
            if verbose:
                print(f"Patch cache {blob_path} is missing {len(missing)} requested patches; re-decoding.")
            return False
        t0 = time.time()
        blob = np.memmap(blob_path, dtype=np.uint8, mode="r").reshape(-1, 3, res, res)
        rows = np.array([stored[p] for p in self.locations])
        self.patches = torch.from_numpy(np.ascontiguousarray(blob[rows]))
        if verbose:
            print(f"Loaded {self.patches.shape[0]} patches at {res}x{res} from {blob_path} "
                  f"in {time.time() - t0:.0f}s (no decode).", flush=True)
        return True

    def _save_to_disk(self, cache_file, verbose):
        blob_path, index_path = self._cache_paths(cache_file)
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing["res"] == self.res and len(existing["patches"]) >= len(self.locations):
                return
        os.makedirs(os.path.dirname(os.path.abspath(blob_path)), exist_ok=True)
        t0 = time.time()
        # Per-process temp name so concurrent training jobs sharing one --patch-cache don't
        # race on a fixed "<blob>.tmp" (see the same fix in gpu_aug.SeCoPatchCache).
        tmp = f"{blob_path}.{os.getpid()}.tmp"
        try:
            self.patches.numpy().tofile(tmp)
            os.replace(tmp, blob_path)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"res": self.res, "patches": self.locations}, f)
        except OSError as exc:
            if os.path.isfile(blob_path):
                if verbose:
                    print(f"Patch cache {blob_path} was written by another process ({exc}); "
                          f"continuing with the in-RAM copy.", flush=True)
            else:
                raise
            return
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        if verbose:
            print(f"Wrote patch cache to {blob_path} ({self.nbytes / 1e9:.2f} GB, "
                  f"{time.time() - t0:.0f}s).", flush=True)

    def __len__(self):
        return len(self.locations)

    @property
    def n_patches(self):
        return self.patches.shape[0]

    @property
    def nbytes(self):
        return self.patches.numel()

    def to(self, device):
        self.patches = self.patches.to(device)
        self.loc_start = self.loc_start.to(device)
        self.loc_count = self.loc_count.to(device)
        self.device = device
        return self

    def sample_views(self, loc_idx, n_views=2, distinct_seasons=False, generator=None):
        """One patch per index; every view starts from that same patch (SimCLR positives)."""
        if distinct_seasons:
            raise ValueError("BigEarthNet has no seasonal revisits; --seasonal-positives is SeCo-only.")
        batch = self.patches.index_select(0, self.loc_start[loc_idx])
        return [batch] * n_views


# ---------------------------------------------------------------------------------------
# CPU DataLoader fallback path (used only with --no-gpu-aug), mirroring SeCoAugmentedDataset.
# ---------------------------------------------------------------------------------------
class BigEarthNetAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, n_views=2, patch_names=None, preload=False, preload_res=None):
        self.s2_root = resolve_s2_root(root)
        self.transform = transform
        self.n_views = n_views
        self.preload_res = preload_res
        if patch_names is None:
            raise ValueError(
                "patch_names is required: enumerate patches with download_bigearthnet.py "
                "(it writes manifest.txt) rather than scanning the ~590k-dir tree here."
            )
        self.patch_names = list(patch_names)
        self._cache = {}
        if preload:
            self._preload()

    def _preload(self):
        from PIL import Image

        for name in self.patch_names:
            patch = (read_rgb_patch(os.path.join(self.s2_root, name)) * 255).astype(np.uint8)
            if self.preload_res is not None and patch.shape[0] != self.preload_res:
                patch = np.asarray(Image.fromarray(patch).resize(
                    (self.preload_res, self.preload_res), Image.BILINEAR))
            self._cache[name] = patch

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        from PIL import Image

        name = self.patch_names[idx]
        cached = self._cache.get(name)
        if cached is None:
            cached = (read_rgb_patch(os.path.join(self.s2_root, name)) * 255).astype(np.uint8)
        img = Image.fromarray(cached)
        return [self.transform(img) for _ in range(self.n_views)], 0
