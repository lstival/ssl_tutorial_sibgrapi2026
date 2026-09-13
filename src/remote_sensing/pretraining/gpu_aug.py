"""
GPU-side augmentation pipeline for SeCo pretraining.

Why this exists. The CPU pipeline in `seco_data.py` (PIL -> torchvision.transforms -> ToTensor)
costs ~2 ms per view. A contrastive step needs batch_size x 2 views, so at batch 256 that is
512 x 2 ms ~= 1.0 s of single-threaded Python per step -- and because the in-RAM preload forces
`num_workers=0` (workers would duplicate the multi-GB patch cache under Windows spawn), that
cost is fully serial with the GPU. Measured on the ViT-S/8 100k run: 1.22 s/step with the GPU
idle ~80% of the time.

This module removes that stall by keeping the decoded corpus as one big uint8 tensor and doing
every augmentation as a batched tensor op on the GPU:

  * `SeCoPatchCache` decodes the .tif bands once, in a process pool, into a single (M, 3, R, R)
    uint8 tensor plus a per-location season index. Placed on the GPU when it fits (100k patches
    at R=128 is 4.9 GB), the per-step batch is then an `index_select` with zero host-to-device
    traffic. The decode can also be persisted to disk (`cache_file`) so the five subprocesses of
    a scaling study do not each repeat it.
  * `gpu_augment` fuses RandomResizedCrop + H/V flip + rotation into ONE `grid_sample` -- the
    crop box, the flips and the rotation are all affine, so composing them into a single
    sampling grid costs one bilinear resample instead of three separate resamples, and produces
    the 64x64 output directly at the resolution we actually train on.

Fidelity to the CPU recipe (`build_seco_augmentations`). The sampled distributions are the
same (scale, ratio, flip and rotation probabilities, jitter strength), with three deliberate
differences, all documented at their call sites below:
  1. RandomResizedCrop's 10-try rejection loop is replaced by clamping the sampled crop to the
     patch -- vectorized, and distributionally equivalent for scale <= 1.0 where rejection
     almost never fires.
  2. Rotation resamples bilinearly (torchvision's RandomRotation defaults to nearest).
  3. Jitter arithmetic is done in float rather than torchvision's uint8-rounded path.
None of these change what the pretext task is; they do mean a GPU-augmented run is not
bit-identical to a CPU-augmented one, so a single comparison (e.g. the corpus-scaling curve in
run_seco_scaling.py) must not mix the two paths.
"""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from seco_data import RGB_BANDS, SECO_MEAN, SECO_STD, list_seasons, read_rgb_patch


def _decode_patch(task):
    """
    Decode one season folder's RGB bands to a (3, res, res) uint8 array.

    Deliberately does NOT go through `read_rgb_patch` when it can avoid it. That function opens
    each band with rasterio/GDAL, which costs ~10 ms per patch here -- 17 min for the 100k
    corpus -- and, worse, serializes across threads, so a thread pool made it *slower* than
    single-threaded. The seco_100k Zenodo release stores each band as a plain 8-bit TIFF, which
    PIL decodes in ~2.9 ms per patch with byte-identical output (verified). Anything else (the
    raw uint16 reflectance layout) falls back to `read_rgb_patch` and its scaling.

    Module-level and taking a single tuple so a ProcessPoolExecutor can pickle it.
    """
    from PIL import Image

    season_path, res = task
    imgs = [Image.open(os.path.join(season_path, f"{band}.tif")) for band in RGB_BANDS]
    if all(im.mode == "L" for im in imgs):
        if res is not None and imgs[0].size[0] != res:
            imgs = [im.resize((res, res), Image.BILINEAR) for im in imgs]
        return np.stack([np.asarray(im) for im in imgs], axis=0)

    patch = (read_rgb_patch(season_path) * 255).astype(np.uint8)  # (H, W, 3)
    if res is not None and patch.shape[0] != res:
        patch = np.asarray(Image.fromarray(patch).resize((res, res), Image.BILINEAR))
    return np.ascontiguousarray(patch.transpose(2, 0, 1))


class SeCoPatchCache:
    """
    The whole pretraining corpus as one uint8 tensor, grouped by location.

    `patches` is (M, 3, R, R) with all seasons of all locations concatenated; `loc_start` and
    `loc_count` index into it, so sampling "one random season of location i" is arithmetic on
    index tensors rather than a filesystem access.
    """

    def __init__(self, seco_root, locations, res=None, workers=8, verbose=True, cache_file=None):
        self.seco_root = seco_root
        self.locations = list(locations)
        self.res = res
        self.device = torch.device("cpu")

        # Enumerate (location, season) pairs first so the destination tensor can be allocated
        # once: filling disjoint slices from threads needs no locking and no per-patch realloc.
        pairs = []
        starts, counts = [], []
        for location in self.locations:
            location_path = os.path.join(seco_root, location)
            seasons = list_seasons(location_path)
            starts.append(len(pairs))
            counts.append(len(seasons))
            for season in seasons:
                pairs.append(os.path.join(location_path, season))
        if not pairs:
            raise RuntimeError(f"No season folders found under {seco_root} for the given locations.")

        if res is None:
            res = read_rgb_patch(pairs[0]).shape[0]
            self.res = res

        if cache_file is not None and self._load_from_disk(cache_file, res, starts, counts, verbose):
            return

        n_bytes = len(pairs) * 3 * res * res
        if verbose:
            print(f"Decoding {len(pairs)} patches ({len(self.locations)} locations) at "
                  f"{res}x{res} into a {n_bytes / 1e9:.2f} GB uint8 cache "
                  f"using {workers} processes...", flush=True)

        self.patches = torch.empty((len(pairs), 3, res, res), dtype=torch.uint8)
        t0 = time.time()
        tasks = [(p, res) for p in pairs]
        # Processes, not threads: image decoding is CPU-bound behind the GIL (and GDAL, on the
        # fallback path, holds an internal lock), so a thread pool contends instead of scaling.
        # A large chunksize keeps the per-task pickling overhead well below the ~3 ms of work.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, arr in enumerate(pool.map(_decode_patch, tasks, chunksize=64)):
                self.patches[i] = torch.from_numpy(arr)
                if verbose and (i + 1) % 20000 == 0:
                    print(f"  {i + 1}/{len(pairs)} patches ({time.time() - t0:.0f}s)", flush=True)
        if verbose:
            print(f"Cache ready in {time.time() - t0:.0f}s.", flush=True)

        self.loc_start = torch.tensor(starts, dtype=torch.long)
        self.loc_count = torch.tensor(counts, dtype=torch.long)

        if cache_file is not None:
            self._save_to_disk(cache_file, verbose)

    # --- optional on-disk cache ----------------------------------------------------------
    # Decoding the full 100k corpus costs ~15 min of .tif reads (300k file opens) and is paid
    # again by every process. run_seco_scaling.py launches five training subprocesses over
    # nested subsets of the same corpus, so without this the study spends over an hour just
    # re-decoding. The store holds ALL locations seen so far as a flat uint8 blob plus a JSON
    # index; a run whose locations are a subset gathers its rows out of the blob.

    @staticmethod
    def _cache_paths(cache_file):
        base = os.path.splitext(cache_file)[0]
        return base + ".u8", base + ".json"

    def _load_from_disk(self, cache_file, res, starts, counts, verbose):
        blob_path, index_path = self._cache_paths(cache_file)
        if not (os.path.isfile(blob_path) and os.path.isfile(index_path)):
            return False
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        if index["res"] != res:
            if verbose:
                print(f"Patch cache {blob_path} is at res {index['res']}, need {res}; re-decoding.")
            return False

        stored = {loc: (s, c) for loc, s, c in
                  zip(index["locations"], index["starts"], index["counts"])}
        missing = [loc for loc in self.locations if loc not in stored]
        if missing:
            if verbose:
                print(f"Patch cache {blob_path} is missing {len(missing)} of the requested "
                      f"locations; re-decoding.")
            return False

        t0 = time.time()
        blob = np.memmap(blob_path, dtype=np.uint8, mode="r").reshape(-1, 3, res, res)
        # The requested locations are generally a scattered subset of the store, so gather the
        # rows location-by-location (each location's seasons are contiguous) into a fresh,
        # compact tensor whose own indexing then matches `starts`/`counts`.
        self.patches = torch.empty((sum(counts), 3, res, res), dtype=torch.uint8)
        for loc, dst_start, n in zip(self.locations, starts, counts):
            src_start, src_n = stored[loc]
            if src_n != n:
                raise RuntimeError(
                    f"Patch cache {blob_path} has {src_n} seasons for location {loc} but the "
                    f"corpus now has {n}; delete the cache to rebuild it."
                )
            self.patches[dst_start:dst_start + n] = torch.from_numpy(
                np.asarray(blob[src_start:src_start + src_n])
            )
        self.loc_start = torch.tensor(starts, dtype=torch.long)
        self.loc_count = torch.tensor(counts, dtype=torch.long)
        if verbose:
            print(f"Loaded {self.patches.shape[0]} patches at {res}x{res} from {blob_path} "
                  f"in {time.time() - t0:.0f}s (no decode).", flush=True)
        return True

    def _save_to_disk(self, cache_file, verbose):
        blob_path, index_path = self._cache_paths(cache_file)
        # Only ever grow the store to a superset; a run over a small subset must not shrink a
        # cache the next (larger) run would otherwise reuse.
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing["res"] == self.res and len(existing["locations"]) >= len(self.locations):
                return
        os.makedirs(os.path.dirname(os.path.abspath(blob_path)), exist_ok=True)
        t0 = time.time()
        # Per-process temp name: when several training jobs share one --patch-cache and each
        # decodes it before the file exists, a fixed "<blob>.tmp" makes the second os.replace
        # fail (the first already renamed it away). A unique suffix keeps each writer's temp
        # file its own; whoever lands last wins, and the bytes are identical either way.
        tmp = f"{blob_path}.{os.getpid()}.tmp"
        try:
            self.patches.numpy().tofile(tmp)
            os.replace(tmp, blob_path)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"res": self.res, "locations": self.locations,
                           "starts": self.loc_start.tolist(), "counts": self.loc_count.tolist()}, f)
        except OSError as exc:
            # A concurrent writer completing the same cache first is fine -- our decode already
            # succeeded and this run has its in-RAM copy; the on-disk store just isn't ours.
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
        """
        Return `n_views` uint8 batches (each (B, 3, R, R)) for the given location indices.

        With `distinct_seasons=False` every view starts from the *same* randomly chosen season
        (SimCLR positives: the pair differs only by augmentation). With `distinct_seasons=True`
        each view is a different seasonal revisit of the same place -- SeCo's own positive-pair
        definition -- which the CPU path could only serve by re-decoding from disk every step.
        """
        start = self.loc_start[loc_idx]
        count = self.loc_count[loc_idx]
        dev = start.device

        if not distinct_seasons:
            # `.minimum(count - 1)` guards the measure-zero case where torch.rand returns
            # exactly 1.0 and the product lands one past the last season of that location.
            offset = (torch.rand(loc_idx.shape, device=dev, generator=generator)
                      * count).long().minimum(count - 1)
            batch = self.patches.index_select(0, start + offset)
            # The same source patch for every view: the pair differs only by augmentation, and
            # returning the same tensor avoids a duplicate gather (each view is augmented
            # independently downstream, out-of-place).
            return [batch] * n_views

        if n_views > 2:
            raise ValueError("distinct_seasons supports at most 2 views.")
        if int(count.min()) < 2:
            raise ValueError("distinct_seasons needs >= 2 seasons at every sampled location.")
        first = (torch.rand(loc_idx.shape, device=dev, generator=generator)
                 * count).long().minimum(count - 1)
        # Offset the second draw by 1..count-1 modulo count: uniform over the other seasons,
        # and distinct from the first by construction (no rejection loop).
        step = 1 + (torch.rand(loc_idx.shape, device=dev, generator=generator)
                    * (count - 1)).long().minimum(count - 2)
        second = (first + step) % count
        views = [self.patches.index_select(0, start + first),
                 self.patches.index_select(0, start + second)]
        return views[:n_views]


class LocationShuffler:
    """
    Epoch-wise shuffled stream of location indices, matching
    `DataLoader(shuffle=True, drop_last=True)`: a fresh permutation each pass, trailing
    partial batch dropped.
    """

    def __init__(self, n_locations, batch_size, device, generator=None):
        self.n = n_locations
        self.batch_size = batch_size
        self.device = device
        self.generator = generator
        self._perm = None
        self._pos = 0
        self._reshuffle()

    def _reshuffle(self):
        self._perm = torch.randperm(self.n, device=self.device, generator=self.generator)
        self._pos = 0

    def next_batch(self):
        if self._pos + self.batch_size > self.n:
            self._reshuffle()
        idx = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx


def _rand(n, low, high, device, generator):
    return torch.rand(n, device=device, generator=generator) * (high - low) + low


def gpu_augment(batch_u8, out_size, mean, std, generator=None,
                scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                hflip_p=0.5, vflip_p=0.5, rot_p=0.5, rot_degrees=90.0,
                jitter_p=0.5, brightness=0.2, contrast=0.2):
    """
    Batched GPU equivalent of `build_seco_augmentations`: RandomResizedCrop -> H/V flip ->
    RandomApply(RandomRotation) -> RandomApply(ColorJitter) -> Normalize.

    `batch_u8`: (B, 3, R, R) uint8 on the target device. Returns (B, 3, out_size, out_size)
    float32, normalized with `mean`/`std` (which must already be device tensors of shape
    (1, 3, 1, 1)).
    """
    device = batch_u8.device
    B = batch_u8.shape[0]
    x = batch_u8.float().div_(255.0)

    # --- RandomResizedCrop, as fractions of the source patch -------------------------------
    # torchvision samples area in `scale` and aspect in `ratio`, retrying up to 10 times when
    # the box falls outside the image, then falls back to a center crop. Here the box is simply
    # clamped to the patch, which is vectorizable and, for scale<=1.0 with these ratios, fires
    # on a small minority of draws (an extreme aspect near area 1.0).
    area = _rand(B, scale[0], scale[1], device, generator)
    log_ratio = _rand(B, math.log(ratio[0]), math.log(ratio[1]), device, generator)
    aspect = torch.exp(log_ratio)
    w = (area * aspect).sqrt().clamp_max_(1.0)   # crop width  as a fraction of the patch
    h = (area / aspect).sqrt().clamp_max_(1.0)   # crop height as a fraction of the patch

    # Crop centre in grid_sample's normalized [-1, 1] coordinates: the top-left corner is
    # uniform over the legal range, so the centre is uniform over [w/2, 1-w/2] (mapped to
    # [-1, 1] by 2c-1).
    cx = 2.0 * (torch.rand(B, device=device, generator=generator) * (1.0 - w) + w / 2) - 1.0
    cy = 2.0 * (torch.rand(B, device=device, generator=generator) * (1.0 - h) + h / 2) - 1.0

    # --- flips and rotation, folded into the same affine map --------------------------------
    fx = 1.0 - 2.0 * (torch.rand(B, device=device, generator=generator) < hflip_p).float()
    fy = 1.0 - 2.0 * (torch.rand(B, device=device, generator=generator) < vflip_p).float()
    rot_on = (torch.rand(B, device=device, generator=generator) < rot_p).float()
    theta = _rand(B, -rot_degrees, rot_degrees, device, generator) * rot_on * (math.pi / 180.0)
    cos, sin = torch.cos(theta), torch.sin(theta)

    # Inverse map from an output pixel (u, v) in [-1, 1]^2 back to source coordinates: rotate
    # inside the crop's own square (which is what rotating the already-resized crop means),
    # then flip, then scale/translate into the crop box. Rotation before the anisotropic
    # (w != h) scaling is exactly torchvision's order -- crop, resize to a square, then rotate.
    mat = torch.zeros(B, 2, 3, device=device)
    mat[:, 0, 0] = w * fx * cos
    mat[:, 0, 1] = w * fx * sin
    mat[:, 0, 2] = cx
    mat[:, 1, 0] = -h * fy * sin
    mat[:, 1, 1] = h * fy * cos
    mat[:, 1, 2] = cy

    grid = F.affine_grid(mat, (B, 3, out_size, out_size), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Black corners. torchvision rotates the crop *after* it has been cut out and resized, so
    # anything the rotation pulls in from beyond the crop's own edge is filled with 0 (black).
    # Fusing the two steps would instead sample real neighbouring pixels from the surrounding
    # patch -- arguably nicer, but a different augmentation distribution (measured: 1.8% black
    # pixels fused vs 4.5% with torchvision). Re-applying the fill keeps the GPU path a drop-in
    # replacement for the CPU recipe: mask out every output pixel whose pre-crop coordinate left
    # the [-1, 1]^2 square, which is the same affine map with the crop box removed.
    rot_only = mat.clone()
    rot_only[:, 0, :2] /= w.view(-1, 1)
    rot_only[:, 1, :2] /= h.view(-1, 1)
    rot_only[:, :, 2] = 0.0
    inside = F.affine_grid(rot_only, (B, 1, out_size, out_size), align_corners=False)
    out = out * (inside.abs() <= 1.0).all(dim=-1).unsqueeze(1).float()

    # --- ColorJitter(brightness, contrast), applied with probability jitter_p ---------------
    if brightness > 0 or contrast > 0:
        on = (torch.rand(B, 1, 1, 1, device=device, generator=generator) < jitter_p).float()
        b_factor = 1.0 + _rand(B, -brightness, brightness, device, generator).view(-1, 1, 1, 1) * on
        c_factor = 1.0 + _rand(B, -contrast, contrast, device, generator).view(-1, 1, 1, 1) * on

        def apply_brightness(t):
            return (t * b_factor).clamp_(0.0, 1.0)

        def apply_contrast(t):
            gray = 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]
            m = gray.mean(dim=(1, 2)).view(-1, 1, 1, 1)
            return ((t - m) * c_factor + m).clamp_(0.0, 1.0)

        # torchvision's ColorJitter shuffles the order of its enabled ops per call; one draw
        # per batch (rather than per sample) keeps that without branching B ways.
        if float(torch.rand((), device=device, generator=generator)) < 0.5:
            out = apply_contrast(apply_brightness(out))
        else:
            out = apply_brightness(apply_contrast(out))

    return (out - mean) / std


def norm_constants(device, mean=SECO_MEAN, std=SECO_STD):
    """Return (mean, std) as (1, 3, 1, 1) device tensors for `gpu_augment`."""
    return (torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1))


def choose_cache_device(nbytes, device, headroom_bytes=3 * 1024 ** 3, verbose=True):
    """
    Put the patch cache on the GPU when it fits alongside the model and its activations,
    otherwise keep it in host RAM. On the GPU a batch costs one `index_select` and no PCIe
    traffic at all; from host RAM it is an extra ~12 MB copy per step, which is still far
    cheaper than the CPU augmentation it replaces.
    """
    if device.type != "cuda":
        return torch.device("cpu")
    free, _total = torch.cuda.mem_get_info()
    fits = nbytes + headroom_bytes < free
    if verbose:
        print(f"Patch cache {nbytes / 1e9:.2f} GB | free VRAM {free / 1e9:.2f} GB -> "
              f"cache on {'GPU' if fits else 'CPU RAM'}")
    return device if fits else torch.device("cpu")
