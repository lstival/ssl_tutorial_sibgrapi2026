"""
Dataset classes for pretraining on the SeCo (Seasonal Contrast) corpus.

Reads the on-disk layout produced by `download_seco.py`:

    <root>/seasonal_contrast_100k/
        <location_id>/
            <season_subdir_1>/
                B1.tif ... B12.tif
            <season_subdir_2>/
                ...

We only ever read the RGB bands (B4, B3, B2 -> R, G, B), matching the EuroSAT RGB tutorial
data (src/remote_sensing/tutorial_rs.py): pretraining and downstream fine-tuning/probing must see the same
number of channels and a comparable spectral normalization (see
docs/remote_sensing_implementation_plan.md, section 3.1).

Each SeCo location has multiple seasonal revisits of the same coordinate. This module offers
two ways to build a positive pair for pretraining:

  - `SeCoAugmentedDataset`: standard SimCLR-style positive pairs -- two independent
    augmentations of the *same* season's patch (used by masking/distillation notebooks too,
    where a single view or a multi-crop set is needed).
  - `SeCoSeasonalDataset`: SeCo's own positive pair definition -- two *different* seasons of
    the same location, optionally still augmented. This is the "free" invariance signal
    specific to remote sensing mentioned in Notebook 1 (no synthetic augmentation needed to
    get a second, legitimately different view of the same place).
"""

import os
import random

import numpy as np
import rasterio
from torch.utils.data import Dataset
from torchvision import transforms

RGB_BANDS = ("B4", "B3", "B2")  # Sentinel-2 true-color order: R, G, B

# Sentinel-2 L1C/L2A reflectance is stored as uint16 with a fixed scale factor; dividing by
# this puts pixel values roughly in [0, 1] before our own mean/std normalization (mirrors the
# convention used by SeCo/SSL4EO-S12 and Sentinel-2 processing baselines).
SENTINEL2_SCALE = 10000.0

# Precomputed per-channel statistics of SeCo RGB (measured directly on the downloaded 10%
# subset: 200 random locations, one random season each, via read_rgb_patch -- see
# pretraining/README.md). Distinct from EUROSAT_MEAN/STD in src/remote_sensing/tutorial_rs.py: EuroSAT is
# processed and clipped differently upstream. Encoders pretrained here are meant to be
# fine-tuned or linear-probed on EuroSAT using EuroSAT's own normalization, not this one --
# only the pretraining forward pass uses these stats (see README in this folder).
SECO_MEAN = (0.1685, 0.1346, 0.0988)
SECO_STD = (0.1360, 0.1046, 0.0907)


def list_locations(seco_root):
    """Return the sorted list of location-ID subdirectories under the extracted SeCo directory."""
    return sorted(
        d for d in os.listdir(seco_root) if os.path.isdir(os.path.join(seco_root, d))
    )


def list_seasons(location_path):
    """Return the sorted list of season subdirectories for one SeCo location."""
    return sorted(
        d for d in os.listdir(location_path) if os.path.isdir(os.path.join(location_path, d))
    )


def read_rgb_patch(season_path, bands=RGB_BANDS):
    """
    Read and stack the given bands from one season folder into a (H, W, C) float32 array,
    scaled to [0, 1]. Returns a numpy array suitable for `Image.fromarray` after converting
    to uint8, or direct tensor conversion.

    The seco_100k Zenodo release stores each band as 8-bit (uint8, 0-255) -- already a
    contrast-stretched preview, not raw 16-bit reflectance -- unlike the raw SeCo/SSL4EO-S12
    collection pipeline, which produces uint16 reflectance scaled by SENTINEL2_SCALE. We
    detect the on-disk dtype per file and scale accordingly so both variants work.
    """
    channels = []
    src_dtype = None
    for band in bands:
        band_path = os.path.join(season_path, f"{band}.tif")
        with rasterio.open(band_path) as src:
            data = src.read(1)
            src_dtype = data.dtype
            channels.append(data.astype(np.float32))
    stacked = np.stack(channels, axis=-1)

    if src_dtype == np.uint8:
        patch = stacked / 255.0
    else:
        patch = stacked / SENTINEL2_SCALE
    return np.clip(patch, 0.0, 1.0)


def to_pil_uint8(patch_float):
    """Convert a (H, W, C) float32 array in [0, 1] to a PIL Image (uint8), for torchvision transforms."""
    from PIL import Image

    return Image.fromarray((patch_float * 255).astype(np.uint8))


class SeCoAugmentedDataset(Dataset):
    """
    Standard positive-pair dataset: `n_views` independent augmentations of the *same* season's
    RGB patch at each location. Used by the contrastive (SimCLR-style) and multi-crop
    (DINO-style) pretraining scripts; also usable as a plain single-view dataset (MAE) by
    setting `n_views=1` and indexing `[0]` on the returned list.
    """

    def __init__(self, seco_root, transform, n_views=2, locations=None, preload=False,
                 preload_res=None):
        self.seco_root = seco_root
        self.transform = transform
        self.n_views = n_views
        self.locations = locations if locations is not None else list_locations(seco_root)
        self.preload_res = preload_res
        # Decoding the RGB .tif bands via rasterio is ~12 ms/patch and, re-run every epoch, is the
        # dominant cost of pretraining -- it starves the GPU (the trainer sees ~4 s stalls on each
        # epoch boundary; see pretraining/README.md profiling notes). With `preload=True` we decode
        # every (location, season) patch to uint8 once, up front, into an in-RAM dict (~2 GB for the
        # 10% SeCo subset: 10k patches). Thereafter __getitem__ does no disk I/O -- only the (cheap,
        # ~0.5 ms) augmentation runs -- so a decoded-in-RAM run is ~5x faster. Preloading happens in
        # the main process before workers are spawned; run with num_workers=0 so the single in-RAM
        # copy is reused (Windows spawn would otherwise duplicate it per worker).
        #
        # `preload_res` downscales each patch to (preload_res, preload_res) as it is cached, which
        # is what makes the full 100k corpus trainable on a 32 GB machine. SeCo patches are 264x264
        # = 204 KB each as uint8, so the cache costs ~2.1 GB for the 10% subset (10k patches) but
        # ~20.9 GB for the full 100k -- more RAM than this machine has. Since the augmentation
        # pipeline immediately RandomResizedCrops down to IMG_SIZE (64), the extra resolution is
        # thrown away anyway; caching at 128 keeps crops of 90-128 px (scale=(0.5,1.0)) before the
        # final downsample, preserving crop diversity at 4.9 GB for the full corpus.
        # NOTE: this changes the augmentation distribution slightly (crops are taken from an
        # already-downsampled patch), so runs used in the same comparison must all pass the same
        # `preload_res` -- otherwise a corpus-size comparison is confounded by resampling.
        self._cache = {}
        if preload:
            self._preload()

    def _preload(self):
        from PIL import Image

        for location in self.locations:
            location_path = os.path.join(self.seco_root, location)
            for season in list_seasons(location_path):
                patch = (read_rgb_patch(os.path.join(location_path, season)) * 255).astype(np.uint8)
                if self.preload_res is not None and patch.shape[0] != self.preload_res:
                    patch = np.asarray(
                        Image.fromarray(patch).resize(
                            (self.preload_res, self.preload_res), Image.BILINEAR
                        )
                    )
                self._cache[(location, season)] = patch

    def __len__(self):
        return len(self.locations)

    def _get_patch_uint8(self, location, location_path, season):
        cached = self._cache.get((location, season))
        if cached is not None:
            return cached
        return (read_rgb_patch(os.path.join(location_path, season)) * 255).astype(np.uint8)

    def __getitem__(self, idx):
        location = self.locations[idx]
        location_path = os.path.join(self.seco_root, location)
        seasons = list_seasons(location_path)
        season = random.choice(seasons)

        from PIL import Image
        img = Image.fromarray(self._get_patch_uint8(location, location_path, season))

        views = [self.transform(img) for _ in range(self.n_views)]
        return views, 0  # dummy label: pretraining is label-free, kept for DataLoader symmetry


class SeCoSeasonalDataset(Dataset):
    """
    SeCo's own positive-pair definition: two *different* seasonal revisits of the same
    location, each independently (optionally) augmented. This is the "seasonal positives"
    variant discussed in Notebook 1 -- a domain-specific alternative to synthetic
    augmentation-only positives, requiring at least 2 seasons per location.

    Locations with fewer than 2 available seasons are skipped at construction time.
    """

    def __init__(self, seco_root, transform, locations=None):
        self.seco_root = seco_root
        self.transform = transform

        candidate_locations = locations if locations is not None else list_locations(seco_root)
        self.locations = [
            loc for loc in candidate_locations
            if len(list_seasons(os.path.join(seco_root, loc))) >= 2
        ]
        if not self.locations:
            raise RuntimeError(
                f"No SeCo locations under {seco_root} have >= 2 seasons; "
                "SeCoSeasonalDataset requires multi-season data."
            )

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        location = self.locations[idx]
        location_path = os.path.join(self.seco_root, location)
        seasons = list_seasons(location_path)
        season_a, season_b = random.sample(seasons, 2)

        patch_a = to_pil_uint8(read_rgb_patch(os.path.join(location_path, season_a)))
        patch_b = to_pil_uint8(read_rgb_patch(os.path.join(location_path, season_b)))

        view_a = self.transform(patch_a)
        view_b = self.transform(patch_b)
        return [view_a, view_b], 0


def build_seco_augmentations(img_size=64, mean=SECO_MEAN, std=SECO_STD):
    """
    Pretraining-time augmentations for SeCo patches. Deliberately identical in spirit to
    `build_rs_augmentations` in src/remote_sensing/tutorial_rs.py (free rotation/flips, mild jitter -- see
    that function's docstring for the remote-sensing rationale), reimplemented here so this
    folder has no import-time dependency on the notebooks package beyond the model
    architecture itself.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_seco_mae_augmentations(img_size=64, mean=SECO_MEAN, std=SECO_STD, min_scale=0.5):
    """
    Augmentations for MAE pretraining: geometry only, no photometric jitter.

    MAE needs augmentation for a different reason than the contrastive/distillation methods.
    Those need it to *define* the pretext task (the positive pair). MAE's pretext task comes
    from masking alone, so it is tempting -- and this is a classic mistake -- to feed it
    deterministic views. But the SeCo 10% subset is only 2,000 locations x 5 seasons = 10,000
    patches; at batch 512 a 10k-step run is ~500 epochs over that fixed set. With no
    augmentation the model sees literally the same 10,000 tensors every epoch, and the
    encoder-decoder pair can lower reconstruction loss by memorizing those specific images
    instead of learning transferable structure. Random resized crops + flips/rotations make
    each epoch's views distinct and force the encoder to generalize.

    Photometric jitter is deliberately excluded (unlike `build_seco_augmentations`): MAE
    regresses raw pixel values, so jittering the input while the target is derived from that
    same jittered image would not corrupt the objective, but it does make the reconstruction
    target's radiometry arbitrary -- and, as in the original MAE recipe (He et al., 2022,
    which uses crop + flip only), it buys nothing. Geometry is what MAE needs.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_seco_eval_transform(img_size=64, mean=SECO_MEAN, std=SECO_STD):
    """Deterministic (no augmentation) transform, e.g. for a held-out reconstruction-loss check."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def read_manifest(manifest_path):
    """Read a location-ID manifest written by download_seco.py (one location ID per line)."""
    with open(manifest_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
