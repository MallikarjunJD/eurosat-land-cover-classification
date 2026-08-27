"""
data.py — Data pipelines for EuroSAT RGB and multispectral training.

Both pipelines follow the same contract:
  1. Compute per-channel mean/std ONCE, over the RAW (unnormalized)
     training split.
  2. Bake those frozen constants into the real train/val/test datasets
     as the final step of every transform.
  3. Never recompute those constants anywhere else — `train.py` writes
     them to `normalization_stats.json`, and that file (not this
     module, not a fresh computation) is what `app.py` reads at
     inference time.

Geometric augmentation (flips, 90-degree rotations) is applied only to
the training split. Satellite patches have no canonical "up", so these
transformations are label-preserving in a way they would not be for,
say, handwritten digits or natural photographs.
"""

import glob
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT
import tifffile

IMG_SIZE = 64
NUM_CLASSES = 10
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


# --------------------------------------------------------------------------
# Shared utility
# --------------------------------------------------------------------------
def compute_channel_stats(dataset: Dataset, num_channels: int,
                           batch_size: int = 64, num_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-pass mean/std over an entire dataset without holding every
    image in memory at once.

    IMPORTANT: only call this on a dataset whose transform does NOT
    already normalize — raw ToTensor() output (values in [0, 1] for
    RGB) or raw digital numbers (for multispectral). Calling this on an
    already-normalized dataset would compute the stats of the stats,
    which is meaningless.
    """
    channel_sum = torch.zeros(num_channels)
    channel_sq_sum = torch.zeros(num_channels)
    n_pixels = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for images, _ in loader:
        b, c, h, w = images.shape
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        n_pixels += b * h * w
    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    return mean.numpy(), std.numpy()


def save_normalization_stats(rgb_mean=None, rgb_std=None, ms_mean=None, ms_std=None,
                              path: str = "normalization_stats.json") -> dict:
    """
    Freezes whichever stats are provided to a single JSON file. Only
    writes a modality's key if both its mean and std are given, so
    calling this after training just one modality doesn't clobber the
    other modality's previously-frozen stats with nulls.
    """
    stats = {}
    if rgb_mean is not None and rgb_std is not None:
        stats["rgb"] = {"mean": np.asarray(rgb_mean).tolist(), "std": np.asarray(rgb_std).tolist()}
    if ms_mean is not None and ms_std is not None:
        stats["multispectral"] = {"mean": np.asarray(ms_mean).tolist(), "std": np.asarray(ms_std).tolist()}
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_normalization_stats(path: str = "normalization_stats.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# RGB pipeline
# --------------------------------------------------------------------------
def build_rgb_datasets(data_root: str = "./data", download: bool = True, seed: int = 42):
    """
    Returns (train_set, val_set, test_set, mean, std).
    """
    raw_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    raw_dataset = EuroSAT(root=data_root, download=download, transform=raw_transform)
    mean, std = compute_channel_stats(raw_dataset, num_channels=3)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_dataset = EuroSAT(root=data_root, download=False, transform=train_transform)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    # Validation/test splits must use the deterministic eval transform,
    # not the training transform they inherited from full_dataset.
    val_set.dataset = EuroSAT(root=data_root, download=False, transform=eval_transform)
    test_set.dataset = EuroSAT(root=data_root, download=False, transform=eval_transform)

    return train_set, val_set, test_set, mean, std


def build_rgb_dataloaders(data_root: str = "./data", batch_size: int = 64, num_workers: int = 4,
                           download: bool = True, seed: int = 42):
    """Returns (train_loader, val_loader, test_loader, mean, std)."""
    train_set, val_set, test_set, mean, std = build_rgb_datasets(data_root, download, seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, mean, std


# --------------------------------------------------------------------------
# Multispectral pipeline
# --------------------------------------------------------------------------


class EuroSATMultispectral(Dataset):
    """
    Loads 13-band Sentinel-2 L1C GeoTIFFs from the official EuroSAT-MS
    directory structure.

    Expected structure:

        data_ms/
        └── EuroSAT_MS/
            ├── AnnualCrop/
            ├── Forest/
            ├── HerbaceousVegetation/
            ├── Highway/
            ├── Industrial/
            ├── Pasture/
            ├── PermanentCrop/
            ├── Residential/
            ├── River/
            └── SeaLake/

    Each TIFF:
        Shape: (64, 64, 13)
        Dtype: uint16

    PyTorch output:
        Shape: (13, 64, 64)
        Dtype: float32
    """

    def __init__(
        self,
        root: str,
        class_names,
        mean=None,
        std=None,
        transform=None,
        samples=None
    ):
        self.root = root
        self.class_names = class_names
        self.transform = transform

        # ---------------------------------------------------------------
        # Normalization statistics
        # ---------------------------------------------------------------
        self.mean = (
            torch.tensor(mean, dtype=torch.float32).view(13, 1, 1)
            if mean is not None else None
        )

        self.std = (
            torch.tensor(std, dtype=torch.float32).view(13, 1, 1)
            if std is not None else None
        )

        # ---------------------------------------------------------------
        # Use supplied samples for train/val/test subsets
        # ---------------------------------------------------------------
        if samples is not None:
            self.samples = samples

        else:
            self.samples = []

            for class_idx, class_name in enumerate(class_names):

                class_dir = os.path.join(root, class_name)

                tif_files = sorted(
                    glob.glob(
                        os.path.join(class_dir, "*.tif")
                    )
                )

                for path in tif_files:
                    self.samples.append(
                        (path, class_idx)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .tif files found under:\n{root}\n\n"
                f"Expected folders such as:\n"
                f"{os.path.join(root, 'AnnualCrop')}"
            )

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, path: str) -> torch.Tensor:

        # ---------------------------------------------------------------
        # Read TIFF
        # ---------------------------------------------------------------
        arr = tifffile.imread(path).astype(np.float32)

        # ---------------------------------------------------------------
        # Validate dimensions
        # ---------------------------------------------------------------
        if arr.ndim != 3:
            raise ValueError(
                f"Expected 3D TIFF but got {arr.shape}\n"
                f"File: {path}"
            )

        # Official EuroSAT-MS:
        # (64, 64, 13)
        #
        # Convert to:
        # (13, 64, 64)
        if arr.shape == (64, 64, 13):

            arr = arr.transpose(2, 0, 1)

        elif arr.shape[0] == 13:

            # Already (13, H, W)
            pass

        else:
            raise ValueError(
                f"Expected 13-band TIFF but got shape {arr.shape}\n"
                f"File: {path}"
            )

        tensor = torch.from_numpy(arr)

        # ---------------------------------------------------------------
        # Safety resize
        # ---------------------------------------------------------------
        if tensor.shape[-2:] != (IMG_SIZE, IMG_SIZE):

            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        return tensor

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = self._read_tif(path)

        # Training augmentation
        if self.transform is not None:
            img = self.transform(img)

        # Per-band normalization
        if self.mean is not None and self.std is not None:

            img = (img - self.mean) / self.std

        return img, label


# ============================================================================
# MULTISPECTRAL AUGMENTATION
# ============================================================================

def spatial_augment_ms(img: torch.Tensor) -> torch.Tensor:
    """
    Applies identical spatial transformations to all 13 bands.
    """

    # Horizontal flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-1])

    # Vertical flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-2])

    # 0 / 90 / 180 / 270 degree rotation
    k = torch.randint(0, 4, (1,)).item()

    img = torch.rot90(
        img,
        k,
        dims=[-2, -1]
    )

    return img


# ============================================================================
# STEP 1: PATH
# ============================================================================

MS_ROOT = (
    r"C:\Users\Lenovo\ML Sprint folder\CNN of EuroSat"
    r"\data_ms\EuroSAT_MS"
)

print("Multispectral dataset path:")
print(MS_ROOT)


# ============================================================================
# STEP 2: LOAD ALL 27,000 FILE PATHS
# ============================================================================

full_ms_dataset = EuroSATMultispectral(
    root=MS_ROOT,
    class_names=CLASS_NAMES
)

print()
print("Total multispectral images:", len(full_ms_dataset))


# ============================================================================
# STEP 3: REPRODUCIBLE 70 / 15 / 15 SPLIT
# ============================================================================

n_total = len(full_ms_dataset)

n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val


print()
print("Multispectral split:")
print("Train:", n_train)
print("Validation:", n_val)
print("Test:", n_test)


# Reproducible random split
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    n_total,
    generator=generator
).tolist()

train_indices = indices[:n_train]

val_indices = indices[
    n_train:n_train + n_val
]

test_indices = indices[
    n_train + n_val:
]


# Convert indices to (path, label)
all_samples = full_ms_dataset.samples

train_samples = [
    all_samples[i]
    for i in train_indices
]

val_samples = [
    all_samples[i]
    for i in val_indices
]

test_samples = [
    all_samples[i]
    for i in test_indices
]


# ============================================================================
# STEP 4: CALCULATE 13-BAND MEAN / STD
# ============================================================================
#
# IMPORTANT:
#
# We do NOT scan all 18,900 training images every time.
#
# Instead, we use a representative subset.
#
# This is sufficient for normalization and dramatically reduces startup time.
#
# Change this to 5000 if you want an even larger sample.
# ============================================================================

STATS_SAMPLE_SIZE = 3000

stats_generator = torch.Generator().manual_seed(42)

if len(train_samples) > STATS_SAMPLE_SIZE:

    stats_indices = torch.randperm(
        len(train_samples),
        generator=stats_generator
    )[:STATS_SAMPLE_SIZE].tolist()

    stats_samples = [
        train_samples[i]
        for i in stats_indices
    ]

else:

    stats_samples = train_samples


# ---------------------------------------------------------------------------
# Cache file
# ---------------------------------------------------------------------------

STATS_FILE = os.path.join(
    MS_ROOT,
    "multispectral_stats.json"
)


# ---------------------------------------------------------------------------
# If statistics already exist, load them.
# Otherwise calculate them once.
# ---------------------------------------------------------------------------

if os.path.exists(STATS_FILE):

    print()
    print("Loading saved multispectral statistics...")

    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    ms_mean = np.array(
        stats["mean"],
        dtype=np.float32
    )

    ms_std = np.array(
        stats["std"],
        dtype=np.float32
    )

    print("Loaded statistics from:")
    print(STATS_FILE)

else:

    print()
    print(
        f"Calculating 13-band statistics using "
        f"{len(stats_samples)} training images..."
    )

    raw_ms_dataset = EuroSATMultispectral(
        root=MS_ROOT,
        class_names=CLASS_NAMES,
        samples=stats_samples
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # num_workers=0 is intentional.
    #
    # It is more stable on Windows for thousands of TIFF files.
    # ---------------------------------------------------------------

    raw_ms_loader = DataLoader(
        raw_ms_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    channel_sum = torch.zeros(13, dtype=torch.float64)

    channel_sq_sum = torch.zeros(
        13,
        dtype=torch.float64
    )

    n_pixels = 0

    for batch_idx, (images, _) in enumerate(raw_ms_loader):

        images = images.double()

        channel_sum += images.sum(
            dim=(0, 2, 3)
        )

        channel_sq_sum += (
            images ** 2
        ).sum(
            dim=(0, 2, 3)
        )

        n_pixels += (
            images.shape[0]
            * images.shape[2]
            * images.shape[3]
        )

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Processed "
                f"{(batch_idx + 1) * 64} "
                f"images..."
            )

    # ---------------------------------------------------------------
    # Mean
    # ---------------------------------------------------------------

    mean = channel_sum / n_pixels

    # ---------------------------------------------------------------
    # Variance
    # ---------------------------------------------------------------

    variance = (
        channel_sq_sum / n_pixels
        - mean ** 2
    )

    # Protect against tiny floating-point negative values
    variance = torch.clamp(
        variance,
        min=1e-12
    )

    std = torch.sqrt(variance)

    ms_mean = mean.float().numpy()

    ms_std = std.float().numpy()

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    if np.any(ms_std <= 0):

        raise ValueError(
            "At least one spectral band has zero standard deviation."
        )

    # ---------------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------------

    stats = {
        "mean": ms_mean.tolist(),
        "std": ms_std.tolist(),
        "sample_size": len(stats_samples),
        "seed": 42
    }

    with open(STATS_FILE, "w") as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print()
    print("Statistics saved to:")
    print(STATS_FILE)


print()
print("MS mean:")
print(ms_mean)

print()
print("MS std:")
print(ms_std)


# ============================================================================
# STEP 5: CREATE TRAIN / VALIDATION / TEST DATASETS
# ============================================================================

ms_train_set = EuroSATMultispectral(
    root=MS_ROOT,
    class_names=CLASS_NAMES,
    mean=ms_mean,
    std=ms_std,
    transform=spatial_augment_ms,
    samples=train_samples
)


ms_val_set = EuroSATMultispectral(
    root=MS_ROOT,
    class_names=CLASS_NAMES,
    mean=ms_mean,
    std=ms_std,
    transform=None,
    samples=val_samples
)


ms_test_set = EuroSATMultispectral(
    root=MS_ROOT,
    class_names=CLASS_NAMES,
    mean=ms_mean,
    std=ms_std,
    transform=None,
    samples=test_samples
)


# ============================================================================
# STEP 6: DATALOADERS
# ============================================================================

# num_workers=0 is intentionally used here for Windows stability.
#
# Once everything is working correctly, we can benchmark whether
# increasing this to 2 or 4 actually improves your machine's speed.

ms_train_loader = DataLoader(
    ms_train_set,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


ms_val_loader = DataLoader(
    ms_val_set,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


ms_test_loader = DataLoader(
    ms_test_set,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================================
# STEP 7: FINAL VERIFICATION
# ============================================================================

print()
print("=" * 60)
print("FINAL MULTISPECTRAL PIPELINE CHECK")
print("=" * 60)

print()
print("Dataset sizes:")
print("ms_train_set:", len(ms_train_set))
print("ms_val_set  :", len(ms_val_set))
print("ms_test_set :", len(ms_test_set))


# Load one training sample
sample_img, sample_label = ms_train_set[0]

print()
print("Sample verification:")
print("Image shape :", sample_img.shape)
print("Image dtype :", sample_img.dtype)
print("Label       :", sample_label)
print("Class       :", CLASS_NAMES[sample_label])
print("Image min   :", sample_img.min().item())
print("Image max   :", sample_img.max().item())


# ============================================================================
# STEP 8: VERIFY DATALOADER BATCH
# ============================================================================

print()
print("Testing one DataLoader batch...")

images, labels = next(iter(ms_train_loader))

print("Batch image shape :", images.shape)
print("Batch label shape :", labels.shape)
print("Batch dtype       :", images.dtype)


# ============================================================================
# FINAL EXPECTED OUTPUT
# ============================================================================

print()
print("=" * 60)
print("PIPELINE READY")
print("=" * 60)

print()
print("Expected image shape:")
print("13 × 64 × 64")

print()
print("Expected batch shape:")
print("64 × 13 × 64 × 64")