"""
Preprocessing module.

Reads raw images, applies transformations (resize, normalize, augment),
and writes processed tensors / split manifests to disk.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


def build_transforms(
    img_size: int,
    augment: bool = False,
) -> transforms.Compose:
    """
    Return a torchvision transform pipeline.

    Args:
        img_size: Target spatial resolution (square).
        augment: If True, include random flips and colour jitter.
    """
    pass


def split_dataset(
    raw_dir: str | Path,
    splits_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    """
    Scan raw_dir for images, stratify-split into train/val/test,
    and save manifest CSV files under splits_dir.

    Returns:
        Dict with keys "train", "val", "test" mapping to
        lists of (image_path, label_index) tuples.
    """
    pass


def preprocess_image(image_path: str | Path, transform: transforms.Compose) -> np.ndarray:
    """Load a single image from disk, apply transform, return numpy array."""
    pass


def preprocess_all(
    raw_dir: str | Path,
    output_dir: str | Path,
    img_size: int = 64,
    augment: bool = False,
) -> None:
    """
    Preprocess every image in raw_dir and save results to output_dir,
    preserving the label sub-folder structure.
    """
    pass


if __name__ == "__main__":
    split_dataset(raw_dir="data/raw", splits_dir="data/splits")
    preprocess_all(raw_dir="data/raw", output_dir="data/processed")
