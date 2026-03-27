"""
PyTorch Dataset for geometric figures.

Reads images (or preprocessed tensors) referenced in a manifest CSV
and exposes them as (tensor, label) pairs for DataLoader consumption.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocess import build_transforms


# Aligned with preprocess.py LABEL_MAP
LABEL_TO_IDX: dict[str, int] = {"circle": 0, "triangle": 1}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}


class GeometricDataset(Dataset):
    """Dataset that loads geometric figure images from a manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Args:
            manifest_path: Path to a CSV with columns: path, label.
            transform: Optional torchvision transform applied to each image.
        """
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    splits_dir: str | Path,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Construct DataLoaders for train / val / test splits.

    Returns:
        Dict with keys "train", "val", "test".
    """
    splits_dir = Path(splits_dir)
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        augment = split == "train"
        transform = build_transforms(img_size=img_size, augment=augment)
        dataset = GeometricDataset(splits_dir / f"{split}.csv", transform=transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders
