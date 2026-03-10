"""
PyTorch Dataset for geometric figures.

Reads images (or preprocessed tensors) referenced in a manifest CSV
and exposes them as (tensor, label) pairs for DataLoader consumption.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocess import build_transforms


LABEL_TO_IDX: dict[str, int] = {"triangle": 0, "circle": 1}
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
            manifest_path: Path to a CSV with columns: image_path, label_index.
            transform: Optional torchvision transform applied to each image.
        """
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pass


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
    pass
