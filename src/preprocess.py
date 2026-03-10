"""
Preprocessing module.

Reads raw images, applies transformations (resize, normalize, augment),
and writes processed tensors / split manifests to disk.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

LABEL_MAP = {"circle": 0, "triangle": 1}


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
    steps = []
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(steps)


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
    raw_dir = Path(raw_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    result: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}

    for label_name, label_idx in LABEL_MAP.items():
        images = sorted((raw_dir / label_name).glob("*.png"))
        images = [str(p) for p in images]
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        result["train"] += [(p, label_idx) for p in images[:n_train]]
        result["val"] += [(p, label_idx) for p in images[n_train: n_train + n_val]]
        result["test"] += [(p, label_idx) for p in images[n_train + n_val:]]

    for split_name, rows in result.items():
        rng.shuffle(rows)
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(rows)

    return result


def preprocess_image(image_path: str | Path, transform: transforms.Compose) -> np.ndarray:
    """Load a single image from disk, apply transform, return numpy array."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    return tensor.numpy()


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
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    transform = build_transforms(img_size=img_size, augment=augment)

    for label_name in LABEL_MAP:
        src_dir = raw_dir / label_name
        dst_dir = output_dir / label_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_dir.glob("*.png")):
            array = preprocess_image(img_path, transform)
            out_path = dst_dir / (img_path.stem + ".npy")
            np.save(out_path, array)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("configs/config.yaml"))
    split_dataset(
        raw_dir=cfg["data"]["raw_dir"],
        splits_dir=cfg["data"]["splits_dir"],
        seed=cfg["data"]["seed"],
    )
    preprocess_all(
        raw_dir=cfg["data"]["raw_dir"],
        output_dir=cfg["data"]["processed_dir"],
        img_size=cfg["data"]["img_size"],
        augment=cfg["augmentation"]["enabled"],
    )
