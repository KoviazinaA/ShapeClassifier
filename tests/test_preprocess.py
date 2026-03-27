"""Tests for the preprocessing module."""

import csv

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from src.preprocess import build_transforms, preprocess_image, split_dataset


class TestBuildTransforms:
    def test_returns_compose(self):
        t = build_transforms(img_size=64)
        assert isinstance(t, transforms.Compose)

    def test_augment_adds_extra_steps(self):
        t_plain = build_transforms(img_size=64, augment=False)
        t_aug = build_transforms(img_size=64, augment=True)
        assert len(t_aug.transforms) > len(t_plain.transforms)


class TestSplitDataset:
    def _make_raw(self, tmp_path, n_per_class: int = 10):
        raw = tmp_path / "raw"
        for label in ("circle", "triangle"):
            d = raw / label
            d.mkdir(parents=True)
            for i in range(n_per_class):
                Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(d / f"{label}_{i:04d}.png")
        return raw

    def test_splits_sum_to_total(self, tmp_path):
        raw = self._make_raw(tmp_path, n_per_class=10)
        splits_dir = tmp_path / "splits"
        result = split_dataset(raw, splits_dir)
        total = sum(len(v) for v in result.values())
        assert total == 20  # 10 per class

    def test_manifests_are_written(self, tmp_path):
        raw = self._make_raw(tmp_path, n_per_class=10)
        splits_dir = tmp_path / "splits"
        split_dataset(raw, splits_dir)
        for split in ("train", "val", "test"):
            assert (splits_dir / f"{split}.csv").exists()

    def test_stratified_class_balance(self, tmp_path):
        raw = self._make_raw(tmp_path, n_per_class=10)
        splits_dir = tmp_path / "splits"
        split_dataset(raw, splits_dir)
        for split in ("train", "val", "test"):
            csv_path = splits_dir / f"{split}.csv"
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            labels = [int(r["label"]) for r in rows]
            # Both classes must appear in each split
            assert 0 in labels and 1 in labels


class TestPreprocessImage:
    def test_output_shape(self, tmp_path):
        img_path = tmp_path / "test.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_path)
        t = build_transforms(img_size=16)
        arr = preprocess_image(img_path, t)
        assert arr.shape == (3, 16, 16)

    def test_pixel_range(self, tmp_path):
        img_path = tmp_path / "test.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_path)
        t = build_transforms(img_size=16)
        arr = preprocess_image(img_path, t)
        assert arr.min() >= -1.0 and arr.max() <= 1.0
