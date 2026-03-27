"""Tests for the image generation module."""

import numpy as np
import pytest

from src.generate import LABELS, generate_circle, generate_dataset, generate_triangle


class TestGenerateTriangle:
    def test_returns_numpy_array(self):
        arr = generate_triangle(img_size=32)
        assert isinstance(arr, np.ndarray)

    def test_correct_shape(self):
        arr = generate_triangle(img_size=32)
        assert arr.shape == (32, 32, 3)

    def test_dtype_uint8(self):
        arr = generate_triangle(img_size=32)
        assert arr.dtype == np.uint8


class TestGenerateCircle:
    def test_returns_numpy_array(self):
        arr = generate_circle(img_size=32)
        assert isinstance(arr, np.ndarray)

    def test_correct_shape(self):
        arr = generate_circle(img_size=32)
        assert arr.shape == (32, 32, 3)

    def test_dtype_uint8(self):
        arr = generate_circle(img_size=32)
        assert arr.dtype == np.uint8


class TestGenerateDataset:
    def test_creates_output_dirs(self, tmp_path):
        generate_dataset(tmp_path, n_samples=2, img_size=16, seed=0)
        assert (tmp_path / "triangle").is_dir()
        assert (tmp_path / "circle").is_dir()

    def test_correct_number_of_images(self, tmp_path):
        generate_dataset(tmp_path, n_samples=3, img_size=16, seed=0)
        for label in LABELS:
            files = list((tmp_path / label).glob("*.png"))
            assert len(files) == 3

    def test_reproducible_with_seed(self, tmp_path):
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        generate_dataset(out_a, n_samples=2, img_size=16, seed=7)
        generate_dataset(out_b, n_samples=2, img_size=16, seed=7)
        for label in LABELS:
            files_a = sorted((out_a / label).glob("*.png"))
            files_b = sorted((out_b / label).glob("*.png"))
            for fa, fb in zip(files_a, files_b):
                from PIL import Image
                assert np.array_equal(np.array(Image.open(fa)), np.array(Image.open(fb)))
