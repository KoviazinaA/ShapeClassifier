"""Tests for the image generation module."""

import numpy as np
import pytest

from src.generate import LABELS, generate_circle, generate_dataset, generate_triangle


class TestGenerateTriangle:
    def test_returns_numpy_array(self):
        pass

    def test_correct_shape(self):
        pass

    def test_dtype_uint8(self):
        pass


class TestGenerateCircle:
    def test_returns_numpy_array(self):
        pass

    def test_correct_shape(self):
        pass

    def test_dtype_uint8(self):
        pass


class TestGenerateDataset:
    def test_creates_output_dirs(self, tmp_path):
        pass

    def test_correct_number_of_images(self, tmp_path):
        pass

    def test_reproducible_with_seed(self, tmp_path):
        pass
