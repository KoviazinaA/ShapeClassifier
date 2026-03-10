"""Tests for the preprocessing module."""

import pytest

from src.preprocess import build_transforms, preprocess_image, split_dataset


class TestBuildTransforms:
    def test_returns_compose(self):
        pass

    def test_augment_adds_extra_steps(self):
        pass


class TestSplitDataset:
    def test_splits_sum_to_total(self, tmp_path):
        pass

    def test_manifests_are_written(self, tmp_path):
        pass

    def test_stratified_class_balance(self, tmp_path):
        pass


class TestPreprocessImage:
    def test_output_shape(self, tmp_path):
        pass

    def test_pixel_range(self, tmp_path):
        pass
