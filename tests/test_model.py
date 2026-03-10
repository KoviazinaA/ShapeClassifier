"""Tests for model definitions."""

import torch
import pytest

from src.model import GeometricCNN, PretrainedClassifier, build_model


class TestGeometricCNN:
    def test_forward_shape(self):
        pass

    def test_output_classes(self):
        pass

    def test_no_nan_in_output(self):
        pass


class TestPretrainedClassifier:
    def test_forward_shape(self):
        pass

    def test_backbone_frozen(self):
        pass


class TestBuildModel:
    def test_returns_cnn_by_default(self):
        pass

    def test_returns_pretrained_when_requested(self):
        pass

    def test_unknown_type_raises(self):
        pass
