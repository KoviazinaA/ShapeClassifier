"""Tests for model definitions."""

import torch
import pytest

from src.model import GeometricCNN, build_model


class TestGeometricCNN:
    def test_forward_shape(self):
        model = GeometricCNN(img_size=64, num_classes=2)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 2)

    def test_output_classes(self):
        model = GeometricCNN(img_size=64, num_classes=3)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 3)

    def test_no_nan_in_output(self):
        model = GeometricCNN(img_size=64, num_classes=2)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert not torch.isnan(out).any()


class TestBuildModel:
    def test_returns_cnn_by_default(self):
        model = build_model("cnn", img_size=64, num_classes=2)
        assert isinstance(model, GeometricCNN)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            build_model("unknown")
