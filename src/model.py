"""
Classification model definitions.

Provides a lightweight CNN baseline and an optional
transfer-learning wrapper around a pretrained backbone.
"""

import torch
import torch.nn as nn


class GeometricCNN(nn.Module):
    """Small convolutional network for triangle / circle classification."""

    def __init__(self, img_size: int = 64, num_classes: int = 2) -> None:
        """
        Args:
            img_size: Spatial resolution of input images (square assumed).
            num_classes: Number of output classes.
        """
        super().__init__()
        # TODO: define convolutional feature extractor
        self.features: nn.Sequential
        # TODO: define classification head
        self.classifier: nn.Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PretrainedClassifier(nn.Module):
    """Transfer-learning wrapper: frozen backbone + custom head."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        """
        Args:
            backbone_name: torchvision model name (e.g. "resnet18", "efficientnet_b0").
            num_classes: Number of output classes.
            freeze_backbone: If True, backbone weights are not updated during training.
        """
        super().__init__()
        # TODO: load pretrained backbone from torchvision.models
        self.backbone: nn.Module
        self.head: nn.Linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def build_model(
    model_type: str = "cnn",
    img_size: int = 64,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function that returns the requested model.

    Args:
        model_type: "cnn" for GeometricCNN or "pretrained" for PretrainedClassifier.
        img_size: Passed to GeometricCNN.
        num_classes: Number of output classes.
        **kwargs: Extra arguments forwarded to the model constructor.
    """
    pass
