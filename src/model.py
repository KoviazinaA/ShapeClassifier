"""
Classification model definitions.

Provides a lightweight CNN baseline and an optional
transfer-learning wrapper around a pretrained backbone.
"""

import torch
import torch.nn as nn
from torchvision import models


class GeometricCNN(nn.Module):
    """Small convolutional network for triangle / circle classification."""

    def __init__(self, img_size: int = 64, num_classes: int = 2) -> None:
        """
        Args:
            img_size: Spatial resolution of input images (square assumed).
            num_classes: Number of output classes.
        """
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # img_size → img_size/2
            conv_block(32, 64),   # → img_size/4
            conv_block(64, 128),  # → img_size/8
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


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
        weights_enum = models.get_model_weights(backbone_name).DEFAULT
        self.backbone = models.get_model(backbone_name, weights=weights_enum)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


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
    if model_type == "cnn":
        return GeometricCNN(img_size=img_size, num_classes=num_classes)
    if model_type == "pretrained":
        return PretrainedClassifier(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn' or 'pretrained'.")
