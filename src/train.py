"""
Training loop module.

Handles model training with configurable optimizer, scheduler,
early stopping, and checkpoint saving.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import build_dataloaders


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training DataLoader.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    pass


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader without gradient updates.

    Returns:
        (avg_loss, accuracy).
    """
    pass


def train(
    config: dict,
    splits_dir: str | Path = "data/splits",
    checkpoint_dir: str | Path = "checkpoints",
) -> nn.Module:
    """
    Full training routine driven by a config dict.

    Args:
        config: Hyper-parameters (see configs/config.yaml for schema).
        splits_dir: Directory containing train/val/test manifest CSVs.
        checkpoint_dir: Where to save best model weights.

    Returns:
        Trained model (best validation checkpoint loaded).
    """
    pass


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
