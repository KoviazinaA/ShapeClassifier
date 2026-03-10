"""
Evaluation and metrics module.

Loads a trained checkpoint and reports accuracy, precision, recall,
F1-score, and confusion matrix on the test split.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import IDX_TO_LABEL, build_dataloaders
from model import build_model


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> nn.Module:
    """Load model weights from a checkpoint file."""
    pass


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

    Returns:
        (y_true, y_pred) as integer numpy arrays.
    """
    pass


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and return a metrics dictionary containing:
        accuracy, per-class precision/recall/f1, confusion matrix.
    """
    pass


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | Path | None = None,
) -> None:
    """
    Plot and optionally save a confusion matrix figure.

    Args:
        cm: Confusion matrix from sklearn.
        class_names: Ordered list of class label strings.
        output_path: If provided, save the figure here instead of showing it.
    """
    pass


def run_evaluation(
    checkpoint_path: str | Path,
    config: dict,
    splits_dir: str | Path = "data/splits",
) -> dict:
    """
    End-to-end evaluation: load model, run inference, print report.

    Returns:
        Metrics dictionary.
    """
    pass


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_evaluation(checkpoint_path="checkpoints/best.pt", config=cfg)
