"""
Evaluation and metrics module.

Loads a trained checkpoint and reports accuracy, precision, recall,
F1-score, and confusion matrix on the test split.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import IDX_TO_LABEL, GeometricDataset, build_dataloaders
from model import build_model
from preprocess import build_transforms


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> nn.Module:
    """Load model weights from a checkpoint file."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


@torch.no_grad()
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
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(preds)

    return np.array(all_true), np.array(all_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and return a metrics dictionary containing:
        accuracy, per-class precision/recall/f1, confusion matrix.
    """
    class_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL)]
    accuracy = (y_true == y_pred).mean()
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": accuracy, "report": report, "confusion_matrix": cm}


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
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sample_grid(
    splits_dir: str | Path,
    model: nn.Module,
    device: torch.device,
    img_size: int,
    n_per_class: int = 4,
    output_path: str | Path | None = None,
) -> None:
    """
    Show a grid of sample images per class with true and predicted labels.

    Args:
        splits_dir: Directory containing test.csv.
        model: Trained model.
        device: torch device.
        img_size: Target image resolution.
        n_per_class: Number of samples to display per class.
        output_path: If provided, save the figure here.
    """
    class_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL)]
    n_classes = len(class_names)
    transform = build_transforms(img_size=img_size, augment=False)
    dataset = GeometricDataset(Path(splits_dir) / "test.csv", transform=transform)

    # Collect n_per_class indices for each class
    class_indices: dict[int, list[int]] = {i: [] for i in range(n_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)
        if all(len(v) == n_per_class for v in class_indices.values()):
            break

    model.eval()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2 + 0.5))
    fig.suptitle("Per-class sample predictions (test set)", fontsize=12)

    for row, cls_idx in enumerate(range(n_classes)):
        for col, sample_idx in enumerate(class_indices[cls_idx]):
            img_tensor, true_label = dataset[sample_idx]
            with torch.no_grad():
                pred_label = model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).item()

            # Denormalize for display
            img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            ax = axes[row][col] if n_classes > 1 else axes[col]
            ax.imshow(img_display)
            color = "green" if pred_label == true_label else "red"
            ax.set_title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]}", fontsize=7, color=color)
            ax.axis("off")

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Sample grid saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        splits_dir=splits_dir,
        img_size=config["data"]["img_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = build_model(
        model_type=config["model"]["type"],
        img_size=config["data"]["img_size"],
        num_classes=config["model"]["num_classes"],
    )
    model = load_checkpoint(checkpoint_path, model, device)

    y_true, y_pred = predict(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred)

    class_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL)]
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}\n")
    print(metrics["report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        output_path="reports/confusion_matrix.png",
    )
    plot_sample_grid(
        splits_dir=splits_dir,
        model=model,
        device=device,
        img_size=config["data"]["img_size"],
        output_path="reports/sample_grid.png",
    )

    return metrics


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_evaluation(checkpoint_path="checkpoints/best.pt", config=cfg)
