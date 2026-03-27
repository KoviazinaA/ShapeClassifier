"""
Training loop module.

Handles model training with configurable optimizer, scheduler,
early stopping, and checkpoint saving.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_dataloaders
from model import build_model


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
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


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
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

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
        backbone_name=config["model"].get("backbone", "resnet18"),
        freeze_backbone=config["model"].get("freeze_backbone", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    best_val_loss = float("inf")
    patience = config["training"]["patience"]
    epochs_without_improvement = 0
    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> Checkpoint saved ({best_checkpoint})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    return model


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
