# Unit Test Instructions

No formal unit-test framework is configured for this PoC. The following manual smoke-tests verify each module in isolation.

## 1 — dataset.py

```bash
cd /data/testClaude/geometric_ml
python src/generate.py      # produces data/raw/
python src/preprocess.py    # produces data/splits/*.csv

python - <<'EOF'
import sys; sys.path.insert(0, "src")
from dataset import GeometricDataset, build_dataloaders, LABEL_TO_IDX

# Label alignment check
assert LABEL_TO_IDX["circle"] == 0 and LABEL_TO_IDX["triangle"] == 1, "Label mismatch!"

ds = GeometricDataset("data/splits/train.csv")
print(f"Train samples: {len(ds)}")
img, label = ds[0]
print(f"Sample shape: {img.size}, label: {label}")

loaders = build_dataloaders("data/splits", img_size=64, batch_size=8, num_workers=0)
for split, loader in loaders.items():
    batch_imgs, batch_labels = next(iter(loader))
    print(f"{split}: batch shape {batch_imgs.shape}, labels {batch_labels[:4].tolist()}")
print("dataset OK")
EOF
```

## 2 — model.py

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import torch
from model import GeometricCNN, build_model

model = GeometricCNN(img_size=64, num_classes=2)
x = torch.randn(4, 3, 64, 64)
out = model(x)
assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"

model2 = build_model("cnn", img_size=64, num_classes=2)
assert isinstance(model2, GeometricCNN)

try:
    build_model("unknown")
    assert False, "Should have raised"
except ValueError:
    pass

print("model OK")
EOF
```

## 3 — train.py

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import torch, yaml
from model import build_model
from dataset import build_dataloaders
from train import train_one_epoch, evaluate

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")
model = build_model("cnn", img_size=64, num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loaders = build_dataloaders("data/splits", img_size=64, batch_size=8, num_workers=0)

loss, acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
print(f"train one epoch: loss={loss:.4f} acc={acc:.4f}")

loss, acc = evaluate(model, loaders["val"], criterion, device)
print(f"evaluate val: loss={loss:.4f} acc={acc:.4f}")

print("train OK")
EOF
```

## 4 — evaluate.py

```bash
# Requires checkpoints/best.pt — run full training first (see build-instructions.md)
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import yaml
from evaluate import run_evaluation

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

metrics = run_evaluation("checkpoints/best.pt", cfg, splits_dir="data/splits")
assert "accuracy" in metrics and "confusion_matrix" in metrics
print(f"Test accuracy: {metrics['accuracy']:.4f}")
print("evaluate OK")
EOF
```
