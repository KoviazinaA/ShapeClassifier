# Build Instructions

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
# From workspace root
pip install -r requirements.txt
```

## Verify Imports

```bash
cd /data/testClaude/geometric_ml/src
python -c "from dataset import GeometricDataset, build_dataloaders; print('dataset OK')"
python -c "from model import GeometricCNN, build_model; print('model OK')"
python -c "from train import train_one_epoch, evaluate, train; print('train OK')"
python -c "from evaluate import run_evaluation; print('evaluate OK')"
```

## Full Pipeline Run

```bash
cd /data/testClaude/geometric_ml

# Step 1 — generate raw images
python src/generate.py

# Step 2 — preprocess (resize, normalise, split into train/val/test CSVs)
python src/preprocess.py

# Step 3 — train
python src/train.py

# Step 4 — evaluate (requires checkpoints/best.pt)
python src/evaluate.py
```

Outputs:
- `checkpoints/best.pt` — best model weights
- `reports/confusion_matrix.png` — confusion matrix figure
- `reports/sample_grid.png` — per-class sample predictions grid
