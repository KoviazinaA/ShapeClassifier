# ShapeClassifier

A end-to-end image classification pipeline that distinguishes **triangles** from **circles** using a custom CNN and optional transfer learning.
The dataset is fully synthetic — generated with `matplotlib` — making the project self-contained and 100 % reproducible.

---

## Motivation

This project demonstrates a complete ML workflow on image data:

- Synthetic dataset generation with controlled variation
- Image preprocessing and train / val / test splitting
- Custom CNN architecture built with PyTorch
- Transfer learning with a frozen pretrained backbone (ResNet-18)
- Evaluation: accuracy, F1-score, confusion matrix

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)

---

## Project Structure

```
ShapeClassifier/
├── configs/
│   └── config.yaml          # All hyper-parameters in one place
├── data/
│   ├── raw/                 # Generated PNG images (triangle/, circle/)
│   ├── processed/           # Resized & normalised images
│   └── splits/              # train / val / test manifest CSVs
├── notebooks/
│   └── exploration.ipynb    # Visualisation & sanity-checks
├── src/
│   ├── generate.py          # Synthetic image generation
│   ├── preprocess.py        # Transforms, augmentation, dataset splitting
│   ├── dataset.py           # PyTorch Dataset + DataLoader factory
│   ├── model.py             # GeometricCNN and PretrainedClassifier
│   ├── train.py             # Training loop with early stopping
│   └── evaluate.py          # Metrics, confusion matrix, report
├── tests/
│   ├── test_generate.py
│   ├── test_preprocess.py
│   └── test_model.py
├── requirements.txt
└── setup.py
```

---

## Pipeline

```
generate.py  →  preprocess.py  →  dataset.py  →  train.py  →  evaluate.py
    │                │                │               │              │
 raw PNGs       splits + CSVs    DataLoaders      checkpoint      metrics
```

**1. Generate** — draws triangles (3 random non-collinear points) and circles (random centre + radius) with randomised fill/edge colours and optional Gaussian noise.

**2. Preprocess** — resizes to a fixed resolution, normalises pixel values, applies optional augmentations (horizontal flip, colour jitter), and writes stratified train / val / test manifests.

**3. Dataset** — `GeometricDataset` reads manifests and serves `(tensor, label)` pairs to PyTorch `DataLoader`.

**4. Train** — configurable optimiser, LR scheduler, and early stopping; saves the best validation checkpoint.

**5. Evaluate** — loads the checkpoint, runs inference on the test split, prints a `sklearn` classification report, and plots a confusion matrix.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (1 000 images per class by default)
python src/generate.py

# 3. Preprocess & split
python src/preprocess.py

# 4. Train
python src/train.py

# 5. Evaluate
python src/evaluate.py
```

All hyper-parameters (image size, number of samples, learning rate, …) live in [`configs/config.yaml`](configs/config.yaml).

---

## Models

| Model | Description |
|---|---|
| `GeometricCNN` | Lightweight custom CNN — conv layers + max-pool + FC head |
| `PretrainedClassifier` | Frozen ResNet-18 backbone with a custom classification head |

Switch between them in `configs/config.yaml`:
```yaml
model:
  type: cnn        # or "pretrained"
```

---

## Results

> _To be filled in after training._

| Model | Test Accuracy | F1 (macro) |
|---|---|---|
| GeometricCNN | — | — |
| ResNet-18 (frozen) | — | — |

---

## Roadmap

- [x] Synthetic data generation
- [x] Preprocessing pipeline
- [ ] GeometricCNN training
- [ ] Transfer learning (ResNet-18)
- [ ] Vision Transformer (ViT) comparison
- [ ] Streamlit demo

---

## Note

This project was also used to test [Claude Code](https://claude.ai/claude-code) — Anthropic's CLI — and its session configuration via `CLAUDE.md`.
