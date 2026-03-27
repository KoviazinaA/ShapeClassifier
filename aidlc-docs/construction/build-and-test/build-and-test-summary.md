# Build and Test Summary

## Project Status

All 4 construction units are complete.

| Module | Status |
|---|---|
| src/generate.py | Implemented (prior session) |
| src/preprocess.py | Implemented (prior session) |
| src/dataset.py | Implemented |
| src/model.py | Implemented |
| src/train.py | Implemented |
| src/evaluate.py | Implemented |

## Key Design Decisions

| Decision | Value |
|---|---|
| Model | GeometricCNN (custom 3-block CNN + AdaptiveAvgPool + FC head) |
| Label alignment | circle=0, triangle=1 (matches preprocess.py) |
| Evaluate output | Console report + confusion matrix PNG + sample grid PNG |
| Hyper-parameters | All in configs/config.yaml — none hardcoded |
| Early stopping | Patience configurable via `training.patience` |

## Quick Run

```bash
pip install -r requirements.txt
python src/generate.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Test Files

| File | Purpose |
|---|---|
| build-instructions.md | Setup and full pipeline run |
| unit-test-instructions.md | Per-module smoke tests |
| integration-test-instructions.md | End-to-end pipeline test |
