# AI-DLC Workflow Plan

## Context Summary

- **Project**: ShapeClassifier — geometric figure classification (circles vs triangles)
- **Type**: Brownfield — 2 modules implemented, 4 skeleton modules to complete
- **Model**: GeometricCNN (custom CNN only)
- **Security extension**: Disabled (PoC/portfolio project)
- **Evaluate output**: classification report + confusion matrix + per-class sample visualisation

## Decisions from Requirements Analysis

| Decision | Value |
|---|---|
| Security rules | Disabled |
| Model | GeometricCNN (custom CNN) |
| Evaluate output | Console report + confusion matrix + per-class samples |
| Bug to fix | Label-index mismatch: preprocess.py uses circle=0, dataset.py uses triangle=0 — align to circle=0, triangle=1 |

---

## Proposed Execution Plan

### INCEPTION PHASE
- [x] Workspace Detection
- [x] Requirements Analysis
- [ ] Workflow Planning ← current

### CONSTRUCTION PHASE — 4 units (implemented in order)

#### Unit 1: dataset.py
- Fix LABEL_TO_IDX to match preprocess.py (circle=0, triangle=1)
- Implement GeometricDataset.__init__ (read manifest CSV)
- Implement __len__ and __getitem__ (load PNG, apply transform)
- Implement build_dataloaders (train/val/test DataLoaders from config)

#### Unit 2: model.py
- Implement GeometricCNN.features (3 conv-BN-ReLU-MaxPool blocks)
- Implement GeometricCNN.classifier (FC head)
- Implement GeometricCNN.forward
- Implement build_model factory (cnn only, raise for unknown type)

#### Unit 3: train.py
- Implement train_one_epoch (forward, loss, backward, accuracy)
- Implement evaluate (no_grad pass, loss + accuracy)
- Implement train (full loop: optimizer, scheduler, early stopping, checkpoint)

#### Unit 4: evaluate.py
- Implement load_checkpoint
- Implement predict (collect y_true, y_pred)
- Implement compute_metrics (accuracy, classification_report, confusion_matrix)
- Implement plot_confusion_matrix (seaborn heatmap, save to disk)
- Add per-class sample visualisation grid
- Implement run_evaluation (end-to-end)

### BUILD AND TEST
- Verify imports resolve
- Smoke-test generation → preprocessing → dataset → model forward pass
- Document run instructions

---

## Stages NOT Executed (and Why)

| Stage | Decision |
|---|---|
| Reverse Engineering | Skipped — codebase fully known from session |
| User Stories | Skipped — no user-facing UI, pure ML pipeline |
| Application Design | Skipped — component boundaries already defined in skeletons |
| Units Generation | Skipped — units derived directly from existing skeleton modules |
| NFR Requirements | Skipped — PoC, no performance SLAs |
| NFR Design | Skipped — follows from NFR skip |
| Infrastructure Design | Skipped — local execution, no cloud infra |
| Security Extension | Disabled by user |
