# Integration Test Instructions

Verifies the full pipeline end-to-end: generate → preprocess → train → evaluate.

```bash
cd /data/testClaude/geometric_ml

# Full pipeline integration test
python - <<'EOF'
import sys, yaml, shutil, pathlib
sys.path.insert(0, "src")

# Clean slate
for d in ("data/raw", "data/processed", "data/splits", "checkpoints", "reports"):
    shutil.rmtree(d, ignore_errors=True)

# 1. Generate
import generate
generate.main()
assert pathlib.Path("data/raw").exists(), "data/raw missing"
print("generate: OK")

# 2. Preprocess
import preprocess
preprocess.main()
for split in ("train.csv", "val.csv", "test.csv"):
    assert pathlib.Path(f"data/splits/{split}").exists(), f"{split} missing"
print("preprocess: OK")

# 3. Train (short run — override epochs and patience for speed)
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["epochs"] = 2
cfg["training"]["patience"] = 2

from train import train
model = train(cfg, splits_dir="data/splits", checkpoint_dir="checkpoints")
assert pathlib.Path("checkpoints/best.pt").exists(), "checkpoint missing"
print("train: OK")

# 4. Evaluate
from evaluate import run_evaluation
metrics = run_evaluation("checkpoints/best.pt", cfg, splits_dir="data/splits")
assert 0.0 <= metrics["accuracy"] <= 1.0
assert pathlib.Path("reports/confusion_matrix.png").exists()
assert pathlib.Path("reports/sample_grid.png").exists()
print(f"evaluate: OK  (accuracy={metrics['accuracy']:.4f})")

print("\nFull pipeline integration: PASSED")
EOF
```
