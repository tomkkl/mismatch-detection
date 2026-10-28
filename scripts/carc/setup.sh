#!/bin/bash
# One-time setup on a CARC login node. Run from the repo root:
#   bash scripts/carc/setup.sh
# Builds .venv, downloads GraphCodeBERT into $HF_HOME, writes the smoke-test config and runs
# the tests (CPU only, ~30 s).
set -euo pipefail
source scripts/carc/env.sh

python3 -c 'import sys; assert sys.version_info >= (3, 10), f"need Python 3.10+, got {sys.version}"'
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

mkdir -p "$HF_HOME" runs
.venv/bin/python - <<'EOF'
import yaml
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
AutoModel.from_pretrained("microsoft/graphcodebert-base")

# Same smoke test as notebooks/train_binary.ipynb: 1 epoch on 64 train / 32 val pairs.
cfg = yaml.safe_load(open("configs/binary.yaml"))
cfg.update(epochs=1, limit_train=64, limit_val=32, output_dir="checkpoints/smoke")
yaml.safe_dump(cfg, open("runs/smoke.yaml", "w"), sort_keys=False)
EOF

.venv/bin/pytest -q
echo "Setup done. Next: sbatch -A <account> --time=00:30:00 scripts/carc/train.slurm runs/smoke.yaml"
