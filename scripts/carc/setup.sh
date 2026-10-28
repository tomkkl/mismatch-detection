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
echo "Downloading GraphCodeBERT to $HF_HOME ..."
.venv/bin/python - <<'EOF'
import yaml
from huggingface_hub import snapshot_download

# Download only: importing transformers' model code pulls in torch and scipy, which takes
# minutes on the login node's network filesystem.
snapshot_download("microsoft/graphcodebert-base", allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])

# Same smoke test as notebooks/train_binary.ipynb: 1 epoch on 64 train / 32 val pairs.
cfg = yaml.safe_load(open("configs/binary.yaml"))
cfg.update(epochs=1, limit_train=64, limit_val=32, output_dir="checkpoints/smoke")
yaml.safe_dump(cfg, open("runs/smoke.yaml", "w"), sort_keys=False)
EOF

echo "Running tests (the first import of torch can take a few minutes here; it is not stuck) ..."
.venv/bin/pytest -q
echo "Setup done. Next: sbatch -A <account> --time=00:30:00 scripts/carc/train.slurm runs/smoke.yaml"
