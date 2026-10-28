#!/bin/bash
#SBATCH --job-name=agentlint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#
# Train one config on one GPU. Run from the repo root after scripts/carc/setup.sh:
#   sbatch -A <account> scripts/carc/train.slurm configs/binary.yaml
#   sbatch -A <account> scripts/carc/train.slurm configs/binary_frozen.yaml
# `myaccount` lists your CARC accounts. Pick a GPU type with e.g. --gres=gpu:a100:1.
set -euo pipefail
CONFIG=${1:?usage: sbatch -A <account> scripts/carc/train.slurm <config.yaml>}

cd "$SLURM_SUBMIT_DIR"
source scripts/carc/env.sh
export HF_HUB_OFFLINE=1   # the model was downloaded by setup.sh

echo "job $SLURM_JOB_ID on $(hostname) | config $CONFIG | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/agentlint train "$CONFIG"
