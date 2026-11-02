#!/bin/bash
#SBATCH --job-name=agentlint-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-%x-%j.out
#
# Score a trained checkpoint on one test set, once. Run from the repo root on CARC, where the
# checkpoint's model weights are (they are not in git):
#   sbatch -A <account> --gres=gpu:a40:1 scripts/carc/score.slurm checkpoints/binary data/splits/full/test.jsonl
# Results go to <checkpoint>/eval/<tag>/. Extra arguments (e.g. --tag) are passed to `agentlint score`.
set -euo pipefail
CHECKPOINT=${1:?usage: sbatch -A <account> scripts/carc/score.slurm <checkpoint> <pairs.jsonl> [--tag T]}
PAIRS=${2:?usage: sbatch -A <account> scripts/carc/score.slurm <checkpoint> <pairs.jsonl> [--tag T]}
shift 2

cd "$SLURM_SUBMIT_DIR"
source scripts/carc/env.sh
export HF_HUB_OFFLINE=1   # model and tokenizer load from the checkpoint folder

echo "job $SLURM_JOB_ID on $(hostname) | score $CHECKPOINT on $PAIRS | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/agentlint score "$CHECKPOINT" "$PAIRS" "$@"
