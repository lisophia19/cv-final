#!/bin/bash
#
# Submit with:
#   sbatch yolo_train.slurm
# Or set a custom job name:
#   sbatch --job-name=yolo_train11s yolo_train.slurm
#   sbatch --job-name=yolo_train11n yolo_train.slurm
#   sbatch --job-name=yolo_tune11s yolo_train.slurm

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J yolo_train
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

# Use submit directory in SLURM jobs; fallback for local runs.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project dir: ${PROJECT_DIR}"

# Activate local virtual environment if present.
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# script
PY_SCRIPT="${PY_SCRIPT:-yolo.py}"

python "${PY_SCRIPT}"

echo "End time: $(date)"
