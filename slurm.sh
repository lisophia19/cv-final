#!/bin/bash
#SBATCH --job-name=dinov2
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

uv run python temp.py

echo "Finished running temp.py"
