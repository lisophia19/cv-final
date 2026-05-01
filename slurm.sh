#!/bin/bash
#SBATCH --job-name=dino_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=dino_train.out
#SBATCH --error=dino_train.err

python temp3.py

echo "Finished running temp3.py"
