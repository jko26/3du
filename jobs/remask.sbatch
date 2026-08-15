#!/bin/bash
#SBATCH --job-name=3du_mask
#SBATCH --partition=gpua100,gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# SCENE=/path/scene sbatch jobs/remask.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"
: "${SCENE:?set SCENE=}"
python -m cork3du remask --scene "$SCENE"
