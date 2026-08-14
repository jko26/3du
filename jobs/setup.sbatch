#!/bin/bash
#SBATCH --job-name=3du_setup
#SBATCH --partition=gpu,gpua100,gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# Do not run setup.sh on a login node. Submit from the clone root:
#   mkdir -p logs && sbatch jobs/setup.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"
bash scripts/setup.sh
