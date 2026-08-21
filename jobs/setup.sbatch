#!/bin/bash
#SBATCH --job-name=3du_setup
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# CPU only (no GPU). If partition=cpu is rejected, override:
#   sbatch -p YOUR_CPU_PARTITION jobs/setup.sbatch
# Submit from the clone root: mkdir -p logs && sbatch jobs/setup.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"
bash scripts/setup.sh
