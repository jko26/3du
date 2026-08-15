#!/bin/bash
#SBATCH --job-name=3du_run
#SBATCH --partition=gpua100,gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# VIDEO=/path.mp4 OUT=/path/scene sbatch jobs/run_scene.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"
: "${VIDEO:?set VIDEO=}"
: "${OUT:?set OUT=}"
python -m cork3du run --video "$VIDEO" --out "$OUT"
