#!/bin/bash
#SBATCH --job-name=3du_ingest
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

# CITY=amsterdam N_CHUNKS=20 sbatch jobs/ingest_wtours.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"

CITY="${CITY:-amsterdam}"
N_CHUNKS="${N_CHUNKS:-20}"
CHUNK_SECONDS="${CHUNK_SECONDS:-20}"

python -m cork3du ingest-wtours \
  --city "$CITY" \
  --n-chunks "$N_CHUNKS" \
  --chunk-seconds "$CHUNK_SECONDS" \
  --out "$CORK3DU_DATA/chunks"
