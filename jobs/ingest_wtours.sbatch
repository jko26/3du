#!/bin/bash
#SBATCH --job-name=3du_ingest
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# CPU only: Hugging Face download + ffmpeg splits. No GPU.
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
