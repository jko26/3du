#!/bin/bash
#SBATCH --job-name=3du_wt20
#SBATCH --partition=gpu,gpua100,gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# After ingest: sbatch jobs/wtours20.sbatch
# CITY=amsterdam sbatch jobs/wtours20.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"

CITY="${CITY:-amsterdam}"
IDX="$(printf '%03d' "${SLURM_ARRAY_TASK_ID:?}")"
VIDEO="$CORK3DU_DATA/chunks/${CITY}/${IDX}.mp4"
OUT="$CORK3DU_DATA/scenes/${CITY}_${IDX}"

if [[ ! -f "$VIDEO" ]]; then
  echo "missing chunk $VIDEO — run jobs/ingest_wtours.sbatch first" >&2
  exit 1
fi

python -m cork3du run --video "$VIDEO" --out "$OUT"
echo "debug: $OUT/masks/mask_debug.mp4"
echo "preview: $OUT/preview.png"
