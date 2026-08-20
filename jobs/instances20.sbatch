#!/bin/bash
#SBATCH --job-name=3du_inst20
#SBATCH --partition=gpu,gpua100,gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jko26@jh.edu

# After wtours20 has stream_out (+ optional remask masks): sbatch jobs/instances20.sbatch
# CITY=amsterdam sbatch jobs/instances20.sbatch
source "${SLURM_SUBMIT_DIR:-.}/jobs/_env.sh"

CITY="${CITY:-amsterdam}"
IDX="$(printf '%03d' "${SLURM_ARRAY_TASK_ID:?}")"
SCENE="$CORK3DU_DATA/scenes/${CITY}_${IDX}"

if [[ ! -d "$SCENE/stream_out/results_output" ]]; then
  echo "missing $SCENE/stream_out — run jobs/wtours20.sbatch first" >&2
  exit 1
fi

python -m cork3du instances --scene "$SCENE"
echo "preview: $SCENE/instances/preview.png"
echo "html:    $SCENE/instances/preview.html"
echo "labeled: $SCENE/instances/cloud_labeled.npy"
echo "odisedb: $SCENE/instances/odise_debug/"
