#!/usr/bin/env bash
# Shared by sbatch scripts. Source after #SBATCH headers.
# Submit from the repo root:  sbatch jobs/ingest_wtours.sbatch
set -euo pipefail

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CORK3DU_ROOT="${CORK3DU_ROOT:-$(cd "$JOBS_DIR/.." && pwd)}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_WEIGHTS="${CORK3DU_WEIGHTS:-$CORK3DU_DATA/weights}"
export CORK3DU_DA3="${CORK3DU_DA3:-$CORK3DU_ROOT/third_party/Depth-Anything-3}"
export CORK3DU_SAM2="${CORK3DU_SAM2:-$CORK3DU_ROOT/third_party/sam2}"
export CORK3DU_SAM2_CKPT="${CORK3DU_SAM2_CKPT:-$CORK3DU_WEIGHTS/sam2.1_hiera_large.pt}"
export PYTHONPATH="${CORK3DU_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "$CORK3DU_ROOT/logs" "$CORK3DU_DATA"/{chunks,scenes,weights,logs}

if [[ -f "$CORK3DU_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$CORK3DU_ROOT/.venv/bin/activate"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  :
fi

cd "$CORK3DU_ROOT"
