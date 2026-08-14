#!/usr/bin/env bash
# Shared by sbatch scripts. Source after #SBATCH headers.
# Submit from the repo root:  sbatch jobs/ingest_wtours.sbatch
set -euo pipefail

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$JOBS_DIR/_cluster_modules.sh"
load_cork3du_modules

export CORK3DU_ROOT="${CORK3DU_ROOT:-$(cd "$JOBS_DIR/.." && pwd)}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_ENV="${CORK3DU_ENV:-$CORK3DU_DATA/env}"
export CORK3DU_WEIGHTS="${CORK3DU_WEIGHTS:-$CORK3DU_DATA/weights}"
export CORK3DU_DA3="${CORK3DU_DA3:-$CORK3DU_ROOT/third_party/Depth-Anything-3}"
export CORK3DU_SAM2="${CORK3DU_SAM2:-$CORK3DU_ROOT/third_party/sam2}"
export CORK3DU_SAM2_CKPT="${CORK3DU_SAM2_CKPT:-$CORK3DU_WEIGHTS/sam2.1_hiera_large.pt}"
export PYTHONPATH="${CORK3DU_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$CORK3DU_ROOT/logs" "$CORK3DU_DATA"/{chunks,scenes,weights,logs}

if [[ -x "$CORK3DU_ENV/bin/python" ]]; then
  link_cluster_torch_into_venv "$CORK3DU_ENV"
  export PATH="$CORK3DU_ENV/bin:$PATH"
  export VIRTUAL_ENV="$CORK3DU_ENV"
elif [[ -f "$CORK3DU_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$CORK3DU_ROOT/.venv/bin/activate"
fi
export PATH="${HOME}/.local/bin:${PATH}"

cd "$CORK3DU_ROOT"
echo "python=$(command -v python) ($(python -c 'import sys; print(sys.executable)'))"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || echo "torch not importable yet"
