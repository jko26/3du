#!/usr/bin/env bash
# Create a venv on project disk that can see the cluster CUDA PyTorch module.
# Run on a login node:  bash scripts/make_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/jobs/_cluster_modules.sh"
load_cork3du_modules

export CORK3DU_ROOT="${CORK3DU_ROOT:-$ROOT}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_ENV="${CORK3DU_ENV:-$CORK3DU_DATA/env}"
PY="${CORK3DU_PYTHON:-$(command -v python3)}"

mkdir -p "$CORK3DU_DATA"
if [[ -x "$CORK3DU_ENV/bin/python" ]]; then
  if grep -q "include-system-site-packages = false" "$CORK3DU_ENV/pyvenv.cfg" 2>/dev/null; then
    echo "existing venv cannot see cluster torch; moving it aside"
    mv "$CORK3DU_ENV" "${CORK3DU_ENV}.nosys.$(date +%Y%m%d%H%M%S)"
  fi
fi
if [[ ! -x "$CORK3DU_ENV/bin/python" ]]; then
  echo "creating venv --system-site-packages → $CORK3DU_ENV  (python=$PY)"
  "$PY" -m venv --system-site-packages "$CORK3DU_ENV"
fi

export PATH="$CORK3DU_ENV/bin:$PATH"
python -m pip install -U pip
python -m pip install -e "$CORK3DU_ROOT"

echo "env ok"
echo "  python=$(command -v python)"
echo "  CORK3DU_ENV=$CORK3DU_ENV"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Then: bash $CORK3DU_ROOT/scripts/setup.sh"
