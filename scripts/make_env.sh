#!/usr/bin/env bash
# Create a venv on project disk (not $HOME, not shared Anaconda).
# Run on a login node. Ingest only needs this + ffmpeg. Torch comes later.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORK3DU_ROOT="${CORK3DU_ROOT:-$ROOT}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_ENV="${CORK3DU_ENV:-$CORK3DU_DATA/env}"
PY="${CORK3DU_PYTHON:-python3}"

mkdir -p "$CORK3DU_DATA"
if [[ ! -x "$CORK3DU_ENV/bin/python" ]]; then
  echo "creating venv → $CORK3DU_ENV  (python=$PY)"
  "$PY" -m venv "$CORK3DU_ENV"
fi

export PATH="$CORK3DU_ENV/bin:$PATH"
python -m pip install -U pip
python -m pip install -e "$CORK3DU_ROOT"

echo "env ok"
echo "  python=$(command -v python)"
echo "  CORK3DU_ENV=$CORK3DU_ENV"
echo "Before wtours20, install CUDA torch into this env, e.g."
echo "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
echo "Then: bash $CORK3DU_ROOT/scripts/setup.sh"
