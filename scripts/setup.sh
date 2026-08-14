#!/usr/bin/env bash
# Clone DA3 + SAM2 next to the git checkout; weights onto $CORK3DU_DATA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORK3DU_ROOT="${CORK3DU_ROOT:-$ROOT}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_WEIGHTS="${CORK3DU_WEIGHTS:-$CORK3DU_DATA/weights}"
export CORK3DU_DA3="${CORK3DU_DA3:-$CORK3DU_ROOT/third_party/Depth-Anything-3}"
export CORK3DU_SAM2="${CORK3DU_SAM2:-$CORK3DU_ROOT/third_party/sam2}"
export CORK3DU_ENV="${CORK3DU_ENV:-$CORK3DU_DATA/env}"

mkdir -p "$CORK3DU_DATA"/{chunks,scenes,weights,logs}
mkdir -p "$CORK3DU_ROOT/third_party" "$CORK3DU_ROOT/logs"

if [[ -x "$CORK3DU_ENV/bin/python" ]]; then
  export PATH="$CORK3DU_ENV/bin:$PATH"
  export VIRTUAL_ENV="$CORK3DU_ENV"
fi

if [[ ! -d "$CORK3DU_DA3/.git" ]]; then
  git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git "$CORK3DU_DA3"
fi
if [[ ! -d "$CORK3DU_SAM2/.git" ]]; then
  git clone --recursive https://github.com/facebookresearch/sam2.git "$CORK3DU_SAM2"
  pip install -e "$CORK3DU_SAM2"
fi

CKPT="$CORK3DU_WEIGHTS/sam2.1_hiera_large.pt"
if [[ ! -f "$CKPT" ]]; then
  mkdir -p "$CORK3DU_WEIGHTS"
  curl -L -o "$CKPT" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

pip install -e "$CORK3DU_ROOT"
echo "setup ok"
echo "  CORK3DU_ROOT=$CORK3DU_ROOT"
echo "  CORK3DU_DATA=$CORK3DU_DATA"
echo "  CORK3DU_DA3=$CORK3DU_DA3"
echo "  CORK3DU_SAM2=$CORK3DU_SAM2"
echo "Install torch/torchvision for this cluster CUDA *before* or after, matching the GPU."
