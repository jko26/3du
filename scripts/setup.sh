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

# shellcheck source=/dev/null
source "$ROOT/jobs/_cluster_modules.sh"
load_cork3du_modules

mkdir -p "$CORK3DU_DATA"/{chunks,scenes,weights,logs}
mkdir -p "$CORK3DU_ROOT/third_party" "$CORK3DU_ROOT/logs"

if [[ -x "$CORK3DU_ENV/bin/python" ]]; then
  link_cluster_torch_into_venv "$CORK3DU_ENV"
  export PATH="$CORK3DU_ENV/bin:$PATH"
  export VIRTUAL_ENV="$CORK3DU_ENV"
fi

if [[ ! -d "$CORK3DU_DA3/.git" ]]; then
  git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git "$CORK3DU_DA3"
fi
if [[ ! -d "$CORK3DU_SAM2/.git" ]]; then
  git clone --recursive https://github.com/facebookresearch/sam2.git "$CORK3DU_SAM2"
fi
pip install -e "$CORK3DU_SAM2"

CKPT="$CORK3DU_WEIGHTS/sam2.1_hiera_large.pt"
if [[ ! -f "$CKPT" ]]; then
  mkdir -p "$CORK3DU_WEIGHTS"
  echo "downloading SAM2 checkpoint via Hugging Face (not fbaipublicfiles)"
  python - <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path
dest = Path("$CKPT")
got = hf_hub_download(
    "facebook/sam2.1-hiera-large",
    "sam2.1_hiera_large.pt",
    local_dir=str(dest.parent),
)
got = Path(got)
if got.resolve() != dest.resolve():
    dest.write_bytes(got.read_bytes())
print("sam2 ckpt", dest, dest.stat().st_size)
PY
fi

pip install -e "$CORK3DU_ROOT"
echo "setup ok"
echo "  CORK3DU_ROOT=$CORK3DU_ROOT"
echo "  CORK3DU_DATA=$CORK3DU_DATA"
echo "  CORK3DU_DA3=$CORK3DU_DA3"
echo "  CORK3DU_SAM2=$CORK3DU_SAM2"
echo "  SAM2_CKPT=$CKPT"
