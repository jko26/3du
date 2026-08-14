#!/usr/bin/env bash
# Clone DA3 + SAM2 next to the git checkout; weights onto $CORK3DU_DATA.
# Installs every import DA3-Streaming / SAM2 / RAFT need at module load.
# Never pip-install torch, torchvision, xformers, gsplat, or numpy>=2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORK3DU_ROOT="${CORK3DU_ROOT:-$ROOT}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_WEIGHTS="${CORK3DU_WEIGHTS:-$CORK3DU_DATA/weights}"
export CORK3DU_DA3="${CORK3DU_DA3:-$CORK3DU_ROOT/third_party/Depth-Anything-3}"
export CORK3DU_SAM2="${CORK3DU_SAM2:-$CORK3DU_ROOT/third_party/sam2}"
export CORK3DU_ENV="${CORK3DU_ENV:-$CORK3DU_DATA/env}"
CONSTRAINT="$ROOT/constraints/hpc.txt"

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

python - <<'PY'
import pathlib, sys
import torch
p = pathlib.Path(torch.__file__).resolve()
print("torch", torch.__version__, "cuda", torch.version.cuda, "→", p)
if "cm/shared" not in str(p) and "pytorch-py311" not in str(p):
    sys.exit(
        f"refusing to continue: torch is not the cluster CUDA module ({p}).\n"
        "  pip uninstall torch torchvision -y\n"
        "  then re-run scripts/make_env.sh / setup.sh"
    )
PY

if [[ ! -d "$CORK3DU_DA3/.git" ]]; then
  git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git "$CORK3DU_DA3"
else
  git -C "$CORK3DU_DA3" submodule update --init --recursive
fi
if [[ ! -d "$CORK3DU_SAM2/.git" ]]; then
  git clone --recursive https://github.com/facebookresearch/sam2.git "$CORK3DU_SAM2"
fi

# Do not `pip install -e DA3` — its pyproject pulls xformers/open3d/torch.
# PYTHONPATH=$CORK3DU_DA3/src is set in jobs/_env.sh and reconstruct.py.
export PYTHONPATH="${CORK3DU_DA3}/src:${CORK3DU_DA3}/da3_streaming:${CORK3DU_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# SAM2's pyproject wants torch>=2.5.1 and will otherwise pip-install a
# CPU/cu13 wheel over the cluster 2.2.0+cu121. Install deps only, then SAM2.
python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  "hydra-core>=1.3.2" "iopath>=0.1.10" "omegaconf>=2.2,<2.4"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
python -m pip install -e "$CORK3DU_SAM2" --no-build-isolation --no-deps

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

python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  -e "$CORK3DU_ROOT" --no-build-isolation

# pypose depends on torch; --no-deps so pip cannot fetch a PyPI torch wheel.
python -m pip install --no-deps pypose

python -m cork3du preflight
echo "setup ok"
echo "  CORK3DU_ROOT=$CORK3DU_ROOT"
echo "  CORK3DU_DATA=$CORK3DU_DATA"
echo "  CORK3DU_DA3=$CORK3DU_DA3"
echo "  CORK3DU_SAM2=$CORK3DU_SAM2"
echo "  SAM2_CKPT=$CKPT"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)"
