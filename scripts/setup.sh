#!/usr/bin/env bash
# Clone DA3 + SAM2 + ODISE next to the git checkout; weights onto $CORK3DU_DATA.
# Installs every import DA3-Streaming / SAM2 / RAFT / ODISE need at module load.
# Never pip-install torch, torchvision, xformers, gsplat, or numpy>=2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORK3DU_ROOT="${CORK3DU_ROOT:-$ROOT}"
export CORK3DU_DATA="${CORK3DU_DATA:-/projects/sinus_clinical_data/3du}"
export CORK3DU_WEIGHTS="${CORK3DU_WEIGHTS:-$CORK3DU_DATA/weights}"
export CORK3DU_DA3="${CORK3DU_DA3:-$CORK3DU_ROOT/third_party/Depth-Anything-3}"
export CORK3DU_SAM2="${CORK3DU_SAM2:-$CORK3DU_ROOT/third_party/sam2}"
export CORK3DU_ODISE="${CORK3DU_ODISE:-$CORK3DU_ROOT/third_party/ODISE}"
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
if [[ ! -d "$CORK3DU_ODISE/.git" ]]; then
  git clone --recursive https://github.com/NVlabs/ODISE.git "$CORK3DU_ODISE"
else
  git -C "$CORK3DU_ODISE" submodule update --init --recursive
fi

# Do not `pip install -e DA3` — its pyproject pulls xformers/open3d/torch.
# Mask2Former is on PYTHONPATH only (do not pip-install; its pins fight SAM2/hydra).
export PYTHONPATH="${CORK3DU_DA3}/src:${CORK3DU_DA3}/da3_streaming:${CORK3DU_ODISE}:${CORK3DU_ODISE}/third_party/Mask2Former:${CORK3DU_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# SAM2's pyproject wants torch>=2.5.1 and will otherwise pip-install a
# CPU/cu13 wheel over the cluster 2.2.0+cu121. Install deps only, then SAM2.
python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  "hydra-core>=1.3.2" "iopath>=0.1.10" "omegaconf>=2.2,<2.4" "einops>=0.8,<1"
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

# --- ODISE (open-vocab panoptic). Never let its resolver downgrade DA3/SAM2 pins. ---
# detectron2 against cluster torch (no torch wheel).
if ! python -c "import detectron2" 2>/dev/null; then
  echo "installing detectron2 against cluster torch (no torch wheel)…"
  python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/detectron2.git"
fi
# detectron2 runtime extras. fvcore pins iopath<0.1.10 which fights SAM2's
# iopath>=0.1.10 — install fvcore --no-deps; keep our newer iopath.
python -m pip install --no-deps "fvcore==0.1.5.post20221221"
python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  "cloudpickle" "termcolor>=1.1" "yacs>=0.1.8" "pycocotools>=2.0.2" \
  "tabulate" "Pillow" "matplotlib" "boto3>=1.21.25"

# ODISE dataset utils (not on PyPI as named packages). --no-deps so they
# cannot fight opencv / numpy pins.
python -m pip install --no-build-isolation --no-deps \
  "git+https://github.com/cocodataset/panopticapi.git"
python -m pip install --no-build-isolation --no-deps \
  "git+https://github.com/lvis-dataset/lvis-api.git" || true

# ODISE + sdkit: --no-deps so they cannot pin einops==0.3 / omegaconf==2.1.1.
python -m pip install -e "$CORK3DU_ODISE" --no-build-isolation --no-deps
python -m pip install --no-deps "stable-diffusion-sdkit==2.1.3" || \
  python -m pip install --no-deps "stable-diffusion-sdkit>=2.1.3"
python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  "timm>=0.6.11" "nltk>=3.6.2" "diffdist>=0.1" "open-clip-torch>=2.0.2" "wandb>=0.12.11" || true

python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  -e "$CORK3DU_ROOT" --no-build-isolation

# pypose depends on torch; --no-deps so pip cannot fetch a PyPI torch wheel.
python -m pip install --no-deps pypose

# Heal: ODISE/sdkit metadata still *asks* for old pins; force DA3/SAM2-compatible stack back.
echo "healing einops / omegaconf / hydra after ODISE install…"
python -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINT" \
  "einops>=0.8,<1" "omegaconf>=2.2,<2.4" "hydra-core>=1.3.2" \
  "antlr4-python3-runtime==4.9.*" "iopath>=0.1.10"
# Re-assert fvcore after any resolver churn (still --no-deps for iopath).
python -m pip install --no-deps "fvcore==0.1.5.post20221221"

python - <<'PY'
import einops
from einops import einsum  # noqa: F401 — DA3 needs this; fails on einops 0.3
print("einops", einops.__version__, "ok (has einsum)")
import omegaconf
print("omegaconf", omegaconf.__version__)
import fvcore
import detectron2
from panopticapi.utils import rgb2id  # noqa: F401
from odise.data import get_openseg_labels  # noqa: F401
print("fvcore", getattr(fvcore, "__version__", "?"), "detectron2+odise.data ok")
PY

python -m cork3du preflight
# Fail the job if ODISE cannot import — instances20 will not work otherwise.
python -m cork3du preflight --check-odise --skip-da3 --skip-sam2
echo "setup ok"
echo "  CORK3DU_ROOT=$CORK3DU_ROOT"
echo "  CORK3DU_DATA=$CORK3DU_DATA"
echo "  CORK3DU_DA3=$CORK3DU_DA3"
echo "  CORK3DU_SAM2=$CORK3DU_SAM2"
echo "  CORK3DU_ODISE=$CORK3DU_ODISE"
echo "  SAM2_CKPT=$CKPT"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)"
