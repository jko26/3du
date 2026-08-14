# Sourced by jobs/_env.sh and scripts. Loads JHU's CUDA PyTorch (py311).
# Do not pip-install torch from PyPI; the proxy blocks NVIDIA's index and
# default PyPI torch is CPU-only.

load_cork3du_modules() {
  if ! type module >/dev/null 2>&1; then
    return 0
  fi
  set +u
  module load shared 2>/dev/null || true
  module load python311
  module load pytorch-py311-cuda12.1-gcc11/2.2.0
  module load ffmpeg 2>/dev/null || true
  set -u
}
