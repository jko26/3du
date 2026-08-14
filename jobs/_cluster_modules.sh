# Sourced by jobs/_env.sh and scripts. Loads JHU's CUDA PyTorch (py311).
# Load this on CPU *and* GPU nodes: the module is libraries on /cm/shared.
# Only GPU sbatch scripts should request --gres=gpu. Do not pip-install torch
# from PyPI; the proxy blocks NVIDIA's index and default PyPI torch is CPU-only.

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

# Nested venvs do not inherit the module's torch. Drop a .pth into the venv
# while the *module* python can still `import torch`.
link_cluster_torch_into_venv() {
  local env_root="${1:?}"
  local site dest
  site="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent.parent)")"
  dest="$(ls -d "$env_root"/lib/python3.*/site-packages | head -1)"
  echo "$site" > "$dest/z_cluster_torch.pth"
  echo "linked cluster torch $site → $dest/z_cluster_torch.pth"
}
