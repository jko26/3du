# 3du

Static 3D reconstruction from monocular walking-tour video.

The Python import name is **`cork3du`** (`python -m cork3du`). The repo and data folder stay `3du`.

**Geometry:** DA3-Streaming (pose + depth).
**Dynamic mask:** floored RAFT residual (`p_resid`) **or** SAM2 40/40 lock on **independent 2-second windows**. Residual-only still masks when SAM2 misses an object. No DAS3R.

Lab experiments live in `../3d-understanding/`. This repo is the compact HPC pipeline.

## Cluster paths

| What | Where |
|---|---|
| Code (`git clone`) | `$HOME/3du` or anywhere (`CORK3DU_ROOT`) |
| Data | **`/projects/sinus_clinical_data/3du`** (`CORK3DU_DATA`) — created on first job |
| Chunks / scenes / weights / logs | `$CORK3DU_DATA/{chunks,scenes,weights,logs}` |

Env vars are `CORK3DU_*` because bash identifiers cannot start with a digit. Jobs `mkdir -p` the data tree. You need write access on `/projects/sinus_clinical_data`.

## First run (20 × 20s Amsterdam chunks)

On the JHU cluster, after `git pull`:

```bash
cd $HOME/projects/3du   # or wherever you cloned
mkdir -p logs           # Slurm opens logs/ before the job script runs

# one-time, on a login node — venv on project disk, using cluster CUDA PyTorch
module load shared
module load python311
module load pytorch-py311-cuda12.1-gcc11/2.2.0
bash scripts/make_env.sh

sbatch jobs/setup.sbatch                  # CPU (shared): clone, pip, checkpoints
sbatch jobs/ingest_wtours.sbatch          # CPU (shared): HF clip + 20×20s splits
sbatch jobs/wtours20.sbatch               # GPU: array 0-19; wait until ingest is done
```

| Job | Partition | GPU? |
|---|---|---|
| `jobs/setup.sbatch` | `shared` | no |
| `jobs/ingest_wtours.sbatch` | `shared` | no |
| `jobs/wtours20.sbatch` / `reconstruct` / `remask` / `run_scene` | `gpua100,gpuh100` (≥40GB) | yes |
| `jobs/instances20.sbatch` | `gpua100,gpuh100` | yes (SAM2 AMG) |

DA3 Nested-Giant OOMs a 16GB `gpu` card (T4/V100) at 60-view chunks. GPU jobs skip that partition. Chunk size is chosen from VRAM: 4/2 on <24GB, 12/6 on 40GB, 24/12 on 80GB.

## Stage 3–4 (static 3D instances)

After a scene has `cloud.npy` (and ideally `masks/final_*.png` from remask):

```bash
sbatch jobs/instances20.sbatch    # array 0-19; SAM2 AMG + superpoint affinity lift
```

This is SAI3D-style (voxel superpoints + multi-view SAM co-occurrence + region grow) using the SAM2 already in the venv — not the official ScanNet/Semantic-SAM stack. By default **all AMG masks are kept** (no dyn/depth drop); masks that hit no cloud points still cannot vote. Tighten with `--max-dyn-frac` / `--min-depth-frac` if needed.

```
$CORK3DU_DATA/scenes/amsterdam_000/instances/
  instance_000.npy …     # (N,6) xyzrgb per instance
  point_instances.npy    # per-point ids (-1 unassigned)
  preview.png / .html    # colored by instance
  amg_debug/frame_XXX.png  # RGB | all AMG | kept | dropped | remask final
  meta.json              # n_things vs n_stuff (ground / huge regions)
```

Always `sbatch` from the clone root so `SLURM_SUBMIT_DIR/jobs/_env.sh` resolves.

Per-chunk outputs:

```
$CORK3DU_DATA/scenes/amsterdam_000/
  stream_out/
  masks/mask_debug.mp4    # RGB | p_resid | SAM2 | window lock | final
  cloud.npy
  preview.png
  preview.html
  meta.json
```

Data: [Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) (CC-BY YouTube). The original HF card is URLs only; ingest pulls the pre-cut Amsterdam clip from [jkoooo/3du-wtours](https://huggingface.co/datasets/jkoooo/3du-wtours) (public, no token).

## CLI

```bash
export CORK3DU_DATA=/projects/sinus_clinical_data/3du
python -m cork3du ingest-wtours --city amsterdam --n-chunks 20 --chunk-seconds 20
python -m cork3du run --video $CORK3DU_DATA/chunks/amsterdam/000.mp4 --out $CORK3DU_DATA/scenes/amsterdam_000
python -m cork3du remask --scene $CORK3DU_DATA/scenes/amsterdam_000
python -m cork3du instances --scene $CORK3DU_DATA/scenes/amsterdam_000
python -m cork3du preview --scene $CORK3DU_DATA/scenes/amsterdam_000
```

## Masking rule

At 5 fps, a 2s window is 10 frames.

1. Frame is **hot** if the SAM2 instance is visible and ≥40% of its pixels have `p_resid ≥ 0.5`.
2. If ≥40% of **visible** frames in that window are hot, paint the instance on every visible frame in the window.
3. Next window is independent (parked → moving unmasks/masks at the cut).
4. `final = (p_resid ≥ 0.5) OR window_lock`.

## Setup notes

After `git pull`, re-run `sbatch jobs/setup.sbatch` (CPU `shared` queue — do not use a login node) so DA3-Streaming’s import-time extras (`pypose`, `evo`, `pycolmap`, `moviepy==1.0.3`, …) land in the venv. `python -m cork3du preflight` lists every missing module at once. Do not `pip install torch` / `xformers` / `numpy>=2`. DA3-Giant may OOM at 32G — bump `--mem` on the GPU sbatch header if needed.

Jobs activate `$CORK3DU_DATA/env` automatically (`CORK3DU_ENV`). Recreate it with `bash scripts/make_env.sh` after `module load shared python311 pytorch-py311-cuda12.1-gcc11/2.2.0` so the venv can see cluster torch (`--system-site-packages`). Do not `pip install torch` from PyPI (CPU, or NVIDIA index 403). ffmpeg comes from `module load ffmpeg` or `imageio-ffmpeg`.
