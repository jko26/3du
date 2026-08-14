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

On the JHU GPU cluster, after `git pull`:

```bash
cd $HOME/3du
mkdir -p logs   # Slurm opens logs/ before the job script runs
# one-time: conda/venv + torch for this CUDA, then
bash scripts/setup.sh

sbatch jobs/ingest_wtours.sbatch          # yt-dlp first 400s, 20×20s mp4s
sbatch jobs/wtours20.sbatch               # array 0-19; wait until ingest is done
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

Data: [Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) (CC-BY YouTube). Ingest does **not** download the full hours-long 4K file.

## CLI

```bash
export CORK3DU_DATA=/projects/sinus_clinical_data/3du
python -m cork3du ingest-wtours --city amsterdam --n-chunks 20 --chunk-seconds 20
python -m cork3du run --video $CORK3DU_DATA/chunks/amsterdam/000.mp4 --out $CORK3DU_DATA/scenes/amsterdam_000
python -m cork3du remask --scene $CORK3DU_DATA/scenes/amsterdam_000
python -m cork3du preview --scene $CORK3DU_DATA/scenes/amsterdam_000
```

## Masking rule

At 5 fps, a 2s window is 10 frames.

1. Frame is **hot** if the SAM2 instance is visible and ≥40% of its pixels have `p_resid ≥ 0.5`.
2. If ≥40% of **visible** frames in that window are hot, paint the instance on every visible frame in the window.
3. Next window is independent (parked → moving unmasks/masks at the cut).
4. `final = (p_resid ≥ 0.5) OR window_lock`.

## Setup notes

Install **torch / torchvision** against the cluster CUDA wheel index before or during `scripts/setup.sh`. DA3-Giant may OOM at 32G — bump `--mem` on the sbatch header if needed.

Ingest needs **yt-dlp** and **ffmpeg** in the same Python the job uses. Shared Anaconda often has no `yt-dlp` on `PATH`; install the module, then resubmit:

```bash
python -m pip install --user yt-dlp
# ffmpeg: module load ffmpeg   # or whatever your cluster provides
```
