"""ffmpeg frames + DA3-Streaming geometry (no masking)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .ffbin import ffmpeg_bin

logger = logging.getLogger(__name__)

DA3_HF_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


def extract_frames_ffmpeg(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float = 5.0,
    width: int = 640,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("frame_*.png")) + list(out_dir.glob("*.jpg")):
        old.unlink()
    pattern = str(out_dir / "frame_%06d.png")
    cmd = [ffmpeg_bin(), "-y", "-i", str(video_path), "-vf", f"fps={fps},scale={width}:-1", pattern]
    logger.info("ffmpeg extract: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    paths = sorted(out_dir.glob("frame_*.png"))
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 frames from {video_path}, got {len(paths)}")
    logger.info("Extracted %d frames @ fps=%s width=%s", len(paths), fps, width)
    return paths


def ensure_da3_weights(weights_dir: Path, *, hf_repo: str = DA3_HF_REPO) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "model.safetensors"
    cfg = weights_dir / "config.json"
    salad = weights_dir / "dino_salad.ckpt"
    if not ckpt.is_file() or not cfg.is_file():
        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s → %s", hf_repo, weights_dir)
        for name in ("config.json", "model.safetensors"):
            path = hf_hub_download(hf_repo, name, local_dir=str(weights_dir))
            dest = weights_dir / name
            if Path(path).resolve() != dest.resolve():
                shutil.copy2(path, dest)
    if not salad.is_file():
        logger.info("Downloading SALAD → %s", salad)
        subprocess.run(["curl", "-L", SALAD_URL, "-o", str(salad)], check=True)
    return weights_dir


def write_streaming_config(
    template_path: Path,
    out_path: Path,
    *,
    weights_dir: Path,
    n_frames: int,
    delete_temp_files: bool = False,
) -> Path:
    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    chunk = min(60, max(8, n_frames))
    overlap = min(chunk // 2, max(0, chunk - 2))
    cfg["Model"]["chunk_size"] = int(chunk)
    cfg["Model"]["overlap"] = int(overlap)
    cfg["Model"]["loop_enable"] = False
    cfg["Model"]["save_depth_conf_result"] = True
    cfg["Model"]["delete_temp_files"] = delete_temp_files
    cfg["Model"]["align_lib"] = "torch"
    cfg["Weights"]["DA3"] = str(weights_dir / "model.safetensors")
    cfg["Weights"]["DA3_CONFIG"] = str(weights_dir / "config.json")
    cfg["Weights"]["SALAD"] = str(weights_dir / "dino_salad.ckpt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return out_path


def run_da3_streaming(
    *,
    da3_streaming_root: Path,
    images_dir: Path,
    output_dir: Path,
    config_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    script = da3_streaming_root / "da3_streaming.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--image_dir",
        str(images_dir),
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
    ]
    da3_repo = da3_streaming_root.parent if da3_streaming_root.name == "da3_streaming" else da3_streaming_root
    from .preflight import apply_da3_pythonpath

    env = apply_da3_pythonpath(da3_repo)
    logger.info("DA3-Streaming: cwd=%s", da3_streaming_root)
    subprocess.run(cmd, cwd=str(da3_streaming_root), env=env, check=True)
    return output_dir


def read_c2w_poses(pose_file: Path) -> np.ndarray:
    poses = []
    with open(pose_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = list(map(float, line.split()))
            if len(nums) == 16:
                poses.append(np.array(nums, dtype=np.float64).reshape(4, 4))
    if not poses:
        raise RuntimeError(f"No poses in {pose_file}")
    return np.stack(poses, axis=0)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _unproject_frame(depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    cam = np.stack([x, y, z], axis=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    world = cam @ R.T + t[None, None, :]
    return world.astype(np.float32)


def fuse_masked_cloud(
    stream_out: Path,
    soft_masks: list[np.ndarray] | None,
    *,
    motion_mask_thre: float = 0.5,
    conf_threshold_coef: float = 0.75,
    max_points: int = 400_000,
    spatial_stride: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    pose_file = stream_out / "camera_poses.txt"
    npz_dir = stream_out / "results_output"
    poses = read_c2w_poses(pose_file)
    npz_files = sorted(npz_dir.glob("frame_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if len(npz_files) != len(poses):
        raise RuntimeError(f"poses ({len(poses)}) != npz ({len(npz_files)})")
    if soft_masks is not None and len(soft_masks) != len(npz_files):
        if not soft_masks:
            soft_masks = None
        else:
            idx = np.linspace(0, len(soft_masks) - 1, len(npz_files)).round().astype(int)
            soft_masks = [soft_masks[i] for i in idx]

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    n_dyn_drop = 0
    n_cand = 0
    for i, (npz_path, c2w) in enumerate(zip(npz_files, poses)):
        data = np.load(npz_path)
        image = data["image"]
        depth = data["depth"].astype(np.float32)
        conf = data["conf"].astype(np.float32)
        K = data["intrinsics"].astype(np.float32)
        world = _unproject_frame(depth, K, c2w.astype(np.float32))
        pts = world[::spatial_stride, ::spatial_stride].reshape(-1, 3)
        cols = image[::spatial_stride, ::spatial_stride].reshape(-1, 3).astype(np.float32) / 255.0
        c = conf[::spatial_stride, ::spatial_stride].reshape(-1)
        finite = np.isfinite(pts).all(axis=1) & (c > 0) & (pts[:, 2] != 0)
        keep = finite
        n_cand += int(keep.sum())
        if soft_masks is not None:
            h, w = depth.shape
            static = resize_mask(soft_masks[i], h, w) <= motion_mask_thre
            static_flat = static[::spatial_stride, ::spatial_stride].reshape(-1)
            before = int(keep.sum())
            keep = keep & static_flat
            n_dyn_drop += before - int(keep.sum())
        all_pts.append(pts[keep])
        all_cols.append(cols[keep])
        all_conf.append(c[keep])

    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), np.float32)
    cols = np.concatenate(all_cols, axis=0) if all_cols else np.zeros((0, 3), np.float32)
    confs = np.concatenate(all_conf, axis=0) if all_conf else np.zeros((0,), np.float32)
    if pts.shape[0] == 0:
        raise RuntimeError("No points after DA3+mask fusion")
    conf_thr = float(np.mean(confs) * conf_threshold_coef) if confs.size else 0.0
    conf_ok = confs >= conf_thr
    pts, cols = pts[conf_ok], cols[conf_ok]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    cloud = np.concatenate([pts.astype(np.float32), np.clip(cols.astype(np.float32), 0, 1)], axis=1)
    stats = {
        "n_frames_fused": len(npz_files),
        "n_candidate_before_conf": int(n_cand),
        "n_dynamic_dropped": int(n_dyn_drop),
        "frac_dynamic_dropped": float(n_dyn_drop / max(n_cand, 1)),
        "conf_threshold": conf_thr,
        "n_cloud_points": int(cloud.shape[0]),
        "spatial_stride": spatial_stride,
        "motion_mask_thre": motion_mask_thre if soft_masks is not None else None,
    }
    return cloud, stats


def load_stream_rgbs(stream_out: Path) -> tuple[list[np.ndarray], list[Path]]:
    npz_dir = stream_out / "results_output"
    files = sorted(npz_dir.glob("frame_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    rgbs = [np.load(p)["image"] for p in files]
    return rgbs, files


def reconstruct_video(
    video_path: Path,
    out_dir: Path,
    *,
    da3_streaming_root: Path,
    weights_cache: Path,
    fps: float = 5.0,
    frame_width: int = 640,
) -> dict[str, Any]:
    """Extract frames and run DA3-Streaming. Masking is a separate remask step."""
    from .preflight import run_preflight

    run_preflight(require_da3_tree=True, require_sam2=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    stream_out = out_dir / "stream_out"
    cfg_path = out_dir / "streaming_config.yaml"
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / "video.mp4"
    if Path(video_path).resolve() != dest.resolve():
        if not dest.exists():
            try:
                dest.symlink_to(Path(video_path).resolve())
            except OSError:
                shutil.copy2(video_path, dest)

    frame_paths = extract_frames_ffmpeg(video_path, images_dir, fps=fps, width=frame_width)
    weights_dir = ensure_da3_weights(weights_cache)
    template = da3_streaming_root / "da3_streaming" / "configs" / "base_config.yaml"
    if not template.is_file():
        template = da3_streaming_root / "configs" / "base_config.yaml"
    write_streaming_config(template, cfg_path, weights_dir=weights_dir, n_frames=len(frame_paths))
    if stream_out.exists():
        shutil.rmtree(stream_out)
    streaming_cwd = da3_streaming_root / "da3_streaming"
    if not (streaming_cwd / "da3_streaming.py").is_file():
        streaming_cwd = da3_streaming_root
    run_da3_streaming(
        da3_streaming_root=streaming_cwd,
        images_dir=images_dir,
        output_dir=stream_out,
        config_path=cfg_path,
    )
    pose_file = stream_out / "camera_poses.txt"
    if pose_file.is_file():
        c2w = read_c2w_poses(pose_file).astype(np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        np.savez_compressed(
            out_dir / "pose.npz",
            data=c2w,
            w2c=w2c,
            inds=np.arange(len(c2w), dtype=np.int32),
        )
    meta = {
        "pipeline": "da3_streaming",
        "geometry_backend": "da3_streaming",
        "da3_model": DA3_HF_REPO,
        "n_video_frames_sampled": int(len(frame_paths)),
        "fps": fps,
        "frame_width": frame_width,
        "stream_out": str(stream_out),
        "video": str(video_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("DA3 reconstruct: %d frames → %s", len(frame_paths), stream_out)
    return meta
