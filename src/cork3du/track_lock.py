"""Residual-seeded SAM2 40/40 lock over independent 2-second windows."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .ffbin import ffmpeg_bin

logger = logging.getLogger(__name__)


def _mask_bool(m: np.ndarray, h: int, w: int) -> np.ndarray:
    m = m > 0.5
    if m.shape[:2] == (h, w):
        return m
    import cv2

    return (
        cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
    )


def apply_track_window_lock(
    p_resid: list[np.ndarray],
    masks_by_obj: dict[int, list[np.ndarray]],
    *,
    pixel_p_thre: float = 0.5,
    pixel_frac_thre: float = 0.4,
    frame_frac_thre: float = 0.4,
    window_seconds: float = 2.0,
    fps: float = 5.0,
    min_area: int = 32,
    min_visible_in_window: int = 3,
) -> tuple[list[np.ndarray], dict[str, Any], dict[int, dict[str, float]]]:
    """
    Per SAM2 track, non-overlapping G-second windows.

    A frame is hot if visible and ≥pixel_frac_thre of instance pixels have
    p_resid ≥ pixel_p_thre. If ≥frame_frac_thre of *visible* frames in a window
    are hot, cookie-cut the instance on every visible frame in that window.
    Adjacent windows are independent.
    """
    n = len(p_resid)
    h, w = p_resid[0].shape
    win = max(1, int(round(float(window_seconds) * float(fps))))
    locked = [np.zeros((h, w), np.float32) for _ in range(n)]
    per_obj: dict[int, dict[str, float]] = {}
    n_windows_locked = 0
    n_windows_skipped = 0

    for oid, masks in masks_by_obj.items():
        hot = np.zeros(n, dtype=bool)
        visible = np.zeros(n, dtype=bool)
        for t in range(n):
            m = _mask_bool(masks[t], h, w)
            area = int(m.sum())
            if area < min_area:
                continue
            visible[t] = True
            frac = float((p_resid[t][m] >= pixel_p_thre).mean())
            hot[t] = frac >= pixel_frac_thre

        n_win_lock = 0
        n_win_skip = 0
        starts = list(range(0, n, win))
        lock_flags = np.zeros(n, dtype=bool)
        for start in starts:
            end = min(start + win, n)
            vis_idx = [t for t in range(start, end) if visible[t]]
            if len(vis_idx) < min_visible_in_window:
                n_win_skip += 1
                continue
            hot_frac = float(np.mean([hot[t] for t in vis_idx]))
            if hot_frac < frame_frac_thre:
                continue
            n_win_lock += 1
            for t in vis_idx:
                lock_flags[t] = True
                m = _mask_bool(masks[t], locked[t].shape[0], locked[t].shape[1])
                if m.any():
                    locked[t][m] = 1.0

        n_windows_locked += n_win_lock
        n_windows_skipped += n_win_skip
        per_obj[oid] = {
            "visible": float(int(visible.sum())),
            "n_hot": float(int(hot.sum())),
            "n_windows_locked": float(n_win_lock),
            "n_windows_skipped": float(n_win_skip),
            "n_locked_frames": float(int(lock_flags.sum())),
        }

    info = {
        "mode": "window_40_40",
        "window_seconds": float(window_seconds),
        "fps": float(fps),
        "window_frames": win,
        "pixel_p_thre": pixel_p_thre,
        "pixel_frac_thre": pixel_frac_thre,
        "frame_frac_thre": frame_frac_thre,
        "min_visible_in_window": min_visible_in_window,
        "n_objects": len(masks_by_obj),
        "n_windows_locked": n_windows_locked,
        "n_windows_skipped_sparse": n_windows_skipped,
    }
    return locked, info, per_obj


def _heatmap(rgb: np.ndarray, p: np.ndarray) -> np.ndarray:
    import cv2

    if p.shape[:2] != rgb.shape[:2]:
        p = cv2.resize(
            p.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    p = np.clip(p, 0, 1)
    heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 0.35, heat, 0.65, 0)


def write_mask_debug(
    images_rgb: list[np.ndarray],
    *,
    p_resid: list[np.ndarray],
    instance_vis: list[np.ndarray],
    locked: list[np.ndarray],
    final: list[np.ndarray],
    out_mp4: Path,
    fps: float = 5.0,
) -> Path:
    import cv2

    dbg = out_mp4.parent / "debug_frames"
    if dbg.exists():
        shutil.rmtree(dbg)
    dbg.mkdir(parents=True)

    def overlay(rgb: np.ndarray, m: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
        if m.shape[:2] != rgb.shape[:2]:
            m = cv2.resize(
                m.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = bgr.copy()
        hit = m > 0.5
        col = np.array(color[::-1], np.float32)
        out[hit] = (out[hit].astype(np.float32) * 0.35 + col * 0.65).astype(np.uint8)
        return out

    for i, rgb in enumerate(images_rgb):
        panel = np.concatenate(
            [
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                _heatmap(rgb, p_resid[i]),
                overlay(rgb, instance_vis[i], color=(0, 255, 255)),
                overlay(rgb, locked[i], color=(255, 128, 0)),
                overlay(rgb, final[i]),
            ],
            axis=1,
        )
        label = np.zeros((28, panel.shape[1], 3), np.uint8)
        ww = rgb.shape[1]
        for text, x in [
            ("RGB", 8),
            ("p_resid", ww + 8),
            ("SAM2", 2 * ww + 8),
            ("window lock", 3 * ww + 8),
            ("final", 4 * ww + 8),
        ]:
            cv2.putText(label, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imwrite(str(dbg / f"{i:05d}.jpg"), np.concatenate([label, panel], 0))

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(), "-y", "-framerate", str(fps),
            "-i", str(dbg / "%05d.jpg"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_mp4),
        ],
        check=True,
        capture_output=True,
    )
    return out_mp4


def remask_scene(
    scene_dir: Path,
    *,
    sam2_root: Path,
    sam2_checkpoint: Path,
    fps: float = 5.0,
    window_seconds: float = 2.0,
    pixel_p_thre: float = 0.5,
    pixel_frac_thre: float = 0.4,
    frame_frac_thre: float = 0.4,
    ego_floor_px: float = 3.0,
    max_points: int = 400_000,
) -> dict[str, Any]:
    from .preview import write_scene_previews
    from .preflight import run_preflight
    from .reconstruct import fuse_masked_cloud, load_stream_rgbs
    from .residual import compute_p_resid
    from .sam2_tracks import find_residual_blob_seeds, run_sam2_tracks

    run_preflight(require_da3_tree=False, require_sam2=True)
    import cv2
    scene_dir = Path(scene_dir)
    stream_out = scene_dir / "stream_out"
    mask_dir = scene_dir / "masks"
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True)

    images_rgb, _ = load_stream_rgbs(stream_out)
    n = len(images_rgb)
    h, w = images_rgb[0].shape[:2]
    p_resid, _valid, resid_info = compute_p_resid(stream_out, ego_floor_px=ego_floor_px)
    seeds = find_residual_blob_seeds(p_resid, pixel_p_thre=pixel_p_thre)
    if not seeds:
        logger.warning("No residual blob seeds; final = p_resid>=%.2f only", pixel_p_thre)
        masks_by_obj = {}
        sam_info = {"n_seeds": 0, "n_objects": 0, "n_frames": n}
    else:
        masks_by_obj, sam_info = run_sam2_tracks(
            images_rgb,
            seeds,
            sam2_root=sam2_root,
            checkpoint=sam2_checkpoint,
            work_dir=mask_dir / "sam2_work",
        )

    instance_vis = [np.zeros((h, w), np.float32) for _ in range(n)]
    for masks in masks_by_obj.values():
        for t, m in enumerate(masks):
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            instance_vis[t] = np.maximum(instance_vis[t], m)

    locked, lock_info, per_obj = apply_track_window_lock(
        p_resid,
        masks_by_obj,
        pixel_p_thre=pixel_p_thre,
        pixel_frac_thre=pixel_frac_thre,
        frame_frac_thre=frame_frac_thre,
        window_seconds=window_seconds,
        fps=fps,
    )
    final = []
    for t in range(n):
        resid_bin = (p_resid[t] >= pixel_p_thre).astype(np.float32)
        final.append(np.maximum(resid_bin, locked[t]))
    for i, m in enumerate(final):
        cv2.imwrite(str(mask_dir / f"final_{i}.png"), (m * 255).astype(np.uint8))

    dbg = write_mask_debug(
        images_rgb,
        p_resid=p_resid,
        instance_vis=instance_vis,
        locked=locked,
        final=final,
        out_mp4=mask_dir / "mask_debug.mp4",
        fps=fps,
    )
    cloud, fuse_stats = fuse_masked_cloud(stream_out, final, motion_mask_thre=0.5, max_points=max_points)
    np.save(scene_dir / "cloud.npy", cloud)
    previews = write_scene_previews(scene_dir, title=f"{scene_dir.name} (p_resid∨2s 40/40 + DA3)")

    meta = {
        "pipeline": "p_resid_sam2_window_40_40",
        "n_frames": n,
        "fps": fps,
        "window_seconds": window_seconds,
        "pixel_p_thre": pixel_p_thre,
        "n_resid_seeds": len(seeds),
        "residual": resid_info,
        "sam2": sam_info,
        "track_lock": lock_info,
        "per_object": {str(k): v for k, v in per_obj.items()},
        "mean_resid_bin_frac": float(np.mean([(p >= pixel_p_thre).mean() for p in p_resid])),
        "mean_locked_frac": float(np.mean([m.mean() for m in locked])),
        "mean_final_frac": float(np.mean([m.mean() for m in final])),
        "n_cloud_points": int(cloud.shape[0]),
        "filter_stats": fuse_stats,
        "mask_debug_mp4": str(dbg),
        "previews": previews,
        "notes": [
            "Final = (p_resid≥0.5) OR 2s-window 40/40 SAM2 lock.",
            "SAM2 seeded from residual blobs only (no DAS3R).",
            "Windows are independent: a later static window is not locked by an earlier hot one.",
        ],
    }
    (mask_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    scene_meta_path = scene_dir / "meta.json"
    old = json.loads(scene_meta_path.read_text()) if scene_meta_path.is_file() else {}
    old.update(
        {
            "pipeline": meta["pipeline"],
            "n_cloud_points": meta["n_cloud_points"],
            "filter_stats": fuse_stats,
            "mask_debug_mp4": str(dbg),
            "masking": meta,
        }
    )
    scene_meta_path.write_text(json.dumps(old, indent=2) + "\n")
    logger.info(
        "remask: final=%.1f%% dropped=%.1f%% windows_locked=%d → %s",
        100 * meta["mean_final_frac"],
        100 * fuse_stats["frac_dynamic_dropped"],
        lock_info["n_windows_locked"],
        dbg,
    )
    return meta
