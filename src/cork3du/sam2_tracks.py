"""SAM2 video tracks from residual-blob seeds (no DAS3R)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BlobSeed:
    frame_idx: int
    y: float
    x: float
    score: float
    area: int


def find_residual_blob_seeds(
    p_resid: list[np.ndarray],
    *,
    pixel_p_thre: float = 0.5,
    min_area: int = 80,
    max_blobs_per_frame: int = 3,
    max_keyframes: int = 12,
    skip_if_frame_frac: float = 0.45,
) -> list[BlobSeed]:
    n = len(p_resid)
    key_idxs = np.linspace(0, n - 1, min(max_keyframes, n), dtype=int)
    seeds: list[BlobSeed] = []
    for fi in key_idxs:
        p = p_resid[int(fi)]
        binary = (p >= pixel_p_thre).astype(np.uint8)
        if float(binary.mean()) >= skip_if_frame_frac:
            continue
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cands: list[BlobSeed] = []
        for lab in range(1, n_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = centroids[lab]
            score = float(p[labels == lab].mean())
            cands.append(
                BlobSeed(frame_idx=int(fi), y=float(cy), x=float(cx), score=score, area=area)
            )
        cands.sort(key=lambda b: b.area, reverse=True)
        seeds.extend(cands[:max_blobs_per_frame])
    return seeds


def _write_jpeg_video_dir(images_rgb: list[np.ndarray], out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for i, rgb in enumerate(images_rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_dir


def run_sam2_tracks(
    images_rgb: list[np.ndarray],
    seeds: list[BlobSeed],
    *,
    sam2_root: Path,
    checkpoint: Path,
    work_dir: Path,
    device: str | None = None,
) -> tuple[dict[int, list[np.ndarray]], dict[str, Any]]:
    if not seeds:
        n = len(images_rgb)
        return {}, {"n_seeds": 0, "n_objects": 0, "n_frames": n}

    import sys

    torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    logger.info("SAM2 video predictor (%s)", torch_device)
    predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=torch_device)
    use_bf16 = str(torch_device).startswith("cuda") and torch.cuda.is_available()
    if use_bf16:
        predictor = predictor.to(dtype=torch.bfloat16)

    video_dir = _write_jpeg_video_dir(images_rgb, work_dir / "sam2_frames")
    n = len(images_rgb)
    h, w = images_rgb[0].shape[:2]
    seeds_by_frame: dict[int, list[BlobSeed]] = {}
    for s in seeds:
        seeds_by_frame.setdefault(s.frame_idx, []).append(s)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.autocast("cuda", enabled=False)
    )
    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(video_dir))
        obj_id = 1
        obj_ids: list[int] = []
        for fi in sorted(seeds_by_frame.keys()):
            for seed in seeds_by_frame[fi]:
                pts = np.array([[seed.x, seed.y]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                add_fn = getattr(predictor, "add_new_points_or_box", None) or predictor.add_new_points
                add_kwargs = dict(
                    inference_state=state,
                    frame_idx=int(fi),
                    obj_id=obj_id,
                    points=pts,
                    labels=labels,
                )
                try:
                    add_fn(**add_kwargs)
                except TypeError:
                    predictor.add_new_points(state, int(fi), obj_id, pts, labels)
                obj_ids.append(obj_id)
                obj_id += 1

        masks_by_obj: dict[int, list[np.ndarray]] = {
            oid: [np.zeros((h, w), np.float32) for _ in range(n)] for oid in obj_ids
        }
        for frame_idx, out_obj_ids, out_masks in predictor.propagate_in_video(state):
            for i, oid in enumerate(out_obj_ids):
                oid = int(oid)
                if oid not in masks_by_obj:
                    continue
                m = out_masks[i]
                if torch.is_tensor(m):
                    m = m.detach().float().cpu().numpy()
                m = np.squeeze(m)
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                masks_by_obj[oid][int(frame_idx)] = (m > 0.0).astype(np.float32)

    info = {
        "n_seeds": len(seeds),
        "n_objects": len(masks_by_obj),
        "n_frames": n,
        "checkpoint": str(checkpoint),
        "device": str(torch_device),
    }
    return masks_by_obj, info
