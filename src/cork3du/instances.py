"""Stage 3–4: SAM2 automatic 2D masks → SAI3D-style superpoint lift → 3D instances.

Uses the SAM2 already in the 3du env (no Semantic-SAM / official SAI3D ScanNet stack).
Dynamic pixels from remask (`masks/final_*.png`) are excluded before lifting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .lift import (
    accumulate_affinity,
    affinity_matrix,
    mask_depth_valid_frac,
    mask_dynamic_overlap,
    merge_superpoints,
    project_points,
    superpoints_in_mask,
)
from .preview import write_preview_html, write_preview_png
from .superpoints import voxel_superpoints

logger = logging.getLogger(__name__)

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _load_dynamic_masks(scene_dir: Path, n: int, h: int, w: int) -> list[np.ndarray] | None:
    paths = sorted((scene_dir / "masks").glob("final_*.png"))
    if len(paths) != n:
        return None
    out = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((m > 127).astype(np.uint8))
    return out


def _instance_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    rgb = rng.random((max(n, 1), 3))
    rgb = 0.35 + 0.65 * rgb
    return rgb.astype(np.float32)


def run_sam2_amg(
    image_rgb: np.ndarray,
    *,
    sam2_root: Path,
    checkpoint: Path,
    device: str,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.85,
    min_mask_region_area: int = 80,
    max_masks: int = 40,
) -> list[np.ndarray]:
    import sys

    import torch

    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    if not hasattr(run_sam2_amg, "_model"):
        logger.info("SAM2 AMG (%s) %s", device, checkpoint)
        model = build_sam2(SAM2_CFG, str(checkpoint), device=device)
        run_sam2_amg._model = model  # type: ignore[attr-defined]
        run_sam2_amg._gen = SAM2AutomaticMaskGenerator(  # type: ignore[attr-defined]
            model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            crop_n_layers=0,
        )
    gen = run_sam2_amg._gen  # type: ignore[attr-defined]
    img = image_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    anns = gen.generate(img)
    anns.sort(key=lambda a: float(a.get("predicted_iou", 0.0)), reverse=True)
    masks = [np.asarray(a["segmentation"]).astype(bool) for a in anns[:max_masks]]
    return masks


def instance_scene(
    scene_dir: Path,
    *,
    sam2_root: Path,
    sam2_checkpoint: Path,
    n_keyframes: int = 8,
    min_depth_frac: float = 0.35,
    max_dyn_frac: float = 0.4,
    min_points: int = 80,
    stuff_frac: float = 0.22,
) -> dict[str, Any]:
    from .preflight import run_preflight
    from .reconstruct import load_stream_rgbs, read_c2w_poses, resize_mask

    run_preflight(require_da3_tree=False, require_sam2=True)
    scene_dir = Path(scene_dir)
    stream_out = scene_dir / "stream_out"
    cloud_path = scene_dir / "cloud.npy"
    if not (stream_out / "results_output").is_dir():
        raise FileNotFoundError(f"missing DA3 stream_out at {stream_out} — run reconstruct first")
    if not cloud_path.is_file():
        raise FileNotFoundError(f"missing {cloud_path} — run remask (or reconstruct+fuse) first")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, npz_files = load_stream_rgbs(stream_out)
    n = len(images)
    h, w = images[0].shape[:2]
    poses = read_c2w_poses(stream_out / "camera_poses.txt")
    if len(poses) != n:
        raise RuntimeError(f"poses {len(poses)} != frames {n}")

    cloud = np.load(cloud_path)
    pts = cloud[:, :3].astype(np.float32)
    cols = cloud[:, 3:6].astype(np.float32) if cloud.shape[1] >= 6 else np.full((pts.shape[0], 3), 0.7)
    if cols.max() > 1.5:
        cols = cols / 255.0
    labels, sp_info = voxel_superpoints(pts, min_points=40)
    n_sp = int(sp_info["n_superpoints"])
    if n_sp < 2:
        raise RuntimeError(f"too few superpoints ({n_sp}) in {scene_dir}")

    dyn = _load_dynamic_masks(scene_dir, n, h, w)
    key_idxs = np.unique(np.linspace(0, n - 1, min(n_keyframes, n), dtype=int))
    co = np.zeros((n_sp, n_sp), dtype=np.float64)
    both = np.zeros((n_sp, n_sp), dtype=np.float64)
    n_masks_kept = 0
    n_masks_drop_dyn = 0
    n_masks_drop_depth = 0

    for fi in key_idxs:
        fi = int(fi)
        data = np.load(npz_files[fi])
        depth = data["depth"].astype(np.float32)
        conf = data["conf"].astype(np.float32)
        k = data["intrinsics"].astype(np.float32)
        dh, dw = depth.shape
        ui, vi, vis = project_points(pts, poses[fi], k, height=dh, width=dw, depth=depth)
        visible_sps = {int(s) for s in np.unique(labels[vis & (labels >= 0)])}

        rgb = images[fi]
        if rgb.shape[0] != dh or rgb.shape[1] != dw:
            rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        masks = run_sam2_amg(rgb, sam2_root=sam2_root, checkpoint=sam2_checkpoint, device=device)
        dyn_f = None
        if dyn is not None:
            dyn_f = resize_mask(dyn[fi].astype(np.float32), dh, dw) > 0.5

        voters_per_mask: list[list[int]] = []
        for m in masks:
            if m.shape != (dh, dw):
                m = cv2.resize(m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask_dynamic_overlap(m, dyn_f, max_frac=max_dyn_frac):
                n_masks_drop_dyn += 1
                continue
            if mask_depth_valid_frac(m, depth, conf) < min_depth_frac:
                n_masks_drop_depth += 1
                continue
            voters = superpoints_in_mask(labels, vis, ui, vi, m)
            if len(voters) >= 1:
                voters_per_mask.append(voters)
                n_masks_kept += 1
        accumulate_affinity(n_sp, voters_per_mask, visible_sps, co, both)
        logger.info("frame %d: sam=%d kept=%d vis_sp=%d", fi, len(masks), len(voters_per_mask), len(visible_sps))

    aff = affinity_matrix(co, both)
    inst_of_sp = merge_superpoints(aff, both)
    point_inst = np.full(pts.shape[0], -1, dtype=np.int32)
    for sp in range(n_sp):
        point_inst[labels == sp] = int(inst_of_sp[sp])

    # compact ids after dropping tiny / stuff
    records: list[dict[str, Any]] = []
    out_dir = scene_dir / "instances"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    next_id = 0
    compact = np.full(pts.shape[0], -1, dtype=np.int32)
    n_raw = int(point_inst.max()) + 1 if point_inst.max() >= 0 else 0
    for raw in range(n_raw):
        sel = point_inst == raw
        n_pts = int(sel.sum())
        if n_pts < min_points:
            continue
        frac = n_pts / max(pts.shape[0], 1)
        xyz = pts[sel]
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        z_span = float(extent[2])
        xy_span = float(np.linalg.norm(extent[:2]))
        is_ground = z_span < 0.18 * max(xy_span, 1e-6) and xyz[:, 2].mean() < np.percentile(pts[:, 2], 25)
        is_stuff = frac >= stuff_frac or is_ground
        rec = {
            "id": next_id,
            "n_points": n_pts,
            "frac": float(frac),
            "centroid": xyz.mean(axis=0).tolist(),
            "is_stuff": bool(is_stuff),
            "is_ground": bool(is_ground),
        }
        inst_cloud = np.concatenate([xyz, np.clip(cols[sel], 0, 1)], axis=1)
        np.save(out_dir / f"instance_{next_id:03d}.npy", inst_cloud.astype(np.float32))
        compact[sel] = next_id
        records.append(rec)
        next_id += 1

    np.save(out_dir / "point_instances.npy", compact)
    pal = _instance_palette(max(next_id, 1))
    colored = cols.copy()
    for i in range(next_id):
        colored[compact == i] = pal[i]
    colored[compact < 0] *= 0.25
    vis_cloud = np.concatenate([pts, colored], axis=1)
    things = [r for r in records if not r["is_stuff"]]
    title = f"{scene_dir.name} instances ({len(things)} things / {next_id} total)"
    png = write_preview_png(vis_cloud, out_dir / "preview.png", title=title)
    html = write_preview_html(vis_cloud, out_dir / "preview.html", title=title)
    meta = {
        "pipeline": "sam2_amg_superpoint_affinity",
        "n_cloud_points": int(pts.shape[0]),
        "n_keyframes": int(len(key_idxs)),
        "key_idxs": [int(i) for i in key_idxs],
        "superpoints": sp_info,
        "n_masks_kept": n_masks_kept,
        "n_masks_drop_dyn": n_masks_drop_dyn,
        "n_masks_drop_depth": n_masks_drop_depth,
        "n_instances": next_id,
        "n_things": len(things),
        "n_stuff": next_id - len(things),
        "had_dynamic_masks": dyn is not None,
        "instances": records,
        "preview_png": str(png),
        "preview_html": str(html),
        "device": device,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    scene_meta = scene_dir / "meta.json"
    old = json.loads(scene_meta.read_text()) if scene_meta.is_file() else {}
    old["instances"] = {k: meta[k] for k in meta if k != "instances"}
    scene_meta.write_text(json.dumps(old, indent=2, default=str) + "\n")
    logger.info(
        "instances: %d things / %d total (%d superpoints, %d sam masks) → %s",
        len(things),
        next_id,
        n_sp,
        n_masks_kept,
        png,
    )
    return meta
