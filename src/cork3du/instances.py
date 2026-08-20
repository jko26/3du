"""Stage 3–4: ODISE open-vocab panoptic → depth/pose unprojection → labeled 3D cloud.

RGB frames go to ODISE (semantic labels + instance masks). Each labeled pixel is
unprojected with DA3 depth + camera pose — same geometry path as fuse_masked_cloud.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .odise_infer import OdiseModel, OdisePrediction, parse_vocab
from .preview import write_preview_html, write_preview_png

logger = logging.getLogger(__name__)


def _instance_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    rgb = rng.random((max(n, 1), 3))
    return (0.35 + 0.65 * rgb).astype(np.float32)


def _overlay_panoptic(rgb: np.ndarray, pred: OdisePrediction, *, alpha: float = 0.45) -> np.ndarray:
    if rgb.dtype != np.uint8:
        base = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        base = rgb
    out = base.astype(np.float32).copy()
    pal = _instance_palette(max((s["id"] for s in pred.segments), default=0) + 1)
    for seg in pred.segments:
        sid = int(seg["id"])
        mb = pred.panoptic == sid
        if not np.any(mb):
            continue
        c = pal[sid % len(pal)].astype(np.float32) * 255.0
        out[mb] = (1.0 - alpha) * out[mb] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def _panel_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_odise_debug_panel(
    rgb: np.ndarray,
    pred: OdisePrediction,
    *,
    out_path: Path,
    frame_idx: int,
) -> Path:
    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb
    ov = _overlay_panoptic(rgb_u8, pred)
    n_thing = sum(1 for s in pred.segments if s.get("isthing"))
    panels = [
        _panel_label(rgb_u8, f"frame {frame_idx} RGB"),
        _panel_label(ov, f"ODISE panoptic n={len(pred.segments)} things={n_thing}"),
    ]
    h = min(p.shape[0] for p in panels)
    w = min(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA) for p in panels]
    strip = np.concatenate(panels, axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return out_path


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
        out.append((m > 127).astype(bool))
    return out


def _unproject_labeled_frame(
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image: np.ndarray,
    pred: OdisePrediction,
    dyn_mask: np.ndarray | None,
    next_global_id: int,
    spatial_stride: int,
    motion_mask_thre: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[int, dict[str, Any]]]:
    """Unproject valid pixels; return xyz, rgb, semantic, instance, conf, next_id, id_meta."""
    from .reconstruct import _unproject_frame, resize_mask

    world = _unproject_frame(depth, K, c2w.astype(np.float32))
    h, w = depth.shape
    pan = pred.panoptic
    if pan.shape != (h, w):
        pan = cv2.resize(pan.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    # Map local panoptic segment id → (global instance id, category_id, isthing, name)
    local_to_global: dict[int, int] = {}
    id_meta: dict[int, dict[str, Any]] = {}
    gid = next_global_id
    for seg in pred.segments:
        lid = int(seg["id"])
        local_to_global[lid] = gid
        id_meta[gid] = {
            "category_id": int(seg["category_id"]),
            "name": str(seg.get("name", f"class_{seg['category_id']}")),
            "isthing": bool(seg.get("isthing", False)),
            "score": seg.get("score"),
        }
        gid += 1

    sem_map = np.full((h, w), -1, dtype=np.int32)
    inst_map = np.full((h, w), -1, dtype=np.int32)
    for lid, g in local_to_global.items():
        mb = pan == lid
        sem_map[mb] = id_meta[g]["category_id"]
        inst_map[mb] = g

    pts = world[::spatial_stride, ::spatial_stride].reshape(-1, 3)
    cols = image[::spatial_stride, ::spatial_stride].reshape(-1, 3).astype(np.float32) / 255.0
    c = conf[::spatial_stride, ::spatial_stride].reshape(-1)
    sem = sem_map[::spatial_stride, ::spatial_stride].reshape(-1)
    inst = inst_map[::spatial_stride, ::spatial_stride].reshape(-1)

    keep = np.isfinite(pts).all(axis=1) & (c > 0) & (pts[:, 2] != 0) & (inst >= 0)
    if dyn_mask is not None:
        static = resize_mask(dyn_mask.astype(np.float32), h, w) <= motion_mask_thre
        keep = keep & static[::spatial_stride, ::spatial_stride].reshape(-1)

    return pts[keep], cols[keep], sem[keep], inst[keep], c[keep], gid, id_meta


def instance_scene(
    scene_dir: Path,
    *,
    odise_root: Path,
    vocab: str | None = None,
    label_sets: Sequence[str] | None = None,
    frame_stride: int = 4,
    spatial_stride: int = 2,
    conf_threshold_coef: float = 0.75,
    max_points: int = 400_000,
    min_points: int = 80,
    write_debug: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    from .preflight import run_preflight
    from .reconstruct import load_stream_rgbs, read_c2w_poses

    run_preflight(require_da3_tree=False, require_sam2=False, require_odise=True)
    scene_dir = Path(scene_dir)
    stream_out = scene_dir / "stream_out"
    if not (stream_out / "results_output").is_dir():
        raise FileNotFoundError(f"missing DA3 stream_out at {stream_out} — run reconstruct first")

    images, npz_files = load_stream_rgbs(stream_out)
    n = len(images)
    h0, w0 = images[0].shape[:2]
    poses = read_c2w_poses(stream_out / "camera_poses.txt")
    if len(poses) != n:
        raise RuntimeError(f"poses {len(poses)} != frames {n}")

    dyn = _load_dynamic_masks(scene_dir, n, h0, w0)
    frame_idxs = list(range(0, n, max(1, int(frame_stride))))
    if frame_idxs[-1] != n - 1:
        frame_idxs.append(n - 1)

    out_dir = scene_dir / "instances"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    debug_dir = out_dir / "odise_debug"
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_vocab(vocab)
    sets = tuple(label_sets) if label_sets else ("COCO", "ADE", "LVIS")

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_sem: list[np.ndarray] = []
    all_inst: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    instance_meta: dict[int, dict[str, Any]] = {}
    next_gid = 0
    debug_paths: list[str] = []
    n_segments = 0

    with OdiseModel(
        odise_root=odise_root,
        label_sets=sets,
        vocab=labels,
        device=device,
    ) as odise:
        for fi in frame_idxs:
            data = np.load(npz_files[fi])
            depth = data["depth"].astype(np.float32)
            conf = data["conf"].astype(np.float32)
            k = data["intrinsics"].astype(np.float32)
            rgb = images[fi]
            dh, dw = depth.shape
            if rgb.shape[0] != dh or rgb.shape[1] != dw:
                rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

            pred = odise.predict(rgb)
            n_segments += len(pred.segments)
            dyn_f = dyn[fi] if dyn is not None else None
            if dyn_f is not None and dyn_f.shape != (dh, dw):
                dyn_f = cv2.resize(dyn_f.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)

            pts, cols, sem, inst, c, next_gid, id_meta = _unproject_labeled_frame(
                depth=depth,
                conf=conf,
                K=k,
                c2w=poses[fi],
                image=rgb,
                pred=pred,
                dyn_mask=dyn_f,
                next_global_id=next_gid,
                spatial_stride=spatial_stride,
            )
            instance_meta.update(id_meta)
            all_pts.append(pts)
            all_cols.append(cols)
            all_sem.append(sem)
            all_inst.append(inst)
            all_conf.append(c)
            logger.info(
                "frame %d: segments=%d points=%d (things=%d)",
                fi,
                len(pred.segments),
                pts.shape[0],
                sum(1 for s in pred.segments if s.get("isthing")),
            )
            if write_debug:
                dbg = write_odise_debug_panel(
                    rgb, pred, out_path=debug_dir / f"frame_{fi:03d}.png", frame_idx=fi
                )
                debug_paths.append(str(dbg))

    if not all_pts or sum(p.shape[0] for p in all_pts) == 0:
        raise RuntimeError(f"No labeled points after ODISE unprojection in {scene_dir}")

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    sem = np.concatenate(all_sem, axis=0)
    inst = np.concatenate(all_inst, axis=0)
    confs = np.concatenate(all_conf, axis=0)

    conf_thr = float(np.mean(confs) * conf_threshold_coef) if confs.size else 0.0
    conf_ok = confs >= conf_thr
    pts, cols, sem, inst = pts[conf_ok], cols[conf_ok], sem[conf_ok], inst[conf_ok]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts, cols, sem, inst = pts[idx], cols[idx], sem[idx], inst[idx]

    # Compact instance ids after dropping tiny regions; keep semantics.
    records: list[dict[str, Any]] = []
    compact = np.full(pts.shape[0], -1, dtype=np.int32)
    next_id = 0
    unique_inst = sorted(int(x) for x in np.unique(inst) if x >= 0)
    for raw in unique_inst:
        sel = inst == raw
        n_pts = int(sel.sum())
        if n_pts < min_points:
            continue
        meta = instance_meta.get(raw, {})
        xyz = pts[sel]
        rec = {
            "id": next_id,
            "n_points": n_pts,
            "frac": float(n_pts / max(pts.shape[0], 1)),
            "centroid": xyz.mean(axis=0).tolist(),
            "category_id": int(meta.get("category_id", -1)),
            "name": str(meta.get("name", "unknown")),
            "isthing": bool(meta.get("isthing", False)),
            "score": meta.get("score"),
        }
        inst_cloud = np.concatenate([xyz, np.clip(cols[sel], 0, 1)], axis=1)
        np.save(out_dir / f"instance_{next_id:03d}.npy", inst_cloud.astype(np.float32))
        compact[sel] = next_id
        records.append(rec)
        next_id += 1

    labeled = np.concatenate(
        [
            pts.astype(np.float32),
            np.clip(cols.astype(np.float32), 0, 1),
            sem.astype(np.float32)[:, None],
            compact.astype(np.float32)[:, None],
        ],
        axis=1,
    )
    np.save(out_dir / "cloud_labeled.npy", labeled)  # (N,8) xyzrgb + semantic + instance
    np.save(out_dir / "point_semantics.npy", sem.astype(np.int32))
    np.save(out_dir / "point_instances.npy", compact)

    cat_catalog: dict[str, Any] = {}
    for rec in records:
        cid = rec["category_id"]
        cat_catalog[str(cid)] = {"name": rec["name"], "isthing": rec["isthing"]}
    for _gid, imeta in instance_meta.items():
        cid = int(imeta["category_id"])
        cat_catalog.setdefault(str(cid), {"name": imeta["name"], "isthing": imeta["isthing"]})
    (out_dir / "labels.json").write_text(json.dumps(cat_catalog, indent=2) + "\n")

    pal = _instance_palette(max(next_id, 1))
    colored = cols.copy()
    for i in range(next_id):
        colored[compact == i] = pal[i]
    colored[compact < 0] *= 0.25
    vis_cloud = np.concatenate([pts, colored], axis=1)
    things = [r for r in records if r["isthing"]]
    title = f"{scene_dir.name} ODISE ({len(things)} things / {next_id} segments)"
    png = write_preview_png(vis_cloud, out_dir / "preview.png", title=title)
    html = write_preview_html(vis_cloud, out_dir / "preview.html", title=title)

    meta = {
        "pipeline": "odise_panoptic_unproject",
        "n_cloud_points": int(pts.shape[0]),
        "n_frames": int(len(frame_idxs)),
        "frame_idxs": [int(i) for i in frame_idxs],
        "frame_stride": int(frame_stride),
        "spatial_stride": int(spatial_stride),
        "conf_threshold": conf_thr,
        "n_odise_segments_2d": int(n_segments),
        "n_instances": next_id,
        "n_things": len(things),
        "n_stuff": next_id - len(things),
        "had_dynamic_masks": dyn is not None,
        "label_sets": list(sets),
        "vocab": vocab,
        "odise_debug": debug_paths,
        "instances": records,
        "preview_png": str(png),
        "preview_html": str(html),
        "device": device or "auto",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    scene_meta = scene_dir / "meta.json"
    old = json.loads(scene_meta.read_text()) if scene_meta.is_file() else {}
    old["instances"] = {k: meta[k] for k in meta if k != "instances"}
    scene_meta.write_text(json.dumps(old, indent=2, default=str) + "\n")
    logger.info(
        "ODISE instances: %d things / %d total from %d frames → %s",
        len(things),
        next_id,
        len(frame_idxs),
        png,
    )
    return meta
