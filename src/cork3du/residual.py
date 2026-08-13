"""Torchvision RAFT residual vs expected ego-flow, floored depth-norm → p_resid."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .reconstruct import load_stream_rgbs, read_c2w_poses

logger = logging.getLogger(__name__)


def sigmoid_recenter(x: np.ndarray, *, center: float, k: float) -> np.ndarray:
    z = np.clip(k * (x.astype(np.float32) - float(center)), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _load_raft(device: torch.device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms()


@torch.inference_mode()
def compute_raft_flow(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    *,
    model,
    transforms,
    device: torch.device,
) -> np.ndarray:
    ta = torch.from_numpy(rgb_a).permute(2, 0, 1).contiguous()
    tb = torch.from_numpy(rgb_b).permute(2, 0, 1).contiguous()
    img1, img2 = transforms(ta, tb)
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    flows = model(img1, img2)
    flow = flows[-1][0].permute(1, 2, 0).detach().float().cpu().numpy()
    h, w = rgb_a.shape[:2]
    if flow.shape[0] != h or flow.shape[1] != w:
        fh, fw = flow.shape[:2]
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= w / float(fw)
        flow[..., 1] *= h / float(fh)
    return flow.astype(np.float32)


def expected_ego_flow(
    depth: np.ndarray,
    K: np.ndarray,
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 1e-4)
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    cam_a = np.stack([x, y, z], axis=-1)
    R_a, t_a = c2w_a[:3, :3], c2w_a[:3, 3]
    R_b, t_b = c2w_b[:3, :3], c2w_b[:3, 3]
    world = cam_a @ R_a.T + t_a[None, None, :]
    cam_b = (world - t_b[None, None, :]) @ R_b
    zb = cam_b[..., 2]
    valid = valid & np.isfinite(zb) & (zb > 1e-4)
    ub = fx * (cam_b[..., 0] / zb) + cx
    vb = fy * (cam_b[..., 1] / zb) + cy
    valid = valid & np.isfinite(ub) & np.isfinite(vb)
    margin = 0.25 * max(h, w)
    valid = valid & (ub > -margin) & (vb > -margin) & (ub < w + margin) & (vb < h + margin)
    flow = np.stack([ub - xs, vb - ys], axis=-1).astype(np.float32)
    flow[~valid] = 0.0
    return flow, valid


def normalize_residual(
    residual_mag: np.ndarray,
    flow_ego: np.ndarray,
    valid: np.ndarray,
    *,
    eps: float = 1.0,
    ego_floor_px: float = 3.0,
) -> np.ndarray:
    ego_mag = np.linalg.norm(flow_ego, axis=-1).astype(np.float32)
    denom = np.maximum(ego_mag + float(eps), float(ego_floor_px))
    out = (residual_mag / denom).astype(np.float32)
    out[~valid] = 0.0
    return out


def compute_p_resid(
    stream_out: Path,
    *,
    pair_stride: int = 1,
    ego_eps_px: float = 1.0,
    ego_floor_px: float = 3.0,
    resid_center: float = 0.35,
    k: float = 10.0,
    device: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Return (p_resid, residual_valid, info) aligned to stream RGB size."""
    poses = read_c2w_poses(stream_out / "camera_poses.txt")
    images, npz_files = load_stream_rgbs(stream_out)
    n = len(npz_files)
    if n != len(poses) or n < 2:
        raise RuntimeError(f"Need matching poses/npz >=2, got poses={len(poses)} npz={n}")

    depths, Ks, confs = [], [], []
    for p in npz_files:
        d = np.load(p)
        depths.append(d["depth"].astype(np.float32))
        Ks.append(d["intrinsics"].astype(np.float32))
        confs.append(
            d["conf"].astype(np.float32)
            if "conf" in d.files
            else np.ones_like(d["depth"], np.float32)
        )

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("RAFT residual on %s", torch_device)
    model, transforms = _load_raft(torch_device)
    h0, w0 = depths[0].shape
    sum_norm = [np.zeros((h0, w0), np.float32) for _ in range(n)]
    sum_cnt = [np.zeros((h0, w0), np.float32) for _ in range(n)]

    def prep_rgb(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if rgb.shape[:2] != shape:
            return cv2.resize(rgb, (shape[1], shape[0]))
        return rgb

    for i in range(0, n - 1, pair_stride):
        j = i + pair_stride
        if j >= n:
            break
        rgb_i = prep_rgb(images[i], depths[i].shape)
        rgb_j = prep_rgb(images[j], depths[j].shape)
        flow_obs = compute_raft_flow(rgb_i, rgb_j, model=model, transforms=transforms, device=torch_device)
        flow_ego, valid = expected_ego_flow(depths[i], Ks[i], poses[i], poses[j])
        valid = valid & (confs[i] > 0)
        if flow_obs.shape[:2] != flow_ego.shape[:2]:
            flow_obs = cv2.resize(
                flow_obs, (flow_ego.shape[1], flow_ego.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        mag = np.linalg.norm(flow_obs - flow_ego, axis=-1).astype(np.float32)
        mag[~valid] = 0.0
        norm = normalize_residual(mag, flow_ego, valid, eps=ego_eps_px, ego_floor_px=ego_floor_px)
        sum_norm[i] += norm
        sum_cnt[i] += valid.astype(np.float32)

        flow_obs_b = compute_raft_flow(rgb_j, rgb_i, model=model, transforms=transforms, device=torch_device)
        flow_ego_b, valid_b = expected_ego_flow(depths[j], Ks[j], poses[j], poses[i])
        valid_b = valid_b & (confs[j] > 0)
        if flow_obs_b.shape[:2] != flow_ego_b.shape[:2]:
            flow_obs_b = cv2.resize(
                flow_obs_b, (flow_ego_b.shape[1], flow_ego_b.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        mag_b = np.linalg.norm(flow_obs_b - flow_ego_b, axis=-1).astype(np.float32)
        mag_b[~valid_b] = 0.0
        norm_b = normalize_residual(
            mag_b, flow_ego_b, valid_b, eps=ego_eps_px, ego_floor_px=ego_floor_px
        )
        sum_norm[j] += norm_b
        sum_cnt[j] += valid_b.astype(np.float32)

    p_resid: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    h, w = images[0].shape[:2]
    for i in range(n):
        cnt = np.maximum(sum_cnt[i], 1e-6)
        valid = sum_cnt[i] >= 0.5
        rn = (sum_norm[i] / cnt).astype(np.float32)
        rn[~valid] = 0.0
        pr = sigmoid_recenter(rn, center=resid_center, k=k)
        pr = np.where(valid, pr, 0.0).astype(np.float32)
        if pr.shape[:2] != (h, w):
            pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
            valid_r = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        else:
            valid_r = valid
        p_resid.append(pr)
        valid_maps.append(valid_r.astype(np.float32))

    info = {
        "backend": "torchvision_raft_large + floored residual_norm + sigmoid",
        "ego_floor_px": ego_floor_px,
        "resid_center": resid_center,
        "n_frames": n,
        "device": str(torch_device),
    }
    return p_resid, valid_maps, info
