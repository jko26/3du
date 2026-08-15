"""Project superpoints into SAM masks and merge by multi-view co-occurrence."""

from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def components(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.p))]
        remap: dict[int, int] = {}
        out = np.empty(len(self.p), dtype=np.int32)
        for i, r in enumerate(roots):
            if r not in remap:
                remap[r] = len(remap)
            out[i] = remap[r]
        return out


def project_points(
    pts: np.ndarray,
    c2w: np.ndarray,
    k: np.ndarray,
    *,
    height: int,
    width: int,
    depth: np.ndarray | None = None,
    rel_depth_tol: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, visible_bool). visible ⇒ in-frame, z>0, optional depth agree."""
    w2c = np.linalg.inv(c2w.astype(np.float64))
    cam = (pts.astype(np.float64) @ w2c[:3, :3].T) + w2c[:3, 3]
    z = cam[:, 2]
    u = k[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + k[0, 2]
    v = k[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + k[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    vis = (z > 1e-4) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if depth is not None:
        d = np.zeros(pts.shape[0], dtype=np.float64)
        ok = vis.copy()
        d[ok] = depth[vi[ok], ui[ok]].astype(np.float64)
        rel = np.abs(d - z) / np.maximum(np.abs(z), 1e-6)
        vis = vis & (d > 0) & (rel < rel_depth_tol)
    return ui, vi, vis


def mask_dynamic_overlap(mask: np.ndarray, dyn: np.ndarray | None, *, max_frac: float = 0.4) -> bool:
    """True if the 2D mask is too dynamic (drop it)."""
    if dyn is None:
        return False
    m = mask.astype(bool)
    if m.sum() == 0:
        return True
    return float(dyn.astype(bool)[m].mean()) > max_frac


def mask_depth_valid_frac(mask: np.ndarray, depth: np.ndarray, conf: np.ndarray | None = None) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    ok = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        ok = ok & (conf > 0)
    return float(ok[m].mean())


def superpoints_in_mask(
    labels: np.ndarray,
    vis: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    mask: np.ndarray,
    *,
    min_visible: int = 8,
    in_frac: float = 0.5,
) -> list[int]:
    """Superpoints whose visible projections mostly fall inside mask."""
    n_sp = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    inside = np.zeros(ui.shape[0], dtype=bool)
    sel = vis
    inside[sel] = mask[vi[sel], ui[sel]].astype(bool)
    hits: list[int] = []
    for sp in range(n_sp):
        members = labels == sp
        vis_sp = members & vis
        n_vis = int(vis_sp.sum())
        if n_vis < min_visible:
            continue
        if float(inside[vis_sp].mean()) >= in_frac:
            hits.append(sp)
    return hits


def accumulate_affinity(
    n_sp: int,
    voters_per_mask: list[list[int]],
    visible_sps: set[int],
    co: np.ndarray,
    both_vis: np.ndarray,
) -> None:
    vis_list = sorted(visible_sps)
    for a_i, a in enumerate(vis_list):
        for b in vis_list[a_i + 1 :]:
            both_vis[a, b] += 1
            both_vis[b, a] += 1
    for voters in voters_per_mask:
        for i, a in enumerate(voters):
            for b in voters[i + 1 :]:
                co[a, b] += 1
                co[b, a] += 1


def affinity_matrix(co: np.ndarray, both_vis: np.ndarray) -> np.ndarray:
    denom = np.maximum(both_vis, 1.0)
    return co / denom


def merge_superpoints(
    aff: np.ndarray,
    both_vis: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.75, 0.5),
    min_shared_views: int = 2,
) -> np.ndarray:
    """Progressive union-find (high affinity first), SAI3D-style."""
    n = aff.shape[0]
    uf = UnionFind(n)
    pairs = [
        (float(aff[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if both_vis[i, j] >= min_shared_views
    ]
    pairs.sort(reverse=True)
    for thr in thresholds:
        for a, i, j in pairs:
            if a >= thr:
                uf.union(i, j)
    return uf.components()
