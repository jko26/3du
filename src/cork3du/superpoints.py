"""Voxel superpoints on a static XYZ cloud (SAI3D-style geometric primitives)."""

from __future__ import annotations

import numpy as np


def adaptive_voxel_size(pts: np.ndarray, *, n_bins: float = 80.0) -> float:
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    extent = float(np.linalg.norm(hi - lo))
    return max(extent / float(n_bins), 1e-4)


def voxel_superpoints(
    pts: np.ndarray,
    *,
    voxel_size: float | None = None,
    min_points: int = 40,
) -> tuple[np.ndarray, dict]:
    """6-connected voxel components. Returns labels in {0..K-1} or -1 (tiny)."""
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32), {"n_superpoints": 0, "voxel_size": 0.0}
    vs = float(voxel_size) if voxel_size else adaptive_voxel_size(pts)
    keys = np.floor(pts / vs).astype(np.int32)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_vox = uniq.shape[0]
    parent = np.arange(n_vox, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    index = {tuple(v.tolist()): i for i, v in enumerate(uniq)}
    for i, v in enumerate(uniq):
        x, y, z = int(v[0]), int(v[1]), int(v[2])
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = index.get((x + dx, y + dy, z + dz))
            if j is not None:
                union(i, j)

    roots = np.array([find(i) for i in range(n_vox)], dtype=np.int32)
    remap: dict[int, int] = {}
    voxel_label = np.empty(n_vox, dtype=np.int32)
    for i, r in enumerate(roots):
        rr = int(r)
        if rr not in remap:
            remap[rr] = len(remap)
        voxel_label[i] = remap[rr]
    labels = voxel_label[inv]
    counts = np.bincount(labels, minlength=int(labels.max()) + 1 if labels.size else 0)
    tiny = np.where(counts < min_points)[0]
    if tiny.size:
        labels = labels.copy()
        labels[np.isin(labels, tiny)] = -1
        keep = sorted(int(i) for i in range(len(counts)) if i not in set(tiny.tolist()))
        remap2 = {old: new for new, old in enumerate(keep)}
        out = np.full(n, -1, dtype=np.int32)
        for i, lab in enumerate(labels):
            if lab >= 0:
                out[i] = remap2[int(lab)]
        labels = out
    n_sp = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    info = {
        "voxel_size": vs,
        "n_voxels": int(n_vox),
        "n_superpoints": n_sp,
        "n_unassigned": int((labels < 0).sum()),
        "min_points": min_points,
    }
    return labels.astype(np.int32), info
