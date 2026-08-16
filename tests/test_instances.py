"""Stage 3–4 lift: superpoints, soft filter, affinity merge."""

from __future__ import annotations

import numpy as np

from cork3du.instances import write_amg_debug_panel
from cork3du.lift import (
    UnionFind,
    affinity_matrix,
    mask_depth_valid_frac,
    mask_dynamic_overlap,
    merge_superpoints,
    superpoints_in_mask,
)
from cork3du.superpoints import voxel_superpoints


def test_two_blobs_two_superpoints():
    a = np.random.default_rng(0).normal(0, 0.02, size=(80, 3))
    b = np.random.default_rng(1).normal(0, 0.02, size=(80, 3)) + np.array([5.0, 0, 0])
    pts = np.concatenate([a, b], axis=0)
    labels, info = voxel_superpoints(pts, voxel_size=0.2, min_points=20)
    assert info["n_superpoints"] == 2
    assert set(labels[:80]) == {0} or set(labels[:80]).isdisjoint(set(labels[80:]))
    assert len(set(labels[:80])) == 1
    assert len(set(labels[80:])) == 1
    assert labels[0] != labels[80]


def test_dynamic_overlap_drops_person_mask():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    dyn = np.zeros((8, 8), dtype=bool)
    dyn[2:6, 2:6] = True
    assert mask_dynamic_overlap(mask, dyn, max_frac=0.4) is True
    # Loose default (0.85): full overlap still drops
    assert mask_dynamic_overlap(mask, dyn, max_frac=0.85) is True
    dyn2 = np.zeros((8, 8), dtype=bool)
    dyn2[0, 0] = True
    assert mask_dynamic_overlap(mask, dyn2, max_frac=0.4) is False
    # Partial bleed (~25%) survives the loose gate
    dyn3 = np.zeros((8, 8), dtype=bool)
    dyn3[2:6, 2:3] = True
    assert mask_dynamic_overlap(mask, dyn3, max_frac=0.85) is False


def test_depth_valid_frac():
    mask = np.ones((4, 4), dtype=bool)
    depth = np.ones((4, 4), dtype=np.float32)
    depth[0] = 0
    assert abs(mask_depth_valid_frac(mask, depth) - 0.75) < 1e-6
    # Loose min_depth_frac=0.10 would keep this mask (0.75 >= 0.10)
    assert mask_depth_valid_frac(mask, depth) >= 0.10


def test_merge_high_affinity():
    aff = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    both = np.array([[0, 3, 3], [3, 0, 3], [3, 3, 0]], dtype=np.float64)
    inst = merge_superpoints(aff, both, thresholds=(0.75, 0.5), min_shared_views=2)
    assert inst[0] == inst[1]
    assert inst[2] != inst[0]


def test_union_find():
    uf = UnionFind(3)
    uf.union(0, 1)
    c = uf.components()
    assert c[0] == c[1]
    assert c[2] != c[0]


def test_superpoints_in_mask():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    vis = np.ones(6, dtype=bool)
    ui = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    vi = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0] = True
    hits = superpoints_in_mask(labels, vis, ui, vi, mask, min_visible=2, in_frac=0.5)
    assert hits == [0]


def test_affinity_from_cooccur():
    co = np.array([[0.0, 4.0], [4.0, 0.0]])
    both = np.array([[0.0, 5.0], [5.0, 0.0]])
    aff = affinity_matrix(co, both)
    assert abs(aff[0, 1] - 0.8) < 1e-9


def test_amg_debug_panel(tmp_path=None):
    from pathlib import Path

    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/cork3du_amg_debug_test")
    root.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :] = (30, 40, 50)
    m1 = np.zeros((48, 64), dtype=bool)
    m1[10:20, 10:30] = True
    m2 = np.zeros((48, 64), dtype=bool)
    m2[25:40, 40:55] = True
    final = np.zeros((48, 64), dtype=bool)
    final[10:20, 10:20] = True
    out = write_amg_debug_panel(
        rgb,
        [m1, m2],
        kept_masks=[m1],
        dropped_dyn=[m2],
        dropped_depth=[],
        final_mask=final,
        out_path=root / "frame_000.png",
        frame_idx=0,
        counts={"sam": 2, "kept": 1, "drop_dyn": 1, "drop_depth": 0},
    )
    assert out.is_file()
    assert out.stat().st_size > 100
