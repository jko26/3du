"""Window 40/40 lock: independent 2s chunks, residual-only still in final."""

from __future__ import annotations

import numpy as np

from cork3du.track_lock import apply_track_window_lock


def _obj_masks(n: int, h: int = 8, w: int = 8, on: slice | None = None) -> list[np.ndarray]:
    on = on or slice(None)
    out = []
    for t in range(n):
        m = np.zeros((h, w), np.float32)
        if t in range(n)[on]:
            m[:, :] = 1.0
        out.append(m)
    return out


def test_locks_whole_window_when_40pct_hot():
    # 10-frame window @ 5fps / 2s. 4 hot of 10 = 40% → lock all 10.
    n = 10
    p = [np.zeros((8, 8), np.float32) for _ in range(n)]
    for t in (0, 1, 2, 3):
        p[t][:, :] = 1.0
    masks = {1: _obj_masks(n)}
    locked, info, _ = apply_track_window_lock(
        p, masks, window_seconds=2.0, fps=5.0, min_visible_in_window=3
    )
    assert info["window_frames"] == 10
    assert all(m.mean() == 1.0 for m in locked)


def test_adjacent_windows_independent():
    # 20 frames, two windows. Only first window is 40% hot.
    n = 20
    p = [np.zeros((8, 8), np.float32) for _ in range(n)]
    for t in (0, 1, 2, 3):
        p[t][:, :] = 1.0
    masks = {1: _obj_masks(n)}
    locked, _, _ = apply_track_window_lock(
        p, masks, window_seconds=2.0, fps=5.0, min_visible_in_window=3
    )
    assert all(locked[t].mean() == 1.0 for t in range(10))
    assert all(locked[t].mean() == 0.0 for t in range(10, 20))


def test_sparse_window_skipped():
    n = 10
    p = [np.ones((8, 8), np.float32) for _ in range(n)]
    masks = {1: _obj_masks(n, on=slice(0, 2))}  # only 2 visible frames
    locked, info, _ = apply_track_window_lock(
        p, masks, window_seconds=2.0, fps=5.0, min_visible_in_window=3
    )
    assert info["n_windows_skipped_sparse"] >= 1
    assert all(m.mean() == 0.0 for m in locked)
