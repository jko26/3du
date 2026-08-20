"""ODISE panoptic unprojection helpers (no GPU / no ODISE weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cork3du.instances import write_odise_debug_panel
from cork3du.odise_infer import OdisePrediction, parse_vocab
from cork3du.reconstruct import _unproject_frame


def test_parse_vocab():
    assert parse_vocab(None) is None
    assert parse_vocab("") is None
    assert parse_vocab("truck, pickup; sky") == [["truck", "pickup"], ["sky"]]


def test_unproject_shape():
    h, w = 4, 6
    depth = np.full((h, w), 2.0, dtype=np.float32)
    K = np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    world = _unproject_frame(depth, K, c2w)
    assert world.shape == (h, w, 3)
    assert np.isfinite(world).all()


def test_odise_debug_panel(tmp_path: Path):
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :] = (30, 40, 50)
    pan = np.zeros((48, 64), dtype=np.int32)
    pan[10:20, 10:30] = 1
    pan[25:40, 40:55] = 2
    pred = OdisePrediction(
        panoptic=pan,
        segments=[
            {"id": 1, "category_id": 0, "isthing": True, "name": "person", "score": 0.9},
            {"id": 2, "category_id": 1, "isthing": True, "name": "car", "score": 0.8},
        ],
    )
    out = write_odise_debug_panel(rgb, pred, out_path=tmp_path / "frame_000.png", frame_idx=0)
    assert out.is_file()
    assert out.stat().st_size > 100
