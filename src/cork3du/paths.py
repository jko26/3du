"""Env-driven roots. Code clone ≠ data root.

Bash identifiers cannot start with a digit, so these are CORK3DU_*
(not 3DU_*). Default data path is still /projects/sinus_clinical_data/3du.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA = Path("/projects/sinus_clinical_data/3du")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def code_root() -> Path:
    return Path(os.environ.get("CORK3DU_ROOT", PACKAGE_ROOT)).resolve()


def data_root() -> Path:
    return Path(os.environ.get("CORK3DU_DATA", str(DEFAULT_DATA))).resolve()


def weights_dir() -> Path:
    override = os.environ.get("CORK3DU_WEIGHTS")
    return Path(override).resolve() if override else data_root() / "weights"


def da3_root() -> Path:
    override = os.environ.get("CORK3DU_DA3")
    if override:
        return Path(override).resolve()
    return code_root() / "third_party" / "Depth-Anything-3"


def sam2_root() -> Path:
    override = os.environ.get("CORK3DU_SAM2")
    if override:
        return Path(override).resolve()
    return code_root() / "third_party" / "sam2"


def sam2_checkpoint() -> Path:
    override = os.environ.get("CORK3DU_SAM2_CKPT")
    if override:
        return Path(override).resolve()
    return weights_dir() / "sam2.1_hiera_large.pt"


def da3_weights_dir() -> Path:
    return weights_dir() / "da3_nested_giant_large_1_1"


def ensure_data_dirs() -> Path:
    root = data_root()
    for sub in ("chunks", "scenes", "weights", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
