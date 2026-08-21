"""Make 2023 ODISE importable on cluster PyTorch 2.2.

ODISE was written for torch 1.13. Preflight must load the same modules as
`OdiseModel`, not just `import odise`.
"""

from __future__ import annotations

import collections.abc
import sys
import types
from pathlib import Path


def install_torch_six() -> None:
    """torch._six was removed in PyTorch 2.0; ODISE still imports it."""
    import torch

    if "torch._six" in sys.modules:
        return
    six = types.ModuleType("torch._six")
    six.inf = getattr(torch, "inf", float("inf"))
    six.nan = getattr(torch, "nan", float("nan"))
    six.string_classes = (str, bytes)
    six.int_classes = int
    six.container_abcs = collections.abc
    six.PY3 = True
    sys.modules["torch._six"] = six
    setattr(torch, "_six", six)


def patch_odise_tree(odise_root: Path) -> None:
    """Rewrite known torch-1.13 imports in the local ODISE clone."""
    loop = Path(odise_root) / "odise" / "engine" / "train_loop.py"
    if loop.is_file():
        text = loop.read_text()
        new = text.replace("from torch._six import inf", "from torch import inf")
        if new != text:
            loop.write_text(new)


def prepare_odise(odise_root: Path | None = None) -> None:
    """Call before any `import odise` / mask2former."""
    from . import paths

    root = Path(odise_root) if odise_root else paths.odise_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    m2f = root / "third_party" / "Mask2Former"
    if m2f.is_dir() and str(m2f) not in sys.path:
        sys.path.insert(0, str(m2f))
    install_torch_six()
    patch_odise_tree(root)


def probe_odise_imports() -> list[str]:
    """Import every module OdiseModel touches. Returns missing-error strings."""
    prepare_odise()
    missing: list[str] = []
    mods = (
        "fvcore",
        "detectron2",
        "detectron2.config",
        "detectron2.data",
        "detectron2.engine",
        "detectron2.evaluation",
        "mask2former",
        "mask2former.data.datasets.register_ade20k_panoptic",
        "panopticapi.utils",
        "odise",
        "odise.data",
        "odise.config",
        "odise.checkpoint",
        "odise.engine.defaults",
        "odise.engine.train_loop",
    )
    import importlib

    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    # Same from-imports as OdiseModel.__init__
    try:
        from odise.checkpoint import ODISECheckpointer  # noqa: F401
        from odise.config import instantiate_odise  # noqa: F401
        from odise.data import get_openseg_labels  # noqa: F401
        from odise.engine.defaults import get_model_from_module  # noqa: F401
    except Exception as exc:
        missing.append(f"OdiseModel imports: {exc}")
    return missing
