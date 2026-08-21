"""Fail fast with every missing import, not one ModuleNotFoundError per job.

DA3-Streaming imports loop-closure and export stacks at module load even when
loop_enable=False. Those packages are listed here so setup.sh / `cork3du preflight`
catch them before ffmpeg/GPU work starts.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

from . import paths

logger = logging.getLogger(__name__)

# Pip name → import name. Skip torch-dependent packages that setup installs --no-deps.
PIP_IMPORTS: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("opencv-python-headless", "cv2"),
    ("Pillow", "PIL"),
    ("pyyaml", "yaml"),
    ("matplotlib", "matplotlib"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("einops", "einops"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("faiss-cpu", "faiss"),
    ("numba", "numba"),
    ("plyfile", "plyfile"),
    ("pandas", "pandas"),
    ("prettytable", "prettytable"),
    ("trimesh", "trimesh"),
    ("scikit-learn", "sklearn"),
    ("addict", "addict"),
    ("evo", "evo"),
    ("moviepy==1.0.3", "moviepy.editor"),
    ("imageio", "imageio"),
    ("pycolmap", "pycolmap"),
    ("rich", "rich"),
    ("omegaconf", "omegaconf"),
    ("requests", "requests"),
    ("hydra-core", "hydra"),
    ("iopath", "iopath"),
    ("pypose", "pypose"),
]

TORCH_IMPORTS = ("torch", "torchvision", "torchvision.models.optical_flow")


def _ensure_sys_path(*dirs: Path) -> None:
    for d in dirs:
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))


def da3_pythonpath(da3_repo: Path) -> list[str]:
    streaming = da3_repo / "da3_streaming"
    src = da3_repo / "src"
    return [str(p) for p in (src, streaming) if p.is_dir()]


def apply_da3_pythonpath(da3_repo: Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """Put DA3 `src/` + `da3_streaming/` on PYTHONPATH (do not pip-install DA3; it pulls xformers)."""
    out = dict(env if env is not None else os.environ)
    extra = da3_pythonpath(da3_repo)
    old = out.get("PYTHONPATH", "")
    out["PYTHONPATH"] = os.pathsep.join(extra + ([old] if old else []))
    return out


def check_runtime_imports(
    *,
    da3_repo: Path | None = None,
    sam2_root: Path | None = None,
    odise_root: Path | None = None,
    require_da3_tree: bool = True,
    require_sam2: bool = True,
    require_odise: bool = False,
) -> list[str]:
    missing: list[str] = []

    for pip_name, mod in PIP_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{mod}  (pip {pip_name}): {exc}")

    # DA3 imports einops.einsum; einops 0.3 (pulled by sdkit) breaks that.
    try:
        from einops import einsum  # noqa: F401
    except Exception as exc:
        missing.append(f"einops.einsum (need einops>=0.8, not 0.3 from sdkit): {exc}")

    for mod in TORCH_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{mod}  (cluster pytorch module, do not pip install): {exc}")

    da3_repo = da3_repo or paths.da3_root()
    sam2_root = sam2_root or paths.sam2_root()
    odise_root = odise_root or paths.odise_root()
    _ensure_sys_path(*[Path(p) for p in da3_pythonpath(da3_repo)])
    if sam2_root.is_dir():
        _ensure_sys_path(sam2_root)
    if odise_root.is_dir():
        _ensure_sys_path(odise_root)
        m2f = odise_root / "third_party" / "Mask2Former"
        if m2f.is_dir():
            _ensure_sys_path(m2f)

    if require_da3_tree:
        if not (da3_repo / "da3_streaming" / "da3_streaming.py").is_file():
            missing.append(f"DA3 clone missing at {da3_repo} (bash scripts/setup.sh)")
        else:
            for mod in (
                "depth_anything_3.api",
                "loop_utils.sim3loop",
                "loop_utils.sim3utils",
                "loop_utils.loop_detector",
            ):
                try:
                    importlib.import_module(mod)
                except Exception as exc:
                    missing.append(f"{mod}: {exc}")

    if require_sam2:
        try:
            importlib.import_module("sam2.build_sam")
        except Exception as exc:
            missing.append(f"sam2.build_sam: {exc}")
        ckpt = paths.sam2_checkpoint()
        if not ckpt.is_file():
            missing.append(f"SAM2 checkpoint missing: {ckpt}")

    if require_odise:
        if not (odise_root / "odise").is_dir():
            missing.append(f"ODISE clone missing at {odise_root} (bash scripts/setup.sh)")
        else:
            from .odise_compat import probe_odise_imports

            missing.extend(probe_odise_imports())

    return missing


def run_preflight(
    *,
    require_da3_tree: bool = True,
    require_sam2: bool = True,
    require_odise: bool = False,
) -> None:
    missing = check_runtime_imports(
        require_da3_tree=require_da3_tree,
        require_sam2=require_sam2,
        require_odise=require_odise,
    )
    if not missing:
        logger.info("preflight ok")
        return
    body = "\n".join(f"  - {m}" for m in missing)
    raise RuntimeError(
        "Missing runtime imports (install all at once, then re-run):\n"
        f"{body}\n\n"
        "On the cluster, with the venv + pytorch module loaded:\n"
        "  bash scripts/setup.sh\n"
        "Do not pip-install torch, torchvision, xformers, or numpy>=2."
    )
