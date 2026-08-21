"""Make 2023 ODISE importable on cluster PyTorch 2.2.

ODISE was written for torch 1.13. Preflight must load the same modules as
`OdiseModel`, not just `import odise`.
"""

from __future__ import annotations

import collections.abc
import hashlib
import logging
import os
import shutil
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenAI CLIP ViT-L/14@336 — open_clip 2.0.2 downloads this from azureedge
# (blocked by the cluster proxy). Same SHA as open_clip/pretrained.py.
CLIP_336_NAME = "ViT-L-14-336px.pt"
CLIP_336_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
CLIP_336_HF = (
    ("camenduru/CLIP", CLIP_336_NAME),
    ("xianbao/clip", CLIP_336_NAME),
    ("lllyasviel/misc", CLIP_336_NAME),
)


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_openai_clip_home(dest: Path) -> None:
    """open_clip 2.0.2 defaults to ~/.cache/clip/<filename>."""
    home = Path.home() / ".cache" / "clip" / dest.name
    home.parent.mkdir(parents=True, exist_ok=True)
    if home.is_file() or home.is_symlink() or home.exists():
        return
    try:
        home.symlink_to(dest)
    except OSError:
        shutil.copy2(dest, home)


def ensure_openai_clip_336() -> Path:
    """Fetch CLIP L/14@336 via Hugging Face (not azureedge / urllib)."""
    from . import paths

    dest = paths.clip_cache_dir() / CLIP_336_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and _sha256(dest) == CLIP_336_SHA256:
        _link_openai_clip_home(dest)
        return dest

    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for repo, filename in CLIP_336_HF:
        try:
            logger.info("Downloading OpenAI CLIP %s from hf.co/%s …", filename, repo)
            got = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(dest.parent),
                )
            )
            if got.resolve() != dest.resolve():
                shutil.copy2(got, dest)
            if _sha256(dest) != CLIP_336_SHA256:
                logger.warning("CLIP checksum mismatch from %s; trying next mirror", repo)
                dest.unlink(missing_ok=True)
                continue
            _link_openai_clip_home(dest)
            logger.info("CLIP weights → %s", dest)
            return dest
        except Exception as exc:
            last = exc
            logger.warning("CLIP mirror %s failed: %s", repo, exc)

    import requests

    urls = (
        f"https://huggingface.co/{repo}/resolve/main/{fn}" for repo, fn in CLIP_336_HF
    )
    for url in urls:
        try:
            logger.info("Downloading CLIP from %s", url)
            with requests.get(url, stream=True, timeout=600, allow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest.with_name(dest.name + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            if _sha256(tmp) != CLIP_336_SHA256:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dest)
            _link_openai_clip_home(dest)
            logger.info("CLIP weights → %s", dest)
            return dest
        except Exception as exc:
            last = exc
            logger.warning("CLIP url failed: %s", exc)
    raise RuntimeError(
        f"Could not download {CLIP_336_NAME} from Hugging Face "
        f"(cluster blocks OpenAI azureedge). Last error: {last}"
    )


def patch_open_clip_download() -> None:
    """Stop open_clip from using urllib against azureedge.net (proxy 403)."""
    try:
        import open_clip.pretrained as pretrained
    except Exception:
        return

    orig = pretrained.download_pretrained_from_url

    def _download(url: str, cache_dir: str | None = None):
        if CLIP_336_NAME in url or CLIP_336_SHA256[:16] in url:
            return str(ensure_openai_clip_336())
        cache_dir = cache_dir or os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        target = os.path.join(cache_dir, os.path.basename(url))
        if os.path.isfile(target):
            return target
        try:
            import requests

            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = target + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, target)
            return target
        except Exception:
            return orig(url, cache_dir=cache_dir)

    pretrained.download_pretrained_from_url = _download


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
    patch_open_clip_download()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


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
    try:
        from odise.checkpoint import ODISECheckpointer  # noqa: F401
        from odise.config import instantiate_odise  # noqa: F401
        from odise.data import get_openseg_labels  # noqa: F401
        from odise.engine.defaults import get_model_from_module  # noqa: F401
    except Exception as exc:
        missing.append(f"OdiseModel imports: {exc}")
    return missing
