"""Download Walking Tours (HF URL list) and split into 20s chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .paths import code_root

logger = logging.getLogger(__name__)


def load_wtours_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (code_root() / "configs" / "wtours.yaml")
    return yaml.safe_load(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_wtours(
    *,
    city: str = "amsterdam",
    n_chunks: int = 20,
    chunk_seconds: int = 20,
    max_height: int = 720,
    out_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    cfg = load_wtours_config(config_path)
    cities = {str(k).lower(): v for k, v in (cfg.get("cities") or {}).items()}
    key = city.lower().replace(" ", "_")
    if key not in cities:
        raise KeyError(f"Unknown city {city!r}. Known: {sorted(cities)}")
    url = cities[key]
    n_chunks = int(n_chunks or cfg.get("n_chunks") or 20)
    chunk_seconds = int(chunk_seconds or cfg.get("chunk_seconds") or 20)
    max_height = int(max_height or cfg.get("max_height") or 720)
    total_s = n_chunks * chunk_seconds

    city_dir = Path(out_dir) / key
    city_dir.mkdir(parents=True, exist_ok=True)
    src = city_dir / "_source.mp4"
    if not src.is_file():
        hf_repo = os.environ.get("CORK3DU_HF_WTOURS") or cfg.get("hf_repo")
        hf_name = f"{key}/_source.mp4"
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download

                logger.info("HF %s %s → %s", hf_repo, hf_name, src)
                got = hf_hub_download(
                    repo_id=str(hf_repo),
                    filename=hf_name,
                    repo_type="dataset",
                    local_dir=str(out_dir),
                )
                got_path = Path(got)
                if got_path.resolve() != src.resolve():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    src.write_bytes(got_path.read_bytes())
            except Exception as e:
                logger.warning("HF download failed (%s); falling back to yt-dlp", e)
    if not src.is_file():
        logger.info("yt-dlp %s first %ds ≤%dp → %s", url, total_s, max_height, src)
        fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--proxy",
            "",
            "-f",
            fmt,
            "--download-sections",
            f"*0-{total_s}",
            "--force-keyframes-at-cuts",
            "-o",
            str(src),
            "--no-playlist",
            url,
        ]
        try:
            import yt_dlp  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "yt-dlp is not installed in this Python. On the cluster run:\n"
                f"  {sys.executable} -m pip install --user yt-dlp"
            ) from e
        env = {
            k: v
            for k, v in os.environ.items()
            if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")
        }
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "YouTube is blocked from this node (proxy/firewall). "
                "Download the first 400s on a laptop, copy it to the cluster, then resubmit ingest:\n"
                f"  python -m yt_dlp -f '{fmt}' --download-sections '*0-{total_s}' "
                f"--force-keyframes-at-cuts -o _source.mp4 --no-playlist {url}\n"
                f"  rsync -avP _source.mp4 <cluster>:{src}\n"
                "Ingest skips yt-dlp when that file already exists."
            ) from e
    if not src.is_file():
        raise FileNotFoundError(src)

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_seconds
        dest = city_dir / f"{i:03d}.mp4"
        if not dest.is_file():
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-t", str(chunk_seconds),
                    "-i", str(src),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                    str(dest),
                ],
                check=True,
                capture_output=True,
            )
        chunks.append(
            {
                "index": i,
                "path": str(dest),
                "start_s": start,
                "duration_s": chunk_seconds,
                "sha256": _sha256(dest),
            }
        )
        logger.info("chunk %03d %ss–%ss → %s", i, start, start + chunk_seconds, dest)

    meta = {
        "city": key,
        "source_url": url,
        "license": cfg.get("license", "CC-BY"),
        "dataset": cfg.get("source"),
        "n_chunks": n_chunks,
        "chunk_seconds": chunk_seconds,
        "max_height": max_height,
        "source_mp4": str(src),
        "chunks": chunks,
    }
    (city_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("ingest done: %d chunks → %s", n_chunks, city_dir)
    return meta
