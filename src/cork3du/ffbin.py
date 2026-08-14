"""Resolve an ffmpeg binary: PATH, then imageio-ffmpeg's static build."""

from __future__ import annotations

import shutil


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is not on PATH and imageio-ffmpeg is not installed.\n"
            "  module load ffmpeg\n"
            "  # or, in the project venv:\n"
            "  python -m pip install imageio-ffmpeg"
        ) from e
