"""CLI: ingest-wtours | reconstruct | remask | run | preview."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import paths


def _log() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--frame-width", type=int, default=640)
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument("--pixel-p-thre", type=float, default=0.5)
    p.add_argument("--pixel-frac-thre", type=float, default=0.4)
    p.add_argument("--frame-frac-thre", type=float, default=0.4)


def cmd_ingest(args: argparse.Namespace) -> None:
    from .ingest import ingest_wtours

    paths.ensure_data_dirs()
    out = Path(args.out) if args.out else paths.data_root() / "chunks"
    meta = ingest_wtours(
        city=args.city,
        n_chunks=args.n_chunks,
        chunk_seconds=args.chunk_seconds,
        max_height=args.max_height,
        out_dir=out,
    )
    print(json.dumps(meta, indent=2, default=str))


def cmd_reconstruct(args: argparse.Namespace) -> None:
    from .reconstruct import reconstruct_video

    paths.ensure_data_dirs()
    meta = reconstruct_video(
        Path(args.video),
        Path(args.out),
        da3_streaming_root=paths.da3_root(),
        weights_cache=paths.da3_weights_dir(),
        fps=args.fps,
        frame_width=args.frame_width,
    )
    print(json.dumps(meta, indent=2, default=str))


def cmd_remask(args: argparse.Namespace) -> None:
    from .track_lock import remask_scene

    paths.ensure_data_dirs()
    meta = remask_scene(
        Path(args.scene),
        sam2_root=paths.sam2_root(),
        sam2_checkpoint=paths.sam2_checkpoint(),
        fps=args.fps,
        window_seconds=args.window_seconds,
        pixel_p_thre=args.pixel_p_thre,
        pixel_frac_thre=args.pixel_frac_thre,
        frame_frac_thre=args.frame_frac_thre,
    )
    print(json.dumps({k: meta[k] for k in meta if k != "per_object"}, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> None:
    from .preflight import run_preflight

    run_preflight()
    cmd_reconstruct(args)
    args.scene = args.out
    cmd_remask(args)


def cmd_preview(args: argparse.Namespace) -> None:
    from .preview import write_scene_previews

    info = write_scene_previews(Path(args.scene))
    print(json.dumps(info, indent=2))


def cmd_preflight(args: argparse.Namespace) -> None:
    from .preflight import run_preflight

    run_preflight(require_da3_tree=not args.skip_da3, require_sam2=not args.skip_sam2)
    print("preflight ok")


def main(argv: list[str] | None = None) -> None:
    _log()
    parser = argparse.ArgumentParser(prog="3du", description="DA3 + residual 2s 40/40 static reconstruction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest-wtours", help="Download Walking Tours section and split 20s chunks")
    p.add_argument("--city", default="amsterdam")
    p.add_argument("--n-chunks", type=int, default=20)
    p.add_argument("--chunk-seconds", type=int, default=20)
    p.add_argument("--max-height", type=int, default=720)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("reconstruct", help="ffmpeg frames + DA3-Streaming")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("remask", help="p_resid + SAM2 2s 40/40 + fuse cloud")
    p.add_argument("--scene", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_remask)

    p = sub.add_parser("run", help="reconstruct + remask")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("preview", help="Regenerate preview.png/html from cloud.npy")
    p.add_argument("--scene", required=True)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("preflight", help="Import-check DA3/SAM2/RAFT deps before a GPU job")
    p.add_argument("--skip-da3", action="store_true")
    p.add_argument("--skip-sam2", action="store_true")
    p.set_defaults(func=cmd_preflight)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
