"""Glanceable 4-view PNG + orbit HTML for a scene cloud.npy."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np


def _load_cloud(path: Path, max_points: int) -> np.ndarray:
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Expected (N,3+) cloud, got {cloud.shape} from {path}")
    n = cloud.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        cloud = cloud[idx]
    xyz = cloud[:, :3].astype(np.float64)
    if cloud.shape[1] >= 6:
        rgb = cloud[:, 3:6].astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        z = xyz[:, 2]
        t = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
        rgb = np.stack([t, 0.55 * (1 - t) + 0.2, 1.0 - t], axis=1)
    return np.concatenate([xyz, rgb], axis=1)


def _center_scale(xyz: np.ndarray) -> np.ndarray:
    c = xyz.mean(axis=0)
    x = xyz - c
    scale = np.linalg.norm(x, axis=1).max()
    if scale < 1e-8:
        scale = 1.0
    return x / scale


def write_preview_png(cloud_xyzrgb: np.ndarray, out_path: Path, *, title: str, point_size: float = 0.6) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = _center_scale(cloud_xyzrgb[:, :3])
    rgb = cloud_xyzrgb[:, 3:6]
    views = [
        ("Top (XY)", xyz[:, 0], xyz[:, 1]),
        ("Front (XZ)", xyz[:, 0], xyz[:, 2]),
        ("Side (YZ)", xyz[:, 1], xyz[:, 2]),
        ("Iso", xyz[:, 0] + 0.55 * xyz[:, 1], xyz[:, 2] + 0.35 * xyz[:, 1]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="#111111")
    fig.suptitle(f"{title}\n{cloud_xyzrgb.shape[0]:,} points", color="white", fontsize=14)
    for ax, (name, xs, ys) in zip(axes.ravel(), views):
        ax.set_facecolor("#111111")
        ax.scatter(xs, ys, c=rgb, s=point_size, linewidths=0, rasterized=True)
        ax.set_title(name, color="#dddddd", fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(colors="#666666")
        for spine in ax.spines.values():
            spine.set_color("#333333")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def write_preview_html(
    cloud_xyzrgb: np.ndarray,
    out_path: Path,
    *,
    title: str,
    max_embed: int = 200_000,
    point_size: float = 0.01,
) -> Path:
    pts = cloud_xyzrgb
    if pts.shape[0] > max_embed:
        idx = np.random.default_rng(1).choice(pts.shape[0], size=max_embed, replace=False)
        pts = pts[idx]
    xyz = _center_scale(pts[:, :3]).astype(np.float32)
    rgb = pts[:, 3:6].astype(np.float32)
    packed = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    b64 = base64.b64encode(packed.tobytes()).decode("ascii")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#0b0b0b;color:#eee;font-family:sans-serif}}
#hud{{position:absolute;left:12px;top:10px;z-index:2;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:8px;font-size:13px}}</style>
</head><body>
<div id="hud"><b>{title}</b><br/>{pts.shape[0]:,} points · drag to orbit</div>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
const raw = Uint8Array.from(atob("{b64}"), c => c.charCodeAt(0));
const f32 = new Float32Array(raw.buffer);
const n = f32.length / 6;
const positions = new Float32Array(n * 3);
const colors = new Float32Array(n * 3);
for (let i = 0; i < n; i++) {{
  positions[i*3]=f32[i*6]; positions[i*3+1]=f32[i*6+1]; positions[i*3+2]=f32[i*6+2];
  colors[i*3]=f32[i*6+3]; colors[i*3+1]=f32[i*6+4]; colors[i*3+2]=f32[i*6+5];
}}
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0b0b);
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0.8, 0.5, 1.2);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size:{point_size}, vertexColors:true, sizeAttenuation:true}})));
addEventListener("resize", () => {{ camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); }});
(function animate(){{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }})();
</script></body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def write_scene_previews(
    scene_dir: Path,
    *,
    max_png_points: int = 200_000,
    max_html_points: int = 200_000,
    title: str | None = None,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    cloud_path = scene_dir / "cloud.npy"
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    cloud = _load_cloud(cloud_path, max_points=max(max_png_points, max_html_points))
    rng = np.random.default_rng(0)
    png_cloud = (
        cloud
        if cloud.shape[0] <= max_png_points
        else cloud[rng.choice(cloud.shape[0], size=max_png_points, replace=False)]
    )
    point_size = 0.35 if cloud.shape[0] > 80_000 else 0.8
    html_point_size = 0.008 if cloud.shape[0] > 80_000 else 0.018
    label = title or f"{scene_dir.name} (DA3 + 2s 40/40)"
    png = write_preview_png(png_cloud, scene_dir / "preview.png", title=label, point_size=point_size)
    html_cloud = (
        cloud
        if cloud.shape[0] <= max_html_points
        else cloud[np.random.default_rng(1).choice(cloud.shape[0], size=max_html_points, replace=False)]
    )
    html = write_preview_html(
        html_cloud,
        scene_dir / "preview.html",
        title=label,
        max_embed=max_html_points,
        point_size=html_point_size,
    )
    return {
        "preview_png": str(png),
        "preview_html": str(html),
        "n_points_rendered": int(png_cloud.shape[0]),
        "cloud_path": str(cloud_path),
    }
