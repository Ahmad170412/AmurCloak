#!/usr/bin/env python
"""Builds the longer synthetic fixtures used for manual clip checks.

A 10s clip with one static red shape, and a 2s clip with none - shapes, not
faces, so these are for eyeballing the pipeline rather than proving detection.
The suites build their own face clips from lena.jpg instead, which is why they
do not need this script to have been run. Runs offline via the bundled ffmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GEN = FIXTURES / "gen"


def build_clip(name: str, duration: int, with_face: bool = True) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / name
    if out.exists():
        return out

    w, h = 640, 360
    img = Path(tempfile.gettempdir()) / "face_source.png"
    if not img.exists():
        import cv2
        import numpy as np

        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
        if with_face:
            cv2.circle(canvas, (w // 2, h // 2), h // 6, (0, 0, 255), -1)  # red face
        cv2.imwrite(str(img), canvas)

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f", "lavfi",
        "-i", f"color=size={w}x{h}:duration={duration}",
        "-i", str(img),
        "-filter_complex", "overlay=x=(W-w)/2:y=(H-h)/2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-t", str(duration),
        str(out),
    ]
    result = subprocess.run(  # noqa: S603 - constant argv, no shell, local binary
        cmd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg build failed for {name}: {result.stderr[-400:]}")
    return out


if __name__ == "__main__":
    build_clip("ten_seconds_face.mp4", 10)
    build_clip("two_seconds_no_face.mp4", 2, with_face=False)
    print("fixtures ready")
