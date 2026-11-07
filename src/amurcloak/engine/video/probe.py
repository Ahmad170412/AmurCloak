"""Clip ingest: what is on disk, and the hard limits on it.

Two limits exist, both intentionally blunt:

* **Duration** - clips longer than one minute are refused at upload with a
  message that says to trim first. Not a soft preference: the tracker's
  usefulness and the render time both scale linearly with frames, and a tool
  that quietly accepts a 5-minute phone video would hold the UI hostage for ten
  minutes with no progress story that makes sense.
* **The read loop.** Container metadata lies often enough that the duration
  printed by ``probe`` is treated as advisory. The limit is enforced again while
  actually reading frames, from the decoder's own timestamps, so a corrupt or
  lying container cannot turn a "3 second" file into an unbounded scan. Where
  those timestamps never advance there is no clock to check against at all, so
  the loop also carries a frame ceiling - a backstop that cannot refuse a real
  clip but does bound a file that refuses to say how long it is.

Pixels and frame rate are never re-scaled. Same truthfulness rule as photos:
the exported clip keeps the source's dimensions and rate, and the preview shows
frames as the export will contain them.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2

SUPPORTED_VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

MAX_CLIP_SEC = 60.0
# The read loop's version of the same limit. Slightly wider than the advertised
# one so a source whose timestamps jitter between frames does not fail a file
# the metadata probe had already accepted at 59.9s.
MAX_SCAN_SEC = MAX_CLIP_SEC + 0.99

# The bound the read loop falls back to when the decoder's clock never moves.
# It matters because duration is enforced against that clock: a container that
# reports the same timestamp for every frame (a raw H.264 stream named .mp4 is
# the easy case) makes the check unreachable, and "70 seconds" then reads as
# zero. At 300fps - faster than any camera writes - this is still more frames
# than an accepted clip can hold, so it never refuses a real file, and it stops
# a multi-hour file from being scanned for hours.
MAX_SCAN_FRAMES = int(MAX_SCAN_SEC * 300)

# Largest frame a clip may decode. 8K, because a bgr24 frame at that size is
# already 100MB and both passes hold one at a time: photos can afford a generous
# pixel cap, a clip pays it on every frame of both passes.
MAX_CLIP_PIXELS = 7680 * 4320

# FFmpeg's own banner, silenced so its stderr is only ever our parsing input.
_FFMPEG_QUIET = ("-hide_banner",)


class VideoError(Exception):
    """A clip could not be read, is over the limits, or encoding failed.

    Message is user-facing: the upload form and the render poll both surface it
    verbatim.
    """


@dataclass
class ClipInfo:
    """What a staged clip says about itself, before any frame is decoded."""

    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    has_audio: bool

    def to_dict(self) -> dict:
        return {
            "kind": "video",
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "frames": self.frame_count,
            "duration": round(self.duration, 2),
            "has_audio": self.has_audio,
        }


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_VIDEO


def unique_output_name(source_name: str, taken: set[str]) -> str:
    """A clip's deliverable is always MP4, so its name is decided here.

    Same collision rule as the photo side: ``holiday.mov`` and ``holiday.mp4``
    in one batch must not overwrite each other in the delivered zip.
    """
    base = f"{Path(source_name).stem}.mp4"
    if base not in taken:
        taken.add(base)
        return base
    stem = Path(base).stem
    n = 2
    while f"{stem}-{n}.mp4" in taken:
        n += 1
    taken.add(f"{stem}-{n}.mp4")
    return f"{stem}-{n}.mp4"


def ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary.

    Import is inside the function so a wheel built without video support (or an
    environment pinned to the 0.9 dependency set) fails at the moment clips are
    actually used, never a moment earlier.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception as exc:  # any import shape means "not available"
        raise VideoError(
            "clip support could not find its ffmpeg. "
            "Reinstall AmurCloak - it ships the encoder it needs."
        ) from exc
    try:
        path = get_ffmpeg_exe()
    except Exception as exc:  # same posture: report, do not traceback
        raise VideoError(f"clip support could not start: {exc}") from exc
    if not Path(path).exists():
        raise VideoError("the bundled ffmpeg encoder is missing from this install")
    return path


@cache
def binary_available() -> bool:
    try:
        ffmpeg_exe()
    except VideoError:
        return False
    return True


def open_capture(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"{Path(path).name} could not be opened as a video")
    return cap


def _ffmpeg_says_audio(path: Path) -> bool:
    """True when ffmpeg's stream listing names an audio track.

    OpenCV exposes no audio API at all, so this is the one question only the
    muxer can answer. One short subprocess per uploaded clip; nothing is
    decoded, the container headers are just parsed.
    """
    try:
        result = subprocess.run(  # noqa: S603 - constant argv, no shell, local binary
            [ffmpeg_exe(), *_FFMPEG_QUIET, "-i", str(path)],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    listing = (result.stderr or b"").decode("latin-1", errors="replace")
    return "Audio:" in listing


def probe(path: str | Path) -> ClipInfo:
    """Validate a staged clip and describe it.

    Duration read from container metadata is a *check*, not a promise: frame
    counts in some containers are placeholders (0 or 1), whereupon duration is
    simply unknown at this stage and the read loop's own clock decides later.
    """
    path = Path(path)
    cap = open_capture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise VideoError(f"{path.name} has no readable video track")

    if width * height > MAX_CLIP_PIXELS:
        raise VideoError(
            f"{path.name} is {width}x{height}; clips are limited to 8K "
            f"({MAX_CLIP_PIXELS / 1e6:.0f}MP). Re-encode it smaller first."
        )

    # A container that cannot count its frames has no duration to compare with
    # the limit. A raw stream reports a negative number here, which would read
    # as "zero length" and let a very long file through, so an impossible count
    # is treated as unknown rather than believed.
    if not 0 <= frames <= MAX_SCAN_FRAMES:
        frames = 0

    duration = frames / fps if frames > 0 else 0.0
    if duration > MAX_CLIP_SEC:
        raise VideoError(
            f"{path.name} runs {duration:.0f}s; clips are limited to "
            f"{int(MAX_CLIP_SEC)}s. Trim it first, then drop it in."
        )

    return ClipInfo(
        fps=fps,
        width=width,
        height=height,
        frame_count=frames,
        duration=duration,
        has_audio=_ffmpeg_says_audio(path),
    )


def probe_free_disk(path: Path) -> int:
    return shutil.disk_usage(str(path)).free
