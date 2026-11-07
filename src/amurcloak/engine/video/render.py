"""The two passes that turn a staged clip into a redacted one.

**Pass A - scan.** Every frame is decoded once, run through YuNet, and
discarded; only the detections are kept. Reading stops at a hard cap on the
decoder's own clock, so a container that lies about its duration cannot stretch
the scan.

**Between passes.** :mod:`.track` builds the per-frame mask list. All of the
clip-specific progress is concentrated there and it is kilobytes, which is the
whole reason the pipeline stays streaming: masks are looked up by frame index,
never recomputed.

**Pass B - render.** The file is streamed again; each frame gets the boxes the
tracker left for it, minus any person the user chose to leave visible, the
*unmodified* photo-mode redaction is applied (the ellipse geometry, the three
cover modes, the antialiased rim - all identical to stills, deliberately), and
the frame is piped straight into ffmpeg's stdin,
which encodes libx264 into a silent intermediate. A cheap second command then
builds the deliverable: that intermediate plus the source's own audio track,
copied straight through when the codec allows and re-encoded to AAC when it
does not. Keeping the two steps apart is deliberate: an exotic audio codec
would otherwise fail a finished three-minute encode and force a full re-render
for what is a seconds-long remux.

Audio: kept by default and copied losslessly, with an explicit off switch.
That default is honest - most people sharing a redacted clip need the sound -
and the choice is visible, because a voice is as identifying as a face and the
tool should say so rather than decide silently.

Output container: MP4. Metadata is dropped wholesale (``-map_metadata -1``);
the container is rebuilt from scratch, so no stray source tag rides along.
"""

from __future__ import annotations

import contextlib
import math
import subprocess
from pathlib import Path

import cv2

from .. import redact
from ..detector import Face, FaceDetector
from ..presets import Settings
from . import probe, track as tracker
from .probe import VideoError

# Inference edge per view. Lower than the photo cap because a clip pays it on
# every frame; tiling is what buys back native resolution, not this number.
CLIP_DETECT_EDGE_CAP = 1024


def clip_detector(settings: Settings) -> FaceDetector:
    """The per-frame detector pass A uses.

    Tiling follows the preset, exactly as it does for stills. It used to be off
    for clips on a cost argument that no longer holds: a single 1024px pass sees
    a 4K frame at a quarter scale, so anything smaller than roughly a hundred
    pixels in the source was invisible - not "hard to find", absent. The price
    is real and measured: a 1080p frame costs about five views and 25-57ms, a
    4K frame seventeen views and ~200ms, so a full minute of 4K takes minutes to
    scan. That is why clips are capped at 60s and the scan is a background
    thread with a progress fraction rather than a blocking request.
    """
    return FaceDetector(
        score=settings.score,
        nms=0.3,
        min_face=settings.min_face,
        detect_edge=min(settings.detect_edge, CLIP_DETECT_EDGE_CAP),
        tiles=settings.tiles,
    )


def scan_frames(
    path: Path, detector: FaceDetector, on_progress=None, total_frames: int = 0
) -> tuple[list[list[Face]], float]:
    """Pass A: detections per frame, streaming. Enforces the duration cap.

    ``on_progress`` receives a 0..1 fraction. The denominator is the probe's
    frame count; without one the callback reports nothing rather than a raw
    frame count, which would be misread as a fraction by every caller.

    The duration cap is measured on the decode clock, and that clock is only as
    good as the container: see ``probe.MAX_SCAN_FRAMES`` for the backstop that
    keeps a file with no advancing timestamps from being read to the end.

    Returns the detections and the frame rate measured from the decoder's own
    timestamps (0.0 when it cannot be measured). Pass B has to encode at *some*
    rate, and the container's stated one is a claim: a variable-rate or
    mislabelled file makes frames/rate disagree with the real duration, so the
    muxed audio drifts away from the picture. The decode clock is the same
    clock pass B reads, which is why measuring it here settles both.
    """
    cap = probe.open_capture(path)
    frames: list[list[Face]] = []
    first_ms: float | None = None
    last_ms: float | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(ms) and ms >= 0:
                if first_ms is None:
                    first_ms = ms
                last_ms = ms
            if ms / 1000 > probe.MAX_SCAN_SEC:
                raise VideoError(
                    f"{path.name} runs past the {int(probe.MAX_CLIP_SEC)}s clip limit "
                    "- trim it first, then drop it in."
                )
            if len(frames) >= probe.MAX_SCAN_FRAMES:
                # The clock above never moved, so the only thing still bounding
                # this file is how many frames it has. Refuse rather than keep
                # decoding something that will not say how long it is.
                raise VideoError(
                    f"{path.name} holds more frames than a "
                    f"{int(probe.MAX_CLIP_SEC)}s clip can, and its timestamps never "
                    "advance, so its length cannot be checked. Re-encode it, then "
                    "drop it in."
                )
            frames.append(detector.detect(frame))
            if on_progress is not None:
                on_progress(min(1.0, len(frames) / total_frames) if total_frames else 0.0)
    finally:
        cap.release()

    measured = 0.0
    if first_ms is not None and last_ms is not None and last_ms > first_ms:
        measured = (len(frames) - 1) / ((last_ms - first_ms) / 1000.0)
    # A rate outside anything a camera records is a timestamp artefact, not a
    # frame rate; the caller falls back to the probe's value.
    if not 1.0 < measured < 1000.0:
        measured = 0.0
    return frames, measured


def render_to_disk(
    source: Path,
    destination: Path,
    masks: list[list[tracker.Mask]],
    settings: Settings,
    *,
    fps: float,
    width: int,
    height: int,
    on_progress=None,
    restored_tracks: frozenset[int] = frozenset(),
) -> None:
    """Pass B: stream frames through redaction, then mux the deliverable.

    ``restored_tracks`` names the tracked people the user chose to leave
    visible. They are dropped here, at the same boundary the preview uses, so
    the delivered clip covers exactly what the map on screen said it would.
    """
    silence = destination.with_name(destination.name + ".silent.mp4")
    cap = probe.open_capture(source)
    try:
        _encode_muted(
            cap,
            silence,
            masks,
            settings,
            fps=fps,
            width=width,
            height=height,
            on_progress=on_progress,
            restored_tracks=restored_tracks,
        )
    finally:
        cap.release()

    deliver(destination, silence, source, settings.keep_audio)


def _encode_muted(
    cap: cv2.VideoCapture,
    silence: Path,
    masks: list[list[tracker.Mask]],
    settings: Settings,
    *,
    fps: float,
    width: int,
    height: int,
    on_progress=None,
    restored_tracks: frozenset[int] = frozenset(),
) -> None:
    """The frame stream, piped into one ffmpeg invocation.

    Raw BGR goes in (what OpenCV hands back, so no colour translation
    anywhere) and H.264 comes out. The audio is not part of this pass at all:
    the mux step adds it, and an exotic audio codec failing there must not
    strand a finished encode.

    This pass fails closed. Every frame written must have a mask list recorded
    for its index by the scan; a frame the scan never saw is refused rather
    than encoded with nothing over it. An empty mask list is a normal result -
    it means no faces were found - but a *missing* entry means the two passes
    disagree about the file, and that is not something to ship.
    """
    if not masks:
        raise VideoError("this clip has no scan to render from")
    exe = probe.ffmpeg_exe()
    command = [
        exe,
        "-hide_banner",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.3f}",
        "-i", "-",
        "-map", "0:v",
        "-map_metadata", "-1",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", str(silence),
    ]
    process = subprocess.Popen(  # noqa: S603 - constant argv, no shell, local binary
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    total = len(masks)
    frames = 0
    overrun = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames >= total:
                # The scan ran out of frames while the decoder had more. Stop
                # before writing a frame that has no mask behind it.
                overrun = True
                break
            present = [m for m in masks[frames] if m.track not in restored_tracks]
            if present:
                redact.apply_redaction(
                    frame,
                    [m.to_face() for m in present],
                    settings.mode,
                    settings.margin,
                    settings.strength,
                )
            _push(process, frame)
            frames += 1
            if on_progress is not None:
                # Pass B owns a bit over half the bar; the scan holds the rest.
                on_progress(min(1.0, frames / total))
    finally:
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        code = process.wait(timeout=60)
        tail = _stderr_tail(process)
    if overrun:
        with contextlib.suppress(OSError):
            silence.unlink(missing_ok=True)
        raise VideoError(
            "the clip handed the renderer more frames than the scan recorded. "
            "Nothing was written rather than risk a frame with no cover on it."
        )
    if frames == 0:
        with contextlib.suppress(OSError):
            silence.unlink(missing_ok=True)
        raise VideoError("no frames could be read from the clip")
    if code != 0:
        with contextlib.suppress(OSError):
            silence.unlink(missing_ok=True)
        detail = f": {tail}" if tail else ""
        raise VideoError(f"the encode failed{detail}")


def _push(process: subprocess.Popen, frame) -> None:
    stdin = process.stdin
    if stdin is None:  # pragma: no cover - Popen is built with stdin=PIPE
        raise VideoError("the encoder's input pipe was never opened")
    try:
        stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        detail = _stderr_tail(process)
        raise VideoError(
            f"the encoder stopped accepting frames{f': {detail}' if detail else ''}"
        ) from exc


def _stderr_tail(process: subprocess.Popen, limit: int = 400) -> str:
    if process.stderr is None:  # pragma: no cover - Popen sets the pipe
        return ""
    try:
        text = (process.stderr.read() or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - diagnostics must never mask the real error
        return ""
    return text.strip()[-limit:]


def deliver(
    destination: Path,
    silence: Path,
    source: Path,
    keep_audio: bool,
) -> None:
    """Mux the finished video track with the source's audio (or none).

    Three attempts, cheapest first: a stock ffmpeg can usually copy the audio
    bitstream as-is; where the source codec does not fit an MP4 (PCM in an AVI,
    say), the fallback re-encodes instead; and with audio switched off none of
    it happens. Video is never re-encoded here - seconds, not minutes.
    """
    # Order matters: every input comes first, then every output option, then the
    # output path as the last positional token. ffmpeg treats options placed
    # before an ``-i`` as input options, so an output option such as ``-map``
    # wedged between two inputs makes the second input fail to parse.
    inputs = [probe.ffmpeg_exe(), "-hide_banner", "-i", str(silence)]
    with_audio = [*inputs, "-i", str(source)]
    tail = ["-map", "0:v", "-map", "1:a?", "-map_metadata", "-1", "-c:v", "copy",
            "-movflags", "+faststart"]
    audio_copy = [*with_audio, *tail, "-c:a", "copy", "-y", str(destination)]
    audio_aac = [*with_audio, *tail, "-c:a", "aac", "-b:a", "160k", "-y", str(destination)]
    no_audio = [
        *inputs, "-map", "0:v", "-map_metadata", "-1", "-c:v", "copy",
        "-movflags", "+faststart", "-an", "-y", str(destination),
    ]

    failures: list[str] = []
    attempts = [audio_copy, audio_aac, no_audio] if keep_audio else [no_audio]
    for attempt in attempts:
        result = subprocess.run(  # noqa: S603 - constant argv, no shell, local binary
            attempt, capture_output=True, timeout=120, check=False
        )
        if result.returncode == 0:
            return  # destination was overwritten by this attempt
        errors = (result.stderr or b"").decode("utf-8", errors="replace").strip()[-400:]
        failures.append(errors or "ffmpeg failed")

    with contextlib.suppress(OSError):
        silence.unlink(missing_ok=True)
    detail = f": {failures[-1]}" if failures else ""
    raise VideoError(f"the redacted clip could not be assembled{detail}")
