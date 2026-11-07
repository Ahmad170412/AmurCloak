#!/usr/bin/env python
"""AmurCloak video tests.

Run with:  .venv/bin/python tests/test_video.py

Same shape as the other two suites: no framework, a tiny runner, and ``require``
rather than ``assert`` so the checks survive ``python -O``. Every fixture is
built on demand by the ffmpeg that ships inside the wheel, so nothing is
downloaded and nothing outside a temp directory is touched.

What these are actually for: the clip path has three ways to ship a face it
promised to cover - a mask that stops following the face, a mask that is missing
for a frame, and a renderer that ignores the person the user chose to leave
visible. Each has a test that fails without its fix, which matters more here
than in the photo path because a still gets looked at once and a clip does not.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# The package lives under src/, so that is what has to be importable here.
sys.path.insert(0, str(ROOT / "src"))

from amurcloak.app import Config, create_app  # noqa: E402
from amurcloak.engine.detector import Face  # noqa: E402
from amurcloak.engine.presets import build  # noqa: E402
from amurcloak.engine.video import (  # noqa: E402
    probe,
    render as video_render,
    track as video_track,
)
from amurcloak.engine.video.probe import VideoError  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
LENA = FIXTURES / "lena.jpg"
SCRATCH = Path(tempfile.mkdtemp(prefix="amurcloak-video-tests-"))

RESULTS: list[tuple[str, bool, str]] = []
TESTS: list[tuple[str, object]] = []


def check(name: str):
    def wrap(fn):
        TESTS.append((name, fn))
        return fn

    return wrap


def require(condition, message: str = "") -> None:
    if not condition:
        raise AssertionError(message or "condition failed")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def ffmpeg(*args: str) -> None:
    """Run the bundled ffmpeg. The argv is built here, never from user input."""
    command = [probe.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(  # noqa: S603 - constant argv, no shell, local binary
        command, capture_output=True, check=False
    )
    require(
        result.returncode == 0,
        result.stderr.decode("utf-8", "replace")[-400:] or "ffmpeg failed",
    )


def solid_clip(path: Path, seconds: float = 0.5, rate: int = 10) -> Path:
    """A solid white clip: a render can be checked on it with a single pixel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color=size=160x120:duration={seconds}:rate={rate}:color=white",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


def face_clip(path: Path, seconds: float = 1.0, rate: int = 10, audio: bool = False) -> Path:
    """Lena, looped into a clip - a face the detector genuinely finds.

    A synthetic circle is not a face: it exercises the plumbing while leaving
    detection, tracking and coverage unproven. This is the smallest fixture that
    proves those.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-loop", "1", "-i", str(LENA)]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += [
        "-t",
        str(seconds),
        "-r",
        str(rate),
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    ffmpeg(*args, str(path))
    return path


def raw_stream_clip(path: Path, seconds: float = 2.0, rate: int = 10) -> Path:
    """A raw H.264 stream wearing an .mp4 name: no container, therefore no clock.

    This is the shape that slipped past the duration limit: every frame reports
    t=0 and the frame count is negative garbage, so a two-minute file reads as
    zero-length and the read loop never has a timestamp to refuse.
    """
    elementary = path.with_suffix(".h264")
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color=size=160x120:duration={seconds}:rate={rate}:color=white",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(elementary),
    )
    elementary.replace(path)
    return path


def upload_clip(client, path: Path) -> str:
    """Post one clip the way the page does, and hand back the session id."""
    upload = client.post(
        "/api/sessions",
        data={
            "settings": (
                '{"mode":"black","preset":"balanced","margin":14,"strength":70,'
                '"metadata":"strip","keep_audio":true}'
            ),
            "files": [(io.BytesIO(path.read_bytes()), path.name)],
        },
        content_type="multipart/form-data",
    )
    require(upload.status_code == 201, f"upload: {upload.data!r}")
    return upload.get_json()["id"]


def preview_payload(client, sid: str, timeout: float = 90.0) -> dict:
    """Wait out the background scan, then hand back the frame it produced."""
    payload: dict = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/sessions/{sid}/preview?index=0&view=result&t=0").get_json()
        if not payload.get("scanning"):
            return payload
        time.sleep(0.2)
    raise AssertionError(f"the scan never finished: {payload}")


def preview_pixels(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["image"])
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    require(frame is not None, "the preview was not a decodable image")
    return frame


def mask_point(payload: dict, face: dict, frame: np.ndarray) -> tuple[float, float]:
    """A mask centre in the pixels of the preview, whatever scale it came back at."""
    scale = frame.shape[1] / payload["width"] if payload.get("width") else 1.0
    return face["mask"]["cx"] * scale, face["mask"]["cy"] * scale


def frames_of(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def render_clip(source: Path, destination: Path, masks, settings=None, restored=()):
    """Render with the fixture's own geometry, then hand back decoded frames."""
    info = probe.probe(source)
    video_render.render_to_disk(
        source,
        destination,
        masks,
        settings or build("balanced", mode="black"),
        fps=info.fps,
        width=info.width,
        height=info.height,
        restored_tracks=frozenset(restored),
    )
    return frames_of(destination)


def patch_mean(frame: np.ndarray, cx: float, cy: float, radius: int = 5) -> float:
    x, y = int(cx), int(cy)
    crop = frame[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius]
    return float(crop.mean()) if crop.size else 0.0


def track_ids(masks) -> int:
    return video_track.count_tracks(masks)


def union_box(row) -> tuple[float, float, float, float]:
    return (
        min(m.x for m in row),
        min(m.y for m in row),
        max(m.x + m.w for m in row),
        max(m.y + m.h for m in row),
    )


# ---------------------------------------------------------------------------
# probe and limits
# ---------------------------------------------------------------------------


@check("a clip probe reports a finite, in-limit geometry")
def test_clip_probe():
    info = probe.probe(face_clip(SCRATCH / "probe.mp4"))
    require(info.frame_count > 0, "the fixture clip has no frames")
    require(info.fps > 0 and info.width > 0 and info.height > 0, "incomplete geometry")
    require(info.duration <= probe.MAX_CLIP_SEC, "the fixture is over the limit")


@check("a clip over the limit is refused, and the message says what to do")
def test_clip_over_limit_refused():
    long_clip = solid_clip(SCRATCH / "long.mp4", seconds=probe.MAX_CLIP_SEC + 2, rate=2)
    try:
        probe.probe(long_clip)
    except VideoError as exc:
        require("trim" in str(exc).lower(), f"unhelpful refusal: {exc}")
    else:
        raise AssertionError("an over-length clip was accepted")


@check("a clip with a dead clock is stopped by its frame count")
def test_no_clock_is_stopped_by_frame_count():
    source = raw_stream_clip(SCRATCH / "raw.mp4", seconds=2.0, rate=10)
    info = probe.probe(source)
    require(info.frame_count == 0, f"a raw stream claimed {info.frame_count} frames")
    require(info.duration == 0.0, f"a raw stream claimed to run {info.duration}s")

    # The real ceiling is thousands of frames; shrink it so the guard is
    # reachable with a fixture that encodes in a tenth of a second.
    original = probe.MAX_SCAN_FRAMES
    probe.MAX_SCAN_FRAMES = 8
    try:
        video_render.scan_frames(source, video_render.clip_detector(build("balanced")))
    except VideoError as exc:
        require("timestamps" in str(exc), f"refused for the wrong reason: {exc}")
    else:
        raise AssertionError("a 20-frame clip with a dead clock was scanned to the end")
    finally:
        probe.MAX_SCAN_FRAMES = original


@check("a frame bigger than 8K is refused before anything decodes it")
def test_eight_k_is_refused():
    source = SCRATCH / "eightk.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=8192x4320:duration=0.2:rate=5:color=gray",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(source),
    )
    try:
        probe.probe(source)
    except VideoError as exc:
        require("8K" in str(exc), f"unhelpful refusal: {exc}")
    else:
        raise AssertionError("an 8192x4320 frame was accepted for redaction")


# ---------------------------------------------------------------------------
# detection settings
# ---------------------------------------------------------------------------


@check("the clip detector follows the preset's tiling and the per-view cap")
def test_clip_detector_follows_preset():
    balanced = video_render.clip_detector(build("balanced"))
    cautious = video_render.clip_detector(build("conservative"))
    require(balanced.tiles is True, "Balanced must tile clips exactly as it tiles photos")
    require(cautious.tiles is False, "Cautious is the single-pass preset")
    require(
        balanced.detect_edge <= video_render.CLIP_DETECT_EDGE_CAP,
        "the per-view inference cap was ignored",
    )


@check("clip audio is kept by default and the off switch works")
def test_audio_setting():
    require(build("balanced").keep_audio is True, "audio should default to kept")
    require(build("balanced", keep_audio=False).keep_audio is False, "the off switch is dead")


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------


@check("one face across two frames keeps one identity")
def test_tracker_identity_count():
    face = Face(x=200.0, y=100.0, w=80.0, h=100.0, score=0.9, landmarks=None)
    masks = video_track.build_masks([[face], [face]])
    require(len(masks) == 2, "one mask list per frame")
    require(len(masks[0]) == 1 and len(masks[1]) == 1, "one mask per face per frame")
    require(masks[0][0].track == masks[1][0].track == 1, "identity did not persist")


@check("a face walking across the frame is covered on every frame")
def test_moving_face_is_covered_on_every_frame():
    start_x, step, face_w, face_h = 40.0, 8.0, 90.0, 110.0
    frames = [
        [
            Face(
                x=start_x + i * step,
                y=200.0,
                w=face_w,
                h=face_h,
                score=0.9,
                landmarks=None,
            )
        ]
        for i in range(90)
    ]

    masks = video_track.build_masks(frames)
    require(len(masks) == len(frames), "the timeline lost frames")

    boxes = set()
    for i, row in enumerate(masks):
        require(len(row) == 1, f"frame {i}: expected one mask, got {len(row)}")
        mask = row[0]
        require(mask.track == 1, f"frame {i}: identity changed to {mask.track}")
        boxes.add((round(mask.x, 1), round(mask.w, 1)))

        face_x = start_x + i * step
        overlap = min(mask.x + mask.w, face_x + face_w) - max(mask.x, face_x)
        require(overlap > 0, f"frame {i}: the mask missed the face entirely")
        # The box trails by the position smoothing's steady-state lag (about
        # 10px at this speed), so this is a strong floor rather than full
        # containment; the mask expands past the face at render time.
        require(
            overlap / face_w >= 0.75,
            f"frame {i}: only {overlap / face_w:.0%} of the face under the mask",
        )

    # "A mask exists on every frame" is satisfied by one frozen box. These two
    # are what pin the cover to the face.
    require(len(boxes) > 10, f"the mask never moved: {len(boxes)} distinct boxes")
    require(masks[0][0].x < masks[-1][0].x, "the mask did not follow the face's direction")


@check("a face that keeps moving during a detector gap stays covered")
def test_grace_follows_motion():
    """Detections stop and never resume, so grace is the only cover there is.

    A face walking at a steady rate is masked for as long as the detector sees
    it, then the detector loses it for ten frames. The cover has to widen along
    the motion, because a frozen box is left behind at 8px a frame - and that is
    exactly the frame an attacker would step to.
    """
    start_x, step, face_w, face_h = 60.0, 8.0, 90.0, 110.0

    def det(i: int) -> Face:
        return Face(
            x=start_x + i * step,
            y=200.0,
            w=face_w,
            h=face_h,
            score=0.9,
            landmarks=None,
        )

    frames = [[det(i)] for i in range(20)] + [[] for _ in range(10)]
    masks = video_track.build_masks(frames)
    require(len(masks) == 30, "the timeline lost frames")
    require(track_ids(masks) == 1, "an unfinished gap must not invent a second identity")

    frozen = masks[19][0]
    frozen_x0, _, frozen_x1, _ = union_box([frozen])
    require(masks[20][0].w > frozen.w, "the cover did not widen along the motion")

    for frame in range(20, 30):
        row = masks[frame]
        require(row, f"frame {frame}: the gap lost its cover entirely")
        x0, _, x1, _ = union_box(row)
        # The cover may only ever grow: the last known box is always inside it.
        require(x0 <= frozen_x0 + 0.5, f"frame {frame}: the frozen box was abandoned")
        require(x1 >= frozen_x1 - 0.5, f"frame {frame}: the cover shrank")

        face_x = start_x + frame * step  # where the face keeps going
        overlap = min(x1, face_x + face_w) - max(x0, face_x)
        require(
            overlap / face_w >= 0.75,
            f"frame {frame}: only {overlap / face_w:.0%} of the moving face covered",
        )


# ---------------------------------------------------------------------------
# render: what the delivered clip actually contains
# ---------------------------------------------------------------------------


@check("a person left visible in the preview is visible in the export")
def test_restored_track_is_left_uncovered_in_the_export():
    source = solid_clip(SCRATCH / "solid.mp4")
    originals = frames_of(source)
    require(originals, "the fixture clip holds no frames")

    box = video_track.Mask(x=40.0, y=30.0, w=40.0, h=60.0, roll=0.0, score=0.9, track=7)
    masks = [[box] for _ in originals]

    covered = render_clip(source, SCRATCH / "covered.mp4", masks)[0]
    require(patch_mean(covered, 60, 60) < 30, "the mask centre was not black")
    require(patch_mean(covered, 8, 8) > 200, "the rest of the frame was altered")

    spared = render_clip(source, SCRATCH / "spared.mp4", masks, restored={7})[0]
    require(patch_mean(spared, 60, 60) > 200, "the spared track was still covered")

    other = render_clip(source, SCRATCH / "other.mp4", masks, restored={8})[0]
    require(patch_mean(other, 60, 60) < 30, "restoring one track spared a different one")


@check("a frame with no recorded mask is refused, not written uncovered")
def test_render_refuses_frames_the_scan_never_saw():
    source = solid_clip(SCRATCH / "short.mp4")
    originals = frames_of(source)
    require(len(originals) >= 4, "need at least four frames for this test")

    box = video_track.Mask(x=40.0, y=30.0, w=40.0, h=60.0, roll=0.0, score=0.9, track=7)
    masks = [[box] for _ in range(len(originals) - 2)]  # the scan saw two fewer
    out = SCRATCH / "mismatch.mp4"

    try:
        render_clip(source, out, masks)
    except VideoError as exc:
        require("more frames" in str(exc), f"wrong refusal: {exc}")
    else:
        raise AssertionError("a short scan was rendered instead of refused")

    require(not out.exists(), "a partial deliverable was left behind")


@check("a detected face comes out black, frame after frame")
def test_export_covers_a_real_face():
    source = face_clip(SCRATCH / "lena.mp4")
    detector = video_render.clip_detector(build("balanced"))
    detections, _rate = video_render.scan_frames(source, detector)
    masks = video_track.build_masks(detections)
    require(any(masks), "the detector found no face in the fixture clip")

    before = frames_of(source)
    after = render_clip(source, SCRATCH / "lena-redacted.mp4", masks)
    require(len(after) == len(before), "the export changed the frame count")

    checked = 0
    for i, (original, redacted) in enumerate(zip(before, after, strict=False)):
        if not masks[i]:
            continue
        for mask in masks[i]:
            cx, cy = mask.x + mask.w / 2.0, mask.y + mask.h / 2.0
            require(
                patch_mean(redacted, cx, cy) < 40,
                f"frame {i}: the face centre is not black ({patch_mean(redacted, cx, cy):.0f})",
            )
            require(
                patch_mean(original, cx, cy) > 40,
                f"frame {i}: nothing to redact at that point - the check is meaningless",
            )
            checked += 1

    require(checked >= 5, f"only {checked} faces were checked")


@check("clip audio survives the export, and the off switch removes it")
def test_clip_audio_kept_and_dropped():
    source = face_clip(SCRATCH / "lena-audio.mp4", audio=True)
    info = probe.probe(source)
    require(info.has_audio, "the fixture clip was built without an audio track")

    masks = [[] for _ in frames_of(source)]
    render_clip(source, SCRATCH / "audio-kept.mp4", masks)
    require(probe.probe(SCRATCH / "audio-kept.mp4").has_audio, "the audio was dropped")

    dropped = render_clip(
        source, SCRATCH / "audio-dropped.mp4", masks, build("balanced", keep_audio=False)
    )
    require(dropped, "the no-audio render produced nothing")
    require(not probe.probe(SCRATCH / "audio-dropped.mp4").has_audio, "the off switch leaked sound")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


@check("the scan measures the source's real frame rate")
def test_scan_measures_the_rate():
    source = solid_clip(SCRATCH / "rate.mp4", seconds=1.0, rate=8)
    detector = video_render.clip_detector(build("balanced"))
    _detections, rate = video_render.scan_frames(source, detector)
    require(abs(rate - 8.0) < 0.5, f"measured {rate:.2f} fps, expected 8")


# ---------------------------------------------------------------------------
# integration: the whole route, end to end
# ---------------------------------------------------------------------------


@check("clip scans share two slots instead of one thread each")
def test_scan_concurrency_is_bounded():
    from amurcloak.engine import sessions as sessions_module

    root = SCRATCH / "conc" / "session"
    root.mkdir(parents=True, exist_ok=True)
    session = sessions_module.Session(root, build("balanced"))
    source = solid_clip(SCRATCH / "conc-source.mp4", seconds=1.0, rate=10)

    class SlowDetector:
        """Stands in for YuNet: the point here is how many run at once."""

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.live = 0
            self.peak = 0

        def detect(self, _frame):
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            time.sleep(0.02)
            with self.lock:
                self.live -= 1
            return []

    detector = SlowDetector()
    items = []
    for index in range(6):
        stored = f"{index:04d}.mp4"
        shutil.copyfile(source, session.in_dir / stored)
        item = sessions_module.Item(name=f"clip-{index}.mp4", stored=stored, kind="video")
        item.meta = {"frames": 10}
        session.items.append(item)
        items.append(item)

    for item in items:
        session.request_clip_scan(item, detector)
    for item in items:
        session.wait_for_clip_scan(item)

    require(detector.peak >= 1, "no scan ran at all")
    require(
        detector.peak <= sessions_module.MAX_CONCURRENT_SCANS,
        f"{detector.peak} scans ran at once, the cap is {sessions_module.MAX_CONCURRENT_SCANS}",
    )


@check("a clip uploads, previews, renders and downloads")
def test_clip_upload_and_render_roundtrip():
    clip = face_clip(SCRATCH / "roundtrip.mp4", seconds=2.0)
    app = create_app(Config(tmp_root=SCRATCH / "app", ttl=3600, max_age=7200))
    client = app.test_client()
    clip_bytes = clip.read_bytes()

    upload = client.post(
        "/api/sessions",
        data={
            "settings": (
                '{"mode":"black","preset":"balanced","margin":14,"strength":70,'
                '"metadata":"strip","keep_audio":true}'
            ),
            "files": [(io.BytesIO(clip_bytes), clip.name)],
        },
        content_type="multipart/form-data",
    )
    require(upload.status_code == 201, f"upload: {upload.data!r}")
    sid = upload.get_json()["id"]

    # Preview: wait out the background scan rather than guessing a duration.
    payload: dict = {}
    deadline = time.time() + 90
    while time.time() < deadline:
        payload = client.get(f"/api/sessions/{sid}/preview?index=0&view=result&t=0").get_json()
        if not payload.get("scanning"):
            break
        time.sleep(0.25)
    require(payload.get("image"), f"no preview image: {payload}")

    render = client.post(f"/api/sessions/{sid}/render")
    require(render.status_code == 202, f"render: {render.data!r}")

    snapshot: dict = {}
    deadline = time.time() + 120
    while time.time() < deadline:
        snapshot = client.get(f"/api/sessions/{sid}").get_json()
        if snapshot["state"] == "ready":
            break
        time.sleep(0.25)
    require(
        snapshot.get("state") == "ready",
        f"render never finished: state={snapshot.get('state')}",
    )

    result = client.get(f"/api/sessions/{sid}/result")
    require(result.status_code == 200, f"download: {result.data!r}")

    out = SCRATCH / "downloaded.mp4"
    out.write_bytes(result.data)
    info = probe.probe(out)
    source_info = probe.probe(clip)
    require(info.duration > 0 and info.width > 0 and info.height > 0, "the download is not a clip")
    require(frames_of(out), "the downloaded clip would not decode")
    # The render encodes at the rate the scan measured, so the delivered clip
    # must still run as long as the source did - a wrong rate shows up here.
    require(
        abs(info.duration - source_info.duration) <= max(0.3, 0.15 * source_info.duration),
        f"the export runs {info.duration:.1f}s against a {source_info.duration:.1f}s source",
    )

    app.dispose()


@check("leaving a person visible reaches the preview, and a re-scan forgets them")
def test_leave_visible_reaches_the_preview():
    clip = face_clip(SCRATCH / "leave-visible.mp4", seconds=1.0)
    app = create_app(Config(tmp_root=SCRATCH / "app-leave", ttl=3600, max_age=7200))
    client = app.test_client()
    sid = upload_clip(client, clip)

    payload = preview_payload(client, sid)
    require(payload.get("faces"), f"nothing to leave visible: {payload.get('note')}")
    face = payload["faces"][0]
    # A clip's unit of choice is the tracked person, so the key has to name one.
    require(face["key"][:1] == ["t"], f"a clip handed back the key {face['key']}")
    require(face["covered"] is True, "a clip started with a face uncovered")

    frame = preview_pixels(payload)
    covered_at = mask_point(payload, face, frame)
    require(patch_mean(frame, *covered_at) < 40, "the face is not covered before the click")

    toggle = client.post(
        f"/api/sessions/{sid}/faces",
        json={"index": 0, "key": face["key"], "covered": False},
    )
    require(toggle.status_code == 200, f"leaving the person visible failed: {toggle.data!r}")
    require(toggle.get_json()["items"][0]["restored"] == 1, "the choice was not recorded")

    payload = preview_payload(client, sid)
    require(payload["restored"] == 1, "the preview still reports the person covered")
    frame = preview_pixels(payload)
    visible_at = mask_point(payload, payload["faces"][0], frame)
    require(
        patch_mean(frame, *visible_at) > 40,
        f"the preview still paints over the person ({patch_mean(frame, *visible_at):.0f})",
    )

    # Track ids are positional and handed out again by any new scan, so a choice
    # kept across one would silently name whoever is track 1 next time. The
    # payload flips tiling because that is the detection knob a request can
    # actually move.
    settings = client.post(
        f"/api/sessions/{sid}/settings", json={"preset": "balanced", "tiles": False}
    )
    require(settings.status_code == 200, f"settings: {settings.data!r}")
    require(settings.get_json()["detection_changed"] is True, "nothing detection-related moved")
    snapshot = client.get(f"/api/sessions/{sid}").get_json()
    require(
        snapshot["items"][0]["restored"] == 0,
        "a clip kept a track id across a re-scan, which can name another person",
    )

    app.dispose()


# ---------------------------------------------------------------------------


def main() -> int:
    print("AmurCloak video tests\n" + "-" * 46)
    try:
        for name, fn in TESTS:
            try:
                fn()
                RESULTS.append((name, True, ""))
            except Exception as exc:  # noqa: BLE001 - a failed check is the report
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc()

        for name, ok, detail in RESULTS:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print(f"      {detail}")
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print("-" * 46)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
