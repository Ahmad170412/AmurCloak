"""Session staging, rendering, and temp-file lifecycle.

Disk is the scarce resource in this project, so the design leans on it directly:

* Everything transient lives under one root (a per-user folder inside the
  system temp directory) that a single reaper owns, so a crash or a hard kill
  cannot leave debris scattered across the filesystem.
* A batch is streamed into a zip: each rendered file is appended and unlinked
  immediately, so the peak footprint is ``inputs + zip + one file`` rather than
  ``inputs + outputs + zip``.
* Inputs are freed automatically once the result has been downloaded, which is
  the natural end of the flow.
* A background reaper deletes idle sessions, and startup purges orphan
  directories left behind by a crash.

Rendering runs on a single worker thread. That is deliberate: it keeps RAM
predictable (one 24MP image in flight at a time, ~72MB) and there is no
throughput to win by running two, since detection already saturates a core.
"""

from __future__ import annotations

import contextlib
import queue
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import photo as photo_io, redact
from .detector import Face, FaceDetector
from .presets import Settings, build as build_settings
from .video import probe as video_probe, render as video_render, track as video_track
from .video.probe import VideoError

# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------

# The temp root must never resolve inside the package: installed, this file
# lives in site-packages, which is not ours to write to. The system temp
# directory is per-user on both macOS and Windows and is cleaned up by the OS.
DEFAULT_TMP = Path(tempfile.gettempdir()) / "amurcloak"

MAX_FILES = 300
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_SESSION_BYTES = 1_500_000_000  # total accepted per upload (1.5GB)
SPACE_MULTIPLIER = 3  # transient copies during an upload
SPACE_HEADROOM = 512 * 1024 * 1024

SESSION_TTL_SEC = 30 * 60
SESSION_MAX_AGE_SEC = 6 * 3600
REAP_INTERVAL_SEC = 60

PREVIEW_MAX_EDGE = 1400
PREVIEW_CACHE_PIXELS = 40_000_000  # don't cache monsters in RAM
PHOTO_CACHE_ENTRIES = 2


class StorageError(Exception):
    """Not enough room, or a limit would be exceeded. Message is user-facing."""


@dataclass
class ClipScan:
    """Everything a clip has produced so far, minus its pixels.

    A clip's scan runs in the background the first time it is previewed or
    rendered: at 25ms a frame it can take a minute of a long clip, and neither
    the preview request that started it nor anyone else should sit on it. So
    the entry carries its own state machine and a completion event the render
    worker can wait on. What is *kept* is only detections and masks - lookups
    for rendering are index reads, never rework.
    """

    state: str = "none"  # none | running | done | error
    progress: float = 0.0
    error: str | None = None
    # Frame rate measured from the decode clock, or 0.0 when it was not
    # measurable. The render encodes at this rather than the container's claim.
    fps: float = 0.0
    # Per-frame detections from pass A, in frame order.
    frames: list[list[Face]] = field(default_factory=list)
    # Tracker output: per-frame masks, same indexing; what render and preview read.
    masks: list[list[video_track.Mask]] = field(default_factory=list)


def free_bytes(path: str | Path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0


def space_needed(incoming_bytes: int) -> int:
    return incoming_bytes * SPACE_MULTIPLIER + SPACE_HEADROOM


def ensure_space(paths: list[Path], incoming_bytes: int) -> None:
    """Refuse a job that would not fit, *before* accepting the bytes.

    Checked against every candidate location because an upload briefly exists in
    two places at once (the request spool and our own staging directory), and
    they can be on different mounts.
    """
    needed = space_needed(incoming_bytes)
    for path in paths:
        if not path.exists():
            continue
        available = free_bytes(path)
        if available and available < needed:
            raise StorageError(
                f"needs about {needed / 1e9:.1f}GB free to accept "
                f"{incoming_bytes / 1e9:.1f}GB safely, but "
                f"{path} has {available / 1e9:.1f}GB. "
                "Clear some space or delete old sessions."
            )


def _shred_file(path: Path) -> None:
    """Best-effort overwrite before unlink. Off by default: it costs a full write pass."""
    with contextlib.suppress(OSError):
        size = path.stat().st_size
        if size == 0:
            path.unlink()
            return
        with path.open("r+b", buffering=0) as handle:
            remaining = size
            chunk = b"\x00" * (1024 * 1024)
            while remaining > 0:
                n = min(remaining, len(chunk))
                handle.write(chunk[:n])
                remaining -= n
    # Unlink even if the overwrite failed: leaving the plaintext behind would
    # defeat the point of having asked for a shred.
    with contextlib.suppress(OSError):
        path.unlink()


def purge_tree(root: Path, shred: bool = False) -> None:
    if not root.exists():
        return
    if shred:
        for path in root.rglob("*"):
            if path.is_file():
                _shred_file(path)
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# session model
# ---------------------------------------------------------------------------


@dataclass
class Item:
    name: str  # original filename, as the user knows it
    stored: str  # filename inside the staging dir
    size: int = 0
    status: str = "staged"  # staged | done | error
    kind: str = "photo"  # photo | video
    faces: int = 0
    scanned: bool = False  # whether detection has run on this file yet
    scan_state: str = "idle"  # idle | running | done | error (clips run in the background)
    scan_progress: float = 0.0
    out_name: str | None = None
    out_size: int = 0
    ms: int = 0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0
    # Face keys the user clicked to leave visible. Per item, and never carried
    # between files: two people are only "the same person" to face recognition,
    # which this tool deliberately does not do. Photos: rounded boxes. Clips:
    # the 4-tuple ("t", track_id) naming a whole tracked person.
    restored: set[tuple[int | str, ...]] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "status": self.status,
            "kind": self.kind,
            "faces": self.faces,
            "scanned": self.scanned,
            "scan_state": self.scan_state,
            "scan_progress": round(self.scan_progress, 3),
            "out_name": self.out_name,
            "out_size": self.out_size,
            "ms": self.ms,
            "error": self.error,
            "meta": self.meta,
            "width": self.width,
            "height": self.height,
            "restored": len(self.restored),
        }


class Session:
    def __init__(self, root: Path, settings: Settings) -> None:
        self.id = root.name
        self.root = root
        self.in_dir = root / "in"
        self.out_dir = root / "out"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.created = time.time()
        self.touched = self.created
        self.settings = settings
        self.items: list[Item] = []
        self.state = "staged"  # staged | queued | rendering | ready | error
        self.progress = 0
        self.result_path: Path | None = None
        self.result_name: str | None = None
        self.result_size = 0
        self.rendered_with: dict | None = None
        self.error: str | None = None
        self.downloaded = False
        self.bytes_in = 0

        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self._photos: dict[str, photo_io.Photo] = {}
        self._detections: dict[str, list[Face]] = {}
        # Per-clip scan state. The decoded clips themselves never enter RAM -
        # these entries hold only per-frame detections and the tracker's masks,
        # which is kilobytes - plus liveness flags for the background scan.
        self._clips: dict[str, ClipScan] = {}
        self._clip_events: dict[str, threading.Event] = {}

    # -- lifecycle --------------------------------------------------------

    def touch(self) -> None:
        self.touched = time.time()

    def age(self) -> float:
        return time.time() - self.created

    def idle(self) -> float:
        return time.time() - self.touched

    def should_reap(self, ttl: float, max_age: float) -> bool:
        return self.idle() > ttl or self.age() > max_age

    # -- cached decode / detection ---------------------------------------

    def load_photo(self, item: Item) -> photo_io.Photo:
        """Decode once and keep it while the user tunes. Bounded, so RAM stays flat."""
        with self.lock:
            cached = self._photos.get(item.stored)
            if cached is not None:
                self._photos[item.stored] = self._photos.pop(item.stored)
                return cached

        source = self.in_dir / item.stored
        if not source.exists():
            raise photo_io.PhotoError(
                "staged copy was freed after download - re-upload to render again"
            )
        decoded = photo_io.load_photo(source)

        with self.lock:
            if decoded.width * decoded.height <= PREVIEW_CACHE_PIXELS:
                self._photos[item.stored] = decoded
                while len(self._photos) > PHOTO_CACHE_ENTRIES:
                    self._photos.pop(next(iter(self._photos)))
        return decoded

    def detections(self, item: Item, detector: FaceDetector) -> list[Face]:
        """Detection is the expensive step; preview must never repeat it."""
        with self.lock:
            cached = self._detections.get(item.stored)
        if cached is not None:
            return cached

        # Serialised so two concurrent preview requests don't both run inference.
        with self.io_lock:
            with self.lock:
                cached = self._detections.get(item.stored)
            if cached is not None:
                return cached

            decoded = self.load_photo(item)
            faces = detector.detect(decoded.bgr)
            with self.lock:
                self._detections[item.stored] = faces
                item.faces = len(faces)
                item.scanned = True
                item.width = decoded.width
                item.height = decoded.height
            return faces

    def forget_detections(self) -> None:
        """Drop when detection settings changed.

        Covers both kinds: photo detections, and clip scans - including one
        running in the background. An in-flight scan detects with the *old*
        settings, so its outputs must not survive: the worker publishes only
        into the entry that is still current, and clearing the dict here forces
        the next preview to start a fresh scan against the new settings.
        """
        with self.lock:
            self._detections.clear()
            clips = list(self._clips.items())
            self._clips.clear()
            self._clip_events.clear()
        for stored, _scan in clips:
            item = next((i for i in self.items if i.stored == stored), None)
            if item is not None and item.kind == "video":
                item.scanned = False
                item.faces = 0
                item.scan_state = "idle"
                item.scan_progress = 0.0

    def covered(self, item: Item, detector: FaceDetector) -> list[Face]:
        """The faces that will actually be masked: detections minus the restored.

        Preview and render both go through here. One list, one truth: if the map
        on screen and the file on disk could disagree about who is hidden, the
        map would be worse than useless.
        """
        with self.lock:
            restored = set(item.restored)
        return [f for f in self.detections(item, detector) if f.key() not in restored]

    def restored_faces(self, item: Item, detector: FaceDetector) -> list[Face]:
        """Detections the user chose to leave visible. The complement of covered()."""
        hidden = {f.key() for f in self.covered(item, detector)}
        return [f for f in self.detections(item, detector) if f.key() not in hidden]

    def set_restored(self, index: int, key: object, restore: bool) -> str | None:
        """Leave one detected person visible, or cover them again. None on success.

        The key is checked against the live detections rather than trusted, so a
        stale or invented box cannot end up in the restore set. It also means the
        only thing this can ever do is *un*-cover a face the detector already
        found with the preset currently in force.
        """
        if not 0 <= index < len(self.items):
            return "no such photo in this batch"
        item = self.items[index]

        if isinstance(key, (list, tuple)) and len(key) == 2 and key[0] == "t":
            return self._set_track_restored(item, key[1], restore)
        return self._set_box_restored(item, key, restore)

    def _set_box_restored(self, item: Item, key: object, restore: bool) -> str | None:
        """The stills side of set_restored: one detected box on one photo."""
        if not isinstance(key, (list, tuple)) or len(key) != 4:
            return "a face key must be four numbers"
        try:
            wanted = tuple(round(float(part)) for part in key)
        except (TypeError, ValueError):
            return "a face key must be four numbers"

        with self.lock:
            detected = self._detections.get(item.stored)
        if detected is None:
            # Nothing detected yet, so there is nothing to check the key against.
            return "this photo has not been scanned yet"

        if not any(face.key() == wanted for face in detected):
            return "no such face in this photo"

        # A clip's restored names are tracker ids, and they are checked against
        # the scan that is actually on file rather than trusted either.
        with self.lock:
            if restore:
                item.restored.add(wanted)
            else:
                item.restored.discard(wanted)
            self._drop_result_locked()
        return None

    # -- clip scans --------------------------------------------------------

    def _set_track_restored(self, item: Item, raw: object, restore: bool) -> str | None:
        """The clip side of set_restored: name a tracked person, not a frame.

        One click means "this person keeps their pixels in the whole clip", so
        the id, not any frame's box, is what is stored.
        """
        try:
            wanted = int(raw)
        except (TypeError, ValueError):
            return "a face key must be four numbers"

        with self.lock:
            scan = self._clips.get(item.stored)
            present = {mask.track for row in scan.masks for mask in row} if scan else set()
        if scan is None or scan.state != "done":
            return "this clip has not been scanned yet"
        if wanted not in present:
            return "no such face in this clip"

        with self.lock:
            if restore:
                item.restored.add(("t", wanted))
            else:
                item.restored.discard(("t", wanted))
            self._drop_result_locked()
        return None

    def clip_scan(self, item: Item) -> ClipScan:
        """The scan entry for a clip item, created empty on first touch."""
        with self.lock:
            return self._clips.setdefault(item.stored, ClipScan())

    def request_clip_scan(self, item: Item, detector: FaceDetector) -> ClipScan:
        """Start the background scan unless one is already running or done.

        Returns whatever state the entry is in; the preview route reads it and
        either serves frames or tells the page "scanning". Only ever starts one
        thread per clip per cache lifetime.
        """
        with self.lock:
            scan = self._clips.setdefault(item.stored, ClipScan())
            if scan.state in ("running", "done"):
                return scan
            if scan.state == "error":
                # A failed scan is retried on demand: transients (disk sleep,
                # a size race) deserve another chance, and the route that
                # notices never has to explain a permanent state.
                pass
            scan.state = "running"
            scan.error = None
            scan.progress = 0.0
            item.scan_state = "running"
            item.scan_progress = 0.0
            event = threading.Event()
            self._clip_events[item.stored] = event
            worker = threading.Thread(
                target=self._clip_scan_worker,
                args=(item, detector, scan, event),
                name=f"amurcloak-clip-{item.stored}",
                daemon=True,
            )
            worker.start()
            return scan

    def wait_for_clip_scan(self, item: Item) -> ClipScan:
        """Block until the background scan for this clip settles. None on entry."""
        with self.lock:
            event = self._clip_events.get(item.stored)
        if event is not None:
            event.wait()
        return self._clips.get(item.stored) or self.clip_scan(item)

    def ensure_clip_done(
        self, item: Item, detector: FaceDetector, on_progress=None
    ) -> ClipScan:
        """A finished scan for this clip, one way or another.

        Running in the background? Wait for it. Already done? Hand it back.
        Failed? Say so. Otherwise - the render worker path, where no preview
        has ever started a scan - run it inline, serialised on ``io_lock`` so
        two clips in one batch cannot race each other's decoder.
        """
        with self.lock:
            scan = self._clips.setdefault(item.stored, ClipScan())
            event = self._clip_events.get(item.stored) if scan.state == "running" else None
        if event is not None:
            event.wait()

        if scan.state == "done":
            return scan
        if scan.state == "error":
            raise VideoError(scan.error or "the clip scan failed")

        # A scan nobody started. Run it here, under the same io_lock the photo
        # detection path uses, and publish as the worker would.
        with self.io_lock:
            with self.lock:
                if scan.state == "done":
                    return scan
                scan.state = "running"
                scan.error = None
                scan.progress = 0.0
                item.scan_state = "running"
                event = threading.Event()
                self._clip_events[item.stored] = event
            source = self.in_dir / item.stored
            try:
                frames, measured_fps = video_render.scan_frames(
                    source,
                    detector,
                    on_progress=on_progress,
                    total_frames=int(item.meta.get("frames") or 0),
                )
                masks = video_track.build_masks(frames)
            except Exception as exc:  # a failed scan is a reported state
                with self.lock:
                    scan.state = "error"
                    scan.error = str(exc)
                    item.scan_state = "error"
                    item.error = str(exc)
                event.set()
                raise
            self._publish_clip(item, scan, frames, masks, measured_fps)
            event.set()
            return scan

    def _publish_clip(
        self,
        item: Item,
        scan: ClipScan,
        frames: list[list[Face]],
        masks: list[list[video_track.Mask]],
        measured_fps: float,
    ) -> None:
        """Publish scan results into the entry that is still current.

        The guard matters because a settings change clears the scan cache and
        spawns a replacement scan while an old one may still be running; the
        loser's outputs must never be mistaken for fresh ones.
        """
        with self.lock:
            if self._clips.get(item.stored) is not scan:
                return
            scan.frames = frames
            scan.masks = masks
            scan.fps = measured_fps
            scan.state = "done"
            scan.progress = 1.0
            item.faces = video_track.count_tracks(masks)
            item.scanned = True
            item.width = int(item.meta.get("width") or 0)
            item.height = int(item.meta.get("height") or 0)
            item.scan_state = "done"
            item.error = None

    def _clip_scan_worker(
        self, item: Item, detector: FaceDetector, scan: ClipScan, event: threading.Event
    ) -> None:
        """The background half. Publishes progress, never raises."""
        source = self.in_dir / item.stored

        def progress(fraction: float) -> None:
            scan.progress = fraction
            item.scan_progress = fraction

        try:
            frames, measured_fps = video_render.scan_frames(
                source,
                detector,
                on_progress=progress,
                total_frames=int(item.meta.get("frames") or 0),
            )
            masks = video_track.build_masks(frames)
        except Exception as exc:  # noqa: BLE001 - recorded on the session, never swallowed silently
            with self.lock:
                # A settings change may have cleared the cache and started a new
                # scan while this one worked; only publish into the entry that
                # is still current, and leave the new scan's event alone.
                if self._clips.get(item.stored) is scan:
                    scan.state = "error"
                    scan.error = str(exc)
                    item.scan_state = "error"
                    item.error = str(exc)
            event.set()
            return

        self._publish_clip(item, scan, frames, masks, measured_fps)
        event.set()

    def _drop_result_locked(self) -> None:
        """Discard any rendered result. Caller holds the lock.

        Shared by settings changes and face toggles, because both change what the
        export would contain and neither may leave a stale download behind.
        """
        if self.rendered_with is None:
            return
        self.clear_outputs()
        self.rendered_with = None
        self.state = "staged"
        self.progress = 0
        for item in self.items:
            if item.status == "done":
                item.status = "staged"

    def apply_settings(self, new_settings: Settings) -> dict:
        """Swap in new settings, discarding any result they invalidate.

        A result rendered under different settings must never remain
        downloadable: handing someone last-render pixels while the UI says
        "blur, margin 0.25" would be a silent privacy failure.
        """
        detection_fields = ("score", "min_face", "detect_edge", "tiles")

        with self.lock:
            previous = self.settings.to_dict()
            incoming = new_settings.to_dict()
            detection_changed = any(
                previous[f] != incoming[f] for f in detection_fields
            )

            self.settings = new_settings
            invalidated = (
                self.rendered_with is not None and self.rendered_with != incoming
            )

            if invalidated:
                self._drop_result_locked()

            self.touch()
            return {
                "detection_changed": detection_changed,
                "result_invalidated": invalidated,
            }

    # -- outputs ----------------------------------------------------------

    def clear_outputs(self) -> None:
        with self.lock:
            if self.out_dir.exists():
                shutil.rmtree(self.out_dir, ignore_errors=True)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.result_path and self.result_path.exists():
                self.result_path.unlink()
            self.result_path = None
            self.result_name = None
            self.result_size = 0
            self.downloaded = False

    def free_inputs(self) -> int:
        """Drop staging bytes. Called after a download, since re-render then means re-upload."""
        with self.lock:
            freed = sum(i.size for i in self.items if (self.in_dir / i.stored).exists())
            shutil.rmtree(self.in_dir, ignore_errors=True)
            self.in_dir.mkdir(parents=True, exist_ok=True)
            self._photos.clear()
            self._detections.clear()
            # Clips: frames were never in RAM, but the masks describe frames
            # that can no longer be read - a re-render needs a re-scan.
            self._clips.clear()
            self._clip_events.clear()
            for item in self.items:
                if item.kind == "video":
                    item.scanned = False
                    item.faces = 0
                    item.scan_state = "idle"
                    item.scan_progress = 0.0
            return freed

    # -- reporting --------------------------------------------------------

    def disk_usage(self) -> dict:
        def tree_size(path: Path) -> int:
            if not path.exists():
                return 0
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    with contextlib.suppress(OSError):
                        total += p.stat().st_size
            return total

        with self.lock:
            inputs = tree_size(self.in_dir)
            outputs = tree_size(self.out_dir)
            result = 0
            if self.result_path and self.result_path.exists():
                result = self.result_path.stat().st_size
            return {
                "inputs": inputs,
                "outputs": outputs,
                "result": result,
                "total": inputs + outputs + result,
                "inputs_freed": not any(
                    (self.in_dir / i.stored).exists() for i in self.items
                ),
            }

    def snapshot(self) -> dict:
        with self.lock:
            disk = self.disk_usage()
            return {
                "id": self.id,
                "state": self.state,
                "progress": self.progress,
                "total": len(self.items),
                "error": self.error,
                "downloaded": self.downloaded,
                "settings": self.settings.to_dict(),
                "items": [i.to_dict() for i in self.items],
                "result": {
                    "name": self.result_name,
                    "size": self.result_size,
                    "ready": bool(self.result_path and self.result_path.exists()),
                },
                "disk": disk,
                "age": round(self.age(), 1),
                "idle": round(self.idle(), 1),
            }


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class SessionStore:
    # Directories the store shares the temp root with. Never touched by the
    # orphan purge. Session ids are ``<timestamp>-<hex>``, so a leading dot is
    # also treated as reserved.
    RESERVED = frozenset({".spool"})

    def __init__(
        self,
        root: Path | None = None,
        *,
        ttl: float = SESSION_TTL_SEC,
        max_age: float = SESSION_MAX_AGE_SEC,
        shred: bool = False,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_TMP
        self.ttl = ttl
        self.max_age = max_age
        self.shred = shred
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.purge_orphans()

    def purge_orphans(self) -> int:
        """Wipe leftovers from a previous run. Nothing here can be live.

        Reserved entries are skipped: the store shares its root with the
        process-level temp dir (``.spool``) that Werkzeug writes multipart
        uploads into. Deleting that out from under a running app makes every
        upload fail, so it must survive the purge.
        """
        removed = 0
        for child in self.root.iterdir():
            if child.name in self.RESERVED or child.name.startswith("."):
                continue
            if child.is_dir():
                purge_tree(child, shred=False)
                removed += 1
            elif child.is_file():
                with contextlib.suppress(OSError):
                    child.unlink()
                    removed += 1
        return removed

    def create(self, settings: Settings | None = None) -> Session:
        sid = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
        session = Session(self.root / sid, settings or build_settings())
        with self.lock:
            self.sessions[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        with self.lock:
            return self.sessions.get(sid)

    def discard(self, sid: str) -> int:
        with self.lock:
            session = self.sessions.pop(sid, None)
        if session is None:
            return 0
        with session.lock:
            freed = session.disk_usage()["total"]
        purge_tree(session.root, shred=self.shred)
        return freed

    def reap(self) -> tuple[int, int]:
        """Drop idle sessions. Returns (count, bytes_freed)."""
        with self.lock:
            stale = [s.id for s in self.sessions.values() if s.should_reap(self.ttl, self.max_age)]
        freed = 0
        for sid in stale:
            freed += self.discard(sid)
        return len(stale), freed

    def usage(self) -> dict:
        with self.lock:
            sessions = list(self.sessions.values())

        total = 0
        for session in sessions:
            # The reaper can discard a session between the snapshot above and
            # this call. That race is expected, so accounting just skips it.
            with contextlib.suppress(Exception):
                total += session.disk_usage()["total"]

        # Include anything on disk not tracked as a live session (should be zero
        # outside of an in-flight upload).
        on_disk = 0
        if self.root.exists():
            for child in self.root.iterdir():
                if child.is_dir():
                    for p in child.rglob("*"):
                        if p.is_file():
                            with contextlib.suppress(OSError):
                                on_disk += p.stat().st_size

        return {
            "sessions": len(sessions),
            "tracked_bytes": total,
            "on_disk_bytes": on_disk,
            "free_bytes": free_bytes(self.root),
            "root": str(self.root),
            "ttl_sec": self.ttl,
        }

    def start_reaper(self, interval: float = REAP_INTERVAL_SEC) -> threading.Thread:
        def loop() -> None:
            while True:
                time.sleep(interval)
                # A sweep that throws must not kill the loop; the next one retries.
                with contextlib.suppress(Exception):
                    self.reap()

        thread = threading.Thread(target=loop, name="amurcloak-reaper", daemon=True)
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


class Renderer:
    """Single-threaded render worker. One job at a time keeps RAM flat."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.queue: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="amurcloak-render", daemon=True)
        self.thread.start()

    def submit(self, session: Session) -> None:
        with session.lock:
            if session.state in ("queued", "rendering"):
                return
            session.state = "queued"
            session.progress = 0
            session.error = None
        self.queue.put(session.id)

    def _loop(self) -> None:
        while True:
            sid = self.queue.get()
            try:
                session = self.store.get(sid)
                if session is not None:
                    self._render(session)
            except Exception as exc:  # noqa: BLE001
                session = self.store.get(sid)
                if session is not None:
                    with session.lock:
                        session.state = "error"
                        session.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.queue.task_done()

    def _detector_for(self, session: Session) -> FaceDetector:
        s = session.settings
        return FaceDetector(
            score=s.score,
            nms=0.3,
            min_face=s.min_face,
            detect_edge=s.detect_edge,
            tiles=s.tiles,
        )

    def _render(self, session: Session) -> None:
        detector = self._detector_for(session)

        with session.lock:
            session.state = "rendering"
            session.touch()
            session.clear_outputs()
            items = list(session.items)

        renderable = [i for i in items if i.status != "error"]
        if not renderable:
            with session.lock:
                session.state = "error"
                session.error = "nothing to render"
            return

        archived = len(renderable) > 1
        zip_tmp = session.root / "result.zip.part"
        zip_final = session.root / "result.zip"
        zf = zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_STORED) if archived else None
        taken: set[str] = set()

        try:
            for position, item in enumerate(renderable):
                started = time.perf_counter()

                # A clip's progress is fractional so the bar tracks the frame
                # it is actually on, not a full step per item that sits at zero
                # for a minute. Photos keep their whole numbers; they share the
                # same scale, items out of len(renderable).
                def advance(fraction: float, position=position) -> None:
                    with session.lock:
                        session.progress = min(position + fraction, len(renderable))

                try:
                    if item.kind == "video":
                        out_name = video_probe.unique_output_name(item.name, taken)
                        # Keep the name the preview already showed, unless the
                        # collision set has claimed it since.
                        if item.out_name is None or item.out_name in taken:
                            item.out_name = out_name
                        taken.add(item.out_name)

                        out_path = session.out_dir / item.out_name
                        self._render_clip(session, item, out_path, advance)
                        out_size = out_path.stat().st_size
                    else:
                        decoded = session.load_photo(item)
                        faces = session.covered(item, detector)

                        if item.out_name is None or item.out_name in taken:
                            item.out_name = photo_io.unique_output_name(
                                item.name, decoded.alpha is not None, taken
                            )
                        else:
                            taken.add(item.out_name)

                        redact.apply_redaction(
                            decoded.bgr,
                            faces,
                            session.settings.mode,
                            session.settings.margin,
                            session.settings.strength,
                        )

                        out_path = session.out_dir / item.out_name
                        photo_io.save_photo(
                            out_path,
                            decoded.bgr,
                            decoded.alpha,
                            decoded.icc if not session.settings.strip_exif else None,
                            quality=session.settings.quality,
                            keep_icc=not session.settings.strip_exif,
                            metadata=session.settings.metadata,
                            spoof_when=session.settings.spoof_when,
                        )
                        out_size = out_path.stat().st_size

                    if zf is not None:
                        # Append then remove, so the batch never exists twice on
                        # disk. ZIP_STORED, so this size is the real contribution.
                        zf.write(out_path, arcname=item.out_name)
                        out_size = zf.getinfo(item.out_name).file_size
                        out_path.unlink()

                    with session.lock:
                        item.status = "done"
                        if item.kind != "video":
                            # Photos: the covered count already reflects any
                            # "leave visible" choice, exactly as 0.9 did.
                            item.scanned = True
                            item.faces = len(faces)
                        item.out_size = out_size
                        item.ms = int((time.perf_counter() - started) * 1000)
                        session.progress = min(position + 1, len(renderable))
                        session.touch()

                except Exception as exc:  # noqa: BLE001
                    with session.lock:
                        item.status = "error"
                        item.error = str(exc)
                        item.ms = int((time.perf_counter() - started) * 1000)
                        session.progress = min(position + 1, len(renderable))
                        session.touch()

            if zf is not None:
                zf.close()
                zf = None

            with session.lock:
                done = [i for i in renderable if i.status == "done"]
                if not done:
                    session.state = "error"
                    session.error = "every file failed"
                    return

                if archived and len(done) > 1:
                    zip_tmp.replace(zip_final)
                    shutil.rmtree(session.out_dir, ignore_errors=True)
                    session.result_path = zip_final
                    session.result_name = "amurcloak-redacted.zip"
                    session.result_size = zip_final.stat().st_size
                else:
                    # Exactly one success: hand back the image itself, not a
                    # zip wrapping a single file.
                    single = done[0]
                    name = single.out_name or single.name
                    if archived and zip_tmp.exists():
                        # Every success was spooled into the zip and unlinked
                        # from out_dir as it went, so this last survivor has to
                        # come back out of the spool. Without this the batch
                        # would report "ready" with no file behind it.
                        with zipfile.ZipFile(zip_tmp) as spool:
                            spool.extract(name, session.out_dir)
                    session.result_path = session.out_dir / name
                    session.result_name = name
                    session.result_size = (
                        session.result_path.stat().st_size
                        if session.result_path.exists()
                        else 0
                    )

                session.state = "ready"
                session.progress = len(renderable)
                session.rendered_with = session.settings.to_dict()
                session.touch()

        finally:
            if zf is not None:
                with contextlib.suppress(Exception):
                    zf.close()
            if zip_tmp.exists():
                zip_tmp.unlink(missing_ok=True)
            # A failed render should not leave half a batch on disk.
            with session.lock:
                if session.state == "error" and session.out_dir.exists():
                    shutil.rmtree(session.out_dir, ignore_errors=True)
                    session.out_dir.mkdir(parents=True, exist_ok=True)

    def _render_clip(
        self,
        session: Session,
        item: Item,
        out_path: Path,
        advance,
    ) -> None:
        """One clip through both passes.

        The scan is shared with the preview path - same cache, same entry -
        so a clip someone has been scrubbing renders immediately.
        """
        source = session.in_dir / item.stored
        info = self._clip_geometry(item, source)

        def scan_progress(fraction: float) -> None:
            advance(fraction * 0.5)

        detector = video_render.clip_detector(session.settings)
        scan = session.ensure_clip_done(item, detector, on_progress=scan_progress)

        # The rate measured while scanning, not the container's claim: a
        # variable-rate or mislabelled file would otherwise encode at the wrong
        # rate and drift away from its own audio.
        fps = scan.fps or info.fps

        def render_progress(fraction: float) -> None:
            # Pass B owns the second half of this item's slice. When the scan
            # was served from cache the first half never spent wall time here,
            # and that is allowed to be visible in the bar's rate.
            advance(0.5 + fraction * 0.5)

        # The people the user clicked to leave visible. Read once, at the start,
        # rather than per frame: the export has to be one decision, and it has to
        # be the same one the preview made. Photo keys are four numbers, so a
        # track id is only ever behind the "t" marker.
        with session.lock:
            restored_tracks = frozenset(
                key[1] for key in item.restored if key and key[0] == "t"
            )

        video_render.render_to_disk(
            source,
            out_path,
            scan.masks,
            session.settings,
            fps=fps,
            width=info.width,
            height=info.height,
            on_progress=render_progress,
            restored_tracks=restored_tracks,
        )

    @staticmethod
    def _clip_geometry(item: Item, source: Path):
        """fps and dimensions from the upload probe, or a fresh probe."""
        fps = float(item.meta.get("fps") or 0)
        width = int(item.meta.get("width") or 0)
        height = int(item.meta.get("height") or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            info = video_probe.probe(source)
            item.meta.update(info.to_dict())
            item.width = info.width
            item.height = info.height
            return info
        return video_probe.ClipInfo(
            fps=fps,
            width=width,
            height=height,
            frame_count=int(item.meta.get("frames") or 0),
            duration=float(item.meta.get("duration") or 0.0),
            has_audio=bool(item.meta.get("has_audio")),
        )
