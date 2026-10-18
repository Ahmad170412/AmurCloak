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
    faces: int = 0
    scanned: bool = False  # whether detection has run on this file yet
    out_name: str | None = None
    out_size: int = 0
    ms: int = 0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "status": self.status,
            "faces": self.faces,
            "scanned": self.scanned,
            "out_name": self.out_name,
            "out_size": self.out_size,
            "ms": self.ms,
            "error": self.error,
            "meta": self.meta,
            "width": self.width,
            "height": self.height,
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
        with self.lock:
            self._detections.clear()

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
                self.clear_outputs()
                self.rendered_with = None
                self.state = "staged"
                self.progress = 0
                for item in self.items:
                    if item.status == "done":
                        item.status = "staged"

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
            for item in renderable:
                started = time.perf_counter()
                try:
                    decoded = session.load_photo(item)
                    faces = session.detections(item, detector)

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
                        item.faces = len(faces)
                        item.scanned = True
                        item.out_size = out_size
                        item.ms = int((time.perf_counter() - started) * 1000)
                        session.progress = min(session.progress + 1, len(renderable))
                        session.touch()

                except Exception as exc:  # noqa: BLE001
                    with session.lock:
                        item.status = "error"
                        item.error = str(exc)
                        item.ms = int((time.perf_counter() - started) * 1000)
                        session.progress = min(session.progress + 1, len(renderable))
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
