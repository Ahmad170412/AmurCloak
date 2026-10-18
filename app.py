"""TigerBlur web app.

Binds to localhost only, by default. Nothing in this process makes an outbound
network call at runtime - the one exception is ``--allow-download``, which
fetches the 232KB face model once if it is missing.

Disk posture:
* ``tempfile.tempdir`` is redirected into the app's own temp root, so the
  multipart spool from an upload lands somewhere the user can see and delete
  rather than hiding in the system temp directory.
* Inputs are freed automatically once a result has been downloaded.
* A background reaper clears idle sessions, and startup purges orphans.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from flask import (
    Flask,
    abort,
    after_this_request,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.exceptions import RequestEntityTooLarge

from engine import photo as photo_io
from engine import pipeline, presets
from engine.detector import FaceDetector, ensure_model
from engine.sessions import (
    DEFAULT_TMP,
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_SESSION_BYTES,
    SESSION_MAX_AGE_SEC,
    SESSION_TTL_SEC,
    Item,
    Renderer,
    SessionStore,
    StorageError,
    ensure_space,
)

CHUNK = 1 << 20


@dataclass
class Config:
    tmp_root: Path = DEFAULT_TMP
    ttl: float = SESSION_TTL_SEC
    max_age: float = SESSION_MAX_AGE_SEC
    shred: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    allow_download: bool = False
    free_after_download: bool = True


def _detector(settings: presets.Settings) -> FaceDetector:
    return FaceDetector(
        score=settings.score,
        nms=0.3,
        min_face=settings.min_face,
        detect_edge=settings.detect_edge,
        tiles=settings.tiles,
    )


def create_app(config: Config | None = None):
    """Build the app.

    Note on ``tempfile.tempdir``: redirecting it is a **process-level** side
    effect, which is correct for production (one app, one process) but would
    leak between instances if an embedder built two apps. ``app.dispose()``
    restores the previous value, and it is safe to call more than once.
    """
    config = config or Config()
    config.tmp_root.mkdir(parents=True, exist_ok=True)

    # The store purges orphaned session dirs from this root on construction, so
    # it has to exist first and be told what to leave alone.
    store = SessionStore(
        config.tmp_root, ttl=config.ttl, max_age=config.max_age, shred=config.shred
    )
    renderer = Renderer(store)
    store.start_reaper()

    # Keep every transient byte inside our own root, including Werkzeug's
    # multipart spool files. This must survive the purge above.
    spool = config.tmp_root / ".spool"
    spool.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(spool)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_SESSION_BYTES
    app.config["MAX_FORM_PARTS"] = MAX_FILES + 10
    app.config["JSON_SORT_KEYS"] = False
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # A local tool gets edited live. Never let a browser sit on a stale app.js.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["CONFIG"] = config

    @app.after_request
    def _fresh_static(response):
        if request.path.startswith(app.static_url_path or "/static"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers.pop("Expires", None)
        return response

    @app.context_processor
    def _asset_stamp():
        # Cache-buster pinned to the newest static file, so an edited stylesheet
        # or script is a brand new URL and the browser cannot reuse the old one.
        newest = 0.0
        static_dir = Path(app.static_folder or "")
        if static_dir.is_dir():
            for path in static_dir.rglob("*"):
                if path.is_file():
                    newest = max(newest, path.stat().st_mtime)
        return {"asset_v": str(int(newest))}

    # -- pages ------------------------------------------------------------

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/config")
    def api_config():
        return jsonify(
            {
                **presets.catalog(),
                "limits": {
                    "max_files": MAX_FILES,
                    "max_file_bytes": MAX_FILE_BYTES,
                    "max_session_bytes": MAX_SESSION_BYTES,
                },
                "usage": store.usage(),
                "supported": sorted(photo_io.SUPPORTED_IN),
                "heif": photo_io.HEIF_AVAILABLE,
                "free_after_download": config.free_after_download,
            }
        )

    @app.get("/api/usage")
    def api_usage():
        return jsonify(store.usage())

    @app.post("/api/reap")
    def api_reap():
        count, freed = store.reap()
        return jsonify({"sessions": count, "freed_bytes": freed, "usage": store.usage()})

    # -- upload -----------------------------------------------------------

    @app.post("/api/sessions")
    def create_session():
        # Werkzeug spools multipart parts into the process-level temp dir, which
        # we point inside our own root. If the user clears that folder while the
        # app is running, put it back rather than failing the upload.
        spool = config.tmp_root / ".spool"
        if not spool.is_dir():
            spool.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(spool)

        try:
            uploads = request.files.getlist("files")
        except OSError as exc:
            return (
                jsonify(
                    {
                        "error": (
                            "the upload could not be written to disk: "
                            f"{exc}. Check that {config.tmp_root} is writable."
                        )
                    }
                ),
                500,
            )
        if not uploads:
            return jsonify({"error": "no files were uploaded"}), 400
        if len(uploads) > MAX_FILES:
            return (
                jsonify({"error": f"{len(uploads)} files exceeds the {MAX_FILES} limit"}),
                413,
            )

        try:
            raw = json.loads(request.form.get("settings") or "{}")
        except json.JSONDecodeError:
            raw = {}
        settings = presets.from_payload(raw)

        incoming = request.content_length or 0
        try:
            ensure_space([config.tmp_root, Path(tempfile.gettempdir())], incoming)
        except StorageError as exc:
            return jsonify({"error": str(exc)}), 507

        session = store.create(settings)
        total = 0

        for index, upload in enumerate(uploads):
            original = Path(upload.filename or f"image-{index + 1}").name
            extension = Path(original).suffix.lower()
            stored = f"{index:04d}{extension}"

            item = Item(name=original, stored=stored)
            if extension not in photo_io.SUPPORTED_IN:
                item.status = "error"
                item.error = (
                    f"{extension or 'no extension'} is not a supported image format"
                )
                session.items.append(item)
                continue

            destination = session.in_dir / stored
            size = 0
            failure: str | None = None
            try:
                with open(destination, "wb") as handle:
                    while True:
                        chunk = upload.stream.read(CHUNK)
                        if not chunk:
                            break
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise StorageError(
                                f"{original} is larger than "
                                f"{MAX_FILE_BYTES // (1024 * 1024)}MB"
                            )
                        if total > MAX_SESSION_BYTES:
                            raise StorageError("this upload is larger than the size limit")
                        handle.write(chunk)
            except StorageError as exc:
                failure = str(exc)
            except Exception as exc:  # noqa: BLE001
                failure = f"could not receive file: {exc}"
            finally:
                upload.close()  # releases Werkzeug's spool file promptly

            if failure:
                destination.unlink(missing_ok=True)
                item.status = "error"
                item.error = failure
                session.items.append(item)
                continue

            item.size = size
            item.meta = photo_io.probe_metadata(destination)
            session.items.append(item)

        session.bytes_in = total
        session.touch()
        return jsonify(session.snapshot()), 201

    # -- tuning -----------------------------------------------------------

    @app.post("/api/sessions/<sid>/settings")
    def update_settings(sid: str):
        session = store.get(sid)
        if session is None:
            abort(404)
        payload = request.get_json(silent=True) or {}
        settings = presets.from_payload(payload)
        outcome = session.apply_settings(settings)
        if outcome["detection_changed"]:
            session.forget_detections()
        return jsonify({"settings": settings.to_dict(), **outcome})

    @app.get("/api/sessions/<sid>/preview")
    def preview(sid: str):
        session = store.get(sid)
        if session is None:
            abort(404)

        index = request.args.get("index", 0, type=int)
        view = request.args.get("view", "result")
        if view not in ("result", "audit"):
            view = "result"
        if not session.items:
            abort(404)
        index = max(0, min(index, len(session.items) - 1))
        item = session.items[index]

        settings = session.settings
        detector = _detector(settings)
        started = time.perf_counter()

        try:
            faces = session.detections(item, detector)
            decoded = session.load_photo(item)
        except photo_io.PhotoError as exc:
            return jsonify({"error": str(exc), "index": index}), 422

        image = pipeline.render_preview_bytes(decoded, faces, settings, view)
        report = pipeline.scene_report(faces, settings, decoded)
        report.update(
            {
                "index": index,
                "view": view,
                "name": item.name,
                "meta": item.meta,
                "image": base64.b64encode(image).decode("ascii"),
                "ms": int((time.perf_counter() - started) * 1000),
                "note": report["note"],
            }
        )
        response = jsonify(report)
        response.headers["Cache-Control"] = "no-store"
        return response

    # -- render -----------------------------------------------------------

    @app.post("/api/sessions/<sid>/render")
    def render(sid: str):
        session = store.get(sid)
        if session is None:
            abort(404)
        if not any(item.status != "error" for item in session.items):
            return jsonify({"error": "there are no photos here to redact"}), 400
        renderer.submit(session)
        return jsonify(session.snapshot()), 202

    @app.get("/api/sessions/<sid>")
    def poll(sid: str):
        session = store.get(sid)
        if session is None:
            abort(404)
        response = jsonify(session.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    # -- result -----------------------------------------------------------

    @app.get("/api/sessions/<sid>/result")
    def result(sid: str):
        session = store.get(sid)
        if session is None:
            abort(404)

        with session.lock:
            path = session.result_path
            name = session.result_name
            if path is None or not path.exists():
                return jsonify({"error": "no result available yet"}), 409

        if config.free_after_download and not session.downloaded:
            @after_this_request
            def _reclaim(response):
                # The staging copies exist to make re-rendering instant. Once the
                # result is in the user's hands they are just occupying disk.
                try:
                    session.downloaded = True
                    session.free_inputs()
                except Exception:
                    pass
                return response

        response = send_file(
            str(path),
            as_attachment=True,
            download_name=name or "tigerblur-output",
            conditional=True,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.delete("/api/sessions/<sid>")
    def discard(sid: str):
        freed = store.discard(sid)
        return jsonify({"freed_bytes": freed, "usage": store.usage()})

    # -- errors -----------------------------------------------------------

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):  # pragma: no cover - exercised via the UI
        return (
            jsonify(
                {
                    "error": (
                        f"this upload is larger than the "
                        f"{MAX_SESSION_BYTES // (1024 * 1024)}MB limit; "
                        "split it into smaller drops"
                    )
                }
            ),
            413,
        )

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify({"error": "no such session"}), 404

    @app.errorhandler(500)
    def server_error(exc):
        # Show the real cause, not Werkzeug's generic wrapper text.
        cause = getattr(exc, "original_exception", None) or exc
        return jsonify({"error": f"the server hit an unexpected problem: {cause}"}), 500

    app.store = store  # type: ignore[attr-defined]
    app.renderer = renderer  # type: ignore[attr-defined]

    def dispose() -> None:
        """Stop the reaper's next sweep, clear sessions, and undo the temp redirect."""
        if tempfile.tempdir == str(spool):
            tempfile.tempdir = previous_tempdir
        try:
            for sid in list(store.sessions):
                store.discard(sid)
        except Exception:
            pass

    app.dispose = dispose  # type: ignore[attr-defined]
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="tigerblur",
        description="Local face redaction for photos. Nothing leaves this machine.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tmp", default=str(DEFAULT_TMP), help="temp root for uploads and output")
    parser.add_argument("--ttl", type=float, default=SESSION_TTL_SEC, help="idle seconds before a session is reaped")
    parser.add_argument("--max-age", type=float, default=SESSION_MAX_AGE_SEC, help="hard session age limit in seconds")
    parser.add_argument("--shred", action="store_true", help="zero-fill temp files before deleting (slower)")
    parser.add_argument("--keep-inputs", action="store_true", help="do not free staged uploads after download")
    parser.add_argument("--allow-download", action="store_true", help="permit a one-time model download if it is missing")
    parser.add_argument("--open", action="store_true", help="open the browser on start")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        model = ensure_model(allow_download=args.allow_download)
    except Exception as exc:
        print(f"\n  Cannot start: {exc}\n", flush=True)
        return 1

    config = Config(
        tmp_root=Path(args.tmp).resolve(),
        ttl=args.ttl,
        max_age=args.max_age,
        shred=args.shred,
        host=args.host,
        port=args.port,
        allow_download=args.allow_download,
        free_after_download=not args.keep_inputs,
    )

    app = create_app(config)
    store: SessionStore = app.store  # type: ignore[attr-defined]
    usage = store.usage()

    url = f"http://{'localhost' if config.host == '127.0.0.1' else config.host}:{config.port}"
    print("\n  TigerBlur")
    print("  " + "-" * 46)
    print(f"  model      {model.name} ({model.stat().st_size / 1024:.0f}KB, checksum verified)")
    print(f"  temp root  {config.tmp_root}")
    print(f"  free disk  {usage['free_bytes'] / 1e9:.1f}GB")
    print(f"  session    reaped after {int(config.ttl / 60)}min idle")
    print("  network    none at runtime - no uploads, no telemetry")
    print("  " + "-" * 46)
    print(f"  open       {url}\n")

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=config.host, port=config.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
