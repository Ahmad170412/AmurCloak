#!/usr/bin/env python
"""End-to-end API tests for AmurCloak.

Run with:  .venv/bin/python tests/test_api.py

Exercises the real Flask app through its test client: upload, tune, preview,
render, poll, download, and the disk-reclamation behaviour.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import uuid
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# src/ for the package under test, ROOT for the tests package itself.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from amurcloak.app import Config, create_app  # noqa: E402
from amurcloak.engine import photo as photo_io  # noqa: E402
from tests.test_engine import build_group_photo  # noqa: E402

LENA = ROOT / "tests" / "fixtures" / "lena.jpg"

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


# Resolved once, at import, before any app has redirected tempfile.tempdir.
# Deriving it later would pick up a leaked redirect pointing at a deleted dir.
BASE_TMP = Path(tempfile.mkdtemp(prefix="amurcloak-tests-"))


def make_app():
    root = BASE_TMP / f"app-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    app = create_app(Config(tmp_root=root, ttl=3600, max_age=7200))
    app.config["TESTING"] = True
    return app, root


def teardown(app, root: Path) -> None:
    app.dispose()
    shutil.rmtree(root, ignore_errors=True)


def upload(client, paths: list[Path], settings: dict | None = None, names=None):
    data = {"settings": json.dumps(settings or {})}
    files = []
    for i, path in enumerate(paths):
        if path is None:
            files.append((io.BytesIO(b"not an image at all"), names[i] if names else "bad.txt"))
        else:
            files.append((io.BytesIO(path.read_bytes()), names[i] if names else path.name))
    data["files"] = files
    return client.post("/api/sessions", data=data, content_type="multipart/form-data")


def render_and_wait(client, sid: str, timeout: float = 120.0) -> dict:
    response = client.post(f"/api/sessions/{sid}/render")
    require(response.status_code == 202, f"render returned {response.status_code}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get(f"/api/sessions/{sid}").get_json()
        if snapshot["state"] in ("ready", "error"):
            return snapshot
        time.sleep(0.2)
    raise AssertionError("render timed out")


# ---------------------------------------------------------------------------


@check("config endpoint exposes presets, modes and limits")
def t_config():
    app, root = make_app()
    try:
        payload = app.test_client().get("/api/config").get_json()
        require(payload["defaults"]["mode"] == "black", "default mode must be black")
        require(len(payload["presets"]) == 3, "expected three presets")
        require(len(payload["modes"]) == 3, "expected three modes")
        require(payload["limits"]["max_files"] > 0, "missing limits")
        require(payload["defaults"]["strip_exif"] is True, "strip_exif must default on")
    finally:
        teardown(app, root)


@check("single photo: upload, preview, render, download as a plain image")
def t_single_photo():
    app, root = make_app()
    try:
        client = app.test_client()
        response = upload(client, [LENA], {"mode": "black", "preset": "balanced"})
        require(response.status_code == 201, f"upload returned {response.status_code}")
        snapshot = response.get_json()
        sid = snapshot["id"]
        require(len(snapshot["items"]) == 1, "expected one item")

        preview = client.get(f"/api/sessions/{sid}/preview?index=0&view=audit")
        require(preview.status_code == 200, "preview failed")
        payload = preview.get_json()
        require(payload["count"] >= 1, f"expected a face in preview, got {payload['count']}")
        require(payload["image"], "preview returned no image")
        require(payload["meta"] is not None, "preview returned no metadata")

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"state was {snapshot['state']}: {snapshot['error']}")
        require(snapshot["result"]["ready"], "no result produced")
        require(snapshot["result"]["name"].endswith(".jpg"), "single output should be a jpg")

        downloaded = client.get(f"/api/sessions/{sid}/result")
        require(downloaded.status_code == 200, "download failed")
        body = downloaded.data
        require(len(body) > 1000, "downloaded file looks empty")

        # The delivered bytes must be a clean JPEG with no metadata.
        out = root / "check.jpg"
        out.write_bytes(body)
        probe = photo_io.probe_metadata(out)
        require(not probe["has_exif"], f"EXIF made it into the download: {probe}")
        require(not probe["has_gps"], "GPS made it into the download")

        # And the face region must actually be black.
        loaded = photo_io.load_photo(out)
        img = loaded.bgr
        require(
            (img[: int(img.shape[0] * 0.05)] > 0).any(),
            "output appears to be entirely black",
        )
        darkest = img.sum(axis=2).min()
        require(darkest == 0, "no fully black pixels found - face may not be redacted")
    finally:
        teardown(app, root)


@check("batch: renders to a zip and frees staged inputs after download")
def t_batch_zip_and_reclaim():
    app, root = make_app()
    try:
        client = app.test_client()
        batch = [LENA, build_group_photo(), LENA]
        response = upload(client, batch, {"mode": "black"}, names=["a.jpg", "group.jpg", "b.jpg"])
        require(response.status_code == 201, "batch upload failed")
        sid = response.get_json()["id"]

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"batch failed: {snapshot['error']}")
        require(
            snapshot["result"]["name"].endswith(".zip"),
            f"multi-file batch should zip, got {snapshot['result']['name']}",
        )
        require(
            all(item["status"] == "done" for item in snapshot["items"]),
            "not every item rendered",
        )
        require(snapshot["disk"]["inputs"] > 0, "staged inputs should exist before download")

        downloaded = client.get(f"/api/sessions/{sid}/result")
        require(downloaded.status_code == 200, "zip download failed")

        archive = zipfile.ZipFile(io.BytesIO(downloaded.data))
        names = archive.namelist()
        require(len(names) == 3, f"expected 3 files in zip, got {names}")
        require("group.jpg" in names, f"original names must survive into the zip: {names}")
        for name in names:
            blob = archive.read(name)
            require(len(blob) > 500, f"{name} looks empty")

        after = client.get(f"/api/sessions/{sid}").get_json()
        require(
            after["disk"]["inputs"] == 0,
            f"inputs should be freed after download, got {after['disk']['inputs']}",
        )
        require(after["disk"]["result"] > 0, "result must survive the reclaim")
    finally:
        teardown(app, root)


@check("changing settings invalidates a stale result")
def t_stale_invalidation():
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black"}).get_json()["id"]
        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", "first render failed")
        require(snapshot["result"]["ready"], "no result to invalidate")

        changed = client.post(
            f"/api/sessions/{sid}/settings", json={"mode": "blur", "margin": 0.3}
        ).get_json()
        require(changed["result_invalidated"], "changing mode should invalidate the result")

        after = client.get(f"/api/sessions/{sid}").get_json()
        require(after["state"] == "staged", f"state should reset, got {after['state']}")
        require(not after["result"]["ready"], "stale result is still downloadable")

        redo = render_and_wait(client, sid)
        require(redo["state"] == "ready", "re-render after settings change failed")
    finally:
        teardown(app, root)


@check("a forgotten session answers 404 on every session route")
def t_forgotten_session_is_404():
    """Pin the contract the page uses to notice that a batch is gone.

    Batches live in memory, so a restarted or reaped server answers 404 for
    every session-scoped call. The front end reads exactly that as "this batch
    is gone" and tears itself down. If a route ever answered something else
    there - a 500, or a 400 - the page would keep a stale or blanked frame on
    screen instead of telling the user, which is the misreporting this guards.
    """
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black"}).get_json()["id"]
        ready = client.get(f"/api/sessions/{sid}/preview?index=0&view=audit")
        require(ready.status_code == 200, "preview should work before the session is gone")

        # A session that exists but stages nothing is not the same as one that is
        # gone. It must not answer 404, because the page would tear itself down.
        empty = app.store.create()
        blank = client.get(f"/api/sessions/{empty.id}/preview?index=0&view=result")
        require(
            blank.status_code == 409,
            f"a live session with nothing staged should be 409, got {blank.status_code}",
        )
        app.store.discard(empty.id)

        # Forget it the way a restart does.
        require(client.delete(f"/api/sessions/{sid}").status_code < 400, "discard failed")
        require(app.store.get(sid) is None, "session should be gone")

        calls = (
            ("get", f"/api/sessions/{sid}"),
            ("get", f"/api/sessions/{sid}/preview?index=0&view=result"),
            ("post", f"/api/sessions/{sid}/settings"),
            ("post", f"/api/sessions/{sid}/render"),
        )
        for method, path in calls:
            response = getattr(client, method)(path)
            require(
                response.status_code == 404,
                f"{method.upper()} {path} should be 404 once the session is gone, "
                f"got {response.status_code}",
            )
    finally:
        teardown(app, root)


@check("detection-setting changes drop the detection cache")
def t_detection_cache_invalidation():
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black", "preset": "balanced"}).get_json()["id"]
        session = app.store.get(sid)

        client.get(f"/api/sessions/{sid}/preview?index=0")
        require(session._detections, "preview should have populated the cache")

        same = client.post(
            f"/api/sessions/{sid}/settings", json={"mode": "blur", "preset": "balanced"}
        ).get_json()
        require(not same["detection_changed"], "mode is not a detection setting")
        require(session._detections, "cache should survive a cosmetic change")

        moved = client.post(
            f"/api/sessions/{sid}/settings", json={"preset": "aggressive"}
        ).get_json()
        require(moved["detection_changed"], "preset change must be detected")
        require(not session._detections, "stale detections were not dropped")
    finally:
        teardown(app, root)


@check("unsupported files are reported per-item, not as a batch failure")
def t_bad_file_isolation():
    app, root = make_app()
    try:
        client = app.test_client()
        response = upload(client, [LENA, None], names=["good.jpg", "notes.txt"])
        require(response.status_code == 201, "batch with a bad file should still stage")
        sid = response.get_json()["id"]

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"good file should render: {snapshot['error']}")

        by_name = {item["name"]: item for item in snapshot["items"]}
        require(by_name["notes.txt"]["status"] == "error", "bad file should be an error item")
        require(by_name["good.jpg"]["status"] == "done", "good file should be done")
        require(
            snapshot["result"]["name"].endswith(".jpg"),
            "one surviving output should not be wrapped in a zip",
        )
    finally:
        teardown(app, root)


@check("the multipart spool survives startup and a real upload")
def t_spool_survives_startup():
    """Werkzeug spools multipart parts to a temp file inside our own root.

    The orphan purge runs at startup and used to delete that directory, which
    made every upload fail with a 500 - and only ever appeared as "I dropped
    photos and nothing happened".
    """
    app, root = make_app()
    big = BASE_TMP / f"big-{uuid.uuid4().hex[:6]}.jpg"
    try:
        require((root / ".spool").is_dir(), "spool dir was not created at startup")

        # Comfortably past Flask's in-memory form limit, so the parser really
        # does roll over to a file on disk.
        rng = np.random.default_rng(7)
        noise = rng.integers(0, 256, (1400, 1400, 3), dtype=np.uint8)
        Image.fromarray(noise).save(big, "JPEG", quality=95)
        require(big.stat().st_size > 600_000, "fixture should force a spool rollover")

        client = app.test_client()
        response = upload(client, [big])
        require(response.status_code == 201, f"upload returned {response.status_code}")

        snapshot = response.get_json()
        require(len(snapshot["items"]) == 1, "the large photo should be staged")
        require(
            snapshot["items"][0]["status"] != "error",
            f"large photo failed to stage: {snapshot['items'][0].get('error')}",
        )
        require((root / ".spool").is_dir(), "the spool must still exist after an upload")
    finally:
        big.unlink(missing_ok=True)
        teardown(app, root)


@check("orphan purge clears sessions but spares reserved folders")
def t_purge_spares_reserved():
    app, root = make_app()
    try:
        orphan = root / "1700000000-deadbeef00"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "leftover.jpg").write_bytes(b"x" * 32)
        (root / ".spool").mkdir(exist_ok=True)

        removed = app.store.purge_orphans()
        require(removed >= 1, "the orphaned session dir should be purged")
        require(not orphan.exists(), "orphaned session dir survived the purge")
        require((root / ".spool").is_dir(), "the reserved spool dir must survive")
    finally:
        teardown(app, root)


@check("deleting a session frees its disk")
def t_delete_frees_disk():
    app, root = make_app()
    try:
        client = app.test_client()
        batch = [LENA, build_group_photo()]
        sid = upload(client, batch).get_json()["id"]

        before = client.get("/api/usage").get_json()
        require(before["on_disk_bytes"] > 0, "expected staged bytes on disk")

        removed = client.delete(f"/api/sessions/{sid}").get_json()
        require(removed["freed_bytes"] > 0, "delete reported nothing freed")
        require(removed["usage"]["sessions"] == 0, "session still tracked")
        require(
            removed["usage"]["on_disk_bytes"] == 0,
            f"bytes remain on disk: {removed['usage']}",
        )
    finally:
        teardown(app, root)


@check("reaper clears idle sessions")
def t_reaper():
    app, root = make_app()
    store = app.store
    try:
        client = app.test_client()
        sid = upload(client, [LENA]).get_json()["id"]
        store.ttl = 0  # everything is instantly idle
        count, freed = store.reap()
        require(count >= 1, "reaper found nothing")
        require(freed > 0, "reaper freed nothing")
        require(store.get(sid) is None, "session survived the reaper")
    finally:
        teardown(app, root)


@check("previews never expose a stale frame after a margin change")
def t_preview_reflects_margin():
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black", "margin": 0.02}).get_json()["id"]

        tight = client.get(f"/api/sessions/{sid}/preview?index=0&view=result").get_json()
        client.post(f"/api/sessions/{sid}/settings", json={"mode": "black", "margin": 0.5})
        wide = client.get(f"/api/sessions/{sid}/preview?index=0&view=result").get_json()

        require(tight["image"] != wide["image"], "preview ignored the margin change")
        require(wide["margin"] == 0.5, f"margin not echoed back: {wide['margin']}")
    finally:
        teardown(app, root)


# ---------------------------------------------------------------------------


def main() -> int:
    print("AmurCloak API tests\n" + "-" * 46)
    for name, fn in TESTS:
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))

    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      {detail}")

    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print("-" * 46)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
