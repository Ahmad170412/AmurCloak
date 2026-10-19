#!/usr/bin/env python
"""End-to-end API tests for AmurCloak.

Run with:  .venv/bin/python tests/test_api.py

Exercises the real Flask app through its test client: upload, tune, preview,
render, poll, download, and the disk-reclamation behaviour.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
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


@check("a batch that loses all but one file at render still delivers that file")
def t_single_survivor_batch():
    # Both names pass the upload extension check, so the batch looks multi-file
    # and the renderer spools to a zip before it can know that one of them will
    # fail to decode. The survivor has to be handed back on its own regardless.
    app, root = make_app()
    try:
        client = app.test_client()
        response = upload(client, [LENA, None], names=["good.jpg", "broken.jpg"])
        require(response.status_code == 201, "batch with a bad file should still stage")
        sid = response.get_json()["id"]

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"good file should render: {snapshot['error']}")
        require(snapshot["result"]["ready"], "session says ready but no result file exists")
        require(snapshot["result"]["size"] > 0, "result has no size")
        require(
            not snapshot["result"]["name"].endswith(".zip"),
            "one surviving output should not be wrapped in a zip",
        )

        downloaded = client.get(f"/api/sessions/{sid}/result")
        require(downloaded.status_code == 200, f"download returned {downloaded.status_code}")
        require(downloaded.data[:2] == b"\xff\xd8", "survivor is not a JPEG")
        require(len(downloaded.data) > 500, "survivor looks empty")

        by_name = {item["name"]: item for item in snapshot["items"]}
        require(by_name["good.jpg"]["status"] == "done", "good file should be done")
        require(by_name["broken.jpg"]["status"] == "error", "undecodable file should be an error")
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
# leaving one person visible
# ---------------------------------------------------------------------------


def build_pair_png() -> Path:
    """Two faces, PNG, so the export's pixels can be compared exactly.

    PNG is the lossless path in and out, which makes "no faces covered" a claim
    worth testing literally rather than approximately.
    """
    gen = LENA.parent / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    out = gen / "pair.png"
    if out.exists():
        return out
    base = cv2.imread(str(LENA))
    width = base.shape[1]
    canvas = np.full((base.shape[0], width * 2, 3), 30, dtype=np.uint8)
    canvas[:, :width] = base
    canvas[:, width:] = base
    cv2.imwrite(str(out), canvas)
    return out


def mean_inside(img: np.ndarray, mask: dict) -> tuple[float, int]:
    """Brightness in the core of a published mask, and how many pixels that is.

    Sampled at 60% of the published ellipse so the anti-aliased edge of the
    painted mask never contributes. If the geometry the API hands the browser
    were wrong, these numbers could not come out right.
    """
    m = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        m,
        (round(mask["cx"]), round(mask["cy"])),
        (max(1, round(mask["ax"] * 0.6)), max(1, round(mask["ay"] * 0.6))),
        mask["angle"],
        0,
        360,
        255,
        -1,
    )
    count = int((m > 0).sum())
    if count == 0:
        return float("nan"), 0
    return float(img[m > 0].mean()), count


@check("leaving one person visible keeps their pixels and covers the others")
def t_restore_one():
    app, root = make_app()
    try:
        client = app.test_client()
        response = upload(client, [build_group_photo()], {"mode": "black", "preset": "balanced"})
        require(response.status_code == 201, "upload failed")
        sid = response.get_json()["id"]

        preview = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        faces = preview["faces"]
        require(len(faces) >= 2, f"need at least two faces to tell apart, got {len(faces)}")
        require(all(face["covered"] for face in faces), "everyone should start covered")
        require(preview["restored"] == 0, "nothing should start visible")
        require(preview["covered"] == len(faces), "the covered count is wrong")
        require(all(face["mask"]["ax"] > 0 for face in faces), "faces need clickable geometry")

        chosen = faces[0]
        response = client.post(
            f"/api/sessions/{sid}/faces",
            json={"index": 0, "key": chosen["key"], "covered": False},
        )
        require(response.status_code == 200, response.get_data(as_text=True))

        after = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        require(after["restored"] == 1, f"expected one person visible, got {after['restored']}")
        require(after["covered"] == len(faces) - 1, "the covered count should drop by one")
        marked = [face for face in after["faces"] if not face["covered"]]
        require(len(marked) == 1, "exactly one face should be marked visible")
        require(marked[0]["key"] == chosen["key"], "the wrong person was restored")

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"render failed: {snapshot['error']}")
        out = root / "one_kept.jpg"
        out.write_bytes(client.get(f"/api/sessions/{sid}/result").data)
        img = photo_io.load_photo(out).bgr

        for entry in after["faces"]:
            mean, count = mean_inside(img, entry["mask"])
            require(count > 50, f"mask too small to judge ({count} px)")
            if entry["covered"]:
                require(mean < 12, f"a covered face came out lit (mean {mean:.1f})")
            else:
                require(mean > 40, f"the restored face was still redacted (mean {mean:.1f})")
    finally:
        teardown(app, root)


@check("leaving everyone visible hands back the photo's own pixels")
def t_restore_everyone():
    app, root = make_app()
    try:
        client = app.test_client()
        pair = build_pair_png()
        response = upload(client, [pair], {"mode": "black", "preset": "balanced"})
        sid = response.get_json()["id"]

        preview = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        faces = preview["faces"]
        require(len(faces) == 2, f"expected the two faces, got {len(faces)}")

        for face in faces:
            response = client.post(
                f"/api/sessions/{sid}/faces",
                json={"index": 0, "key": face["key"], "covered": False},
            )
            require(response.status_code == 200, response.get_data(as_text=True))

        after = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        require(after["restored"] == 2 and after["covered"] == 0, "everyone should be visible")

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"render failed: {snapshot['error']}")
        require(snapshot["result"]["name"].endswith(".png"), "a PNG input should stay a PNG")

        out = root / "all_kept.png"
        out.write_bytes(client.get(f"/api/sessions/{sid}/result").data)
        require(
            np.array_equal(photo_io.load_photo(out).bgr, photo_io.load_photo(pair).bgr),
            "with nobody covered, the export should be the original pixels exactly",
        )
    finally:
        teardown(app, root)


@check("only a box the detector actually found can be restored")
def t_face_toggle_guards():
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black"}).get_json()["id"]
        faces_url = f"/api/sessions/{sid}/faces"

        # Nothing has been detected yet, so there is nothing to check a key against.
        response = client.post(faces_url, json={"index": 0, "key": [1, 2, 3, 4], "covered": False})
        require(response.status_code == 400, "restoring before detection should be refused")

        preview = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        key = preview["faces"][0]["key"]

        refused = [
            ({"index": 0, "key": [1, 2, 3, 4], "covered": False}, "an invented box"),
            ({"index": 0, "key": [1, 2, 3], "covered": False}, "a short key"),
            ({"index": 0, "key": key, "covered": "yes"}, "a non-boolean covered flag"),
            ({"index": 99, "key": key, "covered": False}, "an out-of-range index"),
            ({"key": key, "covered": False}, "a missing index"),
            ({"index": 0, "covered": False}, "a missing key"),
        ]
        for body, label in refused:
            response = client.post(faces_url, json=body)
            require(response.status_code == 400, f"{label} was accepted: {response.status_code}")

        ok = {"index": 0, "key": key, "covered": False}
        require(client.post(faces_url, json=ok).status_code == 200, "a real face was refused")
        require(
            client.get(f"/api/sessions/{sid}/preview?index=0").get_json()["restored"] == 1,
            "the first restore did not take",
        )
        # Idempotent: clicking the same person twice must not count them twice.
        require(client.post(faces_url, json=ok).status_code == 200, "the retry was refused")
        require(
            client.get(f"/api/sessions/{sid}/preview?index=0").get_json()["restored"] == 1,
            "a repeated restore was counted twice",
        )
        # And reversible.
        require(
            client.post(faces_url, json={**ok, "covered": True}).status_code == 200,
            "covering the person again was refused",
        )
        require(
            client.get(f"/api/sessions/{sid}/preview?index=0").get_json()["restored"] == 0,
            "covering again did not clear the restore",
        )
    finally:
        teardown(app, root)


@check("restoring a face discards a result rendered before it")
def t_restore_invalidates_result():
    # Same rule as a settings change, for the same reason: handing over pixels
    # that no longer match what the screen shows is a silent privacy failure.
    app, root = make_app()
    try:
        client = app.test_client()
        sid = upload(client, [LENA], {"mode": "black"}).get_json()["id"]
        preview = client.get(f"/api/sessions/{sid}/preview?index=0").get_json()
        key = preview["faces"][0]["key"]

        snapshot = render_and_wait(client, sid)
        require(snapshot["result"]["ready"], "no result to invalidate")
        # Note: no download here on purpose. A download is what frees the staged
        # inputs, and with them the detections a restore is checked against - so
        # the restore window closes exactly where the reason to use it does.

        response = client.post(
            f"/api/sessions/{sid}/faces",
            json={"index": 0, "key": key, "covered": False},
        )
        require(response.status_code == 200, response.get_data(as_text=True))

        after = client.get(f"/api/sessions/{sid}").get_json()
        require(not after["result"]["ready"], "a stale result survived a restore")
        require(after["state"] == "staged", f"state should be staged, got {after['state']}")
        require(
            client.get(f"/api/sessions/{sid}/result").status_code == 409,
            "the stale file stayed downloadable",
        )
        # The detections must survive the invalidation, or nobody could undo it.
        require(
            client.post(
                f"/api/sessions/{sid}/faces",
                json={"index": 0, "key": key, "covered": True},
            ).status_code
            == 200,
            "detections were lost when the result was dropped",
        )
    finally:
        teardown(app, root)


@check("forged metadata reaches the download, with no thumbnail")
def t_spoof_end_to_end():
    app, root = make_app()
    try:
        client = app.test_client()
        response = upload(
            client,
            [build_group_photo()],
            {"mode": "black", "metadata": "spoof", "spoof_when": "2024-07-01T09:30"},
        )
        require(response.status_code == 201, "upload failed")
        snapshot = response.get_json()
        sid = snapshot["id"]
        require(snapshot["settings"]["metadata"] == "spoof", "the upload dropped the policy")
        require(snapshot["settings"]["spoof_when"] == "2024-07-01T09:30", "the date was dropped")

        # The tuning route has to carry it too, since that is how the UI changes it.
        tuned = client.post(
            f"/api/sessions/{sid}/settings",
            json={"mode": "black", "metadata": "spoof", "spoof_when": "2024-07-01T09:30"},
        ).get_json()
        require(tuned["settings"]["spoof_when"] == "2024-07-01T09:30", "settings dropped the date")

        snapshot = render_and_wait(client, sid)
        require(snapshot["state"] == "ready", f"render failed: {snapshot['error']}")
        out = root / "spoofed.jpg"
        out.write_bytes(client.get(f"/api/sessions/{sid}/result").data)

        exif = Image.open(out).getexif()
        sub = exif.get_ifd(0x8769)
        require(exif.get(0x0132) == "2024:07:01 09:30:00", f"date: {exif.get(0x0132)!r}")
        require(sub.get(0x9003) == "2024:07:01 09:30:00", "DateTimeOriginal missing")
        require(sub.get(0x9004) == "2024:07:01 09:30:00", "DateTimeDigitized missing")
        require(exif.get(0x010F) == "Apple" and exif.get(0x0110) == "iPhone 13", "camera missing")
        require(sub.get(0x8827) and sub.get(0xA434), "the exposure and lens detail are missing")
        require(exif.get_ifd(0x8825).get(1) == "S", "the forged location is not Point Nemo")
        require(0x0201 not in exif and dict(exif.get_ifd(1)) == {}, "a thumbnail was written")

        probe = photo_io.probe_metadata(out)
        require(probe["has_gps"], "the probe should see the forged location")
        require(probe["camera"] == "Apple iPhone 13", f"camera: {probe['camera']!r}")
        require(probe["datetime"] == "2024:07:01 09:30:00", f"datetime: {probe['datetime']!r}")

        # And the point of the tool still holds: the faces are gone.
        img = photo_io.load_photo(out).bgr
        require((img.sum(axis=2) == 0).any(), "the redaction was lost while forging metadata")
    finally:
        teardown(app, root)


# ---------------------------------------------------------------------------
# The three front-end files have to agree with each other. Nothing catches a
# half-finished rename for us - the page still loads, it just quietly loses a
# control or drops a rail into the wrong grid cell - so check them against each
# other rather than by eye.


def front_end_assets():
    app, root = make_app()
    client = app.test_client()
    return (
        app,
        root,
        client.get("/").get_data(as_text=True),
        client.get("/static/app.js").get_data(as_text=True),
        client.get("/static/style.css").get_data(as_text=True),
    )


@check("every id the controller queries exists in the page")
def t_frontend_ids_line_up():
    app, root, html, js, _css = front_end_assets()
    try:
        looked_up = set(re.findall(r'\$\("#([a-z0-9-]+)"\)', js))
        require(
            len(looked_up) > 20,
            f"only found {len(looked_up)} lookups - did app.js change shape?",
        )
        missing = sorted(i for i in looked_up if f'id="{i}"' not in html)
        require(not missing, f"app.js queries ids the page does not define: {missing}")
    finally:
        teardown(app, root)


@check("every column the workspace declares is filled by a pane the page uses")
def t_frontend_grid_areas_are_filled():
    app, root, html, _js, css = front_end_assets()
    try:
        declared: set[str] = set()
        for rule in re.finditer(r"\.workspace\s*\{([^}]*)\}", css):
            for chunk in re.findall(r'grid-template-areas:\s*((?:"[^"]*"\s*)+)', rule.group(1)):
                for row in re.findall(r'"([^"]+)"', chunk):
                    declared.update(row.split())
        require(declared == {"batch", "stage", "mask"}, f"unexpected areas: {sorted(declared)}")

        claimed: dict[str, set[str]] = {}
        for rule in re.finditer(r"\.([a-z][a-z0-9-]*)\s*\{([^}]*)\}", css):
            area = re.search(r"grid-area:\s*([a-z][a-z0-9-]*)", rule.group(2))
            if area:
                claimed.setdefault(area.group(1), set()).add(rule.group(1))

        for area in sorted(declared):
            classes = claimed.get(area, set())
            require(classes, f"the workspace declares a {area!r} column that no rule fills")
            require(
                any(name in html for name in classes),
                f"the {area!r} column is claimed by {sorted(classes)}, none of which the page uses",
            )
    finally:
        teardown(app, root)


@check("the theme is settled before the first paint and in one place")
def t_theme_boots_before_the_first_paint():
    app, root, html, _js, _css = front_end_assets()
    try:
        script_at = html.find("amurcloak-theme")
        link_at = html.find("style.css")
        require(script_at != -1, "nothing on the page decides the theme")
        require(link_at != -1, "the stylesheet is not linked from the page")
        require(
            script_at < link_at,
            "the theme must be decided before the stylesheet, or the page flashes",
        )
        require('id="theme-toggle"' in html, "there is no way to change the theme")
        require('aria-pressed' in html, "the toggle does not expose its state")
        require(
            'content="light dark"' in html,
            "the page should tell the browser both schemes are legitimate",
        )
    finally:
        teardown(app, root)


@check("every custom property the stylesheet uses is defined, and dark overrides only")
def t_css_tokens_are_all_defined():
    app, root, _html, _js, css = front_end_assets()
    try:
        used = set(re.findall(r"var\((--[a-z0-9-]+)\)", css))
        defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
        missing = sorted(used - defined)
        require(not missing, f"custom properties used but never defined: {missing}")

        # The dark block is an override list. A token only it defines would be
        # undefined in the light theme, and a fallback would hide that forever.
        block_at = css.find(':root[data-theme="dark"]')
        require(block_at != -1, "there is no dark theme block")
        open_at = css.index("{", block_at)
        depth = 0
        dark_block = ""
        for index in range(open_at, len(css)):
            if css[index] == "{":
                depth += 1
            elif css[index] == "}":
                depth -= 1
                if depth == 0:
                    dark_block = css[open_at:index]
                    break
        require(dark_block, "the dark theme block is empty")
        dark_tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", dark_block))
        light_tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", css[:block_at]))
        only_dark = sorted(dark_tokens - light_tokens)
        require(not only_dark, f"tokens defined only for the dark theme: {only_dark}")
        require(
            len(dark_tokens) > 15,
            f"only {len(dark_tokens)} tokens in the dark theme - is it still an override list?",
        )
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
