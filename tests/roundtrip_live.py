"""Manual end-to-end check against a running server, with real footage.

Not part of the suites: this drives a server that is already up, so it needs a
clip that only exists on a working machine. Kept because it is the only check
that exercises a real browser-shaped flow at full length.
"""

from __future__ import annotations

import json
import mimetypes
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000"
CLIP = Path(__file__).resolve().parent / "fixtures" / "gen" / "tears_of_steel_faces_15s.mp4"
SETTINGS = {
    "mode": "black",
    "preset": "balanced",
    "margin": 14,
    "strength": 70,
    "metadata": "strip",
    "keep_audio": True,
}


def api(path, data=None, headers=None):
    # Loopback only, and the base is a constant at the top of this file.
    req = urllib.request.Request(BASE + path, data=data, headers=headers or {})  # noqa: S310
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        raw = r.read()
        ctype = r.headers.get("Content-Type", "")
        return r.status, (json.loads(raw) if "json" in ctype else raw)


def upload():
    boundary = uuid.uuid4().hex
    body = bytearray()

    def field(name, value):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")

    field("settings", json.dumps(SETTINGS))
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="files"; filename="{CLIP.name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mimetypes.guess_type(CLIP.name)[0]}\r\n\r\n".encode())
    body.extend(CLIP.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    status, payload = api(
        "/api/sessions",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if status != 201:
        raise SystemExit(f"upload refused: {payload}")
    return payload["id"]


def main() -> None:
    sid = upload()
    print("session", sid)

    deadline = time.time() + 180
    while time.time() < deadline:
        # The preview request is what kicks the scan off, exactly as the UI does.
        _, prev = api(f"/api/sessions/{sid}/preview?index=0&view=result&t=0")
        _, snap = api(f"/api/sessions/{sid}")
        item = snap["items"][0]
        print("  preview", "scanning" if prev.get("scanning") else "ready",
              "| scan", item.get("scan_state"),
              f"{float(item.get('scan_progress') or 0):.0%}",
              "faces", item.get("faces"))
        if not prev.get("scanning"):
            break
        if item.get("scan_state") == "error":
            raise SystemExit(f"scan failed: {item.get('error')}")
        time.sleep(2)
    else:
        raise SystemExit("scan never finished")

    status, _ = api(f"/api/sessions/{sid}/render", data=b"", headers={"Content-Length": "0"})
    print("render accepted", status)

    deadline = time.time() + 300
    while time.time() < deadline:
        _, snap = api(f"/api/sessions/{sid}")
        print("  state", snap["state"], f"{float(snap.get('progress') or 0):.0%}")
        if snap["state"] == "ready":
            break
        if snap["state"] == "error":
            raise SystemExit(f"render failed: {snap.get('error')}")
        time.sleep(2)
    else:
        raise SystemExit("render never finished")

    status, blob = api(f"/api/sessions/{sid}/result")
    out = Path(tempfile.gettempdir()) / "roundtrip_result.mp4"
    out.write_bytes(blob)
    print("downloaded", status, out.stat().st_size, "bytes")
    print("session summary faces:", snap["items"][0].get("faces"))


if __name__ == "__main__":
    main()
