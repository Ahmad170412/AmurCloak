#!/usr/bin/env python
"""AmurCloak engine tests.

Run with:  .venv/bin/python tests/test_engine.py

Deliberately dependency-free (assert + a tiny runner) so the project keeps its
five-package footprint. These are not decoration: the two claims worth proving
are that tiled inference actually recovers small faces, and that metadata really
is gone from the output.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
# The package lives under src/, so that is what has to be importable here.
sys.path.insert(0, str(ROOT / "src"))

from amurcloak.engine import photo, redact  # noqa: E402
from amurcloak.engine.detector import FaceDetector  # noqa: E402
from amurcloak.engine.presets import build  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
GEN = FIXTURES / "gen"
LENA = FIXTURES / "lena.jpg"

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


def lena_bgr() -> np.ndarray:
    require(LENA.exists(), "tests/fixtures/lena.jpg missing")
    return cv2.imread(str(LENA))


def build_group_photo() -> Path:
    """A synthetic group photo: the same face at six sizes across a big canvas.

    Stands in for the real case that matters - a wide shot where a single
    downscaled inference pass loses the people at the back.
    """
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "group.jpg"
    if out.exists():
        return out

    base = lena_bgr()
    canvas = np.full((2000, 3000, 3), 24, dtype=np.uint8)
    placements = [
        (60, 60, 0.55),
        (700, 120, 0.40),
        (1250, 80, 0.30),
        (1700, 150, 0.22),
        (2150, 200, 0.16),
        (2620, 90, 0.12),
    ]
    for x, y, scale in placements:
        small = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        canvas[y : y + h, x : x + w] = small
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def build_rotated_exif_jpeg() -> Path:
    """A landscape crop carrying EXIF orientation 6, plus camera metadata."""
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "rotated_exif.jpg"
    if out.exists():
        return out
    base = Image.open(LENA).convert("RGB").crop((0, 64, 512, 448))  # 512x384
    exif = base.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display
    exif[0x010F] = "TestCam"
    exif[0x0110] = "Model X"
    exif[0x0132] = "2024:07:01 12:00:00"
    base.save(out, exif=exif, quality=95)
    return out


def build_alpha_png() -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "alpha.png"
    if out.exists():
        return out
    base = lena_bgr()
    rgba = cv2.cvtColor(base, cv2.COLOR_BGR2RGBA)
    circle = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.circle(circle, (256, 256), 220, 255, -1)
    rgba[:, :, 3] = circle
    cv2.imwrite(str(out), rgba)
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@check("single face is detected with high confidence")
def t_single():
    faces = FaceDetector(score=0.6, min_face=24).detect(lena_bgr())
    require(len(faces) >= 1, "expected at least one face")
    require(max(f.score for f in faces) > 0.8, "expected a confident detection")


@check("tiled inference recovers faces a single pass loses")
def t_tiling_recall():
    img = cv2.imread(str(build_group_photo()))
    h, w = img.shape[:2]
    print(f"      group photo {w}x{h}")

    plain = FaceDetector(score=0.45, min_face=16, detect_edge=1024, tiles=False)
    tiled = FaceDetector(score=0.45, min_face=16, detect_edge=1024, tiles=True)

    a = plain.detect(img)
    b = tiled.detect(img)
    print(f"      single-pass: {len(a)} faces | tiled: {len(b)} faces")
    print(f"      tiled views: {len(tiled._views(h, w))}")

    require(len(b) >= len(a), "tiling should never lose detections")
    require(len(b) >= 4, f"expected >=4 faces with tiling, got {len(b)}")
    require(len(b) > len(a), "tiling was expected to find strictly more here")


@check("black mode writes pure black over the whole face")
def t_black_is_black():
    img = lena_bgr()
    faces = FaceDetector(score=0.6).detect(img)
    require(faces, "no face found")
    out = img.copy()
    redact.apply_redaction(out, faces, "black", 0.14, 0.7)

    f = faces[0]
    patches = [
        out[int(f.cy), int(f.cx)],
        out[int(f.y + f.h * 0.25), int(f.cx)],
        out[int(f.y + f.h * 0.85), int(f.cx)],
    ]
    for px in patches:
        require(tuple(int(v) for v in px) == (0, 0, 0), f"pixel not black: {px}")


@check("pixelate and blur keep the photo's colours - they never go black")
def t_soft_modes_keep_colour():
    """Regression: the soft modes must resample the face, not darken it.

    The failure this guards against is pixelate/blur compositing against black,
    which still looks like a redaction but is really the black mode in disguise.
    """
    img = lena_bgr()
    faces = FaceDetector(score=0.6).detect(img)
    require(faces, "no face found")
    f = faces[0]

    ell = redact.ellipse_from_face(f, 0.14)
    x0, y0, x1, y1 = redact._rotated_bounds(ell, img.shape[1], img.shape[0])

    def patch_mean(src):
        cy, cx = int(f.cy), int(f.cx)
        return src[cy - 12 : cy + 12, cx - 12 : cx + 12].reshape(-1, 3).mean(axis=0)

    before = patch_mean(img)
    require(before.max() > 40, "fixture patch should carry real colour to preserve")

    for mode in ("pixelate", "blur"):
        out = img.copy()
        redact.apply_redaction(out, faces, mode, 0.14, 0.7)
        after = patch_mean(out)

        # Colour survives, and is emphatically not the black mode.
        require(after.max() > 40, f"{mode} produced a near-black patch: {after.round(1)}")
        drift = float(np.abs(after - before).max())
        require(
            drift < 70,
            f"{mode} shifted the colour too far ({drift:.0f}): "
            f"{before.round(1)} -> {after.round(1)}",
        )

        # It was genuinely redacted rather than passed through untouched.
        roi_before = img[y0:y1, x0:x1].astype(np.float32)
        roi_after = out[y0:y1, x0:x1].astype(np.float32)
        delta = float(np.abs(roi_after - roi_before).mean())
        require(delta > 3.0, f"{mode} left the face region essentially unchanged ({delta:.2f})")


@check("mask covers every facial landmark and most of the face box")
def t_coverage():
    img = lena_bgr()
    faces = FaceDetector(score=0.6).detect(img)
    f = faces[0]
    margin = 0.14
    ell = redact.ellipse_from_face(f, margin)

    # An inscribed ellipse cannot reach the corners of the box - that is
    # geometry, not a bug, and the corners are background/hair. What must hold
    # is that the actual facial features sit well inside the mask.
    require(f.landmarks is not None, "no landmarks to verify against")
    for lx, ly in f.landmarks:
        dx = (float(lx) - ell.cx) / ell.ax
        dy = (float(ly) - ell.cy) / ell.ay
        radius = math.hypot(dx, dy)
        require(radius <= 0.85, f"landmark outside mask (normalised r={radius:.2f})")

    height, width = img.shape[:2]
    x0, y0, x1, y1 = redact._rotated_bounds(ell, width, height)
    alpha = redact._alpha_mask((y1 - y0, x1 - x0), ell, (x0, y0))

    bx0 = max(0, int(f.x) - x0)
    by0 = max(0, int(f.y) - y0)
    patch = alpha[by0 : by0 + int(f.h), bx0 : bx0 + int(f.w)]
    coverage = float(patch.mean())
    print(f"      face-box coverage at margin {margin}: {coverage:.3f}")
    require(coverage >= 0.78, f"face-box coverage too low: {coverage:.3f}")


@check("pixelate and blur actually change pixels")
def t_modes_change_pixels():
    img = lena_bgr()
    faces = FaceDetector(score=0.6).detect(img)
    for mode in ("pixelate", "blur"):
        out = img.copy()
        redact.apply_redaction(out, faces, mode, 0.14, 0.7)
        diff = np.abs(out.astype(int) - img.astype(int)).mean()
        require(diff > 1.0, f"{mode} barely changed the image (mean delta {diff:.2f})")


@check("EXIF orientation is applied, matching Pillow's oracle")
def t_exif_orientation():
    path = build_rotated_exif_jpeg()
    with Image.open(path) as im:
        stored_size = im.size
        oracle = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))

    loaded = photo.load_photo(path)
    ours = cv2.cvtColor(loaded.bgr, cv2.COLOR_BGR2RGB)

    require(loaded.height > loaded.width, "portrait orientation was not applied")
    require(stored_size[0] > stored_size[1], "fixture should be stored landscape")
    require(ours.shape == oracle.shape, f"{ours.shape} != {oracle.shape}")
    require(
        np.array_equal(ours, oracle),
        "orientation-corrected pixels disagree with Pillow",
    )


@check("output carries no EXIF, GPS or camera metadata")
def t_metadata_stripped():
    path = build_rotated_exif_jpeg()
    before = photo.probe_metadata(path)
    require(before["has_exif"], "fixture should carry EXIF")

    loaded = photo.load_photo(path)
    out = GEN / "stripped.jpg"
    photo.save_photo(out, loaded.bgr, loaded.alpha, loaded.icc, quality=95)

    after = photo.probe_metadata(out)
    require(not after["has_exif"], f"EXIF survived: {after}")
    require(not after["has_gps"], f"GPS survived: {after}")
    require(not after["has_xmp"], f"XMP survived: {after}")
    require(after["camera"] is None, f"camera model survived: {after}")
    require(after["datetime"] is None, f"timestamp survived: {after}")


@check("alpha PNG survives the round trip")
def t_alpha():
    path = build_alpha_png()
    loaded = photo.load_photo(path)
    require(loaded.alpha is not None, "alpha channel was dropped on load")
    require(loaded.alpha.shape == (loaded.height, loaded.width), "alpha shape wrong")

    faces = FaceDetector(score=0.6).detect(loaded.bgr)
    redact.apply_redaction(loaded.bgr, faces, "black", 0.14, 0.7)

    out = GEN / "alpha_out.png"
    photo.save_photo(out, loaded.bgr, loaded.alpha, loaded.icc)
    reread = photo.load_photo(out)
    require(reread.alpha is not None, "alpha lost on write")
    require(
        np.array_equal(reread.alpha, loaded.alpha), "alpha values changed"
    )


@check("output format rules hold")
def t_output_rules():
    require(photo.output_extension(".png", False) == ".png")
    require(photo.output_extension(".jpg", False) == ".jpg")
    require(photo.output_extension(".jpg", True) == ".png", "alpha must force PNG")
    require(photo.output_extension(".heic", False) == ".jpg")
    require(photo.output_extension(".tiff", False) == ".tiff")
    taken: set[str] = set()
    a = photo.unique_output_name("holiday.jpg", False, taken)
    b = photo.unique_output_name("holiday.png", False, taken)
    require(a != b, f"name collision: {a} vs {b}")


@check("upscaled downscale guard rejects absurd dimensions")
def t_bomb_guard():
    old = photo.MAX_PIXELS
    photo.MAX_PIXELS = 1000
    try:
        photo.load_photo(LENA)
        raise AssertionError("expected PhotoTooLarge")
    except photo.PhotoTooLarge:
        pass
    finally:
        photo.MAX_PIXELS = old


@check("presets resolve, and defaults stay on the safe mode")
def t_presets():
    s = build("balanced")
    require(s.mode == "black", "default mode must be the irreversible one")
    require(s.strip_exif is True, "metadata stripping must be on by default")
    require(s.margin > 0, "default margin should expand the mask")
    for key in ("conservative", "balanced", "aggressive"):
        p = build(key)
        require(p.score > 0, key)
        require(p.min_face >= 12, key)
    # Untrusted input must not produce a weaker-than-black default.
    from amurcloak.engine.presets import from_payload

    s2 = from_payload({"mode": "nonsense", "margin": "999", "preset": "nope"})
    require(s2.mode == "black", "bad mode fell through to something unsafe")
    require(s2.margin <= 0.6, "margin was not clamped")


# ---------------------------------------------------------------------------


def main() -> int:
    print("AmurCloak engine tests\n" + "-" * 46)
    for name, fn in TESTS:
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {name}")
        if not ok:
            print(f"      {detail}")

    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print("-" * 46)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
