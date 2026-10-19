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
import struct
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
from amurcloak.engine.presets import build, from_payload  # noqa: E402

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
        raise AssertionError("expected PhotoTooLargeError")
    except photo.PhotoTooLargeError:
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
    s2 = from_payload({"mode": "nonsense", "margin": "999", "preset": "nope"})
    require(s2.mode == "black", "bad mode fell through to something unsafe")
    require(s2.margin <= 0.6, "margin was not clamped")


@check("a non-finite slider value degrades to the default, not into the mask")
def t_payload_rejects_non_finite():
    # float('nan') parses fine, so without an explicit guard it would flow into
    # the ellipse geometry and either raise mid-render or build a mask that
    # covers nothing. Both outcomes are unacceptable; the default is not.
    s = from_payload({"margin": "nan", "strength": "inf", "quality": "-inf"})
    require(math.isfinite(s.margin) and s.margin == 0.14, f"margin leaked as {s.margin}")
    require(math.isfinite(s.strength) and s.strength == 0.7, f"strength leaked as {s.strength}")
    require(math.isfinite(s.strength) and s.quality == 95, f"quality leaked as {s.quality}")


# ---------------------------------------------------------------------------
# forged metadata (active defence)
# ---------------------------------------------------------------------------

# EXIF/TIFF tag numbers used by the checks below.
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_ORIENTATION = 0x0112
TAG_X_RESOLUTION = 0x011A
TAG_Y_RESOLUTION = 0x011B
TAG_RESOLUTION_UNIT = 0x0128
TAG_DATETIME = 0x0132
TAG_THUMB_OFFSET = 0x0201
TAG_THUMB_LENGTH = 0x0202
TAG_YCBCR_POSITIONING = 0x0213
TAG_DATETIME_ORIGINAL = 0x9003
TAG_DATETIME_DIGITIZED = 0x9004
TAG_GPS_IFD = 0x8825
TAG_EXIF_IFD = 0x8769

# Tags that live in the Exif sub-directory.
TAG_EXPOSURE_TIME = 0x829A
TAG_FNUMBER = 0x829D
TAG_ISO = 0x8827
TAG_EXIF_VERSION = 0x9000
TAG_SHUTTER_APEX = 0x9201
TAG_APERTURE_APEX = 0x9202
TAG_FOCAL_LENGTH = 0x920A
TAG_COLOR_SPACE = 0xA001
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003
TAG_FOCAL_35MM = 0xA405
TAG_LENS_MAKE = 0xA433
TAG_LENS_MODEL = 0xA434

POINT_NEMO = ([48.0, 52.0, 36.0], [123.0, 23.0, 36.0])


def exif_with_thumbnail(make: str, model: str, when: str, thumb_jpeg: bytes) -> bytes:
    """A hand-built EXIF block carrying a real thumbnail of the original.

    Pillow will not write thumbnail tags for us - they need a resolved offset
    into the APP1 block - so the TIFF structure is assembled by hand. That makes
    it a faithful stand-in for a camera JPEG: a complete second JPEG stream
    (the unredacted preview) living inside the metadata.
    """
    ascii_type, short_type, long_type = 2, 3, 4

    def entry(tag: int, kind: int, count: int, value: bytes) -> bytes:
        return struct.pack("<HHI", tag, kind, count) + value

    def text_entry(tag: int, text: str, offset: int) -> bytes:
        raw = text.encode("ascii") + b"\x00"
        if len(raw) <= 4:
            return entry(tag, ascii_type, len(raw), raw.ljust(4, b"\x00"))
        return entry(tag, ascii_type, len(raw), struct.pack("<I", offset))

    make_b = make.encode("ascii") + b"\x00"
    model_b = model.encode("ascii") + b"\x00"
    when_b = when.encode("ascii") + b"\x00"

    def ifd_size(count: int) -> int:
        return 2 + 12 * count + 4

    ifd0_at = 8
    ifd1_at = ifd0_at + ifd_size(3)
    data_at = ifd1_at + ifd_size(3)
    make_at = data_at
    model_at = make_at + len(make_b)
    when_at = model_at + len(model_b)
    thumb_at = when_at + len(when_b)

    ifd0 = (
        struct.pack("<H", 3)
        + text_entry(TAG_MAKE, make, make_at)
        + text_entry(TAG_MODEL, model, model_at)
        + text_entry(TAG_DATETIME, when, when_at)
        + struct.pack("<I", ifd1_at)  # next IFD = the thumbnail directory
    )
    ifd1 = (
        struct.pack("<H", 3)
        + entry(0x0103, short_type, 1, struct.pack("<HH", 6, 0))  # JPEG
        + entry(TAG_THUMB_OFFSET, long_type, 1, struct.pack("<I", thumb_at))
        + entry(TAG_THUMB_LENGTH, long_type, 1, struct.pack("<I", len(thumb_jpeg)))
        + struct.pack("<I", 0)
    )
    tiff = (
        b"II"
        + struct.pack("<HI", 42, ifd0_at)
        + ifd0
        + ifd1
        + make_b
        + model_b
        + when_b
        + thumb_jpeg
    )
    return b"Exif\x00\x00" + tiff


def gps_degrees(gps: dict) -> tuple[list[float], list[float]]:
    lat = [float(part) for part in gps[2]]
    lon = [float(part) for part in gps[4]]
    return lat, lon


@check("spoofed output carries one consistent, thumbnail-free EXIF block")
def t_spoof_forges_the_block():
    height, width = lena_bgr().shape[:2]
    for ext in (".jpg", ".png", ".tif"):
        path = GEN / f"spoofed{ext}"
        photo.save_photo(
            path,
            lena_bgr(),
            None,
            None,
            quality=95,
            metadata="spoof",
            spoof_when="1453-05-05T06:00",
        )
        with Image.open(path) as out:
            exif = out.getexif()
            require("xmp" not in out.info, f"{ext}: XMP was written")

        # A container that can hold nested directories carries all three
        # timestamps the proper way, in the Exif IFD. A compressed TIFF cannot,
        # so it keeps them in IFD0 instead.
        nested = ext not in (".tif", ".tiff")
        sub = exif.get_ifd(TAG_EXIF_IFD) if nested else exif
        for tag, label in (
            (TAG_DATETIME, "DateTime"),
            (TAG_DATETIME_ORIGINAL, "DateTimeOriginal"),
            (TAG_DATETIME_DIGITIZED, "DateTimeDigitized"),
        ):
            holder = exif if tag == TAG_DATETIME else sub
            require(
                holder.get(tag) == "1453:05:05 06:00:00",
                f"{ext}: {label} was {holder.get(tag)!r}",
            )
        require(exif.get(TAG_MAKE) == "Apple", f"{ext}: make was {exif.get(TAG_MAKE)!r}")
        require(exif.get(TAG_MODEL) == "iPhone 13", f"{ext}: model was {exif.get(TAG_MODEL)!r}")
        require(
            exif.get(TAG_ORIENTATION) == 1,
            f"{ext}: orientation must be 1 so the upright pixels stay upright",
        )
        # A camera states its own resolution; a block missing it was written by a
        # script, which is half of what makes the forged version pass.
        require(exif.get(TAG_X_RESOLUTION) == 72, f"{ext}: XResolution missing")
        require(exif.get(TAG_Y_RESOLUTION) == 72, f"{ext}: YResolution missing")
        require(exif.get(TAG_RESOLUTION_UNIT) == 2, f"{ext}: ResolutionUnit missing")
        require(exif.get(TAG_YCBCR_POSITIONING) == 1, f"{ext}: YCbCrPositioning missing")

        gps = exif.get_ifd(TAG_GPS_IFD)
        if nested:
            require(dict(sub) != {}, f"{ext}: the Exif sub-directory was not written")
            require(sub.get(TAG_EXIF_VERSION) == b"0232", f"{ext}: ExifVersion missing")
            require(sub.get(TAG_ISO), f"{ext}: ISO missing")
            require(sub.get(TAG_FNUMBER), f"{ext}: FNumber missing")
            require(sub.get(TAG_EXPOSURE_TIME), f"{ext}: ExposureTime missing")
            require(
                "iPhone 13" in str(sub.get(TAG_LENS_MODEL)),
                f"{ext}: LensModel was {sub.get(TAG_LENS_MODEL)!r}",
            )
            require(sub.get(TAG_LENS_MAKE) == "Apple", f"{ext}: LensMake missing")
            require(
                sub.get(TAG_PIXEL_X) == width and sub.get(TAG_PIXEL_Y) == height,
                f"{ext}: Exif pixels {sub.get(TAG_PIXEL_X)}x{sub.get(TAG_PIXEL_Y)} "
                f"do not match the {width}x{height} image",
            )
            lat, lon = gps_degrees(gps)
            require(lat == POINT_NEMO[0], f"{ext}: latitude was {lat}")
            require(lon == POINT_NEMO[1], f"{ext}: longitude was {lon}")
            require(gps.get(1) == "S" and gps.get(3) == "W", f"{ext}: hemisphere refs missing")
            require(gps.get(0x001D) == "1453:05:05", f"{ext}: GPS date was {gps.get(0x001D)!r}")
        else:
            # Documented exception: Pillow attaches a nested directory only
            # through its own uncompressed writer, and libtiff rejects the offset
            # it emits for a compressed file - for EXIFIFDOffset just as much as
            # for GPSIFDOffset. Changing the archival format's encoding to plant a
            # fake place is not worth it, so a spoofed TIFF stays flat: the
            # camera, the resolution and the timestamps, and nothing deeper.
            require(not gps, f"{ext}: a compressed TIFF should carry no GPS at all")
            require(
                dict(exif.get_ifd(TAG_EXIF_IFD)) == {},
                f"{ext}: a compressed TIFF should carry no Exif sub-directory",
            )

        require(TAG_THUMB_OFFSET not in exif, f"{ext}: thumbnail pointer present")
        require(TAG_THUMB_LENGTH not in exif, f"{ext}: thumbnail length present")
        require(dict(exif.get_ifd(1)) == {}, f"{ext}: an IFD1 thumbnail directory was written")


@check("the forged block reads as a real capture, not an empty stub")
def t_spoof_looks_like_a_real_capture():
    path = GEN / "spoofed_plausible.jpg"
    photo.save_photo(
        path,
        lena_bgr(),
        None,
        None,
        quality=95,
        metadata="spoof",
        spoof_when="2024-07-01T09:30",
    )
    height, width = lena_bgr().shape[:2]
    with Image.open(path) as out:
        exif = out.getexif()
        sub = exif.get_ifd(TAG_EXIF_IFD)

    # The point of the exercise: everything a viewer asks about the shot is
    # answered, so nothing stands out as missing.
    for tag, label in (
        (TAG_EXIF_VERSION, "ExifVersion"),
        (TAG_ISO, "ISO"),
        (TAG_FNUMBER, "FNumber"),
        (TAG_EXPOSURE_TIME, "ExposureTime"),
        (TAG_SHUTTER_APEX, "ShutterSpeedValue"),
        (TAG_APERTURE_APEX, "ApertureValue"),
        (TAG_FOCAL_LENGTH, "FocalLength"),
        (TAG_FOCAL_35MM, "FocalLengthIn35mmFormat"),
        (TAG_COLOR_SPACE, "ColorSpace"),
        (TAG_LENS_MAKE, "LensMake"),
        (TAG_LENS_MODEL, "LensModel"),
        (TAG_PIXEL_X, "PixelXDimension"),
        (TAG_PIXEL_Y, "PixelYDimension"),
    ):
        require(sub.get(tag) is not None, f"{label} is missing from the forged block")

    require(sub.get(TAG_PIXEL_X) == width, "the recorded width is not the written width")
    require(sub.get(TAG_PIXEL_Y) == height, "the recorded height is not the written height")
    require(sub.get(TAG_COLOR_SPACE) == 65535, "an iPhone reports uncalibrated colour")
    require(sub.get(TAG_FOCAL_35MM) == 26, "the iPhone 13 wide lens is 26mm equivalent")

    # Self-consistency is the whole game. APEX stores shutter and aperture on a
    # log2 scale, so these have to be derived from ExposureTime and FNumber -
    # a block whose stated aperture and computed aperture disagree contradicts
    # itself, and that is the loudest thing a forgery can carry.
    exposure = float(sub.get(TAG_EXPOSURE_TIME))
    require(exposure > 0, "exposure time must be positive")
    require(
        abs(2.0 ** (-float(sub.get(TAG_SHUTTER_APEX))) - exposure) < 1e-3,
        "ShutterSpeedValue does not agree with ExposureTime",
    )
    fnumber = float(sub.get(TAG_FNUMBER))
    require(fnumber > 0, "f-number must be positive")
    require(
        abs(2.0 ** (float(sub.get(TAG_APERTURE_APEX)) / 2.0) - fnumber) < 1e-2,
        "ApertureValue does not agree with FNumber",
    )
    # And the lens string should describe the aperture it claims.
    require(f"f/{fnumber:.1f}" in str(sub.get(TAG_LENS_MODEL)), "the lens name disagrees")


@check("spoofing cannot smuggle the original photo back in as a thumbnail")
def t_spoof_never_carries_the_original():
    # This is the failure that would matter most: an EXIF thumbnail is a small
    # copy of the ORIGINAL image, so a spoofing pass that copied the source's
    # tags would ship the un-redacted face inside the "metadata" of the redacted
    # file. Build a genuine leaky source and prove none of it survives.
    source = GEN / "leaky_source.jpg"
    base = lena_bgr()
    thumb = cv2.resize(base, (80, 80), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    require(ok, "could not encode the thumbnail")
    thumb_bytes = buffer.tobytes()
    Image.open(LENA).convert("RGB").save(
        source,
        exif=exif_with_thumbnail(
            "Originalcam", "Secret Phone", "2011:03:04 05:06:07", thumb_bytes
        ),
        quality=90,
    )

    raw = source.read_bytes()
    require(raw.count(b"\xff\xd8\xff") == 2, "the source should carry a second JPEG stream")
    require(thumb_bytes in raw, "the source thumbnail should really be embedded")
    with Image.open(source) as im:
        source_exif = im.getexif()
    require(source_exif.get(TAG_MAKE) == "Originalcam", "the source should carry a real camera")

    redacted = np.full_like(base, 12)
    out = GEN / "spoofed_from_leaky.jpg"
    photo.save_photo(out, redacted, None, None, quality=95, metadata="spoof")

    data = out.read_bytes()
    require(
        data.count(b"\xff\xd8\xff") == 1,
        "the output must hold exactly one JPEG stream - the redacted one",
    )
    require(thumb_bytes not in data, "the original thumbnail leaked into the spoofed file")
    with Image.open(out) as im:
        exif = im.getexif()
        info = dict(im.info)
    require(TAG_THUMB_OFFSET not in exif and TAG_THUMB_LENGTH not in exif, "thumbnail written")
    require(dict(exif.get_ifd(1)) == {}, "IFD1 thumbnail directory written")
    require(exif.get(TAG_MAKE) == "Apple", "the forged camera should replace the real one")
    require(exif.get(TAG_DATETIME) == "1453:05:05 06:00:00", "forged timestamp missing")
    require("xmp" not in info and "exif" in info, "the forged block should be the only metadata")

    # Forging metadata must not resurrect the pixels underneath it.
    recovered = photo.load_photo(out).bgr.astype(int)
    require(
        abs(int(recovered.mean()) - 12) <= 3,
        f"the redacted pixels did not survive the metadata write: {recovered.mean():.1f}",
    )


@check("spoofing is opt-in, and a junk date degrades to the default")
def t_spoof_is_opt_in():
    # The default is still "no metadata at all" - the simpler, stronger promise.
    default_out = GEN / "stripped_by_default.jpg"
    photo.save_photo(default_out, lena_bgr(), None, None, quality=95)
    with Image.open(default_out) as im:
        require(not im.getexif(), "a default save grew an EXIF block")

    # An unknown policy is ignored rather than guessed at, and a date that cannot
    # be parsed keeps the default instead of inventing one.
    settings = from_payload({"metadata": "whatever", "spoof_when": "not-a-date"})
    require(settings.metadata == "strip", f"unknown policy honoured: {settings.metadata!r}")
    require(settings.spoof_when == "1453-05-05T06:00", f"junk date: {settings.spoof_when!r}")

    settings = from_payload({"metadata": "spoof", "spoof_when": "2024-07-01T09:30"})
    require(settings.metadata == "spoof", "a valid policy was dropped")
    require(settings.spoof_when == "2024-07-01T09:30", f"normalised to {settings.spoof_when!r}")
    # A date with no time is accepted, and lands at midnight.
    require(
        from_payload({"spoof_when": "2024-07-01"}).spoof_when == "2024-07-01T00:00",
        "a date-only value should be accepted at midnight",
    )
    # An impossible date is junk like any other.
    require(
        from_payload({"spoof_when": "1453-02-31"}).spoof_when == "1453-05-05T06:00",
        "31 February should not be accepted",
    )

    recent = GEN / "spoofed_recent.jpg"
    photo.save_photo(
        recent,
        lena_bgr(),
        None,
        None,
        quality=95,
        metadata=settings.metadata,
        spoof_when=settings.spoof_when,
    )
    with Image.open(recent) as im:
        require(im.getexif().get(TAG_DATETIME) == "2024:07:01 09:30:00", "chosen date not used")


@check("the audit map marks a kept face without washing it")
def t_audit_marks_kept_faces():
    base = lena_bgr()
    faces = FaceDetector(score=0.6, min_face=24).detect(base)
    require(faces, "expected a face to draw")

    plain = redact.render_audit(base, faces, 0.14)
    same = redact.render_audit(base, faces, 0.14, [])
    require(np.array_equal(plain, same), "an empty kept list must change nothing at all")

    face = faces[0]
    cx, cy = int(face.cx), int(face.cy)
    washed = redact.render_audit(base, faces, 0.14)
    kept = redact.render_audit(base, [], 0.14, faces)
    require(
        not np.array_equal(washed[cy, cx], base[cy, cx]),
        "a covered face should be washed in the audit map",
    )
    require(
        np.array_equal(kept[cy, cx], base[cy, cx]),
        "a kept face must not be washed - the export will not cover it",
    )
    require(
        not np.array_equal(kept, base),
        "a kept face should still be outlined so the user can see it was found",
    )


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
