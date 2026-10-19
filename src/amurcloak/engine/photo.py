"""Image loading and saving.

Two jobs beyond plain decode/encode, both of which matter more than they look:

1. **EXIF orientation.** Phone photos store pixels sideways plus a tag saying how
   to rotate them. OpenCV ignores that tag, so if you skip this step every
   portrait shot gets its mask in the wrong place. We normalise the pixels and
   drop the tag.

2. **Metadata stripping.** A perfectly blacked-out face can still ship GPS
   coordinates, device serial and exact capture time in EXIF. Redacting the face
   and keeping the metadata is a false sense of security, so metadata is removed
   by default.

Internally the engine speaks **BGR** (OpenCV's convention) so detection never
needs a colour conversion. That is the only boundary where channel order is
translated.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.TiffImagePlugin import IFDRational

from . import presets

try:  # iPhone photos are HEIC; without this most phone libraries simply cannot open.
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # noqa: BLE001  # pragma: no cover - optional dependency
    HEIF_AVAILABLE = False


# A 200MP ceiling exists purely as a decompression-bomb guard. Real photos are
# far below it; a malicious file that claims 10 gigapixels would otherwise be
# decoded straight into RAM.
MAX_PIXELS = 200_000_000

EXIF_ORIENTATION = 0x0112
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_DATETIME = 0x0132
EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_DATETIME_DIGITIZED = 0x9004
EXIF_GPS_IFD = 0x8825
EXIF_EXIF_IFD = 0x8769

# IFD0 tags a camera fills in that a script forgets. Their absence is one of the
# quiet giveaways that a block was written by hand.
EXIF_X_RESOLUTION = 0x011A
EXIF_Y_RESOLUTION = 0x011B
EXIF_RESOLUTION_UNIT = 0x0128
EXIF_YCBCR_POSITIONING = 0x0213

# Sub-tags inside the Exif IFD (a third directory, hung off EXIF_EXIF_IFD). These
# are where a real phone records *how* the shot was taken.
EXIF_EXPOSURE_TIME = 0x829A
EXIF_FNUMBER = 0x829D
EXIF_EXPOSURE_PROGRAM = 0x8822
EXIF_ISO = 0x8827
EXIF_EXIF_VERSION = 0x9000
EXIF_OFFSET_TIME = 0x9010
EXIF_OFFSET_TIME_ORIGINAL = 0x9011
EXIF_OFFSET_TIME_DIGITIZED = 0x9012
EXIF_COMPONENTS = 0x9101
EXIF_SHUTTER_SPEED = 0x9201
EXIF_APERTURE = 0x9202
EXIF_EXPOSURE_BIAS = 0x9204
EXIF_METERING = 0x9207
EXIF_FLASH = 0x9209
EXIF_FOCAL_LENGTH = 0x920A
EXIF_COLOR_SPACE = 0xA001
EXIF_PIXEL_X = 0xA002
EXIF_PIXEL_Y = 0xA003
EXIF_SENSING_METHOD = 0xA217
EXIF_SCENE_TYPE = 0xA301
EXIF_EXPOSURE_MODE = 0xA402
EXIF_WHITE_BALANCE = 0xA403
EXIF_FOCAL_35MM = 0xA405
EXIF_LENS_SPEC = 0xA432
EXIF_LENS_MAKE = 0xA433
EXIF_LENS_MODEL = 0xA434

# Sub-tags inside the GPS IFD (a second tag directory hung off EXIF_GPS_IFD).
GPS_LAT_REF = 1
GPS_LAT = 2
GPS_LON_REF = 3
GPS_LON = 4
GPS_TIME = 7
GPS_DATE = 0x001D

# ---------------------------------------------------------------------------
# Forged metadata (opt-in - see presets.METADATA_POLICIES)
# ---------------------------------------------------------------------------
# What a spoofed file claims to be. A phone is statistically the least
# remarkable thing a shared photo can say it came from.
SPOOF_MAKE = "Apple"
SPOOF_MODEL = "iPhone 13"

# The rest of what an iPhone 13 writes about the exposure. Individually trivial,
# but collectively they are the difference between a block that reads as a photo
# and one that reads as a stub: a real file states its lens, aperture and ISO, and
# "Apple / iPhone 13" followed by nothing at all is its own kind of flag.
SPOOF_EXIF_VERSION = b"0232"  # Exif 2.32, the revision iOS 16-era iPhones write
SPOOF_LENS_MAKE = "Apple"
SPOOF_LENS_MODEL = "iPhone 13 back dual wide camera 5.1mm f/1.6"
SPOOF_FOCAL_MM = IFDRational(51, 10)  # 5.1 mm
SPOOF_FOCAL_35MM = 26  # 35 mm equivalent, the iPhone 13 wide lens
SPOOF_FNUMBER = IFDRational(16, 10)  # f/1.6
SPOOF_EXPOSURE_TIME = IFDRational(1, 60)
SPOOF_ISO = 125
SPOOF_TZ_OFFSET = "+00:00"
SPOOF_DPI = 72

# Point Nemo - 48 52 36 S, 123 23 36 W. The oceanic pole of inaccessibility, and
# the furthest you can get from land anywhere on Earth (~2,688 km).
#
# Fixed, and deliberately not a setting. A forged timestamp is harmless, because
# a wrong time is only a wrong time. A forged *location* is a claim about place,
# and a place can be somebody's front door. Locking it to the emptiest point on
# the planet keeps the feature useful for hiding where you were while making it
# structurally impossible for it to name a home or a workplace.
SPOOF_GPS_LAT = (48, 52, 36)  # degrees, minutes, seconds
SPOOF_GPS_LAT_REF = "S"
SPOOF_GPS_LON = (123, 23, 36)
SPOOF_GPS_LON_REF = "W"

SUPPORTED_IN = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}

# Formats we write back out. TIFF is preserved because it is used for lossless
# archival work; everything else lossy collapses to JPEG.
_LOSSLESS_PRESERVE = {".png", ".tif", ".tiff"}


class PhotoError(Exception):
    """A photo could not be read or written. Message is user-facing."""


class PhotoTooLargeError(PhotoError):
    """The image is larger than the decoding ceiling."""


class UnsupportedFormatError(PhotoError):
    """The file's format is not one we can read or write."""


@dataclass
class Photo:
    bgr: np.ndarray  # HxWx3 uint8
    alpha: np.ndarray | None  # HxW uint8, or None
    icc: bytes | None
    source_ext: str
    width: int
    height: int


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_IN


def output_extension(source_ext: str, has_alpha: bool) -> str:
    """Decide the output format.

    Alpha forces PNG (JPEG cannot carry it). PNG and TIFF stay lossless so we
    never re-compress an already-lossless file. Everything else becomes JPEG -
    HEIC included, because JPEG opens everywhere and HEIC output surprises
    people who expected a file they could attach to anything.
    """
    source_ext = source_ext.lower()
    if has_alpha:
        return ".png"
    if source_ext in _LOSSLESS_PRESERVE:
        return source_ext
    return ".jpg"


def output_name(source_name: str, has_alpha: bool) -> str:
    p = Path(source_name)
    return f"{p.stem}{output_extension(p.suffix, has_alpha)}"


def unique_output_name(source_name: str, has_alpha: bool, taken: set[str]) -> str:
    """Avoid collisions when a batch mixes ``holiday.jpg`` and ``holiday.png``."""
    base = output_name(source_name, has_alpha)
    if base not in taken:
        taken.add(base)
        return base
    stem, ext = Path(base).stem, Path(base).suffix
    n = 2
    while f"{stem}-{n}{ext}" in taken:
        n += 1
    candidate = f"{stem}-{n}{ext}"
    taken.add(candidate)
    return candidate


def load_photo(path: str | Path) -> Photo:
    """Decode to BGR uint8, applying EXIF orientation and splitting alpha."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_IN:
        raise UnsupportedFormatError(f"{ext or 'unknown'} files are not supported")

    try:
        with Image.open(path) as im:
            if im.width * im.height > MAX_PIXELS:
                raise PhotoTooLargeError(
                    f"{im.width}x{im.height} exceeds the {MAX_PIXELS // 1_000_000}MP limit"
                )

            icc = im.info.get("icc_profile")

            # Only pay for the transpose when the tag actually says to rotate.
            orientation = 1
            try:
                orientation = im.getexif().get(EXIF_ORIENTATION, 1) or 1
            except Exception:  # noqa: BLE001 - a broken EXIF block just means "do not rotate"
                orientation = 1

            # exif_transpose hands back a new image; the original stays owned by
            # the ``with`` above and is closed when that block exits.
            upright = ImageOps.exif_transpose(im) if orientation != 1 else im

            has_alpha = upright.mode in ("RGBA", "LA", "PA") or (
                upright.mode == "P" and "transparency" in upright.info
            )
            alpha = None
            if has_alpha:
                rgba = upright.convert("RGBA")
                alpha = np.array(rgba.getchannel("A"))
                rgb = np.asarray(rgba.convert("RGB"))
            else:
                rgb = np.asarray(upright.convert("RGB"))

            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    except PhotoError:
        raise
    except Image.DecompressionBombError as exc:
        raise PhotoTooLargeError(str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedFormatError(f"could not decode {path.name}") from exc
    except Exception as exc:
        # Anything the decoder can throw beyond the two cases above becomes a
        # PhotoError, so a caller only ever has one failure type to handle.
        raise PhotoError(f"could not read {path.name}: {exc}") from exc

    h, w = bgr.shape[:2]
    return Photo(bgr=bgr, alpha=alpha, icc=icc, source_ext=ext, width=w, height=h)


def _dms(parts: tuple[int, int, int]) -> list[IFDRational]:
    """Degrees/minutes/seconds as the three rationals EXIF expects."""
    return [IFDRational(part, 1) for part in parts]


def _apex(value: float) -> IFDRational:
    """A plain ratio in its APEX form, which is how EXIF stores exposure values.

    APEX is a log2 scale, so these have to be *derived* from the exposure time
    and f-number rather than guessed: a file whose ``FNumber`` and
    ``ApertureValue`` disagree is one that contradicts itself, and a contradiction
    is the loudest thing a forged block can carry.
    """
    return IFDRational(round(value * 1000), 1000)


def _forge_exif_ifd(exif: Image.Exif, stamp: str, size: tuple[int, int]) -> None:
    """Fill the Exif sub-directory: everything about the shot but the location."""
    sub = exif.get_ifd(EXIF_EXIF_IFD)
    sub[EXIF_EXIF_VERSION] = SPOOF_EXIF_VERSION
    sub[EXIF_DATETIME_ORIGINAL] = stamp
    sub[EXIF_DATETIME_DIGITIZED] = stamp
    sub[EXIF_OFFSET_TIME] = SPOOF_TZ_OFFSET
    sub[EXIF_OFFSET_TIME_ORIGINAL] = SPOOF_TZ_OFFSET
    sub[EXIF_OFFSET_TIME_DIGITIZED] = SPOOF_TZ_OFFSET
    sub[EXIF_EXPOSURE_TIME] = SPOOF_EXPOSURE_TIME
    sub[EXIF_FNUMBER] = SPOOF_FNUMBER
    # -log2(seconds) and 2*log2(f), the two APEX definitions.
    sub[EXIF_SHUTTER_SPEED] = _apex(-math.log2(float(SPOOF_EXPOSURE_TIME)))
    sub[EXIF_APERTURE] = _apex(2.0 * math.log2(float(SPOOF_FNUMBER)))
    sub[EXIF_EXPOSURE_PROGRAM] = 2  # normal program
    sub[EXIF_EXPOSURE_BIAS] = IFDRational(0, 1)
    sub[EXIF_ISO] = SPOOF_ISO
    sub[EXIF_METERING] = 5  # pattern
    sub[EXIF_FLASH] = 16  # off, did not fire
    sub[EXIF_FOCAL_LENGTH] = SPOOF_FOCAL_MM
    sub[EXIF_FOCAL_35MM] = SPOOF_FOCAL_35MM
    sub[EXIF_LENS_SPEC] = [
        IFDRational(154, 100),
        SPOOF_FOCAL_MM,
        SPOOF_FNUMBER,
        SPOOF_FNUMBER,
    ]
    sub[EXIF_LENS_MAKE] = SPOOF_LENS_MAKE
    sub[EXIF_LENS_MODEL] = SPOOF_LENS_MODEL
    # 65535 is "uncalibrated", which is what an iPhone actually writes when it
    # ships a Display P3 profile alongside it - so this agrees with the ICC
    # profile the caller may be keeping rather than contradicting it.
    sub[EXIF_COLOR_SPACE] = 65535
    # Only claim dimensions we were actually given. A zero here would be a claim
    # that the image is empty, which is worse than saying nothing.
    if size[0] and size[1]:
        sub[EXIF_PIXEL_X] = size[0]
        sub[EXIF_PIXEL_Y] = size[1]
    sub[EXIF_SENSING_METHOD] = 2  # one-chip colour area
    sub[EXIF_SCENE_TYPE] = b"\x01"  # directly photographed
    sub[EXIF_EXPOSURE_MODE] = 0  # auto
    sub[EXIF_WHITE_BALANCE] = 0  # auto
    sub[EXIF_COMPONENTS] = b"\x01\x02\x03\x00"  # Y, Cb, Cr, -


def _forge_gps_ifd(exif: Image.Exif, moment: datetime) -> None:
    """Fill the GPS sub-directory with the fixed mid-ocean point."""
    gps = exif.get_ifd(EXIF_GPS_IFD)
    gps[GPS_LAT_REF] = SPOOF_GPS_LAT_REF
    gps[GPS_LAT] = _dms(SPOOF_GPS_LAT)
    gps[GPS_LON_REF] = SPOOF_GPS_LON_REF
    gps[GPS_LON] = _dms(SPOOF_GPS_LON)
    gps[GPS_DATE] = moment.strftime("%Y:%m:%d")
    # GPSTimeStamp is defined as UTC and takes plain numbers, not a date string.
    gps[GPS_TIME] = _dms((moment.hour, moment.minute, moment.second))


def forged_exif(when: str, *, size: tuple[int, int] | None = None, nested: bool = True) -> bytes:
    """Build a deliberately false EXIF block out of nothing.

    Built from scratch and never copied from the source file. That is not
    tidiness: an EXIF block can carry a *thumbnail*, which is a small copy of the
    original photo. Copying the source tags would therefore ship the unredacted
    face inside the metadata of the redacted file. Starting from an empty block
    means there is no thumbnail, no maker notes, no XMP and no IPTC - only the
    claims below, which is also why the lie stays internally consistent.

    ``when`` is ``YYYY-MM-DDTHH:MM`` (see presets.normalize_when). Orientation is
    written as 1 because the pixels reaching us are already upright; anything
    else would make viewers rotate the export.

    ``size`` is the ``(width, height)`` of the image being written, recorded as
    the Exif image dimensions. Omitted only when the caller cannot know it.

    ``nested=False`` drops the two sub-directories, for containers that cannot
    hold a nested one. Compressed TIFF is the only such case: Pillow attaches a
    nested directory only through its own uncompressed writer, and libtiff
    rejects the offset it emits for a compressed file - which covers
    ``EXIFIFDOffset`` just as much as ``GPSIFDOffset``. Rather than change the
    encoding of the archival format, a spoofed TIFF keeps a flat IFD0: the camera,
    the resolution and all three timestamps, but no lens detail and no location.
    """
    try:
        moment = datetime.fromisoformat(when)
    except (TypeError, ValueError):
        moment = datetime.fromisoformat(presets.DEFAULT_SPOOF_WHEN)
    stamp = moment.strftime("%Y:%m:%d %H:%M:%S")

    exif = Image.Exif()
    exif[EXIF_MAKE] = SPOOF_MAKE
    exif[EXIF_MODEL] = SPOOF_MODEL
    exif[EXIF_ORIENTATION] = 1
    # A real camera states its own resolution, and a missing pair here is a
    # giveaway that nothing ever developed this file.
    exif[EXIF_X_RESOLUTION] = IFDRational(SPOOF_DPI, 1)
    exif[EXIF_Y_RESOLUTION] = IFDRational(SPOOF_DPI, 1)
    exif[EXIF_RESOLUTION_UNIT] = 2  # inches
    exif[EXIF_YCBCR_POSITIONING] = 1  # centred - JPEG default and what iOS writes
    # All three timestamps, set to the same instant: a file where they disagree
    # is a file that announces it was edited.
    exif[EXIF_DATETIME] = stamp

    if nested:
        _forge_exif_ifd(exif, stamp, size or (0, 0))
        _forge_gps_ifd(exif, moment)
    else:
        # No nested directory available, so the capture timestamps fall back to
        # IFD0 - off-spec, but readable, and better than losing them entirely.
        exif[EXIF_DATETIME_ORIGINAL] = stamp
        exif[EXIF_DATETIME_DIGITIZED] = stamp

    return exif.tobytes()


def save_photo(
    path: str | Path,
    bgr: np.ndarray,
    alpha: np.ndarray | None = None,
    icc: bytes | None = None,
    *,
    quality: int = 95,
    keep_icc: bool = True,
    metadata: str = presets.DEFAULT_METADATA,
    spoof_when: str = presets.DEFAULT_SPOOF_WHEN,
) -> None:
    """Write BGR pixels out with nothing but an optional colour profile and a policy.

    ``metadata="strip"`` (the default) writes no metadata at all. ICC is
    deliberately retained in that case only if the caller asks for it: iPhone
    photos are Display P3, and dropping the profile makes them assume sRGB and
    visibly desaturate. Pass ``keep_icc=False`` to drop it too. EXIF, XMP, IPTC,
    GPS and PNG text chunks never reach this function, because the pixels arrive
    as a raw numpy array.

    ``metadata="spoof"`` strips all of the above and then writes the forged
    block from :func:`forged_exif` in its place. That trades an honest absence
    for a false presence, which is why it is opt-in.
    """
    path = Path(path)
    ext = path.suffix.lower()
    profile = icc if (keep_icc and icc) else None
    height, width = bgr.shape[:2]
    exif = None
    if metadata == "spoof":
        # See forged_exif: a compressed TIFF can carry no nested directory at all.
        exif = forged_exif(
            spoof_when,
            size=(width, height),
            nested=ext not in (".tif", ".tiff"),
        )

    # Pillow calls len() on the exif argument, so passing None is not the same as
    # omitting it. Only hand it over when there is one to write.
    extra: dict = {"icc_profile": profile}
    if exif is not None:
        extra["exif"] = exif

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext in (".jpg", ".jpeg"):
            if alpha is not None:
                raise PhotoError("JPEG cannot store transparency")
            image.save(
                path,
                format="JPEG",
                quality=int(quality),
                subsampling=0,  # keep chroma; q95 + 4:4:4 is visually lossless
                optimize=True,
                progressive=True,
                **extra,
            )
        elif ext == ".png":
            # compress_level 9 costs seconds on a 12MP image for a few percent.
            # 6 is the pragmatic point on the curve.
            image.save(
                path,
                format="PNG",
                optimize=True,
                compress_level=6,
                **extra,
            )
        elif ext in (".tif", ".tiff"):
            image.save(
                path,
                format="TIFF",
                compression="tiff_deflate",
                **extra,
            )
        else:
            raise UnsupportedFormatError(f"cannot write {ext} output")
    except PhotoError:
        raise
    except Exception as exc:
        # As above: any encoder failure is reported as a PhotoError.
        raise PhotoError(f"could not write {path.name}: {exc}") from exc


def probe_metadata(path: str | Path) -> dict:
    """Report what identifying metadata a file carries, without decoding pixels.

    This exists so the UI can tell someone *why* stripping matters: "this photo
    has GPS coordinates" is far more persuasive than a checkbox.
    """
    info: dict = {
        "has_exif": False,
        "has_gps": False,
        "has_xmp": False,
        "has_iptc": False,
        "camera": None,
        "datetime": None,
        "icc": False,
    }
    # Best effort: any failure at all just means "no metadata to report".
    with contextlib.suppress(Exception), Image.open(Path(path)) as im:
        info["has_xmp"] = "xmp" in im.info
        info["has_iptc"] = "iptc" in im.info
        info["icc"] = bool(im.info.get("icc_profile"))
        try:
            exif = im.getexif()
        except Exception:  # noqa: BLE001 - a malformed EXIF block means "nothing to report"
            exif = None
        if exif:
            info["has_exif"] = len(exif) > 0
            make = exif.get(EXIF_MAKE)
            model = exif.get(EXIF_MODEL)
            if make or model:
                info["camera"] = " ".join(str(v).strip() for v in (make, model) if v)[:80]
            dt = exif.get(EXIF_DATETIME)
            if dt:
                info["datetime"] = str(dt)[:40]
            gps = exif.get_ifd(EXIF_GPS_IFD)
            info["has_gps"] = bool(gps)
    return info
