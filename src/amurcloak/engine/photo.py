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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

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
EXIF_GPS_IFD = 0x8825

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


def save_photo(
    path: str | Path,
    bgr: np.ndarray,
    alpha: np.ndarray | None = None,
    icc: bytes | None = None,
    *,
    quality: int = 95,
    keep_icc: bool = True,
) -> None:
    """Write BGR pixels out with *no* metadata beyond an optional colour profile.

    ICC is deliberately retained: iPhone photos are Display P3, and dropping the
    profile makes them assume sRGB and visibly desaturate. An ICC profile is a
    colour-space descriptor, not identity data. Pass ``keep_icc=False`` to drop
    it too. EXIF, XMP, IPTC, GPS and PNG text chunks are always discarded - they
    never reach this function because the pixels arrive as a raw numpy array.
    """
    path = Path(path)
    ext = path.suffix.lower()
    profile = icc if (keep_icc and icc) else None

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
                icc_profile=profile,
            )
        elif ext == ".png":
            # compress_level 9 costs seconds on a 12MP image for a few percent.
            # 6 is the pragmatic point on the curve.
            image.save(
                path,
                format="PNG",
                optimize=True,
                compress_level=6,
                icc_profile=profile,
            )
        elif ext in (".tif", ".tiff"):
            image.save(path, format="TIFF", compression="tiff_deflate", icc_profile=profile)
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
