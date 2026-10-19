"""Preview rendering and reporting.

Kept separate from :mod:`engine.sessions` because these are pure functions -
pixels in, pixels out - which makes them trivial to reason about and test.

Truthfulness rule: previews are composed at **full resolution** and only
downscaled for transport. Rendering a preview from an already-shrunk image would
change the blur kernel and pixelation block size, so the preview could disagree
with the delivered file. For a privacy tool that is unacceptable: the map has to
be exactly what you get.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from . import redact
from .detector import Face
from .photo import Photo
from .presets import Settings

CHECKER = 16
CHECKER_LIGHT = 232
CHECKER_DARK = 198

PREVIEW_QUALITY = 84
PREVIEW_MAX_EDGE = 1400


def flatten_alpha(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Composite transparency over a checkerboard so JPEG can show it honestly.

    Built from broadcast 1-D ranges rather than ``np.mgrid``: mgrid on a 24MP
    image allocates two int64 planes (~380MB) and would blow the RAM budget for
    a purely cosmetic step.
    """
    if alpha is None:
        return bgr
    h, w = alpha.shape[:2]
    ys = (np.arange(h, dtype=np.uint8) // CHECKER) % 2
    xs = (np.arange(w, dtype=np.uint8) // CHECKER) % 2
    board = (ys[:, None] ^ xs[None, :]).astype(np.float32)
    background = CHECKER_DARK + board * (CHECKER_LIGHT - CHECKER_DARK)

    weight = (alpha.astype(np.float32) / 255.0)[..., None]
    composited = bgr.astype(np.float32) * weight + background[..., None] * (1.0 - weight)
    return np.clip(composited, 0, 255).astype(np.uint8)


def downscale(image: np.ndarray, max_edge: int = PREVIEW_MAX_EDGE) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def compose_preview(
    decoded: Photo,
    faces: list[Face],
    settings: Settings,
    view: str = "result",
    max_edge: int = PREVIEW_MAX_EDGE,
    *,
    restored: Sequence[Face] = (),
) -> np.ndarray:
    """Full-resolution preview, downscaled for transport.

    ``view='audit'`` outlines every detection so the user can verify nothing was
    missed; ``view='result'`` shows the actual redaction.

    ``faces`` is what will be masked. ``restored`` is only for the audit map,
    which marks the people the user chose to leave visible.
    """
    if view == "audit":
        composed = redact.render_audit(decoded.bgr, faces, settings.margin, restored)
    else:
        composed = decoded.bgr.copy()
        redact.apply_redaction(
            composed,
            faces,
            settings.mode,
            settings.margin,
            settings.strength,
        )
    return downscale(flatten_alpha(composed, decoded.alpha), max_edge)


def encode_preview(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY]
    )
    if not ok:
        raise RuntimeError("preview encode failed")
    return buffer.tobytes()


def render_preview_bytes(
    decoded: Photo,
    faces: list[Face],
    settings: Settings,
    view: str = "result",
    max_edge: int = PREVIEW_MAX_EDGE,
    *,
    restored: Sequence[Face] = (),
) -> bytes:
    return encode_preview(
        compose_preview(decoded, faces, settings, view, max_edge, restored=restored)
    )


def scene_report(
    faces: list[Face],
    settings: Settings,
    decoded: Photo | None = None,
    restored_keys: frozenset[tuple[int, int, int, int]] = frozenset(),
) -> dict:
    """Everything the UI needs to describe the scene in words and numbers.

    ``faces`` is every detection. ``restored_keys`` are the ones the user chose to
    leave visible, so each entry can say whether the export will cover it - the
    UI draws its click targets straight from this, which is what keeps the
    picture on screen and the file on disk in agreement.
    """
    strong = sum(1 for f in faces if f.score >= 0.65)
    weak = len(faces) - strong

    entries = []
    for face in faces:
        key = face.key()
        mark = redact.ellipse_from_face(face, settings.margin)
        entry = face.to_dict()
        entry["key"] = list(key)
        entry["covered"] = key not in restored_keys
        # The mask's real geometry, so the browser can lay a click target exactly
        # over the shape that was painted instead of approximating a box.
        entry["mask"] = {
            "cx": round(mark.cx, 1),
            "cy": round(mark.cy, 1),
            "ax": round(mark.ax, 1),
            "ay": round(mark.ay, 1),
            "angle": round(mark.angle, 2),
        }
        entries.append(entry)

    restored = sum(1 for entry in entries if not entry["covered"])
    report = {
        "count": len(faces),
        "covered": len(faces) - restored,
        "restored": restored,
        "strong": strong,
        "weak": weak,
        "note": redact.confidence_note(faces),
        "faces": entries,
        "mode": settings.mode,
        "margin": settings.margin,
        "strength": settings.strength,
    }
    if decoded is not None:
        report["width"] = decoded.width
        report["height"] = decoded.height
        report["megapixels"] = round(decoded.width * decoded.height / 1e6, 1)
        report["has_alpha"] = decoded.alpha is not None
    return report
