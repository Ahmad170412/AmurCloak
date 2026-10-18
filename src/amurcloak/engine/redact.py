"""Redaction rendering.

Three modes with a real difference in strength, and the code does not pretend
otherwise:

* ``black``  - opaque ellipse. The pixels are destroyed. Nothing to recover.
* ``pixelate`` - mosaic blocks. Destroys most detail; AI un-pixelation exists.
* ``blur``  - Gaussian blur. Natural-looking, and partially invertible.

Geometry note: the mask is an **ellipse fitted and roll-aligned to the face**,
not a circle. A circle either clips the chin and forehead or balloons out over
the background; an ellipse built from the five YuNet landmarks follows a tilted
head properly, which matters as soon as someone cocks their head in a photo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Face

# A face is taller than it is wide, so the vertical margin expands further than
# the horizontal one. Without this the chin and hairline peek out of the mask.
VERTICAL_EXTRA = 1.3

# Beyond this the eye-line angle is more likely landmark noise than a real tilt,
# and a wrong angle would rotate the mask off the face entirely.
MAX_ROLL_DEG = 75.0

AUDIT_ACCENT = (32, 176, 255)  # BGR amber
AUDIT_BOX = (170, 170, 170)
AUDIT_INK = (16, 16, 16)


@dataclass
class Ellipse:
    cx: float
    cy: float
    ax: float
    ay: float
    angle: float


def ellipse_from_face(face: Face, margin: float) -> Ellipse:
    """Fit a redaction ellipse to a detection, including head roll."""
    ax = max(2.0, face.w / 2.0 * (1.0 + margin * 2.0))
    ay = max(2.0, face.h / 2.0 * (1.0 + margin * VERTICAL_EXTRA * 2.0))

    angle = 0.0
    landmarks = face.landmarks
    if landmarks is not None and len(landmarks) >= 2:
        (ex1, ey1), (ex2, ey2) = landmarks[0], landmarks[1]
        angle = math.degrees(math.atan2(float(ey2 - ey1), float(ex2 - ex1)))
        if not math.isfinite(angle) or abs(angle) > MAX_ROLL_DEG:
            angle = 0.0

    return Ellipse(face.cx, face.cy, ax, ay, angle)


def ellipses(faces: list[Face], margin: float) -> list[Ellipse]:
    return [ellipse_from_face(f, margin) for f in faces]


def _rotated_bounds(ell: Ellipse, width: int, height: int) -> tuple[int, int, int, int]:
    """Integer ROI covering the rotated ellipse, clipped to the image."""
    theta = math.radians(ell.angle)
    cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
    half_w = math.sqrt((ell.ax * cos_t) ** 2 + (ell.ay * sin_t) ** 2)
    half_h = math.sqrt((ell.ax * sin_t) ** 2 + (ell.ay * cos_t) ** 2)

    x0 = max(0, math.floor(ell.cx - half_w))
    y0 = max(0, math.floor(ell.cy - half_h))
    x1 = min(width, math.ceil(ell.cx + half_w))
    y1 = min(height, math.ceil(ell.cy + half_h))
    return x0, y0, x1, y1


def _alpha_mask(
    size: tuple[int, int], ell: Ellipse, offset: tuple[int, int]
) -> np.ndarray:
    """Antialiased coverage mask in [0,1] for one ellipse within a ROI."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (round(ell.cx - offset[0]), round(ell.cy - offset[1]))
    axes = (max(1, round(ell.ax)), max(1, round(ell.ay)))

    # Filled hard body, then an antialiased rim. The rim is what removes the
    # stair-stepping a plain filled ellipse leaves on a large mask, and because
    # the rim can only add coverage it never exposes original pixels.
    cv2.ellipse(mask, center, axes, ell.angle, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(mask, center, axes, ell.angle, 0, 360, 255, 2, cv2.LINE_AA)

    return mask.astype(np.float32) / 255.0


def _odd(value: int, low: int = 15, high: int = 201) -> int:
    value = max(low, min(high, value))
    return value if value % 2 == 1 else value + 1


def _blur_kernel(face: Face, strength: float) -> int:
    factor = 0.20 + 0.60 * strength
    return _odd(round(face.short_side * factor))


def _pixelate_roi(roi: np.ndarray, face: Face, strength: float) -> np.ndarray:
    blocks = round(8 + (1.0 - strength) * 14)
    blocks = max(4, min(24, blocks))

    h, w = roi.shape[:2]
    short = max(1.0, min(face.w, face.h))
    k = blocks / short
    down_w = max(1, round(w * k))
    down_h = max(1, round(h * k))

    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_redaction(
    bgr: np.ndarray,
    faces: list[Face],
    mode: str = "black",
    margin: float = 0.14,
    strength: float = 0.7,
) -> int:
    """Redact in place. Returns the number of faces masked."""
    if not faces:
        return 0

    height, width = bgr.shape[:2]
    masked = 0

    for face in faces:
        ell = ellipse_from_face(face, margin)
        x0, y0, x1, y1 = _rotated_bounds(ell, width, height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        roi = bgr[y0:y1, x0:x1]
        alpha = _alpha_mask((roi.shape[0], roi.shape[1]), ell, (x0, y0))[..., None]

        if mode == "black":
            # Darken toward zero. The blend can only ever remove signal, which is
            # why the antialiased rim costs nothing in privacy terms.
            blended = roi.astype(np.float32) * (1.0 - alpha)
        elif mode == "blur":
            ksize = _blur_kernel(face, strength)
            processed = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            blended = roi.astype(np.float32) * (1.0 - alpha) + processed.astype(
                np.float32
            ) * alpha
        elif mode == "pixelate":
            processed = _pixelate_roi(roi, face, strength)
            blended = roi.astype(np.float32) * (1.0 - alpha) + processed.astype(
                np.float32
            ) * alpha
        else:
            raise ValueError(f"unknown redaction mode: {mode}")

        bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        masked += 1

    return masked


def render_audit(bgr: np.ndarray, faces: list[Face], margin: float = 0.14) -> np.ndarray:
    """Annotated copy for the audit map: what will be covered, and how sure we are.

    This is the trust-building view. If a face is visible in the audit map, the
    user knows the tool saw it; if someone is missed, they can see that too and
    pick a more thorough preset before committing.
    """
    canvas = bgr.copy()
    if not faces:
        return canvas

    height, width = canvas.shape[:2]
    tint = canvas.copy()

    for face in faces:
        ell = ellipse_from_face(face, margin)
        center = (round(ell.cx), round(ell.cy))
        axes = (max(1, round(ell.ax)), max(1, round(ell.ay)))
        cv2.ellipse(tint, center, axes, ell.angle, 0, 360, AUDIT_ACCENT, -1, cv2.LINE_AA)

    # Translucent wash so the covered region reads at a glance without hiding
    # the face underneath - the user needs to see who is being covered.
    cv2.addWeighted(tint, 0.22, canvas, 0.78, 0, canvas)

    for index, face in enumerate(faces, start=1):
        ell = ellipse_from_face(face, margin)
        center = (round(ell.cx), round(ell.cy))
        axes = (max(1, round(ell.ax)), max(1, round(ell.ay)))
        cv2.ellipse(canvas, center, axes, ell.angle, 0, 360, AUDIT_ACCENT, 2, cv2.LINE_AA)

        bx0 = int(max(0, face.x))
        by0 = int(max(0, face.y))
        bx1 = int(min(width - 1, face.x + face.w))
        by1 = int(min(height - 1, face.y + face.h))
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), AUDIT_BOX, 1, cv2.LINE_AA)

        label = f"{index}  {face.score:.2f}"
        scale = max(0.42, min(0.9, face.w / 190.0))
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        chip_h = th + baseline + 8
        chip_w = tw + 12

        lx = min(max(0, bx0), max(0, width - chip_w))
        ly = by0 - chip_h - 4
        if ly < 0:
            ly = min(height - chip_h, by1 + 4)
        ly = max(0, ly)

        cv2.rectangle(
            canvas,
            (lx, ly),
            (lx + chip_w, ly + chip_h),
            AUDIT_INK,
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (lx, ly),
            (lx + chip_w, ly + chip_h),
            AUDIT_ACCENT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (lx + 6, ly + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            AUDIT_ACCENT,
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def summary(faces: list[Face]) -> list[dict]:
    return [f.to_dict() for f in faces]


def confidence_note(faces: list[Face]) -> str:
    """A plain-language read on how solid the detections are."""
    if not faces:
        return "No faces found."
    weak = [f for f in faces if f.score < 0.65]
    strongest = max(f.score for f in faces)
    if weak:
        return (
            f"{len(faces)} face{'s' if len(faces) != 1 else ''} found, "
            f"{len(weak)} below 0.65 confidence - check the map closely."
        )
    return (
        f"{len(faces)} face{'s' if len(faces) != 1 else ''} found, "
        f"strongest confidence {strongest:.2f}."
    )
