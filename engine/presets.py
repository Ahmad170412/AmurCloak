"""Tuning presets and the Settings object.

This module is the single source of truth for every tunable number in the
project. The detector, the redactor and the web layer all read from here so a
default can never drift out of sync between them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Redaction modes
# ---------------------------------------------------------------------------
# Ordered safest -> weakest. The order is meaningful: the UI renders it as a
# safety hierarchy and refuses to present blur as equivalent to an opaque fill.

MODE_RANK = ("black", "pixelate", "blur")

MODES: dict[str, dict[str, str]] = {
    "black": {
        "label": "Solid black",
        "blurb": "Opaque. The pixels are destroyed.",
        "detail": (
            "Erases the face region outright. No algorithm can bring it back - "
            "the only mode here with a real guarantee."
        ),
        "recoverability": "irreversible",
    },
    "pixelate": {
        "label": "Pixelate",
        "blurb": "Mosaic blocks. Partly reversible.",
        "detail": (
            "Destroys most detail, but research-grade un-pixelation can partially "
            "reconstruct a familiar face from a still."
        ),
        "recoverability": "partly reversible",
    },
    "blur": {
        "label": "Blur",
        "blurb": "Soft blur. Partly reversible.",
        "detail": (
            "Looks the most natural to viewers, and is the weakest option: Gaussian "
            "blur is a known transform and can be partially inverted."
        ),
        "recoverability": "partly reversible",
    },
}


# ---------------------------------------------------------------------------
# Detection presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    blurb: str  # one line, shown next to the option
    detail: str  # fuller explanation, shown under the group
    score: float  # YuNet confidence threshold
    margin: float  # ellipse expansion as a fraction of face size
    min_face: int  # drop detections smaller than this many px (short side)
    detect_edge: int  # inference input, long edge
    tiles: bool  # extra overlapping tile pass for small faces


PRESETS: dict[str, Preset] = {
    "conservative": Preset(
        key="conservative",
        label="Cautious",
        blurb="Only confident hits.",
        detail=(
            "Masks only high-confidence detections. Fewest false positives, and the "
            "most likely to leave someone unredacted in a crowded shot."
        ),
        score=0.75,
        margin=0.20,
        min_face=32,
        detect_edge=1024,
        tiles=False,
    ),
    "balanced": Preset(
        key="balanced",
        label="Balanced",
        blurb="Confident hits, plus a tiled scan.",
        detail=(
            "The default. Confident detections, plus extra crops of the image that "
            "recover smaller and more distant faces."
        ),
        score=0.65,
        margin=0.14,
        min_face=24,
        detect_edge=1024,
        tiles=True,
    ),
    "aggressive": Preset(
        key="aggressive",
        label="Thorough",
        blurb="Lowest threshold, widest net.",
        detail=(
            "Finds the most faces and may over-mask. Worth it for dense group shots "
            "where missing someone matters more than covering extra."
        ),
        score=0.45,
        margin=0.10,
        min_face=16,
        detect_edge=1280,
        tiles=True,
    ),
}

DEFAULT_PRESET = "balanced"

# The user chose this explicitly: a redacted photo that is still recoverable is
# worse than no tool at all, because it creates false confidence.
DEFAULT_MODE = "black"

DEFAULT_STRENGTH = 0.7
DEFAULT_QUALITY = 95
MIN_FACE_FLOOR = 12  # never let a preset request faces below this
DETECT_EDGE_CAP = 2048  # guard against absurd inference sizes


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass
class Settings:
    """A fully resolved job configuration.

    Built from a preset, then optionally overridden by explicit UI/CLI values.
    Never construct this with partial preset values - use :func:`build`.
    """

    mode: str = DEFAULT_MODE
    preset: str = DEFAULT_PRESET
    score: float = 0.65
    margin: float = 0.14
    min_face: int = 24
    detect_edge: int = 1024
    tiles: bool = True
    strength: float = DEFAULT_STRENGTH
    strip_exif: bool = True
    quality: int = DEFAULT_QUALITY

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def mode_info(self) -> dict[str, str]:
        return MODES.get(self.mode, MODES[DEFAULT_MODE])


def build(
    preset: str = DEFAULT_PRESET,
    *,
    mode: str | None = None,
    margin: float | None = None,
    strength: float | None = None,
    tiles: bool | None = None,
    strip_exif: bool | None = None,
    quality: int | None = None,
) -> Settings:
    """Resolve a preset into a Settings object, applying explicit overrides."""
    p = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])

    settings = Settings(
        mode=DEFAULT_MODE,
        preset=p.key,
        score=p.score,
        margin=p.margin,
        min_face=max(MIN_FACE_FLOOR, p.min_face),
        detect_edge=min(DETECT_EDGE_CAP, p.detect_edge),
        tiles=p.tiles,
    )

    if mode in MODES:
        settings.mode = mode
    if margin is not None:
        settings.margin = clamp(float(margin), 0.0, 0.60)
    if strength is not None:
        settings.strength = clamp(float(strength), 0.0, 1.0)
    if tiles is not None:
        settings.tiles = bool(tiles)
    if strip_exif is not None:
        settings.strip_exif = bool(strip_exif)
    if quality is not None:
        settings.quality = int(clamp(int(quality), 60, 100))

    return settings


def from_payload(payload: dict) -> Settings:
    """Build Settings from an untrusted request payload.

    Every field is validated. A malformed payload degrades to defaults rather
    than raising, because a privacy job that refuses to run is a privacy job
    that does not happen.
    """
    payload = payload or {}

    def _num(key: str) -> float | None:
        try:
            raw = payload.get(key)
            return None if raw is None else float(raw)
        except (TypeError, ValueError):
            return None

    def _flag(key: str) -> bool | None:
        raw = payload.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    preset = payload.get("preset")
    if preset not in PRESETS:
        preset = DEFAULT_PRESET

    mode = payload.get("mode")
    if mode not in MODES:
        mode = DEFAULT_MODE

    return build(
        preset,
        mode=mode,
        margin=_num("margin"),
        strength=_num("strength"),
        tiles=_flag("tiles"),
        strip_exif=_flag("strip_exif"),
        quality=_num("quality"),
    )


def catalog() -> dict:
    """Everything the frontend needs to render its controls."""
    return {
        "modes": [
            {"key": k, **MODES[k], "rank": i} for i, k in enumerate(MODE_RANK)
        ],
        "presets": [
            {
                "key": p.key,
                "label": p.label,
                "blurb": p.blurb,
                "detail": p.detail,
                "margin": p.margin,
                "tiles": p.tiles,
            }
            for p in PRESETS.values()
        ],
        "defaults": {
            "mode": DEFAULT_MODE,
            "preset": DEFAULT_PRESET,
            "strength": DEFAULT_STRENGTH,
            "strip_exif": True,
        },
    }
