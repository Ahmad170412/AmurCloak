"""YuNet face detection.

YuNet is a ~230KB ONNX model that runs inside OpenCV's own DNN engine. That
choice is deliberate: it needs no PyTorch, no onnxruntime, no GPU, and it gives
five facial landmarks. Those landmarks matter - they let the redactor build a
face-fitted, roll-aligned ellipse instead of a lazy square.

Detection runs over several views ("tiling"). A single pass on a downscaled
whole image loses small faces in group photos, because by the time a 6000px
photo is squeezed into a 1024px inference budget a distant face is a few pixels
across. Re-running inference on overlapping crops raises the effective
resolution for those regions at a cost of a few milliseconds per crop.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# OpenCV 5's new graph engine emits a harmless "targets not supported" warning
# at model load. Suppress it before it reaches the log, since a scary-looking
# warning on startup undermines trust in a privacy tool.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MIN_INFER_DIM = 64  # YuNet needs a sane minimum; below this we skip the view
TILE_OVERLAP = 0.15
EDGE_TOUCH_PENALTY = 0.88  # ranking nudge against boxes clipped by a crop edge
MAX_GRID = 4  # 4x4 = 16 crops, plus the full frame; caps work on huge panoramas

# A model instance is expensive to construct (~40ms) and cheap to reuse, but it
# holds mutable input-size state, so inference is serialised on one lock. That
# is fine here: detection is tens of milliseconds and the workload is a single
# local user. Serialising also stops concurrent previews thrashing the CPU.
_CACHE_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_NET_CACHE: dict[tuple, cv2.FaceDetectorYN] = {}


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray = field(repr=False, default=None)  # (5,2) float32
    rank: float = 0.0  # internal ordering key, not the reported confidence

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def short_side(self) -> float:
        return min(self.w, self.h)

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "w": round(self.w, 1),
            "h": round(self.h, 1),
            "score": round(self.score, 3),
        }

    def key(self) -> tuple[int, int, int, int]:
        """Identity for picking a person, derived from the box itself.

        Not a list index: the order detections come back in is an implementation
        detail, and a stale index would silently un-redact the wrong person. The
        rounded box is stable for as long as the detection cache holds the same
        detection, which is exactly the window in which the user is clicking.
        """
        return (round(self.x), round(self.y), round(self.w), round(self.h))


def _iou(a: Face, b: Face) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nms(faces: list[Face], threshold: float) -> list[Face]:
    """Greedy non-maximum suppression on the internal rank key."""
    ordered = sorted(faces, key=lambda f: f.rank, reverse=True)
    kept: list[Face] = []
    for candidate in ordered:
        if all(_iou(candidate, k) <= threshold for k in kept):
            kept.append(candidate)
    return kept


def model_path(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root) / MODEL_FILENAME
    return Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    destination: str | Path | None = None, allow_download: bool = False
) -> Path:
    """Verify the vendored model, optionally fetching it if absent.

    The hash is checked so a corrupted or substituted model file cannot silently
    degrade detection - which for this tool would mean faces escaping redaction.
    Downloading is opt-in; the default posture is that no runtime network call
    ever happens.
    """
    target = Path(destination) if destination else model_path()

    if target.exists():
        if _sha256(target) == MODEL_SHA256:
            return target
        if not allow_download:
            raise FileNotFoundError(
                f"{target} fails its checksum. Delete it and re-run with "
                "--allow-download to fetch a clean copy."
            )

    if not allow_download:
        raise FileNotFoundError(
            f"face model missing at {target}.\n"
            "Re-run with --allow-download to fetch it once "
            "(it is a 232KB file, and nothing else ever touches the network)."
        )

    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, temporary)  # fixed https URL, never user input
    if _sha256(temporary) != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded model failed checksum; refusing to use it")
    temporary.replace(target)
    return target


class FaceDetector:
    """Stateful-but-cheap wrapper around YuNet with multi-view inference."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        score: float = 0.65,
        nms: float = 0.3,
        top_k: int = 200,
        min_face: int = 24,
        detect_edge: int = 1024,
        tiles: bool = True,
    ) -> None:
        self.model = Path(model) if model else model_path()
        if not self.model.exists():
            raise FileNotFoundError(
                f"face model missing at {self.model}. "
                "Re-download it or run with --allow-download."
            )
        self.score = float(score)
        self.nms = float(nms)
        self.top_k = int(top_k)
        self.min_face = int(min_face)
        self.detect_edge = int(detect_edge)
        self.tiles = bool(tiles)

    # -- network cache ----------------------------------------------------

    def _net(self) -> cv2.FaceDetectorYN:
        key = (str(self.model), round(self.score, 3), round(self.nms, 3), self.top_k)
        with _CACHE_LOCK:
            net = _NET_CACHE.get(key)
            if net is None:
                net = cv2.FaceDetectorYN.create(
                    str(self.model),
                    "",
                    (self.detect_edge, self.detect_edge),
                    self.score,
                    self.nms,
                    self.top_k,
                )
                _NET_CACHE[key] = net
            return net

    # -- view geometry ----------------------------------------------------

    def _views(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """Rects to run inference over, full frame first.

        The grid is derived from the image size rather than a fixed threshold:
        we want crops that are *at or just above* the inference budget so they
        run at roughly native resolution. A fixed 2x2 split still downscales a
        3000px photo by 0.6, which is exactly where small faces are lost.
        """
        full = (0, 0, width, height)
        if not self.tiles:
            return [full]

        long_edge = max(height, width)
        if long_edge <= self.detect_edge:
            return [full]  # already at native resolution, nothing to gain

        grid = math.ceil(long_edge / float(self.detect_edge))
        grid = max(2, min(MAX_GRID, grid))

        def cuts(total: int) -> list[tuple[int, int]]:
            stride = total / grid
            pad = stride * TILE_OVERLAP / 2.0
            spans = []
            for i in range(grid):
                start = max(0, round(i * stride - pad))
                end = min(total, round((i + 1) * stride + pad))
                if end - start >= MIN_INFER_DIM:
                    spans.append((start, end))
            return spans

        xs = cuts(width)
        ys = cuts(height)
        views = [full]
        for y0, y1 in ys:
            for x0, x1 in xs:
                if (x0, y0, x1, y1) == full:
                    continue
                views.append((x0, y0, x1, y1))
        return views

    # -- inference --------------------------------------------------------

    def _infer_view(
        self, bgr: np.ndarray, rect: tuple[int, int, int, int]
    ) -> list[Face]:
        x0, y0, x1, y1 = rect
        view = bgr[y0:y1, x0:x1]
        vh, vw = view.shape[:2]
        if vh < MIN_INFER_DIM or vw < MIN_INFER_DIM:
            return []

        long_edge = max(vh, vw)
        scale = min(1.0, self.detect_edge / long_edge)
        if scale < 1.0:
            tw = max(MIN_INFER_DIM, round(vw * scale))
            th = max(MIN_INFER_DIM, round(vh * scale))
            net_in = cv2.resize(view, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            tw, th = vw, vh
            net_in = view
        if min(tw, th) < MIN_INFER_DIM:
            return []

        net = self._net()
        with _INFER_LOCK:
            net.setInputSize((tw, th))
            _retval, faces = net.detect(net_in)
        if faces is None or len(faces) == 0:
            return []

        # Map inference coordinates back to full-image coordinates.
        sx = vw / float(tw)
        sy = vh / float(th)
        is_crop = rect != (0, 0, bgr.shape[1], bgr.shape[0])

        out: list[Face] = []
        for row in np.asarray(faces):
            fx, fy, fw, fh = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            score = float(row[14])
            if fw <= 0 or fh <= 0:
                continue

            box_x = x0 + fx * sx
            box_y = y0 + fy * sy
            box_w = fw * sx
            box_h = fh * sy

            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            landmarks = np.stack(
                [landmarks[:, 0] * sx + x0, landmarks[:, 1] * sy + y0], axis=1
            )

            rank = score
            # A box clipped by a crop boundary is usually a partial face. Rank it
            # below a complete detection of the same face so NMS keeps the good one.
            if is_crop and (
                fx <= 1.0
                or fy <= 1.0
                or fx + fw >= tw - 1.0
                or fy + fh >= th - 1.0
            ):
                rank *= EDGE_TOUCH_PENALTY

            out.append(
                Face(
                    x=box_x,
                    y=box_y,
                    w=box_w,
                    h=box_h,
                    score=score,
                    landmarks=landmarks,
                    rank=rank,
                )
            )
        return out

    # -- public -----------------------------------------------------------

    def detect(self, bgr: np.ndarray) -> list[Face]:
        """Faces in full-image pixel coordinates, left to right, top to bottom."""
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]

        raw: list[Face] = []
        for rect in self._views(h, w):
            raw.extend(self._infer_view(bgr, rect))
        if not raw:
            return []

        kept = [
            f
            for f in raw
            if f.score >= self.score and f.short_side >= self.min_face
        ]
        if not kept:
            return []

        merged = _nms(kept, 0.35)
        merged.sort(key=lambda f: (f.cx, f.cy))
        return merged


def describe_views(height: int, width: int, detect_edge: int, tiles: bool) -> int:
    """Number of inference passes a photo will need. Used for UI cost hints."""
    probe = FaceDetector.__new__(FaceDetector)
    probe.detect_edge = detect_edge
    probe.tiles = tiles
    return len(probe._views(height, width))
