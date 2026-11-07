"""Turn per-frame detections into track ids and temporally coherent masks.

Detection is per frame and ignorant of time. This module is where time enters:

* **Identity.** Detections are matched frame to frame by IoU against the live
  tracks (greedy, highest overlap first, 0.20 minimum). A detection that
  matches nothing becomes a new track; a track that finds nothing goes into
  *grace* - its last known box stays masked for a short while, widened along
  its last known motion, so a half-second occlusion does not blink the mask off
  and on and a face that kept moving does not walk out from under it.
* **Stability.** Track boxes are exponential moving averages of position, size
  and head roll, so the mask drifts with the face instead of shivering at the
  detector's frame-to-frame wobble. This is the difference between a cover that
  reads as deliberate and one that reads as a glitch marking exactly the frames
  you did not want looked at.
* **Safety on onset.** Detection is a frame or two late the moment a face
  enters the frame. Each track's mask is extended *backwards* for a short run,
  so a face is never caught un-covered in the frames the detector had not
  caught up to. The cost is a static stop of background covered slightly
  before the face arrives - the conservative direction for a privacy tool.

The output is small: one box (with roll and track id) per present face per
frame - kilobytes for a full minute, which is why it can live in RAM while the
decode cannot. After this, rendering and preview are frame lookups.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..detector import Face

# Frame a track keeps being masked after its last confident detection. About
# 0.4s of a 30fps phone clip: shorter, and a blink or a hand in front of the
# camera makes the mask flicker; longer, and a face that has genuinely left the
# frame drags its mask across the room.
GRACE_FRAMES = 12
# The matching onset window, same moment scale as the grace period.
ONSET_FRAMES = 12
# An overlap of one fifth of either box is enough to call it "the same face";
# fast pans move a face further than that between frames far less often than
# two faces standing next to each other overlap by more than it accidentally.
IOU_MATCH = 0.20
# EMA factors: positions move every frame, sizes and roll move rarely, so
# position follows faster than shape does.
POS_EMA = 0.45
SIZE_EMA = 0.30
ROLL_EMA = 0.30
VEL_EMA = 0.35

# Grace covers the union of the last known box and a short extrapolation of it,
# because detection lags motion: holding only the last box lets a fast face
# leave its own cover behind. The extrapolation is capped twice - per frame, and
# over the whole grace run - so a noisy velocity can never fling the cover
# across the frame. Both caps are in face widths.
MAX_SPEED_RATIO = 0.35
MAX_GAP_DRIFT = 1.5


@dataclass(frozen=True)
class Mask:
    """One present face in one frame, as found on disk - not a detection."""

    x: float
    y: float
    w: float
    h: float
    roll: float
    score: float
    track: int  # the identity; 1-based, stable for the whole clip

    def to_face(self) -> Face:
        """The Face a redaction pass and the preview layer both expect."""
        return Face(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            score=self.score,
            landmarks=None,
            roll=self.roll,
            track=self.track,
        )


@dataclass
class _Track:
    id: int
    cx: float
    cy: float
    w: float
    h: float
    roll: float
    score: float
    first: int  # first frame seen
    last: int  # last frame seen
    missed: int = 0
    detected: bool = True  # False from the second grace frame onwards
    # The smoothed state at every frame this track was detected on, ascending:
    # ``(frame, cx, cy, w, h, roll, score)``. This is the whole point of the
    # tracker - the render looks its mask up by frame index, so the cover has
    # to be the state *at that frame*. Keeping only the current values would
    # leave one box per track, frozen at wherever the face ended up.
    states: list[tuple[int, float, float, float, float, float, float]] = field(
        default_factory=list
    )
    # Smoothed centre velocity in pixels per frame, used only while the face is
    # missing; it is never applied to a frame that has its own detection.
    vx: float = 0.0
    vy: float = 0.0


def _mask_from(
    track_id: int, state: tuple[int, float, float, float, float, float, float]
) -> Mask:
    """One frame's mask from a recorded ``(frame, cx, cy, w, h, roll, score)``."""
    _, cx, cy, w, h, roll, score = state
    return Mask(
        x=cx - w / 2.0,
        y=cy - h / 2.0,
        w=w,
        h=h,
        roll=roll,
        score=score,
        track=track_id,
    )


def _grace_mask(track: _Track, frame: int) -> Mask:
    """The cover for a frame the detector has lost the face on.

    The last known box is a certainty, the extrapolation is an estimate, so the
    mask is the box containing both. A face that stopped is still covered where
    it stopped; a face that kept moving is covered where it went. Widening can
    only ever add cover, which is the safe direction - and because the frozen
    box is always inside it, nothing that was covered stops being covered.
    """
    last_frame, cx, cy, w, h, roll, score = track.states[-1]
    missed = frame - last_frame
    speed = MAX_SPEED_RATIO * max(1.0, w)
    dx = max(-speed, min(speed, track.vx)) * missed
    dy = max(-speed, min(speed, track.vy)) * missed
    drift = max(abs(dx), abs(dy))
    cap = MAX_GAP_DRIFT * max(1.0, w)
    if drift > cap:
        dx, dy = dx * (cap / drift), dy * (cap / drift)
    px, py = cx + dx, cy + dy
    return Mask(
        x=min(cx, px) - w / 2.0,
        y=min(cy, py) - h / 2.0,
        w=abs(px - cx) + w,
        h=abs(py - cy) + h,
        roll=roll,
        score=score,
        track=track.id,
    )


def _fold(angle: float) -> float:
    """Fold an eye-line angle into (-90, 90], where the roll is well-defined."""
    return (float(angle) + 90.0) % 180.0 - 90.0


def _ema(prev: float, new: float, alpha: float) -> float:
    return prev + alpha * (new - prev)


def _iou(a: Mask | _Track, b: tuple[float, float, float, float]) -> float:
    if isinstance(a, Mask):
        ax, ay = a.x, a.y
    else:
        ax, ay = a.cx - a.w / 2.0, a.cy - a.h / 2.0
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax, ay, ax + a.w, ay + a.h
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    union = a.w * a.h + bw * bh - iw * ih
    return (iw * ih) / union if union > 0 else 0.0


def build_masks(frames: Sequence[Sequence[Face]]) -> list[list[Mask]]:
    """Match detections across frames; return one Mask list per frame.

    The input is the raw per-frame YuNet output, in order. The output has
    exactly one entry per input frame, so a render pass can index it with the
    frame counter and always agree with what was scanned.
    """
    tracks: list[_Track] = []
    next_id = 1

    for index, detections in enumerate(frames):
        # Greedy matching: every (track, detection) pair with IoU above the
        # floor is a candidate, taken strongest first. Two faces overlapping
        # each other or a fast pan can produce a locally wrong assignment, but
        # both are self-healing within grace frames and never leak pixels.
        candidates: list[tuple[float, _Track, Face]] = []
        for track in tracks:
            if track.missed > GRACE_FRAMES:
                continue
            for det in detections:
                iou = _iou(track, (det.x, det.y, det.w, det.h))
                if iou >= IOU_MATCH:
                    candidates.append((iou, track, det))
        candidates.sort(key=lambda c: c[0], reverse=True)

        used: set[int] = set()  # tracks, by id
        taken: set[int] = set()  # detection indices
        for _iou_value, track, det in candidates:
            if track.id in used or id(det) in taken:
                continue
            used.add(track.id)
            taken.add(id(det))

            dcx, dcy = det.cx, det.cy
            track.cx = _ema(track.cx, dcx, POS_EMA)
            track.cy = _ema(track.cy, dcy, POS_EMA)
            track.w = _ema(track.w, det.w, SIZE_EMA)
            track.h = _ema(track.h, det.h, SIZE_EMA)
            if det.roll is not None:
                track.roll = _fold(_ema(track.roll, _fold(det.roll), ROLL_EMA))
            else:
                track.roll = 0.0
            track.score = max(track.score, det.score)
            track.last = index
            track.missed = 0
            track.detected = True
            # Velocity from the previous recorded frame, divided by the frames
            # actually skipped, so a detection after a gap does not read as one
            # enormous step.
            prev_frame, prev_cx, prev_cy = track.states[-1][:3]
            gap = max(1, index - prev_frame)
            track.vx = _ema(track.vx, (track.cx - prev_cx) / gap, VEL_EMA)
            track.vy = _ema(track.vy, (track.cy - prev_cy) / gap, VEL_EMA)
            # Snapshot taken after the smoothing above, so the recorded position
            # is exactly the mask this frame will carry.
            track.states.append(
                (index, track.cx, track.cy, track.w, track.h, track.roll, track.score)
            )

        for det in detections:
            if id(det) in taken:
                continue
            roll = det.roll if det.roll is not None else 0.0
            tracks.append(
                _Track(
                    id=next_id,
                    cx=det.cx,
                    cy=det.cy,
                    w=det.w,
                    h=det.h,
                    roll=_fold(roll),
                    score=det.score,
                    first=index,
                    last=index,
                    states=[
                        (
                            index,
                            det.cx,
                            det.cy,
                            det.w,
                            det.h,
                            _fold(roll),
                            det.score,
                        )
                    ],
                )
            )
            next_id += 1

        for track in tracks:
            if track.id in used:
                continue
            track.missed += 1
            track.detected = False

    # Assemble the timeline. Each frame gets the track's state *as recorded on
    # that frame*, so the mask moves with the face. Frames before the first hit
    # hold that first box; frames after the last one are covered by the frozen
    # box unioned with a capped extrapolation of the last known motion, so a
    # face that keeps moving during a detector gap is still under cover.
    total = len(frames)
    out: list[list[Mask]] = [[] for _ in range(total)]
    for track in tracks:
        states = track.states
        cursor = 0
        last_index = states[-1][0]
        start = max(0, track.first - ONSET_FRAMES)
        end = min(total, track.last + GRACE_FRAMES + 1)
        for frame in range(start, end):
            if frame > last_index:
                out[frame].append(_grace_mask(track, frame))
                continue
            while cursor + 1 < len(states) and states[cursor + 1][0] <= frame:
                cursor += 1
            out[frame].append(_mask_from(track.id, states[cursor]))
    return out


def count_tracks(masks: Sequence[Sequence[Mask]]) -> int:
    """Distinct identities the tracker saw, for the one-line summary.

    Takes the tracker's output, not the raw detections: identity only exists on
    ``Mask`` (a ``Face`` straight from YuNet has ``track is None``).
    """
    return max((m.track for frame in masks for m in frame), default=0)
