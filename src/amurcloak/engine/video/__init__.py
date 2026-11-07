"""Clip ingest, tracking, and streaming redaction.

The same face that a photo clears with one mask has to stay clear for up to
1,800 consecutive frames, so this package owns the one thing photos never
needed: *time*. Detection is per-frame YuNet, exactly as for photos; identity
is not. A tracker stitches per-frame detections into track ids so a moving face
is one person from the first frame to the last, the mask follows smoothly, and
"leave that person visible" means the whole clip at once.

Like everything else in the engine, this is streaming: one frame in flight in
RAM, and two passes over a file that was already fully staged on disk.
"""

__all__ = ["probe", "render", "track"]
