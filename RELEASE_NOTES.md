# AmurCloak 0.10.0

Redact short clips as well as photos. Faces are tracked across frames so the
cover follows a moving person, audio is kept by default with an off switch, and
a person you leave visible stays visible in the export. The clip path shares the
same audit-map, settings and metadata rules as the photo path.

## In this release

- **Short clips.** Upload `.mp4`, `.mov`, `.m4v`, `.webm`, `.mkv` or `.avi`;
  faces are detected per frame, tracked with identity persistence, and covered
  with the same roll-aware masks as photos.
- **Tracking that follows the face.** A face walking across the frame keeps one
  identity, and a detector gap widens the cover along the motion instead of
  freezing it.
- **Leave a person visible, per clip.** Click a covered face to spare that
  tracked person; the choice is recorded by track id and dropped on any re-scan.
- **Audio.** Kept by default, removable with the off switch.
- **Clip limits.** Duration, frame count, resolution and dead-clock guards refuse
  oversized or malformed clips before anything decodes them.
- **Metadata.** Clips follow the same strip/forged policy as photos.
- **Tests.** 57 passing: 18 engine, 22 API, 17 video.

# AmurCloak 0.9.0

The first release. Find every face in a photo and cover it on your own machine,
with the network unplugged.

## Install

One paste, straight from this release:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv tool install https://github.com/Ahmad170412/AmurCloak/releases/latest/download/amurcloak-0.9.0-py3-none-any.whl
```

Then run it:

```bash
amurcloak
```

Windows PowerShell and wheel-from-checkout routes are in the
[README](https://github.com/Ahmad170412/AmurCloak#quick-start).

## In this release

- **Local only.** No uploads, no accounts, no telemetry. The only runtime socket
  is a loopback probe of the app's own port.
- **YuNet detection inside the package.** A 227 KB ONNX model runs through
  OpenCV's DNN module, with its SHA-256 verified at startup. No PyTorch, no
  onnxruntime, no model download.
- **Tiled detection.** Overlapping crops recover small faces that a single
  full-frame pass misses — on the 10-face fixture, 4 faces found instead of 1.
- **Three cover modes.** Solid black (the default, irreversible), pixelate and
  blur, each labelled with whether it can be partly reversed.
- **Roll-aware masks.** The ellipse is fitted to the five detected landmarks and
  aligned to the eye line, so a tilted head is covered without clipping the chin.
- **Audit map.** Numbered boxes, mask geometry and per-face confidence, shown
  before anything is rendered.
- **Leave one person visible.** Click a covered face to give that person their
  pixels back. Per photo, never matched across a batch.
- **Metadata removed by default.** EXIF, GPS, XMP, IPTC and maker notes are
  dropped; the ICC profile is kept so wide-gamut colours survive.
- **Opt-in forged metadata.** A false block built from nothing — chosen date,
  a made-up iPhone 13, and a Point Nemo location — with exposure and lens tags
  derived so the block agrees with itself. No thumbnail from the original can
  ride along.
- **Nine input extensions.** `.bmp`, `.heic`, `.heif`, `.jpeg`, `.jpg`, `.png`,
  `.tif`, `.tiff`, `.webp`; outputs keep the original format, and HEIC/HEIF
  convert to JPEG.
- **Streaming batches.** Outputs are appended to a `ZIP_STORED` archive and
  unlinked as they go; idle sessions are reaped and staged inputs are freed on
  download.
- **Light and dark.** Follows the system until you choose, remembers the choice,
  and settles it before the first paint.

## Requirements

- Python 3.11 or newer — uv supplies one if needed.
- macOS, Windows or Linux. The wheel is pure Python (`py3-none-any`); OpenCV's
  own platform wheels provide the native code.
- About 250 MB on first run, almost all of it OpenCV, into uv's shared cache.
  After that the command works with the network off.

## Testing

```bash
uv run python tests/test_engine.py   # 18/18
uv run python tests/test_api.py      # 22/22
```

## Known limits

- Pixelate and blur are **partly reversible**. Solid black is the only mode with
  a real guarantee.
- Covering faces is not anonymity: the pixels still identify a place and a
  person, and the file no longer volunteers where and when it was taken.
- "Leave visible" is a per-photo decision. The tool has no face recognition and
  will not match the same person across a batch.
- A forged TIFF carries a flat IFD0 — camera, resolution and timestamps — with
  no nested EXIF or GPS directory.
- This is a single-user localhost tool. Do not bind it to a public interface.

## Checksums

```text
SHA256 (amurcloak-0.9.0-py3-none-any.whl)  ca7f31a6da80fc31bf9075e606582ed35b3182c40d8ae8778cdbe4d148d7f5e1
SHA256 (amurcloak-0.9.0.tar.gz)            e7fc60b124b54714b1443350604b16da3fe18bab611707a831844693ccf6566b
```
