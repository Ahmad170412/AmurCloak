<div align="center">

<img src="assets/logo.svg" alt="TigerBlur logo" width="320"/>

# TigerBlur

**Find every face in a photo and cover it — on your own machine, with the network unplugged.**

No uploads, no accounts, no telemetry. Drop photos in, check the map, download redacted copies.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS-000000?logo=apple&logoColor=white)](#requirements)
[![Tests](https://img.shields.io/badge/tests-23%20passing-3f6b52)](#testing)
[![Dependencies](https://img.shields.io/badge/dependencies-5-5d6e79)](#requirements)
[![Network](https://img.shields.io/badge/network-none%20at%20runtime-c97a2b)](#why-tigerblur)
[![Detector](https://img.shields.io/badge/detector-YuNet%20227%20KB-17202a)](models/)

[Quick Start](#quick-start) · [Why](#why-tigerblur) · [Features](#features) · [Configuration](#configuration) · [Testing](#testing)

</div>

---

## Why TigerBlur?

- **Your photos never leave the machine.** There is no server to upload to, no cloud API behind a spinner, and no outbound request anywhere in the runtime. You can pull the network cable out mid-batch and nothing changes.
- **One detection pass is not enough for group photos.** A single frame inference reliably misses the small faces at the back of a crowd. TigerBlur also scans overlapping tiles and merges the results — on the 10-face test fixture that is the difference between finding 1 face and finding 4.
- **The default mode is the one that cannot be undone.** Solid black destroys the pixels. Pixelate and blur are *partially reversible* and are labelled as such in the interface, because "AI un-pixelation" is a real and published technique.
- **It costs you nothing to leave running.** Every transient byte lives under one temp root: batches stream into a ZIP one file at a time, staged inputs are freed the moment you download, and a reaper clears idle sessions. Steady-state disk usage is zero.

---

## What it looks like

The default mode, next to the two softer ones — so the trade-off is visible rather than assumed:

![Solid black, pixelate and blur applied to the same face. Black is opaque; pixelate and blur still leak the structure of the face.](assets/modes.png)

The **audit map** shows you what was found before you commit to anything. If a face is visible here, you know it is covered; if someone is missed, you can see that too and switch to a more aggressive preset:

![The TigerBlur interface showing the audit map.](assets/screenshot-app.png)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
cd TigerBlur
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py --port 5000 --open
```

The `--open` flag opens `http://localhost:5000` in your browser. Drop photos onto the page, or click **Choose photos**.

No API keys, no configuration files, no model download — the detector ships in the repository.

---

## Architecture

A single process, one request at a time, no queue and no worker pool:

```text
   photo
     │
     ▼
 ┌─────────────┐   EXIF orientation applied, ICC profile kept,
 │   decode    │   everything else discarded
 └──────┬──────┘
        ▼
 ┌─────────────┐   YuNet (227 KB ONNX, via OpenCV's DNN module)
 │   detect    ├──▶ full-frame pass
 └──────┬──────┘   └─▶ + overlapping tiles when the long edge is large
        ▼             └─▶ NMS, with edge-touching hits ranked down
 ┌─────────────┐
 │    mask     │   ellipse fitted and roll-aligned to the face from
 └──────┬──────┘   the five detected landmarks — not a circle
        ▼
 ┌─────────────┐   black  · opaque, irreversible
 │   redact    │   pixelate · mosaic blocks, partly reversible
 └──────┬──────┘   blur   · Gaussian, partly reversible
        ▼
 ┌─────────────┐   alpha → PNG      .heic/.heif → JPEG
 │   encode    │   .png → PNG       everything else → JPEG q95
 └──────┬──────┘   .tif/.tiff → TIFF
        ▼
   single file, or a ZIP streamed one entry at a time
```

**Colour order is BGR throughout the engine** (OpenCV's convention) and is converted only at the file boundary, so no module has to guess.

---

## Features

| Feature | Description |
|---|---|
| **Three redaction modes** | Solid black, pixelate, and blur — with an honest note on which are reversible. |
| **Tiled detection** | Overlapping crops recover small faces that a single full-frame pass misses. |
| **Audit map** | Annotated preview with numbered boxes, landmark-fitted ellipses, and per-face confidence, so you can verify coverage before rendering. |
| **Roll-aware mask** | The ellipse is aligned to the eye line, so it follows a tilted head instead of clipping the chin. |
| **Metadata stripping** | EXIF, XMP, IPTC and GPS are dropped by default. The ICC profile is kept, so iPhone Display P3 colours survive. |
| **HEIC support** | iPhone photos work directly, via `pillow-heif`. |
| **Nine input formats** | BMP, HEIC, HEIF, JPEG, PNG, TIFF, WebP — outputs keep the original format. |
| **Streaming batches** | Outputs are appended to a `ZIP_STORED` archive and unlinked as they go, so a batch never exists twice on disk. |
| **Automatic cleanup** | Idle sessions are reaped, inputs are freed on download, and orphans are purged at startup. |
| **Zero network** | No CDN, no webfonts, no telemetry. The interface is plain HTML, CSS and JavaScript with no build step. |

---

## Installation

### Requirements

- **Python 3.11 or newer** (developed and tested on 3.14)
- **macOS** — built and verified here. The code is portable and should run on Linux, but that is untested.
- ~200 MB of disk for the environment. The single largest piece is OpenCV's binary at 120 MB.

Five dependencies, all prebuilt wheels, none of them a framework:

| Package | Why |
|---|---|
| `opencv-python-headless` | DNN inference for YuNet, plus resize/blur/pixelate |
| `numpy` | the pixel maths |
| `flask` | the local web interface |
| `pillow` | decode, encode, EXIF |
| `pillow-heif` | iPhone HEIC/HEIF |

Deliberately absent: **PyTorch**, **onnxruntime**, and any **ffmpeg** binary. YuNet runs through the DNN module OpenCV already ships, which is why the model is 227 KB instead of hundreds of megabytes.

### Set up

```bash
git clone <this-repository> TigerBlur
cd TigerBlur
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
```

`--no-cache-dir` avoids keeping a second copy of the wheels in `~/.cache/pip`.

### Verify the model

The detector is vendored, and its checksum is verified on startup:

```bash
shasum -a 256 models/face_detection_yunet_2023mar.onnx
# 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

---

## Configuration

Everything is a command-line flag. There are no environment variables and no config file.

```bash
.venv/bin/python app.py [flags]
```

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Localhost only — do not expose this. |
| `--port` | `5000` | Change it if the port is already in use. |
| `--tmp` | `.tmp` | Temp root for uploads and output. |
| `--ttl` | `1800` | Seconds of inactivity before a session is reaped. |
| `--max-age` | `21600` | Hard limit on a session's age, in seconds. |
| `--shred` | off | Zero-fill temp files before unlinking. Slower; costs a full write pass. |
| `--keep-inputs` | off | Keep staged uploads after download instead of freeing them. |
| `--allow-download` | off | Permit a one-time model fetch if the vendored model is missing. |
| `--open` | off | Open the browser on start. |
| `--debug` | off | Flask debug mode. |

### Limits

| Limit | Value |
|---|---|
| Files per batch | 300 |
| Single file | 200 MB |
| Total per upload | 1.5 GB |
| Free-space guard | 3× the upload, plus 512 MB headroom |

An upload that will not fit is refused **before** anything is written, and the error states the reason.

---

## Usage

### The interface

1. **Drop photos** anywhere on the page, paste a screenshot, or click **Choose photos**.
2. **Check the audit map.** Every detection is numbered and scored. Switch to **Result** to see the actual redaction.
3. **Pick a cover mode.** Solid black is selected by default. Adjust **Coverage** to push the mask further past the face, and **Strength** to change the blur or block size — the latter is disabled in black mode, because solid black ignores it.
4. **Choose how hard to look.** Cautious, Balanced or Thorough, which trades false positives against missed faces.
5. **Redact photos**, then **Download**. One photo downloads as a single image; several are bundled into a ZIP.

Settings apply to the whole staged set, and changing any of them invalidates a rendered result rather than letting you download pixels that no longer match what is on screen.

### Presets

| Preset | Confidence | Smallest face | Inference edge | Use it for |
|---|---|---|---|---|
| Cautious | 0.75 | 32 px | 1024 | Fewest false positives. Solo portraits. |
| **Balanced** | 0.65 | 24 px | 1024 | The default. Confident hits plus a tiled scan. |
| Thorough | 0.45 | 16 px | 1280 | Crowds and small faces, at the cost of more false positives. |

### Performance

Detection is tens of milliseconds on a phone-sized photo, and a few hundred milliseconds on a 24 MP frame with tiling. Rendering is dominated by JPEG encoding, not by the mask.

---

## Testing

Two suites, no test framework, no fixtures to download:

```bash
.venv/bin/python tests/test_engine.py   # 12/12 — detection, masking, colour, metadata, formats
.venv/bin/python tests/test_api.py      # 11/11 — upload, tune, preview, render, download, disk
```

The API suite drives the real Flask app through its test client, so it covers the actual routes rather than mocks.

Notable coverage:

- Tiling recovers strictly more faces than a single pass on the group fixture.
- Pixelate and blur **keep the photo's colours** and never composite against black.
- Solid black writes pure black across the whole masked region.
- Every facial landmark falls inside the mask.
- Outputs carry no EXIF, GPS or camera metadata.
- Changing settings invalidates a stale result.
- The temp spool survives startup and a real multi-megabyte upload, and the orphan purge spares reserved folders.

The last two exist because both were real bugs — an upload that silently failed with a 500, and a purge that deleted the directory Werkzeug spools uploads into.

---

## Repository Structure

```text
TigerBlur/
├── README.md
├── app.py                        # Flask routes, CLI, config, lifecycle
├── requirements.txt              # 5 dependencies
├── engine/
│   ├── detector.py               # YuNet wrapper: tiling, NMS, ranking
│   ├── redact.py                 # ellipse geometry, three modes, audit map
│   ├── photo.py                  # decode/encode, EXIF, metadata stripping
│   ├── pipeline.py               # full-resolution preview composition
│   ├── presets.py                # modes, presets, settings resolution
│   ├── sessions.py               # staging, disk accounting, renderer, reaper
│   └── __init__.py
├── models/
│   └── face_detection_yunet_2023mar.onnx   # 227 KB, checksum verified
├── templates/
│   └── index.html                # the single page
├── static/
│   ├── style.css                 # the whole stylesheet
│   └── app.js                    # the whole front end, no build step
├── assets/                       # logo and screenshots
└── tests/
    ├── test_engine.py            # 12 tests
    ├── test_api.py               # 11 tests
    └── fixtures/                 # lena.jpg, plus fixtures built on demand
```

---

## Contributing

This is a private tool, so there is no contribution process to speak of. If you do extend it, three rules keep it honest:

1. **Nothing may call out to the network at runtime.** That is the entire point of the project.
2. **A preview must never disagree with the delivered file.** Previews are composed at full resolution and only then downscaled — shrinking first would change the blur kernel and the block size.
3. **Never describe a reversible redaction as safe.** If a mode can be partially undone, the interface has to say so.

Run both suites before committing; they are fast enough that there is no excuse not to.

---

## License

**Not yet chosen.** This repository has no `LICENSE` file, which legally means all rights are reserved and nobody may reuse it. If you intend to share or publish TigerBlur, pick one — MIT or Apache-2.0 are the usual choices for a tool like this, and the vendored YuNet model carries its own upstream licence that is worth checking first.

---

<div align="center">
<sub>Built to be used once and trusted: the photos go in, the faces come out covered, and nothing in between leaves the machine.</sub>
</div>
