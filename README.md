<div align="center">

<img src="assets/logo.svg" alt="AmurCloak logo" width="320"/>

# AmurCloak

**Find every face in a photo and cover it — on your own machine, with the network unplugged.**

No uploads, no accounts, no telemetry. Drop photos in, check the map, download redacted copies.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-000000)](#requirements)
[![Install](https://img.shields.io/badge/install-uv%20tool%20install-de5fe9)](#quick-start)
[![Tests](https://img.shields.io/badge/tests-40%20passing-3f6b52)](#testing)
[![License](https://img.shields.io/badge/license-MIT-97ca00)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-5-5d6e79)](#requirements)
[![Network](https://img.shields.io/badge/network-none%20at%20runtime-c97a2b)](#why-amurcloak)
[![Detector](https://img.shields.io/badge/detector-YuNet%20227%20KB-17202a)](src/amurcloak/models/)

[Quick Start](#quick-start) · [Why](#why-amurcloak) · [Features](#features) · [Configuration](#configuration) · [Testing](#testing)

</div>

---

## Why AmurCloak?

- **Your photos never leave the machine.** There is no server to upload to, no cloud API behind a spinner, and no outbound request anywhere in the runtime — the only socket the program opens is a loopback check of its own port, to tell "already running" apart from "free port". You can pull the network cable out mid-batch and nothing changes.
- **One detection pass is not enough for group photos.** A single frame inference reliably misses the small faces at the back of a crowd. AmurCloak also scans overlapping tiles and merges the results — on the 10-face test fixture that is the difference between finding 1 face and finding 4.
- **The default mode is the one that cannot be undone.** Solid black destroys the pixels. Pixelate and blur are *partially reversible* and are labelled as such in the interface, because "AI un-pixelation" is a real and published technique.
- **It costs you nothing to leave running.** Every transient byte lives under one temp root: batches stream into a ZIP one file at a time, staged inputs are freed the moment you download, and a reaper clears idle sessions. Steady-state disk usage is zero.

---

## What it looks like

The default mode, next to the two softer ones — so the trade-off is visible rather than assumed:

![Solid black, pixelate and blur applied to the same face. Black is opaque; pixelate and blur still leak the structure of the face.](assets/modes.png)

The **audit map** shows you what was found before you commit to anything. If a face is visible here, you know it is covered; if someone is missed, you can see that too and switch to a more aggressive preset:

![The AmurCloak interface showing the audit map.](assets/screenshot-app.png)

The same workspace after dark, switched to the result view. It follows your system until you choose a side, remembers the choice, and settles the theme before the first paint so the page never flashes the wrong one:

![The same interface in dark mode, showing the result view.](assets/screenshot-dark.png)

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

**One paste.** This installs [uv](https://docs.astral.sh/uv/) — which brings its own Python — and then AmurCloak itself, straight from a release. The `export` is there so it works in the terminal you already have open, with no restart:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv tool install https://github.com/Ahmad170412/AmurCloak/releases/latest/download/amurcloak-0.9.0-py3-none-any.whl
```

Windows, in PowerShell — same idea:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"; $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"; uv tool install https://github.com/Ahmad170412/AmurCloak/releases/latest/download/amurcloak-0.9.0-py3-none-any.whl
```

**Then, forever after:**

```bash
amurcloak
```

That wheel is the entire install — one file, and the same one on macOS, Windows and Linux, with every dependency resolved for your machine. If you would rather pull from the package index, `uv tool install amurcloak` does the same job.

It takes the first free port from 5000, opens your browser, and stops on Control-C. Run it a second time and it notices the copy already running instead of starting another server.

No API keys, no configuration files, no model download — the detector ships inside the package.

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
        ▼          faces you left visible are passed through untouched
 ┌─────────────┐   alpha → PNG      .heic/.heif → JPEG
 │   encode    │   .png → PNG       everything else → JPEG q95
 └──────┬──────┘   .tif/.tiff → TIFF
        ▼          EXIF dropped, or a forged block written in its place
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
| **Leave one person visible** | Click any covered face in the preview to give that person their pixels back — the person waving behind the birthday cake stays in the photo. Per photo, and never matched across a batch. |
| **Forged metadata (opt-in)** | Leave stripped, or write a deliberately false block instead: a chosen date on all three timestamp tags, a made-up camera, and a location fixed to the emptiest point in the Pacific — built from nothing, so no thumbnail and no maker notes from the original can tag along. |
| **Light and dark** | One workspace at two exposures, chosen with the toggle in the top bar. It follows your system until you pick a side, then remembers. The theme is settled before the first paint, so there is no flash, and every text-on-background pair in both themes clears WCAG AA. |
| **HEIC support** | iPhone photos work directly, via `pillow-heif`. |
| **Nine input formats** | BMP, HEIC, HEIF, JPEG, PNG, TIFF, WebP — outputs keep the original format. |
| **Streaming batches** | Outputs are appended to a `ZIP_STORED` archive and unlinked as they go, so a batch never exists twice on disk. |
| **Automatic cleanup** | Idle sessions are reaped, inputs are freed on download, and orphans are purged at startup. |
| **No outbound network** | No CDN, no webfonts, no telemetry, no cloud API. The interface is plain HTML, CSS and JavaScript with no build step. |

---

## Installation

### Requirements

- **Python 3.11 or newer.** uv supplies one if you do not have it, which matters: a clean macOS ships a 3.9 stub with a 2021-era pip that cannot install these dependencies.
- **macOS, Windows or Linux.** The package is pure Python (`py3-none-any`), so one 251 KB wheel covers all three. The platform-specific weight comes from OpenCV's own published wheels, fetched per machine.
- **~250 MB on first run**, almost all of it OpenCV, into uv's shared cache. Reinstalling, or any other uv project, reuses it.
- **One download, ever.** After the first run the command works with the network off.

Five dependencies, all prebuilt wheels, none of them a framework:

| Package | Why |
|---|---|
| `opencv-python-headless` | DNN inference for YuNet, plus resize/blur/pixelate |
| `numpy` | the pixel maths |
| `flask` | the local web interface |
| `pillow` | decode, encode, EXIF |
| `pillow-heif` | iPhone HEIC/HEIF |

Deliberately absent: **PyTorch**, **onnxruntime**, and any **ffmpeg** binary. YuNet runs through the DNN module OpenCV already ships, which is why the model is 227 KB instead of hundreds of megabytes.

### Install

The Quick Start above is the whole thing. Two other routes, if you want them:

```bash
git clone <this-repository> AmurCloak
cd AmurCloak && uv tool install .                       # from a checkout
uv tool install dist/amurcloak-0.9.0-py3-none-any.whl   # from a wheel you built
```

Upgrading and removing are each one command:

```bash
uv tool upgrade amurcloak
uv tool uninstall amurcloak
```

`uv tool install` gives each tool its own isolated environment and drops a small shim on your `PATH`. It does not touch your system Python, and uninstalling leaves nothing behind.

### Working on the source

The package lives under `src/`, so it imports without being installed:

```bash
uv run python -m amurcloak
```

`requirements.txt` carries the same pins for anyone using plain `pip` instead of uv.

### Verify the model

The detector is vendored inside the package, and its checksum is verified on startup:

```bash
shasum -a 256 src/amurcloak/models/face_detection_yunet_2023mar.onnx
# 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

---

## Configuration

Everything is a command-line flag. There are no environment variables and no config file.

```bash
amurcloak [flags]
```

Running `amurcloak` with no flags picks the port and opens the browser — that is the intended way to use it. The flags exist for the corners.

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Localhost only — do not expose this. |
| `--port` | first free from `5000` | Pass one explicitly to pin it. |
| `--tmp` | the system temp directory | Where uploads and renders are staged. Never inside the package. |
| `--no-open` | off | Do not open the browser. For SSH, scripts and headless runs. |
| `--ttl` | `1800` | Seconds of inactivity before a session is reaped. |
| `--max-age` | `21600` | Hard limit on a session's age, in seconds. |
| `--shred` | off | Zero-fill temp files before unlinking. Slower; costs a full write pass. |
| `--keep-inputs` | off | Keep staged uploads after download instead of freeing them. |
| `--allow-download` | off | Permit a one-time model fetch if the vendored model is missing. |
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

The page is three columns and the rule never changes: your **photos and the export button** on the left, the **picture** in the middle, and **every setting** on the right, in the order you would reach for them — how hard to look, how to cover, how wide, and what the file claims. The left column's button is the only primary action, and it stays put: it becomes Redact, then the progress bar, then Download, in the same place every time.

1. **Drop photos** anywhere on the page, paste a screenshot, or click **Choose photos**.
2. **Check the audit map.** Every detection is numbered and scored. Switch to **Result** to see the actual redaction.
3. **Leave someone visible, if you want to.** Click a covered face to give that person their pixels back; click again to cover them.
4. **Pick a cover mode.** Solid black is selected by default. Adjust **Coverage** to push the mask further past the face, and **Strength** to change the blur or block size — the latter is disabled in black mode, because solid black ignores it.
5. **Choose how hard to look.** Cautious, Balanced or Thorough, which trades false positives against missed faces.
6. **Decide what the file says about itself.** **Removed** drops all metadata; **Forged** replaces it with a date you choose, a made-up camera and a location in the middle of the ocean.
7. **Redact photos**, then **Download**. One photo downloads as a single image; several are bundled into a ZIP.

Settings apply to the whole staged set, and changing any of them — or leaving a different set of faces visible — invalidates a rendered result rather than letting you download pixels that no longer match what is on screen.

A batch is held in the running server's memory, so quitting `amurcloak` — or leaving a batch untouched past `--ttl` — clears it. The page notices and asks for the photos again rather than leaving a stale preview up, which matters because a frame that no longer matches its settings is worse than no frame at all.

### Leaving someone visible

Every face starts covered, which is the only safe default. Click a face in the preview — the audit map and the result view carry the same click targets — and that person keeps their own pixels in the export. The ring turns green with a `visible` tag under it; click again to cover them, or **Cover everyone again** to reset the photo.

Three details are deliberate:

- **It is per photo.** Faces are never matched across a batch. Two photos of the same person are two independent decisions, because the alternative is exactly the face recognition this tool exists to avoid.
- **The decision is not painted into the image.** The green ring and the tag are browser overlays. Nothing in the exported file hints that a face was spared, and the audit map marks a spared face differently for the same reason.
- **It invalidates a rendered result**, just as changing a setting does, so a download can never disagree with the screen.

### Metadata

Two policies. The default is the one that says nothing at all:

| Policy | What the exported file carries |
|---|---|
| **Removed** (default) | No EXIF, no GPS, no XMP, no IPTC, no maker notes, no thumbnail. The ICC profile is kept so the colours are right. |
| **Forged** | The same stripping, then a false block: the date you choose on `DateTime`, `DateTimeOriginal` and `DateTimeDigitized` alike; `Make`/`Model` set to `Apple`/`iPhone 13`; a full iPhone 13 exposure set — Exif 2.32, the f/1.6 lens, 26 mm-equivalent focal length, exposure time, ISO, colour space and the image's own pixel dimensions; and a location pinned to 48°52′36″S 123°23′36″W — Point Nemo, the most remote point in the ocean, picked precisely so the coordinates can never name a house. |

The forged block is **built from nothing rather than copied and edited**, and that detail is the whole design. An EXIF block can carry a *thumbnail*, which is a small copy of the unredacted photo, so a spoofing pass that edited the original's tags would ship the face it had just covered inside the metadata. The tests assert the output holds exactly one JPEG stream and no thumbnail directory.

It is also **complete, not just false.** A file that says `Apple / iPhone 13` and then stops is its own kind of tell — no camera records a lens and an aperture and then omits the rest — so the forgery carries what a real iPhone 13 would: `ExifVersion` 0232, the lens model string, the 35 mm-equivalent focal length, exposure time and ISO, the pixel dimensions of the image actually written, and the 72 dpi resolution in IFD0. The rule that governs all of it is internal consistency. Shutter speed and aperture are stored in the logarithmic APEX form, so they are *derived* from the exposure time and f-number rather than guessed — a block whose stated aperture contradicts its own computed one is the loudest thing a forgery can carry — and the tests check that arithmetic, along with the recorded dimensions matching the delivered image.

What it is, and is not:

- **It is false metadata, not the absence of metadata.** A missing block and a consistent one read differently to a tool that looks — that is a difference in suspicion, not proof either way.
- **It is not anonymity.** The pixels are what identify a place and a person; all this removes is the file volunteering where and when it was taken.
- **Most platforms strip EXIF on upload anyway.** It matters for files sent directly — email, chat, a USB stick, a shared folder.
- **A `.tif` cannot carry any of the nested detail.** Pillow attaches a nested directory only through its own uncompressed writer, and libtiff rejects the offset it produces for a compressed file — which is true of the Exif sub-directory (`EXIFIFDOffset`) exactly as much as the GPS one. Rather than change the archival format's encoding to plant the detail, a spoofed TIFF keeps a flat IFD0: the camera, the resolution and all three timestamps, and nothing deeper.
- **The default stays `Removed`.** Stamping a false date on a photo is a claim about evidence, so it is something you opt into, not something done for you.

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
uv run python tests/test_engine.py   # 18/18 — detection, masking, colour, metadata, formats
uv run python tests/test_api.py      # 22/22 — upload, tune, preview, render, download, disk
```

The API suite drives the real Flask app through its test client, so it covers the actual routes rather than mocks. Its last four checks fetch the page, the script and the stylesheet and compare them against each other.

Notable coverage:

- Tiling recovers strictly more faces than a single pass on the group fixture.
- Pixelate and blur **keep the photo's colours** and never composite against black.
- Solid black writes pure black across the whole masked region.
- Every facial landmark falls inside the mask.
- Outputs carry no EXIF, GPS or camera metadata by default, and the ICC profile survives.
- Restoring one face leaves that face original **and** the others covered, checked against the same ellipse geometry the browser is handed.
- Restoring everyone hands back the photo's own pixels, exactly — the lossless path in and out.
- A forged block carries the chosen date on all three timestamp tags, the forged camera, a Point Nemo location and no thumbnail: a source file that arrives carrying a thumbnail of the original comes out with exactly one JPEG stream.
- The forged block reads as a real capture rather than a stub: the exposure and lens tags are present, the recorded pixel dimensions match the image actually written, and the APEX shutter and aperture values agree with the exposure time and f-number they were derived from.
- Only a box the detector actually found can be restored; repeating a click does not double-count, junk keys and indices are refused, and a rendered result is discarded when the visible set changes.
- Changing settings invalidates a stale result.
- The temp spool survives startup and a real multi-megabyte upload, and the orphan purge spares reserved folders.
- A batch that loses all but one photo at render still hands that survivor back, as a plain image rather than a zip.
- A non-finite slider value degrades to the default instead of reaching the mask geometry.
- The page, the script and the stylesheet agree with each other: every id the controller queries exists in the markup, every column the workspace declares is filled by a pane the page actually uses, the theme is decided before the stylesheet so it cannot flash, and no custom property is used without being defined — and none is defined only for the dark theme.

Several of these exist because they were real bugs: an upload that silently failed with a 500, a purge that deleted the directory Werkzeug spools uploads into, a batch that reported "ready" with no file behind it, a `"nan"` margin that could have reached the ellipse maths, and a renamed pane whose stylesheet kept the old class name — which cost the rail its grid area and its divider without a single error being raised.

---

## Repository Structure

```text
AmurCloak/
├── README.md
├── pyproject.toml                # metadata, the 5 dependencies, the entry point
├── requirements.txt              # the same pins, for plain pip
├── src/
│   └── amurcloak/
│       ├── __init__.py           # version
│       ├── __main__.py           # python -m amurcloak
│       ├── cli.py                # free-port search, already-running check, browser
│       ├── app.py                # Flask routes, config, flags, lifecycle
│       ├── engine/
│       │   ├── detector.py       # YuNet wrapper: tiling, NMS, ranking
│       │   ├── redact.py         # ellipse geometry, three modes, audit map
│       │   ├── photo.py          # decode/encode, EXIF, metadata policy
│       │   ├── pipeline.py       # full-resolution preview composition
│       │   ├── presets.py        # modes, presets, settings resolution
│       │   ├── sessions.py       # staging, disk accounting, renderer, reaper
│       │   └── __init__.py
│       ├── models/
│       │   └── face_detection_yunet_2023mar.onnx   # 227 KB, checksum verified
│       ├── templates/
│       │   └── index.html        # the single page
│       └── static/
│           ├── style.css         # the whole stylesheet, both themes as tokens
│           └── app.js            # the whole front end, no build step
├── assets/                       # logo and screenshots
└── tests/
    ├── test_engine.py            # 18 tests
    ├── test_api.py               # 22 tests
    └── fixtures/                 # lena.jpg, plus fixtures built on demand
```

---

## Contributing

This is a private tool, so there is no contribution process to speak of. If you do extend it, six rules keep it honest:

1. **Nothing may make an outbound network call at runtime.** That is the entire point of the project. The one request AmurCloak ever makes is a loopback probe of its own port, to tell "already running" apart from "free port" — it is pinned to `127.0.0.1` and deliberately ignores any proxy set in the environment, so it cannot become an outbound request by accident.
2. **A preview must never disagree with the delivered file.** Previews are composed at full resolution and only then downscaled — shrinking first would change the blur kernel and the block size. It is also why the preview and the export share one face list, restored faces included.
3. **Never describe a reversible redaction as safe.** If a mode can be partially undone, the interface has to say so.
4. **Build forged metadata from nothing; never edit the source's tags.** An EXIF block can carry a thumbnail, which is a small copy of the *original* photo — so the obvious "copy the tags and change a few" shortcut would ship the unredacted face inside the metadata of the file that covered it.
5. **Whatever the forged block claims, make it agree with itself.** A block that is merely sparse is a poor forgery, and a block that contradicts itself is a worse one: derive the APEX shutter and aperture values from the exposure time and f-number instead of guessing them, and record the dimensions of the image you actually wrote.
6. **Keep `index.html`, `style.css` and `app.js` in agreement, and let the tests hold you to it.** This front end has no build step, so nothing checks the seams: rename a pane, miss one of the three files, and the page still loads — it just quietly loses a control, or a rail loses the grid area that held it in place. The API suite compares the three files against each other for exactly that reason, and is the cheapest way to find out you renamed something twice.

Run both suites before committing; they are fast enough that there is no excuse not to.

---

## License

MIT — see [`LICENSE`](LICENSE). Use it, fork it, ship it.

One component is somebody else's work: the bundled face detector, `src/amurcloak/models/face_detection_yunet_2023mar.onnx`, is YuNet by Shiqi Yu, licensed separately under MIT. Its notice travels alongside the model at [`src/amurcloak/models/LICENSE`](src/amurcloak/models/LICENSE), which is what redistribution requires.

---

<div align="center">
<sub>Built to be used once and trusted: the photos go in, the faces come out covered, and nothing in between leaves the machine.</sub>
</div>
