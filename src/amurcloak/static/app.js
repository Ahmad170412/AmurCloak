/* AmurCloak front end.
   No framework, no build step, no external requests - the page must keep
   working with the network unplugged, which is the whole point of the tool. */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const el = {
    themeToggle: $("#theme-toggle"),
    dropzone: $("#dropzone"),
    viewer: $("#viewer"),
    canvas: $("#viewer-canvas"),
    previewImg: $("#preview-img"),
    viewerLoading: $("#viewer-loading"),
    faceLayer: $("#face-layer"),
    pickHint: $("#pick-hint"),
    coverAll: $("#cover-all"),
    viewerName: $("#viewer-name"),
    viewerIndex: $("#viewer-index"),
    viewerNote: $("#viewer-note"),
    scrub: $("#scrub"),
    scrubRange: $("#scrub-range"),
    scrubTime: $("#scrub-time"),
    fileList: $("#file-list"),
    filesNote: $("#files-note"),
    railCount: $("#rail-count"),
    modeOptions: $("#mode-options"),
    modeNote: $("#mode-note"),
    presetOptions: $("#preset-options"),
    margin: $("#margin"),
    marginOut: $("#margin-out"),
    strength: $("#strength"),
    strengthOut: $("#strength-out"),
    strengthRow: $("#strength-row"),
    strengthNote: $("#strength-note"),
    metadataOptions: $("#metadata-options"),
    whenRow: $("#when-row"),
    spoofWhen: $("#spoof-when"),
    metadataNote: $("#metadata-note"),
    metadataFlag: $("#metadata-flag"),
    audioOptions: $("#audio-options"),
    audioNote: $("#audio-note"),
    audioGroup: $("#group-audio-clip"),
    renderBtn: $("#render-btn"),
    downloadBtn: $("#download-btn"),
    redactAgain: $("#redact-again"),
    resetBtn: $("#reset-btn"),
    progress: $("#progress"),
    progressFill: $("#progress-fill"),
    progressText: $("#progress-text"),
    actionNote: $("#action-note"),
    formatsLine: $("#formats-line"),
    pickFiles: $("#pick-files"),
    fileInput: $("#file-input"),
    dropcatcher: $("#dropcatcher"),
    staging: $("#staging"),
    stagingTitle: $("#staging-title"),
    stagingTrack: $("#staging-track"),
    stagingFill: $("#staging-fill"),
    stagingNote: $("#staging-note"),
    stagingCancel: $("#staging-cancel"),
    segs: document.querySelectorAll(".seg[data-view]"),
  };

  const state = {
    config: null,
    sid: null,
    snapshot: null,
    index: 0,
    view: "audit",
    t: 0, // where in the current clip the preview is (seconds)
    settings: {
      mode: "black",
      preset: "balanced",
      margin: 0.14,
      strength: 0.7,
      metadata: "strip",
      spoof_when: "",
      keep_audio: true,
    },
    faces: [], // the last preview's detections, for the click targets
    previewToken: 0,
    busy: false,
    uploading: false,
    pending: [], // files dropped but not yet acknowledged by the server
    uploadXhr: null,
    pollTimer: null,
    settingsTimer: null,
    previewTimer: null, // clip "scanning" re-asks
  };

  /* ── theme ─────────────────────────────────────────────────────
     The head script has already committed to a theme before the first paint,
     so this only mirrors that onto the button and remembers a deliberate
     choice. Until there is one, the page keeps following the OS. */

  const THEME_KEY = "amurcloak-theme";

  function storedTheme() {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch {
      return null; // storage disabled
    }
  }

  function paintTheme(dark) {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    el.themeToggle.setAttribute("aria-pressed", String(dark));
    el.themeToggle.title = dark ? "Switch to the light theme" : "Switch to the dark theme";
  }

  function wireTheme() {
    paintTheme(document.documentElement.dataset.theme === "dark");

    el.themeToggle.addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme !== "dark";
      try {
        localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
      } catch {
        // A choice we cannot store is still a choice for this visit.
      }
      paintTheme(dark);
    });

    const query = window.matchMedia("(prefers-color-scheme: dark)");
    if (query.addEventListener) {
      query.addEventListener("change", (event) => {
        if (!storedTheme()) paintTheme(event.matches);
      });
    }
  }

  /* ── helpers ─────────────────────────────────────────────────── */

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const message = (payload && payload.error) || `request failed (${response.status})`;
      const error = new Error(message);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    const digits = value >= 100 || unit === 0 ? 0 : 1;
    return `${value.toFixed(digits)} ${units[unit]}`;
  }

  function plural(n, one, many) {
    return n === 1 ? one : many;
  }

  /* ── controls ────────────────────────────────────────────────── */

  function renderModes() {
    el.modeOptions.innerHTML = "";
    for (const mode of state.config.modes) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "option";
      button.dataset.mode = mode.key;
      button.title = mode.detail;
      button.innerHTML = `
        <span class="swatch swatch-${mode.key}" aria-hidden="true"></span>
        <span class="option-body">
          <span class="option-label">${mode.label}</span>
          <span class="option-hint">${mode.blurb}</span>
        </span>`;
      button.addEventListener("click", () => {
        state.settings.mode = mode.key;
        syncControls();
        pushSettings();
      });
      el.modeOptions.appendChild(button);
    }
  }

  function renderPresets() {
    el.presetOptions.innerHTML = "";
    for (const preset of state.config.presets) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "seg";
      button.dataset.preset = preset.key;
      button.textContent = preset.label;
      button.title = preset.detail;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        state.settings.preset = preset.key;
        // Adopt the preset's own idea of a sensible coverage *before* the
        // repaint, so the slider track and its readout agree with the value.
        setMarginSlider(preset.margin);
        syncControls();
        pushSettings();
      });
      el.presetOptions.appendChild(button);
    }
  }

  const METADATA_LABELS = [
    ["strip", "Removed"],
    ["spoof", "Forged"],
  ];

  /* What each policy means, in the user's terms rather than the spec's. */
  const METADATA_COPY = {
    strip:
      "Everything goes: EXIF, GPS, camera and timestamps. The copy says nothing about where or when it was taken.",
    spoof:
      "Everything goes, then a false block is written in its place: the date below, a made-up camera, and a location fixed to the emptiest point in the Pacific - far enough from any land that it can never name a home.",
  };

  function renderMetadataOptions() {
    el.metadataOptions.innerHTML = "";
    for (const [key, label] of METADATA_LABELS) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "seg";
      button.dataset.metadata = key;
      button.textContent = label;
      button.title = METADATA_COPY[key];
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        state.settings.metadata = key;
        syncControls();
        pushSettings();
      });
      el.metadataOptions.appendChild(button);
    }
  }

  /* What a clip's sound does. Kept by default and copied losslessly, with the
     second option a straight strip - a voice identifies a person as surely as
     a face, so the choice has to be visible rather than made by the tool. */
  const AUDIO_LABELS = [
    ["keep", "Keep"],
    ["remove", "Remove"],
  ];

  const AUDIO_COPY = {
    keep:
      "The clip's own sound stays, copied unchanged from the original file. Toggle this off when the voice is part of what should be hidden.",
    remove:
      "No sound in the exported clip. A voice can identify someone just as a face can.",
  };

  function renderAudioOptions() {
    el.audioOptions.innerHTML = "";
    for (const [key, label] of AUDIO_LABELS) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "seg";
      button.dataset.audio = key;
      button.textContent = label;
      button.title = AUDIO_COPY[key];
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        state.settings.keep_audio = key === "keep";
        syncControls();
        pushSettings();
      });
      el.audioOptions.appendChild(button);
    }
  }

  function setMarginSlider(margin) {
    state.settings.margin = margin;
    el.margin.value = String(Math.round(margin * 100));
  }

  function paintRange(input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const pct = max === min ? 0 : ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${pct}%`);
  }

  function syncControls() {
    for (const button of el.modeOptions.children) {
      button.classList.toggle("is-on", button.dataset.mode === state.settings.mode);
      button.setAttribute("aria-pressed", String(button.dataset.mode === state.settings.mode));
    }
    for (const button of el.presetOptions.children) {
      button.classList.toggle("is-on", button.dataset.preset === state.settings.preset);
      button.setAttribute("aria-pressed", String(button.dataset.preset === state.settings.preset));
    }

    el.marginOut.textContent = `${Math.round(state.settings.margin * 100)}%`;
    el.strengthOut.textContent = `${Math.round(state.settings.strength * 100)}%`;
    paintRange(el.margin);
    paintRange(el.strength);

    const isBlack = state.settings.mode === "black";
    el.strength.disabled = isBlack;
    el.strengthRow.style.opacity = isBlack ? "0.6" : "1";
    el.strengthNote.textContent = isBlack
      ? "Solid black ignores strength - the pixels are gone, not obscured."
      : state.settings.mode === "blur"
        ? "Higher means a wider blur. Blur can be partly reversed, so treat it as the weakest option."
        : "Higher means chunkier blocks. Un-pixelation is possible on stills.";

    const mode = state.config.modes.find((m) => m.key === state.settings.mode);
    el.modeNote.textContent = mode ? mode.detail : "";

    for (const button of el.metadataOptions.children) {
      const on = button.dataset.metadata === state.settings.metadata;
      button.classList.toggle("is-on", on);
      button.setAttribute("aria-pressed", String(on));
    }
    const forging = state.settings.metadata === "spoof";
    el.whenRow.hidden = !forging;
    if (state.settings.spoof_when) el.spoofWhen.value = state.settings.spoof_when;
    el.metadataNote.textContent = METADATA_COPY[state.settings.metadata] || "";

    for (const button of el.audioOptions.children) {
      const on = (button.dataset.audio === "keep") === state.settings.keep_audio;
      button.classList.toggle("is-on", on);
      button.setAttribute("aria-pressed", String(on));
    }
    el.audioNote.textContent = state.settings.keep_audio ? AUDIO_COPY.keep : AUDIO_COPY.remove;
  }

  function pushSettings() {
    if (!state.sid) return;
    // Remember which batch this change belongs to. If the batch is swapped
    // before the debounce fires, or while the POST is in flight, the reply
    // describes a session that is no longer on screen and must be dropped -
    // otherwise a stale 404 would tear down the *new* batch.
    const sid = state.sid;
    clearTimeout(state.settingsTimer);
    state.settingsTimer = setTimeout(async () => {
      if (sid !== state.sid) return;
      try {
        await api(`/api/sessions/${sid}/settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode: state.settings.mode,
            preset: state.settings.preset,
            margin: state.settings.margin,
            strength: state.settings.strength,
            metadata: state.settings.metadata,
            spoof_when: state.settings.spoof_when,
            keep_audio: state.settings.keep_audio,
          }),
        });
        if (sid === state.sid) {
          // The "Heads up" note is the one piece of copy that reads the photos
          // *and* the policy, so switching between Removed and Forged has to
          // redraw it now. The next snapshot may be minutes away.
          renderMetadataFlag();
          refreshPreview();
        }
      } catch (error) {
        if (sid !== state.sid) return;
        if (isSessionGone(error)) {
          sessionLost();
          return;
        }
        // The controls now say one thing and the pixels on screen say another,
        // so clear the frame rather than let a redaction sit under settings it
        // was not made with.
        el.previewImg.removeAttribute("src");
        clearViewerCaption();
        setNote(error.message);
      }
    }, 130);
  }

  /* ── preview ─────────────────────────────────────────────────── */

  function setNote(text) {
    el.viewerNote.textContent = text || "";
  }

  /** A frame with no image must not keep a name sitting next to it. */
  function clearViewerCaption() {
    el.viewerName.textContent = "";
    el.viewerIndex.textContent = "";
  }

  function isSessionGone(error) {
    // Every session-scoped route answers 404 once the store has forgotten the
    // session, so a 404 here means "this batch is gone", not "bad request".
    return Boolean(error) && error.status === 404;
  }

  /**
   * The server keeps batches in memory. Restarting it, or leaving one idle long
   * enough for the reaper, loses the staged photos, and every later call for
   * that session 404s. Nothing can be recovered, so tear the view down
   * coherently and say so.
   *
   * Ignoring this is worse than useless. A mode click would 404 on /settings,
   * silently fail, and leave the *previous* redaction on screen under the new
   * settings - a frame that disagrees with the controls beside it. And a view
   * toggle would 404 on /preview and blank the image while its name stayed put,
   * which reads as a bug rather than as "your batch is gone".
   */
  function sessionLost() {
    if (!state.sid && !state.snapshot) return; // nothing to tear down
    reset({ quiet: true }).then(() => {
      el.actionNote.textContent =
        "This batch is no longer on the server. It starts empty and clears a batch left idle, so the photos need to go in again.";
    });
  }

  function isClipItem(item) {
    return Boolean(item && item.kind === "video");
  }

  function currentItem() {
    if (!state.snapshot || !state.snapshot.items.length) return null;
    return state.snapshot.items[Math.max(0, Math.min(state.index, state.snapshot.items.length - 1))] || null;
  }

  async function refreshPreview() {
    if (!state.sid || !state.snapshot || !state.snapshot.items.length) return;
    const token = ++state.previewToken;
    const item = currentItem();
    const clip = isClipItem(item);
    el.viewerLoading.hidden = false;

    try {
      const tParam = clip ? `&t=${encodeURIComponent(state.t)}` : "";
      const payload = await api(
        `/api/sessions/${state.sid}/preview?index=${state.index}&view=${state.view}${tParam}`
      );
      if (token !== state.previewToken) return; // a newer request already won

      if (payload.scanning) {
        // A clip is being watched for the first time: the whole run of frames
        // has to be seen once before any frame can be shown, so no image yet -
        // say so, keep the spinner, and ask again shortly. A pull loop, not a
        // push: nothing here holds a request open for a minute.
        el.previewImg.removeAttribute("src");
        hideFaces();
        if (payload.scan_error) {
          el.viewerNote.textContent = `The clip could not be read: ${payload.scan_error}`;
          el.viewerLoading.hidden = true;
          return;
        }
        el.viewerNote.textContent =
          "Finding faces in the clip from end to end... this runs once, then the whole clip scrubs freely.";
        state.previewTimer = setTimeout(() => {
          if (token !== state.previewToken) return;
          refreshPreview();
        }, 600);
        return;
      }

      el.previewImg.src = `data:image/jpeg;base64,${payload.image}`;
      el.viewerNote.textContent = payload.note;
      renderFaces(payload);
      syncScrub(clip, payload);

      // The rail learns the face count for whatever was just checked, so it can
      // report real numbers without scanning every photo up front.
      if (item && typeof payload.count === "number") {
        if (item.faces !== payload.count || !item.scanned) {
          item.faces = payload.count;
          item.scanned = true;
          renderFiles();
        }
      }
    } catch (error) {
      if (token !== state.previewToken) return;
      if (isSessionGone(error)) {
        sessionLost();
        return;
      }
      // No image, so no caption either: a named empty frame reads as a bug.
      el.previewImg.removeAttribute("src");
      clearViewerCaption();
      hideFaces();
      setNote(error.message);
    } finally {
      if (token === state.previewToken) el.viewerLoading.hidden = true;
    }
  }

  /* The scrub is clips-only, and the one place the preview's timestamp lives.
     Positions are recomputed from the payload the server answered with, so the
     bar never claims a frame that was not the one rendered. */
  function syncScrub(clip, payload) {
    el.scrub.hidden = !clip;
    if (!clip) return;
    const duration = Number(
      (payload && payload.meta && payload.meta.duration) || 0
    );
    if (duration > 0) {
      el.scrubRange.max = "1000";
      el.scrubRange.dataset.duration = String(duration);
      const pct = Math.max(0, Math.min(1000, Math.round((state.t / duration) * 1000)));
      el.scrubRange.value = String(pct);
      el.scrubTime.textContent = `${formatClock(state.t)} / ${formatClock(duration)}`;
    }
  }

  function formatClock(seconds) {
    const s = Math.max(0, seconds || 0);
    const whole = Math.floor(s);
    const tenth = Math.floor((s - whole) * 10);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}.${tenth}`;
  }

  /* ── picking people ──────────────────────────────────────────── */
  //
  // Everything starts covered. Clicking a covered person leaves them visible;
  // clicking them again puts the cover back. The click targets are the ellipses
  // the server actually painted, published as percentages of the photo, so the
  // shape you click is the shape that was masked - never an approximation.

  /** Line the face layer up with the rendered image, whatever its size. */
  function syncFaceLayer() {
    const img = el.previewImg;
    if (!img.getAttribute("src") || !img.naturalWidth || el.viewer.hidden) {
      el.faceLayer.hidden = true;
      return;
    }
    // offsetLeft/Top are measured from the canvas's padding box, which is also
    // the containing block for an absolutely positioned child, so this lands on
    // the image without having to reason about borders or letterboxing.
    el.faceLayer.style.left = `${img.offsetLeft}px`;
    el.faceLayer.style.top = `${img.offsetTop}px`;
    el.faceLayer.style.width = `${img.offsetWidth}px`;
    el.faceLayer.style.height = `${img.offsetHeight}px`;
    el.faceLayer.hidden = false;
  }

  // The strip itself stays in the layout whether or not it has anything to
  // say: the canvas is measured off what is left over, so a strip that came and
  // went would resize the picture between a photo with faces and one without.
  function hideFaces() {
    state.faces = [];
    el.faceLayer.innerHTML = "";
    el.faceLayer.hidden = true;
    el.pickHint.textContent = "";
    el.coverAll.hidden = true;
  }

  function pickHint(restored) {
    if (!restored) return "Click a covered face to leave that person visible.";
    return restored === 1
      ? "1 person left visible. Click them to cover them again."
      : `${restored} people left visible. Click one to cover them again.`;
  }

  function renderFaces(payload) {
    const faces = payload.faces || [];
    const width = payload.width || 0;
    const height = payload.height || 0;
    state.faces = faces;
    el.faceLayer.innerHTML = "";

    if (!faces.length || !width || !height) {
      el.faceLayer.hidden = true;
      el.pickHint.textContent = "";
      el.coverAll.hidden = true;
      return;
    }

    for (const face of faces) {
      const mask = face.mask;
      if (!mask) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "face-hit";
      if (!face.covered) {
        button.classList.add("is-visible");
        if (mask.cy / height > 0.88) button.classList.add("is-low");
      }
      button.style.left = `${(mask.cx / width) * 100}%`;
      button.style.top = `${(mask.cy / height) * 100}%`;
      button.style.width = `${((mask.ax * 2) / width) * 100}%`;
      button.style.height = `${((mask.ay * 2) / height) * 100}%`;
      button.style.transform = `translate(-50%, -50%) rotate(${mask.angle}deg)`;
      button.title = face.covered
        ? "Covered. Click to leave this person visible."
        : "Left visible. Click to cover them again.";
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", () => toggleFace(face));
      el.faceLayer.appendChild(button);
    }

    const restored = faces.filter((face) => !face.covered).length;
    el.pickHint.textContent = pickHint(restored);
    el.coverAll.hidden = restored === 0;
    syncFaceLayer();
  }

  async function toggleFace(face) {
    if (!state.sid) return;
    const sid = state.sid;
    try {
      const snapshot = await api(`/api/sessions/${sid}/faces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index: state.index,
          key: face.key,
          covered: !face.covered,
        }),
      });
      if (sid !== state.sid) return;
      // The reply is a full snapshot: the restore just invalidated any rendered
      // result, and the rail has to stop offering a download it no longer has.
      applySnapshot(snapshot);
      await refreshPreview();
    } catch (error) {
      if (sid !== state.sid) return;
      if (isSessionGone(error)) {
        sessionLost();
        return;
      }
      setNote(error.message);
    }
  }

  async function coverEveryone() {
    if (!state.sid) return;
    const visible = state.faces.filter((face) => !face.covered);
    if (!visible.length) return;
    const sid = state.sid;
    el.coverAll.disabled = true;
    try {
      let snapshot = null;
      for (const face of visible) {
        snapshot = await api(`/api/sessions/${sid}/faces`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: state.index, key: face.key, covered: true }),
        });
        if (sid !== state.sid) return;
      }
      if (snapshot) applySnapshot(snapshot);
      await refreshPreview();
    } catch (error) {
      if (sid !== state.sid) return;
      if (isSessionGone(error)) {
        sessionLost();
        return;
      }
      setNote(error.message);
    } finally {
      el.coverAll.disabled = false;
    }
  }

  async function selectIndex(index) {    if (!state.snapshot || !state.snapshot.items.length) return;
    state.index = Math.max(0, Math.min(index, state.snapshot.items.length - 1));
    state.t = 0; // each file starts at its own beginning
    renderFiles();
    const item = state.snapshot.items[state.index];
    el.viewerName.textContent = item.name;
    el.viewerIndex.textContent = `${state.index + 1} of ${state.snapshot.items.length}`;
    refreshPreview();
  }

  /* ── files rail ──────────────────────────────────────────────── */

  function renderFiles() {
    const snapshot = state.snapshot;
    const count = snapshot && snapshot.items.length ? snapshot.items.length : state.pending.length;
    el.railCount.hidden = count === 0;
    el.railCount.textContent = count ? String(count) : "";
    if (!snapshot || !snapshot.items.length) {
      if (state.pending.length) {
        renderPendingRows();
        return;
      }
      el.fileList.innerHTML = "";
      el.filesNote.textContent = "Nothing staged yet.";
      return;
    }

    el.fileList.innerHTML = "";

    snapshot.items.forEach((item, index) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "file";
      if (index === state.index) button.classList.add("is-current");
      if (item.status === "error") button.classList.add("is-failed");

      const isClip = isClipItem(item);
      let chip;
      if (item.status === "error") chip = `<span class="chip chip-failed">failed</span>`;
      else if (isClip && item.scan_state === "running" && item.status !== "done")
        chip = `<span class="chip chip-working">watching</span>`;
      else if (item.status === "done") chip = `<span class="chip chip-done">redacted</span>`;
      else if (snapshot.state === "rendering") chip = `<span class="chip chip-working">working</span>`;
      else chip = "";

      const clipText = isClip && item.meta && item.meta.duration
        ? `${Math.ceil(item.meta.duration)}s clip`
        : "";
      const faceText = item.scanned
        ? `${item.faces} ${plural(item.faces, "face", "faces")}` +
          (item.restored ? ` · ${item.restored} left visible` : "")
        : "";

      const statusBits = [chip, clipText && `<span>${clipText}</span>`].filter(Boolean);
      statusBits.push(faceText ? `<span>${faceText}</span>` : "");

      button.innerHTML = `
        <span class="file-num">${index + 1}</span>
        <span class="file-name">${escapeHtml(item.name)}</span>
        <span class="file-size">${formatBytes(item.size)}</span>
        <span class="file-status">${statusBits.join("")}</span>`;
      button.addEventListener("click", () => selectIndex(index));
      li.appendChild(button);
      el.fileList.appendChild(li);
    });

    // The one place disk footprint is reported: it describes *these* photos.
    // Free space on the volume is not shown - the OS already tells you that, and
    // nothing here acts on it. The upload guard errors with the number if it
    // ever actually matters.
    const failed = snapshot.items.filter((i) => i.status === "error").length;
    const total = snapshot.items.length;
    const checked = snapshot.items.filter((i) => i.scanned).length;
    const totalFaces = snapshot.items.reduce((sum, i) => sum + (i.faces || 0), 0);
    const hasClips = snapshot.items.some(isClipItem);

    const parts = [];
    if (checked) {
      // "0 faces found" is the single most important thing this tool can tell
      // you, so it is reported even when the answer is nothing.
      parts.push(
        `${totalFaces} ${plural(totalFaces, "face", "faces")} found` +
          (checked < total ? " so far" : "")
      );
    }
    if (failed) parts.push(`${failed} could not be read`);
    if (snapshot.downloaded) {
      parts.push("staged copies freed");
    } else if (snapshot.disk && snapshot.disk.total) {
      parts.push(`${formatBytes(snapshot.disk.total)} on disk`);
    }

    el.filesNote.textContent =
      parts.join(", ") || (hasClips ? "Click a file to check it before redacting." : "Click a photo to check it before redacting.");
  }

  // Placeholder rows for files that are still in flight. They exist so a drop
  // is acknowledged the instant it lands, even on a slow transfer.
  function renderPendingRows() {
    el.fileList.innerHTML = "";
    state.pending.forEach((file, index) => {
      const li = document.createElement("li");
      const row = document.createElement("span");
      row.className = "file is-staging";
      row.innerHTML = `
        <span class="file-num">${index + 1}</span>
        <span class="file-name">${escapeHtml(file.name)}</span>
        <span class="file-size">${formatBytes(file.size)}</span>
        <span class="file-status"><span class="chip chip-staging">staging</span></span>`;
      li.appendChild(row);
      el.fileList.appendChild(li);
    });

    const count = state.pending.length;
    el.filesNote.textContent = `Staging ${count} ${plural(count, "photo", "photos")}.`;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
    );
  }

  function renderMetadataFlag() {
    const snapshot = state.snapshot;
    if (!snapshot || !snapshot.items.length) {
      el.metadataFlag.hidden = true;
      return;
    }
    const withGps = snapshot.items.filter((i) => i.meta && i.meta.has_gps).length;
    const withExif = snapshot.items.filter((i) => i.meta && i.meta.has_exif).length;
    if (!withGps && !withExif) {
      el.metadataFlag.hidden = true;
      return;
    }
    el.metadataFlag.hidden = false;
    const bits = [];
    if (withGps) {
      bits.push(
        plural(
          withGps,
          "1 photo carries a location stamp",
          `${withGps} photos carry location stamps`
        )
      );
    }
    if (withExif) {
      bits.push(
        plural(
          withExif,
          "1 photo carries camera info",
          `${withExif} photos carry camera info`
        )
      );
    }
    // Deliberately says nothing about the policy. What happens to the metadata
    // is answered by the note under the control it belongs to, and a line here
    // that grew and shrank with the policy would shove that control around.
    el.metadataFlag.textContent = `Heads up: ${bits.join(" and ")}.`;
  }

  function namingFor(items) {
    /** What to call a mixed batch, in one word: photos, clips, or files. */
    const clips = items.filter(isClipItem).length;
    const photos = items.length - clips;
    if (clips && photos) return "files";
    if (clips) return plural(clips, "clip", "clips");
    return plural(photos, "photo", "photos");
  }

  function renderActions() {
    const snapshot = state.snapshot;
    const hasItems = Boolean(snapshot && snapshot.items.length);
    const live = hasItems ? snapshot.items.filter((i) => i.status !== "error") : [];
    const renderable = live.length;

    el.renderBtn.hidden = false;
    el.downloadBtn.hidden = true;
    el.redactAgain.hidden = true;
    el.progress.hidden = true;
    el.renderBtn.disabled = !hasItems || renderable === 0 || state.busy;

    // The clip control only earns its place when a batch actually holds clips.
    el.audioGroup.hidden = !hasItems || !live.some(isClipItem);

    if (!hasItems) {
      el.renderBtn.textContent = "Redact photos";
      el.actionNote.textContent = "";
      return;
    }

    const rendering = snapshot.state === "rendering" || snapshot.state === "queued";

    if (rendering) {
      el.renderBtn.hidden = true;
      el.progress.hidden = false;
      const total = snapshot.total || 1;
      const done = snapshot.progress || 0;
      const noun = namingFor(live);
      if (noun === "clips" || noun === "files") {
        // Clips step continuously, so the honest unit is the bar itself;
        // "rendering 0 of 2" would sit on 0 for a minute and read as hung.
        const pct = Math.min(100, Math.round((done / total) * 100));
        el.progressFill.style.width = `${pct}%`;
        el.progressText.textContent = `Redacting ${noun} ${pct}%`;
      } else {
        el.progressFill.style.width = `${Math.round((done / total) * 100)}%`;
        el.progressText.textContent = `Redacting ${Math.min(done + (done < total ? 1 : 0), total)} of ${total}`;
      }
      el.actionNote.textContent = "";
      return;
    }

    if (snapshot.state === "ready" && snapshot.result.ready) {
      const count = snapshot.items.filter((i) => i.status === "done").length;
      const noun = namingFor(snapshot.items.filter((i) => i.status === "done"));
      const freed = Boolean(snapshot.disk && snapshot.disk.inputs_freed);
      // One primary at a time, and it is whichever comes next. Download takes
      // the slot Redact just vacated instead of stacking a second amber button,
      // so the eye is never asked which of two equals to press.
      el.renderBtn.hidden = true;
      el.downloadBtn.hidden = false;
      el.downloadBtn.textContent = `Download ${count} ${noun}`;
      // Re-rendering reads the staged copies. Once those have been freed there
      // is nothing left to re-check a face against, and the button would only
      // produce an error.
      el.redactAgain.hidden = false;
      el.redactAgain.disabled = state.busy || freed;
      el.actionNote.textContent = snapshot.downloaded
        ? "Staged copies were freed to reclaim space. Drop the files again to redact them differently."
        : `Output is ${formatBytes(snapshot.result.size)}. Staged copies are freed once you download.`;
      return;
    }

    if (snapshot.state === "error") {
      el.renderBtn.textContent = "Try again";
      el.renderBtn.disabled = false;
      el.actionNote.textContent = snapshot.error || "Something went wrong.";
      return;
    }

    el.renderBtn.textContent = `Redact ${renderable} ${namingFor(live)}`;
    el.actionNote.textContent =
      "Faces are covered on this machine. Nothing is sent anywhere.";
  }

  function applySnapshot(snapshot) {
    state.snapshot = snapshot;
    if (state.index >= snapshot.items.length) {
      state.index = Math.max(0, snapshot.items.length - 1);
    }

    const hasItems = snapshot.items.length > 0;
    el.dropzone.hidden = hasItems;
    el.viewer.hidden = !hasItems;
    el.resetBtn.hidden = !hasItems;
    el.scrub.hidden = !hasItems || !isClipItem(currentItem());

    renderFiles();
    renderMetadataFlag();
    renderActions();

    // Once the staged copies are freed there is nothing left to re-check a face
    // against, so the click targets go too. Leaving them up would offer a button
    // whose only possible answer is an error.
    if (hasItems && snapshot.disk && snapshot.disk.inputs_freed) hideFaces();

    if (hasItems) {
      const item = snapshot.items[state.index];
      el.viewerName.textContent = item.name;
      el.viewerIndex.textContent = `${state.index + 1} of ${snapshot.items.length}`;
    }
  }

  /* ── upload ──────────────────────────────────────────────────── */

  function showStaging(title) {
    el.staging.hidden = false;
    el.stagingTitle.textContent = title;
    el.stagingNote.textContent = "";
    el.stagingNote.classList.remove("is-error");
    el.stagingCancel.hidden = false;
    el.stagingCancel.textContent = "Cancel";
    el.stagingFill.style.width = "";
    el.stagingFill.style.background = "";
    el.stagingTrack.classList.add("is-indeterminate");
  }

  function hideStaging() {
    el.staging.hidden = true;
    el.stagingTrack.classList.remove("is-indeterminate");
    el.stagingFill.style.width = "";
    el.stagingFill.style.background = "";
    el.stagingNote.classList.remove("is-error");
  }

  function setStagingNote(text, { error = false } = {}) {
    el.stagingNote.textContent = text || "";
    el.stagingNote.classList.toggle("is-error", error);
  }

  // XHR rather than fetch: only XHR reports upload progress, and progress is the
  // difference between "working" and "nothing happened".
  function uploadRequest(form) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      state.uploadXhr = xhr;
      xhr.open("POST", "/api/sessions");
      xhr.responseType = "json";

      xhr.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) return;
        const pct = Math.round((event.loaded / event.total) * 100);
        el.stagingTrack.classList.remove("is-indeterminate");
        el.stagingFill.style.width = `${pct}%`;
        setStagingNote(`Sending ${pct}% of ${formatBytes(event.total)}`);
      });

      // Bytes are up, but the server is still decoding. The remaining time is
      // unknown, so the bar stops claiming to know.
      xhr.upload.addEventListener("load", () => {
        el.stagingTrack.classList.add("is-indeterminate");
        setStagingNote("Reading the photos and finding faces");
      });

      const settle = (fn) => {
        state.uploadXhr = null;
        fn();
      };

      xhr.addEventListener("load", () =>
        settle(() => {
          const payload = xhr.response;
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(payload);
            return;
          }
          const error = new Error(
            (payload && payload.error) || `the upload failed (${xhr.status})`
          );
          error.status = xhr.status;
          reject(error);
        })
      );
      xhr.addEventListener("error", () =>
        settle(() => reject(new Error("the upload could not reach the server")))
      );
      xhr.addEventListener("abort", () =>
        settle(() => {
          const error = new Error("cancelled");
          error.aborted = true;
          reject(error);
        })
      );

      xhr.send(form);
    });
  }

  function clearStagedView() {
    el.dropzone.hidden = true;
    el.viewer.hidden = true;
    el.resetBtn.hidden = true;
    el.metadataFlag.hidden = true;
    el.previewImg.removeAttribute("src");
    // Drop the previous batch's caption and note too: the viewer is hidden now,
    // but it becomes visible again the instant the new batch lands, and a note
    // describing the *old* photos under the new file's name is a lie.
    clearViewerCaption();
    hideFaces();
    setNote("");
  }

  async function upload(files) {
    const list = Array.from(files).filter((f) => f && f.size > 0);
    if (!list.length || state.uploading) return;

    // Anything already staged is replaced (a staged set is accepted and
    // rendered as one unit), but the cleanup happens in the background so it
    // never delays the drop.
    const previous = state.sid;
    clearTimeout(state.pollTimer);
    clearTimeout(state.settingsTimer);
    clearTimeout(state.previewTimer);

    state.uploading = true;
    state.sid = null;
    state.snapshot = null;
    state.index = 0;
    state.previewToken += 1; // any preview still in flight belongs to the old batch
    state.pending = list.map((file) => ({
      name: file.name || "photo",
      size: file.size,
    }));

    if (previous) {
      api(`/api/sessions/${previous}`, { method: "DELETE" }).catch(() => {
        /* the reaper will get it */
      });
    }

    // Acknowledge the drop immediately: placeholder rows plus an overlay, so
    // there is never a silent gap between dropping and seeing anything.
    clearStagedView();
    renderFiles();
    renderActions();
    showStaging(`Staging ${list.length} ${plural(list.length, "photo", "photos")}`);

    const form = new FormData();
    for (const file of list) form.append("files", file, file.name);
    form.append(
      "settings",
      JSON.stringify({
        mode: state.settings.mode,
        preset: state.settings.preset,
        margin: state.settings.margin,
        strength: state.settings.strength,
        metadata: state.settings.metadata,
        spoof_when: state.settings.spoof_when,
        keep_audio: state.settings.keep_audio,
      })
    );

    try {
      const snapshot = await uploadRequest(form);
      state.pending = [];
      state.uploading = false;
      state.sid = snapshot.id;
      hideStaging();
      applySnapshot(snapshot);
      await selectIndex(0);
    } catch (error) {
      state.pending = [];
      state.uploading = false;

      if (error.aborted) {
        hideStaging();
        el.dropzone.hidden = false;
        renderFiles();
        renderActions();
        return;
      }

      // Keep the overlay up: it is the most visible place on the page, and an
      // upload failure must not look like an upload that did nothing.
      el.stagingTrack.classList.remove("is-indeterminate");
      el.stagingFill.style.width = "100%";
      el.stagingFill.style.background = "var(--rust)";
      el.stagingCancel.hidden = false;
      el.stagingCancel.textContent = "Dismiss";
      setStagingNote(error.message, { error: true });
      el.actionNote.textContent = error.message;
      renderFiles();
      renderActions();
    }
  }

  /* ── render + poll ───────────────────────────────────────────── */

  async function startRender() {
    if (!state.sid) return;
    state.busy = true;
    try {
      const snapshot = await api(`/api/sessions/${state.sid}/render`, { method: "POST" });
      applySnapshot(snapshot);
      pollUntilDone();
    } catch (error) {
      state.busy = false;
      if (isSessionGone(error)) {
        sessionLost();
        return;
      }
      el.actionNote.textContent = error.message;
    }
  }

  function pollUntilDone() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      if (!state.sid) return;
      try {
        const snapshot = await api(`/api/sessions/${state.sid}`);
        applySnapshot(snapshot);
        if (snapshot.state === "ready" || snapshot.state === "error") {
          state.busy = false;
          renderActions();
          return;
        }
        pollUntilDone();
      } catch (error) {
        state.busy = false;
        if (isSessionGone(error)) {
          sessionLost();
          return;
        }
        setNote(error.message);
      }
    }, 320);
  }

  function download() {
    if (!state.sid || !state.snapshot || !state.snapshot.result.ready) return;
    const name = state.snapshot.result.name || "amurcloak-output";
    const link = document.createElement("a");
    link.href = `/api/sessions/${state.sid}/result`;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Inputs are freed server-side on download; refresh so the disk readout agrees.
    setTimeout(() => pollOnce(), 900);
  }

  async function pollOnce() {
    if (!state.sid) return;
    try {
      applySnapshot(await api(`/api/sessions/${state.sid}`));
    } catch (error) {
      // This is the cheap liveness check, so it is also where a restarted or
      // reaped-away session gets noticed before the user touches anything.
      if (isSessionGone(error)) sessionLost();
    }
  }

  async function reset({ quiet = false } = {}) {
    clearTimeout(state.pollTimer);
    clearTimeout(state.settingsTimer);
    clearTimeout(state.previewTimer);
    const sid = state.sid;
    state.sid = null;
    state.snapshot = null;
    state.index = 0;
    state.busy = false;
    state.uploading = false;
    state.pending = [];
    state.previewToken += 1; // drop any preview still in flight for the old batch

    hideStaging();
    el.previewImg.removeAttribute("src");
    clearViewerCaption();
    hideFaces();
    el.dropzone.hidden = false;
    el.viewer.hidden = true;
    el.resetBtn.hidden = true;
    el.metadataFlag.hidden = true;
    renderFiles();
    renderActions();

    if (sid) {
      try {
        await api(`/api/sessions/${sid}`, { method: "DELETE" });
      } catch {
        /* reaped already */
      }
    }
    if (!quiet) setNote("");
  }

  /* ── drag & drop + paste ─────────────────────────────────────── */

  function wireDrops() {
    let depth = 0;

    const show = (on) => {
      el.dropcatcher.hidden = !on;
      const label = el.dropcatcher.querySelector("p");
      label.textContent = state.sid
        ? "Release to swap in these photos"
        : "Release to redact these photos";
    };

    window.addEventListener("dragenter", (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes("Files")) return;
      event.preventDefault();
      depth += 1;
      show(true);
    });
    window.addEventListener("dragover", (event) => {
      event.preventDefault();
    });
    window.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) show(false);
    });
    window.addEventListener("drop", (event) => {
      if (!event.dataTransfer) return;
      event.preventDefault();
      depth = 0;
      show(false);
      if (event.dataTransfer.files && event.dataTransfer.files.length) {
        upload(event.dataTransfer.files);
      }
    });

    window.addEventListener("paste", (event) => {
      const items = event.clipboardData && event.clipboardData.files;
      if (items && items.length) {
        event.preventDefault();
        upload(items);
      }
    });
  }

  function wireControls() {
    el.pickFiles.addEventListener("click", () => el.fileInput.click());
    el.fileInput.addEventListener("change", () => {
      if (el.fileInput.files.length) upload(el.fileInput.files);
      el.fileInput.value = "";
    });

    el.margin.addEventListener("input", () => {
      state.settings.margin = Number(el.margin.value) / 100;
      el.marginOut.textContent = `${el.margin.value}%`;
      paintRange(el.margin);
      pushSettings();
    });

    el.scrubRange.addEventListener("input", () => {
      const item = currentItem();
      const duration = Number(
        (item && item.meta && item.meta.duration) || el.scrubRange.dataset.duration || 0
      );
      if (duration <= 0) return;
      state.t = (Number(el.scrubRange.value) / 1000) * duration;
      el.scrubTime.textContent = `${formatClock(state.t)} / ${formatClock(duration)}`;
      refreshPreview();
    });

    el.strength.addEventListener("input", () => {
      state.settings.strength = Number(el.strength.value) / 100;
      el.strengthOut.textContent = `${el.strength.value}%`;
      paintRange(el.strength);
      pushSettings();
    });

    // "input" rather than "change": a datetime-local field fires "change" only
    // when it is committed, which native pickers do inconsistently, and a
    // half-filled field reads as an empty value that this guard drops. The
    // settings POST is debounced anyway, so an edit does not mean a request.
    el.spoofWhen.addEventListener("input", () => {
      if (el.spoofWhen.value) state.settings.spoof_when = el.spoofWhen.value;
      pushSettings();
    });

    // The image can be laid out later than the marks that describe it, so
    // re-measure whenever either of them moves.
    el.previewImg.addEventListener("load", syncFaceLayer);
    window.addEventListener("resize", syncFaceLayer);
    el.coverAll.addEventListener("click", coverEveryone);

    for (const seg of el.segs) {
      seg.addEventListener("click", () => {
        state.view = seg.dataset.view;
        for (const other of el.segs) other.classList.toggle("is-on", other === seg);
        refreshPreview();
      });
    }

    el.renderBtn.addEventListener("click", startRender);
    el.redactAgain.addEventListener("click", startRender);
    el.downloadBtn.addEventListener("click", download);
    el.resetBtn.addEventListener("click", () => reset());

    el.stagingCancel.addEventListener("click", () => {
      if (state.uploadXhr) {
        state.uploadXhr.abort();
        return;
      }
      hideStaging();
      el.dropzone.hidden = false;
      renderFiles();
      renderActions();
    });

    window.addEventListener("keydown", (event) => {
      if (!state.snapshot || !state.snapshot.items.length) return;
      const tag = (event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (event.key === "ArrowLeft") selectIndex(state.index - 1);
      if (event.key === "ArrowRight") selectIndex(state.index + 1);
    });

    // Coming back to the tab is where a stale batch is most likely and most
    // confusing - the window may have been sitting there across a restart. So
    // check the batch is still there before the user clicks anything.
    window.addEventListener("focus", () => {
      if (state.sid) pollOnce();
    });
  }

  /* ── boot ────────────────────────────────────────────────────── */

  async function boot() {
    let config;
    try {
      config = await api("/api/config");
    } catch (error) {
      el.actionNote.textContent = "Could not reach the AmurCloak server.";
      return;
    }
    state.config = config;

    // Wire the interactions *first*. Drawing the controls is cosmetic; if any of
    // it fails, the page must still be able to accept photos.
    wireControls();
    wireDrops();

    try {
      const defaults = config.defaults;
      state.settings.mode = defaults.mode;
      state.settings.preset = defaults.preset;
      state.settings.strength = defaults.strength;
      state.settings.metadata = defaults.metadata || "strip";
      state.settings.spoof_when = defaults.spoof_when || "";
      state.settings.keep_audio = defaults.keep_audio !== false;
      const preset = config.presets.find((p) => p.key === defaults.preset);
      setMarginSlider(preset ? preset.margin : 0.14);
      el.strength.value = String(Math.round(state.settings.strength * 100));

      renderModes();
      renderPresets();
      renderMetadataOptions();
      renderAudioOptions();
      syncControls();
      renderActions();

      const friendly = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".heic": "HEIC",
        ".heif": "HEIF",
        ".webp": "WebP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
      };
      const names = [
        ...new Set(config.supported.map((e) => friendly[e] || e.replace(".", "").toUpperCase())),
      ];
      const clipBits = [];
      if (config.clips && config.clips.enabled) {
        clipBits.push(
          `${config.clips.formats.map((e) => e.replace(".", "").toUpperCase()).join(", ")} clips up to ${config.clips.max_seconds}s`
        );
      }
      el.formatsLine.textContent =
        `Reads ${names.join(", ")}. Redacted copies keep the original format.` +
        (clipBits.length ? ` Also ${clipBits.join(", ")}; clips come back as MP4.` : "");
    } catch (error) {
      // Photos still work; only the cosmetic chrome is degraded.
      console.error("AmurCloak: could not draw the controls", error);
    }
  }

  wireTheme();
  boot();
})();
