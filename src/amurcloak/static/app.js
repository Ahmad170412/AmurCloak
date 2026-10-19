/* AmurCloak front end.
   No framework, no build step, no external requests - the page must keep
   working with the network unplugged, which is the whole point of the tool. */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const el = {
    dropzone: $("#dropzone"),
    viewer: $("#viewer"),
    canvas: $("#viewer-canvas"),
    previewImg: $("#preview-img"),
    viewerLoading: $("#viewer-loading"),
    faceLayer: $("#face-layer"),
    viewerPick: $("#viewer-pick"),
    pickHint: $("#pick-hint"),
    coverAll: $("#cover-all"),
    viewerName: $("#viewer-name"),
    viewerIndex: $("#viewer-index"),
    viewerNote: $("#viewer-note"),
    fileList: $("#file-list"),
    filesNote: $("#files-note"),
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
    renderBtn: $("#render-btn"),
    downloadBtn: $("#download-btn"),
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
    settings: {
      mode: "black",
      preset: "balanced",
      margin: 0.14,
      strength: 0.7,
      metadata: "strip",
      spoof_when: "",
    },
    faces: [], // the last preview's detections, for the click targets
    previewToken: 0,
    busy: false,
    uploading: false,
    pending: [], // files dropped but not yet acknowledged by the server
    uploadXhr: null,
    pollTimer: null,
    settingsTimer: null,
  };

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

  async function refreshPreview() {
    if (!state.sid || !state.snapshot || !state.snapshot.items.length) return;
    const token = ++state.previewToken;
    el.viewerLoading.hidden = false;

    try {
      const payload = await api(
        `/api/sessions/${state.sid}/preview?index=${state.index}&view=${state.view}`
      );
      if (token !== state.previewToken) return; // a newer request already won
      el.previewImg.src = `data:image/jpeg;base64,${payload.image}`;
      el.viewerNote.textContent = payload.note;
      renderFaces(payload);

      // The rail learns the face count for whatever was just checked, so it can
      // report real numbers without scanning every photo up front.
      const item = state.snapshot && state.snapshot.items[state.index];
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

  function hideFaces() {
    state.faces = [];
    el.faceLayer.innerHTML = "";
    el.faceLayer.hidden = true;
    el.viewerPick.hidden = true;
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
      el.viewerPick.hidden = true;
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
    el.viewerPick.hidden = false;
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
    renderFiles();
    const item = state.snapshot.items[state.index];
    el.viewerName.textContent = item.name;
    el.viewerIndex.textContent = `${state.index + 1} of ${state.snapshot.items.length}`;
    refreshPreview();
  }

  /* ── files rail ──────────────────────────────────────────────── */

  function renderFiles() {
    const snapshot = state.snapshot;
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

      let chip;
      if (item.status === "error") chip = `<span class="chip chip-failed">failed</span>`;
      else if (item.status === "done") chip = `<span class="chip chip-done">redacted</span>`;
      else if (snapshot.state === "rendering") chip = `<span class="chip chip-working">working</span>`;
      else chip = "";

      const faceText = item.scanned
        ? `${item.faces} ${plural(item.faces, "face", "faces")}` +
          (item.restored ? ` · ${item.restored} left visible` : "")
        : "";

      button.innerHTML = `
        <span class="file-num">${index + 1}</span>
        <span class="file-name">${escapeHtml(item.name)}</span>
        <span class="file-size">${formatBytes(item.size)}</span>
        <span class="file-status">${chip}${faceText ? `<span>${faceText}</span>` : ""}</span>`;
      button.addEventListener("click", () => selectIndex(index));
      li.appendChild(button);
      el.fileList.appendChild(li);
    });

    const failed = snapshot.items.filter((i) => i.status === "error").length;
    const total = snapshot.items.length;
    const checked = snapshot.items.filter((i) => i.scanned).length;
    const totalFaces = snapshot.items.reduce((sum, i) => sum + (i.faces || 0), 0);

    // The one place disk footprint is reported: it describes *these* photos.
    // Free space on the volume is not shown - the OS already tells you that, and
    // nothing here acts on it. The upload guard errors with the number if it
    // ever actually matters.
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
      parts.join(", ") || "Click a photo to check it before redacting.";
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
    el.metadataFlag.textContent =
      `Heads up: ${bits.join(" and ")}. ` +
      (state.settings.metadata === "spoof"
        ? "The redacted copies replace all of it with a deliberately false timestamp, a made-up camera and a location in the emptiest stretch of the Pacific."
        : "All of it is stripped from the redacted copies, along with any timestamps.");
  }

  function renderActions() {
    const snapshot = state.snapshot;
    const hasItems = Boolean(snapshot && snapshot.items.length);
    const renderable = hasItems ? snapshot.items.filter((i) => i.status !== "error").length : 0;

    el.renderBtn.hidden = false;
    el.downloadBtn.hidden = true;
    el.progress.hidden = true;
    el.renderBtn.disabled = !hasItems || renderable === 0 || state.busy;

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
      el.progressFill.style.width = `${Math.round((done / total) * 100)}%`;
      el.progressText.textContent = `Redacting ${Math.min(done + (done < total ? 1 : 0), total)} of ${total}`;
      el.actionNote.textContent = "";
      return;
    }

    if (snapshot.state === "ready" && snapshot.result.ready) {
      const count = snapshot.items.filter((i) => i.status === "done").length;
      const freed = Boolean(snapshot.disk && snapshot.disk.inputs_freed);
      el.downloadBtn.hidden = false;
      el.downloadBtn.textContent = `Download ${count} ${plural(count, "photo", "photos")}`;
      el.renderBtn.textContent = "Redact again";
      // Re-rendering reads the staged copies. Once those have been freed there
      // is nothing left to read, and the button would only produce an error.
      el.renderBtn.disabled = state.busy || freed;
      el.actionNote.textContent = snapshot.downloaded
        ? "Staged copies were freed to reclaim space. Drop the photos again to redact them differently."
        : `Output is ${formatBytes(snapshot.result.size)}. Staged copies are freed once you download.`;
      return;
    }

    if (snapshot.state === "error") {
      el.renderBtn.textContent = "Try again";
      el.renderBtn.disabled = false;
      el.actionNote.textContent = snapshot.error || "Something went wrong.";
      return;
    }

    const label = `${renderable} ${plural(renderable, "photo", "photos")}`;
    el.renderBtn.textContent = `Redact ${label}`;
    el.actionNote.textContent = "Faces are covered on this machine. Nothing is sent anywhere.";
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
      const preset = config.presets.find((p) => p.key === defaults.preset);
      setMarginSlider(preset ? preset.margin : 0.14);
      el.strength.value = String(Math.round(state.settings.strength * 100));

      renderModes();
      renderPresets();
      renderMetadataOptions();
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
      el.formatsLine.textContent = `Reads ${names.join(", ")}. Redacted copies keep the original format.`;
    } catch (error) {
      // Photos still work; only the cosmetic chrome is degraded.
      console.error("AmurCloak: could not draw the controls", error);
    }
  }

  boot();
})();
