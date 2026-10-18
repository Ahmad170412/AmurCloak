/* TigerBlur front end.
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
    segs: document.querySelectorAll(".seg"),
  };

  const state = {
    config: null,
    sid: null,
    snapshot: null,
    index: 0,
    view: "audit",
    settings: { mode: "black", preset: "balanced", margin: 0.14, strength: 0.7 },
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
        syncControls();
        // Adopt the preset's own idea of a sensible coverage.
        setMarginSlider(preset.margin);
        pushSettings();
      });
      el.presetOptions.appendChild(button);
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
  }

  function pushSettings() {
    if (!state.sid) return;
    clearTimeout(state.settingsTimer);
    state.settingsTimer = setTimeout(async () => {
      try {
        await api(`/api/sessions/${state.sid}/settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode: state.settings.mode,
            preset: state.settings.preset,
            margin: state.settings.margin,
            strength: state.settings.strength,
          }),
        });
        refreshPreview();
      } catch (error) {
        setNote(error.message);
      }
    }, 130);
  }

  /* ── preview ─────────────────────────────────────────────────── */

  function setNote(text) {
    el.viewerNote.textContent = text || "";
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
      el.previewImg.removeAttribute("src");
      setNote(error.message);
    } finally {
      if (token === state.previewToken) el.viewerLoading.hidden = true;
    }
  }

  async function selectIndex(index) {
    if (!state.snapshot || !state.snapshot.items.length) return;
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
        ? `${item.faces} ${plural(item.faces, "face", "faces")}`
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
      `All of it is stripped from the redacted copies, along with any timestamps.`;
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
      el.downloadBtn.hidden = false;
      el.downloadBtn.textContent = `Download ${count} ${plural(count, "photo", "photos")}`;
      el.renderBtn.textContent = "Redact again";
      el.renderBtn.disabled = state.busy;
      el.actionNote.textContent = snapshot.downloaded
        ? "Staged copies were freed to reclaim disk. The redacted files are still here."
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
      el.actionNote.textContent = error.message;
      state.busy = false;
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
        setNote(error.message);
      }
    }, 320);
  }

  function download() {
    if (!state.sid || !state.snapshot || !state.snapshot.result.ready) return;
    const name = state.snapshot.result.name || "tigerblur-output";
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
    } catch {
      /* session may have been reaped */
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

    hideStaging();
    el.previewImg.removeAttribute("src");
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
  }

  /* ── boot ────────────────────────────────────────────────────── */

  async function boot() {
    let config;
    try {
      config = await api("/api/config");
    } catch (error) {
      el.actionNote.textContent = "Could not reach the TigerBlur server.";
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
      const preset = config.presets.find((p) => p.key === defaults.preset);
      setMarginSlider(preset ? preset.margin : 0.14);
      el.strength.value = String(Math.round(state.settings.strength * 100));

      renderModes();
      renderPresets();
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
      console.error("TigerBlur: could not draw the controls", error);
    }
  }

  boot();
})();
