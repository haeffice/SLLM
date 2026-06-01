// The "실시간 번역" view: a search bar driving an embedded browser, capture
// controls, and a bottom translation overlay band.
window.SLLM = window.SLLM || {};

SLLM.mountTranslateView = function mountTranslateView(container, settings) {
  const state = {
    settings,
    serverBase: settings.serverUrl || "",
    langDir: settings.languageDirection || "en2ko",
    task: settings.taskMode || "translate",
    capturing: false,
    micToggling: false,
    capture: null,
    band: null,
    ws: null,
  };

  // ---- DOM ---------------------------------------------------------------
  const view = el("section", "translate-view");

  const topbar = el("div", "topbar");
  const navBtns = el("div", "nav-btns");
  const backBtn = button("‹", "icon-btn");
  const fwdBtn = button("›", "icon-btn");
  const reloadBtn = button("⟳", "icon-btn");
  navBtns.append(backBtn, fwdBtn, reloadBtn);

  const searchForm = el("form", "search-form");
  const searchInput = el("input", "search-input");
  searchInput.type = "text";
  searchInput.placeholder = "Google 검색 또는 URL 입력";
  searchForm.appendChild(searchInput);

  const controls = el("div", "controls");
  const micBtn = button("마이크 켜기", "ctrl-btn");
  // Direction toggle: knob left = EN→KO (en2ko), knob right = KO→EN (ko2en).
  const langToggle = toggleSwitch("EN→KO", "KO→EN", state.langDir === "ko2en", (isRight) => {
    state.langDir = isRight ? "ko2en" : "en2ko";
    state.settings.languageDirection = state.langDir;
    SLLM.settings.save({ ...state.settings });
    // Live session: flip direction in place; rides the query string otherwise.
    if (state.ws && state.ws.isOpen()) state.ws.setDirection(state.langDir);
  });
  // Task toggle: knob left = 번역 (translate), knob right = 전사 (transcribe).
  const taskToggle = toggleSwitch("번역", "전사", state.task === "transcribe", (isRight) => {
    state.task = isRight ? "transcribe" : "translate";
    state.settings.taskMode = state.task;
    SLLM.settings.save({ ...state.settings });
    if (state.ws && state.ws.isOpen()) state.ws.setTask(state.task);
  });
  const optionsBtn = button("⚙", "ctrl-btn gear");
  optionsBtn.title = "옵션";
  const paramBtn = button("파라미터", "ctrl-btn");
  const statusEl = SLLM.statusDot.create();
  controls.append(micBtn, langToggle, taskToggle, optionsBtn, paramBtn, statusEl);

  const tagsRow = el("div", "tags-row");

  topbar.append(navBtns, searchForm, controls, tagsRow);

  const stage = el("div", "browser-stage");
  const webview = document.createElement("webview");
  webview.className = "embedded-browser";
  webview.setAttribute("partition", "persist:browser");
  webview.setAttribute("src", "https://www.google.com");
  webview.setAttribute("allowpopups", "");

  const bandEl = el("div", "overlay-band");

  stage.append(webview, bandEl);
  view.append(topbar, stage);
  container.appendChild(view);

  // ---- overlay band ------------------------------------------------------
  state.band = SLLM.overlayBand.create(bandEl, {
    onHeightChange: (px) => {
      state.settings.bandHeightPx = px;
      SLLM.settings.save({ ...state.settings });
    },
  });
  applyBandSettings(state.band, state.settings);

  // ---- browser navigation -----------------------------------------------
  searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = searchInput.value.trim();
    if (!q) return;
    const url = /^https?:\/\//i.test(q)
      ? q
      : `https://www.google.com/search?q=${encodeURIComponent(q)}`;
    safe(() => webview.loadURL(url));
  });
  backBtn.addEventListener("click", () => safe(() => webview.canGoBack() && webview.goBack()));
  fwdBtn.addEventListener("click", () => safe(() => webview.canGoForward() && webview.goForward()));
  reloadBtn.addEventListener("click", () => safe(() => webview.reload()));
  webview.addEventListener("did-navigate", (e) => {
    searchInput.value = e.url || searchInput.value;
  });

  // ---- capture + translate (WebSocket stream) ---------------------------
  // The socket is opened lazily on the first mic-on and then kept alive across
  // mic toggles and direction/task switches: those mutate the live session via
  // control messages (see net/client.js) instead of reconnecting. The socket is
  // only torn down on a server-URL change or when the view is destroyed.
  function langs() {
    return state.langDir === "ko2en" ? { src: "ko", tgt: "en" } : { src: "en", tgt: "ko" };
  }

  function makeCapture() {
    return SLLM.AudioCapture.create((bytes) => {
      if (state.ws) state.ws.sendAudio(bytes);
    });
  }

  // Open a socket if one isn't already up. The current direction/task ride the
  // query string so the first frames are decoded the right way. The handlers
  // are guarded with `state.ws === ws` so a previous socket's late-firing close
  // (close() resolves asynchronously) can't clobber a freshly-opened one.
  function connectWs() {
    if (state.ws || !state.serverBase) return;
    const { src, tgt } = langs();
    let ws;
    ws = SLLM.net.connectWs(state.serverBase, { src, tgt, task: state.task }, {
      onOpen: () => {
        if (state.ws !== ws) return;
        SLLM.statusDot.setFromStatus(statusEl, 200);
        // Seed the session with the configured hyperparameters.
        if (state.settings.hyperparameters) ws.setOptions(state.settings.hyperparameters);
      },
      onMessage: (obj) => {
        if (!obj) return;
        if (!("confirmed" in obj || "prediction" in obj || "eos" in obj)) return;
        // Absent keys mean "unchanged" → pass undefined so the band keeps them
        // (avoids wiping confirmed black text while only prediction streams).
        const confirmed = "confirmed" in obj ? obj.confirmed || "" : undefined;
        let prediction;
        if (obj.eos) {
          // End-of-sentence: the tentative tail folded into `confirmed`, so
          // drop any lingering prediction (gray) right away.
          prediction = "";
        } else if ("prediction" in obj) {
          prediction = obj.prediction || "";
        }
        state.band.update(confirmed, prediction);
      },
      onError: () => {
        if (state.ws === ws) SLLM.statusDot.setFromStatus(statusEl, 500);
      },
      onClose: () => {
        if (state.ws !== ws) return; // stale socket we already replaced/closed
        state.ws = null;
        SLLM.statusDot.setFromStatus(statusEl, null);
        // Remote drop: stop the local mic so the button reflects reality.
        if (state.capturing) stopMic();
      },
    });
    state.ws = ws;
  }

  // Drop the current socket as an intentional close: clear state.ws first so the
  // (async) onClose sees a stale ref and skips the remote-drop cleanup.
  function closeWs() {
    if (state.ws) {
      const ws = state.ws;
      state.ws = null;
      ws.close();
    }
  }

  async function startMic() {
    if (!state.serverBase) {
      alert("서버 URL이 설정되지 않았습니다. 옵션에서 서버 URL을 입력하세요.");
      return;
    }
    connectWs();
    state.band.clear();
    state.capture = makeCapture();
    try {
      await state.capture.start(state.settings.audioSource);
      state.capturing = true;
      micBtn.textContent = "마이크 끄기";
      micBtn.classList.add("active");
    } catch (e) {
      alert(`오디오 시작 실패: ${e.message}`);
      await state.capture.stop();
      state.capture = null;
    }
  }

  async function stopMic() {
    if (state.capture) {
      await state.capture.stop();
      state.capture = null;
    }
    state.capturing = false;
    micBtn.textContent = "마이크 켜기";
    micBtn.classList.remove("active");
    // Keep the socket open; just tell the server to reset its decode buffer.
    if (state.ws) state.ws.micOff();
  }

  // Swap the audio source mid-capture without disturbing the socket.
  async function restartCapture() {
    if (state.capture) {
      await state.capture.stop();
      state.capture = null;
    }
    state.capture = makeCapture();
    try {
      await state.capture.start(state.settings.audioSource);
    } catch (e) {
      alert(`오디오 시작 실패: ${e.message}`);
      await stopMic();
    }
  }

  // Serialize toggles: start/stop are async, so a fast double-click could
  // otherwise spin up a second capture and leak the first (mic stuck on).
  micBtn.addEventListener("click", async () => {
    if (state.micToggling) return;
    state.micToggling = true;
    try {
      if (state.capturing) await stopMic();
      else await startMic();
    } finally {
      state.micToggling = false;
    }
  });

  optionsBtn.addEventListener("click", () => {
    SLLM.settings.openOptions(state.settings, async (saved) => {
      const sourceChanged = saved.audioSource !== state.settings.audioSource;
      const serverChanged = saved.serverUrl !== state.settings.serverUrl;
      state.settings = saved;
      state.serverBase = saved.serverUrl || "";
      applyBandSettings(state.band, saved);
      if (serverChanged) {
        // New endpoint — reconnect only if we already held a socket.
        const wasConnected = !!state.ws;
        closeWs();
        refreshHealth();
        if (wasConnected) connectWs();
      }
      if (state.capturing && sourceChanged) {
        await restartCapture();
      }
    });
  });

  paramBtn.addEventListener("click", () => {
    SLLM.settings.openHyperparams(state.settings, (saved) => {
      state.settings = saved;
      // Apply live without reconnect; rides the next connect otherwise.
      if (state.ws && state.ws.isOpen()) state.ws.setOptions(saved.hyperparameters);
    });
  });

  // ---- health / tags -----------------------------------------------------
  async function refreshHealth() {
    if (!state.serverBase) {
      SLLM.tags.render(tagsRow, []);
      SLLM.statusDot.setFromStatus(statusEl, null);
      return;
    }
    try {
      const { status, json } = await SLLM.net.fetchHealth(state.serverBase);
      SLLM.tags.render(tagsRow, SLLM.net.tagsFromHealth(json));
      SLLM.statusDot.setFromStatus(statusEl, status);
    } catch (_) {
      SLLM.tags.render(tagsRow, []);
      SLLM.statusDot.setFromStatus(statusEl, null);
    }
  }
  refreshHealth();

  // ---- teardown ----------------------------------------------------------
  return async function destroy() {
    await stopMic();
    closeWs();
    safe(() => webview.stop());
    view.remove();
  };

};

// shared tiny DOM helpers (module scope)
function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}
function button(text, className) {
  const b = el("button", className);
  b.textContent = text;
  b.type = "button";
  return b;
}
// Sliding two-state toggle. Shows `leftLabel` (knob left) or `rightLabel` (knob
// right); clicking flips and calls onToggle(isRight). startRight picks the
// initial side.
function toggleSwitch(leftLabel, rightLabel, startRight, onToggle) {
  const sw = el("button", "toggle-switch");
  sw.type = "button";
  const label = el("span", "toggle-label");
  const knob = el("span", "toggle-knob");
  sw.append(label, knob);
  let right = !!startRight;
  function apply() {
    sw.classList.toggle("is-right", right);
    label.textContent = right ? rightLabel : leftLabel;
  }
  apply();
  sw.addEventListener("click", () => {
    right = !right;
    apply();
    onToggle(right);
  });
  return sw;
}
function safe(fn) {
  try {
    fn();
  } catch (_) {}
}
function applyBandSettings(band, s) {
  band.setAlphaPct(s.bandTransparencyPct);
  band.setWidth(s.bandWidthPx);
  band.setRecentLines(s.recentLines);
  band.setHeight(s.bandHeightPx);
}
