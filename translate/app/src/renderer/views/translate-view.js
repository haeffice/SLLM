// The "실시간 번역" view: a search bar driving an embedded browser, capture
// controls, and a bottom translation overlay band.
window.SLLM = window.SLLM || {};

SLLM.mountTranslateView = function mountTranslateView(container, settings) {
  const state = {
    settings,
    serverBase: settings.serverUrl || "",
    langDir: settings.languageDirection || "en2ko",
    capturing: false,
    capture: null,
    band: null,
    inFlight: false,
    pending: null,
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
  const langBtn = button(langLabel(state.langDir), "ctrl-btn");
  const optionsBtn = button("옵션", "ctrl-btn");
  const hyperBtn = button("하이퍼파라미터", "ctrl-btn");
  const statusEl = SLLM.statusDot.create();
  controls.append(micBtn, langBtn, optionsBtn, hyperBtn, statusEl);

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
  state.band = SLLM.overlayBand.create(bandEl);
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

  // ---- capture + translate ----------------------------------------------
  function currentLangs() {
    return state.langDir === "ko2en" ? { src: "ko", tgt: "en" } : { src: "en", tgt: "ko" };
  }

  function enqueue(blob) {
    if (state.inFlight) {
      state.pending = blob; // single-slot: keep only the most recent chunk
      return;
    }
    send(blob);
  }

  async function send(blob) {
    state.inFlight = true;
    const { src, tgt } = currentLangs();
    try {
      const { status, json } = await SLLM.net.postTranslate(state.serverBase, blob, src, tgt);
      SLLM.statusDot.setFromStatus(statusEl, status);
      if (json && json.text) state.band.addText(json.text);
    } catch (_) {
      SLLM.statusDot.setFromStatus(statusEl, null);
    } finally {
      state.inFlight = false;
      if (state.pending) {
        const b = state.pending;
        state.pending = null;
        send(b);
      }
    }
  }

  async function startCapture() {
    if (!state.serverBase) {
      alert("서버 URL이 설정되지 않았습니다. 옵션에서 서버 URL을 입력하세요.");
      return;
    }
    state.capture = SLLM.AudioCapture.create(enqueue);
    try {
      await state.capture.start(state.settings.audioSource, state.settings.sampleRate);
      state.capturing = true;
      micBtn.textContent = "마이크 끄기";
      micBtn.classList.add("active");
    } catch (e) {
      state.capture = null;
      alert(`오디오 시작 실패: ${e.message}`);
    }
  }

  async function stopCapture() {
    if (state.capture) await state.capture.stop();
    state.capture = null;
    state.capturing = false;
    state.pending = null;
    micBtn.textContent = "마이크 켜기";
    micBtn.classList.remove("active");
  }

  micBtn.addEventListener("click", () => (state.capturing ? stopCapture() : startCapture()));

  langBtn.addEventListener("click", () => {
    state.langDir = state.langDir === "en2ko" ? "ko2en" : "en2ko";
    langBtn.textContent = langLabel(state.langDir);
    SLLM.settings.save({ ...state.settings, languageDirection: state.langDir });
    state.settings.languageDirection = state.langDir;
  });

  optionsBtn.addEventListener("click", () => {
    SLLM.settings.openOptions(state.settings, async (saved) => {
      const sourceChanged = saved.audioSource !== state.settings.audioSource;
      const rateChanged = saved.sampleRate !== state.settings.sampleRate;
      const serverChanged = saved.serverUrl !== state.settings.serverUrl;
      state.settings = saved;
      state.serverBase = saved.serverUrl || "";
      applyBandSettings(state.band, saved);
      if (serverChanged) refreshHealth();
      if (state.capturing && (sourceChanged || rateChanged)) {
        await stopCapture();
        await startCapture();
      }
    });
  });

  hyperBtn.addEventListener("click", () => SLLM.settings.openHyperparams());

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
    await stopCapture();
    safe(() => webview.stop());
    view.remove();
  };

  // ---- helpers -----------------------------------------------------------
  function langLabel(dir) {
    return dir === "ko2en" ? "한국어 → 영어" : "영어 → 한국어";
  }
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
function safe(fn) {
  try {
    fn();
  } catch (_) {}
}
function applyBandSettings(band, s) {
  band.setAlphaPct(s.bandTransparencyPct);
  band.setWidth(s.bandWidthPx);
  band.setRecentLines(s.recentLines);
}
