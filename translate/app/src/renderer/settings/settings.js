// Settings load/save (via preload IPC) + the Options and Hyperparameters
// dialogs. Persistence lives in main (userData/settings.json).
window.SLLM = window.SLLM || {};

SLLM.settings = (() => {
  // Canonical AIOptions defaults. Mirrors settings-store.js (main process); kept
  // here too so the dialog and the `safe` fill work even if a stored settings
  // file predates a field (load()'s shallow merge can leave it missing).
  const HYPERPARAM_DEFAULTS = {
    waitK: 0,
    kvCache: "not",
    maxWait: -1,
    mode: "api",
    firstChunkMS: 730,
    steadyChunkMS: 730,
    overlapMS: 90,
    llmLeftContextMaxTokens: 1024,
    waitPenalty: 1,
    usePolicy: false,
    repetitionPenalty: 1,
    previewRepetitionPenalty: 1.08,
    previewEveryNWaits: 8,
    previewMaxTokens: -1,
    previewProbDeltaThreshold: 0.81,
    previewUsePreviousPrefix: true,
    previewKvCache: true,
  };

  // Field specs drive the dialog layout. types: int | float | select | toggle.
  // A toggle maps the checkbox to its on/off values (boolean, or 'use'/'not').
  const HYPERPARAMS = [
    { key: "waitK", type: "int" },
    { key: "kvCache", type: "toggle", on: "use", off: "not" },
    { key: "maxWait", type: "int" },
    { key: "mode", type: "select", options: ["api", "local"] },
    { key: "firstChunkMS", type: "int" },
    { key: "steadyChunkMS", type: "int" },
    { key: "overlapMS", type: "int" },
    { key: "llmLeftContextMaxTokens", type: "int" },
    { key: "waitPenalty", type: "float" },
    { key: "usePolicy", type: "toggle", on: true, off: false },
    { key: "repetitionPenalty", type: "float" },
    { key: "previewRepetitionPenalty", type: "float" },
    { key: "previewEveryNWaits", type: "int" },
    { key: "previewMaxTokens", type: "int" },
    { key: "previewProbDeltaThreshold", type: "float" },
    { key: "previewUsePreviousPrefix", type: "toggle", on: true, off: false },
    { key: "previewKvCache", type: "toggle", on: true, off: false },
  ];

  async function load() {
    return window.api.loadSettings();
  }

  async function save(settings) {
    return window.api.saveSettings(settings);
  }

  function buildField(labelText, inputEl) {
    const wrap = document.createElement("label");
    wrap.className = "field";
    const span = document.createElement("span");
    span.textContent = labelText;
    wrap.appendChild(span);
    wrap.appendChild(inputEl);
    return wrap;
  }

  // A toggle/select field lays the label and control out on one row.
  function buildRowField(labelText, inputEl) {
    const wrap = document.createElement("label");
    wrap.className = "field field-row";
    const span = document.createElement("span");
    span.textContent = labelText;
    wrap.append(span, inputEl);
    return wrap;
  }

  function numberInput(value, min, max) {
    const i = document.createElement("input");
    i.type = "number";
    i.value = String(value);
    if (min != null) i.min = String(min);
    if (max != null) i.max = String(max);
    return i;
  }

  function openDialog(titleText, contentEl, onSave) {
    const backdrop = document.createElement("div");
    backdrop.className = "dialog-backdrop";
    const dialog = document.createElement("div");
    dialog.className = "dialog";

    const title = document.createElement("h2");
    title.textContent = titleText;
    dialog.appendChild(title);
    dialog.appendChild(contentEl);

    const actions = document.createElement("div");
    actions.className = "dialog-actions";
    const cancel = document.createElement("button");
    cancel.textContent = "취소";
    const ok = document.createElement("button");
    ok.className = "primary";
    ok.textContent = "저장";
    actions.appendChild(cancel);
    actions.appendChild(ok);
    dialog.appendChild(actions);

    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);

    const close = () => backdrop.remove();
    cancel.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) close();
    });
    ok.addEventListener("click", async () => {
      const ok2 = await onSave();
      if (ok2 !== false) close();
    });
  }

  // Options: server URL, audio source, sample rate, transparency, width, lines.
  function openOptions(current, onApply) {
    const content = document.createElement("div");
    content.className = "dialog-body";

    const serverUrl = document.createElement("input");
    serverUrl.type = "text";
    serverUrl.value = current.serverUrl || "";
    serverUrl.placeholder = "https://192.168.0.42:9001";

    const audioSource = document.createElement("select");
    for (const [v, t] of [["mic", "마이크"], ["system", "시스템 음성"]]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      if (current.audioSource === v) o.selected = true;
      audioSource.appendChild(o);
    }

    const transparency = numberInput(current.bandTransparencyPct, 0, 100);
    const width = numberInput(current.bandWidthPx, 100, 4000);
    const recent = numberInput(current.recentLines, 1, 20);

    content.appendChild(buildField("서버 URL", serverUrl));
    content.appendChild(buildField("오디오 소스", audioSource));
    content.appendChild(buildField("투명도 (0=가림, 100=글씨만)", transparency));
    content.appendChild(buildField("표시 너비 (px)", width));
    content.appendChild(buildField("최근 줄 수", recent));

    openDialog("옵션", content, async () => {
      const next = {
        ...current,
        serverUrl: serverUrl.value.trim(),
        audioSource: audioSource.value,
        bandTransparencyPct: clamp(parseInt(transparency.value, 10), 0, 100),
        bandWidthPx: Math.max(100, parseInt(width.value, 10) || current.bandWidthPx),
        recentLines: Math.max(1, parseInt(recent.value, 10) || current.recentLines),
      };
      const saved = await save(next);
      onApply(saved);
    });
  }

  // Hyperparameters (AIOptions): one field per spec, persisted under
  // settings.hyperparameters. onApply receives the saved settings so the caller
  // can push the new options to the server (see translate-view.js).
  function openHyperparams(current, onApply) {
    const content = document.createElement("div");
    content.className = "dialog-body";
    const hp = { ...HYPERPARAM_DEFAULTS, ...(current.hyperparameters || {}) };
    const inputs = {};

    for (const spec of HYPERPARAMS) {
      const val = hp[spec.key];
      let inputEl;
      if (spec.type === "toggle") {
        inputEl = document.createElement("input");
        inputEl.type = "checkbox";
        inputEl.checked = val === spec.on;
        content.appendChild(buildRowField(spec.key, inputEl));
      } else if (spec.type === "select") {
        inputEl = document.createElement("select");
        for (const opt of spec.options) {
          const o = document.createElement("option");
          o.value = opt;
          o.textContent = opt;
          if (val === opt) o.selected = true;
          inputEl.appendChild(o);
        }
        content.appendChild(buildRowField(spec.key, inputEl));
      } else {
        inputEl = document.createElement("input");
        inputEl.type = "number";
        inputEl.step = spec.type === "float" ? "any" : "1";
        inputEl.value = String(val);
        content.appendChild(buildField(spec.key, inputEl));
      }
      inputs[spec.key] = inputEl;
    }

    openDialog("파라미터", content, async () => {
      const next = {};
      for (const spec of HYPERPARAMS) {
        const el = inputs[spec.key];
        let v;
        if (spec.type === "toggle") v = el.checked ? spec.on : spec.off;
        else if (spec.type === "select") v = el.value;
        else if (spec.type === "float") v = parseFloat(el.value);
        else v = parseInt(el.value, 10);
        // `safe`: fall back to the default on empty/NaN input.
        if (v == null || (typeof v === "number" && Number.isNaN(v))) {
          v = HYPERPARAM_DEFAULTS[spec.key];
        }
        next[spec.key] = v;
      }
      const saved = await save({ ...current, hyperparameters: next });
      onApply(saved);
    });
  }

  function clamp(v, lo, hi) {
    if (Number.isNaN(v)) return lo;
    return Math.min(hi, Math.max(lo, v));
  }

  return { load, save, openOptions, openHyperparams };
})();
