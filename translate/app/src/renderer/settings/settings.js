// Settings load/save (via preload IPC) + the Options and Hyperparameters
// dialogs. Persistence lives in main (userData/settings.json).
window.SLLM = window.SLLM || {};

SLLM.settings = (() => {
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

    const sampleRate = numberInput(current.sampleRate, 8000, 48000);
    const transparency = numberInput(current.bandTransparencyPct, 0, 100);
    const width = numberInput(current.bandWidthPx, 100, 4000);
    const recent = numberInput(current.recentLines, 1, 20);

    content.appendChild(buildField("서버 URL", serverUrl));
    content.appendChild(buildField("오디오 소스", audioSource));
    content.appendChild(buildField("샘플레이트 (Hz)", sampleRate));
    content.appendChild(buildField("투명도 (0=가림, 100=글씨만)", transparency));
    content.appendChild(buildField("표시 너비 (px)", width));
    content.appendChild(buildField("최근 줄 수", recent));

    openDialog("옵션", content, async () => {
      const next = {
        ...current,
        serverUrl: serverUrl.value.trim(),
        audioSource: audioSource.value,
        sampleRate: parseInt(sampleRate.value, 10) || current.sampleRate,
        bandTransparencyPct: clamp(parseInt(transparency.value, 10), 0, 100),
        bandWidthPx: Math.max(100, parseInt(width.value, 10) || current.bandWidthPx),
        recentLines: Math.max(1, parseInt(recent.value, 10) || current.recentLines),
      };
      const saved = await save(next);
      onApply(saved);
    });
  }

  // Hyperparameters: placeholder (wired for later use, per spec).
  function openHyperparams() {
    const content = document.createElement("div");
    content.className = "dialog-body";
    const note = document.createElement("p");
    note.textContent = "하이퍼파라미터 설정은 추후 제공됩니다.";
    content.appendChild(note);
    openDialog("하이퍼파라미터", content, async () => true);
  }

  function clamp(v, lo, hi) {
    if (Number.isNaN(v)) return lo;
    return Math.min(hi, Math.max(lo, v));
  }

  return { load, save, openOptions, openHyperparams };
})();
