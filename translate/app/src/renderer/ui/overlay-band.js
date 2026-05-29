// Bottom translation band over the embedded browser: confirmed text in black
// followed by the tentative prediction in gray, with an adjustable-opacity
// background and a most-recent-N-(visual)-lines window.
//
// Rendering is a typewriter: the BE streams `confirmed`/`prediction` as the
// current FULL strings, and both grow by prefix-extension (e.g. prediction
// "나는" → "나는 밥을" → "나는 밥을 먹었다"). We reveal the newly-arrived tail
// one character at a time — gray for the tentative prediction, black for
// confirmed. When the prediction is cleared and the sentence lands in
// `confirmed`, the gray text disappears and the confirmed text types out black.
window.SLLM = window.SLLM || {};

SLLM.overlayBand = (() => {
  // Cap retained confirmed text (chars); visual clipping shows only the most
  // recent N lines, this just bounds memory over a long session.
  const MAX_CHARS = 4000;
  const TICK_MS = 18; // ~55 chars/sec at one char per tick

  function create(hostEl) {
    const lines = document.createElement("div");
    lines.className = "overlay-lines";
    const textEl = document.createElement("div");
    textEl.className = "overlay-text";
    const confirmedEl = document.createElement("span");
    confirmedEl.className = "overlay-confirmed";
    const predictionEl = document.createElement("span");
    predictionEl.className = "overlay-prediction";
    const caretEl = document.createElement("span");
    caretEl.className = "overlay-caret";
    caretEl.textContent = "▍";
    caretEl.style.display = "none"; // shown only while typing (see render)
    textEl.append(confirmedEl, predictionEl, caretEl);
    lines.appendChild(textEl);
    hostEl.appendChild(lines);

    // target* = what the BE wants displayed; shown* = what's typed so far.
    let targetC = "";
    let targetP = "";
    let shownC = "";
    let shownP = "";
    let timer = null;

    function setAlphaPct(pct) {
      // Spec: 0% = background fully covers content, 100% = text only (no bg).
      // So CSS alpha is the INVERSE of the option value. Do not "simplify".
      const alpha = 1 - Math.min(100, Math.max(0, pct)) / 100;
      hostEl.style.setProperty("--band-alpha", String(alpha));
    }

    function setWidth(px) {
      lines.style.setProperty("--band-width", `${px}px`);
    }

    function setRecentLines(n) {
      lines.style.setProperty("--recent-lines", String(Math.max(1, n)));
    }

    function render() {
      // Window the DISPLAY to the most recent MAX_CHARS; the typewriter state
      // (shownC/targetC) stays full so it remains a clean prefix relationship.
      let c = shownC;
      if (c.length > MAX_CHARS) c = c.slice(c.length - MAX_CHARS);
      confirmedEl.textContent = c;
      predictionEl.textContent = shownP ? (c ? " " : "") + shownP : "";
      // Caret only while typing; gray over a prediction tail, else black.
      caretEl.style.display = timer ? "" : "none";
      caretEl.style.color = shownP ? "#888" : "#000";
    }

    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    function ensure() {
      if (!timer) timer = setInterval(tick, TICK_MS);
    }

    function tick() {
      // A target that is no longer an extension of what's shown means a
      // reset/clear/replacement (e.g. prediction folded into confirmed) — snap
      // to it instantly rather than retyping or showing stale text.
      if (targetC !== shownC && !targetC.startsWith(shownC)) shownC = targetC;
      if (targetP !== shownP && !targetP.startsWith(shownP)) shownP = targetP;

      const pending =
        targetC.length - shownC.length + (targetP.length - shownP.length);
      if (pending > 0) {
        // One char per tick normally; speed up if we've fallen far behind the
        // stream so the displayed text never lags by more than ~1s.
        let step = Math.max(1, Math.ceil(pending / 40));
        while (step-- > 0) {
          if (shownC.length < targetC.length) {
            shownC = targetC.slice(0, shownC.length + 1);
          } else if (shownP.length < targetP.length) {
            shownP = targetP.slice(0, shownP.length + 1);
          } else {
            break;
          }
        }
      }
      if (shownC === targetC && shownP === targetP) stop();
      render();
    }

    // confirmed/prediction are the BE's current full strings. `undefined` means
    // the BE omitted that field this message → keep what we have (the missing
    // key carries no "cleared" meaning). targetC mirrors the latest confirmed,
    // so memory is bounded by the BE payload, not the session length.
    function update(confirmed, prediction) {
      if (confirmed != null) targetC = confirmed;
      if (prediction != null) targetP = prediction;
      ensure();
    }

    function clear() {
      stop();
      targetC = "";
      targetP = "";
      shownC = "";
      shownP = "";
      render();
    }

    return { setAlphaPct, setWidth, setRecentLines, update, clear };
  }

  return { create };
})();
