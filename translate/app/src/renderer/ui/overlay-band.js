// Bottom translation panel over the embedded browser: a bordered, resizable box
// (#e9ecef bg) that scrolls through subtitle history. The most recent lines are
// fully opaque; older lines fade out upward (a CSS mask) and stay reachable by
// scrolling. The scrollbar is hidden.
//
// Rendering is a typewriter: the BE streams `confirmed`/`prediction` as the
// current FULL strings, both growing by prefix-extension (e.g. prediction
// "나는" → "나는 밥을" → "나는 밥을 먹었다"). We reveal the newly-arrived tail one
// character at a time — gray for the tentative prediction, black for confirmed.
window.SLLM = window.SLLM || {};

SLLM.overlayBand = (() => {
  // Cap retained confirmed text (chars); the panel scrolls/fades visually, this
  // just bounds memory over a long session.
  const MAX_CHARS = 8000;
  const TICK_MS = 18; // ~55 chars/sec at one char per tick
  const LINE_PX = 32; // 20px font × 1.6 line-height — used to size the solid zone
  const MIN_H = 48;

  function el(tag, className) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    return e;
  }

  // opts.onHeightChange(px) fires when the user finishes a resize drag.
  function create(hostEl, opts) {
    opts = opts || {};
    const panel = el("div", "overlay-panel");
    const resizer = el("div", "overlay-resizer");
    resizer.title = "드래그하여 자막 높이 조절";
    const scroll = el("div", "overlay-scroll");
    const textEl = el("div", "overlay-text");
    const confirmedEl = el("span", "overlay-confirmed");
    const predictionEl = el("span", "overlay-prediction");
    const caretEl = el("span", "overlay-caret");
    caretEl.textContent = "▍";
    caretEl.style.display = "none"; // shown only while typing (see render)
    textEl.append(confirmedEl, predictionEl, caretEl);
    scroll.appendChild(textEl);
    panel.append(resizer, scroll);
    hostEl.appendChild(panel);

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
      panel.style.setProperty("--band-alpha", String(alpha));
    }

    function setWidth(px) {
      panel.style.setProperty("--band-width", `${px}px`);
    }

    // How many recent lines stay fully opaque before the older-lines fade.
    function setRecentLines(n) {
      const solid = Math.max(1, n || 1) * LINE_PX;
      scroll.style.setProperty("--solid-px", `${solid}px`);
    }

    function setHeight(px) {
      const h = Math.max(MIN_H, px || MIN_H);
      panel.style.height = `${h}px`;
    }

    function render() {
      // Follow the live tail only when already at the bottom, so a user who
      // scrolled up to read history isn't yanked back down by new text.
      const atBottom =
        scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 4;
      let c = shownC;
      if (c.length > MAX_CHARS) c = c.slice(c.length - MAX_CHARS);
      confirmedEl.textContent = c;
      predictionEl.textContent = shownP ? (c ? " " : "") + shownP : "";
      // Caret only while typing; gray over a prediction tail, else black.
      caretEl.style.display = timer ? "" : "none";
      caretEl.style.color = shownP ? "#888" : "#000";
      if (atBottom) scroll.scrollTop = scroll.scrollHeight;
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
      // A target no longer an extension of what's shown means a reset/clear/
      // replacement (e.g. prediction folded into confirmed) — snap to it.
      if (targetC !== shownC && !targetC.startsWith(shownC)) shownC = targetC;
      if (targetP !== shownP && !targetP.startsWith(shownP)) shownP = targetP;

      const pending =
        targetC.length - shownC.length + (targetP.length - shownP.length);
      if (pending > 0) {
        // One char per tick normally; speed up if we've fallen far behind.
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
    // the BE omitted that field this message → keep what we have.
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

    // ---- resize (drag the top handle; grows upward) ----------------------
    resizer.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const startY = e.clientY;
      const startH = panel.offsetHeight;
      const stage = hostEl.parentElement;
      const maxH = (stage ? stage.clientHeight : window.innerHeight) - 16;
      const onMove = (ev) => {
        const h = Math.max(MIN_H, Math.min(maxH, startH + (startY - ev.clientY)));
        panel.style.height = `${h}px`;
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        if (opts.onHeightChange) opts.onHeightChange(panel.offsetHeight);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });

    return { setAlphaPct, setWidth, setRecentLines, setHeight, update, clear };
  }

  return { create };
})();
