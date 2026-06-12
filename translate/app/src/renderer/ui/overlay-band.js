// Translation panel over the embedded browser: a bordered, resizable box
// (#f1f3f5 bg) that scrolls through subtitle history. The most recent lines are
// fully opaque; older lines fade out upward (a CSS mask) and stay reachable by
// scrolling. The scrollbar is hidden.
//
// Two modes (toggled by the top-right action button):
//   - default: a square panel filling the full bottom of the stage edge-to-edge
//     (no margins, no rounded corners). Drag the top header to resize height
//     (grows upward). The pop-out icon switches to drag mode.
//   - drag: a free-floating panel. Drag the header to move it; the eight edge/
//     corner handles resize it. It is clamped fully inside the stage. The ✕ icon
//     switches back to default mode.
//
// Rendering is a typewriter. The BE streams `confirmed` (the finalized line so
// far, cumulative — only grows) and `prediction` (the tentative tail that
// continues after it; it accumulates while a segment streams and arrives EMPTY
// on the frame that promotes that tail into `confirmed`). The displayed line is
// confirmed + tail. The gray tentative tail types out one char at a time. When a
// confirm lands, the prefix it shares with the prior prediction recolors to black
// instantly (no re-typing); only confirmed text that diverges from / extends past
// the prediction types out in black.
window.SLLM = window.SLLM || {};

SLLM.overlayBand = (() => {
  // Cap retained confirmed text (chars); the panel scrolls/fades visually, this
  // just bounds memory over a long session.
  const MAX_CHARS = 8000;
  const TICK_MS = 18; // ~55 chars/sec at one char per tick
  // Min panel height: header (24px) + border + at least ~2 subtitle lines, so the
  // top fade band (~1 line, see the mask in styles.css) never swallows the whole
  // visible panel even when shrunk to the floor.
  const MIN_H = 96;
  const MIN_W = 200;

  // Pop-out (default → drag) and close (drag → default) glyphs. stroke uses
  // currentColor so they inherit the button's text color.
  const ICON_POPOUT =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6"/><path d="M20 4l-8.5 8.5"/><path d="M19 13.5V19a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5.5"/></svg>';
  const ICON_CLOSE =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  const HANDLE_DIRS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];

  function el(tag, className) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    return e;
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  // Length of the leading run where a and b are character-for-character equal.
  function commonPrefixLen(a, b) {
    const n = Math.min(a.length, b.length);
    let i = 0;
    while (i < n && a[i] === b[i]) i++;
    return i;
  }

  // opts.onHeightChange(px) fires when the user finishes a docked resize drag.
  function create(hostEl, opts) {
    opts = opts || {};
    const panel = el("div", "overlay-panel mode-default");

    // Resize handles — visible/active only in drag mode (CSS hides them docked).
    const handlesWrap = el("div", "overlay-handles");
    for (const dir of HANDLE_DIRS) {
      const h = el("div", `overlay-handle h-${dir}`);
      h.dataset.dir = dir;
      handlesWrap.appendChild(h);
    }

    // Top strip: vertical resize handle (default) / move handle (drag), with the
    // mode-toggle action button on the right.
    const header = el("div", "overlay-header");
    const actionBtn = el("button", "overlay-action");
    actionBtn.type = "button";
    actionBtn.innerHTML = ICON_POPOUT;
    actionBtn.title = "새 창처럼 띄우기";
    header.appendChild(actionBtn);

    const scroll = el("div", "overlay-scroll");
    const textEl = el("div", "overlay-text");
    const confirmedEl = el("span", "overlay-confirmed");
    const predictionEl = el("span", "overlay-prediction");
    const caretEl = el("span", "overlay-caret");
    caretEl.textContent = "▍";
    caretEl.style.display = "none"; // shown only while typing (see render)
    textEl.append(confirmedEl, predictionEl, caretEl);
    scroll.appendChild(textEl);
    panel.append(handlesWrap, header, scroll);
    hostEl.appendChild(panel);

    // State from the BE: `confirmedText` (finalized, black — only ever grows) and
    // `tentLine` (the full tentative line = confirmed + the last NON-EMPTY
    // prediction). `tentLine` is FROZEN while prediction comes empty, so its gray
    // tail survives across a (possibly multi-frame) confirmation and only the part
    // confirmed catches up to recolors to black. The displayed line `text` is
    // confirmedText + the part of tentLine past it (gray) — while confirmed still
    // matches the frozen line; once confirmed diverges, the gray is dropped. Chars
    // [0, confirmedLen) render black, the rest gray. `shown` is the typewriter
    // position. Confirmed (black) text is never erased.
    let confirmedText = "";
    let tentLine = "";
    let text = "";
    let confirmedLen = 0;
    let shown = 0;
    let timer = null;

    // ---- layout state ----------------------------------------------------
    let mode = "default";
    let heightPx = MIN_H; // docked height (persisted via onHeightChange)
    // Floating geometry in stage coords; seeded from the docked rect on pop-out.
    let rect = { left: 0, top: 0, width: 320, height: MIN_H };

    function stageEl() {
      return hostEl.parentElement; // .browser-stage (hostEl is inset:0 within it)
    }
    function stageSize() {
      const s = stageEl();
      // Fall back to the window when the stage is missing or momentarily
      // zero-sized (e.g. hidden/minimized) so geometry never collapses to 0.
      if (s && s.clientWidth > 0 && s.clientHeight > 0) {
        return { w: s.clientWidth, h: s.clientHeight };
      }
      return { w: window.innerWidth, h: window.innerHeight };
    }

    function applyRect() {
      panel.style.left = `${rect.left}px`;
      panel.style.top = `${rect.top}px`;
      panel.style.width = `${rect.width}px`;
      panel.style.height = `${rect.height}px`;
    }

    // Pull the floating rect back fully inside the stage (used on pop-out seed
    // and on window resize).
    function clampRectIntoStage() {
      const { w, h } = stageSize();
      rect.width = clamp(rect.width, MIN_W, w);
      rect.height = clamp(rect.height, MIN_H, h);
      rect.left = clamp(rect.left, 0, w - rect.width);
      rect.top = clamp(rect.top, 0, h - rect.height);
    }

    function setAlphaPct(pct) {
      // Spec: 0% = background fully covers content, 100% = text only (no bg).
      // So CSS alpha is the INVERSE of the option value. Do not "simplify".
      const alpha = 1 - Math.min(100, Math.max(0, pct)) / 100;
      panel.style.setProperty("--band-alpha", String(alpha));
    }

    function setHeight(px) {
      heightPx = Math.max(MIN_H, px || MIN_H);
      if (mode === "default") panel.style.height = `${heightPx}px`;
    }

    function render() {
      // Follow the live tail only when already at the bottom, so a user who
      // scrolled up to read history isn't yanked back down by new text.
      const atBottom =
        scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 4;
      // chars [0, confirmedLen) render black (confirmed), [confirmedLen, shown)
      // gray (tentative). `shown` is the typewriter position. On a confirm the
      // prefix shared with the prior prediction is force-revealed at once (see
      // update); only the diverging confirmed tail then types out, in black.
      const blackEnd = Math.min(shown, confirmedLen);
      let black = text.slice(0, blackEnd);
      const gray = shown > confirmedLen ? text.slice(confirmedLen, shown) : "";
      // Bound retained black text (memory) without disturbing the live tail.
      if (black.length > MAX_CHARS) black = black.slice(black.length - MAX_CHARS);
      confirmedEl.textContent = black;
      predictionEl.textContent = gray;
      // Caret only while typing actual text; gray over a prediction tail, else black.
      caretEl.style.display = timer && text.length > 0 ? "" : "none";
      caretEl.style.color = gray ? "#888" : "#000";
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
      if (shown >= text.length) {
        stop();
        render();
        return;
      }
      // One char per tick normally; speed up if we've fallen far behind.
      const step = Math.max(1, Math.ceil((text.length - shown) / 40));
      shown = Math.min(text.length, shown + step);
      if (shown >= text.length) stop();
      render();
    }

    // Display rules:
    //   1) prediction grows (confirmed unchanged): the appended text types out one
    //      char at a time in gray.
    //   2) prediction empty, confirmed grows over one or more frames: hold the
    //      frozen tentative line (the last non-empty prediction). As confirmed
    //      catches up to it, the matching prefix recolors to black AT ONCE (no
    //      re-typing). The first frame confirmed DIVERGES from the frozen line, the
    //      remaining gray is dropped and the diverging confirmed types out in black.
    function update(confirmed, prediction) {
      // Confirmed only ever GROWS — a shorter / stale value is ignored so black is
      // never erased.
      const prevConfirmedLen = confirmedText.length;
      if (confirmed && confirmed.length >= confirmedText.length) {
        confirmedText = confirmed;
      }
      // (Re)freeze the tentative line only on a NON-EMPTY prediction. An empty
      // prediction is a confirm-sequence frame: keep the frozen line so its gray
      // tail persists until confirmed either reaches or diverges from it.
      if (prediction) {
        tentLine = confirmedText + prediction;
      }

      // Gray = the frozen tentative line beyond confirmed, but ONLY while confirmed
      // is still a prefix of it. The moment confirmed diverges (startsWith fails),
      // the gray is dropped and the diverging confirmed types out in black below.
      const grayTail =
        tentLine.length > confirmedText.length && tentLine.startsWith(confirmedText)
          ? tentLine.slice(confirmedText.length)
          : "";
      const next = confirmedText + grayTail;

      confirmedLen = confirmedText.length;

      // Rewind the typewriter to where the new line diverges from the old, so a
      // corrected tail re-types from there (an unchanged prefix is never re-typed).
      const common = commonPrefixLen(text, next);
      if (shown > common) shown = common;
      text = next;
      // On a frame that newly confirms text, the confirmed prefix the user already
      // saw predicted (the common prefix, capped at confirmedLen) must appear at
      // once — never re-typed — even if the typewriter hadn't reached it. The
      // diverging confirmed tail past `common` still types out (black, via render's
      // confirmedLen split). Prediction-only frames don't force-reveal, so the gray
      // tail keeps typing smoothly.
      if (confirmedLen > prevConfirmedLen) {
        const reveal = Math.min(common, confirmedLen);
        if (shown < reveal) shown = reveal;
      }
      if (shown > text.length) shown = text.length;

      // Reflect the instant (recolored) prefix immediately; then run the timer only
      // if there is still text left to type out.
      render();
      if (shown < text.length) ensure();
      else stop();
    }

    function clear() {
      stop();
      confirmedText = "";
      tentLine = "";
      text = "";
      confirmedLen = 0;
      shown = 0;
      render();
    }

    // ---- mode switching --------------------------------------------------
    function enterDrag() {
      // Seed the floating rect from the panel's current on-screen position so it
      // lifts off exactly where it was docked.
      const s = stageEl();
      if (s) {
        const pr = panel.getBoundingClientRect();
        const sr = s.getBoundingClientRect();
        rect = {
          left: pr.left - sr.left,
          top: pr.top - sr.top,
          width: pr.width,
          height: pr.height,
        };
      }
      clampRectIntoStage();
      mode = "drag";
      panel.classList.remove("mode-default");
      panel.classList.add("mode-drag");
      applyRect();
      actionBtn.innerHTML = ICON_CLOSE;
      actionBtn.title = "하단에 도킹";
    }

    function exitDrag() {
      mode = "default";
      panel.classList.remove("mode-drag");
      panel.classList.add("mode-default");
      // Hand width/position back to the CSS class; keep the persisted height.
      panel.style.left = "";
      panel.style.top = "";
      panel.style.width = "";
      panel.style.right = "";
      panel.style.bottom = "";
      panel.style.height = `${heightPx}px`;
      actionBtn.innerHTML = ICON_POPOUT;
      actionBtn.title = "새 창처럼 띄우기";
    }

    actionBtn.addEventListener("click", () => {
      if (mode === "default") enterDrag();
      else exitDrag();
    });

    // Pointer-capture drag: moves keep arriving on captureEl even when the
    // cursor crosses the embedded <webview> (which would otherwise swallow
    // them). onDone runs once on pointerup/cancel.
    function beginDrag(captureEl, e0, onMove, onDone) {
      try { captureEl.setPointerCapture(e0.pointerId); } catch (_) {}
      const move = (ev) => onMove(ev);
      const end = (ev) => {
        captureEl.removeEventListener("pointermove", move);
        captureEl.removeEventListener("pointerup", end);
        captureEl.removeEventListener("pointercancel", end);
        try { captureEl.releasePointerCapture(e0.pointerId); } catch (_) {}
        if (onDone) onDone();
      };
      captureEl.addEventListener("pointermove", move);
      captureEl.addEventListener("pointerup", end);
      captureEl.addEventListener("pointercancel", end);
    }

    // ---- header drag: resize height (default) / move panel (drag) ---------
    header.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".overlay-action")) return; // let the button click
      e.preventDefault();
      if (mode === "default") {
        const startY = e.clientY;
        const startH = panel.offsetHeight; // border-box → equals the CSS height
        const maxH = stageSize().h; // full-bleed docked panel can fill the stage
        beginDrag(
          header,
          e,
          (ev) => {
            heightPx = clamp(startH + (startY - ev.clientY), MIN_H, maxH);
            panel.style.height = `${heightPx}px`;
          },
          () => opts.onHeightChange && opts.onHeightChange(heightPx)
        );
      } else {
        const startX = e.clientX;
        const startY = e.clientY;
        const startL = rect.left;
        const startT = rect.top;
        beginDrag(header, e, (ev) => {
          rect.left = startL + (ev.clientX - startX);
          rect.top = startT + (ev.clientY - startY);
          clampRectIntoStage(); // keeps the panel fully on-screen (any stage size)
          applyRect();
        });
      }
    });

    // ---- edge/corner resize (drag mode) ----------------------------------
    handlesWrap.addEventListener("pointerdown", (e) => {
      const handle = e.target.closest(".overlay-handle");
      if (!handle) return;
      e.preventDefault();
      const dir = handle.dataset.dir;
      const startX = e.clientX;
      const startY = e.clientY;
      const s = { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
      const right = s.left + s.width;
      const bottom = s.top + s.height;
      beginDrag(handlesWrap, e, (ev) => {
        const { w, h } = stageSize();
        const dx = ev.clientX - startX;
        const dy = ev.clientY - startY;
        // Anchor the opposite edge so resizing never jitters at the limits.
        let L = s.left, T = s.top, R = right, B = bottom;
        if (dir.includes("e")) R = clamp(right + dx, s.left + MIN_W, w);
        if (dir.includes("w")) L = clamp(s.left + dx, 0, right - MIN_W);
        if (dir.includes("s")) B = clamp(bottom + dy, s.top + MIN_H, h);
        if (dir.includes("n")) T = clamp(s.top + dy, 0, bottom - MIN_H);
        rect = { left: L, top: T, width: R - L, height: B - T };
        clampRectIntoStage(); // normalize (no-op for normal stages; guards tiny ones)
        applyRect();
      });
    });

    // Keep a floating panel inside the stage when the window/stage shrinks.
    const onWinResize = () => {
      if (mode !== "drag") return;
      clampRectIntoStage();
      applyRect();
    };
    window.addEventListener("resize", onWinResize);

    function destroy() {
      stop();
      window.removeEventListener("resize", onWinResize);
    }

    return { setAlphaPct, setHeight, update, clear, destroy };
  }

  return { create };
})();
