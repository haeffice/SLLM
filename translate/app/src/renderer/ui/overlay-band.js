// Bottom translation band: bold black text over the embedded browser, with an
// adjustable-opacity background and a most-recent-N-(visual)-lines window.
window.SLLM = window.SLLM || {};

SLLM.overlayBand = (() => {
  // Cap retained DOM nodes; visual clipping (max-height + overflow) shows only
  // the most recent N lines, this just bounds memory.
  const MAX_NODES = 40;

  function create(hostEl) {
    const lines = document.createElement("div");
    lines.className = "overlay-lines";
    hostEl.appendChild(lines);

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

    function addText(text) {
      if (!text) return;
      const line = document.createElement("div");
      line.className = "overlay-line";
      line.textContent = text;
      lines.appendChild(line);
      while (lines.childElementCount > MAX_NODES) {
        lines.removeChild(lines.firstChild);
      }
    }

    function clear() {
      lines.innerHTML = "";
    }

    return { setAlphaPct, setWidth, setRecentLines, addText, clear };
  }

  return { create };
})();
