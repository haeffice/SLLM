// Renders model architecture tags (from /health) as chips, with a fallback.
window.SLLM = window.SLLM || {};

SLLM.tags = (() => {
  const FALLBACK = ["Speech-LLM", "offline"];

  function render(container, tags) {
    container.innerHTML = "";
    const list = tags && tags.length ? tags : FALLBACK;
    for (const t of list) {
      const chip = document.createElement("span");
      chip.className = "tag-chip";
      chip.textContent = t;
      container.appendChild(chip);
    }
  }

  return { render };
})();
