// HTTP status indicator: 2xx green, 4xx red, 5xx purple, null/offline outline.
window.SLLM = window.SLLM || {};

SLLM.statusDot = (() => {
  function create() {
    const el = document.createElement("span");
    el.className = "status-dot status-dot--none";
    el.title = "server status";
    return el;
  }

  function setFromStatus(el, status) {
    el.classList.remove(
      "status-dot--none",
      "status-dot--ok",
      "status-dot--warn",
      "status-dot--err"
    );
    if (status == null) {
      el.classList.add("status-dot--none");
    } else if (status >= 200 && status < 300) {
      el.classList.add("status-dot--ok");
    } else if (status >= 400 && status < 500) {
      el.classList.add("status-dot--warn");
    } else {
      el.classList.add("status-dot--err");
    }
    el.title = status == null ? "offline" : `HTTP ${status}`;
  }

  return { create, setFromStatus };
})();
