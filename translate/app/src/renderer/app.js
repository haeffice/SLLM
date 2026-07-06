// Bootstrap: sidebar routing + "실시간 번역" mount / toggle-reset.
window.SLLM = window.SLLM || {};

(function () {
  const sidebar = document.getElementById("sidebar");
  const content = document.getElementById("content");

  let activeFeature = null;
  let destroyActive = null; // teardown closure for the mounted view
  let busy = false; // serialize clicks so an in-flight mount/teardown can't race

  async function clearActive() {
    if (destroyActive) {
      await destroyActive();
      destroyActive = null;
    }
    content.innerHTML = "";
    activeFeature = null;
    for (const b of sidebar.querySelectorAll(".nav-item")) {
      b.classList.remove("active");
    }
  }

  function showPlaceholder() {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = "왼쪽 메뉴에서 기능을 선택하세요.";
    content.appendChild(ph);
  }

  async function selectFeature(feature, navEl) {
    if (busy) return;
    busy = true;
    try {
      // Tapping the active feature again resets/clears the view.
      if (activeFeature === feature) {
        await clearActive();
        showPlaceholder();
        return;
      }
      await clearActive();
      activeFeature = feature;
      navEl.classList.add("active");

      if (feature === "translate") {
        const settings = await SLLM.settings.load();
        destroyActive = SLLM.mountTranslateView(content, settings);
      }
    } finally {
      busy = false;
    }
  }

  sidebar.addEventListener("click", (e) => {
    const item = e.target.closest(".nav-item");
    if (!item) return;
    selectFeature(item.dataset.feature, item);
  });

  showPlaceholder();
})();
