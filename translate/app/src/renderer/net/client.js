// Server communication: POST /translate per audio chunk, GET /health for tags.
window.SLLM = window.SLLM || {};

SLLM.net = (() => {
  async function postTranslate(base, wavBlob, src, tgt) {
    const url = `${base}/translate?src=${encodeURIComponent(src)}&tgt=${encodeURIComponent(tgt)}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "audio/wav" },
        body: wavBlob,
        signal: controller.signal,
      });
      const json = await res.json().catch(() => ({}));
      return { status: res.status, ok: res.ok, json };
    } finally {
      clearTimeout(timer);
    }
  }

  async function fetchHealth(base) {
    const res = await fetch(`${base}/health`, { method: "GET" });
    const json = await res.json().catch(() => ({}));
    return { status: res.status, ok: res.ok, json };
  }

  // Pull the architecture tags for the server's default model out of /health.
  function tagsFromHealth(json) {
    if (!json || !json.models) return [];
    const def = json.default_model;
    const entry = (def && json.models[def]) || Object.values(json.models)[0];
    return (entry && Array.isArray(entry.tags)) ? entry.tags : [];
  }

  return { postTranslate, fetchHealth, tagsFromHealth };
})();
